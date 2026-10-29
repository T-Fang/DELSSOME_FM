"""nn/ and train/ (stage 1): batches from the training store, the network's invariances,
the heads' guarantees, that it can fit one batch, and a two-step Lightning run with MLflow.
Needs torch and lightning (the delssome_fm_train env); skipped elsewhere."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
# A few threads: on a shared many-core node torch's default (one per core) is ~200x slower.
torch.set_num_threads(4)

from delssome_fm.config import Stage1Config                             # noqa: E402
from delssome_fm.nn.template_encoder import GraphBatch                  # noqa: E402
from delssome_fm.train.data import TrainingStore                        # noqa: E402
from delssome_fm.train.losses import GROUPS, group_losses               # noqa: E402
from delssome_fm.train.fit import fit                                    # noqa: E402
from delssome_fm.train.stage1 import Stage1Module, upper_triangle        # noqa: E402


def tiny_config(store_dir: Path, tmp_path: Path) -> Stage1Config:
    return Stage1Config(
        store_dir=store_dir, output_dir=tmp_path / "ckpt", seed=0, d_model=32, d_template=16,
        template_layers=2, backbone_layers=2, n_heads=4, ffn_multiplier=2, dropout=0.0,
        fc_rank=8, fc_offset=True, pair_head="bilinear", pair_hidden=16, cost_loss=False,
        fc_autoencoder="none", stat_loss=True, catch24_loss=False, fc_net_loss=False,
        max_distance=8,
        max_degree=8, token_scheme="region", train_models=2,
        draws_per_model=100, oversample_coupled=False, coupled_share=0.5,
        coupled_threshold=0.05, coupled_draws_only=False, batch_size=2, edges_per_step=64,
        learning_rate=1e-3, weight_decay=0.0, warmup_steps=1, max_steps=2, val_rows=16,
        val_every_steps=1, final_eval_rows=16, num_workers=0, precision="32-true",
        gradient_clip=1.0,
        log_every_steps=1, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifact_dir=tmp_path / "artifacts", mlflow_experiment="test")


@pytest.fixture
def setup(tiny_store, tmp_path):
    cfg, _, _ = tiny_store
    store = TrainingStore(cfg.store_dir, "train")
    batch = store[list(range(min(8, len(store))))]
    torch.manual_seed(0)                                   # deterministic initial weights
    module = Stage1Module(tiny_config(cfg.store_dir, tmp_path), store.n_fcd_levels,
                          (0.4, 1.3)).eval()
    return store, batch, module


def test_batch_shapes_and_targets(setup):
    store, batch, _ = setup
    B = len(batch.theta)
    assert batch.theta.shape == (B, 68, 5) and batch.psi.shape == (B, 2)
    assert batch.sc.shape == (B, 68, 68) and float(batch.sc.max()) <= 1.0
    g = batch.graph
    M = g.features.shape[1]
    assert g.features.shape == (B, M, 21) and g.distances.shape == (B, M, M)
    assert batch.targets["pairwise"].shape == (B, 2278)
    for v in batch.targets.values():
        assert torch.isfinite(v).all()
    # absent parameters are 0 and have no Par node; present ones are log10 in [-1, 1]
    present = g.regional_par >= 0
    assert (batch.theta[:, :, ~present.all(0)] == 0).all() or present.all()
    assert batch.theta.abs().max() <= 1.0 + 1e-6


def test_subset_rows_takes_the_first_models_and_draws(tiny_store):
    from delssome_fm.train.data import Composition, model_mean_fc, subset_rows
    cfg, _, _ = tiny_store
    store = TrainingStore(cfg.store_dir, "train")
    model = np.asarray(store.rows["model"])
    n_rows = store.models["n_rows"]
    natural = lambda n, k: Composition(n, k, None, 0.05)                 # noqa: E731
    rows = subset_rows(store, natural(2, 1))
    assert model[rows].tolist() == [m for m in (0, 1) if n_rows[m] > 0]   # one draw each
    every = subset_rows(store, natural(len(n_rows), 100))
    assert every.tolist() == list(range(len(model)))
    with pytest.raises(ValueError, match="train_models"):
        subset_rows(store, natural(len(n_rows) + 1, 100))
    # composition: split the models at their median mean FC, then ask for 1 of each kind
    fc = model_mean_fc(store)
    threshold = float(np.nanmedian(fc))
    chosen = np.unique(model[subset_rows(store, Composition(2, 100, 0.5, threshold))])
    assert len(chosen) == 2 and (fc[chosen] > threshold).sum() == 1
    with pytest.raises(ValueError, match="coupled models wanted"):
        subset_rows(store, Composition(2, 100, 1.0, float(np.nanmax(fc))))


def test_outputs_ignore_graph_padding(setup):
    """Padding the DAGs with extra masked nodes must not change any output."""
    _, batch, module = setup
    g = batch.graph
    extra = 5
    pad = lambda x, value: torch.nn.functional.pad(x, value=value, pad=_pad_spec(x, g, extra))
    padded = GraphBatch(features=pad(g.features, 0.0), degree=pad(g.degree, 0),
                        distances=pad(g.distances, -1), node_mask=pad(g.node_mask, False),
                        regional_par=g.regional_par, global_par=g.global_par,
                        deriv_roots=g.deriv_roots, agg=g.agg, out_root=g.out_root)
    with torch.no_grad():
        a = module.backbone(batch.theta, batch.psi, batch.sc, g)
        b = module.backbone(batch.theta, batch.psi, batch.sc, padded)
    torch.testing.assert_close(a.h_cls, b.h_cls, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a.regions, b.regions, rtol=1e-4, atol=1e-5)


def _pad_spec(x, g, extra):
    """F.pad spec that extends every node axis of x (those of size M) by `extra`."""
    M = g.features.shape[1]
    spec = []
    for size in reversed(x.shape[1:]):
        spec += [0, extra if size == M else 0]
    return spec


def test_outputs_do_not_depend_on_parameter_order(setup):
    """Permuting the regional parameters together with their Par nodes changes nothing:
    the network identifies a parameter by where it sits, not by its column."""
    _, batch, module = setup
    perm = torch.tensor([2, 0, 1, 3, 4])
    g = batch.graph
    permuted = dataclasses.replace(g, regional_par=g.regional_par[:, perm])
    with torch.no_grad():
        a = module.backbone(batch.theta, batch.psi, batch.sc, g)
        b = module.backbone(batch.theta[..., perm], batch.psi, batch.sc, permuted)
    torch.testing.assert_close(a.h_cls, b.h_cls, rtol=1e-4, atol=1e-5)


HEADS = [("bilinear", True), ("mlp_symmetric", False), ("mlp_concat", False)]


def _module_with_head(store, tmp_path, head, offset):
    cfg = dataclasses.replace(tiny_config(tmp_path, tmp_path),
                              pair_head=head, fc_offset=offset)
    torch.manual_seed(0)
    return Stage1Module(cfg, store.n_fcd_levels, (0.4, 1.3)).eval()


@pytest.mark.parametrize("head,offset", HEADS)
def test_heads_give_a_cdf_and_a_symmetric_fc(setup, tmp_path, head, offset):
    store, batch, _ = setup
    module = _module_with_head(store, tmp_path, head, offset)
    with torch.no_grad():
        out = module.backbone(batch.theta, batch.psi, batch.sc, batch.graph)
        pred = module.heads(out, upper_triangle()[:10])
        swapped = module.heads.pairwise(out, upper_triangle()[:10].flip(1))
    assert (pred.fcd.diff(dim=-1) >= -1e-6).all()
    torch.testing.assert_close(pred.fcd[:, -1], torch.ones(len(pred.fcd)))
    torch.testing.assert_close(pred.pairwise, swapped)


def test_an_offset_belongs_to_the_bilinear_head_only():
    from delssome_fm.nn.heads import Stage1Heads
    with pytest.raises(ValueError, match="bilinear"):
        Stage1Heads(16, 4, 10, fc_offset=True, pair_head="mlp_symmetric")
    with pytest.raises(ValueError, match="pair_head"):
        Stage1Heads(16, 4, 10, fc_offset=False, pair_head="mlp")


@pytest.mark.parametrize("head,offset", HEADS)
def test_the_network_can_fit_one_batch(setup, tmp_path, head, offset):
    """Each group's loss falls when training on one batch. The pairwise loss falls least with
    the bilinear head: a rank-k factorisation cannot fit the near-full-rank FC of the tiny
    corpus's 40-frame runs."""
    store, batch, _ = setup
    module = _module_with_head(store, tmp_path, head, offset)
    module.train()
    optimiser = torch.optim.Adam(module.parameters(), lr=3e-3)
    edges = torch.arange(2278)
    history = []
    for _ in range(150):
        optimiser.zero_grad()
        losses = group_losses(module.predict_statistics(batch, edges), batch.targets, edges)
        sum(losses.values()).backward()
        optimiser.step()
        history.append({g: float(v.detach()) for g, v in losses.items()})
    first, last = history[0], history[-1]
    assert set(first) == set(GROUPS)
    assert last["regional"] < 0.6 * first["regional"]
    assert last["fc_moments"] < 0.2 * first["fc_moments"]
    assert last["fcd"] < 0.2 * first["fcd"]
    assert last["pairwise"] < first["pairwise"]


def test_two_step_lightning_run_logs_to_mlflow(tiny_store, tmp_path):
    cfg, _, _ = tiny_store
    stage1 = tiny_config(cfg.store_dir, tmp_path)
    config_file = tmp_path / "stage1.yaml"
    config_file.write_text("for the artifact\n")
    checkpoints = fit(stage1, config_file, run_name="smoke")
    assert (checkpoints / "last.ckpt").is_file()
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=stage1.mlflow_tracking_uri)
    run = client.search_runs([client.get_experiment_by_name("test").experiment_id])[0]
    assert {"train/total", "val/total", "val/r2_pairwise"} <= set(run.data.metrics)
    assert np.isfinite(run.data.metrics["val/total"])
    # raw-unit cost terms at validation, and the evaluation of the best checkpoint
    assert any(k.startswith("val/cost_ks_") for k in run.data.metrics)
    for split in ("train", "val", "test"):
        assert np.isfinite(run.data.metrics[f"final/{split}/total"])
    assert (checkpoints / "final_eval.npz").is_file()
    assert (tmp_path / "artifacts").is_dir() and not Path("mlruns").exists()
    # the checkpoint rebuilds the module and evaluates on a packed split
    from delssome_fm.train.evaluate import evaluate, load_stage1
    metrics = evaluate(load_stage1(checkpoints / "last.ckpt"), cfg.store_dir, "val")
    assert metrics["rows"] > 0 and all(np.isfinite(metrics[k]) for k in ("total", "r2_pairwise"))
    from delssome_fm.train.evaluate import per_model_errors
    per_model = per_model_errors(load_stage1(checkpoints / "last.ckpt"), cfg.store_dir, "val")
    assert len(per_model["total"]) == len(per_model["model"]) == len(per_model["mean_fc"]) > 0
    assert np.isfinite(per_model["total"]).all()


@pytest.mark.parametrize("head,offset", HEADS)
def test_cost_loss_trains_with_finite_gradients(setup, tmp_path, head, offset):
    """With cost_loss, the step adds 1 - r, d and KS on all edges; a flat initial FC must
    not make the correlation's gradient NaN."""
    store, batch, _ = setup
    cfg = dataclasses.replace(tiny_config(tmp_path, tmp_path), pair_head=head, fc_offset=offset,
                              cost_loss=True)
    torch.manual_seed(0)
    module = Stage1Module(cfg, store.n_fcd_levels, (0.4, 1.3)).train()
    module.log = module.log_dict = lambda *a, **k: None          # no trainer attached
    loss = module.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    assert len(module.weighting.groups) == 7


def test_parameter_encoding_follows_the_store_policy():
    from delssome_fm.spec.sampler import SINH_SCALE, SINH_Z
    from delssome_fm.train.data import encode_parameters
    x = np.array([[0.1, 1.0, 10.0, np.nan]])
    np.testing.assert_allclose(encode_parameters(x, "multiplier").numpy(), [[-1, 0, 1, 0]])
    v = np.array([[-1e4, -1.0, 0.0, 1.0, 1e4, np.nan]])
    enc = encode_parameters(v, "sinh").numpy()[0]
    np.testing.assert_allclose(enc[:5], np.arcsinh(v[0, :5] / SINH_SCALE) / SINH_Z, rtol=1e-6)
    assert enc[0] == pytest.approx(-1, abs=1e-3) and enc[4] == pytest.approx(1, abs=1e-3)
    assert enc[2] == 0 and enc[5] == 0                      # zero, and NaN padding
    with pytest.raises(ValueError, match="policy"):
        encode_parameters(x, "flat")


def test_evaluation_refuses_a_store_of_another_policy(setup, tiny_store):
    from delssome_fm.train.evaluate import check_parameter_policy
    store, _, module = setup
    check_parameter_policy(module, store)                  # same store: fine
    store.parameter_policy = "sinh"
    with pytest.raises(ValueError, match="policy"):
        check_parameter_policy(module, store)



# ---------------------------------------------------------------- catch24 and the FC autoencoder


def test_loss_groups_follow_the_switches():
    from delssome_fm.train.losses import COST_GROUPS, loss_groups
    assert loss_groups(True, False, False) == GROUPS
    assert loss_groups(True, True, True) == GROUPS + ("catch24",) + COST_GROUPS
    assert loss_groups(False, True, False) == ("catch24",)
    with pytest.raises(ValueError, match="nothing to train"):
        loss_groups(False, False, False)
    with pytest.raises(ValueError, match="needs stat_loss"):
        loss_groups(False, True, True)


@pytest.mark.parametrize("stat_loss,cost_loss", [(True, True), (False, False)])
def test_catch24_head_trains_with_or_without_the_statistics(setup, tmp_path, stat_loss,
                                                            cost_loss):
    store, batch, _ = setup
    assert batch.targets["catch24"].shape == (len(batch.theta), 68, 24)
    cfg = dataclasses.replace(tiny_config(tmp_path, tmp_path), pair_head="mlp_concat",
                              fc_offset=False, catch24_loss=True, stat_loss=stat_loss,
                              cost_loss=cost_loss)
    torch.manual_seed(0)
    module = Stage1Module(cfg, store.n_fcd_levels, (0.4, 1.3)).train()
    module.log = module.log_dict = lambda *a, **k: None          # no trainer attached
    loss = module.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss) and module.heads.catch24 is not None
    assert module.heads.catch24.net[0].weight.grad is not None
    trained = module.weighting.groups
    assert ("catch24" in trained) and (("regional" in trained) == stat_loss)
    assert module.selection == (GROUPS if stat_loss else ()) + ("catch24",)


def _tiny_autoencoder(tiny_store, tmp_path):
    from delssome_fm.config import FCAutoencoderConfig
    from delssome_fm.train.autoencoder import fit_autoencoder
    cfg, _, _ = tiny_store
    ae_cfg = FCAutoencoderConfig(
        store_dir=cfg.store_dir, output_dir=tmp_path / "ae", seed=0, latent_dim=8,
        hidden=(32,), batch_size=4, learning_rate=1e-3, weight_decay=0.0, warmup_steps=1,
        max_steps=3, val_rows=8, val_every_steps=1, num_workers=0, precision="32-true",
        gradient_clip=1.0,
        log_every_steps=1, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifact_dir=tmp_path / "artifacts", mlflow_experiment="test_ae")
    return fit_autoencoder(ae_cfg, Path("fc_autoencoder.yaml"), run_name="tiny")


def test_fc_autoencoder_trains_logs_and_reloads(tiny_store, tmp_path):
    from delssome_fm.train.autoencoder import load_fc_autoencoder
    path = _tiny_autoencoder(tiny_store, tmp_path)
    model, norm = load_fc_autoencoder(path)
    cfg, _, _ = tiny_store
    with np.load(cfg.store_dir / "normalisation.npz") as n:
        assert norm == pytest.approx((float(n["pairwise_mean"]), float(n["pairwise_sd"])))
    assert model.latent_dim == 8 and model(torch.zeros(2, 2278)).shape == (2, 2278)
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    run = client.search_runs([client.get_experiment_by_name("test_ae").experiment_id])[0]
    assert {"train/mse", "val/mse", "val/r2"} <= set(run.data.metrics)


def test_latent_head_decodes_all_edges_with_a_frozen_decoder(setup, tiny_store, tmp_path):
    store, batch, _ = setup
    path = _tiny_autoencoder(tiny_store, tmp_path)
    norm = (float(store.norm["pairwise_mean"]), float(store.norm["pairwise_sd"]))
    cfg = dataclasses.replace(tiny_config(tmp_path, tmp_path), pair_head="latent",
                              fc_offset=False, fc_autoencoder=str(path), cost_loss=True)
    torch.manual_seed(0)
    module = Stage1Module(cfg, store.n_fcd_levels, norm).train()
    module.log = module.log_dict = lambda *a, **k: None
    with torch.no_grad():
        out = module.backbone(batch.theta, batch.psi, batch.sc, batch.graph)
        some = module.heads.pairwise(out, upper_triangle()[:10])
        flipped = module.heads.pairwise(out, upper_triangle()[:10].flip(1))
        full = module.heads.pairwise(out, upper_triangle())
    torch.testing.assert_close(some, flipped)
    torch.testing.assert_close(some, full[:, :10])
    module.training_step(batch, 0).backward()
    assert all(p.grad is None for p in module.heads.fc_decoder.parameters())   # frozen
    assert module.heads.fc_latent.net[0].weight.grad is not None
    with pytest.raises(ValueError, match="z-scored"):
        Stage1Module(cfg, store.n_fcd_levels, (norm[0] + 1, norm[1]))


def test_lightning_run_with_catch24_selects_on_the_trained_groups(tiny_store, tmp_path):
    cfg, _, _ = tiny_store
    stage1 = dataclasses.replace(tiny_config(cfg.store_dir, tmp_path), catch24_loss=True)
    config_file = tmp_path / "stage1.yaml"
    config_file.write_text("for the artifact\n")
    fit(stage1, config_file, run_name="c24")
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=stage1.mlflow_tracking_uri)
    run = client.search_runs([client.get_experiment_by_name("test").experiment_id])[0]
    m = run.data.metrics
    assert {"val/selection", "val/catch24", "final/test/catch24"} <= set(m)
    assert m["val/selection"] == pytest.approx(m["val/total"] + m["val/catch24"], rel=1e-4)


def test_catch24_fit_reports_per_feature_r2(tiny_store, tmp_path):
    from delssome_fm.train.report import catch24_fit
    cfg, _, _ = tiny_store
    store = TrainingStore(cfg.store_dir, "train")
    stage1 = dataclasses.replace(tiny_config(cfg.store_dir, tmp_path), catch24_loss=True)
    torch.manual_seed(0)
    module = Stage1Module(stage1, store.n_fcd_levels, (0.4, 1.3)).eval()
    fit_ = catch24_fit(module, "test", max_rows=8, n_points=5)
    assert fit_["r2"].shape == (24,) and fit_["true"].shape == fit_["pred"].shape == (5, 24)
    assert np.isfinite(fit_["r2"][np.isfinite(fit_["r2"])]).all()
    with pytest.raises(ValueError, match="no catch24 head"):
        catch24_fit(Stage1Module(tiny_config(cfg.store_dir, tmp_path), store.n_fcd_levels,
                                 (0.4, 1.3)), "test")


# ---------------------------------------------------------------- Yeo-7 network FC


def test_network_fc_target_is_packed_and_served(setup):
    store, batch, _ = setup
    assert batch.targets["fc_net"].shape == (len(batch.theta), 21)
    assert torch.isfinite(batch.targets["fc_net"]).all()
    assert store.norm["fc_net_mean"].shape == (21,)
    # served z-scored with the store's statistics (like regional, pairwise, catch24)
    rows = sorted(range(min(8, len(store))))
    raw = np.asarray(store.rows["fc_net"][rows])
    expected = (raw - store.norm["fc_net_mean"]) / store.norm["fc_net_sd"]
    np.testing.assert_allclose(batch.targets["fc_net"].numpy(), expected, rtol=1e-5, atol=1e-6)


def test_fc_net_head_trains_alone(setup, tmp_path):
    from delssome_fm.train.losses import loss_groups
    store, batch, _ = setup
    assert loss_groups(False, False, False, True) == ("fc_net",)
    cfg = dataclasses.replace(tiny_config(tmp_path, tmp_path), pair_head="mlp_concat",
                              fc_offset=False, stat_loss=False, fc_net_loss=True)
    torch.manual_seed(0)
    module = Stage1Module(cfg, store.n_fcd_levels, (0.4, 1.3)).train()
    module.log = module.log_dict = lambda *a, **k: None
    optimiser = torch.optim.Adam(module.parameters(), lr=3e-3)
    first = None
    for _ in range(100):
        optimiser.zero_grad()
        loss = group_losses(module.predict_statistics(batch, torch.arange(10)),
                            batch.targets, torch.arange(10))["fc_net"]
        loss.backward()
        optimiser.step()
        first = float(loss) if first is None else first
    assert float(loss) < 0.2 * first
    assert module.selection == ("fc_net",) and module.weighting.groups == ("fc_net",)


def test_lightning_run_with_fc_net_logs_it(tiny_store, tmp_path):
    cfg, _, _ = tiny_store
    stage1 = dataclasses.replace(tiny_config(cfg.store_dir, tmp_path), fc_net_loss=True,
                                 stat_loss=False)
    config_file = tmp_path / "stage1.yaml"
    config_file.write_text("for the artifact\n")
    fit(stage1, config_file, run_name="net")
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=stage1.mlflow_tracking_uri)
    run = client.search_runs([client.get_experiment_by_name("test").experiment_id])[0]
    m = run.data.metrics
    assert {"val/fc_net", "final/test/fc_net", "final/train/fc_net"} <= set(m)
    assert m["val/selection"] == pytest.approx(m["val/fc_net"], rel=1e-5)


def test_network_fc_fit_scores_direct_and_downsampled_predictions(tiny_store, tmp_path):
    from delssome_fm.train.report import network_fc_fit
    cfg, _, _ = tiny_store
    store = TrainingStore(cfg.store_dir, "train")
    stage1 = dataclasses.replace(tiny_config(cfg.store_dir, tmp_path), fc_net_loss=True)
    torch.manual_seed(0)
    module = Stage1Module(stage1, store.n_fcd_levels, (0.4, 1.3)).eval()
    r = network_fc_fit(module, "test", max_rows=8)
    assert r["true"].shape == r["direct"].shape == r["downsampled"].shape == (len(r["rows"]), 21)
    assert np.all(np.abs(r["true"]) <= 1) and np.all(np.abs(r["downsampled"]) <= 1 + 1e-9)
    for k in ("direct", "downsampled"):
        assert r[f"{k}_mse"].shape == (len(r["rows"]),) and np.isfinite(r[f"{k}_mse"]).all()


# ---------------------------------------------------------------- coupled draws only


def test_coupled_draws_keep_only_rows_above_the_threshold(tiny_store):
    from delssome_fm.train.data import Composition, coupled_rows, subset_rows
    cfg, _, _ = tiny_store
    store = TrainingStore(cfg.store_dir, "train")
    mean_fc = np.asarray(store.rows["fc_moments"][:, 0])
    threshold = float(np.median(mean_fc))
    n = len(store.models["n_rows"])
    every = subset_rows(store, Composition(n, 100, None, threshold))
    coupled = subset_rows(store, Composition(n, 100, None, threshold, coupled_draws=True))
    assert 0 < len(coupled) < len(every)
    assert (mean_fc[coupled] > threshold).all()
    assert coupled.tolist() == [r for r in every if mean_fc[r] > threshold]
    assert coupled_rows(store, every[::-1], threshold).tolist() == coupled.tolist()


def test_coupled_only_run_validates_and_evaluates_on_coupled_draws(tiny_store, tmp_path):
    cfg, _, _ = tiny_store
    store = TrainingStore(cfg.store_dir, "val")
    threshold = float(np.median(np.asarray(store.rows["fc_moments"][:, 0])))
    stage1 = dataclasses.replace(tiny_config(cfg.store_dir, tmp_path), coupled_draws_only=True,
                                 coupled_threshold=threshold, train_models=len(
                                     TrainingStore(cfg.store_dir, "train").models["n_rows"]))
    config_file = tmp_path / "stage1.yaml"
    config_file.write_text("for the artifact\n")
    checkpoints = fit(stage1, config_file, run_name="coupled")
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=stage1.mlflow_tracking_uri)
    m = client.search_runs([client.get_experiment_by_name("test").experiment_id])[0].data.metrics
    # final/* is already on coupled rows for this run, so it equals final_coupled/*
    for split in ("train", "val", "test"):
        assert m[f"final/{split}/total"] == pytest.approx(m[f"final_coupled/{split}/total"])
        assert m[f"final/{split}/rows_uncoupled"] == 0
    assert (checkpoints / "final_eval_coupled.npz").is_file()


# ---------------------------------------------------------------- parameter tokens (2026-10-29)


@pytest.fixture
def ptok(tiny_store_sinh, tmp_path):
    """A sinh-policy store, a batch, and a parameter-token module with the MLP FC head."""
    cfg, _, _ = tiny_store_sinh
    store = TrainingStore(cfg.store_dir, "train")
    batch = store[list(range(min(8, len(store))))]
    torch.manual_seed(0)
    module = Stage1Module(_ptok_config(cfg.store_dir, tmp_path), store.n_fcd_levels,
                          (0.4, 1.3)).eval()
    return store, batch, module


def _ptok_config(store_dir, tmp_path, **changes):
    return dataclasses.replace(tiny_config(store_dir, tmp_path), token_scheme="parameter",
                               pair_head="mlp_concat", fc_offset=False, **changes)


def test_const_values_are_served_encoded_under_the_sinh_policy_only(ptok, setup):
    from delssome_fm.spec.sampler import SINH_SCALE, SINH_Z
    store, batch, _ = ptok
    g = batch.graph
    const = g.features[..., 8] == 1                                     # Const one-hot
    assert g.const_value.shape == g.node_mask.shape and const.any()
    assert (g.const_value[~const] == 0).all()
    m = int(store.rows["model"][0])                                     # the batch's first row
    lo, hi = store.models["node_offset"][m], store.models["node_offset"][m + 1]
    value = np.asarray(store.node_value[lo:hi])
    expected = np.arcsinh(value[const[0, :hi - lo].numpy()] / SINH_SCALE) / SINH_Z
    np.testing.assert_allclose(g.const_value[0, :hi - lo][const[0, :hi - lo]].numpy(), expected,
                               rtol=1e-6)
    assert setup[1].graph.const_value is None                           # multiplier policy


def test_constants_become_global_parameters():
    from delssome_fm.nn.template_encoder import constants_as_global_parameters
    features = torch.zeros(4, 21)
    features[0, 8], features[0, 15], features[0, 16] = 1, -1, 2.5      # Const -10^2.5
    features[1, 9], features[1, 17] = 1, 1                              # regional Par
    features[2, 9], features[2, 17] = 1, -1                             # global Par
    features[3, 10], features[3, 18] = 1, -1                            # neighbour Var
    out = constants_as_global_parameters(features)
    torch.testing.assert_close(out[0], features[2])         # a constant is a global parameter
    torch.testing.assert_close(out[1:], features[1:])


def test_parameter_tokens_ignore_padding_and_absent_entries(ptok):
    """Extra masked graph nodes, values of absent parameters, and the Const columns the
    encoder recodes must not change any output."""
    _, batch, module = ptok
    g = batch.graph
    extra = 5
    pad = lambda x, value: torch.nn.functional.pad(x, value=value, pad=_pad_spec(x, g, extra))
    padded = GraphBatch(features=pad(g.features, 0.0), degree=pad(g.degree, 0),
                        distances=pad(g.distances, -1), node_mask=pad(g.node_mask, False),
                        regional_par=g.regional_par, global_par=g.global_par,
                        deriv_roots=g.deriv_roots, agg=g.agg, out_root=g.out_root,
                        const_value=pad(g.const_value, 0.0))
    absent_p = (g.regional_par < 0)[:, None, :].expand_as(batch.theta)
    theta = torch.where(absent_p, 0.7, batch.theta)
    psi = torch.where(g.global_par < 0, -0.3, batch.psi)
    relabelled = dataclasses.replace(g, features=g.features.clone())
    relabelled.features[..., 15:17] = 9.0                   # sign and log10|c|: recoded away
    with torch.no_grad():
        a = module.backbone(batch.theta, batch.psi, batch.sc, g)
        for args in ((batch.theta, batch.psi, batch.sc, padded), (theta, psi, batch.sc, g),
                     (batch.theta, batch.psi, batch.sc, relabelled)):
            b = module.backbone(*args)
            torch.testing.assert_close(a.h_cls, b.h_cls, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(a.regions, b.regions, rtol=1e-4, atol=1e-5)
    assert (g.regional_par < 0).any() and (g.global_par < 0).any()      # the test bites


def test_parameter_tokens_read_global_values(ptok):
    """A global parameter's value and a constant's value both change the output. Only by
    attention, so weakly in an untrained network (~1e-4 of h_CLS, 2026-10-29): the threshold
    asks for more than float rounding, not for a size."""
    _, batch, module = ptok
    g = batch.graph
    psi = torch.where(g.global_par >= 0, batch.psi + 0.2, batch.psi)
    const = dataclasses.replace(g, const_value=torch.where(g.const_value != 0,
                                                           g.const_value + 0.2, 0.0))
    with torch.no_grad():
        a = module.backbone(batch.theta, batch.psi, batch.sc, g).h_cls
        b = module.backbone(batch.theta, psi, batch.sc, g).h_cls
        c = module.backbone(batch.theta, batch.psi, batch.sc, const).h_cls
    assert (a - b).abs().max() > 1e-5 and (a - c).abs().max() > 1e-5
    with pytest.raises(ValueError, match="Const values"):
        module.backbone(batch.theta, batch.psi, batch.sc, dataclasses.replace(g, const_value=None))


def test_parameter_tokens_do_not_depend_on_parameter_order(ptok):
    _, batch, module = ptok
    perm = torch.tensor([2, 0, 1, 3, 4])
    g = batch.graph
    permuted = dataclasses.replace(g, regional_par=g.regional_par[:, perm])
    with torch.no_grad():
        a = module.backbone(batch.theta, batch.psi, batch.sc, g)
        b = module.backbone(batch.theta[..., perm], batch.psi, batch.sc, permuted)
    torch.testing.assert_close(a.h_cls, b.h_cls, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a.regions, b.regions, rtol=1e-4, atol=1e-5)


def test_parameter_tokens_fit_one_batch_with_the_cost_loss(ptok, tmp_path):
    store, batch, _ = ptok
    torch.manual_seed(0)
    module = Stage1Module(_ptok_config(tmp_path, tmp_path, cost_loss=True), store.n_fcd_levels,
                          (0.4, 1.3)).train()
    module.log = module.log_dict = lambda *a, **k: None          # no trainer attached
    loss = module.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    optimiser = torch.optim.Adam(module.parameters(), lr=3e-3)
    edges = torch.arange(2278)
    history = []
    for _ in range(150):
        optimiser.zero_grad()
        losses = group_losses(module.predict_statistics(batch, edges), batch.targets, edges)
        sum(losses.values()).backward()
        optimiser.step()
        history.append({g: float(v.detach()) for g, v in losses.items()})
    first, last = history[0], history[-1]          # as test_the_network_can_fit_one_batch
    assert last["regional"] < 0.6 * first["regional"]
    assert last["fc_moments"] < 0.2 * first["fc_moments"]
    assert last["fcd"] < 0.2 * first["fcd"]
    assert last["pairwise"] < first["pairwise"]


def test_lightning_run_with_parameter_tokens(tiny_store_sinh, tiny_store, tmp_path):
    cfg, _, _ = tiny_store_sinh
    stage1 = _ptok_config(cfg.store_dir, tmp_path)
    config_file = tmp_path / "stage1.yaml"
    config_file.write_text("for the artifact\n")
    checkpoints = fit(stage1, config_file, run_name="ptok")
    from delssome_fm.train.evaluate import evaluate, load_stage1
    module = load_stage1(checkpoints / "last.ckpt")
    assert module.backbone.token_scheme == "parameter"
    metrics = evaluate(module, cfg.store_dir, "val")
    assert metrics["rows"] > 0 and np.isfinite(metrics["total"])
    # a multiplier-policy store has no constants on the parameters' scale: refused up front
    with pytest.raises(ValueError, match="sinh-policy store"):
        fit(_ptok_config(tiny_store[0].store_dir, tmp_path), config_file, run_name="refused")
