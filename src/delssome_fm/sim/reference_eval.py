"""Stage-1 test on the reference models (2026-10-19): simulate MFM, FIC and Hopf at their
published test fits (the 50 sets of build step 6) with the corpus protocol, so that a stage-1
model's predictions can be compared with numerical simulations of real models.

The network reads each free parameter as log10 of a positive multiplier of the constant in
its card slot, the convention of the synthetic corpus (sampler.py: the sign and magnitude live
in the constant, the multiplier is 10^U(-1, 1)). The reference cards carry published values
instead (MFM's drive and Hopf's a are negative; sigma ~ 0.005 with constant 1). `in_convention`
re-expresses a card and its fitted sets in the corpus convention: every free parameter's
constant is multiplied by m = the signed median of its fitted values, and the parameter
becomes value / m. The equations are unchanged. A set with any region whose value has the
other sign than the median cannot be represented (a corpus parameter never changes sign) and
is excluded; values beyond 0.1-10x the median are kept and counted as outside the training
range.

That is the multiplier policy (corpus_d10). Under the sinh policy (corpus_sinh, 2026-10-25)
the network reads each region's actual coefficient, so `actual_values` only folds each free
parameter's card scale into its values (a free slot's scale becomes +-1, as in the sinh
corpus) and every fitted set is usable, MFM's sign-changing ones included.

Each usable set is simulated `n_repeats` times with independent noise and initial states
(x0 ~ U(-s, s) as in the corpus) on the test group's SC, and written as a corpus-style chunk
per repeat, so train/pack.pack_chunks packs it into a store split <model>_r<k>.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import jax
import numpy as np

from delssome_fm.config import CorpusConfig, ReferenceEvalConfig, ReproduceConfig, SimConfig
from delssome_fm.sim import reproduce
from delssome_fm.sim.corpus import simulate_given
from delssome_fm.spec.card import Coef, ModelCard, card_to_dict
from delssome_fm.spec.compile import compile_card
from delssome_fm.spec.reference import load_reference


def fitted_sets(rcfg: ReproduceConfig, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The model's fitted test sets in card units: theta (S, N, P), psi (S, K), and the test
    SC (N, N) rescaled to max 0.02 (sim/reproduce.py)."""
    model = next(m for m in rcfg.models if m.name == name)
    inputs = reproduce.load_test_inputs(rcfg)
    original = reproduce.load_original(rcfg, model)
    sets = [reproduce.card_inputs(name, original.params[s], inputs.sc) for s in range(rcfg.n_sets)]
    return np.stack([t for t, _ in sets]), np.stack([p for _, p in sets]), inputs.sc


def in_convention(card: ModelCard, theta: np.ndarray, psi: np.ndarray,
                  ) -> tuple[ModelCard, np.ndarray, np.ndarray, dict[str, float]]:
    """The card with each free parameter's constants multiplied by the signed median of its
    fitted values, and theta, psi divided by it (module docstring). Returns the new card,
    theta, psi and the medians by parameter name."""
    medians = {p: float(np.median(theta[..., j])) for j, p in enumerate(card.regional_params)}
    medians.update({p: float(np.median(psi[:, k])) for k, p in enumerate(card.global_params)})
    if any(m == 0 for m in medians.values()):
        raise ValueError(f"{card.name}: a parameter's median is 0: {medians}")
    scale = [medians[p] for p in card.regional_params]
    return (_rescale(card, medians), theta / np.array(scale),
            psi / np.array([medians[p] for p in card.global_params]), medians)


def actual_values(card: ModelCard, theta: np.ndarray, psi: np.ndarray,
                  ) -> tuple[ModelCard, np.ndarray, np.ndarray, dict[str, float]]:
    """The card with each free parameter's slot scales divided by m_p (the scale of its first
    slot), and theta, psi multiplied by it: the parameter now holds the actual coefficient
    (sinh policy), every slot scale is +-1, and the equations are unchanged. Raises if a
    parameter's slots differ in |scale| (no single actual coefficient). Returns the new card,
    theta, psi and the m_p by parameter name."""
    slots: dict[str, list[float]] = {}
    _collect_scales(card, slots)
    if any(not np.allclose(np.abs(s), abs(s[0])) for s in slots.values()):
        raise ValueError(f"{card.name}: a parameter's slots differ in |scale|: {slots}")
    m = {p: s[0] for p, s in slots.items()}
    return (_rescale(card, {p: 1.0 / v for p, v in m.items()}),
            theta * np.array([m[p] for p in card.regional_params]),
            psi * np.array([m[p] for p in card.global_params]), m)


def _collect_scales(x, slots: dict[str, list[float]]) -> None:
    if isinstance(x, Coef):
        if x.param:
            slots.setdefault(x.param, []).append(x.scale)
    elif isinstance(x, tuple):
        for v in x:
            _collect_scales(v, slots)
    elif dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            _collect_scales(getattr(x, f.name), slots)


def _rescale(x, medians: dict[str, float]):
    """Every Coef(param=p) in a card, with its scale multiplied by medians[p]."""
    if isinstance(x, Coef):
        return dataclasses.replace(x, scale=x.scale * medians[x.param]) if x.param else x
    if isinstance(x, tuple):
        return tuple(_rescale(v, medians) for v in x)
    if dataclasses.is_dataclass(x):
        return dataclasses.replace(x, **{f.name: _rescale(getattr(x, f.name), medians)
                                         for f in dataclasses.fields(x)})
    return x


def simulate_model(corpus_cfg: CorpusConfig, sim_cfg: SimConfig, cfg: ReferenceEvalConfig,
                   rcfg: ReproduceConfig, name: str) -> dict:
    """Simulate one reference model's usable fitted sets n_repeats times and write chunk
    <output_dir>/chunks/<name>_r<k>.npz (+ .cards.jsonl) per repeat; returns a summary."""
    theta, psi, sc = fitted_sets(rcfg, name)
    if corpus_cfg.parameter_policy == "sinh":
        card, theta, psi, factors = actual_values(load_reference(name), theta, psi)
        usable = np.ones(len(theta), bool)
    else:
        card, theta, psi, factors = in_convention(load_reference(name), theta, psi)
        usable = (theta > 0).all(axis=(1, 2)) & (psi > 0).all(axis=1)
    sets = np.flatnonzero(usable)
    model = compile_card(card)
    out = cfg.output_dir / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    code = cfg.models.index(name)
    for k in range(cfg.n_repeats):
        x0 = np.stack([np.random.default_rng([cfg.seed, code, k, s]).uniform(
            -corpus_cfg.initial_state_scale, corpus_cfg.initial_state_scale,
            (sc.shape[0], card.n_states)) for s in sets])
        base = jax.random.fold_in(jax.random.fold_in(jax.random.key(cfg.seed), code), k)
        keys = jax.numpy.stack([jax.random.fold_in(base, int(s)) for s in sets])
        res = simulate_given(dataclasses.replace(corpus_cfg, batch_size=len(sets)), sim_cfg,
                             model, theta[sets], psi[sets], x0, keys, sc)
        res.update(model=np.zeros(len(sets), np.int64), draw=sets)
        path = out / f"{name}_r{k}"
        np.savez(path.with_suffix(".npz"), **res)
        path.with_suffix(".cards.jsonl").write_text(json.dumps(
            {"split": "reference", "index": 0, "sc_group": 0, "card": card_to_dict(card)}) + "\n")
    t = np.abs(theta[sets])
    return {"model": name, "sets": int(len(sets)), "excluded_sets": int((~usable).sum()),
            "parameter_policy": corpus_cfg.parameter_policy, "factors": factors, "sc": sc,
            "log10_abs_theta_range": [float(np.log10(t).min()), float(np.log10(t).max())]}


def pack_model(cfg: ReferenceEvalConfig, sim_cfg: SimConfig, name: str, sc: np.ndarray,
               parameter_policy: str) -> list:
    """Pack each repeat's chunk into split <name>_r<k> of the store <output_dir>/store, z-scored
    with cfg.training_store's statistics (train/pack.pack_chunks)."""
    from delssome_fm.train.pack import n_window_pairs, pack_chunks
    store = cfg.output_dir / "store"
    out = [pack_chunks([cfg.output_dir / "chunks" / f"{name}_r{k}.npz"], [0],
                       n_window_pairs(sim_cfg), store / f"{name}_r{k}", sc[None],
                       {"model": name, "repeat": k, "parameter_policy": parameter_policy})
           for k in range(cfg.n_repeats)]
    shutil.copy(cfg.training_store / "normalisation.npz", store / "normalisation.npz")
    return out
