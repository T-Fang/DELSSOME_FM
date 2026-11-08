"""Run the after-training evaluation (train/fit.final_evaluation) on an existing checkpoint
and log it into that run's MLflow record, for runs trained before it existed. Use a GPU job:

    python scripts/submit_job.py --name final_eval --walltime 02:00:00 --mem 64G --ncpus 9 \\
        --ngpus 1 --conda-env delssome_fm_train --submit -- \\
        python scripts/evaluate_final.py --run-name d10_n300k_natural \\
        --checkpoint outputs/stage1/d10_n300k_natural/step80000.ckpt

Every val and test row, leaving final/train/* as it was (2026-11-08):

    ... python scripts/evaluate_final.py --run-name sinh_orig \
        --checkpoint outputs/stage1/sinh_orig/step140000.ckpt --splits val test \
        --set final_eval_rows=null
"""

import argparse
from pathlib import Path

import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger

from delssome_fm.config import override
from delssome_fm.train.data import Stage1DataModule
from delssome_fm.train.evaluate import load_stage1
from delssome_fm.train.fit import FINAL_SPLITS, final_evaluation
from delssome_fm.train.report import training_composition


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-name", required=True, help="the MLflow run to log into")
    p.add_argument("--coupled", action="store_true",
                   help="evaluate coupled draws only, logged as final_coupled/<split>/*")
    p.add_argument("--splits", nargs="+", default=list(FINAL_SPLITS), choices=FINAL_SPLITS,
                   help="splits to evaluate (val and test always; default all three)")
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="evaluate under other settings than the run's, e.g. final_eval_rows=null "
                        "cma_draws=true (all rows, with the CMA-ES draws; 2026-11-05)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = load_stage1(args.checkpoint).to(device)
    module.cfg = override(module.cfg, args.set)
    cfg = module.cfg
    client = mlflow.MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    runs = client.search_runs([client.get_experiment_by_name(cfg.mlflow_experiment).experiment_id],
                              filter_string=f"attributes.run_name = '{args.run_name}'")
    if len(runs) != 1:
        raise ValueError(f"expected one MLflow run named {args.run_name!r}, found {len(runs)}")
    logger = MLFlowLogger(experiment_name=cfg.mlflow_experiment, tracking_uri=cfg.mlflow_tracking_uri,
                          run_id=runs[0].info.run_id)
    data = Stage1DataModule(cfg.store_dir, cfg.batch_size, cfg.val_rows, cfg.num_workers, cfg.seed,
                            training_composition(module), cma=cfg.cma_draws)
    data.setup()
    summary = final_evaluation(module, data, logger, args.checkpoint.parent, coupled=args.coupled,
                               splits=tuple(args.splits))
    for split, s in summary.items():
        print(split, {k: round(v, 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
