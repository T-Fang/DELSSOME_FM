"""Analytic features (train/analytic.analytic_single_model) for the reference-model stores of
scripts/reference_eval_data.py and reference_variants_data.py, so that a network reading
analytic inputs can be evaluated on them. One split per call, or every split with --all
(--procs local processes).

    python scripts/analytic_reference.py --all --procs 8
    python scripts/analytic_reference.py --split mfm_r0
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

from delssome_fm.config import ReferenceEvalConfig, SimConfig, load_config
from delssome_fm.spec.card import card_from_dict
from delssome_fm.train.analytic import analytic_single_model


def _split(job: tuple) -> tuple[str, dict]:
    config, sim_config, split, surrogates = job
    cfg = load_config(config, ReferenceEvalConfig)
    name = split.split("_")[0]
    line = (cfg.output_dir / "chunks" / f"{name}_r0.cards.jsonl").read_text().splitlines()[0]
    seed = [cfg.seed, cfg.models.index(name), sum(split.encode())]      # one stream per split
    return split, analytic_single_model(cfg.output_dir / "store" / split,
                                        card_from_dict(json.loads(line)["card"]),
                                        load_config(sim_config, SimConfig), seed, surrogates)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reference_eval_sinh.yaml"))
    p.add_argument("--sim-config", type=Path, default=Path("configs/sim.yaml"))
    p.add_argument("--split", default=None, help="e.g. mfm_r0")
    p.add_argument("--all", action="store_true")
    p.add_argument("--surrogates", type=int, default=4)
    p.add_argument("--procs", type=int, default=1)
    args = p.parse_args()
    cfg = load_config(args.config, ReferenceEvalConfig)
    store = cfg.output_dir / "store"
    if args.all == (args.split is not None):
        p.error("give --split or --all")
    splits = (sorted(d.name for d in store.iterdir() if (d / "summary.json").is_file())
              if args.all else [args.split])
    jobs = [(args.config, args.sim_config, s, args.surrogates) for s in splits]
    with get_context("spawn").Pool(args.procs) as pool:
        for split, shares in pool.imap_unordered(_split, jobs):
            print(split, {k: round(v, 3) for k, v in shares.items()}, flush=True)


if __name__ == "__main__":
    main()
