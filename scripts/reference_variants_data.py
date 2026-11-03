"""Reference-model variants (sim/reference_eval.py, 2026-11-04): simulate one chunk of draws of
a variant (scrambled: the fitted sets with i.i.d. regional maps; random: corpus draws of the
card), or pack a variant's chunks into store split <model>_<variant> of the reference store.
Run the chunks as CPU jobs, then pack:

    python scripts/reference_variants_data.py simulate --model fic --variant random \\
        --first 0 --n 100
    python scripts/reference_variants_data.py pack --model fic --variant random --n-draws 5000
    then: scripts/reference_variants_compare.py (training env)
"""

import argparse
import json
from pathlib import Path

from delssome_fm.config import (CorpusConfig, ReferenceEvalConfig, ReproduceConfig, SimConfig,
                                load_config)
from delssome_fm.sim.reference_eval import VARIANTS, pack_variant, simulate_variant


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=("simulate", "pack"))
    p.add_argument("--config", type=Path, default=Path("configs/reference_eval_sinh.yaml"))
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--first", type=int, default=0, help="simulate: first draw")
    p.add_argument("--n", type=int, default=100, help="simulate: number of draws")
    p.add_argument("--n-draws", type=int, default=None, help="pack: draws 0..n-1 expected")
    args = p.parse_args()

    cfg = load_config(args.config, ReferenceEvalConfig)
    sim_cfg = load_config(Path("configs/sim.yaml"), SimConfig)
    rcfg = load_config(Path("configs/reproduce.yaml"), ReproduceConfig)
    if args.action == "simulate":
        corpus_cfg = load_config(cfg.corpus_config, CorpusConfig)
        print(json.dumps(simulate_variant(corpus_cfg, sim_cfg, cfg, rcfg, args.model,
                                          args.variant, args.first, args.n)))
    else:
        if args.n_draws is None:
            p.error("pack needs --n-draws")
        print(json.dumps(pack_variant(cfg, sim_cfg, rcfg, args.model, args.variant,
                                      args.n_draws)))


if __name__ == "__main__":
    main()
