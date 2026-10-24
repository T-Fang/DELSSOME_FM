"""Stage-1 test on the reference models, step 1 of 2 (sim/reference_eval.py): simulate the
fitted test sets of each reference model with the corpus protocol and pack them into a store.
About 300 simulations; run it as a CPU job:

    python scripts/submit_job.py --name reference_eval --walltime 04:00:00 --mem 16G \\
        --submit -- python scripts/reference_eval_data.py
    then: scripts/plot_reference_eval.py (training env)
"""

import argparse
import json
from pathlib import Path

from delssome_fm.config import (CorpusConfig, ReferenceEvalConfig, ReproduceConfig, SimConfig,
                                load_config)
from delssome_fm.sim.reference_eval import pack_model, simulate_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/reference_eval.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config, ReferenceEvalConfig)
    corpus_cfg = load_config(cfg.corpus_config, CorpusConfig)
    sim_cfg = load_config(Path("configs/sim.yaml"), SimConfig)
    rcfg = load_config(Path("configs/reproduce.yaml"), ReproduceConfig)
    summary = {}
    for name in cfg.models:
        s = simulate_model(corpus_cfg, sim_cfg, cfg, rcfg, name)
        packed = pack_model(cfg, sim_cfg, name, s.pop("sc"), corpus_cfg.parameter_policy)
        summary[name] = {**s, "rows_per_repeat": [r["rows"] for r in packed]}
        print(json.dumps(summary[name]))
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
