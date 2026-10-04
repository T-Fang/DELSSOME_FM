"""Submit one shell command as a cluster job. Prints a dry run unless --submit is given.

    python scripts/submit_job.py --name verify --walltime 01:00:00 --mem 16G \\
        -- python scripts/verify_groups.py
"""

import argparse
import shlex
from pathlib import Path

from delssome_fm.cluster.submit import Resources, submit
from delssome_fm.config import ClusterConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/cluster.yaml"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--walltime", required=True, help="HH:MM:SS")
    parser.add_argument("--mem", required=True, help="e.g. 16G")
    parser.add_argument("--ngpus", type=int, default=0)
    parser.add_argument("--ncpus", type=int, default=1)
    parser.add_argument("--gpu-type", default="L40S")
    parser.add_argument("--submit", action="store_true", help="actually submit (default: dry run)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="after --, the command to run")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    resources = Resources(walltime=args.walltime, memory=args.mem, ngpus=args.ngpus,
                          ncpus=args.ncpus, gpu_type=args.gpu_type)
    submit(shlex.join(command), args.name, resources, load_config(args.config, ClusterConfig),
           dry_run=not args.submit)


if __name__ == "__main__":
    main()
