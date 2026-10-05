"""Submit one job to the CBIG PBS cluster through CBIG_pbsubmit.

`submit` turns a shell command, a job name and resource requests into a bash job script
(conda activation; for GPU jobs, `module load cuda/<version>`, echoed into the job log; then
the command), and calls CBIG_pbsubmit with that script's path. Passing a script path rather
than a compound command is what CBIG_pbsubmit requires when it has to forward a submission
from a compute node to the headnode.

CBIG_pbsubmit is always run on the headnode, where it calls qsub directly. Elsewhere (this
project's sandbox runs on `compiler`) the call is forwarded with `cluster.headnode_ssh`. The
script's own forwarding uses plain `ssh headnode`, which the sandbox rejects (its
/etc/ssh/ssh_config.d files show the wrong owner), and it cannot submit GPU jobs at all.

Dry run is the default. It prints the script and the command it would run, and writes
nothing.

This module does not decide what to run or how to split work into jobs; the corpus and
training code do that. It does not write job manifests either; each job writes its own.
"""

from __future__ import annotations

import re
import shlex
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from delssome_fm.config import ClusterConfig

GPU_TYPES: tuple[str, ...] = ("L40S", "H200", "H100")  # queues gpuQ-<type> in CBIG_pbsubmit
MAX_GPUS = 4  # per-user limit stated in CBIG_pbsubmit -help
_WALLTIME = re.compile(r"^\d{2,3}:[0-5]\d:[0-5]\d$")
_MEMORY = re.compile(r"^\d+(G|GB|M|MB)$")
_JOB_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,62}$")


@dataclass(frozen=True)
class Resources:
    """Resource request. `ngpus = 0` is a CPU job; `gpu_type` is used only when ngpus > 0."""

    walltime: str          # HH:MM:SS
    memory: str            # e.g. "16G"
    ngpus: int = 0
    ncpus: int = 1
    gpu_type: str = "L40S"

    def __post_init__(self) -> None:
        if not _WALLTIME.match(self.walltime):
            raise ValueError(f"walltime must be HH:MM:SS, found {self.walltime!r}")
        if not _MEMORY.match(self.memory):
            raise ValueError(f"memory must look like '16G' or '512MB', found {self.memory!r}")
        if not 0 <= self.ngpus <= MAX_GPUS:
            raise ValueError(f"ngpus must be in [0, {MAX_GPUS}], found {self.ngpus}")
        if self.ncpus < 1:
            raise ValueError(f"ncpus must be at least 1, found {self.ncpus}")
        if self.gpu_type not in GPU_TYPES:
            raise ValueError(f"gpu_type must be one of {GPU_TYPES}, found {self.gpu_type!r}")


@dataclass(frozen=True)
class Job:
    """Everything needed to submit: the script text, where it goes, and the submit argv."""

    name: str
    script_path: Path
    script: str
    argv: tuple[str, ...]


def build_job(command: str, name: str, resources: Resources, cluster: ClusterConfig,
              stamp: str) -> Job:
    """Pure: the job script and CBIG_pbsubmit argv. `stamp` keeps file names unique."""
    if not _JOB_NAME.match(name):
        raise ValueError(f"job name must match {_JOB_NAME.pattern}, found {name!r}")
    if not command.strip():
        raise ValueError("command is empty")
    stem = f"{name}_{stamp}"
    script_path = cluster.job_dir / f"{stem}.sh"
    lines = [
        "#!/bin/bash -l",  # login shell, so `module` is defined on the compute node
        "set -eo pipefail",
        f"echo \"[delssome_fm] job {name} on $(hostname) at $(date -Is)\"",
        f"source {shlex.quote(str(cluster.conda_init))}",
        f"conda activate {shlex.quote(cluster.conda_env)}",
    ]
    if resources.ngpus > 0:
        module = f"cuda/{cluster.cuda_version}"
        lines += [f"module load {module}", f"echo \"[delssome_fm] CUDA module: {module}\""]
    lines += [f"cd {shlex.quote(str(cluster.repo_dir))}", command, ""]
    argv = [str(cluster.pbsubmit), "-cmd", str(script_path),
            "-walltime", resources.walltime, "-mem", resources.memory,
            "-ncpus", str(resources.ncpus), "-name", name,
            "-joberr", str(cluster.job_dir / f"{stem}.err"),
            "-jobout", str(cluster.job_dir / f"{stem}.out")]
    if resources.ngpus > 0:
        argv += ["-ngpus", str(resources.ngpus), "-gpu_type", resources.gpu_type]
    return Job(name=name, script_path=script_path, script="\n".join(lines), argv=tuple(argv))


def submit(command: str, name: str, resources: Resources, cluster: ClusterConfig,
           dry_run: bool = True) -> Job:
    """Submit `command` as one job. With dry_run (the default) only print what would happen."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    job = build_job(command, name, resources, cluster, stamp)
    argv = submission_argv(job, cluster, socket.gethostname())
    if dry_run:
        print(f"# dry run: would write {job.script_path}\n{job.script}")
        print("# dry run: would run\n" + shlex.join(argv))
        return job
    if not cluster.pbsubmit.is_file():
        raise FileNotFoundError(f"CBIG_pbsubmit not found at {cluster.pbsubmit}")
    cluster.job_dir.mkdir(parents=True, exist_ok=True)
    if job.script_path.exists():
        raise FileExistsError(f"job script already exists: {job.script_path}")
    job.script_path.write_text(job.script)
    job.script_path.chmod(0o755)
    subprocess.run(argv, check=True)
    return job


def submission_argv(job: Job, cluster: ClusterConfig, host: str) -> list[str]:
    """The CBIG_pbsubmit call as run on the headnode: directly there, via ssh elsewhere."""
    if host == "headnode":
        return list(job.argv)
    return [*cluster.headnode_ssh, shlex.join(job.argv)]
