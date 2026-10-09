"""cluster/submit.py. Nothing here submits a job: subprocess.run is replaced so a call fails."""

import shlex

from pathlib import Path

import pytest

from conftest import REPO_ROOT
from delssome_fm.cluster import submit as submit_mod
from delssome_fm.cluster.submit import Resources, build_job, submit
from delssome_fm.config import ClusterConfig, load_config


@pytest.fixture
def cluster(tmp_path) -> ClusterConfig:
    cfg = load_config(REPO_ROOT / "configs" / "cluster.yaml", ClusterConfig)
    return ClusterConfig(pbsubmit=cfg.pbsubmit, conda_init=cfg.conda_init,
                         conda_env=cfg.conda_env, cuda_version="12.9",
                         repo_dir=cfg.repo_dir, job_dir=tmp_path / "jobs",
                         headnode_ssh=cfg.headnode_ssh)


@pytest.fixture(autouse=True)
def no_subprocess(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to call CBIG_pbsubmit")
    monkeypatch.setattr(submit_mod.subprocess, "run", refuse)


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def test_cpu_job_argv_and_script(cluster):
    job = build_job("python scripts/verify_groups.py", "verify", Resources("02:00:00", "16G"),
                    cluster, stamp="T0")
    argv = list(job.argv)
    assert argv[0] == str(cluster.pbsubmit)
    assert _flag(argv, "-cmd") == str(cluster.job_dir / "verify_T0.sh")
    assert (_flag(argv, "-walltime"), _flag(argv, "-mem"), _flag(argv, "-name")) == \
        ("02:00:00", "16G", "verify")
    assert _flag(argv, "-joberr").endswith("verify_T0.err")
    assert "-ngpus" not in argv
    assert "module load" not in job.script
    assert job.script.startswith("#!/bin/bash -l")
    assert job.script.rstrip().endswith("python scripts/verify_groups.py")
    assert f"conda activate {cluster.conda_env}" in job.script


def test_gpu_job_loads_and_reports_cuda_module(cluster):
    job = build_job("python -c 'import jax'", "gpu", Resources("00:30:00", "32G", ngpus=1,
                    gpu_type="H100"), cluster, stamp="T0")
    argv = list(job.argv)
    assert (_flag(argv, "-ngpus"), _flag(argv, "-gpu_type")) == ("1", "H100")
    lines = job.script.splitlines()
    load = lines.index("module load cuda/12.9")
    assert lines[load + 1] == 'echo "[delssome_fm] CUDA module: cuda/12.9"'
    assert load < lines.index("python -c 'import jax'")


def test_dry_run_is_default_and_writes_nothing(cluster, capsys):
    job = submit("echo hi", "dry", Resources("00:01:00", "1G"), cluster)
    assert not job.script_path.exists() and not cluster.job_dir.exists()
    out = capsys.readouterr().out
    assert "dry run" in out and str(cluster.pbsubmit) in out


def test_off_headnode_the_call_is_forwarded_over_ssh(cluster):
    from delssome_fm.cluster.submit import submission_argv
    job = build_job("echo hi", "gpu", Resources("00:01:00", "1G", ngpus=1), cluster, stamp="T0")
    remote = submission_argv(job, cluster, host="compiler")
    assert remote[:len(cluster.headnode_ssh)] == list(cluster.headnode_ssh)
    assert shlex.split(remote[-1]) == list(job.argv)  # the whole pbsubmit call, quoted once
    assert submission_argv(job, cluster, host="headnode") == list(job.argv)


@pytest.mark.parametrize("kwargs", [dict(walltime="2:00", memory="1G"),
                                    dict(walltime="01:00:00", memory="1 GB"),
                                    dict(walltime="01:00:00", memory="1G", ngpus=5),
                                    dict(walltime="01:00:00", memory="1G", gpu_type="A100")])
def test_invalid_resources_raise(kwargs):
    with pytest.raises(ValueError):
        Resources(**kwargs)


def test_invalid_job_name_raises(cluster):
    with pytest.raises(ValueError, match="job name"):
        build_job("echo", "bad name", Resources("00:01:00", "1G"), cluster, stamp="T0")


def test_pbsubmit_path_in_config_exists():
    cfg = load_config(REPO_ROOT / "configs" / "cluster.yaml", ClusterConfig)
    if not Path("/home/ftian/storage").exists():
        pytest.skip("lab storage not mounted")
    assert cfg.pbsubmit.is_file() and cfg.conda_init.is_file()


def test_batch_dry_run_writes_nothing_and_uses_one_remote_call(cluster, capsys, monkeypatch):
    from delssome_fm.cluster.submit import submit_batch
    monkeypatch.setattr(submit_mod.socket, "gethostname", lambda: "compiler")
    reqs = [(f"echo {i}", f"job{i}", Resources("00:01:00", "1G")) for i in range(3)]
    jobs = submit_batch(reqs, cluster)
    assert len(jobs) == 3 and len({j.script_path for j in jobs}) == 3
    assert not cluster.job_dir.exists()
    out = capsys.readouterr().out
    remote = out.strip().splitlines()[-1]
    assert remote.startswith(shlex.join(cluster.headnode_ssh)) and "bash" in remote
