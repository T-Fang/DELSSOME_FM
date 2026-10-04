"""The manifest every job writes next to its output (brief §10.3): the config files, the
seed, the git commit and the wall time, which together make the output reproducible.

It refuses to record an unknown commit: running outside a git repository raises. A dirty
working tree is allowed but recorded, with the list of modified files.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_state(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {out.stderr.strip()}")
        return out.stdout.strip()

    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty), "modified": dirty}


def write_manifest(path: Path, configs: list[Path], seed: int, wall_time_s: float,
                   extra: dict[str, Any] | None = None) -> None:
    """Write `path` (JSON). `configs` are copied in verbatim, not referenced."""
    path = Path(path)
    repo = Path(__file__).resolve().parents[3]
    manifest = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "git": git_state(repo),
        "seed": seed,
        "wall_time_s": wall_time_s,
        "configs": {str(p): Path(p).read_text() for p in configs},
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
