from pathlib import Path

import pytest

from delssome_fm.config import DataConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"


@pytest.fixture(scope="session")
def data_cfg() -> DataConfig:
    return load_config(DATA_CONFIG_PATH, DataConfig)


def data_available() -> bool:
    """True when the real HCP-YA data is reachable (on the lab cluster or its NAS)."""
    cfg = load_config(DATA_CONFIG_PATH, DataConfig)
    return cfg.subject_list.is_file() and cfg.reference_fc_dir.is_dir()
