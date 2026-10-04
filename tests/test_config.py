"""The strict YAML -> dataclass loader."""

from pathlib import Path

import pytest
import yaml

from conftest import DATA_CONFIG_PATH
from delssome_fm.config import (ClusterConfig, DataConfig, ReproduceConfig, SimConfig,
                                 SplitBounds, load_config)


def _write(tmp_path, raw) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _raw() -> dict:
    return yaml.safe_load(DATA_CONFIG_PATH.read_text())


def test_data_yaml_loads_into_typed_fields():
    cfg = load_config(DATA_CONFIG_PATH, DataConfig)
    assert isinstance(cfg.subject_list, Path)
    assert cfg.splits == SplitBounds(train=(0, 680), val=(680, 860), test=(860, 1029))
    assert (cfg.group_size, cfg.group_stride, cfg.n_subjects) == (50, 10, 1029)


def test_missing_field_raises(tmp_path):
    raw = _raw()
    del raw["group_stride"]
    with pytest.raises(KeyError, match="group_stride"):
        load_config(_write(tmp_path, raw), DataConfig)


def test_unknown_field_raises(tmp_path):
    raw = _raw()
    raw["splits"]["holdout"] = [0, 1]
    with pytest.raises(ValueError, match="holdout"):
        load_config(_write(tmp_path, raw), DataConfig)


@pytest.mark.parametrize("field,value", [("group_size", "50"), ("group_size", True),
                                         ("subject_list", 3)])
def test_wrong_type_raises(tmp_path, field, value):
    raw = _raw()
    raw[field] = value
    with pytest.raises(TypeError, match=field):
        load_config(_write(tmp_path, raw), DataConfig)


def test_wrong_tuple_length_raises(tmp_path):
    raw = _raw()
    raw["splits"]["val"] = [680, 860, 900]
    with pytest.raises(ValueError, match="splits.val"):
        load_config(_write(tmp_path, raw), DataConfig)


@pytest.mark.parametrize("name,cls", [("data", DataConfig), ("sim", SimConfig),
                                      ("cluster", ClusterConfig), ("reproduce", ReproduceConfig)])
def test_every_config_file_loads(name, cls):
    """Catches YAML pitfalls such as 1.0e6 (read as a string; write 1.0e+6)."""
    load_config(DATA_CONFIG_PATH.parent / f"{name}.yaml", cls)
