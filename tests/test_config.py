"""The strict YAML -> dataclass loader."""

from pathlib import Path

import pytest
import yaml

from conftest import DATA_CONFIG_PATH
from delssome_fm.config import (ClusterConfig, CorpusConfig, DataConfig, GeneralisationConfig,
                                 ReproduceConfig, SimConfig, SplitBounds, Stage1Config,
                                 TrainingDataConfig, load_config)


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


CONFIG_CLASSES = {"data": DataConfig, "sim": SimConfig, "cluster": ClusterConfig,
                  "corpus": CorpusConfig, "reproduce": ReproduceConfig,
                  "reproduce_once": ReproduceConfig, "training_data": TrainingDataConfig,
                  "stage1": Stage1Config, "stage1_smoke": Stage1Config,
                  "stage1_reg": Stage1Config, "stage1_offset": Stage1Config,
                  "stage1_offset_reg": Stage1Config, "generalisation": GeneralisationConfig,
                  "corpus_d10": CorpusConfig,
                  "training_data_d10": TrainingDataConfig,
                  "stage1_d10": Stage1Config}


@pytest.mark.parametrize("name,cls", sorted(CONFIG_CLASSES.items()))
def test_every_config_file_loads(name, cls):
    """Catches YAML pitfalls such as 1.0e6 (read as a string; write 1.0e+6)."""
    load_config(DATA_CONFIG_PATH.parent / f"{name}.yaml", cls)


def test_every_config_file_is_checked():
    files = {p.stem for p in DATA_CONFIG_PATH.parent.glob("*.yaml")}
    assert files == set(CONFIG_CLASSES), f"add {files - set(CONFIG_CLASSES)} to CONFIG_CLASSES"


def test_overrides_are_typed_and_strict(tmp_path):
    from delssome_fm.config import override, to_yaml
    cfg = load_config(DATA_CONFIG_PATH.parent / "stage1.yaml", Stage1Config)
    new = override(cfg, ["train_models=10000", "dropout=0.1", "fc_offset=true"])
    assert (new.train_models, new.dropout, new.fc_offset) == (10000, 0.1, True)
    assert new.store_dir == cfg.store_dir                       # untouched fields kept
    with pytest.raises(ValueError, match="field=value"):
        override(cfg, ["no_such_field=1"])
    with pytest.raises(TypeError):
        override(cfg, ["train_models=many"])
    path = tmp_path / "effective.yaml"                          # to_yaml round-trips
    path.write_text(to_yaml(new))
    assert load_config(path, Stage1Config) == new
