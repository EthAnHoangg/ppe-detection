from pathlib import Path

import pytest
import yaml

from ppe.config import load_config


@pytest.fixture
def configs(tmp_path):
    (tmp_path / "common.yaml").write_text(
        yaml.safe_dump(
            {
                "project": "PPE-Detection",
                "dataset": {"id": "abc123", "version": "v1"},
                "imgsz": 640,
                "seeds": [0, 1, 2],
                "epochs": 100,
                "patience": 20,
                "smoke": {"images": 16},
            }
        )
    )
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({"model": {"name": "yolo26s"}, "patience": 50, "dataset": {"version": "v2"}}))
    return model


def test_model_config_overrides_common_and_merges_nested_keys(configs):
    cfg = load_config(configs)
    assert cfg["patience"] == 50
    assert cfg["imgsz"] == 640
    assert cfg["dataset"] == {"id": "abc123", "version": "v2"}
    assert cfg["model"]["name"] == "yolo26s"


def test_seed_defaults_to_first_listed_seed(configs):
    assert load_config(configs)["seed"] == 0


def test_seed_and_epochs_overrides_apply(configs):
    cfg = load_config(configs, seed=2, epochs=5)
    assert cfg["seed"] == 2
    assert cfg["epochs"] == 5


def test_seed_outside_common_seeds_is_rejected(configs):
    with pytest.raises(ValueError, match="seed 7"):
        load_config(configs, seed=7)


def test_smoke_allows_any_seed_and_runs_one_epoch(configs):
    cfg = load_config(configs, seed=7, smoke=True)
    assert cfg["smoke"]["enabled"] is True
    assert cfg["seed"] == 7
    assert cfg["epochs"] == 1


def test_non_smoke_run_is_marked_not_smoke(configs):
    assert load_config(configs)["smoke"]["enabled"] is False


def test_data_dir_requires_smoke(configs, tmp_path):
    with pytest.raises(ValueError, match="--data-dir"):
        load_config(configs, data_dir=tmp_path)
    assert load_config(configs, smoke=True, data_dir=tmp_path)["data_dir"] == str(tmp_path)


def test_committed_configs_load_with_a_valid_default_seed():
    for name in ("yolo26s.yaml", "fasterrcnn.yaml"):
        cfg = load_config(Path(__file__).parent.parent / "configs" / name)
        assert cfg["seed"] in cfg["seeds"]
        assert cfg["imgsz"] == 640
