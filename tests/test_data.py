import pytest
import yaml

from ppe.data import CLASS_NAMES, dataset_root, train_val_dirs, write_yolo_yaml


@pytest.fixture
def root(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / "css" / split / "images").mkdir(parents=True)
    return tmp_path / "css"


def test_class_names_match_dataset_id_order():
    assert CLASS_NAMES == [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
    ]


def test_train_val_dirs_returns_train_and_valid_only(root):
    train, valid = train_val_dirs(root)
    assert (train.name, valid.name) == ("train", "valid")


def test_train_val_dirs_fails_clearly_when_a_split_is_missing(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="valid"):
        train_val_dirs(tmp_path)


def test_yolo_yaml_has_absolute_train_val_paths_and_no_test_key(root, tmp_path):
    out = write_yolo_yaml(root, tmp_path / "run")
    data = yaml.safe_load(out.read_text())
    assert data["train"] == str((root / "train").resolve())
    assert data["val"] == str((root / "valid").resolve())
    assert "test" not in data
    assert data["nc"] == 10
    assert data["names"] == CLASS_NAMES


def test_dataset_root_uses_data_dir_when_given(root):
    assert dataset_root({"data_dir": str(root), "dataset": {"id": None}}) == root


def test_dataset_root_without_registered_id_points_to_registration_script():
    with pytest.raises(ValueError, match="register_dataset.py"):
        dataset_root({"data_dir": None, "dataset": {"id": None}})
