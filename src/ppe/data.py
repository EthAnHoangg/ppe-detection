"""Dataset access. Training code only ever sees the train and valid splits."""

from pathlib import Path

import yaml

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


def fetch_dataset(dataset_id: str) -> Path:
    """Download (or reuse the cached) ClearML dataset version; the copy is read-only."""
    from clearml import Dataset

    return Path(Dataset.get(dataset_id=dataset_id).get_local_copy())


def dataset_root(cfg: dict) -> Path:
    """Local `--data-dir` for smoke runs, otherwise the ClearML dataset named in the config."""
    if cfg["data_dir"]:
        return Path(cfg["data_dir"])
    if not cfg["dataset"]["id"]:
        raise ValueError("configs/common.yaml has no dataset.id; run scripts/register_dataset.py first and commit the ID")
    return fetch_dataset(cfg["dataset"]["id"])


def train_val_dirs(root: Path) -> tuple[Path, Path]:
    train, valid = Path(root) / "train", Path(root) / "valid"
    for split in (train, valid):
        if not split.is_dir():
            raise FileNotFoundError(f"dataset split missing: {split}")
    return train, valid


def write_yolo_yaml(root: Path, out_dir: Path) -> Path:
    """Write the Ultralytics data YAML into `out_dir` (the dataset cache is read-only). No test key, on purpose."""
    train, valid = train_val_dirs(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {"train": str(train.resolve()), "val": str(valid.resolve()), "nc": len(CLASS_NAMES), "names": CLASS_NAMES},
            sort_keys=False,
        )
    )
    return path
