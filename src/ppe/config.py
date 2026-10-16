"""Load a model config merged over `common.yaml`, plus the few allowed CLI overrides."""

import argparse
import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    model_config: Path,
    seed: int | None = None,
    epochs: int | None = None,
    smoke: bool = False,
    data_dir: Path | None = None,
) -> dict:
    """Merge `model_config` over `common.yaml` in the same directory and apply overrides.

    Everything else is changed by editing a config file, so each run's settings trace to a commit.
    """
    model_config = Path(model_config)
    common = yaml.safe_load((model_config.parent / "common.yaml").read_text())
    cfg = _deep_merge(common, yaml.safe_load(model_config.read_text()))

    cfg["seed"] = cfg["seeds"][0] if seed is None else seed
    if not smoke and cfg["seed"] not in cfg["seeds"]:
        raise ValueError(f"seed {cfg['seed']} is not in common.seeds {cfg['seeds']}; add it there or use --smoke")
    if data_dir is not None and not smoke:
        raise ValueError("--data-dir is only allowed with --smoke; real runs fetch the dataset from ClearML")

    cfg["smoke"] = {**cfg.get("smoke", {}), "enabled": smoke}
    if smoke:
        cfg["epochs"] = 1
    if epochs is not None:
        cfg["epochs"] = epochs
    cfg["data_dir"] = None if data_dir is None else str(data_dir)
    return cfg


def parse_args(description: str) -> dict:
    """Shared CLI for the training entry points; returns the loaded config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="model config, e.g. configs/yolo26s.yaml")
    parser.add_argument("--seed", type=int, help="must be one of common.seeds unless --smoke")
    parser.add_argument("--epochs", type=int, help="override the epoch budget")
    parser.add_argument("--smoke", action="store_true", help="1 epoch on a few images, logged to the smoke sub-project")
    parser.add_argument("--data-dir", type=Path, help="local dataset root instead of ClearML (only with --smoke)")
    args = parser.parse_args()
    return load_config(args.config, seed=args.seed, epochs=args.epochs, smoke=args.smoke, data_dir=args.data_dir)
