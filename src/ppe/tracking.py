"""The single place that opens a ClearML run, so every run follows the same conventions."""

import os
import subprocess
import warnings
from collections.abc import Mapping

REQUIRED_KEYS = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")


def run_identity(cfg: dict) -> tuple[str, str, list[str]]:
    """Project, run name and tags. The member is not in the name: ClearML records the user."""
    project = cfg["project"] + ("/smoke" if cfg["smoke"]["enabled"] else "")
    name = f"{cfg['model']['name']} · seed{cfg['seed']}"
    tags = [f"model:{cfg['model']['name']}", f"data:{cfg['dataset']['version']}", f"seed:{cfg['seed']}"]
    if cfg["smoke"]["enabled"]:
        tags.append("smoke")
    return project, name, tags


def check_credentials(env: Mapping[str, str], offline: bool) -> None:
    if offline:
        return
    missing = [key for key in REQUIRED_KEYS if not env.get(key)]
    if missing:
        raise RuntimeError(
            f"ClearML credentials missing: {', '.join(missing)}. "
            "Set them (and the CLEARML_*_HOST variables) as shown in .env.example."
        )


def _warn_if_dirty() -> None:
    try:
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    if status.strip():
        warnings.warn(
            "Uncommitted changes: ClearML stores the diff, but runs tagged 'report' should come from a clean commit.",
            stacklevel=2,
        )


def start_run(cfg: dict):
    """Open the ClearML Task for this run and attach the merged config as its hyperparameters."""
    from clearml import Task

    check_credentials(os.environ, offline=Task.is_offline())
    _warn_if_dirty()
    project, name, tags = run_identity(cfg)
    task = Task.init(
        project_name=project,
        task_name=name,
        tags=tags,
        output_uri=True,
        reuse_last_task_id=False,
        # Weights and plots are uploaded explicitly (by Ultralytics' callback or our loop), not on every torch.save.
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},
    )
    task.connect(cfg, name="config")
    return task
