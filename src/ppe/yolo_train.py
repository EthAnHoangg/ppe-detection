"""Train YOLO26 on the Construction Site Safety data, logging to ClearML.

    uv run python -m ppe.yolo_train --config configs/yolo26s.yaml --seed 0

Ultralytics' ClearML callback attaches to the run opened by `start_run` and logs metrics, plots and best.pt.
"""

from pathlib import Path

from ppe.config import parse_args
from ppe.data import dataset_root, train_val_dirs, write_yolo_yaml
from ppe.evaluate import run_validation_eval
from ppe.predict import YoloPredictor
from ppe.seed import seed_everything
from ppe.tracking import start_run


def train_kwargs(cfg: dict, data_yaml: Path, run_dir: Path) -> dict:
    yolo = cfg["yolo"]
    kwargs = {
        "data": str(data_yaml),
        "imgsz": cfg["imgsz"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "seed": cfg["seed"],
        "deterministic": True,
        "batch": yolo["batch"],
        "workers": yolo["workers"],
        "project": str(run_dir),
        "name": f"{cfg['model']['name']}-seed{cfg['seed']}",
        **yolo.get("train_args", {}),
    }
    if cfg["smoke"]["enabled"]:
        kwargs["fraction"] = [cfg["smoke"]["images"], cfg["smoke"]["images"]]
    return kwargs


def main() -> None:
    cfg = parse_args("Train YOLO26 with ClearML tracking")
    task = start_run(cfg)
    seed_everything(cfg["seed"])
    run_dir = Path("runs").resolve() / task.id
    root = dataset_root(cfg)
    data_yaml = write_yolo_yaml(root, run_dir)

    from ultralytics import YOLO

    model = YOLO(cfg["yolo"]["weights"])
    try:
        model.train(**train_kwargs(cfg, data_yaml, run_dir))
    except AttributeError as err:
        # ClearML offline mode can't register output models; Ultralytics tries to at train end, after saving weights.
        if not (task.is_offline() and "DummyModel" in str(err)):
            raise
        print(f"offline: training finished, model not registered in ClearML (weights are under {run_dir})")

    # Best weights are saved (and, online, uploaded) before this point, so an evaluation failure can't lose them.
    _, valid_dir = train_val_dirs(root)
    run_validation_eval(task, cfg, YoloPredictor.load(Path(model.trainer.best), cfg["imgsz"]), valid_dir)

if __name__ == "__main__":
    main()
