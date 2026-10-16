"""Score a trained model on a split with the shared evaluator, and log the results to ClearML.

    uv run python -m ppe.evaluate --task <training-run-id>        # (re-)run validation evaluation into that run
    uv run python -m ppe.evaluate --final --tag report            # final test evaluation, once per run
"""

import argparse
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import yaml
from PIL import Image

from ppe.predict import make_predictor
from ppe_detection_evaluation import EvaluationFramework, ModelConfig
from ppe_detection_evaluation.metrics import compute_latency

IMAGE_SIZE = (640, 640)
RETRYABLE = {"failed", "stopped"}


class Sample(NamedTuple):
    image_id: str
    image: np.ndarray  # RGB uint8 (H, W, 3)
    gt_boxes: np.ndarray  # normalised cx, cy, w, h
    gt_labels: np.ndarray  # 0-based class IDs


def _read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
    values = np.array([[float(v) for v in row[:5]] for row in rows])
    return values[:, 1:5], values[:, 0].astype(np.int64)


def read_split(split_dir: Path, limit: int | None = None) -> list[Sample]:
    samples = []
    for path in sorted((Path(split_dir) / "images").glob("*.jpg"))[:limit]:
        with Image.open(path) as img:
            image = np.asarray(img.convert("RGB"))
        if image.shape[1::-1] != IMAGE_SIZE:
            raise ValueError(f"{path.name} is {image.shape[1]}x{image.shape[0]}; the evaluator assumes {IMAGE_SIZE}")
        boxes, labels = _read_labels(Path(split_dir) / "labels" / f"{path.stem}.txt")
        samples.append(Sample(path.stem, image, boxes, labels))
    return samples


def time_predictions(predict: Callable, images: Sequence, *, warmup: int, n: int, sync: Callable[[], None]) -> list[float]:
    """Batch-1 latency in ms: `warmup` untimed calls, then `n` timed calls cycling through `images`."""
    timings = []
    for i in range(warmup + n):
        image = images[i % len(images)]
        sync()
        start = time.perf_counter()
        predict(image)
        sync()
        if i >= warmup:
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _default_sync() -> Callable[[], None]:
    import torch

    return torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)


def evaluate_split(predictor, split_dir: Path, model_name: str, *, latency_images: int, warmup: int,
                   limit: int | None = None, sync: Callable[[], None] | None = None) -> EvaluationFramework:
    samples = read_split(split_dir, limit)
    framework = EvaluationFramework(ModelConfig(model_name), image_size=IMAGE_SIZE)
    for s in samples:
        p = predictor.predict(s.image)
        framework.add(s.image_id, p.boxes, p.scores, p.labels, s.gt_boxes, s.gt_labels)

    timings = time_predictions(predictor.predict, [s.image for s in samples], warmup=warmup, n=latency_images,
                               sync=sync or _default_sync())
    per_image = defaultdict(list)
    for k, ms in enumerate(timings):
        per_image[(warmup + k) % len(samples)].append(ms)
    for index, values in per_image.items():
        framework.records[index].latency_ms = float(np.mean(values))

    framework.evaluate()
    # Latency stats over every timed call, not per-image means (images are timed unequally often).
    framework.results["latency"] = compute_latency([SimpleNamespace(latency_ms=t) for t in timings])
    return framework


def _num(value) -> float:
    return float("nan") if value is None else float(value)


def headline_metrics(results: dict) -> dict[str, float]:
    """Flat metric name -> value, shared by ClearML logging and the seed summary. None becomes NaN."""
    detection = results["detection"]
    metrics = {}
    for group in ("all", "ppe"):
        metrics[f"mAP50/{group}"] = _num(detection["groups"][group]["mAP50"])
        metrics[f"mAP50-95/{group}"] = _num(detection["groups"][group]["mAP"])
    for name, values in detection["per_class"].items():
        metrics[f"AP50/{name}"] = _num(values["AP50"])
    for key, value in results["violation"]["mean"].items():
        metrics[f"violation/{key}"] = _num(value)
    metrics["image_any/AP"] = _num(results["image_level"]["any"]["ap"])
    latency = results["latency"]
    if latency.get("num_images"):
        for key in ("mean_ms", "p95_ms", "fps"):
            metrics[f"latency/{key}"] = _num(latency[key])
    return metrics


def log_eval(task, framework: EvaluationFramework, prefix: str, threshold: float | None = None) -> dict[str, float]:
    """Headline scalars under '<prefix> eval' plus two JSON artifacts: results and per-image predictions."""
    metrics = headline_metrics(framework.results)
    logger = task.get_logger()
    for name, value in metrics.items():
        if not math.isnan(value):  # NaN = class absent from this split; ClearML charts can't show it
            logger.report_scalar(f"{prefix} eval", name, value, iteration=0)

    with tempfile.TemporaryDirectory() as tmp:
        saved = json.loads(framework.save(Path(tmp) / "full.json").read_text())
        parts = {
            f"{prefix}_eval": {"model": framework.config.name, "threshold": threshold,
                               "settings": saved["settings"], "results": saved["results"]},
            f"{prefix}_predictions": {"records": saved["records"]},
        }
        for name, payload in parts.items():
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(payload, indent=2))
            task.upload_artifact(name, artifact_object=str(path), wait_on_upload=True)
    return metrics


def run_validation_eval(task, cfg: dict, predictor, valid_dir: Path) -> dict[str, float]:
    """Score `predictor` on the validation split and log into the training run (smoke runs: few images, short timing)."""
    # Model uploads are asynchronous and ClearML drops pending ones if the process crashes,
    # so finish them before anything here can fail.
    task.flush(wait_for_uploads=True)
    smoke = cfg["smoke"]["enabled"]
    framework = evaluate_split(
        predictor, valid_dir, cfg["model"]["name"],
        latency_images=cfg["eval"]["smoke_latency_images"] if smoke else cfg["eval"]["latency_images"],
        warmup=1 if smoke else cfg["eval"]["latency_warmup"],
        limit=cfg["smoke"]["images"] if smoke else None,
    )
    return log_eval(task, framework, "valid")


def select_final_runs(runs: list[dict], existing: dict[str, list[str]], explicit: bool) -> tuple[list[str], list[tuple[str, str]]]:
    """Which training runs may get their one final test evaluation."""
    todo, skipped = [], []
    for run in runs:
        blocking = [s for s in existing.get(run["id"], []) if s not in RETRYABLE]
        if run["status"] != "completed":
            skipped.append((run["id"], f"not completed ({run['status']})"))
        elif not run["has_model"]:
            skipped.append((run["id"], "no output model"))
        elif run["smoke"] and not explicit:
            skipped.append((run["id"], "smoke run (pass it with --task to evaluate it)"))
        elif blocking:
            skipped.append((run["id"], f"test set already used for this run (final-eval {blocking[0]})"))
        else:
            todo.append(run["id"])
    return todo, skipped


def _run_config(run) -> dict:
    return run.get_parameters_as_dict(cast=True)["config"]


def _run_info(run) -> dict:
    cfg = _run_config(run)
    return {"id": run.id, "status": str(run.get_status()), "smoke": bool(cfg["smoke"]["enabled"]),
            "has_model": bool(run.models["output"])}


def _final_eval_statuses(run_id: str) -> list[str]:
    from clearml import Task

    return [str(t.get_status()) for t in Task.get_tasks(tags=["final-eval"], task_filter={"parent": run_id})]


def _weights(run) -> Path:
    return Path(run.models["output"][-1].get_local_copy())


def _valid_threshold(run) -> float | None:
    if "valid_eval" not in run.artifacts:
        return None
    return json.loads(Path(run.artifacts["valid_eval"].get_local_copy()).read_text()).get("threshold")


def final_evaluate(run, eval_settings: dict) -> dict[str, float]:
    """One final test evaluation as a child task of `run`, using the run's own dataset version."""
    from clearml import Task

    from ppe.data import fetch_dataset

    cfg = _run_config(run)
    project = cfg["project"] + ("/smoke" if cfg["smoke"]["enabled"] else "/evaluation")
    task = Task.init(project_name=project, task_name=f"{run.name} · test", tags=[*run.get_tags(), "final-eval"],
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     auto_connect_frameworks={"pytorch": False, "matplotlib": False})
    task.set_parent(run.id)
    task.connect({"training_run": run.id, "dataset_id": cfg["dataset"]["id"], **eval_settings}, name="eval")
    predictor = make_predictor(cfg["model"]["name"], _weights(run), cfg["imgsz"])
    framework = evaluate_split(predictor, fetch_dataset(cfg["dataset"]["id"]) / "test", cfg["model"]["name"],
                               latency_images=eval_settings["latency_images"], warmup=eval_settings["latency_warmup"])
    metrics = log_eval(task, framework, "test", threshold=_valid_threshold(run))
    task.close()
    return metrics


def revalidate_into(run, cfg: dict, eval_settings: dict, predictor, valid_dir: Path) -> None:
    """Log a fresh validation evaluation into the existing training run `run`.

    Deliberately not Task.init(continue_last_task=...): that would overwrite the run's recorded commit and diff
    and offset reported iterations.
    """
    cfg = {**cfg}
    cfg.setdefault("eval", eval_settings)  # runs trained before evaluation existed
    run.mark_started(force=True)
    run_validation_eval(run, cfg, predictor, valid_dir)
    run.flush(wait_for_uploads=True)
    run.mark_completed()


def revalidate(run_id: str, eval_settings: dict) -> None:
    """(Re-)run the validation evaluation into an existing training run."""
    from clearml import Task

    from ppe.data import fetch_dataset

    run = Task.get_task(task_id=run_id)
    cfg = _run_config(run)
    predictor = make_predictor(cfg["model"]["name"], _weights(run), cfg["imgsz"])
    revalidate_into(run, cfg, eval_settings, predictor, fetch_dataset(cfg["dataset"]["id"]) / "valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained runs with the shared evaluator")
    parser.add_argument("--task", help="training run ID")
    parser.add_argument("--final", action="store_true", help="evaluate on the TEST split (once per run)")
    parser.add_argument("--tag", help="with --final: evaluate every training run carrying this tag, e.g. report")
    parser.add_argument("--allow-cpu", action="store_true", help="allow --final without a GPU (latency not comparable)")
    parser.add_argument("--common", type=Path, default=Path("configs/common.yaml"))
    args = parser.parse_args()

    from clearml import Task

    from ppe.tracking import check_credentials

    check_credentials(os.environ, offline=Task.is_offline())
    common = yaml.safe_load(args.common.read_text())
    eval_settings = common["eval"]

    if not args.final:
        if not args.task:
            parser.error("--task is required without --final")
        revalidate(args.task, eval_settings)
        return

    if not (args.task or args.tag):
        parser.error("--final needs --task or --tag")
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no CUDA device: final latency must come from the reference GPU (use --allow-cpu to override)")

    runs = [Task.get_task(task_id=args.task)] if args.task else Task.get_tasks(
        project_name=common["project"], tags=[args.tag], task_filter={"type": ["training"]})
    runs = [r for r in runs if "final-eval" not in r.get_tags()]  # never treat an evaluation as a training run
    todo, skipped = select_final_runs([_run_info(r) for r in runs], {r.id: _final_eval_statuses(r.id) for r in runs},
                                      explicit=bool(args.task))
    for run_id, reason in skipped:
        print(f"skip {run_id}: {reason}")
    if not todo:
        raise SystemExit("no eligible runs to evaluate")
    for run in (r for r in runs if r.id in todo):
        metrics = final_evaluate(run, eval_settings)
        print(f"{run.name}: test mAP50(all)={metrics['mAP50/all']:.4f} mAP50-95(all)={metrics['mAP50-95/all']:.4f}")


if __name__ == "__main__":
    main()
