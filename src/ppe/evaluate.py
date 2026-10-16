"""Score a trained model on a split with the shared evaluator, and log the results to ClearML.

    uv run python -m ppe.evaluate --task <training-run-id>        # (re-)run validation evaluation into that run
    uv run python -m ppe.evaluate --final --tag report            # final test evaluation, once per run
"""

import json
import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
from PIL import Image

from ppe_detection_evaluation import EvaluationFramework, ModelConfig
from ppe_detection_evaluation.metrics import compute_latency

IMAGE_SIZE = (640, 640)


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
    smoke = cfg["smoke"]["enabled"]
    framework = evaluate_split(
        predictor, valid_dir, cfg["model"]["name"],
        latency_images=cfg["eval"]["smoke_latency_images"] if smoke else cfg["eval"]["latency_images"],
        warmup=1 if smoke else cfg["eval"]["latency_warmup"],
        limit=cfg["smoke"]["images"] if smoke else None,
    )
    return log_eval(task, framework, "valid")
