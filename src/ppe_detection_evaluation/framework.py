import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .classes import DEFAULT_CLASS_NAMES
from .metrics import (
    compute_detection_ap,
    compute_image_level_metrics,
    compute_latency,
    compute_ppe_confusion,
    compute_violation_metrics,
)
from .model_configs import GT_CONFIG, ModelConfig
from .records import ImageRecord


class EvaluationFramework:
    def __init__(
        self,
        config: ModelConfig,
        *,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        image_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.001,
        max_detections: int = 100,
    ) -> None:
        self.config = config
        self.class_names = tuple(class_names)
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.records: list[ImageRecord] = []
        self.results: dict[str, Any] = {}

    def add(
        self,
        image_id: str,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: Sequence[int | str] | np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: Sequence[int | str] | np.ndarray,
        latency_ms: float | None = None,
    ) -> ImageRecord:
        boxes = self._to_xyxy(pred_boxes, self.config)
        scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        labels = self._to_class_ids(pred_labels, self.config.label_offset)

        keep = np.flatnonzero(scores >= self.score_threshold)
        keep = keep[np.argsort(-scores[keep], kind="stable")][: self.max_detections]

        record = ImageRecord(
            image_id=image_id,
            gt_boxes=self._to_xyxy(gt_boxes, GT_CONFIG),
            gt_labels=self._to_class_ids(gt_labels, 0),
            pred_boxes=boxes[keep],
            pred_scores=scores[keep],
            pred_labels=labels[keep],
            latency_ms=latency_ms,
        )
        self.records.append(record)
        return record

    def evaluate(self) -> dict[str, Any]:
        if not self.records:
            raise ValueError("No images recorded; call add() first")
        self.results = {
            "detection": compute_detection_ap(self.records, self.class_names),
            "violation": compute_violation_metrics(self.records, self.class_names),
            "image_level": compute_image_level_metrics(self.records, self.class_names),
            "confusion": compute_ppe_confusion(self.records, self.class_names),
            "latency": compute_latency(self.records),
        }
        return self.results

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "settings": {
                "class_names": list(self.class_names),
                "image_size": list(self.image_size),
                "score_threshold": self.score_threshold,
                "max_detections": self.max_detections,
            },
            "results": self.results,
            "records": [r.to_dict() for r in self.records],
        }
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationFramework":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = data["config"]
        if config.get("source_size") is not None:
            config["source_size"] = tuple(config["source_size"])
        settings = data["settings"]
        fw = cls(
            ModelConfig(**config),
            class_names=settings["class_names"],
            image_size=tuple(settings["image_size"]),
            score_threshold=settings["score_threshold"],
            max_detections=settings["max_detections"],
        )
        fw.records = [ImageRecord.from_dict(r) for r in data["records"]]
        fw.results = data["results"]
        return fw

    def _to_xyxy(self, boxes: np.ndarray, config: ModelConfig) -> np.ndarray:
        box_format = config.box_format
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if box_format == "xywh":
            b = np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
        elif box_format == "cxcywh":
            half_w, half_h = b[:, 2] / 2, b[:, 3] / 2
            b = np.column_stack([b[:, 0] - half_w, b[:, 1] - half_h, b[:, 0] + half_w, b[:, 1] + half_h])

        width, height = self.image_size
        if config.normalized:
            b = b * [width, height, width, height]
        elif config.source_size is not None:
            sx, sy = width / config.source_size[0], height / config.source_size[1]
            b = b * [sx, sy, sx, sy]
        return np.clip(b, 0, [width, height, width, height])

    def _to_class_ids(self, labels: Sequence[int | str] | np.ndarray, offset: int) -> np.ndarray:
        ids = []
        for label in np.asarray(labels).reshape(-1).tolist():
            if isinstance(label, str):
                if label not in self.class_names:
                    raise ValueError(f"Unknown class name {label!r}")
                ids.append(self.class_names.index(label))
            else:
                ids.append(int(label) - offset)
        out = np.array(ids, dtype=np.int64)
        if len(out) and (out.min() < 0 or out.max() >= len(self.class_names)):
            raise ValueError(f"Class id out of range for {len(self.class_names)} classes")
        return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
