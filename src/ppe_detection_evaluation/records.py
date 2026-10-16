from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ImageRecord:
    image_id: str
    gt_boxes: np.ndarray
    gt_labels: np.ndarray
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_labels: np.ndarray
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "gt_boxes": self.gt_boxes.tolist(),
            "gt_labels": self.gt_labels.tolist(),
            "pred_boxes": self.pred_boxes.tolist(),
            "pred_scores": self.pred_scores.tolist(),
            "pred_labels": self.pred_labels.tolist(),
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=d["image_id"],
            gt_boxes=np.asarray(d["gt_boxes"], dtype=np.float64).reshape(-1, 4),
            gt_labels=np.asarray(d["gt_labels"], dtype=np.int64),
            pred_boxes=np.asarray(d["pred_boxes"], dtype=np.float64).reshape(-1, 4),
            pred_scores=np.asarray(d["pred_scores"], dtype=np.float64),
            pred_labels=np.asarray(d["pred_labels"], dtype=np.int64),
            latency_ms=d.get("latency_ms"),
        )
