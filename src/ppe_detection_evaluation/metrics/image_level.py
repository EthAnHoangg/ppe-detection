from collections.abc import Sequence
from typing import Any

import numpy as np

from ..classes import DEFAULT_CLASS_NAMES, VIOLATION_CLASSES
from ..records import ImageRecord
from .matching import ClassMatches, max_f1, pr_curve, precision_at_recall, step_average_precision


def compute_image_level_metrics(
    records: Sequence[ImageRecord],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    violation_classes: Sequence[str] = VIOLATION_CLASSES,
    *,
    recall_target: float = 0.85,
) -> dict[str, Any]:
    class_names = tuple(class_names)
    targets = {"any": [class_names.index(n) for n in violation_classes]}
    targets |= {n: [class_names.index(n)] for n in violation_classes}
    return {name: _image_level(records, ids, recall_target) for name, ids in targets.items()}


def _image_level(records: Sequence[ImageRecord], class_ids: list[int], recall_target: float) -> dict[str, Any]:
    positive = np.array([np.isin(r.gt_labels, class_ids).any() for r in records], dtype=bool)
    scores = np.array(
        [r.pred_scores[np.isin(r.pred_labels, class_ids)].max(initial=0.0) for r in records],
        dtype=np.float64,
    )
    curve = pr_curve(ClassMatches(scores, positive, int(positive.sum())))
    return {
        "num_images": len(records),
        "num_positive": int(positive.sum()),
        "ap": step_average_precision(curve),
        "auroc": _auroc(scores, positive),
        "precision_at_recall": precision_at_recall(curve, recall_target),
        "max_f1": max_f1(curve),
    }


def _auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
