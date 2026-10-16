from collections.abc import Sequence
from typing import Any

import numpy as np

from ..classes import DEFAULT_CLASS_NAMES, VIOLATION_CLASSES
from ..records import ImageRecord
from .matching import PRCurve, match_class, max_f1, nanmean, pr_curve, precision_at_recall

LAMR_FPPI_POINTS = np.logspace(-2.0, 0.0, 9)


def compute_violation_metrics(
    records: Sequence[ImageRecord],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    violation_classes: Sequence[str] = VIOLATION_CLASSES,
    *,
    iou_threshold: float = 0.5,
    recall_target: float = 0.85,
    fppi_target: float = 0.1,
) -> dict[str, Any]:
    """Per violation class, at IoU ``iou_threshold``:

        precision_at_recall -- best precision while finding >= ``recall_target`` of violations
        recall_at_fppi      -- best recall with <= ``fppi_target`` false positives per image
        lamr                -- log-average miss rate over FPPI 1e-2..1e0 (lower is better)
        max_f1              -- best F1 and its threshold, precision and recall
        max_recall          -- recall at the lowest kept score

    ``mean`` averages each scalar over the violation classes.
    """
    class_names = tuple(class_names)
    num_images = len(records)
    per_class = {}
    for name in violation_classes:
        curve = pr_curve(match_class(records, class_names.index(name), iou_threshold))
        fppi = curve.fp / max(num_images, 1)
        per_class[name] = {
            "num_gt": curve.num_gt,
            "precision_at_recall": precision_at_recall(curve, recall_target),
            "recall_at_fppi": _recall_at_fppi(curve, fppi, fppi_target),
            "lamr": _log_average_miss_rate(curve, fppi),
            "max_f1": max_f1(curve),
            "max_recall": float(curve.recall[-1]) if len(curve.recall) else 0.0,
        }

    scalar_keys = ("precision_at_recall", "recall_at_fppi", "lamr", "max_recall")
    mean = {k: nanmean([v[k] for v in per_class.values()]) for k in scalar_keys}
    mean["max_f1"] = nanmean([v["max_f1"]["f1"] for v in per_class.values()])
    return {
        "settings": {
            "iou_threshold": iou_threshold,
            "recall_target": recall_target,
            "fppi_target": fppi_target,
            "num_images": num_images,
        },
        "per_class": per_class,
        "mean": mean,
    }


def _recall_at_fppi(curve: PRCurve, fppi: np.ndarray, target: float) -> float:
    if curve.num_gt == 0:
        return float("nan")
    within = fppi <= target
    return float(curve.recall[within].max()) if within.any() else 0.0


def _log_average_miss_rate(curve: PRCurve, fppi: np.ndarray) -> float:
    if curve.num_gt == 0:
        return float("nan")
    miss_rate = np.r_[1.0, 1.0 - curve.recall]
    fppi = np.r_[-1.0, fppi]
    sampled = [miss_rate[np.flatnonzero(fppi <= ref)[-1]] for ref in LAMR_FPPI_POINTS]
    return float(np.exp(np.mean(np.log(np.maximum(sampled, 1e-10)))))
