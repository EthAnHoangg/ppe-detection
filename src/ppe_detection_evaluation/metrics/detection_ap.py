from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..classes import DEFAULT_CLASS_NAMES, PPE_CLASSES
from ..records import ImageRecord
from .matching import coco_average_precision, match_class, nanmean

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)

SIZE_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}


def compute_detection_ap(
    records: Sequence[ImageRecord],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    class_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    class_names = tuple(class_names)
    groups = class_groups or {"all": class_names, "ppe": PPE_CLASSES}
    group_ids = {g: [class_names.index(n) for n in names] for g, names in groups.items()}

    ranges: dict[str, tuple[float, float] | None] = {"all": None, **SIZE_RANGES}
    ap = {size: np.full((len(class_names), len(IOU_THRESHOLDS)), np.nan) for size in ranges}
    num_gt = {size: np.zeros(len(class_names), dtype=np.int64) for size in ranges}
    for size, area_range in ranges.items():
        for c in range(len(class_names)):
            for t, iou in enumerate(IOU_THRESHOLDS):
                matches = match_class(records, c, float(iou), area_range)
                ap[size][c, t] = coco_average_precision(matches)
            num_gt[size][c] = matches.num_gt

    def summarize(size: str) -> dict[str, dict[str, float]]:
        per_class_ap = np.array([nanmean(row) for row in ap[size]])
        return {
            g: {"mAP50": nanmean(ap[size][ids, 0]), "mAP": nanmean(per_class_ap[ids])}
            for g, ids in group_ids.items()
        }

    return {
        "iou_thresholds": IOU_THRESHOLDS.tolist(),
        "per_class": {
            name: {
                "num_gt": int(num_gt["all"][c]),
                "AP50": float(ap["all"][c, 0]),
                "AP75": float(ap["all"][c, 5]),
                "AP": nanmean(ap["all"][c]),
            }
            for c, name in enumerate(class_names)
        },
        "groups": summarize("all"),
        "by_size": {
            size: {
                "num_gt": {name: int(num_gt[size][c]) for c, name in enumerate(class_names)},
                "groups": summarize(size),
            }
            for size in SIZE_RANGES
        },
    }
