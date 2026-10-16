from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..classes import DEFAULT_CLASS_NAMES, PPE_PAIRS
from ..records import ImageRecord
from .matching import box_iou


def compute_ppe_confusion(
    records: Sequence[ImageRecord],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    pairs: Mapping[str, str] = PPE_PAIRS,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    class_names = tuple(class_names)
    out = {}
    for present, absent in pairs.items():
        for true_name, other_name in ((present, absent), (absent, present)):
            counts = _count(
                records, class_names.index(true_name), class_names.index(other_name), iou_threshold
            )
            localised = counts["correct"] + counts["swapped"]
            n = localised + counts["missed"]
            out[f"{true_name} -> {other_name}"] = {
                "num_gt": n,
                **counts,
                "swap_rate": counts["swapped"] / localised if localised else float("nan"),
                "miss_rate": counts["missed"] / n if n else float("nan"),
            }
    return out


def _count(records: Sequence[ImageRecord], true_id: int, other_id: int, iou_threshold: float) -> dict[str, int]:
    correct = swapped = missed = 0
    for r in records:
        gt = r.gt_boxes[r.gt_labels == true_id]
        if not len(gt):
            continue
        sel = np.isin(r.pred_labels, [true_id, other_id])
        boxes, scores, labels = r.pred_boxes[sel], r.pred_scores[sel], r.pred_labels[sel]
        ious = box_iou(gt, boxes)
        for g in range(len(gt)):
            overlapping = np.flatnonzero(ious[g] >= iou_threshold)
            if not len(overlapping):
                missed += 1
            elif labels[overlapping[scores[overlapping].argmax()]] == true_id:
                correct += 1
            else:
                swapped += 1
    return {"correct": correct, "swapped": swapped, "missed": missed}
