from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..records import ImageRecord

RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass
class ClassMatches:
    scores: np.ndarray
    tp: np.ndarray
    num_gt: int


@dataclass
class PRCurve:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    num_gt: int


def box_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_class(
    records: Sequence[ImageRecord],
    class_id: int,
    iou_threshold: float,
    area_range: tuple[float, float] | None = None,
) -> ClassMatches:
    scores, tps = [], []
    num_gt = 0
    for r in records:
        gt = r.gt_boxes[r.gt_labels == class_id]
        sel = r.pred_labels == class_id
        order = np.argsort(-r.pred_scores[sel], kind="stable")
        pb, ps = r.pred_boxes[sel][order], r.pred_scores[sel][order]

        if area_range is None:
            gt_ignore = np.zeros(len(gt), dtype=bool)
        else:
            gt_area = box_area(gt)
            gt_ignore = (gt_area < area_range[0]) | (gt_area > area_range[1])
        num_gt += int((~gt_ignore).sum())

        ious = box_iou(pb, gt)
        matched = np.zeros(len(gt), dtype=bool)
        tp = np.zeros(len(pb), dtype=bool)
        ignore = np.zeros(len(pb), dtype=bool)
        for i in range(len(pb)):
            candidates = ~matched & (ious[i] >= iou_threshold)
            if not candidates.any():
                continue
            preferred = candidates & ~gt_ignore
            pool = preferred if preferred.any() else candidates
            j = np.flatnonzero(pool)[ious[i, pool].argmax()]
            matched[j] = True
            tp[i] = True
            ignore[i] = gt_ignore[j]
        if area_range is not None:
            pred_area = box_area(pb)
            ignore |= ~tp & ((pred_area < area_range[0]) | (pred_area > area_range[1]))

        scores.append(ps[~ignore])
        tps.append(tp[~ignore])

    if not scores:
        return ClassMatches(np.zeros(0), np.zeros(0, dtype=bool), num_gt)
    return ClassMatches(np.concatenate(scores), np.concatenate(tps), num_gt)


def coco_average_precision(matches: ClassMatches) -> float:
    if matches.num_gt == 0:
        return float("nan")
    if not len(matches.scores):
        return 0.0
    order = np.argsort(-matches.scores, kind="mergesort")
    tp = np.cumsum(matches.tp[order])
    fp = np.cumsum(~matches.tp[order])
    recall = tp / matches.num_gt
    precision = np.maximum.accumulate((tp / (tp + fp))[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(q.mean())


def pr_curve(matches: ClassMatches) -> PRCurve:
    order = np.argsort(-matches.scores, kind="mergesort")
    scores = matches.scores[order]
    tp = np.cumsum(matches.tp[order])
    fp = np.cumsum(~matches.tp[order])
    last = np.r_[scores[1:] != scores[:-1], True] if len(scores) else np.zeros(0, dtype=bool)
    scores, tp, fp = scores[last], tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / matches.num_gt if matches.num_gt else np.full(len(tp), np.nan)
    return PRCurve(scores, precision, recall, tp, fp, matches.num_gt)


def step_average_precision(curve: PRCurve) -> float:
    if curve.num_gt == 0:
        return float("nan")
    if not len(curve.recall):
        return 0.0
    return float(np.sum(np.diff(curve.recall, prepend=0.0) * curve.precision))


def precision_at_recall(curve: PRCurve, target: float) -> float:
    if curve.num_gt == 0:
        return float("nan")
    reached = curve.recall >= target
    return float(curve.precision[reached].max()) if reached.any() else 0.0


def max_f1(curve: PRCurve) -> dict[str, float | None]:
    if curve.num_gt == 0 or not len(curve.recall):
        nan_or_zero = float("nan") if curve.num_gt == 0 else 0.0
        return {"f1": nan_or_zero, "threshold": None, "precision": nan_or_zero, "recall": nan_or_zero}
    p, r = curve.precision, curve.recall
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(f1.argmax())
    return {"f1": float(f1[i]), "threshold": float(curve.thresholds[i]), "precision": float(p[i]), "recall": float(r[i])}


def nanmean(values: Sequence[float] | np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")
