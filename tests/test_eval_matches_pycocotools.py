"""The moved evaluator must keep matching pycocotools exactly (with <=100 detections per image)."""

import contextlib
import io

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ppe_detection_evaluation import ImageRecord
from ppe_detection_evaluation.metrics.detection_ap import compute_detection_ap


def _synthetic(seed, n_img=40):
    rng = np.random.default_rng(seed)
    records, images, anns, dets = [], [], [], []
    aid = 1
    for i in range(n_img):
        k = rng.integers(0, 12)
        xy = rng.uniform(0, 560, (k, 2))
        gt = np.c_[xy, np.minimum(xy + rng.uniform(5, 200, (k, 2)), 640)]
        gl = rng.integers(0, 10, k)
        keep = rng.random(k) < 0.8
        pb = gt[keep] + rng.normal(0, 8, (keep.sum(), 4))
        pl = gl[keep].copy()
        swap = rng.random(len(pl)) < 0.1
        pl[swap] = rng.integers(0, 10, swap.sum())
        nf = rng.integers(0, 15)
        fxy = rng.uniform(0, 560, (nf, 2))
        pb = np.r_[pb, np.c_[fxy, fxy + rng.uniform(5, 150, (nf, 2))]]
        pl = np.r_[pl, rng.integers(0, 10, nf)]
        pb = np.clip(pb, 0, 640)
        pb[:, 2:] = np.maximum(pb[:, 2:], pb[:, :2] + 1)
        ps = rng.random(len(pb))
        order = np.argsort(-ps)[:100]
        pb, pl, ps = pb[order], pl[order], ps[order]
        records.append(ImageRecord(str(i), gt, gl, pb, ps, pl))
        images.append({"id": i, "width": 640, "height": 640})
        for b, label in zip(gt, gl):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": aid, "image_id": i, "category_id": int(label), "bbox": [b[0], b[1], w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
        for b, label, s in zip(pb, pl, ps):
            dets.append({"image_id": i, "category_id": int(label),
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(s)})
    return records, images, anns, dets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detection_ap_matches_pycocotools(seed):
    records, images, anns, dets = _synthetic(seed)
    ours = compute_detection_ap(records)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(10)]}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert ours["groups"]["all"]["mAP"] == pytest.approx(ev.stats[0], abs=1e-9)
    assert ours["groups"]["all"]["mAP50"] == pytest.approx(ev.stats[1], abs=1e-9)
    for size, index in {"small": 3, "medium": 4, "large": 5}.items():
        assert ours["by_size"][size]["groups"]["all"]["mAP"] == pytest.approx(ev.stats[index], abs=1e-9)
    precision = ev.eval["precision"][:, :, :, 0, 2]
    for c, value in enumerate(ours["per_class"].values()):
        p = precision[:, :, c]
        assert value["AP"] == pytest.approx(np.mean(p[p > -1]), abs=1e-9)
