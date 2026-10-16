import math
import time

import numpy as np
import pytest
from PIL import Image

from ppe.evaluate import evaluate_split, headline_metrics, read_split, time_predictions
from ppe.predict import Prediction


def _write(split, name, size, label_text):
    (split / "images").mkdir(parents=True, exist_ok=True)
    (split / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(split / "images" / f"{name}.jpg")
    if label_text is not None:
        (split / "labels" / f"{name}.txt").write_text(label_text)


@pytest.fixture
def split(tmp_path):
    s = tmp_path / "valid"
    _write(s, "a", (640, 640), "0 0.5 0.5 0.2 0.2\n5 0.25 0.25 0.1 0.3\n")
    _write(s, "b", (640, 640), "2 0.7 0.7 0.1 0.1\n")
    _write(s, "c", (640, 640), "")      # background image
    _write(s, "d", (640, 640), None)    # label file missing entirely
    return s


def test_read_split_returns_rgb_images_and_yolo_ground_truth(split):
    samples = read_split(split)
    assert [s.image_id for s in samples] == ["a", "b", "c", "d"]
    assert samples[0].image.shape == (640, 640, 3) and samples[0].image.dtype == np.uint8
    assert samples[0].gt_labels.tolist() == [0, 5]
    assert samples[0].gt_boxes.tolist()[1] == pytest.approx([0.25, 0.25, 0.1, 0.3])
    assert samples[2].gt_boxes.shape == (0, 4) and samples[3].gt_labels.shape == (0,)


def test_read_split_limit_keeps_first_n(split):
    assert len(read_split(split, limit=2)) == 2


def test_read_split_rejects_non_640_images(tmp_path):
    s = tmp_path / "valid"
    _write(s, "odd", (320, 240), "")
    with pytest.raises(ValueError, match="odd.jpg"):
        read_split(s)


def test_time_predictions_excludes_warmup_and_cycles_images():
    seen = []

    def predict(image):
        seen.append(image)
        time.sleep(0.002)

    timings = time_predictions(predict, ["x", "y", "z"], warmup=2, n=5, sync=lambda: None)
    assert len(timings) == 5
    assert len(seen) == 7
    assert seen == ["x", "y", "z", "x", "y", "z", "x"]
    assert all(t >= 2.0 for t in timings)


class _GroundTruthPredictor:
    """Returns each image's ground truth as perfect predictions (images are matched by identity)."""

    def __init__(self, samples):
        self._samples = samples

    def predict(self, image):
        s = next(s for s in self._samples if s.image is image)
        cx, cy, w, h = (s.gt_boxes * 640).reshape(-1, 4).T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        return Prediction(boxes, np.ones(len(boxes)), s.gt_labels)


def test_evaluate_split_perfect_predictor_scores_one(split, monkeypatch):
    import ppe.evaluate as ev

    samples = read_split(split)
    monkeypatch.setattr(ev, "read_split", lambda split_dir, limit=None: samples)
    fw = evaluate_split(_GroundTruthPredictor(samples), split, "fake", latency_images=3, warmup=1, sync=lambda: None)
    detection = fw.results["detection"]
    assert detection["groups"]["all"]["mAP50"] == pytest.approx(1.0)
    assert fw.results["latency"]["num_images"] == 3
    # warmup=1 uses image 0; the 3 timed calls land on images 1-3
    assert [r.latency_ms is not None for r in fw.records] == [False, True, True, True]


def test_headline_metrics_flattens_results_and_maps_none_to_nan():
    results = {
        "detection": {"groups": {"all": {"mAP50": 0.5, "mAP": 0.3}, "ppe": {"mAP50": None, "mAP": 0.2}},
                      "per_class": {"Hardhat": {"AP50": 0.9}, "Mask": {"AP50": None}}},
        "violation": {"mean": {"lamr": 0.4, "max_f1": 0.6}},
        "image_level": {"any": {"ap": 0.7}},
        "latency": {"num_images": 3, "mean_ms": 10.0, "p95_ms": 12.0, "fps": 100.0},
    }
    m = headline_metrics(results)
    assert m["mAP50/all"] == 0.5 and m["mAP50-95/ppe"] == 0.2
    assert math.isnan(m["mAP50/ppe"]) and math.isnan(m["AP50/Mask"])
    assert m["violation/lamr"] == 0.4 and m["image_any/AP"] == 0.7
    assert m["latency/mean_ms"] == 10.0 and m["latency/fps"] == 100.0


def test_headline_metrics_without_latency_omits_latency_keys():
    results = {"detection": {"groups": {"all": {"mAP50": 0.5, "mAP": 0.3}, "ppe": {"mAP50": 0.5, "mAP": 0.3}},
                             "per_class": {}},
               "violation": {"mean": {}}, "image_level": {"any": {"ap": 0.1}}, "latency": {"num_images": 0}}
    assert not any(k.startswith("latency/") for k in headline_metrics(results))
