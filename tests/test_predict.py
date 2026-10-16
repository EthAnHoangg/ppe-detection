import numpy as np
import pytest
import torch

from ppe.predict import FrcnnPredictor, Prediction, YoloPredictor, frcnn_to_prediction, make_predictor, yolo_to_prediction


def test_yolo_conversion_gives_numpy_types_and_int_labels():
    p = yolo_to_prediction(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([0.9]), torch.tensor([5.0]))
    assert p.boxes.dtype == np.float64 and p.boxes.shape == (1, 4)
    assert p.scores.tolist() == pytest.approx([0.9])
    assert p.labels.dtype == np.int64 and p.labels.tolist() == [5]


def test_frcnn_conversion_shifts_labels_back_to_zero_based():
    out = {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
           "scores": torch.tensor([0.8, 0.3]), "labels": torch.tensor([1, 10])}
    p = frcnn_to_prediction(out)
    assert p.labels.tolist() == [0, 9]
    assert p.boxes.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_empty_outputs_give_correctly_shaped_empty_arrays():
    p = frcnn_to_prediction({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)})
    assert p.boxes.shape == (0, 4) and p.scores.shape == (0,) and p.labels.shape == (0,)
    q = yolo_to_prediction(torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0))
    assert q.boxes.shape == (0, 4) and q.labels.dtype == np.int64


class _FakeBoxes:
    xyxy = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    conf = torch.tensor([0.5])
    cls = torch.tensor([2.0])


class _FakeYolo:
    def __init__(self):
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return [type("R", (), {"boxes": _FakeBoxes()})()]


def test_yolo_predictor_passes_bgr_and_settings():
    fake = _FakeYolo()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    p = YoloPredictor(fake, imgsz=640).predict(rgb)
    source, kwargs = fake.calls[0]
    assert source[0, 0].tolist() == [0, 0, 255]  # red is last channel in BGR
    assert source.flags["C_CONTIGUOUS"]
    assert kwargs == {"imgsz": 640, "conf": 0.001, "max_det": 100, "verbose": False}
    assert p.labels.tolist() == [2]


def test_frcnn_predictor_loads_weights_and_sets_limits(tmp_path):
    from ppe.frcnn.train import build_model

    weights = tmp_path / "best.pt"
    torch.save(build_model(imgsz=640, pretrained=False).state_dict(), weights)
    predictor = FrcnnPredictor.load(weights, imgsz=640)
    assert predictor.model.roi_heads.score_thresh == 0.001
    assert predictor.model.roi_heads.detections_per_img == 100
    p = predictor.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert isinstance(p, Prediction) and p.boxes.shape[1] == 4


def test_make_predictor_rejects_unknown_family(tmp_path):
    with pytest.raises(ValueError, match="ssd300"):
        make_predictor("ssd300", tmp_path / "w.pt", 640)
