"""Model-specific predictors that all return the same `Prediction` format.

Input images are RGB uint8 arrays (H, W, 3). Output boxes are pixel x1, y1, x2, y2 and labels are 0-based class IDs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

CONF = 0.001  # keep low-confidence boxes for AP (report Section 4.2)
MAX_DET = 100  # matches the evaluator's per-image cap


@dataclass
class Prediction:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def _prediction(boxes, scores, labels) -> Prediction:
    return Prediction(
        boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float64).reshape(-1),
        labels=np.asarray(labels).reshape(-1).astype(np.int64),
    )


def yolo_to_prediction(xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor) -> Prediction:
    return _prediction(xyxy.cpu().numpy(), conf.cpu().numpy(), cls.cpu().numpy())


def frcnn_to_prediction(output: dict) -> Prediction:
    """torchvision labels are 1..10 (0 is background); shift back to 0..9."""
    return _prediction(output["boxes"].cpu().numpy(), output["scores"].cpu().numpy(), output["labels"].cpu().numpy() - 1)


class YoloPredictor:
    def __init__(self, model, imgsz: int):
        self.model = model
        self.imgsz = imgsz

    @classmethod
    def load(cls, weights: Path, imgsz: int) -> "YoloPredictor":
        from ultralytics import YOLO

        return cls(YOLO(str(weights)), imgsz)

    def predict(self, image: np.ndarray) -> Prediction:
        bgr = np.ascontiguousarray(image[..., ::-1])  # Ultralytics treats numpy input as BGR
        result = self.model.predict(bgr, imgsz=self.imgsz, conf=CONF, max_det=MAX_DET, verbose=False)[0]
        return yolo_to_prediction(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)


class FrcnnPredictor:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @classmethod
    def load(cls, weights: Path, imgsz: int) -> "FrcnnPredictor":
        from ppe.frcnn.train import build_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(imgsz, pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.roi_heads.score_thresh = CONF
        model.roi_heads.detections_per_img = MAX_DET
        return cls(model.to(device).eval(), device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Prediction:
        return frcnn_to_prediction(self.model([to_tensor(image).to(self.device)])[0])


def make_predictor(model_name: str, weights: Path, imgsz: int):
    """Pick the predictor from the run's model name (e.g. 'yolo26s', 'fasterrcnn-r50')."""
    if model_name.startswith("yolo"):
        return YoloPredictor.load(weights, imgsz)
    if model_name.startswith("fasterrcnn"):
        return FrcnnPredictor.load(weights, imgsz)
    raise ValueError(f"unknown model family for {model_name!r}; expected a yolo* or fasterrcnn* model name")
