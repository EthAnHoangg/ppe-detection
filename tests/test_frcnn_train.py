import random

import numpy as np
import pytest
import torch

from ppe.frcnn.train import build_model, dataloader_workers, learning_rate, summarise_map
from ppe.seed import seed_everything


def test_seed_everything_makes_python_numpy_and_torch_repeatable():
    seed_everything(3)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(3)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first


def test_summarise_map_uses_yolo_metric_names_and_class_names():
    result = {
        "map": torch.tensor(0.4),
        "map_50": torch.tensor(0.6),
        "map_per_class": torch.tensor([0.1, 0.9]),
        "classes": torch.tensor([1, 6], dtype=torch.int32),
    }
    summary = summarise_map(result)
    assert summary["metrics/mAP50-95(B)"] == torch.tensor(0.4).item()
    assert summary["metrics/mAP50(B)"] == torch.tensor(0.6).item()
    assert summary["per_class"] == {"Hardhat": torch.tensor(0.1).item(), "Person": torch.tensor(0.9).item()}


def test_summarise_map_handles_a_single_class_scalar():
    result = {
        "map": torch.tensor(0.5),
        "map_50": torch.tensor(0.5),
        "map_per_class": torch.tensor(0.5),
        "classes": torch.tensor(10, dtype=torch.int32),
    }
    assert summarise_map(result)["per_class"] == {"vehicle": 0.5}


def test_build_model_predicts_eleven_classes_at_the_shared_image_size():
    model = build_model(imgsz=640, pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 11
    assert model.transform.min_size == (640,)
    assert model.transform.max_size == 640


def test_learning_rate_warms_up_linearly_then_steps_down_at_milestones():
    kwargs = dict(base=0.01, warmup_iters=100, milestones=[3, 5], gamma=0.1)
    assert learning_rate(step=0, epoch=0, **kwargs) == pytest.approx(0.01 * 0.001)
    assert learning_rate(step=50, epoch=0, **kwargs) == pytest.approx(0.01 * (0.001 + 0.999 * 0.5))
    assert learning_rate(step=100, epoch=1, **kwargs) == pytest.approx(0.01)
    assert learning_rate(step=400, epoch=3, **kwargs) == pytest.approx(0.001)
    assert learning_rate(step=900, epoch=6, **kwargs) == pytest.approx(0.0001)


def test_dataloader_workers_are_disabled_on_macos_only():
    # On macOS, torch's shm manager processes inherit multiprocessing's resource-tracker pipe and hang interpreter exit.
    assert dataloader_workers(4, platform="darwin") == 0
    assert dataloader_workers(4, platform="linux") == 4
