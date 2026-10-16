import json
from pathlib import Path

import numpy as np

from ppe.evaluate import log_eval, run_validation_eval
from ppe_detection_evaluation import EvaluationFramework, ModelConfig


class _Logger:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, title, series, value, iteration):
        self.scalars.append((title, series, value, iteration))


class _Task:
    def __init__(self):
        self.logger = _Logger()
        self.artifacts = {}

    def get_logger(self):
        return self.logger

    def upload_artifact(self, name, artifact_object, wait_on_upload=False):
        self.artifacts[name] = json.loads(Path(artifact_object).read_text())
        return True


def _framework():
    fw = EvaluationFramework(ModelConfig("fake"))
    # Only a Hardhat box, perfectly predicted: other classes have no ground truth, so their AP is NaN.
    fw.add("a", np.array([[288.0, 288.0, 352.0, 352.0]]), np.array([0.9]), np.array([0]),
           np.array([[0.5, 0.5, 0.1, 0.1]]), np.array([0]), latency_ms=5.0)
    fw.evaluate()
    return fw


def test_log_eval_skips_nan_and_uploads_artifacts():
    task = _Task()
    metrics = log_eval(task, _framework(), "valid")
    logged = {series: value for title, series, value, _ in task.logger.scalars}
    assert all(title == "valid eval" for title, *_ in task.logger.scalars)
    assert logged["AP50/Hardhat"] == 1.0
    assert "AP50/Mask" not in logged and np.isnan(metrics["AP50/Mask"])
    assert set(task.artifacts) == {"valid_eval", "valid_predictions"}
    assert task.artifacts["valid_eval"]["model"] == "fake"
    assert task.artifacts["valid_eval"]["threshold"] is None
    assert "results" in task.artifacts["valid_eval"] and "records" not in task.artifacts["valid_eval"]
    assert len(task.artifacts["valid_predictions"]["records"]) == 1


def test_run_validation_eval_uses_smoke_limits(monkeypatch, tmp_path):
    import ppe.evaluate as ev

    calls = {}

    def fake_evaluate_split(predictor, split_dir, model_name, **kwargs):
        calls.update(kwargs, split_dir=split_dir, model_name=model_name)
        return _framework()

    monkeypatch.setattr(ev, "evaluate_split", fake_evaluate_split)
    cfg = {"model": {"name": "yolo26s"}, "smoke": {"enabled": True, "images": 16},
           "eval": {"latency_warmup": 20, "latency_images": 200, "smoke_latency_images": 5}}
    run_validation_eval(_Task(), cfg, predictor=None, valid_dir=tmp_path)
    assert (calls["limit"], calls["latency_images"], calls["warmup"]) == (16, 5, 1)
    cfg["smoke"]["enabled"] = False
    run_validation_eval(_Task(), cfg, predictor=None, valid_dir=tmp_path)
    assert (calls["limit"], calls["latency_images"], calls["warmup"]) == (None, 200, 20)
