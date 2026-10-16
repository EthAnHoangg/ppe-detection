import math

import pytest

from ppe.summarise import aggregate, to_markdown


def _results(map50, latency=10.0):
    return {"detection": {"groups": {"all": {"mAP50": map50, "mAP": 0.1}, "ppe": {"mAP50": map50, "mAP": 0.1}},
                          "per_class": {"Mask": {"AP50": None}}},
            "violation": {"mean": {}}, "image_level": {"any": {"ap": 0.5}},
            "latency": {"num_images": 3, "mean_ms": latency, "p95_ms": latency, "fps": 1000 / latency}}


def test_aggregate_mean_std_and_n_per_model():
    s = aggregate([("yolo26s", _results(0.5)), ("yolo26s", _results(0.7)), ("fasterrcnn-r50", _results(0.4))])
    mean, std, n = s["yolo26s"]["mAP50/all"]
    assert mean == pytest.approx(0.6) and std == pytest.approx(0.1414213, abs=1e-6) and n == 2


def test_single_seed_std_is_nan_not_zero():
    _, std, n = aggregate([("m", _results(0.5))])["m"]["mAP50/all"]
    assert math.isnan(std) and n == 1


def test_aggregate_ignores_missing_values():
    mean, std, n = aggregate([("m", _results(0.5)), ("m", _results(0.6))])["m"]["AP50/Mask"]
    assert math.isnan(mean) and n == 0


def test_markdown_has_one_column_per_model():
    md = to_markdown(aggregate([("a", _results(0.5)), ("b", _results(0.4))]))
    header = md.splitlines()[0]
    assert "| metric | a | b |" == header
    assert "0.500" in md


def test_collect_entries_groups_by_model_and_data_version():
    from ppe.summarise import collect_entries

    tasks = [
        {"tags": ["model:yolo26s", "data:v1", "seed:0", "final-eval"], "results": _results(0.5)},
        {"tags": ["model:yolo26s", "data:v2", "seed:0", "final-eval"], "results": _results(0.9)},
    ]
    assert [group for group, _ in collect_entries(tasks)] == ["yolo26s (data:v1)", "yolo26s (data:v2)"]


def test_collect_entries_rejects_duplicate_seeds():
    from ppe.summarise import collect_entries

    tasks = [{"tags": ["model:yolo26s", "data:v1", "seed:0"], "results": _results(0.5)},
             {"tags": ["model:yolo26s", "data:v1", "seed:0"], "results": _results(0.6)}]
    with pytest.raises(ValueError, match="seed:0"):
        collect_entries(tasks)
