from pathlib import Path

from ppe.yolo_train import train_kwargs


def _cfg(smoke=False):
    return {
        "model": {"name": "yolo26s"},
        "imgsz": 640,
        "epochs": 100,
        "patience": 20,
        "seed": 1,
        "smoke": {"enabled": smoke, "images": 16},
        "yolo": {"batch": 16, "workers": 4, "train_args": {"cos_lr": True}},
    }


def test_train_kwargs_carry_shared_settings_and_extra_args():
    kwargs = train_kwargs(_cfg(), Path("/r/data.yaml"), Path("/r"))
    assert kwargs["data"] == "/r/data.yaml"
    assert (kwargs["imgsz"], kwargs["epochs"], kwargs["patience"], kwargs["seed"]) == (640, 100, 20, 1)
    assert kwargs["deterministic"] is True
    assert (kwargs["batch"], kwargs["workers"], kwargs["cos_lr"]) == (16, 4, True)
    assert (kwargs["project"], kwargs["name"]) == ("/r", "yolo26s-seed1")
    assert "fraction" not in kwargs


def test_smoke_limits_train_and_val_image_counts():
    assert train_kwargs(_cfg(smoke=True), Path("/r/data.yaml"), Path("/r"))["fraction"] == [16, 16]
