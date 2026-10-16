import torch
from PIL import Image

from ppe.frcnn.dataset import YoloDetectionDataset, horizontal_flip, yolo_to_target


def test_converts_normalised_cxcywh_to_absolute_xyxy():
    target = yolo_to_target("5 0.5 0.5 0.2 0.4", width=640, height=320)
    assert torch.allclose(target["boxes"], torch.tensor([[256.0, 96.0, 384.0, 224.0]]))


def test_shifts_class_ids_by_one_for_background():
    target = yolo_to_target("0 0.5 0.5 0.1 0.1\n9 0.5 0.5 0.1 0.1", width=100, height=100)
    assert target["labels"].tolist() == [1, 10]


def test_keeps_every_box_in_order():
    text = "0 0.25 0.25 0.1 0.1\n7 0.75 0.75 0.2 0.2\n5 0.5 0.5 0.5 0.5\n"
    target = yolo_to_target(text, width=100, height=100)
    assert target["boxes"].shape == (3, 4)
    assert target["labels"].tolist() == [1, 8, 6]


def test_empty_label_gives_correctly_shaped_empty_tensors():
    target = yolo_to_target("", width=640, height=640)
    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].shape == (0,)
    assert target["labels"].dtype == torch.int64


def _write_split(root, name, size, label_text):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(root / "images" / f"{name}.jpg")
    (root / "labels" / f"{name}.txt").write_text(label_text)


def test_dataset_returns_image_tensor_and_target_using_real_image_size(tmp_path):
    _write_split(tmp_path, "a", (200, 100), "2 0.5 0.5 1.0 1.0")
    image, target = YoloDetectionDataset(tmp_path)[0]
    assert image.shape == (3, 100, 200)
    assert image.dtype == torch.float32 and image.max() <= 1.0
    assert torch.allclose(target["boxes"], torch.tensor([[0.0, 0.0, 200.0, 100.0]]))
    assert target["labels"].tolist() == [3]


def test_dataset_limit_keeps_first_n_images(tmp_path):
    for name in "abc":
        _write_split(tmp_path, name, (10, 10), "")
    assert len(YoloDetectionDataset(tmp_path, limit=2)) == 2


def test_horizontal_flip_mirrors_boxes_and_keeps_x1_left_of_x2():
    image = torch.zeros(3, 10, 100)
    image[:, :, 0] = 1.0
    target = {"boxes": torch.tensor([[10.0, 2.0, 30.0, 8.0]]), "labels": torch.tensor([1])}
    flipped_image, flipped = horizontal_flip(image, target)
    assert torch.equal(flipped["boxes"], torch.tensor([[70.0, 2.0, 90.0, 8.0]]))
    assert flipped_image[0, 0, -1] == 1.0
    assert torch.equal(target["boxes"], torch.tensor([[10.0, 2.0, 30.0, 8.0]]))
