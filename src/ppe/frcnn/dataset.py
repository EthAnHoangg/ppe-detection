"""YOLO-format detection data for torchvision's Faster R-CNN."""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def yolo_to_target(label_text: str, width: int, height: int) -> dict[str, torch.Tensor]:
    """Convert YOLO label rows (`cls cx cy w h`, normalised) to a torchvision target.

    Boxes become absolute `[x1, y1, x2, y2]`, and labels shift by +1 because
    torchvision reserves class 0 for background.
    """
    rows = [line.split() for line in label_text.splitlines() if line.strip()]
    if not rows:
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}
    values = torch.tensor([[float(v) for v in row[:5]] for row in rows], dtype=torch.float32)
    cx, cy, w, h = values[:, 1] * width, values[:, 2] * height, values[:, 3] * width, values[:, 4] * height
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    labels = values[:, 0].to(torch.int64) + 1
    return {"boxes": boxes, "labels": labels}


class YoloDetectionDataset(torch.utils.data.Dataset):
    """Images from `root/images` paired with labels from `root/labels`."""

    def __init__(self, root: Path, limit: int | None = None, hflip: bool = False):
        self.images = sorted((Path(root) / "images").glob("*.jpg"))[:limit]
        self.labels_dir = Path(root) / "labels"
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        path = self.images[index]
        with Image.open(path) as img:
            image = to_tensor(img.convert("RGB"))
        label_path = self.labels_dir / f"{path.stem}.txt"
        label_text = label_path.read_text() if label_path.exists() else ""
        target = yolo_to_target(label_text, width=image.shape[2], height=image.shape[1])
        if self.hflip and torch.rand(()) < 0.5:
            image, target = horizontal_flip(image, target)
        return image, target


def horizontal_flip(image: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    width = image.shape[2]
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] = width - target["boxes"][:, [2, 0]]
    return image.flip(-1), {**target, "boxes": boxes}
