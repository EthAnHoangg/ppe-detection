from dataclasses import dataclass

BOX_FORMATS = ("xyxy", "xywh", "cxcywh")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    box_format: str = "xyxy"
    normalized: bool = False
    source_size: tuple[int, int] | None = None
    label_offset: int = 0

    def __post_init__(self) -> None:
        if self.box_format not in BOX_FORMATS:
            raise ValueError(f"box_format must be one of {BOX_FORMATS}, got {self.box_format!r}")


GT_CONFIG = ModelConfig("ground_truth", box_format="cxcywh", normalized=True)

FASTER_RCNN = ModelConfig("Faster R-CNN", label_offset=1)
SSD300 = ModelConfig("SSD300", label_offset=1)

YOLO26_M = ModelConfig("YOLO26-M")
RTDETR_L = ModelConfig("RT-DETR-L")

ALL_MODELS: tuple[ModelConfig, ...] = (FASTER_RCNN, SSD300, YOLO26_M, RTDETR_L)
