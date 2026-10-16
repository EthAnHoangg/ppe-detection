"""Train torchvision Faster R-CNN on the Construction Site Safety data, logging to ClearML.

    uv run python -m ppe.frcnn.train --config configs/fasterrcnn.yaml --seed 0
"""

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ppe.config import parse_args
from ppe.data import CLASS_NAMES, dataset_root, train_val_dirs
from ppe.frcnn.dataset import YoloDetectionDataset
from ppe.seed import seed_everything
from ppe.tracking import start_run

NUM_CLASSES = len(CLASS_NAMES) + 1  # + background


def build_model(imgsz: int, pretrained: bool = True) -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN with a fresh 11-way head, resizing to `imgsz` like YOLO (not 800/1333)."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=imgsz, max_size=imgsz)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def summarise_map(result: dict) -> dict:
    """torchmetrics mAP output -> YOLO's metric names (so both models share charts) plus per-class mAP50-95."""
    per_class = result["map_per_class"].flatten().tolist()
    classes = result["classes"].flatten().tolist()
    return {
        "metrics/mAP50-95(B)": result["map"].item(),
        "metrics/mAP50(B)": result["map_50"].item(),
        "per_class": {CLASS_NAMES[c - 1]: v for c, v in zip(classes, per_class)},
    }


def dataloader_workers(configured: int, platform: str = sys.platform) -> int:
    """0 on macOS: torch's shm-manager processes there inherit the resource-tracker pipe and hang interpreter exit."""
    return 0 if platform == "darwin" else configured


def _collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        metric.update([{k: v.cpu() for k, v in out.items()} for out in outputs], list(targets))
    return summarise_map(metric.compute())


def learning_rate(step: int, epoch: int, base: float, warmup_iters: int, milestones: list[int], gamma: float) -> float:
    """Linear warmup over the first `warmup_iters` steps, then x`gamma` at each milestone epoch passed."""
    warmup = 0.001 + 0.999 * step / warmup_iters if step < warmup_iters else 1.0
    return base * warmup * gamma ** sum(epoch >= m for m in milestones)


def main() -> None:
    cfg = parse_args("Train Faster R-CNN with ClearML tracking")
    task = start_run(cfg)
    logger = task.get_logger()
    seed_everything(cfg["seed"])
    hp = cfg["frcnn"]

    train_dir, valid_dir = train_val_dirs(dataset_root(cfg))
    limit = cfg["smoke"]["images"] if cfg["smoke"]["enabled"] else None
    train_set = YoloDetectionDataset(train_dir, limit=limit, hflip=True)
    valid_set = YoloDetectionDataset(valid_dir, limit=limit)
    generator = torch.Generator().manual_seed(cfg["seed"])
    workers = dataloader_workers(hp["workers"])
    train_loader = DataLoader(
        train_set, batch_size=hp["batch"], shuffle=True, num_workers=workers,
        collate_fn=_collate, generator=generator,
    )
    valid_loader = DataLoader(valid_set, batch_size=hp["batch"], num_workers=workers, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["imgsz"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"])
    epochs = cfg["epochs"]
    milestones = [math.ceil(f * epochs) for f in hp["lr_milestones"]]
    warmup_iters = min(hp["warmup_iters"], len(train_loader) * epochs)

    run_dir = Path("runs").resolve() / task.id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_map, epochs_without_gain, step = -1.0, 0, 0

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            lr = learning_rate(step, epoch, hp["lr"], warmup_iters, milestones, hp["lr_gamma"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()} at step {step}; lower frcnn.lr in the config")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % hp["log_every"] == 0:
                logger.report_scalar("train (per step)", "loss", loss.item(), iteration=step)
                for name, value in losses.items():
                    logger.report_scalar("train (per step)", name, value.item(), iteration=step)
                logger.report_scalar("lr", "lr", lr, iteration=step)
            step += 1

        summary = evaluate(model, valid_loader, device)
        for name in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            logger.report_scalar("metrics", name, summary[name], iteration=epoch)
        for class_name, value in summary["per_class"].items():
            logger.report_scalar("val per-class mAP50-95", class_name, value, iteration=epoch)
        print(f"epoch {epoch}: mAP50={summary['metrics/mAP50(B)']:.4f} mAP50-95={summary['metrics/mAP50-95(B)']:.4f}")

        if summary["metrics/mAP50-95(B)"] > best_map:
            best_map, epochs_without_gain = summary["metrics/mAP50-95(B)"], 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg["patience"]:
                print(f"early stop: no val mAP50-95 gain for {cfg['patience']} epochs")
                break

    logger.report_single_value("best mAP50-95(B)", best_map)
    if task.is_offline():
        print(f"offline: model not registered in ClearML; best val mAP50-95 {best_map:.4f}, weights at {best_path}")
    else:
        from clearml import OutputModel

        output = OutputModel(
            task=task, name=f"{cfg['model']['name']}-seed{cfg['seed']}", framework="PyTorch",
            label_enumeration={name: i + 1 for i, name in enumerate(CLASS_NAMES)},
        )
        output.update_weights(weights_filename=str(best_path), auto_delete_file=False)
        print(f"best val mAP50-95 {best_map:.4f}; weights at {best_path}")

    # Weights are saved/uploaded above, so an evaluation failure can't lose them.
    from ppe.evaluate import run_validation_eval
    from ppe.predict import FrcnnPredictor

    run_validation_eval(task, cfg, FrcnnPredictor.load(best_path, cfg["imgsz"]), valid_dir)

if __name__ == "__main__":
    main()
