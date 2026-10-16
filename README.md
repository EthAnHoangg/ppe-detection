# PPE detection: YOLO26 vs Faster R-CNN

Trains YOLO26s (Ultralytics) and Faster R-CNN R50-FPN v2 (torchvision) on the Roboflow Construction Site Safety dataset. Every run is tracked in the shared ClearML project `PPE-Detection`. The design is in `docs/superpowers/specs/2026-10-16-clearml-mlops-design.md`.

## Setup (once per machine: Colab terminal, Kaggle, lab GPU)

```bash
git clone <repo-url> && cd <repo>
pip install uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked        # Linux installs CUDA torch, ~2-3 GB
cp .env.example .env    # fill in your ClearML keys (app.clear.ml -> Settings -> Workspace)
set -a; source .env; set +a
```

## Train

```bash
uv run python -m ppe.yolo_train  --config configs/yolo26s.yaml    --seed 0
uv run python -m ppe.frcnn.train --config configs/fasterrcnn.yaml --seed 0
```

- `--seed` must be one of `seeds` in `configs/common.yaml`.
- `--smoke` runs 1 epoch on 16 train and 16 val images, logged to `PPE-Detection/smoke`. Use it to check that a new machine works.
- `--epochs N` overrides the epoch budget.
- Every other setting lives in `configs/`. Change it there and commit before running, so each run traces back to a commit.

To use a run's numbers in the report, tag the run `report` in the ClearML UI. Report runs should come from a clean commit.

## Dataset

The data comes from ClearML by the ID in `configs/common.yaml`. One account holder registers it once:

```bash
curl -L -o data/css.zip https://www.kaggle.com/api/v1/datasets/download/snehilsanyal/construction-site-safety-image-dataset-roboflow
unzip -q data/css.zip -d data && rm data/css.zip
uv run python scripts/register_dataset.py --data-dir data/css-data
# then put the printed ID in configs/common.yaml (dataset.id) and commit
```

The training code reads only `train/` and `valid/`. `test/` is kept for the final evaluation.

## Tests

```bash
uv run pytest
# offline end-to-end check without a ClearML account, using local data:
CLEARML_OFFLINE_MODE=1 uv run python -m ppe.yolo_train --config configs/yolo26s.yaml --smoke --data-dir data/css-data
```

On an Intel Mac, uv installs torch 2.2.2, the last release with Intel-Mac builds. That is enough for tests and smoke runs. Real training belongs on a GPU machine.
