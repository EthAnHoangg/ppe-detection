# ClearML MLOps workflow for PPE detection: design

Date: 2026-10-16
Status: approved in conversation, pending written-spec review

## Goal

Team experiment tracking. Every training run of YOLO26 and Faster R-CNN goes into one shared ClearML project, with its settings, metrics, code version, data version and trained weights. The team can then compare the two models and pick the runs whose numbers go into the report.

### Done when

An account holder can do all of the following:

1. Clone the repo, run `uv sync --locked` and set their ClearML keys.
2. Train either model with one command.
3. See the run in the shared ClearML project with its settings, per-epoch metrics, git commit, dataset ID and a downloadable output model.

## Constraints and decisions

| Topic | Decision |
|---|---|
| ClearML hosting | Free hosted tier at app.clear.ml: 3 user accounts, 100 GB artifact storage, 1M API calls/month. Up to 3 members train; the rest see results through shared read-only run links or exported tables and plots. |
| ClearML features in scope | Experiment tracking, dataset versioning (ClearML Data), model storage (output models). |
| Out of scope | Remote execution (clearml-agent and queues), evaluation as tracked tasks, pipelines, hyperparameter optimisation, resuming interrupted runs. The final test-set evaluation is a separate, untracked step. |
| Code distribution | A private, team-shared git repository. Members run scripts from a terminal on Colab, Kaggle or a lab GPU machine. No notebooks. |
| Repo contents | Code, configs, tests and this design doc only. The report files (`Assignment1.docx`, `Assignment1.md`, `Evaluation_draft.md`) and `CLAUDE.md` are gitignored, as are `data/`, `models/`, `runs/`, `.venv/` and `.env`. |
| Models | YOLO26s (Ultralytics) and Faster R-CNN with a ResNet-50 FPN backbone (torchvision). |
| Logging approach | One shared entry point (`start_run`) opens the ClearML Task for both models. For YOLO, the Ultralytics ClearML callback attaches to that existing Task (`Task.current_task()`) and logs metrics, plots and weights. For Faster R-CNN, our own training loop reports through `task.get_logger()`. MMDetection was rejected because mmcv lags current PyTorch. |
| Environment | The existing uv project: Python 3.12, ultralytics 8.4.178; torch 2.2.2 on Intel Macs and 2.14.1 on Linux. This work adds `clearml`, `pycocotools` and `torchmetrics`, plus `pytest` as a dev dependency. |

## Repository layout

```
pyproject.toml, uv.lock, .python-version, .gitignore, .env.example
configs/
  common.yaml          # shared settings that must match across models
  yolo26s.yaml         # YOLO-specific settings
  fasterrcnn.yaml      # Faster R-CNN-specific settings
src/ppe/
  __init__.py
  config.py            # load and merge common + model config, apply CLI overrides
  tracking.py          # start_run(): the only module that calls Task.init
  data.py              # fetch_dataset(), YOLO data-yaml writer, class names
  yolo_train.py        # entry point: python -m ppe.yolo_train
  frcnn/
    __init__.py
    dataset.py         # YOLO txt labels -> torchvision targets
    train.py           # entry point: python -m ppe.frcnn.train
scripts/
  register_dataset.py  # one-time upload of data/css-data as ClearML dataset v1
tests/
docs/superpowers/specs/  # this file
```

The project becomes an installable package (src layout with a build backend), so entry points run as `uv run python -m ppe.<module>` without patching `sys.path`.

## Components

### `config.py`

- `load_config(model_config_path, overrides) -> dict` reads `configs/common.yaml`, deep-merges the model YAML over it, then applies CLI overrides.
- Only these overrides are allowed: `--seed`, `--epochs` and `--smoke`. Any other setting is changed by editing a config file and committing it.
- `seed` must be one of `common.seeds`, unless `--smoke` is set.

### `tracking.py`

`start_run(cfg, model_name) -> Task` does the following:

1. Checks that `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are set. If not, it raises an error naming the missing variables and pointing to `.env.example`. This check is skipped when offline mode is active, which the tests use.
2. Warns, but continues, if the git working tree has uncommitted changes. ClearML stores the diff.
3. Calls `Task.init` with:
   - `project_name`: `common.project`, or `<project>/smoke` when `--smoke` is set.
   - `task_name`: `"<model_name> · seed<N>"`.
   - `tags`: `model:<name>`, `data:<version>`, `seed:<N>`, plus `smoke` when relevant.
4. Attaches the merged config with `task.connect(cfg)`, so the values appear as editable hyperparameters in the UI.
5. Returns the Task.

ClearML records the git commit, the uncommitted diff, installed packages and machine info automatically.

### `data.py`

- `fetch_dataset(dataset_id) -> Path` calls `Dataset.get(dataset_id).get_local_copy()` and returns the cached root, which contains `train/` and `valid/`. The cached copy is read-only.
- `train_val_dirs(root) -> (train, valid)` is the only accessor the training code uses. Nothing in the training path returns or touches `test/`.
- `write_yolo_yaml(root, out_dir) -> Path` writes the Ultralytics data YAML into the run's output directory (not into the read-only cache). The YAML holds absolute `train`/`val` paths, `nc: 10` and the class names. It deliberately has no `test` key.
- `CLASS_NAMES` lists the ten classes in ID order: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle.

### `yolo_train.py`

1. Parses the arguments and calls `load_config`.
2. Calls `start_run`.
3. Seeds the run, then calls `fetch_dataset` and `write_yolo_yaml`.
4. Calls `YOLO(cfg.yolo.weights).train(...)` with `data`, `imgsz`, `epochs`, `patience`, `seed`, `deterministic=True`, `batch`, `project=runs/`, `name=<task id>`, and the YOLO hyperparameters from the config.
5. The Ultralytics callback logs per-epoch losses, P/R/mAP50/mAP50-95, plots, and `best.pt` as the output model.

With `--smoke`, it trains for 1 epoch with `fraction` set so only a few images are used.

### `frcnn/dataset.py`

- A `torch.utils.data.Dataset` over an `images/` and `labels/` directory pair.
- Each YOLO row `cls cx cy w h` (normalised) becomes an absolute `[x1, y1, x2, y2]` box, using the actual image width and height, with `label = cls + 1` (torchvision reserves 0 for background).
- An empty label file yields `boxes` of shape `(0, 4)` as float32 and `labels` of shape `(0,)` as int64.
- Images are returned as float tensors in [0, 1]. No augmentation beyond horizontal flip on train.

### `frcnn/train.py`

- `fasterrcnn_resnet50_fpn_v2` from torchvision with COCO-pretrained weights. The box predictor is replaced for 11 classes (10 plus background).
- `min_size = max_size = common.imgsz` (640). This overrides torchvision's 800/1333 default so both models see the same resolution.
- The optimizer and learning-rate schedule come from `fasterrcnn.yaml`.
- Logs `train/loss` and its components every `log_every` steps.
- After each epoch:
  - computes validation mAP50 and mAP50-95 (overall and per class) with `torchmetrics` `MeanAveragePrecision` (pycocotools backend), and reports them as `val/mAP50`, `val/mAP50-95` and `val/mAP50-95/<class>`;
  - applies early stopping on `val/mAP50-95`, with patience from `common`;
  - saves the best checkpoint locally.
- At the end, the best checkpoint is uploaded with `OutputModel(task=task, framework="PyTorch").update_weights(...)`.
- Metric titles shared with YOLO use the same names wherever possible, so both models plot on the same charts.
- `--smoke` runs 1 epoch on a handful of images.

### `scripts/register_dataset.py`

A one-time run by one account holder:

- `Dataset.create(dataset_project=<project>/datasets, dataset_name="construction-site-safety", dataset_version="1")`.
- Adds `data/css-data/{train,valid,test}` and `README.*.txt`, then uploads and finalises.
- Prints the dataset ID, which is then written into `configs/common.yaml` (`dataset.id`, `dataset.version: v1`) and committed.

The test split is included in the dataset so the later untracked test evaluation can use the same version. Training code never reads it. A de-duplicated split later becomes dataset version 2, created with v1 as its parent.

## Configuration

`configs/common.yaml`:

| Key | Value |
|---|---|
| `project` | `PPE-Detection` |
| `dataset.id` | Filled after registration |
| `dataset.version` | `v1` |
| `imgsz` | 640 |
| `seeds` | [0, 1, 2] |
| `epochs` | Maximum epochs (team decision; Section 4 bracket) |
| `patience` | Early stopping on validation mAP50-95 |

Model files:

- `configs/yolo26s.yaml`: `weights: yolo26s.pt`, `batch`, and optimizer settings left at the YOLO26 defaults unless the team decides otherwise.
- `configs/fasterrcnn.yaml`: `batch`, `lr`, `momentum`, `weight_decay`, the LR schedule and `log_every`.

The numeric values in brackets in Section 4 of the report become concrete in these files. The report then cites the committed config.

## Data flow (one run)

1. One-time machine setup:
   - `git clone`, `pip install uv`, `uv sync --locked` (on Linux this installs CUDA torch 2.14.1, about 2–3 GB).
   - Export the five `CLEARML_*` variables: `API_HOST`, `WEB_HOST`, `FILES_HOST`, `API_ACCESS_KEY`, `API_SECRET_KEY`.
2. `uv run python -m ppe.yolo_train --config configs/yolo26s.yaml --seed 0`
3. `load_config`, then `start_run` (Task created, config attached, git state captured).
4. `fetch_dataset` (cached after the first download).
5. Train. Metrics stream to the Task, and the best weights are uploaded as the output model.
6. Runs are compared in the ClearML UI. A run whose numbers go into the report is tagged `report` by hand, and read-only links are shared with members who don't have an account.

## Error handling

| Situation | Behaviour |
|---|---|
| ClearML keys missing | Fail before any download or training, naming the missing variables. |
| Uncommitted code changes | Warning only; ClearML stores the diff. Runs tagged `report` should come from clean commits (convention, not enforced). |
| Colab or Kaggle disconnects mid-run | The run is left aborted or failed in ClearML and is rerun from scratch. Resume is out of scope; revisit if Faster R-CNN can't finish within one session. |
| ClearML unreachable mid-run | The ClearML SDK buffers and retries; training is not interrupted. |
| `--seed` not in `common.seeds` (non-smoke) | Refuse to start. |

## Testing

`uv run pytest` runs on this Intel Mac on CPU, with no ClearML account; tests use ClearML offline mode (`Task.set_offline(True)`).

Unit tests:

- **Label conversion:**
  - normalised cxcywh to absolute xyxy, using the real image size;
  - class `+1` offset;
  - multiple boxes;
  - an empty label file gives correctly shaped empty tensors.
- **Config:**
  - merge order, with the model config overriding common;
  - only the allowed overrides apply;
  - an invalid seed is rejected;
  - `--smoke` routes the run to the smoke sub-project.
- **Tracking:**
  - task name and tags are formatted correctly;
  - missing keys raise the right error.
- **Data:**
  - the YOLO YAML has the right class names and `nc`, and no `test` key;
  - `train_val_dirs` never returns the test directory.

Smoke runs (offline, on this Mac):

- `--smoke` for both entry points against the local `data/css-data`, with `fetch_dataset` bypassed in offline mode through a `--data-dir` option that is only valid together with `--smoke`.
- This also confirms whether ultralytics 8.4.178 trains under torch 2.2.2 on this Mac. If it doesn't, Mac smoke runs are dropped and the Colab smoke run below is the end-to-end check.

Acceptance: one online `--smoke` run per model on Colab with a real account. The run must show its config, metrics, git commit, dataset tag and a downloadable output model in the dashboard.

## Open items (team decisions, not blockers)

- The values for `epochs`, `patience`, `seeds`, the YOLO batch size and the Faster R-CNN optimizer settings.
- Which three members hold ClearML accounts. The first of them creates the workspace and registers the dataset.
- Creating the private GitHub repo and inviting the team. This is done by a member; nothing is pushed by the agent without approval.
