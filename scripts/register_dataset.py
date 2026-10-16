"""One-time: upload the Construction Site Safety data as a ClearML dataset version.

    uv run python scripts/register_dataset.py --data-dir data/css-data

Prints the dataset ID; put it in configs/common.yaml (dataset.id) and commit.
For a later version (e.g. a de-duplicated split) pass --version 2 --parent <v1 id>.
"""

import argparse
import os
from pathlib import Path

from clearml import Dataset

from ppe.tracking import check_credentials

SPLITS = ("train", "valid", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder holding train/, valid/, test/")
    parser.add_argument("--project", default="PPE-Detection/datasets")
    parser.add_argument("--name", default="construction-site-safety")
    parser.add_argument("--version", default="1")
    parser.add_argument("--parent", help="ID of the dataset version this one is derived from")
    args = parser.parse_args()

    check_credentials(os.environ, offline=False)
    missing = [s for s in SPLITS if not (args.data_dir / s / "images").is_dir()]
    if missing:
        raise SystemExit(f"{args.data_dir} is missing splits: {', '.join(missing)}")

    dataset = Dataset.create(
        dataset_name=args.name,
        dataset_project=args.project,
        dataset_version=args.version,
        parent_datasets=[args.parent] if args.parent else None,
    )
    for split in SPLITS:
        # Only images and labels: skips e.g. labels.cache files Ultralytics writes next to local data.
        dataset.add_files(path=args.data_dir / split, wildcard=["*.jpg", "*.txt"], dataset_path=split)
    for readme in sorted(args.data_dir.glob("README*.txt")):
        dataset.add_files(path=readme)
    # Single-threaded: ClearML's parallel zip merge intermittently fails with 'Bad file descriptor'.
    dataset.upload(max_workers=1)
    dataset.finalize()
    print(f"dataset id: {dataset.id}\nset dataset.id in configs/common.yaml to this value and commit")


if __name__ == "__main__":
    main()
