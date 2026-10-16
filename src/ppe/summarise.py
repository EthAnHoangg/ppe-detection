"""Mean ± std over seeds of the final test evaluations, per model (report Section 4.2).

    uv run python -m ppe.summarise [--tag report]
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from ppe.evaluate import headline_metrics


def aggregate(entries: list[tuple[str, dict]]) -> dict[str, dict[str, tuple[float, float, int]]]:
    """(model, results) pairs -> model -> metric -> (mean, sample std, n). NaN values are left out of n."""
    by_model: dict[str, dict[str, list[float]]] = {}
    for model, results in entries:
        for name, value in headline_metrics(results).items():
            by_model.setdefault(model, {}).setdefault(name, []).append(value)
    summary = {}
    for model, metrics in by_model.items():
        summary[model] = {}
        for name, values in metrics.items():
            x = np.array([v for v in values if not math.isnan(v)])
            mean = float(x.mean()) if len(x) else float("nan")
            std = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
            summary[model][name] = (mean, std, len(x))
    return summary


def _cell(mean: float, std: float, n: int) -> str:
    if n == 0:
        return "–"
    return f"{mean:.3f} ± {std:.3f} (n={n})" if not math.isnan(std) else f"{mean:.3f} (n={n})"


def to_markdown(summary: dict) -> str:
    models = sorted(summary)
    metrics = sorted({m for model in models for m in summary[model]})
    lines = [f"| metric | {' | '.join(models)} |", f"|---|{'---|' * len(models)}"]
    for metric in metrics:
        cells = [_cell(*summary[model].get(metric, (float("nan"), float("nan"), 0))) for model in models]
        lines.append(f"| {metric} | {' | '.join(cells)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", help="only final evaluations whose training run carried this tag")
    parser.add_argument("--project", default="PPE-Detection/evaluation")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    from clearml import Task

    from ppe.tracking import check_credentials

    check_credentials(os.environ, offline=False)
    tags = ["final-eval", *([args.tag] if args.tag else [])]
    # ClearML ORs tag filters by default; "__$all" makes it require every tag.
    tasks = Task.get_tasks(project_name=args.project, tags=["__$all", *tags], task_filter={"status": ["completed"]})
    entries = []
    for task in tasks:
        model = next((t.split(":", 1)[1] for t in task.get_tags() if t.startswith("model:")), "unknown")
        results = json.loads(Path(task.artifacts["test_eval"].get_local_copy()).read_text())["results"]
        entries.append((model, results))
    if not entries:
        raise SystemExit(f"no completed final-eval tasks in {args.project} with tags {tags}")

    summary = aggregate(entries)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "mean", "std", "n"])
        for model, metrics in sorted(summary.items()):
            for name, (mean, std, n) in sorted(metrics.items()):
                writer.writerow([model, name, mean, std, n])
    markdown = to_markdown(summary)
    (args.out / "summary.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
