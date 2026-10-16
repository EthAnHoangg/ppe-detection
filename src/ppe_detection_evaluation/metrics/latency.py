from collections.abc import Sequence
from typing import Any

import numpy as np

from ..records import ImageRecord


def compute_latency(records: Sequence[ImageRecord]) -> dict[str, Any]:
    t = np.array([r.latency_ms for r in records if r.latency_ms is not None], dtype=np.float64)
    if not len(t):
        return {"num_images": 0}
    mean = float(t.mean())
    return {
        "num_images": len(t),
        "mean_ms": mean,
        "std_ms": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "fps": 1000.0 / mean if mean > 0 else float("inf"),
    }
