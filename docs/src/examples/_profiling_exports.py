"""Compatibility helpers for gallery profiling exports."""

from __future__ import annotations

import json
from pathlib import Path

from scope_profiler import plot_durations as _plot_durations


def plot_durations(*args, metrics=("total",), **kwargs):
    """Export several metrics with scope-profiler's one-metric API."""
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    data_filepath = kwargs.pop("data_filepath", None)
    if not data_filepath or len(metrics) == 1:
        return _plot_durations(*args, metric=metrics[0], data_filepath=data_filepath, **kwargs)

    path = Path(data_filepath)
    payloads = []
    results = []
    for metric in metrics:
        temporary_path = path.with_name(f"{path.stem}.{metric}{path.suffix}")
        results.extend(_plot_durations(*args, metric=metric, data_filepath=temporary_path, **kwargs))
        payloads.append(json.loads(temporary_path.read_text()))
        temporary_path.unlink()

    merged = payloads[0]
    merged["bars"] = [bar for payload in payloads for bar in payload["bars"]]
    merged["metrics"] = list(metrics)
    path.write_text(json.dumps(merged))
    return results
