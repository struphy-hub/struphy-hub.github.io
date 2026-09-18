"""Shared output helpers for the gallery scripts: figure files, profiling exports, metadata.

Every helper writes to the current directory, which is `docs/public/examples/` when a script is
run as described in README.md, and names its files after the script stem.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# A gantt bar is one call, not an aggregate, and a run with thousands of steps draws tens of
# thousands of near-identical bars -- heavy enough to hang the tab. Keeping the first
# GANTT_MAX_INTERVALS in time order leaves the setup phase plus several complete step-loop iterations.
GANTT_MAX_INTERVALS = 5000


def save_figure(figure, stem: str, *, width: int = 1100, height: int = 650, suffix: str = "") -> None:
    """Write `<stem><suffix>.png` (static fallback) and `<stem><suffix>.html` (interactive)."""
    png_path = Path(f"{stem}{suffix}.png")
    html_path = Path(f"{stem}{suffix}.html")
    figure.write_image(png_path, width=width, height=height, scale=2)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {html_path.resolve()}")


def merge_metadata(stem: str, **fields) -> Path:
    """Add result fields to `<stem>.metadata.json`, keeping the structural fields already there."""
    path = Path(f"{stem}.metadata.json")
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(fields)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {path.resolve()}")
    return path


def export_profiling(sim, stem: str) -> dict:
    """Export scope-profiler data of a run made with `profiling_activated=True`.

    Writes the raw HDF5 plus the plot-data JSON (durations, gantt, region statistics) that the
    example page renders as native Plotly figures. Returns the metadata fields that point to them,
    ready for `merge_metadata`.
    """
    from _profiling_exports import plot_durations
    from scope_profiler import plot_gantt, read_h5, write_region_statistics_json

    profile_reader = read_h5(sim.profiling_filepath)
    profile_h5_path = Path(f"{stem}-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    # `stack_children` splits each bar into the region's own time plus one segment per region it
    # calls, which is what the page's durations chart stacks; it only decomposes total/avg.
    # `sort_by` fixes the region order the chart draws, biggest total first.
    durations_path = Path(f"{stem}-durations.json")
    plot_durations(
        [profile_reader],
        ranks=[0],
        metrics=("total", "avg"),
        sort_by="total",
        stack_children=True,
        data_filepath=durations_path,
        data_format="json",
        verbose=False,
    )
    # The stacked export is a dense region x segment grid, but a call graph is sparse (~90% zeros
    # for regions that never call each other); the chart reads a missing pair as null, so dropping
    # them cuts the file ~10x with no change on the page.
    durations_payload = json.loads(durations_path.read_text())
    durations_payload["bars"] = [bar for bar in durations_payload["bars"] if bar["value_seconds"]]
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path(f"{stem}-gantt.json")
    region_stats_path = Path(f"{stem}-region-stats.json")
    plot_gantt([profile_reader], ranks=[0], data_filepath=gantt_path, data_format="json", verbose=False)
    write_region_statistics_json([profile_reader], region_stats_path, ranks=[0])

    gantt_payload = json.loads(gantt_path.read_text())
    if len(gantt_payload["intervals"]) > GANTT_MAX_INTERVALS:
        gantt_payload["intervals"] = sorted(gantt_payload["intervals"], key=lambda c: c["start_seconds"])[
            :GANTT_MAX_INTERVALS
        ]
        gantt_path.write_text(json.dumps(gantt_payload))

    print(f"Saved {profile_h5_path.resolve()}")
    print(f"Saved {durations_path.resolve()}, {gantt_path.resolve()}, {region_stats_path.resolve()}")
    return {
        "profilingData": f"/examples/{profile_h5_path.name}",
        "profilingDurations": f"/examples/{durations_path.name}",
        "profilingGantt": f"/examples/{gantt_path.name}",
        "profilingRegionStats": f"/examples/{region_stats_path.name}",
    }
