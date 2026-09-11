"""Re-export the durations plot-data behind each example page's Performance tile.

Each example script in ``docs/src/examples/`` already writes this file with the
right options when it runs, so this is for rebuilding the committed artifact
from an existing ``*-profile.h5`` -- after changing how the chart reads the
data, say -- without re-running a simulation that takes minutes.

It passes the same two options the tile depends on, and they must stay in step
with the ``plot_durations`` call in those scripts:

* ``--stack-children`` splits each bar into the region's own time plus one
  segment per region it calls, so a bar shows where its time actually went
  instead of only how tall it is. The decomposition has to come from the
  exporter -- ``@scope-profiler/plotly``'s durations builder stacks when, and
  only when, the bars carry a ``segment`` field.
* ``--sort-by total`` orders the regions by total time, descending, which is the
  order the chart draws them in (both exported metrics share it).

Stacking only decomposes ``total`` and ``avg`` -- the exporter refuses the other
statistics rather than silently ignoring the flag -- so those are the two metrics
the page offers. ``min``/``max`` are still in the region-statistics table below
the chart.

The gantt, flame and region-statistics exports are left alone; this touches
durations only.

Usage:

    python scripts/export_durations_data.py            # all examples
    python scripts/export_durations_data.py maxwell-wave
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "public" / "examples"
METRICS = ("total", "avg")


def prune_zero_segments(payload: dict) -> dict:
    """Drop the zero-valued region/segment pairs the exporter pads the grid with.

    The export is a dense region x segment matrix, but a call graph is sparse:
    for the examples here roughly 90% of the entries are zeros for pairs where
    one region never calls the other. The durations builder reads missing pairs
    as null and Plotly skips them, so dropping the zeros changes nothing on the
    page and takes the file from ~220 KB to ~20 KB.
    """
    bars = payload.get("bars", [])
    payload["bars"] = [bar for bar in bars if bar.get("value_seconds")]
    return payload


def export(profile: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(
            [
                "scope-profiler", "export", "plot-data", str(profile),
                "--output", scratch,
                "--label", "profiling_data",
                "--format", "json",
                "--plots", "durations",
                "--metrics", *METRICS,
                "--sort-by", "total",
                "--stack-children",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        exported = Path(scratch) / "durations_data.json"
        payload = prune_zero_segments(json.loads(exported.read_text(encoding="utf-8")))

    before = destination.stat().st_size if destination.exists() else 0
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"  {destination.name}: {len(payload['bars'])} bars, "
        f"{before / 1024:.0f} KB -> {destination.stat().st_size / 1024:.0f} KB"
    )


def main(names: list[str]) -> int:
    if shutil.which("scope-profiler") is None:
        print("scope-profiler is not on PATH; install it to regenerate profiling data.")
        return 1

    profiles = sorted(EXAMPLES_DIR.glob("*-profile.h5"))
    if names:
        wanted = {name.removesuffix("-profile.h5").removesuffix(".h5") for name in names}
        profiles = [path for path in profiles if path.name.removesuffix("-profile.h5") in wanted]
        if not profiles:
            print(f"No profiles matched {sorted(wanted)}")
            return 1

    print(f"Re-exporting durations data for {len(profiles)} example(s):")
    for profile in profiles:
        export(profile, EXAMPLES_DIR / f"{profile.name.removesuffix('-profile.h5')}-durations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
