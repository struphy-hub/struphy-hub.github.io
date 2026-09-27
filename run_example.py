"""Run one gallery example and add what the website needs on top of the example's own output.

An example script in `docs/src/examples/` is a plain script: run as `python <script>.py`, it
simulates, post-processes and saves its figures in the current directory, as `<stem>.html`, `.png`
and `.plotly.json` for the page's main figure and `<stem>-<key>.*` for the others (`--show` also
shows them). The page finds the other figures by the keys in `docs/src/data/example-config.ts`,
which also holds their alt texts and captions.

This runner runs the script in `docs/public/examples/`, where the site reads the figures, and adds
the website's part: it profiles the simulation, exports the profiling data (merging its fields
into `<stem>.metadata.json`, which `generate_examples.py` wrote) and copies the PNGs to
`docs/public/images/examples/` as thumbnails.

Scripts that still publish by themselves (they import `struphy_plots.gallery`) are run as they are.

Usage (under MPI, prefix `mpirun -n 4`; `python cli.py run <example>` calls this):

    python run_example.py maxwell-wave               # simulate, post-process, publish
    python run_example.py maxwell-wave --pproc-only  # post-process an earlier run again
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "docs" / "src" / "examples"
OUTPUT_DIR = ROOT / "docs" / "public" / "examples"
IMAGES_DIR = ROOT / "docs" / "public" / "images" / "examples"


def publishes_by_itself(script: Path) -> bool:
    """Whether the script writes the website's files itself, with the gallery helpers (the older convention)."""
    return "struphy_plots.gallery" in script.read_text()


def profiled_simulations():
    """Profile every simulation the script runs, and collect the simulations it creates.

    The script's own `sim.run()` then records scope-profiler data, as the website's profiling
    charts need, without the script asking for it.
    """
    from struphy import Simulation

    created = []
    original_init, original_run = Simulation.__init__, Simulation.run

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    def run(self, *args, **kwargs):
        kwargs.setdefault("profiling_activated", True)
        return original_run(self, *args, **kwargs)

    Simulation.__init__, Simulation.run = __init__, run
    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one gallery example and publish its results.")
    parser.add_argument("example", help="the script stem, e.g. maxwell-wave")
    parser.add_argument("--pproc-only", action="store_true", help="post-process an earlier run instead of simulating")
    args = parser.parse_args(argv)

    stem = args.example
    script = EXAMPLES_DIR / f"{stem}.py"
    if not script.is_file():
        raise SystemExit(f"no example {script.relative_to(ROOT)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_DIR)  # the scripts save their figures and scratch runs in the current directory

    sys.argv = [str(script), *(["--pproc-only"] if args.pproc_only else [])]
    if publishes_by_itself(script):
        runpy.run_path(str(script), run_name="__main__")
        return

    simulations = profiled_simulations()
    runpy.run_path(str(script), run_name="__main__")

    from struphy_plots.gallery import export_profiling, is_root, merge_metadata

    if not (OUTPUT_DIR / f"{stem}.plotly.json").is_file() and is_root():
        raise SystemExit(f"{stem}: the script saved no main figure {stem}.plotly.json")
    # The first simulation is the one the page describes (generate_examples.py uses the same).
    profiled = [sim for sim in simulations if Path(getattr(sim, "profiling_filepath", "") or "").is_file()]
    fields = export_profiling(profiled[0], stem) if profiled else {}
    merge_metadata(stem, **fields)
    if is_root():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for png in [OUTPUT_DIR / f"{stem}.png", *OUTPUT_DIR.glob(f"{stem}-*.png")]:
            if png.is_file():
                shutil.copyfile(png, IMAGES_DIR / png.name)


if __name__ == "__main__":
    main()
