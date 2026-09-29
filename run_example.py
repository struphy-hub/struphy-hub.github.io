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

Scripts that still publish by themselves (they import `plasma_plots.gallery`) are run as they are.

Examples listed in `.github/precomputed-examples.json` are too expensive to simulate here: their
archived run is downloaded (into `.cache/precomputed-examples/`), checked against its sha256 and
unpacked into `docs/public/examples/struphy_gallery_runs/`, and the script only post-processes it.
`--simulate` runs such an example's simulation anyway, e.g. on a cluster to make the archive
(`scripts/pack_precomputed_run.py` packs it).

Usage (under MPI, prefix `mpirun -n 4`; `python cli.py run <example>` calls this):

    python run_example.py maxwell-wave               # simulate, post-process, publish
    python run_example.py maxwell-wave --pproc-only  # post-process an earlier run again
    mpirun -n 32 python run_example.py itpa-tae-linear-mhd --simulate  # scripts/submit_precomputed_run.py's job
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "docs" / "src" / "examples"
OUTPUT_DIR = ROOT / "docs" / "public" / "examples"
IMAGES_DIR = ROOT / "docs" / "public" / "images" / "examples"
PRECOMPUTED = ROOT / ".github" / "precomputed-examples.json"
REGISTRY = PRECOMPUTED.relative_to(ROOT)  # as named in messages
DOWNLOADS = ROOT / ".cache" / "precomputed-examples"
RUNS_DIR = OUTPUT_DIR / "struphy_gallery_runs"  # the scripts' EnvironmentOptions(out_folders=...)


def precomputed_run(stem: str) -> dict | None:
    """The archived run of this example from `.github/precomputed-examples.json`, if it has one."""
    if not PRECOMPUTED.is_file():
        return None
    return json.loads(PRECOMPUTED.read_text()).get(stem)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_precomputed_run(stem: str, entry: dict) -> None:
    """Download, check and unpack the archived run into RUNS_DIR, unless it is there already."""
    run = RUNS_DIR / entry["sim_folder"]
    if (run / "data").is_dir():
        print(f"{stem}: using the existing run {os.path.relpath(run, ROOT)}")
        return
    if not entry.get("url") or not entry.get("sha256"):
        raise SystemExit(
            f"{stem}: no archived run published yet; set its url and sha256 in "
            f"{REGISTRY}, or simulate it with --simulate"
        )
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOADS / f"{stem}-{entry['sha256'][:12]}.tar.gz"
    if not archive.is_file() or sha256(archive) != entry["sha256"]:
        print(f"{stem}: downloading {entry['url']}")
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(entry["url"], partial)
        if sha256(partial) != entry["sha256"]:
            partial.unlink()
            raise SystemExit(f"{stem}: the download does not match the sha256 in {REGISTRY}")
        partial.replace(archive)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tops = {Path(member.name).parts[0] for member in tar.getmembers()}
        if tops != {entry["sim_folder"]}:
            raise SystemExit(f"{stem}: the archive must hold only {entry['sim_folder']}/, not {sorted(tops)}")
        tar.extractall(RUNS_DIR, filter="data")
    print(f"{stem}: unpacked {archive.name} into {os.path.relpath(run, ROOT)}")


def publishes_by_itself(script: Path) -> bool:
    """Whether the script writes the website's files itself, with the gallery helpers (the older convention)."""
    return "plasma_plots.gallery" in script.read_text()


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
    parser.add_argument(
        "--simulate", action="store_true", help="simulate even an example whose archived run is downloaded otherwise"
    )
    args = parser.parse_args(argv)

    stem = args.example
    script = EXAMPLES_DIR / f"{stem}.py"
    if not script.is_file():
        raise SystemExit(f"no example {script.relative_to(ROOT)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_DIR)  # the scripts save their figures and scratch runs in the current directory

    pproc_only = args.pproc_only
    entry = precomputed_run(stem)
    if entry is not None and not args.simulate:
        if any(int(os.environ.get(name, "1")) > 1 for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE")):
            raise SystemExit(f"{stem}: post-process its archived run on one rank (or pass --simulate)")
        fetch_precomputed_run(stem, entry)
        pproc_only = True

    sys.argv = [str(script), *(["--pproc-only"] if pproc_only else [])]
    if publishes_by_itself(script):
        runpy.run_path(str(script), run_name="__main__")
        return

    simulations = profiled_simulations()
    runpy.run_path(str(script), run_name="__main__")

    from plasma_plots.gallery import export_profiling, is_root, merge_metadata

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
