#!/usr/bin/env python
"""Run, inspect and clean the Struphy gallery examples from one place.

Every example is a script in ``docs/src/examples/`` (see its README.md). Producing its page
takes a few steps that CI (``.github/workflows/build-site.yml``) does for each script:
generate the structural metadata, run the script inside ``docs/public/examples/``, copy the
PNGs to ``docs/public/images/examples/`` and delete the simulation scratch output.
This tool does the same locally for the examples you name, and prints every file it made.

    python cli.py list                      # what examples exist, and which have been run
    python cli.py run orszag-tang-vortex    # metadata + run + clean-up
    python cli.py run orszag dam --open     # unique prefixes work; --open shows their pages
    python cli.py run orszag --mpi 4        # optional: run on 4 MPI ranks (rank 0 writes the files)
    python cli.py show orszag-tang-vortex   # paths of the files of an earlier run
    python cli.py clean --all               # remove everything a run generated

Only the standard library is used, so ``list``, ``show`` and ``clean`` work without Struphy.
``metadata`` and ``run`` need Struphy (``pip install ./submodules/struphy``) and ``run`` also
its compiled kernels (``struphy compile``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "docs" / "src" / "examples"
OUTPUT_DIR = (
    ROOT / "docs" / "public" / "examples"
)  # scripts run here and write their files here
IMAGES_DIR = (
    ROOT / "docs" / "public" / "images" / "examples"
)  # thumbnails, copied there by the scripts
SCRATCH = (
    "struphy_gallery_runs",
    "struphy.log",
)  # simulation scratch output, never part of the site
DEV_SERVER = "http://localhost:4321"


# --------------------------------------------------------------------------- terminal output

_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _style(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(text: str) -> str:
    return _style("1", text)


def dim(text: str) -> str:
    return _style("2", text)


def green(text: str) -> str:
    return _style("32", text)


def red(text: str) -> str:
    return _style("31", text)


def yellow(text: str) -> str:
    return _style("33", text)


def human_size(path: Path) -> str:
    size = float(path.stat().st_size)
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return ""


def human_time(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:02.0f}s" if minutes else f"{secs:.1f}s"


# --------------------------------------------------------------------------- example lookup


def all_stems() -> list[str]:
    """The scripts of the gallery; files starting with an underscore are shared helpers."""
    return sorted(
        p.stem for p in SCRIPTS_DIR.glob("*.py") if not p.name.startswith("_")
    )


def metadata_of(stem: str) -> dict:
    path = OUTPUT_DIR / f"{stem}.metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def resolve(names: list[str], everything: bool) -> list[str]:
    """Expand ``--all`` and unique prefixes (``orszag`` -> ``orszag-tang-vortex``) into script stems."""
    known = all_stems()
    if everything:
        return known
    if not names:
        fail("Name at least one example, or pass --all. See `python cli.py list`.")
    stems: list[str] = []
    for name in names:
        name = name.removesuffix(".py").rsplit("/", 1)[-1]
        matches = [name] if name in known else [s for s in known if s.startswith(name)]
        if len(matches) != 1:
            hint = (
                f"Ambiguous, could be: {', '.join(matches)}"
                if matches
                else "No such example."
            )
            close = [s for s in known if name in s]
            if not matches and close:
                hint += f" Did you mean: {', '.join(close)}?"
            fail(f"{name!r}: {hint}\nRun `python cli.py list` for all examples.")
        if matches[0] not in stems:
            stems.append(matches[0])
    return stems


def fail(message: str) -> None:
    print(red("error: ") + message, file=sys.stderr)
    sys.exit(2)


@dataclass
class Artifacts:
    """The files an example's run produced, in the two places the site reads them from."""

    figures: list[Path] = field(
        default_factory=list
    )  # interactive .plotly.json/.html and static .png
    data: list[Path] = field(default_factory=list)  # metadata and profiling exports
    thumbnails: list[Path] = field(
        default_factory=list
    )  # copies of the PNGs for the gallery

    @property
    def has_interactive(self) -> bool:
        return any(p.name.endswith(".plotly.json") for p in self.figures)

    def all(self) -> list[Path]:
        return [*self.figures, *self.data, *self.thumbnails]


def collect(stem: str, since: float | None = None) -> Artifacts:
    """Find the generated files of an example, optionally only those written after ``since``."""
    prefixes = {stem}  # every generated file is named after its script

    def files(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        found = {
            p
            for prefix in prefixes
            for p in directory.glob(f"{prefix}*")
            if p.is_file() and p.name != "README.md"
        }
        return sorted(p for p in found if since is None or p.stat().st_mtime >= since)

    artifacts = Artifacts()
    for path in files(OUTPUT_DIR):
        if path.suffix in (".html", ".png") or path.name.endswith(".plotly.json"):
            artifacts.figures.append(path)
        else:
            artifacts.data.append(path)
    artifacts.thumbnails = [p for p in files(IMAGES_DIR) if p.suffix == ".png"]
    return artifacts


def print_artifacts(stem: str, artifacts: Artifacts, indent: str = "  ") -> None:
    groups = (
        (
            "Interactive figures",
            [p for p in artifacts.figures if p.name.endswith(".plotly.json")],
        ),
        (
            "Standalone HTML figures",
            [p for p in artifacts.figures if p.suffix == ".html"],
        ),
        ("Static images", [p for p in artifacts.figures if p.suffix == ".png"]),
        ("Metadata and profiling", artifacts.data),
        ("Gallery thumbnails", artifacts.thumbnails),
    )
    for title, paths in groups:
        if not paths:
            continue
        print(f"{indent}{bold(title)}")
        for path in paths:
            print(f"{indent}  {path}  {dim(human_size(path))}")
    page = f"{DEV_SERVER}/examples/{stem}/"
    print(f"{indent}{bold('Page')} (while `npm run dev` runs in docs/)")
    print(f"{indent}  {page}")


def open_example_page(stem: str) -> None:
    """Open the page that renders the generated Plotly JSON."""
    webbrowser.open(f"{DEV_SERVER}/examples/{stem}/")


# --------------------------------------------------------------------------- commands


def python_env_problem(need_kernels: bool) -> str | None:
    if importlib.util.find_spec("struphy") is None:
        return (
            "Struphy is not importable in this Python. From the repository root:\n"
            "  python -m venv .venv && source .venv/bin/activate\n"
            "  pip install ./submodules/struphy"
            + ("\n  struphy compile" if need_kernels else "")
        )
    for module in ("plotly", "kaleido"):
        if need_kernels and importlib.util.find_spec(module) is None:
            return f"The {module!r} package is missing (needed to write the figures): pip install {module}"
    return None


def cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for stem in all_stems():
        meta = metadata_of(stem)
        artifacts = collect(stem)
        rows.append(
            {
                "example": stem,
                "name": meta.get("name", ""),
                "model": meta.get("model", ""),
                "generated": artifacts.has_interactive,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    width = max(len(r["example"]) for r in rows)
    model_width = max(len(r["model"]) for r in rows)
    print(bold(f"{'example':<{width}}  {'model':<{model_width}}  run   title"))
    for r in rows:
        mark = green("yes  ") if r["generated"] else dim("no   ")
        print(
            f"{r['example']:<{width}}  {r['model']:<{model_width}}  {mark} {r['name']}"
        )
    done = sum(r["generated"] for r in rows)
    print(
        dim(
            f"\n{len(rows)} examples, {done} with generated figures in docs/public/examples/."
        )
    )
    print(
        dim("Run one with: python cli.py run <example>   (a unique prefix is enough)")
    )
    return 0


def run_command(command: list[str], cwd: Path, log: Path | None) -> int:
    """Run a command, streaming its output to the terminal or, with ``log``, into that file."""
    if log is None:
        return subprocess.call(command, cwd=cwd)
    with log.open("w") as handle:
        return subprocess.call(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT
        )


def cmd_metadata(args: argparse.Namespace) -> int:
    stems = resolve(args.examples, args.all)
    problem = python_env_problem(need_kernels=False)
    if problem:
        fail(problem)
    return subprocess.call(
        [sys.executable, str(ROOT / "generate_examples.py"), *stems], cwd=ROOT
    )


def clean_scratch() -> None:
    for name in SCRATCH:
        target = OUTPUT_DIR / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def run_one(
    stem: str, args: argparse.Namespace, *, postprocess_only: bool = False
) -> tuple[bool, float, Artifacts, str]:
    """Run or post-process one example; returns (ok, seconds, artifacts, message)."""
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_metadata:
        print(dim("  1/2 generating metadata"))
        code = subprocess.call(
            [sys.executable, str(ROOT / "generate_examples.py"), stem], cwd=ROOT
        )
        if code:
            return (
                False,
                time.time() - started,
                Artifacts(),
                "metadata generation failed",
            )

    action = "post-processing" if postprocess_only else "running"
    print(dim(f"  2/2 {action} docs/src/examples/{stem}.py in docs/public/examples/"))
    log = None
    if args.quiet:
        log = Path(os.environ.get("TMPDIR", "/tmp")) / f"struphy-gallery-{stem}.log"
    run_started = time.time() - 1  # file times can be coarser than the clock
    # Hardware threads count as slots, and a machine with fewer of them still runs.
    launcher = (
        [
            "mpirun",
            "--use-hwthread-cpus",
            "--oversubscribe",
            "--bind-to",
            "none",
            "-n",
            str(args.mpi),
        ]
        if args.mpi > 1
        else []
    )
    code = run_command(
        [
            *launcher,
            sys.executable,
            str(SCRIPTS_DIR / f"{stem}.py"),
            *(["--pproc"] if postprocess_only else []),
        ],
        OUTPUT_DIR,
        log,
    )
    if code:
        tail = ""
        if log is not None:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
            tail = f"\n{tail}\n(full output: {log})"
        tail += "\nIf the error mentions Chrome (PNG export by Kaleido), install it once with `plotly_get_chrome`."
        return (
            False,
            time.time() - started,
            collect(stem, run_started),
            f"script exited with code {code}{tail}",
        )

    # Preserve the actual domain used by this run before clearing the scratch
    # folder that contains Struphy's run_metadata.json.
    domain_code = subprocess.call(
        [
            sys.executable,
            str(ROOT / "generate_example_domain.py"),
            stem,
            "--metadata-root",
            str(OUTPUT_DIR / "struphy_gallery_runs"),
            "--output-dir",
            str(ROOT / "docs" / "public" / "example-domains"),
        ],
        cwd=ROOT,
    )
    if domain_code:
        return (
            False,
            time.time() - started,
            collect(stem, run_started),
            "domain export from run metadata failed",
        )

    if not getattr(args, "keep_scratch", False) and not postprocess_only:
        clean_scratch()

    artifacts = collect(stem, run_started)
    if not artifacts.has_interactive:
        return (
            False,
            time.time() - started,
            artifacts,
            "the script finished but wrote no interactive .plotly.json figure",
        )
    return True, time.time() - started, artifacts, ""


def cmd_run(args: argparse.Namespace) -> int:
    stems = resolve(args.examples, args.all)
    problem = python_env_problem(need_kernels=True)
    if problem:
        fail(problem)
    if args.mpi > 1 and shutil.which("mpirun") is None:
        fail("--mpi needs `mpirun` (OpenMPI or MPICH) and mpi4py in this Python.")
    if args.mpi > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per rank, as in CI
    results = []
    for number, stem in enumerate(stems, 1):
        print(bold(f"\n[{number}/{len(stems)}] {stem}"))
        ok, seconds, artifacts, message = run_one(stem, args)
        results.append((stem, ok, seconds))
        if ok:
            print(green(f"\n  {stem}: done in {human_time(seconds)}. Generated files:"))
            print_artifacts(stem, artifacts)
            if args.keep_scratch:
                print(
                    f"  {bold('Simulation data')} (kept)\n    {OUTPUT_DIR / SCRATCH[0]}"
                )
            if args.open:
                open_example_page(stem)
        else:
            print(red(f"\n  {stem}: FAILED after {human_time(seconds)}: {message}"))
            if artifacts.all():
                print_artifacts(stem, artifacts)
            if not args.keep_going:
                break
    if len(stems) > 1 or not all(ok for _, ok, _ in results):
        print(bold("\nSummary"))
        for stem, ok, seconds in results:
            print(
                f"  {green('ok    ') if ok else red('FAILED')} {stem:<{max(map(len, stems))}}  {human_time(seconds)}"
            )
        skipped = [s for s in stems if s not in {r[0] for r in results}]
        for stem in skipped:
            print(
                f"  {yellow('skipped')} {stem}  (stopped after a failure; use --keep-going to continue)"
            )
    print(
        dim(
            "\nEverything above is generated and git-ignored; CI rebuilds it. Remove it with: python cli.py clean"
        )
    )
    return 0 if all(ok for _, ok, _ in results) and len(results) == len(stems) else 1


def cmd_pproc(args: argparse.Namespace) -> int:
    """Post-process existing simulation output without running time integration."""
    stems = resolve(args.examples, args.all)
    problem = python_env_problem(need_kernels=False)
    if problem:
        fail(problem)
    if args.mpi > 1 and shutil.which("mpirun") is None:
        fail("--mpi needs `mpirun` (OpenMPI or MPICH) and mpi4py in this Python.")
    if args.mpi > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    results = []
    for number, stem in enumerate(stems, 1):
        print(bold(f"\n[{number}/{len(stems)}] {stem}"))
        ok, seconds, artifacts, message = run_one(stem, args, postprocess_only=True)
        results.append((stem, ok, seconds))
        if ok:
            print(
                green(
                    f"\n  {stem}: post-processing done in {human_time(seconds)}. Generated files:"
                )
            )
            print_artifacts(stem, artifacts)
            if args.open:
                open_example_page(stem)
        else:
            print(red(f"\n  {stem}: FAILED after {human_time(seconds)}: {message}"))
            if artifacts.all():
                print_artifacts(stem, artifacts)
            if not args.keep_going:
                break
    if len(stems) > 1 or not all(ok for _, ok, _ in results):
        print(bold("\nSummary"))
        for stem, ok, seconds in results:
            print(
                f"  {green('ok    ') if ok else red('FAILED')} {stem:<{max(map(len, stems))}}  {human_time(seconds)}"
            )
    return 0 if all(ok for _, ok, _ in results) and len(results) == len(stems) else 1


def cmd_show(args: argparse.Namespace) -> int:
    status = 0
    for stem in resolve(args.examples, args.all):
        artifacts = collect(stem)
        print(bold(stem))
        if not artifacts.all():
            print(yellow(f"  Nothing generated yet. Run: python cli.py run {stem}"))
            status = 1
            continue
        print_artifacts(stem, artifacts)
        if args.open:
            open_example_page(stem)
    return status


def cmd_clean(args: argparse.Namespace) -> int:
    """Delete generated files (all git-ignored); the metadata only with ``--metadata``, as the site build needs it."""
    stems = resolve(args.examples, args.all)
    doomed: list[Path] = []
    for stem in stems:
        artifacts = collect(stem)
        doomed += [
            p
            for p in artifacts.all()
            if args.metadata or not p.name.endswith(".metadata.json")
        ]
    doomed += [OUTPUT_DIR / name for name in SCRATCH if (OUTPUT_DIR / name).exists()]
    doomed = sorted(set(doomed))
    if not doomed:
        print("Nothing to clean.")
        return 0
    for path in doomed:
        print(f"{dim('would remove' if args.dry_run else 'removing')} {path}")
        if not args.dry_run:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return 0


# --------------------------------------------------------------------------- argument parsing


def add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "examples",
        nargs="*",
        metavar="EXAMPLE",
        help="script name(s); a unique prefix is enough",
    )
    parser.add_argument("-a", "--all", action="store_true", help="every example")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Run, inspect and clean the Struphy gallery examples.",
        epilog="Examples live in docs/src/examples/; see the README.md there for how to add one.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p = sub.add_parser("list", help="list the examples and whether their figures exist")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser(
        "metadata",
        help="regenerate the structural metadata JSON (needs Struphy, no kernels)",
    )
    add_selection(p)
    p.set_defaults(func=cmd_metadata)

    p = sub.add_parser(
        "run",
        help="metadata, run, thumbnails and clean-up for examples, as CI does (needs compiled Struphy)",
        description="Build the page assets of one or more examples and print the files that were generated.",
    )
    add_selection(p)
    p.add_argument(
        "--open",
        action="store_true",
        help=f"open each example page afterwards (expects the docs dev server at {DEV_SERVER})",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="hide the simulation output (shown only on failure)",
    )
    p.add_argument(
        "--keep-scratch",
        action="store_true",
        help="keep struphy_gallery_runs/ (raw HDF5 run data)",
    )
    p.add_argument(
        "-n",
        "--mpi",
        type=int,
        default=1,
        metavar="N",
        help="run on N MPI ranks (default 1)",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="do not regenerate the metadata first",
    )
    p.add_argument(
        "-k",
        "--keep-going",
        action="store_true",
        help="continue with the next example after a failure",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "pproc",
        help="post-process existing simulation output without running a simulation",
        description="Regenerate gallery figures from existing Struphy output using each example's --pproc path.",
    )
    add_selection(p)
    p.add_argument(
        "--open",
        action="store_true",
        help=f"open each example page afterwards (expects the docs dev server at {DEV_SERVER})",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="hide post-processing output (shown only on failure)",
    )
    p.add_argument(
        "-n",
        "--mpi",
        type=int,
        default=1,
        metavar="N",
        help="post-process on N MPI ranks (default 1)",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="do not regenerate the metadata first",
    )
    p.add_argument(
        "-k",
        "--keep-going",
        action="store_true",
        help="continue with the next example after a failure",
    )
    p.set_defaults(func=cmd_pproc)

    p = sub.add_parser(
        "show", help="print the paths of the files generated by earlier runs"
    )
    add_selection(p)
    p.add_argument(
        "--open",
        action="store_true",
        help=f"open each example page (expects the docs dev server at {DEV_SERVER})",
    )
    p.set_defaults(func=cmd_show)

    p = sub.add_parser(
        "clean",
        help="remove generated figures, thumbnails, profiling data and scratch output",
    )
    add_selection(p)
    p.add_argument(
        "--metadata",
        action="store_true",
        help="also remove the metadata JSON files (`npm run dev` needs them)",
    )
    p.add_argument(
        "-n", "--dry-run", action="store_true", help="only print what would be removed"
    )
    p.set_defaults(func=cmd_clean)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print(red("\ninterrupted"), file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
