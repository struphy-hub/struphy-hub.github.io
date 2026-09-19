#!/usr/bin/env python
"""Check that the generated example files are what the site build expects, before it is built.

The site reads one `<stem>.metadata.json` per example script, and links a download route and, once the
example has been run, its figures. A problem in any of them shows up only late in `npm run build`, or,
worse, after the examples have been run for half an hour. This script checks them in a second:

    python scripts/check_examples.py                   # the metadata generated without running anything
    python scripts/check_examples.py --require-figures # also the figures of a finished run

It needs only the standard library. Exit status 1 if anything is wrong, with every problem listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "docs" / "src" / "examples"
OUTPUT_DIR = ROOT / "docs" / "public" / "examples"
ROUTES_DIR = ROOT / "docs" / "src" / "pages" / "examples"

# What generate_examples.py writes for every example, and the example pages read without a guard.
REQUIRED = ("name", "description", "model", "equationsMarkdown", "domain", "steps")


def _reject_constant(name: str):
    # Python writes NaN and Infinity, but they are not JSON, and the site's JSON.parse refuses them.
    raise ValueError(f"{name} is not valid JSON")


def check_metadata(stem: str) -> list[str]:
    path = OUTPUT_DIR / f"{stem}.metadata.json"
    if not path.is_file():
        return [
            f"{stem}: no {path.relative_to(ROOT)} (run `python generate_examples.py`)"
        ]
    try:
        data = json.loads(path.read_text(), parse_constant=_reject_constant)
    except ValueError as error:
        return [f"{stem}: {path.name} is not valid JSON: {error}"]
    problems = [
        f"{stem}: {path.name} lacks `{key}`"
        for key in REQUIRED
        if data.get(key) in (None, "")
    ]
    if not isinstance(data.get("steps"), int) or data.get("steps", 0) <= 0:
        problems.append(
            f"{stem}: `steps` in {path.name} is not a positive integer: {data.get('steps')!r}"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--require-figures",
        action="store_true",
        help="also require <stem>.html of every example",
    )
    args = parser.parse_args()

    stems = sorted(
        p.stem for p in SCRIPTS_DIR.glob("*.py") if not p.name.startswith("_")
    )
    problems: list[str] = []
    if not stems:
        problems.append(f"no example scripts found in {SCRIPTS_DIR.relative_to(ROOT)}")
    for stem in stems:
        problems += check_metadata(stem)
        route = ROUTES_DIR / f"{stem}.py.ts"
        if not route.is_file():
            problems.append(
                f"{stem}: no download route {route.relative_to(ROOT)} (its 'Download .py' link would be a 404)"
            )
        if args.require_figures and not (OUTPUT_DIR / f"{stem}.html").is_file():
            problems.append(
                f"{stem}: no {stem}.html in {OUTPUT_DIR.relative_to(ROOT)}; the example did not produce its figure"
            )
    # Metadata of a script that no longer exists would still get a page.
    for path in sorted(OUTPUT_DIR.glob("*.metadata.json")):
        if path.name.removesuffix(".metadata.json") not in stems:
            problems.append(
                f"{path.name}: no example script docs/src/examples/{path.name.removesuffix('.metadata.json')}.py"
            )

    if problems:
        print(f"{len(problems)} problem(s) in the example files:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"OK: {len(stems)} examples"
        + (", with figures" if args.require_figures else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
