#!/usr/bin/env python3
"""Read and validate the gallery examples assigned to the Pitagora runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELECTION_FILE = ROOT / ".github" / "pitagora-examples.txt"
SCRIPTS_DIR = ROOT / "docs" / "src" / "examples"
OUTPUT_DIR = ROOT / "docs" / "public" / "examples"
ARTIFACT_DIRS = (
    OUTPUT_DIR,
    ROOT / "docs" / "public" / "images" / "examples",
    ROOT / "docs" / "public" / "example-domains",
)


def selected_examples() -> list[str]:
    if not SELECTION_FILE.is_file():
        raise ValueError(f"missing {SELECTION_FILE.relative_to(ROOT)}")
    examples = [
        line.partition("#")[0].strip()
        for line in SELECTION_FILE.read_text(encoding="utf-8").splitlines()
    ]
    examples = [example for example in examples if example]
    if not examples:
        raise ValueError(f"{SELECTION_FILE.relative_to(ROOT)} contains no examples")
    duplicates = sorted(
        {example for example in examples if examples.count(example) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate Pitagora examples: {', '.join(duplicates)}")
    missing = [
        example for example in examples if not (SCRIPTS_DIR / f"{example}.py").is_file()
    ]
    if missing:
        raise ValueError(f"no gallery script for: {', '.join(missing)}")
    return examples


def artifact_files(example: str) -> list[Path]:
    return sorted(
        path
        for directory in ARTIFACT_DIRS
        if directory.is_dir()
        for path in directory.glob(f"{example}*")
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="print the selection as a JSON array"
    )
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="require figure files for every selected example",
    )
    args = parser.parse_args()
    try:
        examples = selected_examples()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.verify_output:
        expected_outputs = [
            OUTPUT_DIR / f"{example}.{suffix}"
            for example in examples
            for suffix in ("plotly.json", "html")
        ]
        missing = [path.name for path in expected_outputs if not path.is_file()]
        if missing:
            print(
                f"error: missing Pitagora output: {', '.join(missing)}", file=sys.stderr
            )
            return 1
        print("Downloaded Pitagora artifacts:")
        for example in examples:
            print(f"  {example}")
            for path in artifact_files(example):
                print(f"    {path.relative_to(ROOT)}")
        return 0

    print(json.dumps(examples) if args.json else "\n".join(examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
