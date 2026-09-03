"""Export all available Struphy domains as VTK PolyData files."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from struphy import domains
from struphy.geometry.base import Domain

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "docs" / "public" / "domains"

TORUS_MAPPINGS = {
    "Tokamak",
    "GVECunit",
    "DESCunit",
    "IGAPolarTorus",
    "HollowTorus",
}


def json_safe(value):
    """Return a JSON-safe representation of a domain parameter value."""
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parameter_descriptions(cls: type[Domain]) -> dict[str, str]:
    """Extract NumPy-style parameter descriptions from a domain docstring."""
    doc = inspect.getdoc(cls) or ""
    match = re.search(
        r"(?:^|\n)Parameters\n-+\n(?P<body>.*?)(?=\n(?:Note|Notes|Returns|Attributes|Examples|References|Raises|See Also)\n-+|\Z)",
        doc,
        re.S,
    )
    if not match:
        return {}

    descriptions: dict[str, str] = {}
    current: list[str] = []
    for line in match.group("body").splitlines():
        # NumPy parameter headers are unindented after ``inspect.getdoc``;
        # requiring that distinction avoids treating description text such as
        # ``Epsilon: inverse aspect ratio`` as another parameter.
        header = re.match(r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*.+$", line)
        if header:
            current = [
                name.strip()
                for name in header.group(1).split(",")
                if name.strip() != "self"
            ]
            for name in current:
                descriptions[name] = ""
        elif current and line.strip():
            text = line.strip()
            for name in current:
                descriptions[name] = f"{descriptions[name]} {text}".strip()
    return descriptions


def _grid(
    xyz,
    first: int,
    second: int,
    *,
    highlight_rows=(),
    markers=(),
    omit_last_column=False,
):
    """Serialize one mapped logical grid for the browser renderer."""
    return {
        "a": np.asarray(xyz[first]).tolist(),
        "b": np.asarray(xyz[second]).tolist(),
        "highlight_rows": list(highlight_rows),
        "markers": [list(marker) for marker in markers],
        "omit_last_column": omit_last_column,
    }


def export_grid(domain, name, output_dir, plane, n1=16, n2=65):
    """Export the same grid constructions used by ``Domain.show()``.

    For toroidal mappings, x-z is the ``eta3 = 0`` cross-section and x-y is
    the two-branch top view (``eta2 = 0`` and ``eta2 = .5``).  These are the
    two views drawn by Struphy itself; treating them as generic centred
    Cartesian slices makes HollowTorus degenerate to lines.
    """
    e1 = np.linspace(0.0, 1.0, n1)
    e2 = np.linspace(0.0, 1.0, n2)
    is_not_cube = domain.kind_map < 10 or domain.kind_map > 19
    is_torus = name in TORUS_MAPPINGS

    if plane == "xy" and is_torus:
        # Domain.show() top view, including both sides for non-cube mappings.
        branches = [domain(e1, 0.0, e2, squeeze_out=True)]
        if is_not_cube:
            branches.append(domain(e1, 0.5, e2, squeeze_out=True))
        grids = [_grid(branch, 0, 1, highlight_rows=(0,)) for branch in branches]
    elif plane == "xz" and is_torus:
        # Domain.show() first panel: physical poloidal section at eta3 = 0.
        section = domain(e1, e2, 0.0, squeeze_out=True)
        marker_columns = (0, n2 // 2) if is_not_cube else (0,)
        grids = [
            _grid(
                section,
                0,
                2,
                markers=((0, j) for j in marker_columns),
                omit_last_column=is_not_cube,
            )
        ]
    elif plane == "xy":
        # Domain.show() first panel for non-toroidal mappings.
        section = domain(e1, e2, 0.0, squeeze_out=True)
        marker_columns = (0, n2 // 2) if is_not_cube else (0,)
        grids = [
            _grid(
                section,
                0,
                1,
                markers=((0, j) for j in marker_columns),
                omit_last_column=is_not_cube,
            )
        ]
    elif plane == "xz":
        # Domain.show() top view for non-toroidal mappings.
        branches = [domain(e1, 0.0, e2, squeeze_out=True)]
        if is_not_cube:
            branches.append(domain(e1, 0.5, e2, squeeze_out=True))
        grids = [_grid(branch, 0, 2, highlight_rows=(0,)) for branch in branches]
    elif plane == "yz" and is_torus:
        # The y-z counterpart of the top view. Domain.show() has no third
        # panel, so use the two quarter-turn branches rather than a projection
        # that collapses either logical coordinate.
        branches = [domain(e1, 0.25, e2, squeeze_out=True)]
        if is_not_cube:
            branches.append(domain(e1, 0.75, e2, squeeze_out=True))
        grids = [_grid(branch, 1, 2, highlight_rows=(0,)) for branch in branches]
    elif plane == "yz":
        section = domain(0.5, e1, e2, squeeze_out=True)
        grids = [_grid(section, 1, 2)]
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    data = {
        "name": name,
        "plane": plane,
        "axes": list(plane),
        "grids": grids,
    }
    output_file = output_dir / f"{name}-{plane}.json"
    output_file.write_text(
        json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def domain_classes() -> list[tuple[str, type[Domain]]]:
    """Return concrete domains defined by Struphy's public domains module."""
    return sorted(
        (
            (name, cls)
            for name, cls in vars(domains).items()
            if isinstance(cls, type)
            and issubclass(cls, Domain)
            and cls is not Domain
            and cls.__module__ == domains.__name__
            and not inspect.isabstract(cls)
        ),
        key=lambda item: item[0].lower(),
    )


def export_domains(output_dir: Path, *, strict: bool = False) -> int:
    """Instantiate domains with their defaults and export their geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = domain_classes()
    failures: list[tuple[str, BaseException]] = []
    catalogue: dict[str, dict[str, object]] = {}

    for name, domain_class in classes:
        output_file = output_dir / f"{name}.vtp"
        print(f"Exporting {name} -> {output_file}")

        try:
            domain = domain_class()
            domain.export_geometry(filename=str(output_file))
            for plane in ("xy", "xz", "yz"):
                export_grid(domain, name, output_dir, plane)
            catalogue[name] = {
                "parameters": json_safe(getattr(domain, "params", {})),
                "parameter_descriptions": parameter_descriptions(domain_class),
            }
        except (Exception, SystemExit) as error:
            failures.append((name, error))
            detail = str(error) or type(error).__name__
            print(f"  Skipped {name}: {detail}")

    catalogue_file = output_dir / "catalogue.json"
    catalogue_file.write_text(
        json.dumps({"domains": catalogue}, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    generated = len(classes) - len(failures)
    print(f"\nGenerated {generated} domain file(s) in {output_dir}.")

    if failures:
        names = ", ".join(name for name, _ in failures)
        print(f"Skipped {len(failures)} unavailable domain(s): {names}.")
        return 1 if strict else 0

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code if any optional domain cannot be exported",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(export_domains(args.output_dir, strict=args.strict))
