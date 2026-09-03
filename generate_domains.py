"""Export all available Struphy domains as VTK PolyData files."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from struphy import domains
from struphy.geometry.base import Domain


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "docs" / "public" / "domains"

def export_grid(domain, name, output_dir, plane, n1=16, n2=65):
    e1, e2 = __import__('numpy').linspace(0., 1., n1), __import__('numpy').linspace(0., 1., n2)
    np = __import__('numpy')
    # Take each 2D view through the center of the omitted logical coordinate,
    # matching a centered slice rather than a boundary face.
    coordinates = {'xy': (e1, e2, 0.5), 'xz': (e1, 0.5, e2), 'yz': (0.5, e1, e2)}[plane]
    xyz = domain(*coordinates, squeeze_out=True)
    data = {'name': name, 'plane': plane, 'x': np.asarray(xyz[0]).tolist(), 'y': np.asarray(xyz[1]).tolist(), 'z': np.asarray(xyz[2]).tolist()}
    (output_dir / f'{name}-{plane}.json').write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf-8')


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

    for name, domain_class in classes:
        output_file = output_dir / f"{name}.vtp"
        print(f"Exporting {name} -> {output_file}")

        try:
            domain_class().export_geometry(filename=str(output_file))
            for plane in ('xy', 'xz', 'yz'):
                export_grid(domain_class(), name, output_dir, plane)
        except (Exception, SystemExit) as error:
            failures.append((name, error))
            detail = str(error) or type(error).__name__
            print(f"  Skipped {name}: {detail}")

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
