"""Export the actual geometry used by one completed gallery example.

The simulation writes ``run_metadata.json`` before its time loop.  This
script reads that results record, reconstructs its domain, and emits the same
VTK PolyData and 2-D slice grids consumed by the Domains-page renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from struphy.geometry.base import Domain

from generate_domains import export_grid, json_safe


def export_example_domain(slug: str, metadata_root: Path, output_dir: Path) -> Path:
    matches = sorted(metadata_root.glob("**/run_metadata.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run_metadata.json below {metadata_root}, found {len(matches)}."
        )

    metadata_path = matches[0]
    results = json.loads(metadata_path.read_text(encoding="utf-8"))
    domain = Domain.from_dict(results["domain"])
    output_dir.mkdir(parents=True, exist_ok=True)

    domain.export_geometry(filename=str(output_dir / f"{slug}.vtp"))
    for plane in ("xy", "xz", "yz"):
        export_grid(domain, slug, output_dir, plane)

    (output_dir / f"{slug}.json").write_text(
        json.dumps(
            {
                "name": results["domain"]["type"],
                "parameters": json_safe(results["domain"].get("params", {})),
                "run_metadata": str(metadata_path),
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported {results['domain']['type']} for {slug} from {metadata_path}")
    return output_dir / f"{slug}.vtp"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Example script stem")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_example_domain(args.slug, args.metadata_root, args.output_dir)
