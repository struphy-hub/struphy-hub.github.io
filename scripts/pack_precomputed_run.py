"""Pack the run of a precomputed gallery example for upload, and print its registry entry.

After simulating it (from the repository root, on a cluster):

    mpirun -n 32 python run_example.py itpa-tae-linear-mhd --simulate
    python scripts/pack_precomputed_run.py itpa-tae-linear-mhd

This writes `<stem>-run.tar.gz` of `docs/public/examples/struphy_gallery_runs/<sim_folder>/`
(without its `post_processing/`, which the website regenerates), and prints its sha256 and the
commands to upload it to a release and fill in `.github/precomputed-examples.json`. It uploads
nothing itself. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRECOMPUTED = ROOT / ".github" / "precomputed-examples.json"
RUNS_DIR = ROOT / "docs" / "public" / "examples" / "struphy_gallery_runs"
REPOSITORY = "struphy-hub/struphy-hub.github.io"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", help="the script stem, e.g. itpa-tae-linear-mhd")
    parser.add_argument("--tag", default="precomputed-examples-v1", help="the release to upload to")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="where to write the archive")
    args = parser.parse_args()

    entry = json.loads(PRECOMPUTED.read_text()).get(args.example)
    if entry is None:
        raise SystemExit(f"{args.example} is not listed in {PRECOMPUTED.relative_to(ROOT)}")
    run = RUNS_DIR / entry["sim_folder"]
    if not (run / "data").is_dir():
        raise SystemExit(f"no simulation output in {run}; run it first with run_example.py --simulate")

    archive = args.output / f"{args.example}-run.tar.gz"

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        return None if len(parts) > 1 and parts[1] == "post_processing" else info

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run, arcname=entry["sim_folder"], filter=keep)

    digest = hashlib.sha256()
    with archive.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    size = archive.stat().st_size
    url = f"https://github.com/{REPOSITORY}/releases/download/{args.tag}/{archive.name}"
    print(f"Wrote {archive} ({size / 1e6:.0f} MB)")
    if size > 2e9:
        print("warning: GitHub release assets are limited to 2 GB; save fewer fields or snapshots")
    print("\nUpload it (creates the release the first time):")
    print(f"  gh release view {args.tag} -R {REPOSITORY} || gh release create {args.tag} -R {REPOSITORY} "
          f"--title 'Precomputed example runs' --notes 'Archived simulation output of gallery examples too expensive for CI.'")
    print(f"  gh release upload {args.tag} {archive} -R {REPOSITORY} --clobber")
    print(f"\nThen set in {PRECOMPUTED.relative_to(ROOT)}:")
    print(json.dumps({args.example: {**entry, "url": url, "sha256": digest.hexdigest()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
