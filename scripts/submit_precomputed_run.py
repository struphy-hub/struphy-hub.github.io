"""Simulate a precomputed gallery example as a Slurm job, and pack its run for upload.

The website's CI only post-processes these examples (see `.github/precomputed-examples.json`). This
makes their archived run on a cluster, with the same slurm-script-generator jobs as the Pitagora
CI (`scripts/slurm_jobs.py`): from the repository root, in the virtual environment with Struphy,

    python scripts/submit_precomputed_run.py itpa-tae-linear-mhd --account <account> --partition <partition>

The job generates the example's metadata, runs `run_example.py <stem> --simulate` on the registry
entry's `job.mpi_ranks` ranks (which also profiles it) and packs the run with
`scripts/pack_precomputed_run.py`, whose printed upload commands and registry entry end the job's
log. This waits for the job and prints its logs; `--no-wait` only submits it, and `--dry-run` only
writes the batch script. It uploads nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRECOMPUTED = ROOT / ".github" / "precomputed-examples.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slurm_jobs import MODULES, environment_commands, example_job, submit_and_wait  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", help="the script stem, e.g. itpa-tae-linear-mhd")
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--time", help="override the registry's job.time, e.g. 12:00:00")
    parser.add_argument("--mpi-ranks", type=int, help="override the registry's job.mpi_ranks")
    parser.add_argument(
        "--module", action="append", dest="modules",
        help=f"a module to load, instead of Pitagora's (repeat it); default: {' '.join(MODULES)}",
    )
    parser.add_argument(
        "--venv", type=Path, default=os.environ.get("VIRTUAL_ENV"),
        help="the virtual environment with Struphy (default: the active one)",
    )
    parser.add_argument("--no-wait", action="store_true", help="submit the job and return")
    parser.add_argument("--dry-run", action="store_true", help="only write the batch script")
    args = parser.parse_args()

    entry = json.loads(PRECOMPUTED.read_text()).get(args.example)
    if entry is None:
        raise SystemExit(f"{args.example} is not listed in {PRECOMPUTED.relative_to(ROOT)}")
    if args.venv is None:
        raise SystemExit("activate the virtual environment with Struphy, or pass --venv")
    job = entry.get("job", {})
    mpi_ranks = args.mpi_ranks or job.get("mpi_ranks", 1)
    stem = shlex.quote(args.example)

    commands = environment_commands(Path(args.venv).resolve(), mpi_ranks)
    commands += [
        "export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1",
        f"python generate_examples.py {stem}",
        # mpirun takes the ranks and hosts from the Slurm allocation.
        f"{f'mpirun -n {mpi_ranks} ' if mpi_ranks > 1 else ''}python run_example.py {stem} --simulate",
        f"python scripts/pack_precomputed_run.py {stem} --output {shlex.quote(str(ROOT))}",
    ]
    job_name = f"struphy-precomputed-{args.example}-{time.strftime('%Y%m%d-%H%M%S')}"
    script = example_job(
        job_name=job_name,
        account=args.account,
        partition=args.partition,
        workspace=ROOT,
        log_dir=ROOT,
        commands=commands,
        mpi_ranks=mpi_ranks,
        time=args.time or job.get("time", "01:00:00"),
        mem=job.get("mem"),
        modules=args.modules,
    )
    script_path = ROOT / f"{job_name}.sbatch"
    if args.dry_run:
        script.save(script_path)
        print(f"Wrote {script_path.relative_to(ROOT)}; submit it with sbatch.")
        return 0
    if args.no_wait:
        job_id = script.submit_job(path=str(script_path), verbose=True)
        print(f"Submitted {job_name} as Slurm job {job_id}; its log is slurm-{job_name}-{job_id}.out")
        return 0
    submit_and_wait(script, job_name, script_path, ROOT, poll_interval=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
