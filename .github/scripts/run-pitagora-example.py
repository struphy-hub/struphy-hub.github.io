#!/usr/bin/env python3
"""Generate, submit, and wait for one serial Pitagora gallery example."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue


MODULES = [
    "gcc/12.3.0",
    "python/3.11.7",
    "hdf5/1.14.3--gcc--12.3.0",
    "cmake/3.27.9",
    "netcdf-fortran/4.6.1--gcc--12.3.0",
    "netlib-scalapack/2.2.0--openmpi--4.1.6--gcc--12.3.0-ucx1.20",
]


def tail_log(path: Path, lines: int = 200) -> None:
    """Print a bounded batch log in a collapsible GitHub Actions group."""
    if not path.is_file():
        return
    print(f"::group::{path.name}")
    print("\n".join(path.read_text(errors="replace").splitlines()[-lines:]))
    print("::endgroup::")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", help="Gallery example stem")
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    args = parser.parse_args()

    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    virtual_env = Path(os.environ["VIRTUAL_ENV"]).resolve()
    run_id = os.environ["GITHUB_RUN_ID"]
    run_attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    job_name = f"struphy-{args.example}-{run_id}-{run_attempt}"
    script_path = runner_temp / f"{job_name}.sbatch"
    stdout = workspace / f"slurm-{job_name}-%j.out"
    stderr = workspace / f"slurm-{job_name}-%j.err"

    script = SlurmScript(
        job_name=job_name,
        account=args.account,
        partition=args.partition,
        nodes=1,
        ntasks=1,
        cpus_per_task=1,
        time="110",
        chdir=str(workspace),
        output=str(stdout),
        error=str(stderr),
        modules=MODULES,
        custom_commands=[
            "set -euo pipefail",
            f"source {shlex.quote(str(virtual_env / 'bin' / 'activate'))}",
            "export STRUPHY_MPI=0",
            f"python cli.py run {shlex.quote(args.example)}",
        ],
    )
    job_id = script.submit_job(path=str(script_path), verbose=True)
    print(f"Submitted {job_name} as Slurm job {job_id}")

    try:
        state = SQueue().wait_until_done(
            job_id=job_id, poll_interval=15, check=True
        )[job_id]
        print(f"Slurm job {job_id} finished with state {state or 'unknown'}.")
    finally:
        tail_log(workspace / f"slurm-{job_name}-{job_id}.out")
        tail_log(workspace / f"slurm-{job_name}-{job_id}.err")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
