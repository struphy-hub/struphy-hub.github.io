#!/usr/bin/env python3
"""Generate, submit, and wait for one serial Pitagora gallery example."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from slurm_jobs import environment_commands, example_job, submit_and_wait  # noqa: E402

# Orszag--Tang writes 401 field snapshots and post-processing loads them all
# before evaluating the output grid. It exceeds the debug partition's default
# memory allocation; the small gallery examples do not need this request.
MEMORY_BY_EXAMPLE = {"orszag-tang-vortex": "64GB"}
MPI_RANKS_BY_EXAMPLE = {"orszag-tang-vortex": 4}


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
    mpi_ranks = MPI_RANKS_BY_EXAMPLE.get(args.example, 1)
    commands = environment_commands(virtual_env, mpi_ranks)
    commands.append(f"python cli.py run {shlex.quote(args.example)} --mpi {mpi_ranks}")

    script = example_job(
        job_name=job_name,
        account=args.account,
        partition=args.partition,
        workspace=workspace,
        log_dir=workspace,
        commands=commands,
        mpi_ranks=mpi_ranks,
        mem=MEMORY_BY_EXAMPLE.get(args.example),
    )
    submit_and_wait(script, job_name, runner_temp / f"{job_name}.sbatch", workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
