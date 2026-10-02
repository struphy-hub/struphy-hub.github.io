"""Slurm jobs of gallery examples, generated and watched with slurm-script-generator.

Shared by `.github/scripts/run-pitagora-example.py` (CI on the self-hosted Pitagora runner) and
`scripts/submit_precomputed_run.py` (the long runs whose archives CI downloads).
"""

from __future__ import annotations

import shlex
from pathlib import Path

from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue

# The module stack of Pitagora's GitHub runner.
MODULES = [
    "gcc/12.3.0",
    "python/3.11.7",
    "hdf5/1.14.3--gcc--12.3.0",
    "cmake/3.27.9",
    "netcdf-fortran/4.6.1--gcc--12.3.0",
    "netlib-scalapack/2.2.0--openmpi--4.1.6--gcc--12.3.0-ucx1.20",
]


def environment_commands(virtual_env: Path, mpi_ranks: int) -> list[str]:
    """Strict bash, the virtual environment, and Struphy's MPI switch for this many ranks."""
    commands = [
        "set -euo pipefail",
        f"source {shlex.quote(str(virtual_env / 'bin' / 'activate'))}",
    ]
    if mpi_ranks == 1:
        # A one-task Slurm allocation is not an MPI launch.
        commands.append("export STRUPHY_MPI=0")
    else:
        # mpi4py must see the Open MPI environment of the launch.
        commands.append("unset STRUPHY_MPI")
    return commands


def example_job(
    *,
    job_name: str,
    account: str,
    partition: str,
    workspace: Path,
    log_dir: Path,
    commands: list[str],
    mpi_ranks: int = 1,
    time: str = "00:30:00",
    mem: str | None = None,
    modules: list[str] | None = None,
) -> SlurmScript:
    """One single-node job in ``workspace``, logging to ``log_dir/slurm-<job_name>-<id>.out/.err``."""
    return SlurmScript(
        job_name=job_name,
        account=account,
        partition=partition,
        nodes=1,
        ntasks=mpi_ranks,
        cpus_per_task=1,
        time=time,
        mem=mem,
        chdir=str(workspace),
        output=str(log_dir / f"slurm-{job_name}-%j.out"),
        error=str(log_dir / f"slurm-{job_name}-%j.err"),
        modules=MODULES if modules is None else modules,
        custom_commands=commands,
    )


def tail_log(path: Path, lines: int = 200) -> None:
    """Print a bounded batch log in a collapsible GitHub Actions group."""
    if not path.is_file():
        return
    print(f"::group::{path.name}")
    print("\n".join(path.read_text(errors="replace").splitlines()[-lines:]))
    print("::endgroup::")


def submit_and_wait(
    script: SlurmScript, job_name: str, script_path: Path, log_dir: Path, poll_interval: float = 15
) -> str | None:
    """Submit the job, wait for it (raising unless it completed) and print its logs; return its state."""
    job_id = script.submit_job(path=str(script_path), verbose=True)
    print(f"Submitted {job_name} as Slurm job {job_id}")
    try:
        state = SQueue().wait_until_done(job_id=job_id, poll_interval=poll_interval, check=True)[job_id]
        print(f"Slurm job {job_id} finished with state {state or 'unknown'}.")
    finally:
        tail_log(log_dir / f"slurm-{job_name}-{job_id}.out")
        tail_log(log_dir / f"slurm-{job_name}-{job_id}.err")
    return state
