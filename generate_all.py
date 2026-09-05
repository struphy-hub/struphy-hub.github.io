"""Run every catalogue generator (models, domains, equilibria, perturbations) in one go.

Equivalent to running each of these from the repo root, in order:

    python generate_models.py
    python generate_domains.py docs/public/domains
    python generate_equilibrium_slices.py
    python generate_perturbations.py

See .github/workflows/deploy.yml for the CI pipeline this mirrors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent

STEPS = [
    ["generate_models.py"],
    ["generate_domains.py", "docs/public/domains"],
    ["generate_equilibrium_slices.py"],
    ["generate_perturbations.py"],
]


def main() -> None:
    for step in STEPS:
        script = step[0]
        print(f"\n=== Running {' '.join(step)} ===")
        result = subprocess.run([sys.executable, *step], cwd=REPO_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"{script} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
