"""Generate page metadata for the Struphy example gallery.

Each script in `docs/src/examples/` builds its `Simulation` at module scope
and guards the actual run (and plotting) behind `if __name__ == "__main__":`.
That lets this script import each example without running its simulation,
read the resulting `sim` object, and write out the name, description,
governing equations, and configuration summary the gallery page needs --
so the page never duplicates values the script already knows.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent / "docs" / "src" / "examples"
OUTPUT_DIR = Path(__file__).parent / "docs" / "public" / "examples"

VISUALIZATION_LIBRARIES = ("plotly", "matplotlib", "bokeh", "altair")


def integrator_of(model) -> str | None:
    """Return the first propagator's integration algorithm, if any declares one."""
    for propagator in vars(model.propagators).values():
        algo = getattr(getattr(propagator, "options", None), "algo", None)
        if algo:
            return str(algo).capitalize()
    return None


def visualization_of(namespace: dict) -> str | None:
    """Detect which plotting library a script imported, from its module namespace."""
    for value in namespace.values():
        module_name = getattr(value, "__name__", "")
        library = module_name.split(".")[0]
        if library in VISUALIZATION_LIBRARIES:
            return library.capitalize()
    return None


def describe_domain(domain) -> str:
    params = {k: v for k, v in getattr(domain, "params", {}).items() if v not in (0.0, 1.0)}
    if not params:
        return type(domain).__name__
    joined = ", ".join(f"{key}={value:g}" for key, value in params.items())
    return f"{type(domain).__name__} ({joined})"


def build_metadata(sim, namespace: dict) -> dict:
    model = sim.model
    metadata = {
        "name": sim.name,
        "description": sim.description,
        "model": type(model).name(),
        "equationsMarkdown": type(model).pde_markdown(),
        "domain": describe_domain(sim.domain),
        "grid": " × ".join(str(n) for n in sim.grid.num_elements),
        "degree": " × ".join(str(p) for p in sim.derham_opts.degree),
        "steps": round(sim.time_opts.Tend / sim.time_opts.dt),
    }
    integrator = integrator_of(model)
    if integrator:
        metadata["integrator"] = integrator
    visualization = visualization_of(namespace)
    if visualization:
        metadata["visualization"] = visualization
    return metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for script in sorted(EXAMPLES_DIR.glob("*.py")):
        # `run_name` deliberately isn't "__main__", so each script's own
        # `if __name__ == "__main__":` block (the run + plotting) stays skipped.
        namespace = runpy.run_path(str(script), run_name=script.stem)
        sim = namespace.get("sim")
        if sim is None:
            print(f"Skipping {script.name}: no module-level `sim` found")
            continue

        output_path = OUTPUT_DIR / f"{script.stem}.metadata.json"
        existing = json.loads(output_path.read_text()) if output_path.exists() else {}
        metadata = {**existing, **build_metadata(sim, namespace)}
        output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
