"""Generate the data behind the /simulation/ page: the anatomy of a Struphy run.

A Struphy simulation is one `Simulation` object, and its constructor arguments
are the pieces a run is assembled from -- the model, where it lives (domain,
equilibrium, grid), how it is discretized (derham options), how it steps
(time options), and where it writes (environment, profiling). Most of those
pieces have a catalogue on this site already, so the page doubles as a map of
the rest of it.

Both halves are read from struphy rather than restated:

* the constructor's parameters, their types, defaults and docstring lines;
* for the parameters that take an options dataclass, the fields that bundle
  carries, so the page can show what is actually configurable in each;
* the phase names `Simulation.run` wraps its setup in, which are the same
  region names the profiling charts on the example pages show.

Which site page documents a given argument is site knowledge, not something
struphy knows, so that one mapping is declared here.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path

from struphy.io.options import DerhamOptions, EnvironmentOptions, ProfilingOptions, Time
from struphy.simulation.sim import Simulation

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "simulation-anatomy.json"

# Where each ingredient is catalogued on this site.
CATALOGUE_LINKS = {
    "model": {"href": "/models/", "label": "Models"},
    "domain": {"href": "/domains/", "label": "Domains"},
    "equil": {"href": "/equilibria/", "label": "Equilibria"},
    "time_opts": {"href": "/time-integration/", "label": "Time integration"},
    "derham_opts": {"href": "/feec/", "label": "FEEC basics"},
    "grid": {"href": "/feec/", "label": "FEEC basics"},
}

# The order the page presents the arguments in: what the simulation solves,
# where it lives, how it is discretized, how it steps, and how it runs. The
# constructor's own order is close to this but mixes identity in with physics.
GROUPS = [
    {
        "name": "What is solved",
        "blurb": "The physics: a model, which brings its own species, variables and propagators.",
        "parameters": ["model"],
    },
    {
        "name": "Where it lives",
        "blurb": "The mapped geometry, the background it is initialized from, and the element grid covering it.",
        "parameters": ["domain", "equil", "grid"],
    },
    {
        "name": "How it is discretized",
        "blurb": "Spline degrees and boundary conditions for the de Rham complex the fields live in.",
        "parameters": ["derham_opts"],
    },
    {
        "name": "How it steps",
        "blurb": "Step size, end time, and how the model's propagators are composed over a step.",
        "parameters": ["time_opts"],
    },
    {
        "name": "How it runs",
        "blurb": "Output folders, restarts, MPI, and what the profiler records — everything around the physics.",
        "parameters": ["env", "profiling_opts", "comm", "logging_level"],
    },
    {
        "name": "How it is labelled",
        "blurb": "Carried into the run's metadata, so a saved run says what it was.",
        "parameters": ["name", "description", "params_path"],
    },
]

# ProfilingOptions carries ~30 fields covering every backend scope-profiler
# supports; listing them all would bury the page. The rest are shown in full.
FIELD_LIMIT = 12


def docstring_parameters(cls: type) -> dict[str, str]:
    """Map parameter name -> its description in a numpydoc ``Parameters`` block."""
    docstring = inspect.getdoc(cls) or ""
    match = re.search(r"\nParameters\n-+[ \t]*\n([\s\S]*?)(?:\n\S.*\n-+[ \t]*\n|$)", docstring)
    if not match:
        return {}

    descriptions: dict[str, str] = {}
    names: list[str] = []
    lines: list[str] = []

    def flush() -> None:
        text = " ".join(line.strip() for line in lines if line.strip())
        for name in names:
            descriptions[name] = text

    for line in match.group(1).split("\n"):
        header = re.match(r"([\w, ]+?)\s*:\s*\S", line)
        if header and not line.startswith((" ", "\t")):
            flush()
            names = [part.strip() for part in header.group(1).split(",") if part.strip()]
            lines = []
        elif names:
            lines.append(line)
    flush()

    return descriptions


def annotation_sources() -> dict[str, str]:
    """Each parameter's annotation as written in the signature.

    The resolved objects are not always what the code says: without mpi4py
    installed, ``comm: MPI.Intracomm`` resolves through a stub and reports its
    type as ``dummy``. Reading the source keeps the page showing the annotation
    a reader would find in ``sim.py``.
    """
    source = inspect.getsource(Simulation.__init__)
    signature = source[source.index("(") : source.index("):")]
    return dict(re.findall(r"^\s*(\w+)\s*:\s*([^=\n]+?)\s*(?:=[^\n]*)?,?$", signature, re.M))


def type_name(annotation, declared: str = "") -> str:
    if declared:
        return declared
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def default_repr(parameter: inspect.Parameter) -> str | None:
    """How a default reads on the page.

    Most defaults are an empty options dataclass -- ``Time()``, ``Cuboid()`` --
    whose repr is the whole configuration; the class name alone is what the
    reader needs, with the fields listed separately.
    """
    if parameter.default is inspect.Parameter.empty:
        return None
    value = parameter.default
    if dataclasses.is_dataclass(value) or type(value).__module__.startswith("struphy"):
        return f"{type(value).__name__}()"
    return repr(value)


def option_fields(cls: type) -> list[dict] | None:
    """The configurable fields of an options dataclass, with their defaults."""
    if not dataclasses.is_dataclass(cls):
        return None
    descriptions = docstring_parameters(cls)
    fields = []
    for field in dataclasses.fields(cls):
        default = field.default
        fields.append(
            {
                "name": field.name,
                "default": None if default is dataclasses.MISSING else str(default),
                "description": descriptions.get(field.name, ""),
            }
        )
    return fields


SETUP_REGION = "setup: total"


def run_phases() -> list[str]:
    """The setup phases `run()` instruments, in source order.

    These are the same region names the example pages' profiling charts plot,
    so the page can name them and mean exactly what those charts show.
    `setup: total` is the region wrapping all of them rather than one of them,
    so it is reported separately instead of as a sibling step.
    """
    source = inspect.getsource(Simulation.run)
    regions = dict.fromkeys(re.findall(r'profile_region\(\s*"(setup: [^"]+)"', source))
    if SETUP_REGION not in regions:
        raise SystemExit(f"run() no longer wraps setup in a {SETUP_REGION!r} region")
    return [region for region in regions if region != SETUP_REGION]


def generate(output_file: Path = OUTPUT_FILE) -> None:
    signature = inspect.signature(Simulation.__init__)
    descriptions = docstring_parameters(Simulation)
    declared = annotation_sources()

    option_classes = {
        "time_opts": Time,
        "derham_opts": DerhamOptions,
        "env": EnvironmentOptions,
        "profiling_opts": ProfilingOptions,
        "grid": type(signature.parameters["grid"].default),
    }

    parameters = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        fields = option_fields(option_classes.get(name)) if name in option_classes else None
        parameters[name] = {
            "name": name,
            "type": type_name(parameter.annotation, declared.get(name, "")),
            "default": default_repr(parameter),
            "required": parameter.default is inspect.Parameter.empty,
            "description": descriptions.get(name, ""),
            "link": CATALOGUE_LINKS.get(name),
            "fields": None if fields is None else fields[:FIELD_LIMIT],
            "fieldsTruncated": bool(fields) and len(fields) > FIELD_LIMIT,
            "fieldCount": len(fields) if fields else 0,
        }

    missing = set(parameters) - {name for group in GROUPS for name in group["parameters"]}
    if missing:
        raise SystemExit(f"Simulation gained parameters this page does not place: {sorted(missing)}")

    data = {
        "groups": [
            {**group, "parameters": [parameters[name] for name in group["parameters"] if name in parameters]}
            for group in GROUPS
        ],
        "runPhases": run_phases(),
        "setupRegion": SETUP_REGION,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"Generated {len(parameters)} Simulation arguments in {len(data['groups'])} groups "
        f"and {len(data['runPhases'])} run phases in {output_file}"
    )


if __name__ == "__main__":
    generate()
