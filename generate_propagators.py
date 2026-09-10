"""Generate the static propagator catalogue used by the Struphy website.

A propagator is the unit Struphy splits a model into: each one advances a
subset of the model's variables over a time step, and models are assembled by
composing them. `model-graph.json` already links two models when they share a
propagator class -- this generator builds the other direction of that index,
so a propagator page can list the models that use it and where in the
splitting sequence it sits.

Three things are read straight off the class, so the catalogue can't drift
from the code:

* the class docstring -- the equations the propagator discretizes;
* its inner ``Variables`` class -- which variables it advances, and the
  variable kinds / discrete spaces each one accepts. These are declared as
  ``assert`` statements in the property setters, so they are recovered by
  parsing the source rather than by instantiating anything;
* its inner ``Options`` dataclass -- the configurable knobs, their choices
  (from ``Literal`` annotations) and defaults, described by the ``Parameters``
  section of the ``Options`` docstring.

``AdiabaticPhi`` and ``FaradayExtended`` still inherit the abstract templates
from ``Propagator`` instead of declaring their own; they are emitted with
empty variable/option lists rather than with the base class's ``var1`` /
``opt1`` placeholders.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
import textwrap
import typing
from pathlib import Path

import struphy.propagators as propagator_module
from catalogue_docs import extract_math
from struphy.models.utils import get_models
from struphy.propagators.base import Propagator
from struphy.utils.docstring_converter import rst_to_html

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "propagators.json"
CATEGORIES = ("Fluid", "Kinetic", "Hybrid", "Toy")

FEEC_KINDS = {"FEECVariable"}
PARTICLE_KINDS = {"PICVariable", "SPHVariable"}

DOCS_BASE = "https://struphy-hub.github.io/struphy/sections/subsections/propagators.html"


def collect_propagator_classes() -> list[tuple[str, type]]:
    """Every concrete Propagator subclass exported by ``struphy.propagators``."""
    classes = {
        value.__name__: value
        for value in vars(propagator_module).values()
        if inspect.isclass(value)
        and issubclass(value, Propagator)
        and value is not Propagator
    }
    return sorted(classes.items())


def declared_variables(propagator: type) -> list[dict]:
    """Recover the variables a propagator advances from its ``Variables`` class.

    Each variable is a property whose setter asserts what it will accept::

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable | SPHVariable)
            assert new.space in ("Particles6D", "DeltaFParticles6D")

    Those two asserts are the authoritative declaration of the variable's kind
    and admissible discrete spaces, so they are read out of the syntax tree.
    Instantiating the class instead would only yield ``None`` placeholders.
    """
    variables_cls = getattr(propagator, "Variables", None)
    if variables_cls is None or variables_cls is Propagator.Variables:
        return []

    module = ast.parse(textwrap.dedent(inspect.getsource(variables_cls)))
    variables = []
    for node in module.body[0].body:
        is_setter = isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "setter"
            for decorator in node.decorator_list
        )
        if not is_setter:
            continue

        kinds: list[str] = []
        spaces: list[str] = []
        for statement in node.body:
            if not isinstance(statement, ast.Assert):
                continue
            test = statement.test
            if isinstance(test, ast.Call) and getattr(test.func, "id", "") == "isinstance":
                # `isinstance(new, PICVariable | SPHVariable)` -- the union's members.
                kinds = [
                    name.id
                    for name in ast.walk(test.args[1])
                    if isinstance(name, ast.Name)
                ]
            elif isinstance(test, ast.Compare) and getattr(test.left, "attr", "") == "space":
                # `new.space == "Hcurl"` or `new.space in ("Hcurl", "Hdiv")`.
                spaces = [
                    constant.value
                    for constant in ast.walk(test.comparators[0])
                    if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
                ]

        variables.append({"name": node.name, "kinds": kinds, "spaces": spaces})

    return variables


def class_description(propagator: type) -> tuple[str, list[dict]]:
    """The class docstring -- the equations the propagator discretizes -- as HTML.

    Half of the propagator docstrings open with a ``:ref:`FEEC <gempic>``` link
    into the Sphinx documentation, which ``rst_to_html`` passes through as raw
    markup (it handles ``:class:`` and ``:math:``, not ``:ref:``). The site has
    no Sphinx inventory to resolve those targets against, so they are flattened
    to their link text before conversion.
    """
    docstring = inspect.getdoc(propagator) or ""
    # A few classes still carry a `Parameters` section describing constructor
    # arguments from before variables and options moved onto the inner classes.
    # It is stale, and what it documents is covered by the tables on the page.
    # (the underline is allowed trailing whitespace -- one docstring has it)
    docstring = re.split(r"\nParameters\n-+[ \t]*\n", docstring, maxsplit=1)[0]
    # `:ref:`FEEC <gempic>`` -> "FEEC"; `:ref:`time_discret`` -> "time_discret".
    docstring = re.sub(r":ref:`([^`<]+?)\s*(?:<[^`>]*>)?`", r"\1", docstring)
    if not docstring.strip():
        return "", []
    stripped, math_items = extract_math(docstring)
    return rst_to_html(stripped, forced_heading_level=3).strip(), math_items


def inline_rst_to_html(text: str) -> tuple[str, list[dict]]:
    """Render a one-paragraph RST fragment (an option description) as inline HTML.

    The full ``rst_to_html`` converter works on whole docstrings and emits
    block elements, which is the wrong shape for a table cell -- and these
    fragments only ever use three bits of markup: inline math, cross-reference
    roles, and ``literals``. Math is pulled out into the same placeholder-token
    form the rest of the catalogue uses, so the page renders it with KaTeX.
    """
    math_items: list[dict] = []

    def save_inline(match: re.Match) -> str:
        token = f"@@MATH{len(math_items)}@@"
        math_items.append({"token": token, "latex": match.group(1), "display": False})
        return token

    # Before escaping, so LaTeX containing `<` or `&` reaches KaTeX intact.
    stripped = re.sub(r":math:`([^`]+)`", save_inline, text)
    stripped = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # `:class:`~struphy.a.B`` -- the site has no Sphinx inventory to link into,
    # so these become the bare class name.
    stripped = re.sub(
        r":\w+:`~?([^`]+)`",
        lambda match: f"<code>{match.group(1).rsplit('.', 1)[-1]}</code>",
        stripped,
    )
    stripped = re.sub(r"``([^`]+)``", r"<code>\1</code>", stripped)
    return stripped, math_items


def parameter_descriptions(cls: type) -> dict[str, str]:
    """Map parameter name -> prose from a numpydoc ``Parameters`` section.

    struphy has no numpydoc parser, and the full RST-to-HTML converter would
    turn the section into a definition list rather than something addressable
    per option, so the block is split by hand: a non-indented ``name : type``
    line opens an entry and the indented lines under it are its description.
    """
    docstring = inspect.getdoc(cls) or ""
    match = re.search(r"\nParameters\n-+\n([\s\S]*?)(?:\n\S.*\n-+\n|$)", docstring)
    if not match:
        return {}

    descriptions: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if current is None:
            return
        text = " ".join(line.strip() for line in lines if line.strip())
        # Options docstrings spell out each choice as a bullet list after the
        # summary sentence; the choices are already carried structurally by the
        # `Literal` annotation, so only the summary is kept here.
        text = re.split(r"\s*-\s+``", text, maxsplit=1)[0]
        descriptions[current] = text.strip()

    for line in match.group(1).split("\n"):
        header = re.match(r"(\w+)\s*:\s*\S", line)
        if header and not line.startswith((" ", "\t")):
            flush()
            current, lines = header.group(1), []
        elif current is not None:
            lines.append(line)
    flush()

    return descriptions


def declared_options(propagator: type) -> list[dict]:
    """The propagator's configurable options, with choices and defaults."""
    options_cls = getattr(propagator, "Options", None)
    if options_cls is None or options_cls is Propagator.Options:
        return []

    descriptions = parameter_descriptions(options_cls)
    options = []
    for field in dataclasses.fields(options_cls):
        annotation = field.type
        if isinstance(annotation, str):
            # `from __future__ import annotations` in the defining module leaves
            # the annotation as source text; the Literal choices are still
            # readable from it, which is all this needs.
            choices = re.findall(r"[\"']([^\"']+)[\"']", annotation) if "Literal" in annotation else []
            type_name = annotation
        elif typing.get_origin(annotation) is typing.Literal:
            choices = [str(choice) for choice in typing.get_args(annotation)]
            type_name = "Literal"
        else:
            choices = []
            type_name = getattr(annotation, "__name__", str(annotation))

        default = field.default
        if default is dataclasses.MISSING:
            default_repr = None
        elif isinstance(default, float):
            default_repr = f"{default:.6g}"
        else:
            default_repr = str(default)

        description_html, description_math = inline_rst_to_html(
            descriptions.get(field.name, "")
        )
        options.append(
            {
                "name": field.name,
                "type": type_name,
                "choices": choices,
                "default": default_repr,
                "descriptionHtml": description_html,
                "descriptionMath": description_math,
            }
        )

    return options


def categorize(variables: list[dict]) -> str:
    """Group a propagator by what it advances, not by which model uses it.

    Field / Particle / Coupling is the distinction that actually matters when
    reading a splitting scheme: whether a step touches the FEEC fields, the
    markers, or -- the interesting case in a hybrid model -- both at once.
    """
    kinds = {kind for variable in variables for kind in variable["kinds"]}
    has_field = bool(kinds & FEEC_KINDS)
    has_particles = bool(kinds & PARTICLE_KINDS)
    if has_field and has_particles:
        return "Coupling"
    if has_particles:
        return "Particle"
    if has_field:
        return "Field"
    return "Other"


def summarize(propagator: type, variables: list[dict]) -> str:
    """A one-line, plain-text blurb for the index cards and the search index.

    Deliberately not taken from the docstring: almost every propagator opens
    with the same "FEEC discretization of the following equations: find ... such
    that" boilerplate followed by the equations themselves, which distinguishes
    nothing in a list of 44. What it advances does.
    """
    names = [variable["name"] for variable in variables]
    if not names:
        docstring = inspect.getdoc(propagator) or ""
        intro = docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ").strip()
        # These paragraphs run straight into a `.. math::` block, so they read
        # as sentence fragments once the equation is dropped.
        return intro if intro.endswith(".") else f"{intro} …"
    if len(names) == 1:
        return f"Advances {names[0]}."
    return f"Advances {', '.join(names[:-1])} and {names[-1]}."


def collect_usage() -> dict[str, list[dict]]:
    """Map propagator class name -> the models using it, with splitting position.

    ``model.propagators`` holds one attribute per step, in the order the model
    composes them, so the attribute name doubles as the step's name in the
    splitting scheme and its index as the step number.
    """
    usage: dict[str, list[dict]] = {}
    for category in CATEGORIES:
        for model_cls in get_models(category):
            model = model_cls()
            steps = vars(model.propagators)
            for index, (step, instance) in enumerate(steps.items()):
                usage.setdefault(type(instance).__name__, []).append(
                    {
                        "model": model_cls.__name__,
                        "type": category,
                        "step": step,
                        "index": index,
                        "steps": len(steps),
                    }
                )
    return usage


def generate(output_file: Path = OUTPUT_FILE) -> None:
    usage = collect_usage()

    catalogue = []
    for name, propagator in collect_propagator_classes():
        variables = declared_variables(propagator)
        description_html, description_math = class_description(propagator)
        catalogue.append(
            {
                "className": name,
                "module": propagator.__module__,
                "category": categorize(variables),
                "summary": summarize(propagator, variables),
                "descriptionHtml": description_html or "<p>No description is available.</p>",
                "descriptionMath": description_math,
                "variables": variables,
                "options": declared_options(propagator),
                "usedBy": sorted(
                    usage.get(name, []), key=lambda entry: (entry["type"], entry["model"])
                ),
                "docsUrl": f"{DOCS_BASE}#struphy.propagators.{propagator.__module__.split('.')[-1]}.{name}",
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")

    unused = [entry["className"] for entry in catalogue if not entry["usedBy"]]
    print(f"Generated {len(catalogue)} propagators in {output_file}")
    if unused:
        print(f"  {len(unused)} not used by any model: {', '.join(unused)}")


if __name__ == "__main__":
    generate()
