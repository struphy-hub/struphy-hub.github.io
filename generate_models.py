"""Generate the static model catalogue used by the Struphy website."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from struphy.models.utils import get_models
from struphy.utils.docstring_converter import _extract_math_directives, rst_to_html

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "models.json"
CATEGORIES = ("Fluid", "Kinetic", "Hybrid", "Toy")


def short_description(model: type) -> str:
    """Return the introductory sentence from a model class docstring."""
    docstring = inspect.getdoc(model) or ""
    introduction = docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ").strip()
    # Render simple Sphinx class references as ordinary text in the static catalogue.
    return re.sub(r":class:`(?:~[^`]*\.)?([^`.]+)`", r"\1", introduction)


def extract_math(rst: str) -> tuple[str, list[dict]]:
    """Pull raw LaTeX out of ``.. math::`` blocks and ``:math:`...``` roles.

    Each occurrence is replaced with a placeholder token so the surrounding
    RST (headings, lists, code blocks) can still go through struphy's
    ``rst_to_html`` unharmed. The real LaTeX is returned separately so the
    site can render it with KaTeX instead of struphy's Unicode-approximation
    converter (which exists only because VS Code hover tooltips can't load a
    real math renderer -- a constraint that doesn't apply to a web page).
    """
    math_items: list[dict] = []

    def save_block(content: str) -> str:
        raw = "\n".join(line.strip() for line in content.strip().split("\n") if line.strip())
        # A block that already opens with its own environment (``\begin{bmatrix}``,
        # ``\begin{aligned}``, ...) is self-contained -- pass it through as-is.
        if raw.lstrip().startswith("\\begin{"):
            latex = raw
        else:
            # Otherwise docstrings write bare multi-row systems -- either
            # linebroken with an explicit `\\[2mm]`, or as blank-line-separated
            # paragraphs each ending in a comma -- relying on Sphinx's math
            # directive to implicitly lay them out as an aligned system. KaTeX
            # needs that made explicit, or a lone `&`/`\\` is a syntax error
            # outside an alignment environment.
            groups = [
                " ".join(line.strip() for line in group.split("\n") if line.strip())
                for group in re.split(r"\n\s*\n", content.strip())
            ]
            needs_aligned = len(groups) > 1 or any("&" in group for group in groups)
            latex = r" \\ ".join(groups)
            if needs_aligned:
                latex = f"\\begin{{aligned}} {latex} \\end{{aligned}}"
        token = f"@@MATH{len(math_items)}@@"
        math_items.append({"token": token, "latex": latex, "display": True})
        return token

    stripped = _extract_math_directives(rst, save_block)

    def save_inline(match: re.Match) -> str:
        token = f"@@MATH{len(math_items)}@@"
        math_items.append({"token": token, "latex": match.group(1), "display": False})
        return token

    stripped = re.sub(r":math:`([^`]+)`", save_inline, stripped)
    return stripped, math_items


def documentation_html(model: type, attribute: str, fallback: str) -> tuple[str, list[dict]]:
    """Convert a model's notebook documentation helper docstring to HTML.

    Returns the HTML (with math replaced by placeholder tokens) alongside the
    list of extracted LaTeX expressions, for KaTeX rendering at the Astro
    build step.
    """
    rst = inspect.getdoc(getattr(model, attribute, None)) or ""
    if not rst:
        return fallback, []
    stripped, math_items = extract_math(rst)
    html = rst_to_html(stripped, forced_heading_level=3).strip()
    return (html or fallback), math_items


# Maps each catalogue field to the (docstring attribute, fallback HTML) it is built from.
DOC_FIELDS = {
    "pde": ("doc_pde", "<p>No PDE description is available.</p>"),
    "longDescription": ("doc_long_description", "<p>No long description is available.</p>"),
    "normalization": ("doc_normalization", "<p>No normalization information is available.</p>"),
    "discretization": ("doc_discretization", "<p>No discretization information is available.</p>"),
    "scalarQuantities": ("doc_scalar_quantities", "<p>No tracked scalar quantities are documented.</p>"),
    "useCases": ("doc_use_cases", "<p>No use cases are documented.</p>"),
    "cannotBeUsedFor": ("doc_cannot_be_used_for", "<p>No limitations are documented.</p>"),
    "examples": ("doc_examples", "<p>No examples are available.</p>"),
}


def generate(output_file: Path = OUTPUT_FILE) -> None:
    """Write models from Struphy's supported public catalogue API as JSON."""
    catalogue = []
    for category in CATEGORIES:
        for model in get_models(category):
            entry = {
                "className": model.__name__,
                "name": model.name(),
                "type": model.model_type(),
                "description": short_description(model),
            }
            for field, (attribute, fallback) in DOC_FIELDS.items():
                html, math = documentation_html(model, attribute, fallback)
                entry[f"{field}Html"] = html
                entry[f"{field}Math"] = math
            catalogue.append(entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(catalogue)} models in {output_file}")


if __name__ == "__main__":
    generate()
