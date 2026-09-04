"""Generate the static model catalogue used by the Struphy website."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from struphy.models.utils import get_models
from struphy.utils.docstring_converter import rst_to_html

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "models.json"
CATEGORIES = ("Fluid", "Kinetic", "Hybrid", "Toy")


def short_description(model: type) -> str:
    """Return the introductory sentence from a model class docstring."""
    docstring = inspect.getdoc(model) or ""
    introduction = docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ").strip()
    # Render simple Sphinx class references as ordinary text in the static catalogue.
    return re.sub(r":class:`(?:~[^`]*\.)?([^`.]+)`", r"\1", introduction)


def documentation_html(model: type, attribute: str, fallback: str) -> str:
    """Convert a model's notebook documentation helper docstring to HTML."""
    rst = inspect.getdoc(getattr(model, attribute, None)) or ""
    return rst_to_html(rst, forced_heading_level=3).strip() if rst else fallback


def generate(output_file: Path = OUTPUT_FILE) -> None:
    """Write models from Struphy's supported public catalogue API as JSON."""
    catalogue = []
    for category in CATEGORIES:
        for model in get_models(category):
            catalogue.append(
                {
                    "className": model.__name__,
                    "name": model.name(),
                    "type": model.model_type(),
                    "description": short_description(model),
                    "pdeHtml": documentation_html(
                        model, "doc_pde", "<p>No PDE description is available.</p>"
                    ),
                    "longDescriptionHtml": documentation_html(
                        model,
                        "doc_long_description",
                        "<p>No long description is available.</p>",
                    ),
                    "normalizationHtml": documentation_html(
                        model,
                        "doc_normalization",
                        "<p>No normalization information is available.</p>",
                    ),
                    "discretizationHtml": documentation_html(
                        model,
                        "doc_discretization",
                        "<p>No discretization information is available.</p>",
                    ),
                    "scalarQuantitiesHtml": documentation_html(
                        model,
                        "doc_scalar_quantities",
                        "<p>No tracked scalar quantities are documented.</p>",
                    ),
                    "useCasesHtml": documentation_html(
                        model, "doc_use_cases", "<p>No use cases are documented.</p>"
                    ),
                    "cannotBeUsedForHtml": documentation_html(
                        model,
                        "doc_cannot_be_used_for",
                        "<p>No limitations are documented.</p>",
                    ),
                    "examplesHtml": documentation_html(
                        model, "doc_examples", "<p>No examples are available.</p>"
                    ),
                }
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(catalogue)} models in {output_file}")


if __name__ == "__main__":
    generate()
