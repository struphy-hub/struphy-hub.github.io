"""Generate the static model catalogue used by the Struphy website."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from struphy.models.utils import get_models


OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "models.json"
CATEGORIES = ("Fluid", "Kinetic", "Hybrid", "Toy")


def short_description(model: type) -> str:
    """Return the introductory sentence from a model class docstring."""
    docstring = inspect.getdoc(model) or ""
    introduction = docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ").strip()
    # Render simple Sphinx class references as ordinary text in the static catalogue.
    return re.sub(r":class:`(?:~[^`]*\.)?([^`.]+)`", r"\1", introduction)


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
                }
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(catalogue)} models in {output_file}")


if __name__ == "__main__":
    generate()
