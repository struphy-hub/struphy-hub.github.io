"""Generate the kinetic-background catalogue used by the Struphy website.

Kinetic backgrounds are the velocity-space counterpart of the fluid equilibria
already in the catalogue: the distribution :math:`f_0` a kinetic species is
initialized from, and the denominator of the particle weights in the
control-variate method.

Unlike the fluid equilibria and perturbations, which vary over *space* and are
therefore sampled on xy/xz/yz planes, what distinguishes one kinetic background
from another lives in *velocity* space -- and in different velocity coordinates
per class (Cartesian, :math:`(v_\\parallel, \\mu)`, ...). So the sampling here
is done in velocity, at a fixed point in logical space, through struphy's own
``KineticBackground.reduced_eval``, which integrates out the axes that are not
plotted and knows each coordinate's admissible range (symmetric about zero for
:math:`v_\\parallel`, non-negative for :math:`\\mu` and :math:`v_\\perp`).

The output is one self-contained JSON file, like ``models.json`` and
``propagators.json``, rather than the two-stage per-item pipeline the
space-sampled catalogues use: there are only a handful of backgrounds, and
their plot data does not fit that pipeline's fixed plane/field shape.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
from catalogue_docs import class_description
from struphy.fields_background import equils
from struphy.kinetic_background import maxwellians
from struphy.kinetic_background.base import KineticBackground
from struphy.utils.docstring_converter import latex_to_unicode

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "kinetic-backgrounds.json"

# Velocity space is sampled at one point in logical space. It only matters for
# backgrounds whose moments vary spatially (CanonicalMaxwellian2D, or any
# background given callable moments), but sampling every class at the same
# point keeps the plots comparable -- and is what the pages state.
ETA = (0.5, 0.5, 0.5)

PROFILE_RESOLUTION = 121
SLICE_RESOLUTION = 48
INTEGRATION_RESOLUTION = 40

# CanonicalMaxwellian2D is parameterized by the canonical toroidal momentum, so
# it cannot be constructed without a magnetic equilibrium to supply the poloidal
# flux. The others take defaults.
CONSTRUCTOR_ARGS = {
    "CanonicalMaxwellian2D": lambda: {"equil": equils.AdhocTorus()},
}

# Axis labels per velocity coordinate system, indexed as (v1, v2, v3).
VELOCITY_LABELS = {
    "cartesian": ("vx", "vy", "vz"),
    "vpara_mu": ("v∥", "μ"),
    "vpara_vperp": ("v∥", "v⊥"),
    "vpara_energy": ("v∥", "ε"),
}

DOCS_BASE = "https://struphy-hub.github.io/struphy/sections/subsections/kinetic_backgrounds.html"


def available_backgrounds() -> list[tuple[str, KineticBackground]]:
    """Instantiate every public kinetic background."""
    backgrounds = []
    for name, cls in sorted(vars(maxwellians).items()):
        if not inspect.isclass(cls) or cls.__module__ != maxwellians.__name__:
            continue
        if not issubclass(cls, KineticBackground):
            continue
        kwargs = CONSTRUCTOR_ARGS.get(name, dict)()
        try:
            backgrounds.append((name, cls(**kwargs)))
        except Exception as error:  # noqa: BLE001 -- report and keep going
            print(f"Skipped {name}: {type(error).__name__}: {error}")
    return backgrounds


def axis_labels(background: KineticBackground) -> list[str]:
    labels = VELOCITY_LABELS.get(background.velocity_coords, ())
    # Fall back to the raw axis keys for a coordinate system added later.
    return [labels[index] if index < len(labels) else f"v{index + 1}" for index in range(background.vdim)]


def integration_resolution(background: KineticBackground, plotted: tuple[int, ...]) -> tuple:
    """Build ``reduced_eval``'s per-axis resolution argument.

    One entry per phase-space axis (3 space + ``vdim`` velocity): ``None`` for
    the axes being plotted, a float for the space axes (meaning "evaluate here"
    rather than "integrate over"), and a quadrature-point count for the velocity
    axes that are integrated out.
    """
    resolution: list = list(ETA)
    for index in range(background.vdim):
        resolution.append(None if index in plotted else INTEGRATION_RESOLUTION)
    return tuple(resolution)


def finite(values: np.ndarray, significant: int = 6) -> np.ndarray:
    """Sanitize for JSON and drop precision no plot can show, to keep the file small."""
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore"):
        magnitude = np.where(clean == 0.0, 0.0, np.floor(np.log10(np.abs(clean))))
    factor = 10.0 ** np.clip(significant - 1 - magnitude, -15, 15)
    return np.round(clean * factor) / factor


def sample_profile(background: KineticBackground, axis: int) -> dict:
    """The background reduced to a single velocity axis, others integrated out."""
    values, points, _, _ = background.reduced_eval(
        dim_1=f"v{axis + 1}",
        resol=PROFILE_RESOLUTION,
        integrate_resol=integration_resolution(background, (axis,)),
    )
    return {
        "axis": axis_labels(background)[axis],
        "coordinates": finite(points).tolist(),
        "values": finite(values).tolist(),
    }


def sample_slice(background: KineticBackground) -> dict:
    """The background over its first two velocity axes, the rest integrated out."""
    values, points1, points2, _ = background.reduced_eval(
        dim_1="v1",
        dim_2="v2",
        resol=SLICE_RESOLUTION,
        integrate_resol=integration_resolution(background, (0, 1)),
    )
    labels = axis_labels(background)
    return {
        "axes": labels[:2],
        "coordinates": {
            labels[0]: finite(points1).tolist(),
            labels[1]: finite(points2).tolist(),
        },
        "values": finite(values).tolist(),
    }


def summarize(cls: type, limit: int = 190) -> str:
    """A one-sentence plain-text blurb for the index cards and the search index.

    These docstrings lead with inline math -- "depending on two velocities
    :math:`(v_\\parallel, \\mu)`" -- which the site's generic `excerpt` helper
    replaces with the words "mathematical formulation", turning the sentence
    into nonsense. struphy's own `latex_to_unicode` renders it properly instead;
    it exists for VS Code hover tooltips, and a text-only card blurb has the
    same constraint.
    """
    docstring = inspect.getdoc(cls) or ""
    intro = re.split(r"\nParameters\n-+[ \t]*\n", docstring, maxsplit=1)[0]
    intro = re.split(r"\n\s*\.\. \w+::", intro, maxsplit=1)[0]
    # Math first: the generic role strip below would otherwise unwrap `:math:`
    # too, and the LaTeX inside it would then be eaten as unknown commands.
    intro = re.sub(r":math:`([^`]+)`", lambda m: latex_to_unicode(m.group(1)), intro)
    intro = re.sub(r":\w+:`(?:~[^`]*\.)?([^`]+)`", lambda m: m.group(1).rsplit(".", 1)[-1], intro)
    # latex_to_unicode marks sub/superscripts it has no character for with HTML
    # tags, and leaves commands it doesn't know as-is; neither belongs in text.
    intro = re.sub(r"</?su[bp]>", "", intro)
    intro = re.sub(r"\\[a-zA-Z]+\s*", "", intro)
    intro = " ".join(intro.split())

    sentence = re.split(r"(?<=[.!?])\s+", intro, maxsplit=1)[0] if intro else ""
    if len(sentence) > limit:
        sentence = f"{sentence[:limit].rsplit(' ', 1)[0]}…"
    return sentence


def parameter_descriptions(cls: type) -> dict[str, str]:
    """Map parameter name -> prose from the class docstring's ``Parameters`` section.

    A single entry may document several parameters at once (``n, ui, vthi :
    tuple``), so each name in the header shares the description.
    """
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


def json_parameters(parameters: dict) -> dict:
    """Constructor parameters as JSON-safe values.

    Moments are stored as ``(background, perturbation)`` tuples, where the
    background may be a float or a callable and the perturbation is usually
    ``None`` -- so callables are reduced to their name and everything else
    unrecognized to its repr.
    """

    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items() if key != "self"}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if callable(value):
            return getattr(value, "__name__", type(value).__name__)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__

    return convert(parameters)


def generate(output_file: Path = OUTPUT_FILE) -> None:
    catalogue = []
    for name, background in available_backgrounds():
        description_html, description_math = class_description(type(background))
        entry = {
            "className": name,
            "summary": summarize(type(background)),
            "vdim": background.vdim,
            "velocityCoords": background.velocity_coords,
            "velocityLabels": axis_labels(background),
            "volumeForm": bool(background.volume_form),
            "descriptionHtml": description_html or "<p>No description is available.</p>",
            "descriptionMath": description_math,
            "parameters": json_parameters(getattr(background, "params", {})),
            "parameterDescriptions": parameter_descriptions(type(background)),
            "constructedWith": sorted(CONSTRUCTOR_ARGS.get(name, dict)()),
            "eta": list(ETA),
            "profiles": [sample_profile(background, axis) for axis in range(background.vdim)],
            "slice": sample_slice(background) if background.vdim >= 2 else None,
            "docsUrl": f"{DOCS_BASE}#struphy.kinetic_background.maxwellians.{name}",
        }
        catalogue.append(entry)
        plots = f"{len(entry['profiles'])} profile(s)" + (", 1 slice" if entry["slice"] else "")
        print(f"  {name}: vdim={background.vdim}, {background.velocity_coords or 'no velocity space'}, {plots}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(catalogue, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(f"Generated {len(catalogue)} kinetic backgrounds in {output_file}")


if __name__ == "__main__":
    generate()
