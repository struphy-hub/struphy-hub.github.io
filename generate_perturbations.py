import inspect
import json
import re
from pathlib import Path

import numpy as np
from struphy.initial import perturbations

OUTPUT_DIR = Path(__file__).parent / "docs" / "src" / "data" / "perturbations"

# A handful of perturbations are flat by default (mode numbers of zero cancel
# out cos/sin terms). These presets pick small nonzero mode numbers so the
# catalogue is visually informative.
PRESETS = {
    "ModesCos": {"ls": (2,), "ms": (3,)},
    "ModesCosCos": {"ls": (2,), "ms": (3,)},
    "ModesCosSin": {"ls": (2,), "ms": (3,)},
    "ModesSin": {"ls": (2,), "ms": (3,)},
    "ModesSinCos": {"ls": (2,), "ms": (3,)},
    "ModesSinSin": {"ls": (2,), "ms": (3,)},
}

# Perturbations that cannot be evaluated as a plain function of coordinates
# (e.g. random FEEC coefficient noise) are skipped.
SKIP = {"Noise"}


def sample_slice(perturbation, plane="xy", value=0.5, extent=(0.02, 0.98), n=41):
    """Sample a perturbation on a 2-D logical-space slice."""
    u = np.linspace(*extent, n)
    v = np.linspace(*extent, n)
    U, V = np.meshgrid(u, v, indexing="xy")

    if plane == "xy":
        E1, E2, E3 = U, V, value
        axes = ("eta1", "eta2")
    elif plane == "xz":
        E1, E2, E3 = U, value, V
        axes = ("eta1", "eta3")
    elif plane == "yz":
        E1, E2, E3 = value, U, V
        axes = ("eta2", "eta3")
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    values = np.asarray(perturbation(E1, E2, E3), dtype=float)
    if values.ndim == 0:
        values = np.full(U.shape, values.item())

    return {
        "plane": plane,
        "value": value,
        "axes": list(axes),
        "coordinates": {axes[0]: u.tolist(), axes[1]: v.tolist()},
        "shape": [len(v), len(u)],
        "fields": {
            "u": {
                "kind": "scalar",
                "values": np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            }
        },
    }


def sample_centerline(perturbation, axis="eta1", extent=(0.02, 0.98), n=121):
    """Sample a perturbation along one logical coordinate line, others fixed at 0.5."""
    t = np.linspace(*extent, n)
    half = np.full_like(t, 0.5)
    coordinates = {
        "eta1": (t, half, half),
        "eta2": (half, t, half),
        "eta3": (half, half, t),
    }
    E1, E2, E3 = coordinates[axis]
    values = np.asarray(perturbation(E1, E2, E3), dtype=float)
    if values.ndim == 0:
        values = np.full(t.shape, values.item())
    return {
        "axis": axis,
        "coordinates": t.tolist(),
        "fields": {
            "u": np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        },
    }


def export_perturbation(name, perturbation, output_dir=OUTPUT_DIR):
    """Export one perturbation's slices and centerlines as JSON for the docs site."""
    slices = {plane: sample_slice(perturbation, plane=plane) for plane in ("xy", "xz", "yz")}
    data = {
        "name": name,
        "type": type(perturbation).__name__,
        "slice": slices["xy"],
        "slices": slices,
        "centerline": sample_centerline(perturbation, axis="eta1"),
        "centerlines": {
            axis: sample_centerline(perturbation, axis=axis) for axis in ("eta1", "eta2", "eta3")
        },
        "given_in_basis": perturbation.given_in_basis,
        "parameters": _json_parameters(getattr(perturbation, "params", {})),
        "parameter_descriptions": parameter_descriptions(type(perturbation)),
    }
    if not has_variation(data):
        raise ValueError("all sampled perturbation values are constant")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, separators=(",", ":"), allow_nan=False)
        output_file.write("\n")
    return output_path


def has_variation(data, tolerance=1e-10) -> bool:
    """Return whether the sampled field varies meaningfully anywhere."""
    samples = []
    for slice_data in data.get("slices", {}).values():
        samples.extend(np.asarray(slice_data["fields"]["u"]["values"], dtype=float).ravel())
    for line in data.get("centerlines", {}).values():
        samples.extend(np.asarray(line["fields"]["u"], dtype=float).ravel())
    return bool(samples) and np.ptp(np.asarray(samples)) > tolerance


def _json_parameters(parameters):
    """Convert perturbation parameters to JSON-safe values for the catalogue."""

    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items() if key != "self"}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if callable(value):
            return getattr(value, "__name__", str(value))
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return convert(parameters)


def parameter_descriptions(cls) -> dict[str, str]:
    """Extract NumPy-style ``Parameters`` descriptions from a perturbation docstring."""
    doc = inspect.getdoc(cls) or ""
    match = re.search(
        r"\nParameters\n-+\n(?P<body>.*?)(?:\n\n(?:Note|Notes|Returns|Attributes|Examples)\n-+|\Z)",
        doc,
        re.S,
    )
    if not match:
        return {}
    descriptions: dict[str, str] = {}
    current: str | None = None
    for line in match.group("body").splitlines():
        header = re.match(r"^\s*([A-Za-z_]\w*)\s*:\s*.+$", line)
        if header:
            current = header.group(1)
            descriptions[current] = ""
        elif current and line.strip():
            descriptions[current] = f"{descriptions[current]} {line.strip()}".strip()
    return descriptions


def available_perturbations() -> list[tuple[str, object]]:
    """Instantiate every public perturbation that works with its defaults."""
    result = []
    for name, cls in sorted(vars(perturbations).items(), key=lambda item: item[0].lower()):
        if not inspect.isclass(cls) or cls.__module__ != perturbations.__name__:
            continue
        if name in SKIP:
            continue
        try:
            kwargs = PRESETS.get(name, {}).copy()
            result.append((name, cls(**kwargs)))
        except (Exception, SystemExit) as error:
            print(f"Skipped {name}: {error or type(error).__name__}")
    return result


if __name__ == "__main__":
    for perturbation_name, perturbation in available_perturbations():
        try:
            output_path = export_perturbation(perturbation_name, perturbation)
            print(f"Exported {output_path}")
        except (Exception, SystemExit) as error:
            print(f"Skipped {perturbation_name}: {error or type(error).__name__}")
