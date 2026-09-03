import json
import inspect
import re
from pathlib import Path

import numpy as np
from struphy.fields_background import equils


OUTPUT_DIR = Path(__file__).parent / "docs" / "src" / "data" / "equilibria"

# A few analytic equilibria use flat profile defaults. These presets keep the
# catalogue visually informative while remaining representative examples.
PRESETS = {
    "ScrewPinch": {"R0": 3.0, "n1": 2.0, "n2": 2.0, "na": 0.25},
    "ShearedSlab": {"n1": 2.0, "n2": 2.0, "na": 0.25},
    "ConstantVelocity": {"density_profile": "gaussian_xy", "p0": 0.35},
    "AdhocTorusQPsi": {"beta": 6.0, "q0": 0.35, "q1": 5.0},
    "CircularTokamak": {"B0": 4.0, "Bp": 60.0},
    "CurrentSheet": {"delta": 0.08, "amp": 8.0},
    "HomogenSlab": {"B0x": 1.2, "B0y": 0.8, "B0z": 1.0, "beta": 2.0},
}


def sample_slice(eq, plane="xy", value=0.0, extent=None, n=201):
    """Sample an equilibrium on a 2-D Cartesian slice."""
    if extent is None:
        a = eq.params.get("a", 1.0)
        extent = (-a, a)

    u = np.linspace(*extent, n)
    v = np.linspace(*extent, n)
    U, V = np.meshgrid(u, v, indexing="xy")

    if plane == "xy":
        X, Y, Z = U, V, value
        axes = ("x", "y")
    elif plane == "xz":
        X, Y, Z = U, value, V
        axes = ("x", "z")
    elif plane == "yz":
        X, Y, Z = value, U, V
        axes = ("y", "z")
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    fields = {}
    for field in ("p_xyz", "n_xyz"):
        method = getattr(eq, field, None)
        if method is not None:
            values = np.asarray(method(X, Y, Z))
            if values.ndim == 0:
                values = np.full(U.shape, values.item())
            fields[field.removesuffix("_xyz")] = {
                "kind": "scalar",
                "values": np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            }

    for field in ("b_xyz", "j_xyz"):
        method = getattr(eq, field, None)
        if method is None:
            continue
        components = method(X, Y, Z)
        if components is None:
            continue
        fields[field.removesuffix("_xyz")] = {
            "kind": "vector",
            "components": {
                axis: np.nan_to_num(np.asarray(component), nan=0.0, posinf=0.0, neginf=0.0).tolist()
                for axis, component in zip(("x", "y", "z"), components)
            },
        }
        if field == "b_xyz":
            magnitude = np.sqrt(sum(np.asarray(component) ** 2 for component in components))
            fields["bmag"] = {"kind": "scalar", "values": np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0).tolist()}

    return {
        "plane": plane,
        "value": value,
        "axes": list(axes),
        "coordinates": {axes[0]: u.tolist(), axes[1]: v.tolist()},
        "shape": [len(v), len(u)],
        "fields": fields,
    }


def sample_centerline(eq, axis="x", extent=None, n=401):
    """Sample scalar equilibrium profiles along one Cartesian coordinate line."""
    if extent is None:
        a = eq.params.get("a", 1.0)
        extent = (-a, a)
    x = np.linspace(*extent, n)
    zeros = np.zeros_like(x)
    coordinates = {"x": (x, zeros, zeros), "y": (zeros, x, zeros), "z": (zeros, zeros, x)}
    X, Y, Z = coordinates[axis]
    fields = {}
    for field in ("p_xyz", "n_xyz"):
        method = getattr(eq, field, None)
        if method is None:
            continue
        values = np.asarray(method(X, Y, Z))
        if values.ndim == 0:
            values = np.full(x.shape, values.item())
        fields[field.removesuffix("_xyz")] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).tolist()
    magnetic = getattr(eq, "b_xyz", None)
    if magnetic is not None:
        components = magnetic(X, Y, Z)
        if components is not None:
            magnitude = np.sqrt(sum(np.asarray(component) ** 2 for component in components))
            fields["bmag"] = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0).tolist()
    return {"axis": axis, "coordinates": x.tolist(), "fields": fields}


def export_equilibrium(name, eq, output_dir=OUTPUT_DIR, **slice_options):
    """Export one equilibrium slice as JSON for the documentation site."""
    slices = {
        plane: sample_slice(eq, plane=plane, **slice_options)
        for plane in ("xy", "xz", "yz")
    }
    data = {
        "name": name,
        "type": type(eq).__name__,
        "slice": slices["xy"],
        "slices": slices,
        "centerline": sample_centerline(eq),
        "centerlines": {axis: sample_centerline(eq, axis=axis) for axis in ("x", "y", "z")},
        "parameters": _json_parameters(getattr(eq, "params", {})),
        "parameter_descriptions": parameter_descriptions(type(eq)),
    }
    if not has_variation(data):
        raise ValueError("all sampled equilibrium fields are constant")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, separators=(",", ":"), allow_nan=False)
        output_file.write("\n")
    return output_path


def has_variation(data, tolerance=1e-10) -> bool:
    """Return whether at least one sampled scalar field varies meaningfully."""
    samples = []
    for slice_data in data.get("slices", {}).values():
        for field in slice_data.get("fields", {}).values():
            if field.get("kind") == "scalar":
                samples.extend(np.asarray(field.get("values", []), dtype=float).ravel())
    for line in data.get("centerlines", {}).values():
        samples.extend(np.asarray(line.get("fields", {}).get("bmag", []), dtype=float).ravel())
    return bool(samples) and any(np.ptp(values) > tolerance for values in [np.asarray(samples)])


def _json_parameters(parameters):
    """Convert equilibrium parameters to JSON-safe values for the catalogue."""
    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
    return convert(parameters)


def parameter_descriptions(cls) -> dict[str, str]:
    """Extract NumPy-style ``Parameters`` descriptions from an equilibrium docstring."""
    doc = inspect.getdoc(cls) or ""
    match = re.search(r"\nParameters\n-+\n(?P<body>.*?)(?:\n\n(?:Note|Notes|Returns|Attributes)\n-+|\Z)", doc, re.S)
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


def available_equilibria() -> list[tuple[str, object]]:
    """Instantiate every public equilibrium that works with its defaults.

    Some equilibria depend on optional external packages or data files. Those
    are skipped so the catalogue can still be built with the analytic fields.
    """
    result = []
    for name, cls in sorted(vars(equils).items(), key=lambda item: item[0].lower()):
        if not inspect.isclass(cls) or cls.__module__ != equils.__name__:
            continue
        try:
            kwargs = PRESETS.get(name, {}).copy()
            if name.startswith("GenericCartesianFluidEquilibrium"):
                kwargs.update({
                    "p_xyz": lambda x, y, z: 1.0 + 0.5 * np.sin(x),
                    "n_xyz": lambda x, y, z: 1.0 + 0.25 * np.cos(y),
                })
            result.append((name, cls(**kwargs)))
        except (Exception, SystemExit) as error:
            print(f"Skipped {name}: {error or type(error).__name__}")
    return result


if __name__ == "__main__":
    for equilibrium_name, equilibrium in available_equilibria():
        equilibrium_name = equilibrium_name.lower()
        output_path = export_equilibrium(equilibrium_name, equilibrium)
        print(f"Exported {output_path}")
