"""Generate the time-integration data used by the Struphy website.

The /time-integration/ page is the time-side companion to /feec/: where that one
explains how Struphy discretizes fields in space, this one covers how a model
gets advanced from one time level to the next. Two levels are involved:

* the **splitting scheme**, which composes a model's propagators into a step --
  ``StruphyModel.integrate`` implements Lie-Trotter and Strang, and the shape of
  each is read off that method here rather than described from memory;
* the **Butcher tableaux** available to the propagators that step explicitly,
  taken from ``ButcherTableau``'s own coefficient arrays.

The per-model step sequences the page draws are not generated here: they come
from ``propagators.json``, which already records each propagator's position in
every model that uses it.
"""

from __future__ import annotations

import inspect
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from struphy.io.options import LiteralOptions, Time
from struphy.models.base import StruphyModel
from struphy.ode.utils import ButcherTableau

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "time-integration.json"

# How each splitting algorithm lays a model's propagator list out over one step.
# `shape` is what the page draws; it mirrors the branches of
# `StruphyModel.integrate`, which is verified below against the live source.
SPLIT_ALGOS = {
    "LieTrotter": {
        "order": 1,
        "shape": "sequential",
        "summary": "Apply every propagator once, in the model's order, each over the full step.",
        "detail": (
            "The cheapest composition: one pass through the propagator list. It is first-order "
            "accurate in time, because the propagators do not commute -- the error left over from "
            "applying them in sequence rather than simultaneously is O(dt²) per step."
        ),
        "minimumPropagators": 1,
    },
    "Strang": {
        "order": 2,
        "shape": "palindromic",
        "summary": "Half-steps down the list, one full step of the last propagator, then the half-steps again in reverse.",
        "detail": (
            "Making the composition palindromic cancels the leading error term, giving second-order "
            "accuracy for roughly twice the work per step. It needs at least two propagators, so a "
            "model with a single step can only run Lie-Trotter."
        ),
        "minimumPropagators": 2,
    },
}


def verify_split_algos() -> None:
    """Fail loudly if `integrate` gains, loses or renames a splitting branch.

    The page's diagrams encode the shape of each branch, which no data structure
    in struphy exposes -- so this at least ties the set of documented algorithms
    to the source, instead of letting the page drift silently.
    """
    source = inspect.getsource(StruphyModel.integrate)
    in_source = set(re.findall(r'split_algo == "(\w+)"', source))
    declared = set(LiteralOptions.SplitAlgos.__args__)

    if in_source != declared:
        raise SystemExit(f"LiteralOptions.SplitAlgos {sorted(declared)} does not match integrate() {sorted(in_source)}")
    if declared != set(SPLIT_ALGOS):
        raise SystemExit(f"Struphy implements {sorted(declared)}, this generator documents {sorted(SPLIT_ALGOS)}")

    # Strang asserts it has something to fold in half; the page says so.
    if "assert len(self.prop_list) > 1" not in source:
        print("  note: Strang no longer asserts more than one propagator -- check minimumPropagators")


def as_fraction(value: float) -> str:
    """Render a tableau coefficient the way it is written in a Butcher tableau.

    The coefficients are exact small rationals stored as floats, so 1/6 comes
    back as 0.16666666666666666; showing that on the page would be noise.
    """
    fraction = Fraction(float(value)).limit_denominator(1000)
    if float(fraction) != float(value) and abs(float(fraction) - float(value)) > 1e-12:
        return f"{value:.6g}"
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def butcher_tableaux() -> list[dict]:
    tableaux = []
    for algo in ButcherTableau.__available_methods__:
        tableau = ButcherTableau(algo)
        tableaux.append(
            {
                "algo": algo,
                "stages": int(tableau.n_stages),
                "order": int(tableau.conv_rate),
                # a is strictly lower-triangular (explicit methods only), so each
                # row i only carries its first i entries.
                "a": [[as_fraction(value) for value in row[:index]] for index, row in enumerate(np.asarray(tableau.a))],
                "b": [as_fraction(value) for value in np.asarray(tableau.b)],
                "c": [as_fraction(value) for value in np.asarray(tableau.c)],
            }
        )
    return sorted(tableaux, key=lambda entry: (entry["order"], entry["stages"], entry["algo"]))


def generate(output_file: Path = OUTPUT_FILE) -> None:
    verify_split_algos()

    defaults = Time()
    data = {
        "splitAlgos": [{"name": name, **details} for name, details in SPLIT_ALGOS.items()],
        "butcher": butcher_tableaux(),
        "defaults": {
            "dt": defaults.dt,
            "Tend": defaults.Tend,
            "split_algo": defaults.split_algo,
            "butcher": ButcherTableau().algo,
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"Generated {len(data['splitAlgos'])} splitting schemes and "
        f"{len(data['butcher'])} Butcher tableaux in {output_file}"
    )


if __name__ == "__main__":
    generate()
