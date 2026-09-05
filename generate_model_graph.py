"""Generate a model-relationship graph for the /models/relationships/ page.

Two models are linked when they use at least one propagator of the same
class (e.g. both use `VariationalMomentumAdvection`) -- a concrete, numerical
similarity, unlike category alone. Node/edge positions are precomputed here
with a small force-directed layout so the page can render static SVG with no
client-side physics simulation or extra JS dependency.

Also records each model's FEEC variable spaces (H1 / Hcurl / Hdiv / L2), which
the /feec/ page's "who uses this space" cross-reference reads.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from struphy.models.utils import get_models

OUTPUT_FILE = Path(__file__).parent / "docs" / "src" / "data" / "model-graph.json"
CATEGORIES = ("Fluid", "Kinetic", "Hybrid", "Toy")

FEEC_SPACES = {"H1", "Hcurl", "Hdiv", "L2", "H1vec"}


def collect_models() -> list[dict]:
    models = []
    for category in CATEGORIES:
        for model_cls in get_models(category):
            model = model_cls()
            variables = []
            for species_name, species in model.species.items():
                for var_name, var in species.variables.items():
                    space = getattr(var, "space", None)
                    variables.append(
                        {
                            "species": species_name,
                            "name": var_name,
                            "space": space,
                            "isFeec": space in FEEC_SPACES,
                        }
                    )
            propagators = sorted({type(p).__name__ for p in vars(model.propagators).values()})
            models.append(
                {
                    "id": model_cls.__name__,
                    "type": category,
                    "variables": variables,
                    "propagators": propagators,
                }
            )
    return models


def build_edges(models: list[dict]) -> list[dict]:
    edges = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            shared = sorted(set(a["propagators"]) & set(b["propagators"]))
            if shared:
                edges.append({"source": a["id"], "target": b["id"], "shared": shared, "weight": len(shared)})
    return edges


def layout(models: list[dict], edges: list[dict], iterations: int = 400, seed: int = 7) -> dict[str, tuple[float, float]]:
    """A minimal Fruchterman-Reingold-style force layout, so the page can ship
    static positions instead of running a physics simulation in the browser.
    """
    rng = random.Random(seed)
    ids = [m["id"] for m in models]
    n = len(ids)
    index = {node_id: i for i, node_id in enumerate(ids)}
    pos = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)])

    edge_pairs = [(index[e["source"]], index[e["target"]], e["weight"]) for e in edges]
    area = float(n)
    k = np.sqrt(area / max(n, 1))

    for step in range(iterations):
        disp = np.zeros_like(pos)

        # Repulsion between every pair (keeps unrelated nodes apart).
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = 1.0
            dist = np.clip(dist, 0.01, None)
            force = (k * k) / dist
            disp[i] += (delta / dist[:, None] * force[:, None]).sum(axis=0)

        # Attraction along edges (stronger for more shared propagators).
        for i, j, weight in edge_pairs:
            delta = pos[i] - pos[j]
            dist = max(np.linalg.norm(delta), 0.01)
            force = (dist * dist) / k * (1 + 0.4 * weight)
            direction = delta / dist
            disp[i] -= direction * force
            disp[j] += direction * force

        cooling = 1 - step / iterations
        disp_len = np.linalg.norm(disp, axis=1)
        disp_len = np.clip(disp_len, 0.01, None)
        max_step = 0.3 * cooling + 0.01
        pos += (disp / disp_len[:, None]) * np.minimum(disp_len, max_step)[:, None]

    # Normalize to a 140x100 viewBox (landscape) with margin.
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    span = np.clip(maxs - mins, 1e-6, None)
    normalized = (pos - mins) / span * np.array([124, 84]) + np.array([8, 8])

    return {node_id: (float(normalized[i, 0]), float(normalized[i, 1])) for node_id, i in index.items()}


def generate() -> None:
    models = collect_models()
    edges = build_edges(models)
    positions = layout(models, edges)

    nodes = []
    for model in models:
        x, y = positions[model["id"]]
        degree = sum(1 for e in edges if model["id"] in (e["source"], e["target"]))
        nodes.append(
            {
                "id": model["id"],
                "type": model["type"],
                "variables": model["variables"],
                "propagators": model["propagators"],
                "degree": degree,
                "x": round(x, 2),
                "y": round(y, 2),
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(nodes)} nodes and {len(edges)} edges in {OUTPUT_FILE}")


if __name__ == "__main__":
    generate()
