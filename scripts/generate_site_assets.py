"""Generate lightweight scientific imagery for the Struphy landing page."""

from pathlib import Path

import pyvista as pv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "public" / "images"


def render_tokamak_geometry() -> None:
    """Render the Struphy-generated Tokamak domain for web delivery."""
    mesh = pv.read(ROOT / "docs" / "public" / "domains" / "Tokamak.vtp")
    plotter = pv.Plotter(off_screen=True, window_size=(1440, 1120))
    plotter.set_background("#07131d")
    plotter.add_mesh(
        mesh,
        color="#55e6e1",
        style="wireframe",
        line_width=1.15,
        opacity=0.74,
        show_edges=True,
        edge_color="#8af5ef",
    )
    plotter.camera_position = [(5.4, -5.8, 3.8), mesh.center, (0, 0, 1)]
    plotter.camera.zoom(1.12)
    output_png = OUTPUT / "tokamak-domain.png"
    plotter.screenshot(output_png, transparent_background=False)
    plotter.close()

    image = Image.open(output_png).convert("RGB")
    image.save(OUTPUT / "tokamak-domain.webp", "WEBP", quality=80, method=6)
    output_png.unlink()


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render_tokamak_geometry()
