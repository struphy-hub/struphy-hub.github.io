"""Generate lightweight scientific imagery for the Struphy landing page."""

from pathlib import Path

import pyvista as pv
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "public" / "images"


def render_gvec_geometry() -> None:
    """Render the Struphy-generated GVEC unit domain for web delivery."""
    mesh = pv.read(ROOT / "GVECunit.vtp")
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1400))
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
    plotter.camera_position = [(7.1, -7.8, 5.9), (0, 0, 0), (0, 0, 1)]
    plotter.camera.zoom(1.16)
    output_png = OUTPUT / "gvec-domain.png"
    plotter.screenshot(output_png, transparent_background=False)
    plotter.close()

    image = Image.open(output_png).convert("RGB")
    image.save(OUTPUT / "gvec-domain.webp", "WEBP", quality=91, method=6)
    output_png.unlink()


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render_gvec_geometry()
