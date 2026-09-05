"""Verify a time-dependent Poisson solve against its analytic solution.

This compact gallery example follows Struphy's maintained Poisson verification
test. A cosine-mode charge density oscillates in time; Struphy's FEEC solver
recovers the potential at every step, compared here against the closed-form
solution.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from struphy import (
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    grids,
    perturbations,
)
from struphy.models import Poisson

# A time-dependent right-hand side needs the extra `source` propagator.
model = Poisson(with_t_dep_source=True)

# A 1D interval, resolved in the x-direction only.
domain = domains.Cuboid(l1=-5.0, r1=5.0)
grid = grids.TensorProductGrid(num_elements=(48, 1, 1))
time_opts = Time(dt=0.1, Tend=2.0)

# The source oscillates as cos(omega t), driving a single cosine spatial mode.
omega = 2 * np.pi
model.propagators.source.options = model.propagators.source.Options(omega=omega)

wavenumber = 2
amplitude = 0.1
model.em_fields.source.add_perturbation(
    perturbations.ModesCos(ls=(wavenumber,), amps=(amplitude,)),
)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="poisson_source",
)
sim = Simulation(
    model=model,
    name="Poisson time-dependent source",
    description=(
        "Drive a 1D Poisson solve with an oscillating cosine-mode charge "
        "density and compare Struphy’s FEEC potential against the exact "
        "solution, at every time step."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
)

if __name__ == "__main__":
    sim.run()
    sim.pproc()
    sim.load_plotting_data()

    # Exact solution of -d^2(phi)/dx^2 = rho(t, x), rho = A cos(k x) cos(omega t).
    Lx = domain.params["r1"] - domain.params["l1"]
    k = wavenumber * 2 * np.pi / Lx

    def phi_exact(x, t):
        return amplitude / k**2 * np.cos(k * x) * np.cos(omega * t)

    phi = sim.spline_values.em_fields.phi_log.data
    x = sim.grids_phy[0][:, 0, 0]
    times = sorted(phi.keys())

    phi_scale = amplitude / k**2
    max_relative_error = 0.0
    frames = []
    for t in times:
        phi_h = np.asarray(phi[t][0])[:, 0, 0]
        phi_e = phi_exact(x, t)
        max_relative_error = max(max_relative_error, float(np.max(np.abs(phi_h - phi_e))) / phi_scale)
        frames.append(
            go.Frame(
                name=f"{t:.2f}",
                data=[
                    go.Scatter(x=x, y=phi_h),
                    go.Scatter(x=x, y=phi_e),
                ],
            ),
        )

    print(f"Max relative error over the run: {max_relative_error:.5f}")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=np.asarray(phi[times[0]][0])[:, 0, 0],
                mode="lines",
                name="Struphy (FEEC)",
                line={"color": "#168aad", "width": 3},
            ),
            go.Scatter(
                x=x,
                y=phi_exact(x, times[0]),
                mode="lines",
                name="Exact",
                line={"color": "#d62828", "width": 2, "dash": "dot"},
            ),
        ],
        frames=frames,
    )
    figure.update_layout(
        title="Poisson potential: FEEC solution vs. exact",
        xaxis_title="x [a.u.]",
        yaxis_title="φ [a.u.]",
        template="plotly_white",
        autosize=True,
        yaxis={"range": [-1.15 * phi_scale, 1.15 * phi_scale]},
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(44,62,80,0.25)",
            "borderwidth": 1,
        },
        margin={"l": 60, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.32,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {"args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": frame.name, "method": "animate"}
                    for frame in frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.2,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    png_path = Path("poisson-source.png")
    html_path = Path("poisson-source.html")
    figure.write_image(png_path, width=1100, height=650, scale=2)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {html_path.resolve()}")

    metadata_path = Path("poisson-source.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["maxRelativeError"] = max_relative_error
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
