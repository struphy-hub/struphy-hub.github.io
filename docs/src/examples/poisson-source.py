"""Verify a time-dependent Poisson solve against its analytic solution.

This compact gallery example follows Struphy's maintained Poisson verification
test. A cosine-mode charge density oscillates in time; Struphy's FEEC solver
recovers the potential at every step, compared here against the closed-form
solution.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import argparse

import numpy as np
import plotly.graph_objects as go
from struphy import (EnvironmentOptions, Simulation, Time, domains, grids,
                     perturbations)
from struphy.models import Poisson

# Problem parameters: a single cosine mode in space, oscillating in time.
WAVENUMBER = 2
AMPLITUDE = 0.1
OMEGA = 2 * np.pi


def create_simulation() -> Simulation:
    # A time-dependent right-hand side needs the extra `source` propagator.
    model = Poisson(with_t_dep_source=True)

    # A 1D interval, resolved in the x-direction only.
    domain = domains.Cuboid(l1=-5.0, r1=5.0)
    grid = grids.TensorProductGrid(num_elements=(48, 1, 1))
    time_opts = Time(dt=0.01, Tend=2.0)

    # The source oscillates as cos(omega t), driving a single cosine spatial mode.
    model.propagators.source.options = model.propagators.source.Options(omega=OMEGA)

    model.em_fields.source.add_perturbation(
        perturbations.ModesCos(ls=(WAVENUMBER,), amps=(AMPLITUDE,)),
    )

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="poisson_source",
    )
    simulation = Simulation(
        model=model,
        name="Poisson time-dependent source",
        description=(
            "Drive a 1D Poisson solve with an oscillating cosine-mode charge "
            "density and compare Struphy’s FEEC potential against the exact "
            "solution, at every time step."
            r" The prescribed source is $$\rho(x,t)=0.1\cos(2\pi x/5)\cos(2\pi t),$$"
            r" on the periodic interval :math:`-5\le x<5`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
    )
    return simulation


def pproc(sim: Simulation):

    from struphy.utils._gallery import export_profiling, merge_metadata, save_figure

    output = sim.output.process(create_vtk=False)
    domain = sim.domain

    # Exact solution of -d^2(phi)/dx^2 = rho(t, x), rho = A cos(k x) cos(omega t).
    Lx = domain.params["r1"] - domain.params["l1"]
    k = WAVENUMBER * 2 * np.pi / Lx

    def phi_exact(x, t):
        return AMPLITUDE / k**2 * np.cos(k * x) * np.cos(OMEGA * t)

    phi = output.fields.em_fields.phi
    phi_line = phi.isel(e2=0, e3=0)
    if "component" in phi_line.dims:
        phi_line = phi_line.isel(component=0)
    x = np.asarray(phi_line.X)
    times = np.asarray(phi_line.t)

    phi_scale = AMPLITUDE / k**2
    max_relative_error = 0.0
    frames = []
    for t in times:
        phi_h = np.asarray(phi_line.sel(t=t))
        phi_e = phi_exact(x, t)
        max_relative_error = max(
            max_relative_error, float(np.max(np.abs(phi_h - phi_e))) / phi_scale
        )
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
                y=np.asarray(phi_line.isel(t=0)),
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
                        "args": [
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.2,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    save_figure(figure, "poisson-source")

    profiling = export_profiling(sim, "poisson-source")

    merge_metadata(
        "poisson-source",
        maxRelativeError=max_relative_error,
        **profiling,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Poisson source example.")
    argparser.add_argument(
        "--pproc",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    args = argparser.parse_args()

    simulation = create_simulation()

    if args.pproc:
        pproc(simulation)
    else:
        simulation.run(profiling_activated=True)
        pproc(simulation)
