"""Shear relaxation of an incompressible fluid between no-slip walls, with incompressible SPH.

A viscous fluid fills a channel with no-slip walls at y = 0 and y = H. Its velocity starts as the
shear mode u_x = U sin(pi y / H), plus a compressive wave u_x = e sin(2 pi x) along the channel. Struphy's
`IncompressibleNavierStokesSPH` moves the particles with the fluid and, after every step, projects the
velocity onto a divergence-free field (Chorin's projection: a Poisson solve for the pressure). The
projection removes the compressive wave at once, and the shear mode, which is already divergence-free,
decays by viscous diffusion: u_x(y, t) = U sin(pi y / H) exp(-mu (pi / H)^2 t).

Based on the verification test of the model in Struphy
(models/tests/verification/test_verif_IncompressibleNavierStokesSPH.py).

Requires Struphy 3.3 with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import (
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
)
from struphy.initial.base import GenericPerturbation
from struphy.models import IncompressibleNavierStokesSPH
from struphy.ode.utils import ButcherTableau

viscosity = 0.1  # mu
height = 1.0  # H, the channel is 1 long in x and H high in y
shear_amplitude = 0.5  # U
compressive_amplitude = 0.2  # e
decay_rate = viscosity * (np.pi / height) ** 2  # the exact decay rate of the shear mode
boxes = 8
markers_per_box = 16
bins = 16


def create_simulation() -> Simulation:
    model = IncompressibleNavierStokesSPH(with_B0=False, with_viscosity=True)
    model.propagators.push_eta.options = model.propagators.push_eta.Options(
        butcher=ButcherTableau(algo="forward_euler"),
    )
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(
        kernel_type="gaussian_2d", mu=viscosity
    )

    domain = domains.Cuboid(r1=1.0, r2=height)
    # The pressure is a finite element field; the walls are free (no boundary condition) for it.
    grid = grids.TensorProductGrid(num_elements=(boxes, boxes, 1))
    derham_opts = DerhamOptions(degree=(2, 2, 1), bcs=(None, ("free", "free"), None))

    model.fluid.set_markers(
        loading_params=LoadingParameters(ppb=markers_per_box, loading="tesselation"),
        weights_params=WeightsParameters(),
        # Markers reflect off the walls, and the SPH sum sees them as no-slip; x is periodic.
        boundary_params=BoundaryParameters(
            bc=("periodic", "reflect", "periodic"), bc_sph=("periodic", "noslip", "periodic")
        ),
        sorting_params=SortingParameters(boxes_per_dim=(boxes, boxes, 1), dims_mask=(True, True, False)),
        saving_params=SavingParameters(
            binning_plots=(
                BinningPlot(slice="e2", n_bins=(bins,), ranges=(0.0, 1.0), output_quantity="current_1"),
                BinningPlot(slice="e1", n_bins=(bins,), ranges=(0.0, 1.0), output_quantity="current_1"),
            ),
        ),
        bufsize=2,
    )
    model.fluid.density.add_background(equils.ConstantVelocity())
    model.fluid.density.add_perturbation(
        del_u1=GenericPerturbation(
            fun=lambda e1, e2, e3: shear_amplitude * np.sin(np.pi * e2 / height)
            + compressive_amplitude * np.sin(2 * np.pi * e1)
        )
    )

    env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="incompressible_shear_relaxation")
    sim = Simulation(
        model=model,
        name="Incompressible shear relaxation",
        description=(
            "A viscous fluid between no-slip walls starts with a shear flow and a compressive wave. "
            "Incompressible SPH with a pressure projection removes the wave at once and lets the shear decay "
            "viscously, at the exact rate."
            r" Before the pressure projection, $$u_x(x,y,0)=0.5\sin(\pi y)+0.2\sin(2\pi x),\qquad u_y(x,y,0)=0,$$"
            r" with :math:`n=1`, viscosity :math:`\nu=0.1`, and no-slip walls at :math:`y=0,1`."
        ),
        env=env,
        time_opts=Time(dt=0.01, Tend=3.0, split_algo="LieTrotter"),
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def save(figure, name: str, *, show: bool = False, frame: int | None = None, still=None, width=1100, height=650):
    """Save a Plotly figure as ``<name>.html``, ``<name>.png`` and ``<name>.plotly.json``.

    ``figure`` is a plot of struphy-plots drawn with ``backend="plotly"``, or a
    ``plotly.graph_objects.Figure``. ``show`` shows it first. For an animation, the PNG shows
    ``frame`` (default: the first), or the figure ``still`` instead. Under MPI only rank 0 writes.
    """
    import struphy_plots
    from struphy_plots.plotting import PlotResult

    if not struphy_plots.is_plotting_rank():
        return
    result = figure if isinstance(figure, PlotResult) else PlotResult(figure, None)
    if show:
        result.show()
    result.save(f"{name}.html")
    image = PlotResult(still, None) if still is not None else result
    image.save(f"{name}.png", frame=frame, width=width, height=height, scale=2)
    result.save(f"{name}.plotly.json")
def pproc(sim: Simulation, show: bool = False):
    from plotly.subplots import make_subplots

    output = sim.output
    output.pproc()

    across = output.evaluate("fluid/e2_current_1/f")  # (t, e2): u_x against y, averaged over x
    along = output.evaluate("fluid/e1_current_1/f")  # (t, e1): u_x against x, averaged over y
    times = across.t.values
    y, x = across.eta2.values * height, along.eta1.values
    if not (np.isfinite(across.values).all() and np.isfinite(along.values).all()):
        raise RuntimeError("Non-finite SPH result: refusing to publish the run")

    # The amplitude of each mode, projected on it. A bin average lowers a mode by sinc(k dx / 2).
    # The shear mode sin(pi y / H) is half a wave along eta2 = y / H.
    shear = across.struphy.analysis.project_mode(dim="eta2", number=0.5, bin_correction=True)
    wave = along.struphy.analysis.project_mode(dim="eta1", number=1, bin_correction=True).values
    exact_shear = shear_amplitude * np.exp(-decay_rate * times)
    fitted_rate = -abs(shear).struphy.analysis.growth_rate().rate
    profile_error = float(
        across.isel(t=-1).struphy.analysis.error(exact_shear[-1] * np.sin(np.pi * y / height), norm="max")
    )
    shear = shear.values
    wave_left = float(np.abs(wave[times >= 0.1]).max() / compressive_amplitude)
    print(f"shear decay rate {fitted_rate:.3f} (exact {decay_rate:.3f}); compressive wave after t = 0.1: "
          f"{100 * wave_left:.1f}% of its initial amplitude; profile error at the end {profile_error:.4f}")

    dense_y = np.linspace(0.0, height, 201)

    def traces(index):
        return [
            go.Scatter(x=dense_y, y=exact_shear[index] * np.sin(np.pi * dense_y / height), mode="lines",
                       name="exact", line={"color": "#111", "width": 2, "dash": "dash"}),
            go.Scatter(x=y, y=across.values[index], mode="lines+markers", name="SPH",
                       line={"color": "#d62828", "width": 2}, marker={"size": 6}),
            go.Scatter(x=x, y=along.values[index], mode="lines+markers", name="SPH", showlegend=False,
                       line={"color": "#d62828", "width": 2}, marker={"size": 6}, xaxis="x2", yaxis="y2"),
        ]

    picks = np.unique(np.linspace(0, len(times) - 1, min(80, len(times)), dtype=int))
    figure = make_subplots(
        rows=2, cols=2, vertical_spacing=0.22, horizontal_spacing=0.12,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("Shear profile u_x(y)", "Compressive wave u_x(x)", "Amplitude of the shear mode"),
    )
    figure.add_trace(traces(0)[0], row=1, col=1)
    figure.add_trace(traces(0)[1], row=1, col=1)
    figure.add_trace(traces(0)[2], row=1, col=2)
    figure.add_scatter(x=times, y=exact_shear, mode="lines", name="exact decay", showlegend=False,
                       line={"color": "#111", "width": 2, "dash": "dash"}, row=2, col=1)
    figure.add_scatter(x=times, y=np.abs(shear), mode="lines", name="SPH", showlegend=False,
                       line={"color": "#d62828", "width": 2}, row=2, col=1)
    figure.frames = [go.Frame(name=f"{times[i]:.3f}", data=traces(i), traces=[0, 1, 2]) for i in picks]
    figure.update_layout(
        title="Incompressible SPH: the pressure projection and viscous shear decay", template="plotly_white",
        margin={"l": 70, "r": 30, "t": 90, "b": 140}, legend={"orientation": "h", "y": 1.1},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.12,
                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 60, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}]}]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.07, "currentvalue": {"prefix": "t = "},
                  "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}], "label": frame.name, "method": "animate"} for frame in figure.frames]}],
    )
    limit = 1.1 * max(shear_amplitude, compressive_amplitude)
    figure.update_xaxes(title_text="y", range=[0, height], row=1, col=1)
    figure.update_yaxes(title_text="u_x", range=[-0.05, limit], row=1, col=1)
    figure.update_xaxes(title_text="x", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="u_x", range=[-limit, limit], row=1, col=2)
    figure.update_xaxes(title_text="t", row=2, col=1)
    figure.update_yaxes(title_text="amplitude", type="log", row=2, col=1)
    save(figure, "incompressible-shear-relaxation", width=1000, height=850, show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the incompressible shear relaxation example.")
    argparser.add_argument(
        "--pproc-only",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    argparser.add_argument("--show", action="store_true", help="Show the figures before saving them.")
    args = argparser.parse_args()

    simulation = create_simulation()
    if not args.pproc_only:
        simulation.run()
    pproc(simulation, show=args.show)
