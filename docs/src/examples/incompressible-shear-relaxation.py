"""Shear relaxation of an incompressible fluid between no-slip walls, with incompressible SPH.

A viscous fluid fills a channel with no-slip walls at y = 0 and y = H. Its velocity starts as the
shear mode u_x = U sin(pi y / H), plus a compressive wave u_x = e sin(2 pi x) along the channel. Struphy's
`IncompressibleNavierStokesSPH` moves the particles with the fluid and, after every step, projects the
velocity onto a divergence-free field (Chorin's projection: a Poisson solve for the pressure). The
projection removes the compressive wave at once, and the shear mode, which is already divergence-free,
decays by viscous diffusion: u_x(y, t) = U sin(pi y / H) exp(-mu (pi / H)^2 t).

Based on the verification test of the model in Struphy
(models/tests/verification/test_verif_IncompressibleNavierStokesSPH.py).

Requires Struphy 3.3 with compiled kernels (`struphy compile`).
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


# The pressure is a finite element field; the walls are free (no boundary condition) for it.


def create_simulation() -> Simulation:
    model = IncompressibleNavierStokesSPH(with_B0=False, with_viscosity=True)
    model.propagators.push_eta.options = model.propagators.push_eta.Options(
        butcher=ButcherTableau(algo="forward_euler"),
    )
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(
        kernel_type="gaussian_2d", mu=viscosity
    )
    domain = domains.Cuboid(r1=1.0, r2=height)
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
    simulation = Simulation(
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
    return simulation

def pproc(sim: Simulation):

    from plotly.subplots import make_subplots

    from _gallery import export_profiling, is_root, merge_metadata, save_figure

    output = sim.output
    output.pproc()

    across = output.evaluate("fluid/e2_current_1/f")  # (t, e2): u_x against y, averaged over x
    along = output.evaluate("fluid/e1_current_1/f")  # (t, e1): u_x against x, averaged over y
    times = across.t.values
    y, x = across.e2.values * height, along.e1.values
    if not (np.isfinite(across.values).all() and np.isfinite(along.values).all()):
        raise RuntimeError("Non-finite SPH result: refusing to publish the run")

    # The amplitude of each mode, projected on it. A bin average lowers a mode by sinc(k dx / 2).
    shear_bins = np.sinc(np.pi / height * (height / bins) / 2 / np.pi)
    wave_bins = np.sinc(2 * np.pi * (1.0 / bins) / 2 / np.pi)
    shear = 2 * np.mean(across.values * np.sin(np.pi * y / height), axis=1) / shear_bins
    wave = 2 * np.mean(along.values * np.sin(2 * np.pi * x), axis=1) / wave_bins
    exact_shear = shear_amplitude * np.exp(-decay_rate * times)
    fitted_rate = float(-np.polyfit(times, np.log(np.abs(shear)), 1)[0])
    profile_error = float(np.max(np.abs(across.values[-1] - exact_shear[-1] * np.sin(np.pi * y / height))))
    wave_left = float(np.abs(wave[times >= 0.1]).max() / compressive_amplitude)
    if is_root():
        print(f"shear decay rate {fitted_rate:.3f} (exact {decay_rate:.3f}); compressive wave after t = 0.1: "
              f"{100 * wave_left:.1f}% of its initial amplitude; profile error at the end {profile_error:.4f}")

    dense_y = np.linspace(0.0, height, 201)
    dense_x = np.linspace(0.0, 1.0, 201)

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
    save_figure(figure, "incompressible-shear-relaxation", width=1000, height=850)

    merge_metadata(
        "incompressible-shear-relaxation",
        viscosity=viscosity, exactDecayRate=decay_rate, measuredDecayRate=fitted_rate,
        profileError=profile_error, compressiveWaveRemaining=wave_left,
        markers=boxes * boxes * markers_per_box, **export_profiling(sim, "incompressible-shear-relaxation"),
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the example.")
    argparser.add_argument("--pproc", action="store_true", help="Run post-processing on an existing simulation instead of running a new one.")
    args = argparser.parse_args()
    simulation = create_simulation()
    if args.pproc:
        pproc(simulation)
    else:
        simulation.run(profiling_activated=True)
        pproc(simulation)
