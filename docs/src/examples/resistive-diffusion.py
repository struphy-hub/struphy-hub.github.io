"""Resistive diffusion of a magnetic field, with Struphy's ViscoResistiveMHD model.

In a resistive plasma at rest the magnetic field obeys a diffusion equation, dB/dt = eta B'', where eta is the resistivity.
A sinusoidal field B_z = B_0 sin(k x) therefore decays without changing its shape, as exp(-eta k^2 t), and the magnetic energy
decays twice as fast, as exp(-2 eta k^2 t). Struphy's variational discretization turns the lost magnetic energy into
thermal energy (Ohmic heating) and keeps the total energy constant to the accuracy of its nonlinear solver. The field is weak, so that
the pressure gradient it produces sets only a negligible flow. A scan over three resistivities confirms the rate.

Install dependencies and compile the kernels (Python 3.10 or newer):

    pip install "struphy[pproc]>=3.4.0"
    pip install "plasma-plots[plotly]>=0.1.1"
    struphy compile

PNG exports require Chrome or Chromium. If Chrome is not installed, run:

    kaleido_get_chrome

Save this file and run it from the directory where you want the output:

    python resistive-diffusion.py

Figures are saved in the current directory; add --show to display them before saving.
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.linear_algebra.solver import NonlinearSolverParameters
from struphy.models import ViscoResistiveMHD
from plasma_plots import save_figure

length = 2 * np.pi
mode_number = 2
wavenumber = 2 * np.pi * mode_number / length
amplitude = 0.1
resistivities = (0.05, 0.1, 0.2)  # the scan; the example itself is the eta = 0.1 run


class CompatibleNonlinearSolverParameters(NonlinearSolverParameters):
    """Bridge Struphy code paths that use both attribute and mapping access."""

    def __getitem__(self, key):
        return getattr(self, key)


def create_simulation(eta=0.1, folder="resistive_diffusion") -> Simulation:
    """The resistive-MHD model with resistivity eta and a sine mode in the out-of-plane field B_z."""
    time_opts = Time(dt=0.05, Tend=8.0, split_algo="LieTrotter")
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    domain = domains.Cuboid(l1=0.0, r1=length, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    equil = equils.HomogenSlab(B0z=1.0, n0=1.0, beta=2.0)
    model = ViscoResistiveMHD(with_viscosity=False, with_resistivity=True)
    model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="full")
    model.propagators.variat_resist.options = model.propagators.variat_resist.Options(
        model="full",
        eta=eta,
        nonlin_solver=CompatibleNonlinearSolverParameters(type="Newton"),
    )
    # Logical 3-forms include det(DF) = length, giving a physical density 1 and a uniform thermal pressure.
    volume = length
    model.mhd.density.add_background(FieldsBackground(values=(volume,)))
    model.mhd.entropy.add_background(FieldsBackground(values=(0.6 * volume,)))
    model.mhd.velocity.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
    model.em_fields.b_field.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesSin(ls=(mode_number,), amps=(amplitude,), Lx=length, comp=2, given_in_basis="physical")
    )
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder=folder),
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        name="Resistive diffusion of a magnetic field",
        description=(
            "A sinusoidal magnetic field in a resistive plasma at rest decays without changing shape, at the rate "
            "η k². Struphy's variational discretization converts the lost magnetic energy into thermal energy and keeps "
            "the total energy constant."
            r" The initial state is $$\mathbf{B}(x,0)=(0,0,0.1\sin(2x)),\qquad \mathbf{u}(x,0)=0,\qquad n(x,0)=1,$$"
            r" on :math:`0\le x<2\pi`, with resistivities :math:`\eta\in\{0.05,0.1,0.2\}`."
        ),
    )


def field_profile(run):
    """Time, position and B_z(t, x) of a post-processed run."""
    b_z = run.evaluate("em_fields/b_field_xyz").isel(component=2, eta2=0, eta3=0)
    return b_z.t.values, b_z.eta1.values * length, b_z.values


def mode_amplitude(run):
    """Amplitude of the sin(k x) mode of B_z over time, from a post-processed run."""
    b_z = run.evaluate("em_fields/b_field_xyz").isel(component=2, eta2=0, eta3=0)
    return b_z.plasma.analysis.project_mode(dim="eta1", number=mode_number)


def pproc(sim: Simulation, show: bool = False):
    from plotly.subplots import make_subplots

    runs = {0.1: sim.output}
    for eta in resistivities:
        if eta != 0.1:
            runs[eta] = create_simulation(eta, f"resistive_diffusion_eta{eta}").output
    for run in runs.values():
        run.pproc(physical=True)

    amplitudes, fitted = {}, {}
    for eta, run in runs.items():
        amplitudes[eta] = abs(mode_amplitude(run))
        amplitudes[eta].attrs = {"label": f"η = {eta}: Struphy"}
        fitted[eta] = -amplitudes[eta].plasma.analysis.growth_rate().rate
    exact_rate = {eta: eta * wavenumber**2 for eta in runs}
    if not all(np.isfinite(list(fitted.values()))):
        raise RuntimeError("A decay rate could not be fitted")
    for eta in runs:
        print(f"eta = {eta}: decay rate {fitted[eta]:.4f} (exact {exact_rate[eta]:.4f})")

    run = runs[0.1]
    eta_main = 0.1
    times, x, values = field_profile(run)
    magnetic = np.asarray(run.scalars["en_mag"])
    thermal = np.asarray(run.scalars["en_thermo"])
    total = np.asarray(run.scalars["en_tot"])
    scalar_times = np.asarray(run.time)[: len(total)]
    energy_drift = float(run.scalars["en_tot"].plasma.analysis.relative_error().max())
    print(f"Maximum relative drift of the total energy: {energy_drift:.2e}")

    def profile_traces(index):
        decay = np.exp(-exact_rate[eta_main] * times[index])
        return [
            go.Scatter(x=x, y=values[index], mode="lines", name="Struphy", line={"color": "#d62828", "width": 3}),
            go.Scatter(x=x, y=amplitude * decay * np.sin(wavenumber * x), mode="lines", name="exact",
                       line={"color": "#111", "width": 2, "dash": "dash"}),
        ]

    picks = np.unique(np.linspace(0, len(times) - 1, min(80, len(times)), dtype=int))
    figure = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                           subplot_titles=("Magnetic field B_z", "Energy: magnetic energy becomes heat"))
    for trace in profile_traces(0):
        figure.add_trace(trace, row=1, col=1)
    figure.add_scatter(x=scalar_times, y=(thermal - thermal[0]) / (magnetic[0] - magnetic[-1]), mode="lines",
                       name="thermal energy gained", line={"color": "#f77f00", "width": 2}, row=2, col=1)
    figure.add_scatter(x=scalar_times, y=(magnetic - magnetic[0]) / (magnetic[0] - magnetic[-1]), mode="lines",
                       name="magnetic energy lost", line={"color": "#168aad", "width": 2}, row=2, col=1)
    figure.add_scatter(x=scalar_times, y=(total - total[0]) / (magnetic[0] - magnetic[-1]), mode="lines",
                       name="total energy change", line={"color": "#111", "width": 2, "dash": "dot"}, row=2, col=1)
    figure.frames = [go.Frame(name=f"{times[i]:.2f}", data=profile_traces(i), traces=[0, 1]) for i in picks]
    figure.update_layout(
        title="Resistive decay of a magnetic field", template="plotly_white",
        margin={"l": 70, "r": 30, "t": 150, "b": 140}, title_y=0.97, legend={"orientation": "h", "y": 1.13},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.12,
                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 60, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}]}]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.07, "currentvalue": {"prefix": "t = "},
                  "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}], "label": frame.name, "method": "animate"} for frame in figure.frames]}],
    )
    figure.update_xaxes(title_text="x", range=[0, length], row=1, col=1)
    figure.update_yaxes(title_text="B_z", range=[-1.2 * amplitude, 1.2 * amplitude], row=1, col=1)
    figure.update_xaxes(title_text="t", row=2, col=1)
    figure.update_yaxes(title_text="energy change / magnetic energy lost", row=2, col=1)
    save_figure(figure, "resistive-diffusion", width=900, height=850, show=show)

    first, *others = amplitudes.values()
    decay = first.plasma.plot.timeseries(
        *others,
        logy=True,
        reference={f"η = {eta}: exp(−η k² t)": lambda t, eta=eta: amplitude * np.exp(-exact_rate[eta] * t) for eta in runs},
        title="Decay of the field amplitude",
        backend="plotly",
    )
    decay.fig.update_yaxes(title_text="amplitude of the sin(kx) mode")
    save_figure(decay, "resistive-diffusion-decay", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the resistive diffusion example.")
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
        for eta in resistivities:
            if eta != 0.1:
                create_simulation(eta, f"resistive_diffusion_eta{eta}").run()
    pproc(simulation, show=args.show)
