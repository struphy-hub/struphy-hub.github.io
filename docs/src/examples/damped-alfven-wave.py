"""A resistively damped Alfvén wave, with Struphy's ViscoResistiveMHD model.

A transverse velocity mode u_y = U sin(k x) in a plasma with a uniform field B0 along x splits into two Alfvén waves that
run in opposite directions and make a standing wave, oscillating at omega = v_A k with v_A = B0 / sqrt(rho). Finite
resistivity eta lets the field lines slip through the plasma, and the wave amplitude decays as exp(-gamma t) with
gamma = eta k^2 / 2. (Viscosity would add nu k^2 / 2; it is left out because Struphy's viscosity propagator does not run in this
configuration.) A scan over three resistivities compares the decay with the exact rate.

Requires Struphy 3.3 with compiled kernels (`struphy compile`).
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

length = 2 * np.pi
wavenumber = 2 * np.pi / length  # one wavelength in the box
b0 = 1.0
alfven_speed = b0  # the density is 1
frequency = alfven_speed * wavenumber
amplitude = 0.05
resistivities = (0.05, 0.1, 0.2)  # the scan; the example itself is the eta = 0.1 run


class CompatibleNonlinearSolverParameters(NonlinearSolverParameters):
    """Bridge Struphy code paths that use both attribute and mapping access."""

    def __getitem__(self, key):
        return getattr(self, key)


def make_simulation(eta, folder, *, time_opts, domain, equil, grid, derham_opts, **extra):
    """Resistive MHD with resistivity eta, a uniform field along x and a transverse velocity mode."""
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
    model.em_fields.b_field.add_background(FieldsBackground(values=(b0, 0.0, 0.0)))
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(ls=(1,), amps=(amplitude,), Lx=length, comp=1, given_in_basis="physical")
    )
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder=folder),
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        **extra,
    )


def create_simulation() -> Simulation:
    time_opts = Time(dt=0.05, Tend=20.0, split_algo="Strang")
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    domain = domains.Cuboid(l1=0.0, r1=length, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    equil = equils.HomogenSlab(B0x=b0, n0=1.0, beta=2.0)
    simulation = make_simulation(
        0.1,
        "damped_alfven_wave",
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        name="Resistively damped Alfvén wave",
        description=(
            "A standing Alfvén wave in a resistive plasma oscillates at the Alfvén frequency while its amplitude decays "
            "at the rate η k² / 2. Struphy's nonlinear resistive MHD is run at three resistivities and compared with "
            "the exact rate."
            r" The initial fields are $$\mathbf{u}(x,0)=(0,0.05\sin x,0),\qquad \mathbf{B}(x,0)=(1,0,0),$$"
            r" with :math:`n(x,0)=1` on :math:`0\le x<2\pi`; the scan uses :math:`\eta\in\{0.05,0.1,0.2\}`."
        ),
    )
    return simulation

def pproc(sim: Simulation):

    from plotly.subplots import make_subplots

    from _gallery import export_profiling, is_root, merge_metadata, save_extra_figure, save_figure

    runs = {0.1: sim.output}
    for eta in resistivities:
        if eta != 0.1:
            runs[eta] = make_simulation(
                eta,
                f"damped_alfven_wave_eta{eta}",
                time_opts=sim.time_opts,
                domain=sim.domain,
                equil=sim.equil,
                grid=sim.grid,
                derham_opts=sim.derham_opts,
            ).output
    for run in runs.values():
        run.pproc(physical=True)

    amplitudes, fitted, measured_frequency = {}, {}, {}
    for eta, run in runs.items():
        times, x, values = velocity_profile(run)
        amplitudes[eta] = mode_amplitude(x, values)
        peak_times, peak_values = envelope_peaks(times, amplitudes[eta])
        fitted[eta] = float(-np.polyfit(peak_times, np.log(peak_values), 1)[0])
        crossings = np.where(np.diff(np.sign(amplitudes[eta])) != 0)[0]
        measured_frequency[eta] = float(np.pi / np.mean(np.diff(times[crossings])))
    exact_rate = {eta: eta * wavenumber**2 / 2 for eta in runs}
    if not all(np.isfinite(list(fitted.values()) + list(measured_frequency.values()))):
        raise RuntimeError("A decay rate or frequency could not be fitted")
    if is_root():
        for eta in runs:
            print(f"eta = {eta}: damping rate {fitted[eta]:.4f} (exact {exact_rate[eta]:.4f}), "
                  f"frequency {measured_frequency[eta]:.4f} (v_A k = {frequency:.4f})")

    eta_main = 0.1
    run = runs[eta_main]
    times, x, values = velocity_profile(run)
    gamma = exact_rate[eta_main]
    total = np.asarray(run.scalars["en_tot"])
    energy_drift = float(np.max(np.abs(total / total[0] - 1.0)))
    print(f"Maximum relative drift of the total energy: {energy_drift:.2e}")

    def profile_traces(index):
        envelope = amplitude * np.exp(-gamma * times[index]) * np.sin(wavenumber * x)
        return [
            go.Scatter(x=x, y=envelope, mode="lines", name="envelope ± U exp(−γt)", line={"color": "#111", "width": 1.5, "dash": "dash"}),
            go.Scatter(x=x, y=-envelope, mode="lines", showlegend=False, line={"color": "#111", "width": 1.5, "dash": "dash"}),
            go.Scatter(x=x, y=values[index], mode="lines", name="Struphy", line={"color": "#d62828", "width": 3}),
        ]

    picks = np.unique(np.linspace(0, len(times) - 1, min(100, len(times)), dtype=int))
    figure = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                           subplot_titles=("Transverse velocity u_y along the field", "Amplitude of the mode"))
    for trace in profile_traces(0):
        figure.add_trace(trace, row=1, col=1)
    figure.add_scatter(x=times, y=amplitudes[eta_main], mode="lines", name="amplitude", showlegend=False,
                       line={"color": "#d62828", "width": 2}, row=2, col=1)
    for sign in (1, -1):
        figure.add_scatter(x=times, y=sign * amplitude * np.exp(-gamma * times), mode="lines", showlegend=False,
                           line={"color": "#111", "width": 1.5, "dash": "dash"}, row=2, col=1)
    figure.frames = [go.Frame(name=f"{times[i]:.2f}", data=profile_traces(i), traces=[0, 1, 2]) for i in picks]
    figure.update_layout(
        title="A resistively damped standing Alfvén wave", template="plotly_white",
        margin={"l": 70, "r": 30, "t": 150, "b": 140}, title_y=0.97, legend={"orientation": "h", "y": 1.13},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.12,
                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 60, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}]}]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.07, "currentvalue": {"prefix": "t = "},
                  "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}], "label": frame.name, "method": "animate"} for frame in figure.frames]}],
    )
    figure.update_xaxes(title_text="x", range=[0, length], row=1, col=1)
    figure.update_yaxes(title_text="u_y", range=[-1.2 * amplitude, 1.2 * amplitude], row=1, col=1)
    figure.update_xaxes(title_text="t", row=2, col=1)
    figure.update_yaxes(title_text="amplitude of sin(kx)", row=2, col=1)
    save_figure(figure, "damped-alfven-wave", width=900, height=850)

    colors = {0.05: "#168aad", 0.1: "#d62828", 0.2: "#f77f00"}
    decay = go.Figure()
    for eta, amp in amplitudes.items():
        peak_times, peak_values = envelope_peaks(times, amp)
        decay.add_scatter(x=peak_times, y=peak_values, mode="markers", name=f"η = {eta}: peaks of Struphy",
                          marker={"color": colors[eta], "size": 9})
        decay.add_scatter(x=times, y=amplitude * np.exp(-exact_rate[eta] * times), mode="lines",
                          name=f"η = {eta}: exp(−η k² t / 2)", line={"color": colors[eta], "width": 1.5, "dash": "dash"})
    decay.update_layout(
        title="Decay of the wave amplitude", template="plotly_white", autosize=True,
        xaxis_title="t", yaxis_title="amplitude of the sin(kx) mode", yaxis_type="log",
        margin={"l": 75, "r": 30, "t": 80, "b": 60},
    )
    figures = [
        save_extra_figure(
            decay, "damped-alfven-wave", "decay",
            alt="Peaks of the amplitude of a damped Alfvén wave at three resistivities, on the exact exponentials",
            caption=(
                "The successive amplitude peaks of the wave at three resistivities, on log axes where the exact decay "
                "exp(−η k² t / 2) is a straight line. The fitted rates differ from η k² / 2 by "
                + ", ".join(f"{100 * (fitted[eta] / exact_rate[eta] - 1):+.1f} % (η = {eta})" for eta in runs)
                + "."
            ),
        ),
    ]

    merge_metadata(
        "damped-alfven-wave",
        wavenumber=wavenumber,
        alfvenFrequency=frequency,
        dampingRates={str(eta): {"measured": fitted[eta], "exact": exact_rate[eta]} for eta in runs},
        measuredFrequencies={str(eta): measured_frequency[eta] for eta in runs},
        maxEnergyDrift=energy_drift,
        figures=figures,
        **export_profiling(sim, "damped-alfven-wave"),
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
