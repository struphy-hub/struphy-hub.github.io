"""Langmuir waves and their Landau damping at several wavenumbers, with Struphy's VlasovAmpereOneSpecies model.

A small electrostatic density perturbation cos(k x) in a uniform Maxwellian plasma (thermal speed 1, plasma frequency 1) oscillates as a Langmuir wave, at a
frequency above the plasma frequency because of the thermal motion, while it is damped by Landau damping. The complex
frequency omega(k) = omega_r + i gamma is the root of the Vlasov dispersion relation

    1 + (1 + zeta Z(zeta)) / k^2 = 0,   zeta = omega / (sqrt(2) k),

with the plasma dispersion function Z. Four runs with wavenumbers between 0.3 and 0.6 (different box lengths) measure the oscillation
frequency and the damping rate of the electric field, and compare them with this root. The fluid estimate omega^2 = 1 + 3 k^2 (Bohm-Gross) is
shown for reference: it ignores the kinetic effects and misses the frequency by several per cent already at k = 0.5.

Requires Struphy 3.3 with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import (
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
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import VlasovAmpereOneSpecies

wavenumbers = (0.3, 0.4, 0.5, 0.6)  # the scan; the example itself is the k = 0.5 run
amplitude = 0.001


def create_simulation(k=0.5, folder="langmuir_wave_dispersion") -> Simulation:
    """Vlasov-Ampère in a periodic box of length 2 pi / k with one cosine mode in the density."""
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    time_opts = Time(dt=0.05, Tend=20.0, split_algo="LieTrotter")
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
    model.em_fields.e_field.save_data = True
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=1000),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=2.0,
    )
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    mode = perturbations.ModesCos(amps=(amplitude,), ls=(1,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, mode)))
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder=folder),
        time_opts=time_opts,
        domain=domains.Cuboid(r1=2 * np.pi / k),
        grid=grid,
        derham_opts=derham_opts,
        name="Langmuir wave dispersion",
        description=(
            "A density perturbation in a uniform Maxwellian plasma oscillates as a Langmuir wave and is Landau damped. "
            "Runs at four wavenumbers give the oscillation frequency and the damping rate against k, and are compared with "
            "the root of the kinetic dispersion relation and with the fluid Bohm–Gross estimate."
            r" Each run starts from a zero-drift, unit-thermal-speed Maxwellian with $$n(x,0)=1+10^{-3}\cos(kx),\qquad L_x=2\pi/k,$$"
            r" for :math:`k\in\{0.3,0.4,0.5,0.6\}`."
        ),
    )


def mode_amplitude(run):
    """The amplitude of the sin(k x) mode of the electric field E_x over time, from a post-processed run."""
    cells = run.grid.num_elements[0]
    e_x = run.evaluate(
        "em_fields/e_field", eta1=np.linspace(0.0, 1.0, cells + 1), eta2=0.0, eta3=0.0, representation="1"
    ).isel(component=0)
    return e_x.struphy.analysis.project_mode(dim="eta1", number=1)


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
    runs = {0.5: sim.output}
    for k in wavenumbers:
        if k != 0.5:
            runs[k] = create_simulation(k, f"langmuir_wave_dispersion_k{k}").output
    for run in runs.values():
        run.pproc()

    from struphy_plots.theory.kinetic import bohm_gross, langmuir

    exact, measured_frequency, measured_rate, amplitudes = {}, {}, {}, {}
    for k in sorted(runs):
        # The complex Langmuir frequency omega_r + i gamma, the root of the Vlasov dispersion relation.
        omega = langmuir(k)
        exact[k] = (omega.real, omega.imag)
        mode = mode_amplitude(runs[k])
        times, values = mode.t.values, mode.values
        amplitudes[k] = values
        # The first time units hold a transient of phase-mixing modes that are not the Langmuir wave, so the fit starts at t = 1
        # and stops at t = 12. The measured values depend on this window at the level of a few per cent.
        window = (times > 1.0) & (times < 12.0)
        in_window = np.flatnonzero(window)
        crossings = in_window[:-1][np.diff(np.sign(values[in_window])) != 0]
        roots = times[crossings] - values[crossings] * (times[crossings + 1] - times[crossings]) / (
            values[crossings + 1] - values[crossings]
        )
        measured_frequency[k] = float(np.pi / np.mean(np.diff(roots)))
        measured_rate[k] = abs(mode).struphy.analysis.damping_rate(window=(1.0, 12.0)).rate
    if not (np.isfinite(list(measured_frequency.values())).all() and np.isfinite(list(measured_rate.values())).all()):
        raise RuntimeError("A frequency or a damping rate could not be measured")
    for k in sorted(runs):
        print(f"k = {k}: omega_r = {measured_frequency[k]:.4f} (kinetic {exact[k][0]:.4f}), "
              f"gamma = {measured_rate[k]:.4f} (kinetic {exact[k][1]:.4f})")

    from plotly.subplots import make_subplots

    k_line = np.linspace(0.25, 0.65, 60)
    lines = langmuir(k_line)
    ks = sorted(runs)
    figure = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                           subplot_titles=("Oscillation frequency", "Damping rate"))
    figure.add_scatter(x=k_line, y=bohm_gross(k_line).real, mode="lines", name="Bohm–Gross √(1 + 3k²)",
                       line={"color": "#888", "width": 1.5, "dash": "dot"}, row=1, col=1)
    figure.add_scatter(x=k_line, y=lines.real, mode="lines", name="kinetic dispersion relation",
                       line={"color": "#111", "width": 2, "dash": "dash"}, row=1, col=1)
    figure.add_scatter(x=ks, y=[measured_frequency[k] for k in ks], mode="markers", name="Struphy (PIC)",
                       marker={"color": "#d62828", "size": 11}, row=1, col=1)
    figure.add_scatter(x=k_line, y=lines.imag, mode="lines", showlegend=False,
                       line={"color": "#111", "width": 2, "dash": "dash"}, row=1, col=2)
    figure.add_scatter(x=ks, y=[measured_rate[k] for k in ks], mode="markers", showlegend=False,
                       marker={"color": "#d62828", "size": 11}, row=1, col=2)
    figure.update_layout(
        title="Langmuir waves: frequency and Landau damping against wavenumber", template="plotly_white",
        margin={"l": 70, "r": 30, "t": 110, "b": 70}, legend={"orientation": "h", "y": 1.2},
    )
    figure.update_xaxes(title_text="wavenumber k λ_D")
    figure.update_yaxes(title_text="ω_r / ω_p", row=1, col=1)
    figure.update_yaxes(title_text="γ / ω_p", row=1, col=2)
    save(figure, "langmuir-wave-dispersion", width=1200, height=560, show=show)

    colors = {0.3: "#168aad", 0.4: "#2a9d8f", 0.5: "#f77f00", 0.6: "#d62828"}
    signals = go.Figure()
    for k in ks:
        times = mode_amplitude(runs[k]).t.values
        # The exact envelope exp(gamma t), scaled to the first peak of the wave (after the initial transient).
        after = times > 1.0
        first_peak = np.argmax(np.abs(amplitudes[k][after][:40]))
        t_peak = times[after][first_peak]
        scale = abs(amplitudes[k][after][first_peak]) * np.exp(-exact[k][1] * t_peak)
        signals.add_scatter(x=times, y=amplitudes[k], mode="lines", name=f"k = {k}", line={"color": colors[k], "width": 2})
        for sign in (1, -1):
            signals.add_scatter(x=times[after], y=sign * scale * np.exp(exact[k][1] * times[after]), mode="lines",
                                showlegend=False, line={"color": colors[k], "width": 1, "dash": "dash"})
    signals.update_layout(
        title="Electric field of the Langmuir wave", template="plotly_white", autosize=True,
        xaxis_title="t [1/ω_p]", yaxis_title="amplitude of the sin(kx) mode of E_x",
        margin={"l": 75, "r": 30, "t": 80, "b": 60},
    )
    save(signals, "langmuir-wave-dispersion-signals", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the langmuir wave dispersion example.")
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
        for k in wavenumbers:
            if k != 0.5:
                create_simulation(k, f"langmuir_wave_dispersion_k{k}").run()
    pproc(simulation, show=args.show)
