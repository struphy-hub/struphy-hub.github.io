"""Ordinary electromagnetic waves and the plasma-frequency cutoff.

For k perpendicular to B0 and E parallel to B0, the cold-plasma O mode obeys
omega**2 = omega_p**2 + c**2*k**2. Four standing modes are evolved together;
their frequencies are measured independently from electric-field zero crossings.
This periodic initial-value experiment measures propagating modes, not reflection
from an interface. The cutoff is the k -> 0 limit of the dispersion relation.

Reference: https://farside.ph.utexas.edu/teaching/315/Waveshtml/node75.html
Requires the pinned Struphy with compiled kernels (`struphy compile`).
"""

import numpy as np

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, perturbations
from struphy.models import ColdPlasma
from struphy.linear_algebra.solver import SolverParameters

stem = "ordinary-mode-dispersion"
length = 8.0 * np.pi
mode_numbers = (1, 2, 4, 8)
amplitude = 0.02  # electric-field amplitude per mode
wavenumbers = 2.0 * np.pi * np.array(mode_numbers) / length
frequencies = np.sqrt(1.0 + wavenumbers**2)  # c = omega_p = 1
model = ColdPlasma(alpha=1.0, epsilon=1.0)
for propagator in (model.propagators.maxwell, model.propagators.ohm, model.propagators.jxb):
    propagator.options = propagator.Options(solver_params=SolverParameters(tol=1e-12))
model.em_fields.e_field.add_perturbation(
    perturbations.ModesCos(ls=mode_numbers, amps=(amplitude,) * len(mode_numbers),
                           comp=2, Lx=length, given_in_basis="physical"),
)
domain = domains.Cuboid(r1=length)
grid = grids.TensorProductGrid(num_elements=(64, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.02, Tend=30.0, split_algo="Strang")
sim = Simulation(
    model=model, name="Ordinary waves and the plasma cutoff",
    description=(
        "Four ordinary electromagnetic waves oscillate across a uniform magnetic field. "
        "Their electric field is parallel to the background field, so the electrons feel no "
        "magnetic force in this polarization. Measured frequencies follow the cold-plasma dispersion relation"
        r" $$\omega^2=\omega_p^2+c^2k^2,\qquad \omega_p=c=1.$$"
        r" Initially :math:`E_z(x,0)=0.02\sum_{n\in\{1,2,4,8\}}\cos(nx/4)` and the magnetic perturbation and current vanish. "
        "The dispersion curve approaches the plasma-frequency cutoff as the wavelength grows; "
        "below it, a uniform cold plasma has no propagating ordinary mode."
    ),
    env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="ordinary_mode_dispersion"),
    time_opts=time_opts, domain=domain, grid=grid, derham_opts=derham_opts,
    equil=equils.HomogenSlab(B0z=1.0, n0=1.0),
)


if __name__ == "__main__":
    from plotly.subplots import make_subplots
    from _gallery import export_profiling, merge_metadata, save_extra_figure, save_figure, space_time_figure

    output = sim.run(profiling_activated=True)
    output.pproc(physical=True)
    field = output.evaluate("em_fields/e_field_xyz").isel(component=2, e2=0, e3=0)
    energy = output.evaluate("total_energy")
    times, x = field.t.values, field.e1.values * length
    if not np.isfinite(field.values).all() or not np.isfinite(energy.values).all() or not np.isclose(times[-1], time_opts.Tend):
        raise RuntimeError("The ordinary-mode run is incomplete or non-finite")
    # Drop the repeated endpoint of the periodic spatial evaluation grid.
    basis = np.cos(wavenumbers[:, None] * x[None, :-1])
    signals = 2.0 * field.values[:, :-1] @ basis.T / (len(x) - 1)
    measured = []
    for signal in signals.T:
        crossing = np.flatnonzero(np.diff(np.signbit(signal)))
        if crossing.size < 4:
            raise RuntimeError("Too few zero crossings to measure an ordinary-mode frequency")
        roots = times[crossing] - signal[crossing] * np.diff(times)[crossing] / np.diff(signal)[crossing]
        measured.append(float(np.pi / np.mean(np.diff(roots))))
    measured = np.array(measured)
    reference = amplitude * np.cos(times[:, None] * frequencies[None, :])
    frequency_error = float(np.max(np.abs(measured / frequencies - 1.0)))
    mode_error = float(np.max(np.abs(signals - reference)) / amplitude)
    energy_drift = float(np.max(np.abs(energy.values / energy.values[0] - 1.0)))
    if frequency_error > 0.01 or mode_error > 0.1 or energy_drift > 1e-6:
        raise RuntimeError(f"Ordinary-mode check failed: frequency={frequency_error:.3g}, field={mode_error:.3g}, energy={energy_drift:.3g}")

    figure = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                           subplot_titles=("Dispersion and cutoff", "Four independently measured modes"))
    k = np.linspace(0, 2.2, 200)
    figure.add_scatter(x=k, y=np.sqrt(1 + k**2), name="ω² = 1 + k²", line={"color": "#222"}, row=1, col=1)
    figure.add_scatter(x=wavenumbers, y=measured, mode="markers", name="Measured frequencies",
                       marker={"size": 10, "color": "#d62828"}, row=1, col=1)
    figure.add_hrect(y0=0, y1=1, fillcolor="#168aad", opacity=0.08, line_width=0, row=1, col=1)
    figure.add_hline(y=1, line_dash="dot", annotation_text="ω_p: cutoff", row=1, col=1)
    for j, color in enumerate(("#168aad", "#d62828", "#f77f00", "#6a4c93")):
        figure.add_scatter(x=times, y=signals[:, j] / amplitude + 2.5 * j, name=f"k = {wavenumbers[j]:g}",
                           line={"color": color}, row=1, col=2)
        figure.add_scatter(x=times[::25], y=reference[::25, j] / amplitude + 2.5 * j, mode="markers",
                           name="Exact oscillation", showlegend=j == 0,
                           marker={"color": "#222", "symbol": "circle-open", "size": 4}, row=1, col=2)
    figure.update_xaxes(title_text="k c / ω_p", row=1, col=1)
    figure.update_yaxes(title_text="ω / ω_p", range=[0, 2.5], row=1, col=1)
    figure.update_xaxes(title_text="t ω_p", row=1, col=2)
    figure.update_yaxes(title_text="E-mode / A + vertical offset", row=1, col=2)
    figure.update_layout(title="Ordinary electromagnetic waves in a cold plasma", template="plotly_white",
                         legend={"orientation": "h", "y": -0.2}, margin={"l": 70, "r": 30, "t": 100, "b": 140})
    save_figure(figure, stem, height=650)
    space_time = space_time_figure(field, space="e1", x_values=x, xaxis_title="x",
                                   title="Superposed ordinary waves: E_z(x, t)", colorbar_title="E_z")
    figures = [save_extra_figure(space_time, stem, "space-time", alt="Space-time map of four superposed ordinary electromagnetic waves",
                                caption="The longer waves oscillate near the plasma frequency; shorter waves oscillate faster.")]
    merge_metadata(stem, measuredFrequencies=measured.tolist(), exactFrequencies=frequencies.tolist(),
                   maxFrequencyError=frequency_error, maxModeError=mode_error, maxEnergyDrift=energy_drift,
                   figures=figures, **export_profiling(sim, stem))
