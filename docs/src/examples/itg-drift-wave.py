"""ITG drift wave: a temperature-gradient-driven instability in a cylinder.

A magnetized plasma column with radial density and temperature gradients is
unstable to drift waves: a small helical density perturbation taps the free
energy in the ion-temperature gradient (ITG) and grows exponentially,
extracting energy from the background profiles -- the basic mechanism behind
ITG turbulence, one of the dominant sources of turbulent transport in
magnetically confined fusion plasmas.

Adapted from Struphy's maintained example
(examples/DriftKineticElectrostaticAdiabatic/itg_cylindre), at reduced
resolution and run length to keep it a quick gallery run.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import argparse

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.colors import sample_colorscale

from struphy import (
    BaseUnits,
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
    maxwellians,
)
from struphy.initial.base import GenericPerturbation
from struphy.models import DriftKineticElectrostaticAdiabatic
from struphy.propagators import implicit_diffusion

# A magnetized annular column: radius in [a1, a2], periodic in angle and length.
a1, a2, length = 0.1, 14.5, 1506.759067

# Radial density and temperature profiles, each decaying smoothly across the
# annulus, plus a tiny helical (m = 5, n = 1) seed perturbation in density.
mid_radius = (a1 + a2) / 2
perturbation_amplitude = 1e-6
mode_poloidal, mode_toroidal = 5, 1
density_gradient, temperature_gradient = 0.055, 0.27586
temperature_width = 1.45
density_width = temperature_width / 2
perturbation_width = 4 * density_width / temperature_width

_n_norm = (a2 - a1) / np.sum(
    np.exp(-density_gradient * density_width * np.tanh((np.linspace(a1, a2, 100_000) - mid_radius) / density_width)) * (a2 - a1) / 100_000,
)


def density_profile(r):
    return _n_norm * np.exp(-density_gradient * density_width * np.tanh((r - mid_radius) / density_width))


def temperature_profile(r):
    return np.exp(-temperature_gradient * temperature_width * np.tanh((r - mid_radius) / temperature_width))


def density_init(*etas):
    eta1 = etas[0][:, 0] if len(etas) == 1 else etas[0]
    return density_profile(a1 + (a2 - a1) * eta1)


def thermal_velocity_init(*etas):
    eta1 = etas[0][:, 0] if len(etas) == 1 else etas[0]
    return np.sqrt(temperature_profile(a1 + (a2 - a1) * eta1))


def perturbation_func(*etas):
    if len(etas) == 1:
        eta1, eta2, eta3 = etas[0][:, 0], etas[0][:, 1], etas[0][:, 2]
    else:
        eta1, eta2, eta3 = etas
    r = a1 + (a2 - a1) * eta1
    angle, z = 2 * np.pi * eta2, length * eta3
    return (
        density_profile(r)
        * perturbation_amplitude
        * np.exp(-((r - mid_radius) ** 2) / perturbation_width**2)
        * np.cos(2 * np.pi * mode_toroidal * z / length + mode_poloidal * angle)
    )


def density_xyz(x, y, z):
    return density_profile(np.sqrt(x**2 + y**2))


def pressure_xyz(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return density_profile(r) * temperature_profile(r)

# Time window of the exponential growth, and the highest poloidal mode shown in the mode-resolved figures.
GROWTH_WINDOW = (25.0, 175.0)
MAX_POLOIDAL_MODE = 12


def minor_radius(array):
    """Radius r = a1 + (a2 - a1) * e1 of the `e1` coordinate of `array`."""
    return a1 + (a2 - a1) * np.asarray(array.e1)


def mode_amplitudes(phi):
    """Fourier amplitudes of a real field in (e2, e3), as a complex array with dimensions (t, e1, m, n).

    m >= 0 is the poloidal and n the axial mode number. The periodic end points e2 = 1 and e3 = 1 repeat the
    first ones and are dropped. A field `A cos(m*theta + 2*pi*n*z/length)` has |phi_mn| = A.
    """
    field = phi.isel(e2=slice(None, -1), e3=slice(None, -1)).transpose("t", "e1", "e2", "e3")
    n_theta, n_z = field.sizes["e2"], field.sizes["e3"]
    spectrum = np.fft.fft(np.fft.rfft(field.values, axis=2), axis=3) / (n_theta * n_z)
    spectrum[:, :, 1:] *= 2.0  # fold the negative m
    if n_theta % 2 == 0:
        spectrum[:, :, -1] /= 2.0  # the Nyquist mode has no partner
    return xr.DataArray(
        spectrum,
        dims=("t", "e1", "m", "n"),
        coords={"t": field.t, "e1": field.e1, "m": np.arange(spectrum.shape[2]), "n": np.fft.fftfreq(n_z, d=1.0 / n_z).astype(int)},
    )


def radial_rms(spectrum):
    """Radial rms of |phi_mn|, over e1."""
    return np.sqrt((np.abs(spectrum) ** 2).mean("e1"))


def exponential_rate(amplitude, window):
    """Growth rate of each column m of `amplitude(t, m)` from a log-linear fit on `window`."""
    selected = amplitude.sel(t=slice(*window))
    times = selected.t.values
    rates = []
    for m in selected.m.values:
        values = selected.sel(m=m).values
        positive = values > 0
        rates.append(np.polyfit(times[positive], np.log(values[positive]), 1)[0] if positive.sum() > 2 else np.nan)
    return np.asarray(rates)


def log10_or_nan(values):
    values = np.asarray(values, dtype=float)
    return np.log10(np.where(values > 0, values, np.nan))


def create_simulation() -> Simulation:
    model = DriftKineticElectrostaticAdiabatic(base_units=BaseUnits(kBT=1.0), epsilon=1.0, use_diagnostic_poisson=True)
    model.em_fields.phi.save_data = True
    domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=length)
    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)
    grid = grids.TensorProductGrid(num_elements=(12, 20, 4), mpi_dims_mask=(True, True, True))
    derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))
    time_opts = Time(dt=5.0, Tend=250.0, split_algo="LieTrotter")

    density_bins = BinningPlot(slice="e1_e2", n_bins=(32, 64), ranges=((0.0, 1.0), (0.0, 1.0)))
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=15, loading="sobol_standard", spatial="uniform", moments=(0.0, 0.0, 2.0, 2.0)),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        sorting_params=SortingParameters(do_sort=True, boxes_per_dim=(6, 6, 2), sorting_frequency=0),
        saving_params=SavingParameters(binning_plots=(density_bins,)),
        bufsize=2.0,
    )

    model.propagators.gc_poisson.options.solver_params = implicit_diffusion.SolverParameters(maxiter=3000, tol=1e-14)
    model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(algo="explicit", evaluate_e_field=True)
    model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(algo="explicit", evaluate_e_field=True)

    equil.n_xyz = density_xyz
    equil.p_xyz = pressure_xyz

    background = maxwellians.GyroMaxwellian2D(n=(density_init, None), vth_para=(thermal_velocity_init, None), vth_perp=(thermal_velocity_init, None))
    model.kinetic_ions.var.add_background(background)
    perturbation = GenericPerturbation(perturbation_func)
    model.kinetic_ions.var.add_initial_condition(
        maxwellians.GyroMaxwellian2D(n=(density_init, perturbation), vth_para=(thermal_velocity_init, None), vth_perp=(thermal_velocity_init, None)),
    )

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="itg_drift_wave",
    )
    sim = Simulation(
        model=model,
        name="ITG drift wave",
        description=(
            "A magnetized plasma column with radial density and temperature "
            "gradients drives a helical drift wave unstable — the basic "
            "mechanism behind ion-temperature-gradient (ITG) turbulence."
            r" The initial helical density seed is $$\frac{\delta n}{n_0(r)}=10^{-6}\exp\!\left[-\frac{(r-7.3)^2}{2^2}\right]\cos\!\left(5\theta+\frac{2\pi z}{L_z}\right),$$"
            r" with :math:`L_z=1506.759067`. The background temperature is :math:`T_0(r)=\exp[-0.27586\cdot1.45\tanh((r-7.3)/1.45)]`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation):
    from plotly.subplots import make_subplots

    from _gallery import (
        export_profiling,
        heatmap_figure,
        heatmap_movie,
        merge_metadata,
        save_extra_figure,
        save_figure,
        space_time_figure,
    )

    output = sim.output.process(create_vtk=False)

    # The field-projected density perturbation (cleaner than the raw,
    # particle-noise-dominated PIC histogram) is the standard diagnostic for
    # this kind of drift instability.
    rho = output.fields.diagnostics.rho
    rho = rho.isel(component=0) if "component" in rho.dims else rho
    times = np.asarray(rho.t)
    perturbation_energy = np.asarray(output.norm(rho, squared=True))

    growth_window = (times > GROWTH_WINDOW[0]) & (times < GROWTH_WINDOW[1])
    growth_rate = float(np.polyfit(times[growth_window], np.log(perturbation_energy[growth_window]), 1)[0] / 2)
    print(f"Measured growth rate: {growth_rate:.5f}")

    figure = go.Figure(
        data=[
            go.Scatter(x=times, y=perturbation_energy, mode="lines", name="Struphy (drift-kinetic)", line={"color": "#168aad", "width": 3}),
        ],
    )
    figure.update_layout(
        title="ITG drift wave: density perturbation energy",
        xaxis_title="t [a.u.]",
        yaxis_title="‖δn‖² [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    save_figure(figure, "itg-drift-wave")

    # ---- further figures: the potential and its Fourier modes --------------------------------------------
    phi = output.evaluate("em_fields/phi")
    if not np.isfinite(phi.values).all():
        raise RuntimeError("Non-finite electrostatic potential: refusing to publish the run")
    radius = minor_radius(phi)
    spectrum = mode_amplitudes(phi)
    figures = []

    # The potential on a poloidal cross section (radius against angle) at the start of the axis.
    potential = phi.isel(e3=0, drop=True)
    limit = float(np.percentile(np.abs(potential.values), 99.7))
    still_index = int(0.9 * (potential.sizes["t"] - 1))  # the middle of the run is still too faint on this scale
    movie, _ = heatmap_movie(
        potential,
        x="e1",
        y="e2",
        x_values=radius,
        y_values=2 * np.pi * np.asarray(potential.e2),
        title="ITG drift wave: electrostatic potential φ",
        xaxis_title="r [a.u.]",
        yaxis_title="θ [rad]",
        colorbar_title="φ [a.u.]",
        colorscale="RdBu",
        zmin=-limit,
        zmax=limit,
    )
    figures.append(
        save_extra_figure(
            movie,
            "itg-drift-wave",
            "potential",
            alt="Animated electrostatic potential in radius and poloidal angle",
            caption=(
                "The electrostatic potential at z = 0 against radius and poloidal angle. The colour scale is "
                "fixed and symmetric about zero, so growth of the perturbation shows up as deepening colour."
            ),
            static_z=potential.transpose("t", "e2", "e1").values[still_index],
            static_active=still_index,
        )
    )

    # Amplitude of each poloidal mode (radial rms at the seeded axial mode number) and its growth rate.
    amplitude = radial_rms(spectrum.sel(n=mode_toroidal)).sel(m=slice(1, MAX_POLOIDAL_MODE))
    rates = exponential_rate(amplitude, GROWTH_WINDOW)
    wavenumber = amplitude.m.values / float(radius.mean())
    colors = sample_colorscale("Viridis", np.linspace(0, 1, amplitude.sizes["m"]))
    growth_figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=("Amplitude of each poloidal mode", "Growth rate per mode"),
    )
    for color, m in zip(colors, amplitude.m.values):
        growth_figure.add_scatter(
            x=amplitude.t.values,
            y=amplitude.sel(m=m).values,
            mode="lines",
            name=f"m = {m}",
            line={"color": color, "width": 4 if m == mode_poloidal else 2},
            row=1,
            col=1,
        )
    growth_figure.add_scatter(x=wavenumber, y=rates, mode="lines+markers", name="growth rate", showlegend=False, line={"color": "#168aad"}, row=1, col=2)
    growth_figure.add_vline(x=mode_poloidal / float(radius.mean()), line={"dash": "dash", "color": "#d1495b"}, annotation_text="seeded mode", row=1, col=2)
    growth_figure.update_yaxes(type="log", title_text="radial rms of |φ<sub>m,n=1</sub>| [a.u.]", row=1, col=1)
    growth_figure.update_xaxes(title_text="t [a.u.]", row=1, col=1)
    growth_figure.update_xaxes(title_text="k<sub>θ</sub> = m / r<sub>mid</sub> [a.u.]", row=1, col=2)
    growth_figure.update_yaxes(title_text="γ [a.u.]", row=1, col=2)
    growth_figure.update_xaxes(dtick=0.25, row=1, col=2)
    growth_figure.update_layout(
        title="ITG drift wave: growth of the poloidal modes",
        template="plotly_white",
        margin={"l": 80, "r": 30, "t": 90, "b": 60},
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "xanchor": "left"},
    )
    figures.append(
        save_extra_figure(
            growth_figure,
            "itg-drift-wave",
            "mode-growth",
            alt="Amplitudes of the poloidal Fourier modes against time and their fitted growth rates",
            caption=(
                "Left: radial rms amplitude of the potential's poloidal Fourier modes at the seeded axial mode "
                f"number, with the seeded m = {mode_poloidal} in bold. Right: exponential growth rate of each, "
                f"fitted for {GROWTH_WINDOW[0]:g} < t < {GROWTH_WINDOW[1]:g}, against the poloidal wavenumber."
            ),
        )
    )

    # The (m, n) spectrum at the first and last saved time.
    both = radial_rms(spectrum.isel(t=[0, -1])).sel(m=slice(0, MAX_POLOIDAL_MODE))
    axial = np.sort(both.n.values)
    both = both.sel(n=axial)
    z_values = [log10_or_nan(both.isel(t=i).transpose("n", "m").values) for i in (0, 1)]
    z_max = float(np.nanmax(z_values))
    z_min = max(float(np.nanmin(z_values)), z_max - 4.0)
    spectrum_figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.08,
        shared_yaxes=True,
        subplot_titles=[f"t = {float(t):g}" for t in both.t.values],
    )
    for column, z in enumerate(z_values, start=1):
        spectrum_figure.add_trace(
            go.Heatmap(
                z=z,
                x=both.m.values,
                y=axial,
                zmin=z_min,
                zmax=z_max,
                colorscale="Viridis",
                showscale=column == 2,
                colorbar={"title": "log₁₀ |φ<sub>mn</sub>|"},
            ),
            row=1,
            col=column,
        )
        spectrum_figure.update_xaxes(title_text="poloidal m", row=1, col=column)
    spectrum_figure.update_yaxes(title_text="axial n", dtick=1, row=1, col=1)
    spectrum_figure.update_layout(title="ITG drift wave: Fourier spectrum of φ", template="plotly_white", margin={"l": 70, "r": 30, "t": 90, "b": 60})
    figures.append(
        save_extra_figure(
            spectrum_figure,
            "itg-drift-wave",
            "spectrum",
            alt="Poloidal and axial Fourier spectrum of the potential at the first and last time",
            caption="Radial rms of the potential's Fourier amplitudes in the poloidal (m) and axial (n) mode numbers, on a logarithmic colour scale, at the first and last saved time.",
        )
    )

    # Radial structure of the seeded mode.
    interior = slice(1, -1)  # phi = 0 at the Dirichlet boundaries r = a1, a2, which a log axis cannot show
    profile = np.abs(spectrum.sel(m=mode_poloidal, n=mode_toroidal)).isel(e1=interior)
    picks = np.unique(np.linspace(0, profile.sizes["t"] - 1, 6).astype(int))
    radial_figure = go.Figure()
    for color, index in zip(sample_colorscale("Plasma", np.linspace(0, 0.9, len(picks))), picks):
        radial_figure.add_scatter(x=radius[interior], y=profile.isel(t=index).values, mode="lines", name=f"t = {float(profile.t[index]):g}", line={"color": color, "width": 3})
    radial_figure.update_layout(
        title=f"ITG drift wave: radial structure of the m = {mode_poloidal} mode",
        xaxis_title="r [a.u.]",
        yaxis_title=f"|φ<sub>m={mode_poloidal},n={mode_toroidal}</sub>(r)| [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        margin={"l": 80, "r": 30, "t": 80, "b": 60},
    )
    figures.append(
        save_extra_figure(
            radial_figure,
            "itg-drift-wave",
            "radial-structure",
            alt="Radial profile of the seeded potential mode at several times",
            caption="Radial profile of the seeded Fourier mode of the potential at evenly spaced times, on a logarithmic axis.",
        )
    )

    # Where and when does the potential grow?
    rms = np.sqrt((phi**2).mean(("e2", "e3"))).isel(e1=interior)
    rms_map = heatmap_figure(
        xr.DataArray(log10_or_nan(rms.values), dims=("t", "e1"), coords={"t": rms.t, "e1": rms.e1}),
        x="e1",
        y="t",
        x_values=radius[interior],
        title="ITG drift wave: where the potential grows",
        xaxis_title="r [a.u.]",
        yaxis_title="t [a.u.]",
        colorbar_title="log₁₀ φ<sub>rms</sub>",
    )
    figures.append(
        save_extra_figure(
            rms_map,
            "itg-drift-wave",
            "radial-time",
            alt="Root-mean-square potential over flux surfaces as a function of radius and time",
            caption="The root-mean-square of the potential over each flux surface (poloidal angle and axis), against radius and time, on a logarithmic colour scale.",
        )
    )

    # The flux-surface-averaged (m = n = 0) density change: does the profile flatten?
    density = output.evaluate("diagnostics/rho")
    density = density.isel(component=0, drop=True) if "component" in density.dims else density
    zonal = density.isel(e2=slice(None, -1), e3=slice(None, -1)).mean(("e2", "e3"))
    zonal = zonal - zonal.isel(t=0)
    figures.append(
        save_extra_figure(
            space_time_figure(
                zonal,
                space="e1",
                x_values=radius,
                title="ITG drift wave: change of the flux-surface-averaged density",
                colorbar_title="δ⟨ρ⟩ [a.u.]",
                xaxis_title="r [a.u.]",
            ),
            "itg-drift-wave",
            "profile-change",
            alt="Change of the flux-surface-averaged density against radius and time",
            caption="The density projected on the field grid, averaged over each flux surface, minus its initial value: the change of the radial profile as the wave grows.",
        )
    )

    profiling = export_profiling(sim, "itg-drift-wave")

    merge_metadata(
        "itg-drift-wave",
        measuredGrowthRate=growth_rate,
        modeNumbers=[mode_poloidal, mode_toroidal],
        figures=figures,
        **profiling,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the itg drift wave example.")
    argparser.add_argument(
        "--pproc",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    args = argparser.parse_args()

    simulation = create_simulation()
    if not args.pproc:
        # Profile every propagator, pusher and solver call in the simulation.
        simulation.run(profiling_activated=True)
    pproc(simulation)
