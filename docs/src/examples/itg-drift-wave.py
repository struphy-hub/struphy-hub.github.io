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

import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

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

model = DriftKineticElectrostaticAdiabatic(base_units=BaseUnits(kBT=1.0), epsilon=1.0, use_diagnostic_poisson=True)
model.em_fields.phi.save_data = True

# A magnetized annular column: radius in [a1, a2], periodic in angle and length.
a1, a2, length = 0.1, 14.5, 1506.759067
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
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    sim.run(profiling_activated=True)
    sim.pproc(create_vtk=False)
    sim.load_plotting_data()

    # The field-projected density perturbation (cleaner than the raw,
    # particle-noise-dominated PIC histogram) is the standard diagnostic for
    # this kind of drift instability.
    rho = sim.spline_values.diagnostics.rho_log.data
    times = np.asarray(sorted(rho.keys()))
    perturbation_energy = np.array([float(np.sum(np.asarray(rho[t][0]) ** 2)) for t in times])

    growth_window = (times > 25.0) & (times < 175.0)
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

    png_path = Path("itg-drift-wave.png")
    html_path = Path("itg-drift-wave.html")
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

    # Export scope-profiler's plot-data JSON (durations, gantt, flame,
    # region statistics) via its Python API, so the example page can render
    # native Plotly figures from the real run above -- not a separate report.
    from scope_profiler import plot_durations, plot_flame, plot_gantt, read_h5, write_region_statistics_json

    profile_reader = read_h5(sim.profiling_filepath)
    profile_h5_path = Path("itg-drift-wave-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    durations_bars = []
    durations_payload = None
    for metric in ("avg", "min", "max", "total"):
        metric_path = Path(f"itg-drift-wave-durations-{metric}.tmp.json")
        plot_durations([profile_reader], ranks=[0], metric=metric, data_filepath=metric_path, data_format="json", verbose=False)
        metric_payload = json.loads(metric_path.read_text())
        if durations_payload is None:
            durations_payload = metric_payload
        durations_bars.extend(metric_payload["bars"])
        metric_path.unlink()
    durations_payload["bars"] = durations_bars
    durations_path = Path("itg-drift-wave-durations.json")
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("itg-drift-wave-gantt.json")
    flame_path = Path("itg-drift-wave-flame.json")
    region_stats_path = Path("itg-drift-wave-region-stats.json")
    plot_gantt([profile_reader], ranks=[0], data_filepath=gantt_path, data_format="json", verbose=False)
    plot_flame([profile_reader], ranks=[0], data_filepath=flame_path, data_format="json", verbose=False)
    write_region_statistics_json([profile_reader], region_stats_path, ranks=[0])

    # A flame graph draws one node per *individual call*, unlike the other
    # charts (which aggregate) -- a run with thousands of steps produces
    # tens of thousands of near-identical nodes, which is both unreadable
    # and heavy enough to hang the browser tab. Truncating to the first
    # FLAME_MAX_CALLS calls (in call order, which is chronological and
    # nests parents before children) keeps the full setup phase plus
    # several complete iterations of the step loop -- enough to see the
    # real call hierarchy -- without the repetition.
    FLAME_MAX_CALLS = 3000
    flame_payload = json.loads(flame_path.read_text())
    if len(flame_payload["calls"]) > FLAME_MAX_CALLS:
        flame_payload["calls"] = sorted(flame_payload["calls"], key=lambda c: c["call_id"])[:FLAME_MAX_CALLS]
        flame_path.write_text(json.dumps(flame_payload))

    # A gantt bar is also one call, not an aggregate -- the same truncation
    # (kept in time order, so it's setup plus the same early portion of the
    # step loop shown in the flame chart above) keeps it fast to render.
    GANTT_MAX_INTERVALS = 5000
    gantt_payload = json.loads(gantt_path.read_text())
    if len(gantt_payload["intervals"]) > GANTT_MAX_INTERVALS:
        gantt_payload["intervals"] = sorted(gantt_payload["intervals"], key=lambda c: c["start_seconds"])[:GANTT_MAX_INTERVALS]
        gantt_path.write_text(json.dumps(gantt_payload))
    print(f"Saved {profile_h5_path.resolve()}")
    print(f"Saved {durations_path.resolve()}, {gantt_path.resolve()}, {flame_path.resolve()}, {region_stats_path.resolve()}")

    metadata_path = Path("itg-drift-wave.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredGrowthRate"] = growth_rate
    metadata["modeNumbers"] = [mode_poloidal, mode_toroidal]
    metadata["profilingData"] = "/examples/itg-drift-wave-profile.h5"
    metadata["profilingDurations"] = "/examples/itg-drift-wave-durations.json"
    metadata["profilingGantt"] = "/examples/itg-drift-wave-gantt.json"
    metadata["profilingFlame"] = "/examples/itg-drift-wave-flame.json"
    metadata["profilingRegionStats"] = "/examples/itg-drift-wave-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
