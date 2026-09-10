"""Weibel instability: temperature anisotropy generates a magnetic field.

A plasma with a hotter perpendicular temperature than parallel temperature is
unstable to spontaneous magnetic field generation: small magnetic
perturbations grow exponentially, tapping the excess perpendicular thermal
energy, until they're strong enough to isotropize the distribution.

Adapted from Struphy's maintained example
(examples/VlasovMaxwellOneSpecies/weibel_instability), at reduced particle
count and run length to keep it a quick gallery run.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import json
import os
import shutil
from pathlib import Path

import h5py
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
from struphy.models import VlasovMaxwellOneSpecies

model = VlasovMaxwellOneSpecies(alpha=1.0, epsilon=-1.0, measure_gauss_law=True)
model.em_fields.e_field.save_data = True

wavenumber = 1.25
domain = domains.Cuboid(r1=2 * np.pi / wavenumber)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.1, Tend=200.0, split_algo="LieTrotter")

# A colder parallel (vth1) than perpendicular (vth2) thermal spread -- the
# temperature anisotropy that Weibel feeds on.
vth1 = 0.02 / np.sqrt(2)
vth2 = vth1 * np.sqrt(12)
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(
        Np=20_000,
        set_zero_velocity=(False, False, True),
        moments=(0.0, 0.0, 0.0, vth1, vth2, 1.0),
        seed=1234,
    ),
    # The control-variate weighting violates Gauss's law for this setup, so it's disabled here.
    weights_params=WeightsParameters(control_variate=False),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
    saving_params=SavingParameters(),
    bufsize=2.0,
)

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(vth1=(vth1, None), vth2=(vth2, None)))

# A tiny seed perturbation in B_z, needed to trigger the (otherwise exact) instability.
magnetic_perturbation_amplitude = -1e-4
model.em_fields.b_field.add_perturbation(
    perturbations.ModesCos(amps=(magnetic_perturbation_amplitude,), ls=(1,), comp=2),
)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="weibel_instability",
)
sim = Simulation(
    model=model,
    name="Weibel instability",
    description=(
        "A temperature-anisotropic plasma spontaneously generates a magnetic "
        "field: a tiny seed perturbation grows exponentially, tapping the "
        "excess perpendicular thermal energy."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    sim.run(profiling_activated=True)
    sim.pproc(create_vtk=False)
    sim.load_plotting_data()

    # Total magnetic (B3) field energy at each saved time, summed over the grid.
    b_field = sim.spline_values.em_fields.b_field_log.data
    times = sorted(b_field.keys())
    grid_shape = sim.grids_phy[0].shape
    cell_volume = float(np.prod([1.0 / max(n - 1, 1) for n in grid_shape]))
    magnetic_energy = np.array([float(np.sum(np.asarray(b_field[t][2]) ** 2)) * cell_volume / 2 for t in times])
    time = np.asarray(times)

    # Fit the growth rate over the clean exponential window (roughly the
    # middle third of the run, before saturation).
    growth_window = (time > time[-1] / 5) & (time < 2 * time[-1] / 5)
    growth_rate = float(np.polyfit(time[growth_window], np.log(magnetic_energy[growth_window]), 1)[0] / 2)
    print(f"Measured growth rate (in |B3|, from the energy fit): {growth_rate:.5f}")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=magnetic_energy, mode="lines", name="|B₃|² / 2 (Struphy)", line={"color": "#168aad", "width": 3}),
        ],
    )
    figure.update_layout(
        title="Weibel instability: magnetic field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="|B₃|² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    png_path = Path("weibel-instability.png")
    html_path = Path("weibel-instability.html")
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
    profile_h5_path = Path("weibel-instability-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    durations_bars = []
    durations_payload = None
    for metric in ("avg", "min", "max", "total"):
        metric_path = Path(f"weibel-instability-durations-{metric}.tmp.json")
        plot_durations([profile_reader], ranks=[0], metric=metric, data_filepath=metric_path, data_format="json", verbose=False)
        metric_payload = json.loads(metric_path.read_text())
        if durations_payload is None:
            durations_payload = metric_payload
        durations_bars.extend(metric_payload["bars"])
        metric_path.unlink()
    durations_payload["bars"] = durations_bars
    durations_path = Path("weibel-instability-durations.json")
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("weibel-instability-gantt.json")
    flame_path = Path("weibel-instability-flame.json")
    region_stats_path = Path("weibel-instability-region-stats.json")
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

    metadata_path = Path("weibel-instability.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredGrowthRate"] = growth_rate
    metadata["profilingData"] = "/examples/weibel-instability-profile.h5"
    metadata["profilingDurations"] = "/examples/weibel-instability-durations.json"
    metadata["profilingGantt"] = "/examples/weibel-instability-gantt.json"
    metadata["profilingFlame"] = "/examples/weibel-instability-flame.json"
    metadata["profilingRegionStats"] = "/examples/weibel-instability-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
