"""Diocotron instability: a sheared E×B ring develops rippled edges.

An annular ring of charge, confined by a magnetic field, has a sheared E×B
rotation profile at its inner and outer edges. That shear is unstable: a
tiny azimuthal (mode number m) perturbation grows, rippling the ring's edges
-- the onset of the diocotron instability, a non-neutral-plasma analogue of
the Kelvin-Helmholtz instability, which given enough time rolls those
ripples up into a rotating pattern of discrete vortices.

Adapted from Struphy's maintained example
(examples/ToyGyrokinetic/diocotron_instability), at reduced resolution and
run length to keep it a quick gallery run. Parameters follow Crouseilles,
Mehrenberger & Vecil (2014), https://doi.org/10.1140/epjd/e2014-50180-9.

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
    perturbations,
)
from struphy.models import ToyDrift

model = ToyDrift(epsilon=1.0, alpha=1.0, base_units=BaseUnits(kBT=1.0))

# An annular ring, r in [1, 10], with a uniform background field.
domain = domains.HollowCylinder(a1=1.0, a2=10.0, Lz=10.0)
equil = equils.HomogenSlab()
grid = grids.TensorProductGrid(num_elements=(32, 64, 1), mpi_dims_mask=(False, True, False))
derham_opts = DerhamOptions(degree=(3, 3, 1), bcs=(("dirichlet", "dirichlet"), None, None))
time_opts = Time(dt=0.02, Tend=25.0, split_algo="LieTrotter")

# A binned e1-e2 (radial-angular) density snapshot at every step, for the animation.
density_bins = BinningPlot(slice="e1_e2", n_bins=(64, 64), ranges=((0.0, 1.0), (0.0, 1.0)))
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=20, loading="sobol_standard", spatial="disc"),
    weights_params=WeightsParameters(control_variate=True, reject_weights=True, threshold=0.0001),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(8, 8, 1), do_sort=True, sorting_frequency=5),
    saving_params=SavingParameters(binning_plots=(density_bins,)),
    bufsize=2.0,
)

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options()
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
    algo="discrete_gradient_1st_order_newton",
    evaluate_e_field=True,
)

# A uniform-density ring between r = 4 and r = 5, seeded with a tiny m = 4 azimuthal mode.
r_minus, r_plus, mode_number = 4.0, 5.0, 4
a1, a2 = domain.params["a1"], domain.params["a2"]
eta_minus, eta_plus = (r_minus - a1) / (a2 - a1), (r_plus - a1) / (a2 - a1)


def ring_density(etas, r_minus=r_minus, r_plus=r_plus):
    radial = a1 + (a2 - a1) * etas[:, 0]
    return 1.0 * ((r_minus <= radial) & (radial < r_plus))


model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(0.0, None)))
perturbation = perturbations.ModesCos(amps=(1e-6,), ms=(mode_number,), perb_domain=((eta_minus, eta_plus), None, None))
model.kinetic_ions.var.add_initial_condition(maxwellians.GyroMaxwellian2D(n=(ring_density, perturbation)))

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="diocotron_instability",
)
sim = Simulation(
    model=model,
    name="Diocotron instability",
    description=(
        "A sheared E×B ring of charge is unstable: a tiny azimuthal "
        "perturbation grows, rippling the ring's edges — the onset of a "
        "rotating pattern of vortices."
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

    density = sim.f.kinetic_ions.e1_e2_density
    radius = a1 + (a2 - a1) * density.grid_e1
    angle_deg = 360.0 * density.grid_e2
    frames_data = density.f_binned  # (n_saved_times, n_radius_bins, n_angle_bins)
    times = np.linspace(0.0, time_opts.Tend, len(frames_data))

    # A simple measure of how far the ring has departed from its initial,
    # axisymmetric shape: the standard deviation of density around each
    # radius, averaged over the ring -- ~0 initially, growing as vortices form.
    asymmetry = np.array([float(np.mean(np.std(frame, axis=1))) for frame in frames_data])
    growth_window = (times > 5.0) & (times < 15.0)
    growth_rate = float(np.polyfit(times[growth_window], np.log(asymmetry[growth_window] + 1e-12), 1)[0])
    print(f"Measured asymmetry growth rate: {growth_rate:.4f}")

    frames = [
        go.Frame(name=f"{t:.1f}", data=[go.Heatmap(z=frame, x=angle_deg, y=radius, zmin=0, zmax=1.2, colorscale="Viridis")])
        for t, frame in zip(times, frames_data)
    ]
    # Default to the final (most visibly rippled) frame -- both for the
    # interactive page's initial view and for the static PNG export, which
    # can only ever capture the base `data`, not the animation frames.
    figure = go.Figure(
        data=[go.Heatmap(z=frames_data[-1], x=angle_deg, y=radius, zmin=0, zmax=1.2, colorscale="Viridis", colorbar={"title": "density"})],
        frames=frames,
    )
    figure.update_layout(
        title="Diocotron instability: ring density n(r, θ)",
        xaxis_title="θ [deg]",
        yaxis_title="r [a.u.]",
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.28,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {"args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": frame.name, "method": "animate"}
                    for frame in frames
                ],
                "active": len(frames) - 1,
                "x": 0.12,
                "len": 0.88,
                "y": -0.18,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    png_path = Path("diocotron-instability.png")
    html_path = Path("diocotron-instability.html")
    figure.write_image(png_path, width=1100, height=750, scale=2)
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
    profile_h5_path = Path("diocotron-instability-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    durations_bars = []
    durations_payload = None
    for metric in ("avg", "min", "max", "total"):
        metric_path = Path(f"diocotron-instability-durations-{metric}.tmp.json")
        plot_durations([profile_reader], ranks=[0], metric=metric, data_filepath=metric_path, data_format="json", verbose=False)
        metric_payload = json.loads(metric_path.read_text())
        if durations_payload is None:
            durations_payload = metric_payload
        durations_bars.extend(metric_payload["bars"])
        metric_path.unlink()
    durations_payload["bars"] = durations_bars
    durations_path = Path("diocotron-instability-durations.json")
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("diocotron-instability-gantt.json")
    flame_path = Path("diocotron-instability-flame.json")
    region_stats_path = Path("diocotron-instability-region-stats.json")
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

    metadata_path = Path("diocotron-instability.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredGrowthRate"] = growth_rate
    metadata["modeNumber"] = mode_number
    metadata["profilingData"] = "/examples/diocotron-instability-profile.h5"
    metadata["profilingDurations"] = "/examples/diocotron-instability-durations.json"
    metadata["profilingGantt"] = "/examples/diocotron-instability-gantt.json"
    metadata["profilingFlame"] = "/examples/diocotron-instability-flame.json"
    metadata["profilingRegionStats"] = "/examples/diocotron-instability-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
