"""Two-stream instability: exponential growth from counter-streaming beams.

Two counter-streaming Maxwellian populations are kinetically unstable: a tiny
density perturbation grows exponentially, drawing free energy out of the
relative beam motion, until the field is strong enough to trap particles and
the growth saturates -- the classic two-stream instability.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/two_stream).

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
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import VlasovAmpereOneSpecies

model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
model.em_fields.e_field.save_data = True

domain = domains.Cuboid(r1=31.42)
grid = grids.TensorProductGrid(num_elements=(128, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.1, Tend=50.0, split_algo="LieTrotter")

# 1000 particles per cell, drawn with a mean drift of +/-3 built into the loading moments.
# A binned x-v phase-space snapshot at every step gives the classic two-stream "movie".
phase_space_bins = BinningPlot(slice="e1_v1", n_bins=(64, 64), ranges=((0.0, 1.0), (-10.0, 10.0)))
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=1000, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0)),
    weights_params=WeightsParameters(control_variate=True),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
    saving_params=SavingParameters(binning_plots=(phase_space_bins,)),
    bufsize=0.4,
)

model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# Two counter-streaming Maxwellians (u1 = +/-3), each seeded with the same cosine mode.
perturbation_amplitude = 0.001
perturbation = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
background = maxwellians.Maxwellian3D(n=(0.5, None), u1=(3.0, None)) + maxwellians.Maxwellian3D(n=(0.5, None), u1=(-3.0, None))
model.kinetic_ions.var.add_background(background)
init = maxwellians.Maxwellian3D(n=(0.5, perturbation), u1=(3.0, None)) + maxwellians.Maxwellian3D(
    n=(0.5, perturbation),
    u1=(-3.0, None),
)
model.kinetic_ions.var.add_initial_condition(init)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="two_stream",
)
sim = Simulation(
    model=model,
    name="Two-stream instability",
    description=(
        "Two counter-streaming Maxwellian beams are kinetically unstable — a "
        "tiny perturbation grows exponentially, drawing energy from the beams, "
        "until particle trapping saturates the growth."
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

    with h5py.File(os.path.join(env.path_out, "data", "data_proc0.hdf5"), "r") as f:
        time = np.asarray(f["time"]["value"])
        field_energy = np.asarray(f["scalar"]["electric_energy"])

    # Fit the exponential growth rate over the clean linear-growth window
    # (before trapping saturates it, roughly t in [5, 25] for this setup).
    linear = (time > 5.0) & (time < 25.0)
    growth_rate = float(np.polyfit(time[linear], np.log(field_energy[linear]), 1)[0] / 2)
    print(f"Measured growth rate: {growth_rate:.4f} (expected: ~0.2845, from the linear dispersion relation)")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=field_energy, mode="lines", name="Struphy (PIC)", line={"color": "#168aad", "width": 3}),
        ],
    )
    figure.update_layout(
        title="Two-stream instability: electric field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="E² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    png_path = Path("two-stream-instability.png")
    html_path = Path("two-stream-instability.html")
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

    # The classic two-stream "movie": phase-space (x, v) density, showing the
    # two beams' initially flat bands roll up into the characteristic vortex
    # ("cat's eye") pattern as the instability traps particles.
    sim.pproc(create_vtk=False)
    sim.load_plotting_data()
    phase_space = sim.f.kinetic_ions.e1_v1_density
    position = phase_space.grid_e1 * domain.params["r1"]
    velocity = phase_space.grid_v1
    phase_frames_data = phase_space.f_binned  # (n_saved_times, n_position_bins, n_velocity_bins)
    phase_times = np.linspace(0.0, time_opts.Tend, len(phase_frames_data))

    # Each frame auto-scales its own color range -- the interesting signal is
    # the *shape* (flat bands vs. trapped vortex), not the absolute density,
    # which grows sharply once particles bunch up.
    phase_frames = [
        go.Frame(
            name=f"{t:.1f}",
            data=[go.Heatmap(z=frame.T, x=position, y=velocity, zmin=0, colorscale="Viridis")],
        )
        for t, frame in zip(phase_times, phase_frames_data)
    ]
    phase_figure = go.Figure(
        data=[
            go.Heatmap(
                z=phase_frames_data[0].T,
                x=position,
                y=velocity,
                zmin=0,
                colorscale="Viridis",
                colorbar={"title": "f(x, v)"},
            ),
        ],
        frames=phase_frames,
    )
    phase_figure.update_layout(
        title="Two-stream instability: phase-space density f(x, v)",
        xaxis_title="x [a.u.]",
        yaxis_title="v [a.u.]",
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
                    {
                        "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in phase_frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.18,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    phase_png_path = Path("two-stream-instability-phasespace.png")
    phase_html_path = Path("two-stream-instability-phasespace.html")
    # A well-developed frame (not t=0) makes for a more informative static export.
    phase_figure.data[0].z = phase_frames_data[len(phase_frames_data) // 2].T
    phase_figure.write_image(phase_png_path, width=1100, height=650, scale=2)
    phase_figure.data[0].z = phase_frames_data[0].T
    phase_figure.write_html(
        phase_html_path,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Saved {phase_png_path.resolve()}")
    print(f"Saved {phase_html_path.resolve()}")

    # Export scope-profiler's plot-data JSON (durations, gantt, flame,
    # region statistics) via its Python API, so the example page can render
    # native Plotly figures from the real run above -- not a separate report.
    from scope_profiler import plot_durations, plot_flame, plot_gantt, read_h5, write_region_statistics_json

    profile_reader = read_h5(sim.profiling_filepath)
    profile_h5_path = Path("two-stream-instability-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    durations_bars = []
    durations_payload = None
    for metric in ("avg", "min", "max", "total"):
        metric_path = Path(f"two-stream-instability-durations-{metric}.tmp.json")
        plot_durations([profile_reader], ranks=[0], metric=metric, data_filepath=metric_path, data_format="json", verbose=False)
        metric_payload = json.loads(metric_path.read_text())
        if durations_payload is None:
            durations_payload = metric_payload
        durations_bars.extend(metric_payload["bars"])
        metric_path.unlink()
    durations_payload["bars"] = durations_bars
    durations_path = Path("two-stream-instability-durations.json")
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("two-stream-instability-gantt.json")
    flame_path = Path("two-stream-instability-flame.json")
    region_stats_path = Path("two-stream-instability-region-stats.json")
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

    metadata_path = Path("two-stream-instability.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredGrowthRate"] = growth_rate
    metadata["expectedGrowthRate"] = 0.2845
    metadata["phaseSpaceThumbnail"] = "/images/examples/two-stream-instability-phasespace.png"
    metadata["phaseSpaceInteractive"] = "/examples/two-stream-instability-phasespace.html"
    metadata["profilingData"] = "/examples/two-stream-instability-profile.h5"
    metadata["profilingDurations"] = "/examples/two-stream-instability-durations.json"
    metadata["profilingGantt"] = "/examples/two-stream-instability-gantt.json"
    metadata["profilingFlame"] = "/examples/two-stream-instability-flame.json"
    metadata["profilingRegionStats"] = "/examples/two-stream-instability-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
