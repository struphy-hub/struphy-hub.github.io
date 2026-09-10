"""Trace full-orbit test particles through a tokamak's magnetic field.

A handful of markers are seeded from a Maxwellian and pushed with Struphy's
full-orbit (Vlasov) pusher through the static magnetic field of an analytic
tokamak equilibrium, on a flux-aligned `Tokamak` domain. No fields are solved
self-consistently -- this traces single-particle motion in a fixed background
field, so each particle should gyrate around a field line while it circulates
(or bounces) through the torus, conserving its speed exactly.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    grids,
)
from struphy.fields_background import equils
from struphy.kinetic_background import maxwellians
from struphy.models import Vlasov
from struphy.pic.base import BoundaryParameters, LoadingParameters, SavingParameters

model = Vlasov(charge_number=1, mass_number=1.0)
model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D())

# Load a small population; only a handful of markers have their full orbit saved.
# eta1 is the radial flux coordinate on this domain, so it must reflect at the
# plasma edge/axis rather than wrap periodically like the two angular
# coordinates (eta2, eta3) -- periodic in eta1 would teleport a particle from
# the outer edge back to the magnetic axis. The tracked markers start at a
# modest radius (rather than anywhere in [0, 1]) so their orbits stay clear of
# that reflecting boundary for most of the run.
n_tracked = 5
tracked_start = tuple((0.35, None, None, None, None, None) for _ in range(n_tracked))
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(Np=300, seed=7, specific_markers=tracked_start),
    saving_params=SavingParameters(n_markers=n_tracked),
    boundary_params=BoundaryParameters(bc=("reflect", "periodic", "periodic")),
)

# A flux-aligned tokamak domain, built by field-line tracing an analytic
# axisymmetric MHD equilibrium. A stronger-than-default field (B0) shrinks the
# Larmor radius, so gyration shows up as tight loops on top of the smooth
# guiding-center motion instead of dominating it.
equil = equils.AdhocTorus(B0=8.0)
domain = domains.Tokamak(equilibrium=equil, num_elements=(4, 16), degree=(2, 3))
grid = grids.TensorProductGrid(num_elements=(8, 12, 4))
derham_opts = DerhamOptions(degree=(1, 2, 1))
time_opts = Time(dt=0.005, Tend=20.0)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="vlasov_tokamak",
)
sim = Simulation(
    model=model,
    name="Vlasov particle orbits in a tokamak",
    description=(
        "Trace full-orbit test particles through a tokamak’s magnetic field "
        "and watch them gyrate around field lines while circulating -- or "
        "bouncing -- through the torus."
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

    # (time, particle, [x, y, z, v1, v2, v3, weight, id]) in physical coordinates.
    orbits = np.asarray(sim.orbits.kinetic_ions)

    # A magnetic field alone does no work, so each particle's speed should be
    # conserved -- a genuine accuracy check on the pusher, not just a demo.
    speed = np.linalg.norm(orbits[:, :, 3:6], axis=2)
    max_relative_speed_drift = float(np.max(np.abs(speed - speed[0]) / speed[0]))
    print(f"Max relative drift in particle speed (should be ~0): {max_relative_speed_drift:.5f}")

    # The plasma boundary (outer flux surface), for visual context around the orbits.
    boundary_x, boundary_y, boundary_z = domain.outer_boundary_mesh(n2=40, n3=80)

    n_particles = orbits.shape[1]
    n_samples = orbits.shape[0]
    palette = ["#168aad", "#d62828", "#f77f00", "#6a4c93", "#43aa8b"]

    # A moving "comet" per particle: a short, fading trail ending in a bright
    # head marker, animated along the (already-computed) trajectory. The full
    # path stays visible underneath, dimmed, for context.
    n_frames = 150
    tail_length = 250
    frame_indices = np.linspace(0, n_samples - 1, n_frames, dtype=int)
    comet_trace_start = 1 + n_particles  # after the boundary surface + dimmed full paths

    def comet_arrays(p: int, idx: int):
        start = max(0, idx - tail_length)
        return orbits[start : idx + 1, p, 0], orbits[start : idx + 1, p, 1], orbits[start : idx + 1, p, 2]

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=boundary_x,
            y=boundary_y,
            z=boundary_z,
            colorscale=[[0, "#a9c4d8"], [1, "#a9c4d8"]],
            showscale=False,
            opacity=0.25,
            hoverinfo="skip",
            name="Plasma boundary",
            showlegend=True,
        ),
    )
    for p in range(n_particles):
        figure.add_trace(
            go.Scatter3d(
                x=orbits[:, p, 0],
                y=orbits[:, p, 1],
                z=orbits[:, p, 2],
                mode="lines",
                line={"width": 1.5, "color": palette[p % len(palette)]},
                opacity=0.25,
                showlegend=False,
                hoverinfo="skip",
            ),
        )
    # Default to a well-developed moment (not t=0) -- both for the interactive
    # page's initial view and for the static PNG export, which can only ever
    # capture the base `data`, not the animation frames.
    default_frame_index = len(frame_indices) // 2
    default_idx = frame_indices[default_frame_index]
    for p in range(n_particles):
        cx, cy, cz = comet_arrays(p, default_idx)
        sizes = [3] * (len(cx) - 1) + [7]
        figure.add_trace(
            go.Scatter3d(
                x=cx,
                y=cy,
                z=cz,
                mode="lines+markers",
                line={"width": 5, "color": palette[p % len(palette)]},
                marker={"size": sizes, "color": palette[p % len(palette)]},
                name=f"particle {p + 1}",
            ),
        )

    frames = []
    for idx in frame_indices:
        frame_data = []
        for p in range(n_particles):
            cx, cy, cz = comet_arrays(p, idx)
            sizes = [3] * (len(cx) - 1) + [7]
            frame_data.append(
                go.Scatter3d(x=cx, y=cy, z=cz, mode="lines+markers", marker={"size": sizes}),
            )
        frames.append(go.Frame(name=f"{idx * time_opts.dt:.2f}", data=frame_data, traces=list(range(comet_trace_start, comet_trace_start + n_particles))))
    figure.frames = frames

    figure.update_layout(
        title="Full-orbit particle trajectories in a tokamak",
        scene={
            "xaxis_title": "x [a.u.]",
            "yaxis_title": "y [a.u.]",
            "zaxis_title": "z [a.u.]",
            "aspectmode": "data",
        },
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 60, "b": 90},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.75)"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.08,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 13, "redraw": False}, "fromcurrent": False, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
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
                "active": default_frame_index,
                "x": 0.12,
                "len": 0.88,
                "y": -0.02,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    png_path = Path("vlasov-tokamak.png")
    html_path = Path("vlasov-tokamak.html")
    figure.write_image(png_path, width=1100, height=850, scale=2)
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
    profile_h5_path = Path("vlasov-tokamak-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    durations_bars = []
    durations_payload = None
    for metric in ("avg", "min", "max", "total"):
        metric_path = Path(f"vlasov-tokamak-durations-{metric}.tmp.json")
        plot_durations([profile_reader], ranks=[0], metric=metric, data_filepath=metric_path, data_format="json", verbose=False)
        metric_payload = json.loads(metric_path.read_text())
        if durations_payload is None:
            durations_payload = metric_payload
        durations_bars.extend(metric_payload["bars"])
        metric_path.unlink()
    durations_payload["bars"] = durations_bars
    durations_path = Path("vlasov-tokamak-durations.json")
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("vlasov-tokamak-gantt.json")
    flame_path = Path("vlasov-tokamak-flame.json")
    region_stats_path = Path("vlasov-tokamak-region-stats.json")
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

    metadata_path = Path("vlasov-tokamak.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["maxRelativeSpeedDrift"] = max_relative_speed_drift
    metadata["trackedParticles"] = n_tracked
    metadata["profilingData"] = "/examples/vlasov-tokamak-profile.h5"
    metadata["profilingDurations"] = "/examples/vlasov-tokamak-durations.json"
    metadata["profilingGantt"] = "/examples/vlasov-tokamak-gantt.json"
    metadata["profilingFlame"] = "/examples/vlasov-tokamak-flame.json"
    metadata["profilingRegionStats"] = "/examples/vlasov-tokamak-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
