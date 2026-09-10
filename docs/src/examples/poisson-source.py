"""Verify a time-dependent Poisson solve against its analytic solution.

This compact gallery example follows Struphy's maintained Poisson verification
test. A cosine-mode charge density oscillates in time; Struphy's FEEC solver
recovers the potential at every step, compared here against the closed-form
solution.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from struphy import (
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    grids,
    perturbations,
)
from struphy.models import Poisson

# A time-dependent right-hand side needs the extra `source` propagator.
model = Poisson(with_t_dep_source=True)

# A 1D interval, resolved in the x-direction only.
domain = domains.Cuboid(l1=-5.0, r1=5.0)
grid = grids.TensorProductGrid(num_elements=(48, 1, 1))
time_opts = Time(dt=0.1, Tend=2.0)

# The source oscillates as cos(omega t), driving a single cosine spatial mode.
omega = 2 * np.pi
model.propagators.source.options = model.propagators.source.Options(omega=omega)

wavenumber = 2
amplitude = 0.1
model.em_fields.source.add_perturbation(
    perturbations.ModesCos(ls=(wavenumber,), amps=(amplitude,)),
)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="poisson_source",
)
sim = Simulation(
    model=model,
    name="Poisson time-dependent source",
    description=(
        "Drive a 1D Poisson solve with an oscillating cosine-mode charge "
        "density and compare Struphy’s FEEC potential against the exact "
        "solution, at every time step."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
)

if __name__ == "__main__":
    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    sim.run(profiling_activated=True)
    sim.pproc()
    sim.load_plotting_data()

    # Exact solution of -d^2(phi)/dx^2 = rho(t, x), rho = A cos(k x) cos(omega t).
    Lx = domain.params["r1"] - domain.params["l1"]
    k = wavenumber * 2 * np.pi / Lx

    def phi_exact(x, t):
        return amplitude / k**2 * np.cos(k * x) * np.cos(omega * t)

    phi = sim.spline_values.em_fields.phi_log.data
    x = sim.grids_phy[0][:, 0, 0]
    times = sorted(phi.keys())

    phi_scale = amplitude / k**2
    max_relative_error = 0.0
    frames = []
    for t in times:
        phi_h = np.asarray(phi[t][0])[:, 0, 0]
        phi_e = phi_exact(x, t)
        max_relative_error = max(max_relative_error, float(np.max(np.abs(phi_h - phi_e))) / phi_scale)
        frames.append(
            go.Frame(
                name=f"{t:.2f}",
                data=[
                    go.Scatter(x=x, y=phi_h),
                    go.Scatter(x=x, y=phi_e),
                ],
            ),
        )

    print(f"Max relative error over the run: {max_relative_error:.5f}")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=np.asarray(phi[times[0]][0])[:, 0, 0],
                mode="lines",
                name="Struphy (FEEC)",
                line={"color": "#168aad", "width": 3},
            ),
            go.Scatter(
                x=x,
                y=phi_exact(x, times[0]),
                mode="lines",
                name="Exact",
                line={"color": "#d62828", "width": 2, "dash": "dot"},
            ),
        ],
        frames=frames,
    )
    figure.update_layout(
        title="Poisson potential: FEEC solution vs. exact",
        xaxis_title="x [a.u.]",
        yaxis_title="φ [a.u.]",
        template="plotly_white",
        autosize=True,
        yaxis={"range": [-1.15 * phi_scale, 1.15 * phi_scale]},
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(44,62,80,0.25)",
            "borderwidth": 1,
        },
        margin={"l": 60, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.32,
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
                "x": 0.12,
                "len": 0.88,
                "y": -0.2,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    png_path = Path("poisson-source.png")
    html_path = Path("poisson-source.html")
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
    profile_h5_path = Path("poisson-source-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    durations_bars = []
    durations_payload = None
    for metric in ("avg", "min", "max", "total"):
        metric_path = Path(f"poisson-source-durations-{metric}.tmp.json")
        plot_durations([profile_reader], ranks=[0], metric=metric, data_filepath=metric_path, data_format="json", verbose=False)
        metric_payload = json.loads(metric_path.read_text())
        if durations_payload is None:
            durations_payload = metric_payload
        durations_bars.extend(metric_payload["bars"])
        metric_path.unlink()
    durations_payload["bars"] = durations_bars
    durations_path = Path("poisson-source-durations.json")
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("poisson-source-gantt.json")
    flame_path = Path("poisson-source-flame.json")
    region_stats_path = Path("poisson-source-region-stats.json")
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

    metadata_path = Path("poisson-source.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["maxRelativeError"] = max_relative_error
    metadata["profilingData"] = "/examples/poisson-source-profile.h5"
    metadata["profilingDurations"] = "/examples/poisson-source-durations.json"
    metadata["profilingGantt"] = "/examples/poisson-source-gantt.json"
    metadata["profilingFlame"] = "/examples/poisson-source-flame.json"
    metadata["profilingRegionStats"] = "/examples/poisson-source-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
