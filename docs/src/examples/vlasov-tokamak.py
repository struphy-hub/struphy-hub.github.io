"""Trace full-orbit test particles through a tokamak's magnetic field.

A handful of markers are seeded from a Maxwellian and pushed with Struphy's
full-orbit (Vlasov) pusher through the static magnetic field of an analytic
tokamak equilibrium, on a flux-aligned `Tokamak` domain. No fields are solved
self-consistently -- this traces single-particle motion in a fixed background
field, so each particle should gyrate around a field line while it circulates
(or bounces) through the torus, conserving its speed exactly.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import argparse

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


# Load a small population; only a handful of markers have their full orbit saved.
# eta1 is the radial flux coordinate on this domain, so it must reflect at the
# plasma edge/axis rather than wrap periodically like the two angular
# coordinates (eta2, eta3) -- periodic in eta1 would teleport a particle from
# the outer edge back to the magnetic axis. The tracked markers start at a
# modest radius (rather than anywhere in [0, 1]) so their orbits stay clear of
# that reflecting boundary for most of the run.
n_tracked = 5
tracked_start = tuple((0.35, None, None, None, None, None) for _ in range(n_tracked))

# A flux-aligned tokamak domain, built by field-line tracing an analytic
# axisymmetric MHD equilibrium. A stronger-than-default field (B0) shrinks the
# Larmor radius, so gyration shows up as tight loops on top of the smooth
# guiding-center motion instead of dominating it.

def create_simulation() -> Simulation:
    model = Vlasov(charge_number=1, mass_number=1.0)
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D())
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(Np=300, seed=7, specific_markers=tracked_start),
        saving_params=SavingParameters(n_markers=n_tracked),
        boundary_params=BoundaryParameters(bc=("reflect", "periodic", "periodic")),
    )
    equil = equils.AdhocTorus(B0=8.0)
    domain = domains.Tokamak(equilibrium=equil, num_elements=(4, 16), degree=(2, 3))
    grid = grids.TensorProductGrid(num_elements=(8, 12, 4))
    derham_opts = DerhamOptions(degree=(1, 2, 1))
    time_opts = Time(dt=0.005, Tend=20.0)
    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="vlasov_tokamak",
    )
    simulation = Simulation(
        model=model,
        name="Vlasov particle orbits in a tokamak",
        description=(
            "Trace full-orbit test particles through a tokamak’s magnetic field "
            "and watch them gyrate around field lines while circulating -- or "
            "bouncing -- through the torus."
            r" Velocities are drawn from the unit-thermal-speed Maxwellian $$f_0(\mathbf{v})=(2\pi)^{-3/2}e^{-|\mathbf{v}|^2/2}.$$"
            r" The five tracked particles start at logical radius :math:`\eta_1=0.35`, with random angles, in an equilibrium with field parameter :math:`B_0=8`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )
    return simulation

def pproc(sim: Simulation):

    from _gallery import export_profiling, merge_metadata, save_figure

    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    output = sim.output.process(create_vtk=False)

    # (time, particle, [x, y, z, v1, v2, v3, weight, id]) in physical coordinates.
    orbits = np.asarray(output.orbits.kinetic_ions)

    # A magnetic field alone does no work, so each particle's speed should be
    # conserved -- a genuine accuracy check on the pusher, not just a demo.
    speed = np.linalg.norm(np.asarray(output.orbits.kinetic_ions.sel(quantity=["v1", "v2", "v3"])), axis=2)
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
                        "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": False, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
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

    save_figure(figure, "vlasov-tokamak", height=850)

    # The same orbits, projected onto each coordinate plane -- a simpler,
    # non-animated companion to the 3D view above, useful for reading off
    # gyration radius and axial excursion without having to rotate anything.
    from plotly.subplots import make_subplots

    projections = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    projection_figure = make_subplots(rows=1, cols=3, subplot_titles=[f"{a}{b} plane" for a, b, _, _ in projections])
    for col, (label_a, label_b, i, j) in enumerate(projections, start=1):
        for p in range(n_particles):
            projection_figure.add_trace(
                go.Scatter(
                    x=orbits[:, p, i],
                    y=orbits[:, p, j],
                    mode="lines",
                    line={"width": 1.5, "color": palette[p % len(palette)]},
                    name=f"particle {p + 1}",
                    legendgroup=f"particle {p + 1}",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
        projection_figure.update_xaxes(title_text=f"{label_a} [a.u.]", row=1, col=col)
        # scaleanchor to the subplot's own x-axis keeps each projection's
        # aspect ratio physical, so a circular gyro-orbit still looks circular.
        projection_figure.update_yaxes(title_text=f"{label_b} [a.u.]", scaleanchor=f"x{col if col > 1 else ''}", row=1, col=col)
    projection_figure.update_layout(
        title="Full-orbit particle trajectories: coordinate-plane projections",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
    )

    save_figure(projection_figure, "vlasov-tokamak", width=1500, height=560, suffix="-projections")

    profiling = export_profiling(sim, "vlasov-tokamak")

    merge_metadata(
        "vlasov-tokamak",
        maxRelativeSpeedDrift=max_relative_speed_drift,
        trackedParticles=n_tracked,
        projectionsThumbnail="/images/examples/vlasov-tokamak-projections.png",
        projectionsInteractive="/examples/vlasov-tokamak-projections.plotly.json",
        **profiling,
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
