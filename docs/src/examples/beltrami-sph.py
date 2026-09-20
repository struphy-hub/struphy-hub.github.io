"""Pressureless SPH markers orbiting in a stationary Beltrami force field.

The two-dimensional velocity

    u = (-cos(pi x) sin(pi y), sin(pi x) cos(pi y))

is a stationary solution of the pressureless momentum equation when the external
potential is p = (sin(pi x)^2 + sin(pi y)^2) / 2, because (u . grad)u = -grad p.
Markers initialized with this velocity should therefore keep following the same
Eulerian velocity field. Their particle Hamiltonian |v|^2 / 2 + p is conserved as
well. The animation shows the marker motion on stream-function contours while the
second panel reports both numerical errors.

Adapted from Struphy's ``tutorial_beltrami_sph.ipynb`` tutorial.

Requires Struphy 3.3 with compiled kernels (`struphy compile`).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    equils,
    grids,
)
from struphy.models import PressureLessSPH
from struphy.ode.utils import ButcherTableau

box_min = -0.5
box_max = 0.5
boxes = 16
markers_per_box = 4


def velocity(x, y, z):
    """Stationary cellular-flow velocity in physical coordinates."""
    del z
    return (
        -np.cos(np.pi * x) * np.sin(np.pi * y),
        np.sin(np.pi * x) * np.cos(np.pi * y),
        np.zeros_like(x),
    )


def potential(x, y, z):
    """Potential whose negative gradient accelerates the stationary flow."""
    del z
    return 0.5 * (np.sin(np.pi * x) ** 2 + np.sin(np.pi * y) ** 2)


def density(x, y, z):
    """Uniform initial density."""
    del y, z
    return np.ones_like(x)


beltrami_flow = equils.GenericCartesianFluidEquilibrium(
    u_xyz=velocity,
    p_xyz=potential,
    n_xyz=density,
)

model = PressureLessSPH(epsilon=1.0)
model.propagators.push_eta.options = model.propagators.push_eta.Options(
    butcher=ButcherTableau(algo="forward_euler"),
)
model.propagators.push_v.potential = beltrami_flow.p0
model.cold_fluid.set_markers(
    loading_params=LoadingParameters(ppb=markers_per_box, loading="tesselation"),
    weights_params=WeightsParameters(),
    boundary_params=BoundaryParameters(bc=("reflect", "reflect", "periodic")),
    sorting_params=SortingParameters(
        boxes_per_dim=(boxes, boxes, 1),
        dims_mask=(True, True, False),
    ),
    saving_params=SavingParameters(n_markers=1.0),
    bufsize=0.5,
)
model.cold_fluid.var.add_background(beltrami_flow)

domain = domains.Cuboid(
    l1=box_min,
    r1=box_max,
    l2=box_min,
    r2=box_max,
    l3=0.0,
    r3=1.0,
)
env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="beltrami_sph",
)
sim = Simulation(
    model=model,
    name="SPH in a Beltrami force field",
    description=(
        "Pressureless SPH markers circulate through a stationary cellular flow driven by a prescribed "
        "Beltrami potential. The computed marker velocity is compared with the exact Eulerian velocity "
        "field, while conservation of each marker's kinetic-plus-potential energy checks the orbit integration."
    ),
    env=env,
    time_opts=Time(dt=0.02, Tend=4.0, split_algo="Strang"),
    domain=domain,
    grid=grids.TensorProductGrid(num_elements=(32, 32, 1)),
    derham_opts=DerhamOptions(
        degree=(3, 3, 1),
        bcs=(("free", "free"), ("free", "free"), None),
    ),
)


if __name__ == "__main__":
    from _gallery import export_profiling, is_root, merge_metadata, save_figure

    output = sim.run(profiling_activated=True)
    output.pproc()

    orbits = output.evaluate("cold_fluid")
    times = np.asarray(orbits.t.values)
    x, y, z = (np.asarray(orbits.sel(quantity=name).values) for name in ("x", "y", "z"))
    v1, v2, v3 = (np.asarray(orbits.sel(quantity=name).values) for name in ("v1", "v2", "v3"))
    values = np.stack((x, y, z, v1, v2, v3), axis=-1)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite Beltrami marker orbit: refusing to publish the run")
    if not np.isclose(times[-1], sim.time_opts.Tend):
        raise RuntimeError(f"Incomplete Beltrami run: stopped at t={times[-1]:g}")

    exact_v1, exact_v2, _ = velocity(x, y, z)
    initial_speed_rms = float(np.sqrt(np.mean(v1[0] ** 2 + v2[0] ** 2)))
    velocity_error = np.sqrt(np.mean((v1 - exact_v1) ** 2 + (v2 - exact_v2) ** 2, axis=1))
    velocity_error /= initial_speed_rms

    hamiltonian = 0.5 * (v1**2 + v2**2 + v3**2) + potential(x, y, z)
    hamiltonian_scale = np.maximum(np.abs(hamiltonian[0]), 1e-14)
    energy_error = np.max(np.abs(hamiltonian - hamiltonian[0]) / hamiltonian_scale, axis=1)
    max_velocity_error = float(np.max(velocity_error))
    max_energy_error = float(np.max(energy_error))
    if max_velocity_error > 0.1 or max_energy_error > 0.1:
        raise RuntimeError(
            "Beltrami verification failed: "
            f"{max_velocity_error=:.3g}, {max_energy_error=:.3g}"
        )
    if is_root():
        print(
            f"maximum relative velocity RMS error {max_velocity_error:.3e}; "
            f"maximum per-marker Hamiltonian drift {max_energy_error:.3e}"
        )

    # The stream function psi has u = (d_y psi, -d_x psi); its contours are exact trajectories.
    mesh = np.linspace(box_min, box_max, 161)
    mesh_x, mesh_y = np.meshgrid(mesh, mesh)
    stream_function = np.cos(np.pi * mesh_x) * np.cos(np.pi * mesh_y) / np.pi
    marker_color = np.cos(np.pi * x[0]) * np.cos(np.pi * y[0]) / np.pi
    color_limit = float(np.max(np.abs(marker_color)))

    def marker_trace(index):
        return go.Scatter(
            x=x[index],
            y=y[index],
            mode="markers",
            name="SPH markers",
            marker={
                "size": 5,
                "color": marker_color,
                "colorscale": "RdBu",
                "cmin": -color_limit,
                "cmax": color_limit,
                "line": {"width": 0},
                "colorbar": {"title": "initial ψ", "x": 0.47},
            },
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )

    contour = go.Contour(
        x=mesh,
        y=mesh,
        z=stream_function,
        contours={"coloring": "none", "showlabels": False},
        line={"color": "rgba(40,55,65,.32)", "width": 1},
        showscale=False,
        hoverinfo="skip",
        name="exact streamlines",
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=(0.58, 0.42),
        horizontal_spacing=0.14,
        subplot_titles=("Markers on exact streamlines", "Verification errors"),
    )
    figure.add_trace(contour, row=1, col=1)
    figure.add_trace(marker_trace(0), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=times, y=np.maximum(velocity_error, 1e-16), mode="lines", name="velocity RMS"),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(x=times, y=np.maximum(energy_error, 1e-16), mode="lines", name="Hamiltonian drift"),
        row=1,
        col=2,
    )

    picks = np.unique(np.linspace(0, len(times) - 1, min(100, len(times)), dtype=int))
    figure.frames = [
        go.Frame(name=f"{times[index]:.2f}", data=[marker_trace(index)], traces=[1]) for index in picks
    ]
    figure.update_layout(
        title="Pressureless SPH in a stationary Beltrami flow",
        template="plotly_white",
        margin={"l": 65, "r": 35, "t": 90, "b": 135},
        legend={"orientation": "h", "y": 1.08},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": -0.2,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 45, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    }
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "len": 0.88,
                "y": -0.12,
                "currentvalue": {"prefix": "t = "},
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in figure.frames
                ],
            }
        ],
    )
    figure.update_xaxes(title_text="x", range=[box_min, box_max], constrain="domain", row=1, col=1)
    figure.update_yaxes(
        title_text="y",
        range=[box_min, box_max],
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="t", row=1, col=2)
    figure.update_yaxes(title_text="relative error", type="log", range=[-7, -0.7], row=1, col=2)

    final_data = [contour, marker_trace(-1), figure.data[2], figure.data[3]]
    save_figure(
        figure,
        "beltrami-sph",
        width=1100,
        height=680,
        static_data=final_data,
        static_active=len(figure.frames) - 1,
    )
    merge_metadata(
        "beltrami-sph",
        markers=int(x.shape[1]),
        maxVelocityError=max_velocity_error,
        maxHamiltonianDrift=max_energy_error,
        tutorial="https://struphy-hub.github.io/struphy/_collections/tutorials/tutorial_beltrami_sph.html",
        **export_profiling(sim, "beltrami-sph"),
    )
