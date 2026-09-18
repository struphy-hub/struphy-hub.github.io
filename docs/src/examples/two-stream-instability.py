"""Two-stream instability: exponential growth from counter-streaming beams.

Two counter-streaming Maxwellian populations are kinetically unstable: a tiny
density perturbation grows exponentially, drawing free energy out of the
relative beam motion, until the field is strong enough to trap particles and
the growth saturates -- the classic two-stream instability.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/two_stream).

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

from pathlib import Path

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
grid = grids.TensorProductGrid(num_elements=(256, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.1, Tend=50.0, split_algo="LieTrotter")

# 1000 particles per cell, drawn with a mean drift of +/-3 built into the loading moments.
# A binned x-v phase-space snapshot at every step gives the classic two-stream "movie".
phase_space_bins = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-10.0, 10.0)))
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
    from _gallery import export_profiling, merge_metadata, save_figure

    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    sim.run(profiling_activated=True)

    output = sim.output
    time = np.asarray(output.time)
    field_energy = np.asarray(output.scalars["electric_energy"])

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

    save_figure(figure, "two-stream-instability")

    # The classic two-stream "movie": phase-space (x, v) density, showing the
    # two beams' initially flat bands roll up into the characteristic vortex
    # ("cat's eye") pattern as the instability traps particles.
    output.process(create_vtk=False)
    phase_space = output.distributions.kinetic_ions.e1_v1_density.f
    position = np.asarray(phase_space.e1) * domain.params["r1"]
    velocity = np.asarray(phase_space.v1)
    phase_frames_data = np.asarray(phase_space)  # (time, position, velocity)
    phase_times = np.asarray(phase_space.t)

    # Each frame auto-scales its own color range -- the interesting signal is
    # the *shape* (flat bands vs. trapped vortex), not the absolute density,
    # which grows sharply once particles bunch up.
    # Every saved timestep is a full-resolution heatmap; embedding all of them
    # (one per simulation step) bloats the exported page to tens of megabytes
    # and makes it stutter. Subsample to a fixed cap of animation frames --
    # plenty for a smooth-looking movie, and lets the browser actually keep up.
    n_phase_frames = min(150, len(phase_frames_data))
    phase_frame_indices = np.linspace(0, len(phase_frames_data) - 1, n_phase_frames, dtype=int)
    phase_frames = [
        go.Frame(
            name=f"{phase_times[idx]:.1f}",
            data=[go.Heatmap(z=phase_frames_data[idx].T, x=position, y=velocity, zmin=0, colorscale="Viridis")],
        )
        for idx in phase_frame_indices
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

    profiling = export_profiling(sim, "two-stream-instability")

    merge_metadata(
        "two-stream-instability",
        measuredGrowthRate=growth_rate,
        expectedGrowthRate=0.2845,
        phaseSpaceThumbnail="/images/examples/two-stream-instability-phasespace.png",
        phaseSpaceInteractive="/examples/two-stream-instability-phasespace.html",
        **profiling,
    )
