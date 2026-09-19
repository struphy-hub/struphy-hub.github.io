"""Two-stream instability: exponential growth from counter-streaming beams.

Two counter-streaming Maxwellian populations are kinetically unstable: a tiny
density perturbation grows exponentially, drawing free energy out of the
relative beam motion, until the field is strong enough to trap particles and
the growth saturates -- the classic two-stream instability.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/two_stream).

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

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
background = maxwellians.Maxwellian3D(n=(0.5, None), u1=(3.0, None)) + maxwellians.Maxwellian3D(
    n=(0.5, None), u1=(-3.0, None)
)
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
    from _gallery import (
        export_profiling,
        heatmap_figure,
        heatmap_movie,
        merge_metadata,
        save_extra_figure,
        save_figure,
    )

    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    output = sim.run(profiling_activated=True)

    field_energy = output.evaluate("electric_energy")

    # Fit the exponential growth rate over the clean linear-growth window
    # (before trapping saturates it, roughly t in [5, 25] for this setup).
    growth_rate = field_energy.struphy.analysis.growth_rate(window=(5.0, 25.0), amplitude=True).rate
    print(f"Measured growth rate: {growth_rate:.4f} (expected: ~0.2845, from the linear dispersion relation)")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=field_energy.t.values,
                y=field_energy.values,
                mode="lines",
                name="Struphy (PIC)",
                line={"color": "#168aad", "width": 3},
            ),
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

    output.pproc()
    f = output.evaluate("kinetic_ions/e1_v1_density/f")  # (t, e1, v1)

    # The classic two-stream "movie": phase-space (x, v) density, showing the
    # two beams' initially flat bands roll up into the characteristic vortex
    # ("cat's eye") pattern as the instability traps particles.
    phase_space, phase_static = heatmap_movie(
        f,
        x="e1",
        y="v1",
        x_values=f.e1.values * domain.params["r1"],
        title="Two-stream instability: phase-space density f(x, v)",
        xaxis_title="x [a.u.]",
        yaxis_title="v [a.u.]",
        colorbar_title="f(x, v)",
    )

    # The same distribution averaged over space: f(v, t).
    velocity_time = heatmap_figure(
        f.struphy.analysis.spatial_average(),
        x="t",
        y="v1",
        title="Two-stream instability: space-averaged distribution f(v, t)",
        xaxis_title="t [a.u.]",
        yaxis_title="v [a.u.]",
        colorbar_title="f(v)",
        zmin=0.0,
    )

    figures = [
        save_extra_figure(
            velocity_time,
            "two-stream-instability",
            "velocity-time",
            alt="Space-averaged velocity distribution as a function of time",
            caption=(
                "The distribution averaged over space, f(v, t). The two beams stay narrow and separate until about t = 20, when the instability saturates; over the following ten time units they merge into a single broad distribution that fills the gap between them."
            ),
        ),
        save_extra_figure(
            phase_space,
            "two-stream-instability",
            "phasespace",
            static_z=phase_static,
            alt="Phase-space density rolled up into the classic two-stream 'cat's eye' vortex pattern",
            caption=(
                "The classic two-stream picture: the phase-space density f(x, v); drag the slider or press Play. The two beams, initially flat bands at v = ±3, are bent by the growing wave and roll up into a trapped-particle “cat’s eye” hole. The frame shown is from the middle of the run (t = 25)."
            ),
        ),
    ]

    profiling = export_profiling(sim, "two-stream-instability")

    merge_metadata(
        "two-stream-instability",
        measuredGrowthRate=growth_rate,
        expectedGrowthRate=0.2845,
        figures=figures,
        **profiling,
    )
