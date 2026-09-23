"""Bump-on-tail instability: a minority beam drives Langmuir waves.

A small ("bump") population of fast particles riding on the tail of an
otherwise Maxwellian distribution is a classic source of free energy: it
drives Langmuir waves unstable, transferring energy from the hot minority
population to the growing field until particle trapping saturates it.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/bump_on).

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import argparse

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

# A 90% bulk Maxwellian plus a 10% "bump" population drifting at u1 = -4.5.
perturbation_amplitude = 0.05


def create_simulation() -> Simulation:
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
    model.em_fields.e_field.save_data = True

    domain = domains.Cuboid(r1=62.83)
    grid = grids.TensorProductGrid(num_elements=(64, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    time_opts = Time(dt=0.1, Tend=60.0, split_algo="LieTrotter")

    # A high-resolution binned x-v snapshot at every step: the bulk sits near v = 3,
    # the bump near v = -4.5.
    phase_space_bins = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-8.0, 8.0)))
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=2000, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0)),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(binning_plots=(phase_space_bins,)),
        bufsize=2.0,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")
    perturbation = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
    bulk = maxwellians.Maxwellian3D(n=(0.9, None), u1=(3.0, None))
    bump = maxwellians.Maxwellian3D(n=(0.1, None), u1=(-4.5, None), vth1=(0.5, None))
    model.kinetic_ions.var.add_background(bulk + bump)
    init_bump = maxwellians.Maxwellian3D(n=(0.1, perturbation), u1=(-4.5, None), vth1=(0.5, None))
    model.kinetic_ions.var.add_initial_condition(bulk + init_bump)

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="bump_on_tail",
    )
    sim = Simulation(
        model=model,
        name="Bump-on-tail instability",
        description=(
            "A minority “bump” of fast particles on the tail of an otherwise "
            "Maxwellian distribution drives Langmuir waves unstable, feeding "
            "energy into the field until particle trapping saturates it."
            r" The bulk has :math:`n_b=0.9`, :math:`u_{x,b}=3` and :math:`v_{\mathrm{th},x,b}=1`."
            r" Only the fast population is perturbed: $$n_h(x,0)=0.1+0.05\cos(2\pi x/L),\qquad u_{x,h}=-4.5,\qquad v_{\mathrm{th},x,h}=0.5,$$"
            r" with :math:`L=62.83`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation):
    domain = sim.domain
    from _gallery import (
        export_profiling,
        heatmap_figure,
        heatmap_movie,
        merge_metadata,
        save_extra_figure,
        save_figure,
    )

    output = sim.output

    field_energy = output.evaluate("electric_energy")

    # Fit the exponential growth rate over the clean linear-growth window.
    growth_rate = field_energy.struphy.analysis.growth_rate(window=(5.0, 25.0), amplitude=True).rate
    print(f"Measured growth rate: {growth_rate:.4f}")

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
        title="Bump-on-tail instability: electric field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="E² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    save_figure(figure, "bump-on-tail")

    # Evaluate the saved products on their grids, then look at them in more than one way.
    output.pproc()
    length = domain.params["r1"]
    f = output.evaluate("kinetic_ions/e1_v1_density/f")  # (t, e1, v1)

    # The distribution averaged over space, f(v, t): how the whole velocity distribution changes,
    # with less sampling noise than the individual x-v bins.
    f_of_v = f.struphy.analysis.spatial_average()
    velocity_time = heatmap_figure(
        f_of_v,
        x="t",
        y="v1",
        title="Bump-on-tail instability: space-averaged distribution f(v, t)",
        xaxis_title="t [a.u.]",
        yaxis_title="v [a.u.]",
        colorbar_title="f(v)",
        zmin=0.0,
    )

    # The x-v phase space as a movie.
    phase_space, phase_static = heatmap_movie(
        f,
        x="e1",
        y="v1",
        x_values=f.e1.values * length,
        title="Bump-on-tail instability: phase-space density f(x, v)",
        xaxis_title="x [a.u.]",
        yaxis_title="v [a.u.]",
        colorbar_title="f(x, v)",
    )

    figures = [
        save_extra_figure(
            phase_space,
            "bump-on-tail",
            "phasespace",
            static_z=phase_static,
            alt="Phase-space density of the bump-on-tail instability",
            caption=(
                "The phase-space density f(x, v) of the run above, resolved on 64 spatial cells and displayed in 128 × 128 bins; drag the slider or press Play. Initially the bulk (v ≈ 3) and the bump (v ≈ −4.5) are almost uniform in x. The frame shown is from the middle of the run, where both populations have developed strong structure in x and spread far beyond their initial velocity widths."
            ),
        ),
        save_extra_figure(
            velocity_time,
            "bump-on-tail",
            "velocity-time",
            alt="Space-averaged velocity distribution as a function of time",
            caption=(
                "The distribution averaged over space, f(v, t) (the mean of the binned f over the 64 cells). The bulk (v ≈ 3, peak f ≈ 0.36) and the much smaller bump (v ≈ −4.5, peak f ≈ 0.08) start as separate populations. As the wave grows, both broaden and the region between them fills in."
            ),
        ),
    ]

    profiling = export_profiling(sim, "bump-on-tail")

    merge_metadata(
        "bump-on-tail",
        measuredGrowthRate=growth_rate,
        figures=figures,
        **profiling,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the bump on tail example.")
    argparser.add_argument(
        "--pproc",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    args = argparser.parse_args()

    simulation = create_simulation()
    if not args.pproc:
        # Profile every propagator, pusher and solver call in the simulation.
        simulation.run(profiling_activated=True)
    pproc(simulation)
