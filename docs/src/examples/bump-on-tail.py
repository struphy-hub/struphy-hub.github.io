"""Bump-on-tail instability: a minority beam drives Langmuir waves.

A small ("bump") population of fast particles riding on the tail of an
otherwise Maxwellian distribution is a classic source of free energy: it
drives Langmuir waves unstable, transferring energy from the hot minority
population to the growing field until particle trapping saturates it.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/bump_on).

Requires Struphy with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

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
from struphy_plots import save_figure

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


def pproc(sim: Simulation, show: bool = False):
    domain = sim.domain

    output = sim.output

    field_energy = output.scalars["electric_energy"]

    # Fit the exponential growth rate over the clean linear-growth window.
    growth_rate = field_energy.struphy.analysis.growth_rate(window=(5.0, 25.0), amplitude=True).rate
    print(f"Measured growth rate: {growth_rate:.4f}")

    figure = field_energy.struphy.plot.timeseries(logy=True, title="Bump-on-tail instability: electric field energy", backend="plotly")

    save_figure(figure, "bump-on-tail", show=show)

    # Evaluate the saved products on their grids, then look at them in more than one way.
    output.pproc()
    length = domain.params["r1"]
    f = output.evaluate("kinetic_ions/e1_v1_density/f")  # (t, e1, v1)

    # The distribution averaged over space, f(v, t): how the whole velocity distribution changes,
    # with less sampling noise than the individual x-v bins.
    f_of_v = f.struphy.analysis.spatial_average()
    velocity_time = f_of_v.struphy.plot.slice(
        x="t",
        y="v1",
        title="Bump-on-tail instability: space-averaged distribution f(v, t)",
        xlabel="t [a.u.]",
        ylabel="v [a.u.]",
        colorbar_label="f(v)",
        cmap="viridis",
        vmin=0.0,
        backend="plotly",
    )

    # The x-v phase space as a movie.
    phase_space = f.assign_coords(eta1=f.eta1.values * length).struphy.plot.animation(
        x="eta1",
        y="v1",
        max_frames=150,
        vmin=0.0,
        shared_clim=False,
        cmap="viridis",
        title="Bump-on-tail instability: phase-space density f(x, v)",
        xlabel="x [a.u.]",
        ylabel="v [a.u.]",
        colorbar_label="f(x, v)",
        backend="plotly",
    )

    save_figure(phase_space, "bump-on-tail-phasespace", frame=len(phase_space.fig.frames) // 2, show=show)
    save_figure(velocity_time, "bump-on-tail-velocity-time", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the bump on tail example.")
    argparser.add_argument(
        "--pproc-only",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    argparser.add_argument("--show", action="store_true", help="Show the figures before saving them.")
    args = argparser.parse_args()

    simulation = create_simulation()
    if not args.pproc_only:
        simulation.run()
    pproc(simulation, show=args.show)
