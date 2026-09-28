"""Two-stream instability: exponential growth from counter-streaming beams.

Two counter-streaming Maxwellian populations are kinetically unstable: a tiny
density perturbation grows exponentially, drawing free energy out of the
relative beam motion, until the field is strong enough to trap particles and
the growth saturates -- the classic two-stream instability.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/two_stream).

Requires Struphy with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np

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

# Two counter-streaming Maxwellians (u1 = +/-3), each seeded with the same cosine mode.
perturbation_amplitude = 0.001


def create_simulation() -> Simulation:
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
        bufsize=2.0,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")
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
            r" The two beams start with $$n_\pm(x,0)=0.5+10^{-3}\cos(2\pi x/L),\qquad u_{x,\pm}=\pm3,$$"
            r" where :math:`L=31.42` and each beam has thermal speed :math:`v_{\mathrm{th}}=1`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation, show: bool = False):
    from struphy_plots.theory.kinetic import two_stream

    domain = sim.domain

    output = sim.output

    field_energy = output.scalars["electric_energy"]

    # Fit the exponential growth rate over the clean linear-growth window
    # (before trapping saturates it, roughly t in [5, 25] for this setup).
    growth_rate = field_energy.struphy.analysis.growth_rate(window=(5.0, 25.0), amplitude=True).rate
    # The linear kinetic theory of two Maxwellian beams at u = ±3 with v_th = 1, for the box mode k = 2π/L.
    expected = two_stream(2 * np.pi / domain.params["r1"], beam_speed=3.0, thermal_speed=1.0).imag
    print(f"Measured growth rate: {growth_rate:.4f} (expected: ~{expected:.4f}, from the linear dispersion relation)")

    figure = field_energy.struphy.plot.timeseries(logy=True, title="Two-stream instability: electric field energy", backend="plotly")

    save_figure(figure, "two-stream-instability", show=show)

    output.pproc()
    f = output.evaluate("kinetic_ions/e1_v1_density/f")  # (t, e1, v1)

    # The classic two-stream "movie": phase-space (x, v) density, showing the
    # two beams' initially flat bands roll up into the characteristic vortex
    # ("cat's eye") pattern as the instability traps particles.
    phase_space = f.assign_coords(eta1=f.eta1.values * domain.params["r1"]).struphy.plot.animation(
        x="eta1",
        y="v1",
        max_frames=150,
        vmin=0.0,
        shared_clim=False,
        cmap="viridis",
        title="Two-stream instability: phase-space density f(x, v)",
        xlabel="x [a.u.]",
        ylabel="v [a.u.]",
        colorbar_label="f(x, v)",
        backend="plotly",
    )

    # The same distribution averaged over space: f(v, t).
    velocity_time = f.struphy.analysis.spatial_average().struphy.plot.slice(
        x="t",
        y="v1",
        title="Two-stream instability: space-averaged distribution f(v, t)",
        xlabel="t [a.u.]",
        ylabel="v [a.u.]",
        colorbar_label="f(v)",
        cmap="viridis",
        vmin=0.0,
        backend="plotly",
    )

    save_figure(velocity_time, "two-stream-instability-velocity-time", show=show)
    save_figure(phase_space, "two-stream-instability-phasespace", frame=len(phase_space.fig.frames) // 2, show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the two stream instability example.")
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
