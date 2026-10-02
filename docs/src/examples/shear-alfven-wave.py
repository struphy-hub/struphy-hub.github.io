"""Verify shear-Alfvén wave dispersion with Struphy's ShearAlfven model.

A broadband transverse velocity perturbation excites shear-Alfvén waves
propagating along a uniform background magnetic field. Struphy evolves the
linearized MHD induction/momentum equations with FEEC, and the numerical
dispersion relation is compared against the exact Alfvén speed
v_A = B0 / sqrt(n0) = 1.

Requires Struphy with compiled kernels (`struphy compile`) and plasma-plots with Plotly
(`pip install "plasma-plots[plotly]==0.1.1"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.models import ShearAlfven
from plasma_plots import save_figure


def create_simulation() -> Simulation:
    # Model and its single (linear) propagator.
    model = ShearAlfven()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options()

    # A periodic one-dimensional domain embedded in 3D, aligned with the
    # background field so waves can propagate along B0.
    domain = domains.Cuboid(r3=20.0)
    equil = equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.1)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 128))
    derham_opts = DerhamOptions(degree=(1, 1, 3))
    time_opts = Time(dt=0.05, Tend=50.0)

    # Broadband noise in the two components transverse to B0 excites several
    # shear-Alfvén modes at once, same trick as the Maxwell light-wave example.
    model.mhd.velocity.add_background(FieldsBackground())
    model.mhd.velocity.add_perturbation(
        perturbations.Noise(amp=0.05, comp=0, seed=123),
    )
    model.mhd.velocity.add_perturbation(
        perturbations.Noise(amp=0.05, comp=1, seed=123),
    )

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="shear_alfven_wave",
    )
    sim = Simulation(
        model=model,
        name="Shear-Alfvén wave dispersion",
        description=(
            "Excite a broadband transverse velocity perturbation and recover the "
            "shear-Alfvén dispersion relation ω = v_A k with Struphy’s linearized "
            "MHD solver."
            r" Both transverse velocity components are seeded with coefficient-noise amplitude :math:`A=0.05`."
            r" The background is $$\mathbf{B}_0=\mathbf{e}_z,\qquad n_0=1,\qquad L_z=20,$$"
            r" so the reference Alfvén speed is :math:`v_A=B_0/\sqrt{n_0}=1`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation, show: bool = False):
    domain = sim.domain

    # Run, evaluate the FEEC fields on a grid, and load the result.
    output = sim.output.pproc(create_vtk=False)

    # The (k, omega) power spectrum of u_x along z, and a fit of its branch.
    from plasma_plots.analysis import power_spectrum

    velocity = output.fields.mhd.velocity
    u_x = velocity.isel(component=0, eta1=0, eta2=0)
    u_x = u_x.assign_coords(eta3=u_x.eta3 * (domain.params["r3"] - domain.params["l3"]))  # physical z
    spectrum = power_spectrum(u_x, dim="eta3")
    # Peaks count above half the column's peak amplitude, i.e. a quarter of its peak power.
    branch = spectrum.plasma.analysis.fit_branches(n_branches=1, noise_level=0.5**2, order=10)[0]
    phase_velocity = float(branch.velocity)
    print(f"Measured Alfvén speed: {phase_velocity:.5f} (exact: 1.0)")

    # The normalized power spectrum over 15 decades, for omega, k >= 0, with the exact and the fitted branch.
    figure = spectrum.plasma.plot.dispersion(
        kmin=0,
        branches={"Alfvén wave, v_A = 1": lambda k: k},
        fits=[branch],
        dynamic_range=15,
        cmap="plasma",
        omega_max=float(spectrum.k.max()),
        title="Shear-Alfvén wave dispersion",
        backend="plotly",
    )

    save_figure(figure, "shear-alfven-wave", show=show)

    # The same field along z, over time: waves travelling in both directions leave diagonal
    # stripes, whose slope is the wave speed.
    transverse = output.evaluate(
        "mhd/velocity", eta1=0.0, eta2=0.0, eta3=np.linspace(0.0, 1.0, output.grid.num_elements[2] + 1),
        representation="2",
    ).isel(component=0)  # (t, eta3)
    space_time = transverse.assign_coords(eta3=transverse.eta3.values * domain.params["r3"]).plasma.plot.slice(
        x="eta3",
        y="t",
        symmetric=True,
        cmap="RdBu_r",
        title="Shear-Alfvén waves: transverse velocity u(z, t)",
        xlabel="z [a.u.]",
        ylabel="t [a.u.]",
        colorbar_label="u₁ (logical component)",
        backend="plotly",
    )
    save_figure(space_time, "shear-alfven-wave-space-time", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the shear alfven wave example.")
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
