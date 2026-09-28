"""Verify a time-dependent Poisson solve against its analytic solution.

This compact gallery example follows Struphy's maintained Poisson verification
test. A cosine-mode charge density oscillates in time; Struphy's FEEC solver
recovers the potential at every step, compared here against the closed-form
solution.

Requires Struphy with compiled kernels (`struphy compile`) and plasma-plots with Plotly
(`pip install "plasma-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np
from struphy import (EnvironmentOptions, Simulation, Time, domains, grids,
                     perturbations)
from struphy.models import Poisson
from plasma_plots import save_figure

# Problem parameters: a single cosine mode in space, oscillating in time.
WAVENUMBER = 2
AMPLITUDE = 0.1
OMEGA = 2 * np.pi


def create_simulation() -> Simulation:
    # A time-dependent right-hand side needs the extra `source` propagator.
    model = Poisson(with_t_dep_source=True)

    # A 1D interval, resolved in the x-direction only.
    domain = domains.Cuboid(l1=-5.0, r1=5.0)
    grid = grids.TensorProductGrid(num_elements=(48, 1, 1))
    time_opts = Time(dt=0.01, Tend=2.0)

    # The source oscillates as cos(omega t), driving a single cosine spatial mode.
    model.propagators.source.options = model.propagators.source.Options(omega=OMEGA)

    model.em_fields.source.add_perturbation(
        perturbations.ModesCos(ls=(WAVENUMBER,), amps=(AMPLITUDE,)),
    )

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="poisson_source",
    )
    simulation = Simulation(
        model=model,
        name="Poisson time-dependent source",
        description=(
            "Drive a 1D Poisson solve with an oscillating cosine-mode charge "
            "density and compare Struphy’s FEEC potential against the exact "
            "solution, at every time step."
            r" The prescribed source is $$\rho(x,t)=0.1\cos(2\pi x/5)\cos(2\pi t),$$"
            r" on the periodic interval :math:`-5\le x<5`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
    )
    return simulation


def pproc(sim: Simulation, show: bool = False):
    output = sim.output.pproc(create_vtk=False)
    domain = sim.domain

    # Exact solution of -d^2(phi)/dx^2 = rho(t, x), rho = A cos(k x) cos(omega t).
    Lx = domain.params["r1"] - domain.params["l1"]
    k = WAVENUMBER * 2 * np.pi / Lx

    def phi_exact(x, t):
        return AMPLITUDE / k**2 * np.cos(k * x) * np.cos(OMEGA * t)

    phi = output.fields.em_fields.phi
    phi_line = phi.isel(eta2=0, eta3=0)
    if "component" in phi_line.dims:
        phi_line = phi_line.isel(component=0)
    # The largest pointwise error over the whole run, relative to the amplitude of the exact potential.
    phi_scale = AMPLITUDE / k**2
    max_error = phi_line.plasma.analysis.error(phi_exact, norm="max", dims=("t", "eta1"), args=("X", "t"))
    max_relative_error = float(max_error) / phi_scale
    print(f"Max relative error over the run: {max_relative_error:.5f}")

    figure = phi_line.plasma.plot.line_animation(
        x="eta1",
        reference={"Exact": phi_exact},
        x_of=lambda eta1: domain.params["l1"] + Lx * eta1,
        xlabel="x [a.u.]",
        ylim=(-1.15 * phi_scale, 1.15 * phi_scale),
        title="Poisson potential: FEEC solution vs. exact",
        backend="plotly",
    )

    save_figure(figure, "poisson-source", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Poisson source example.")
    argparser.add_argument(
        "--pproc-only",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    argparser.add_argument("--show", action="store_true", help="Show the figures before saving them.")
    args = argparser.parse_args()

    simulation = create_simulation()

    if args.pproc_only:
        pproc(simulation, show=args.show)
    else:
        simulation.run()
        pproc(simulation, show=args.show)
