"""Verify light-wave dispersion with Struphy's Maxwell model.

This compact gallery example follows Struphy's maintained Maxwell verification
test. It excites a broadband electric field, evolves Maxwell's equations with
FEEC, and plots the numerical dispersion relation against omega = c k.

Requires Struphy with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it shows its figures.
"""

import argparse

import numpy as np

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    grids,
    perturbations,
)
from struphy.models import Maxwell


def create_simulation() -> Simulation:
    # Model and structure-preserving Maxwell propagator.
    model = Maxwell()
    model.propagators.maxwell.options = model.propagators.maxwell.Options(
        algo="implicit",
    )

    # A periodic one-dimensional domain embedded in 3D.
    domain = domains.Cuboid(r3=20.0)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 128))
    derham_opts = DerhamOptions(degree=(1, 1, 3))
    time_opts = Time(dt=0.01, Tend=50.0)

    # Broadband noise excites several light-wave modes at once.
    model.em_fields.e_field.add_perturbation(
        perturbations.Noise(amp=0.1, comp=0, seed=123),
    )
    model.em_fields.e_field.add_perturbation(
        perturbations.Noise(amp=0.1, comp=1, seed=123),
    )

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="maxwell_light_wave",
    )
    sim = Simulation(
        model=model,
        name="Maxwell light-wave dispersion",
        description=(
            "Excite a broadband electric field and recover the vacuum dispersion "
            "relation ω = ck with Struphy’s FEEC Maxwell solver."
            r" The two transverse electric components start with coefficient-noise amplitude :math:`A=0.1`, while :math:`\mathbf{B}(z,0)=0`."
            r" On the periodic interval :math:`L_z=20`, the allowed wavenumbers are $$k_n=\frac{2\pi n}{20},\qquad n\in\mathbb{Z}.$$"
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation):
    """Measure the speed of light from the (k, omega) spectrum; return the figures and the measured values."""
    from struphy_plots.analysis import fit_dispersion_branches, power_spectrum

    output = sim.output
    length = sim.domain.params["r3"] - sim.domain.params["l3"]

    # E_x along z over time, at the grid points of the cells and the periodic endpoint, in physical z.
    cells = output.grid.num_elements[2]
    e_x = output.evaluate(
        "em_fields/e_field", eta1=0.0, eta2=0.0, eta3=np.linspace(0.0, 1.0, cells + 1), representation="1"
    ).isel(component=0)
    e_x = e_x.assign_coords(z=("eta3", e_x.eta3.values * length)).swap_dims(eta3="z")

    # The branch of the (k, omega) power spectrum: its peaks count above half the peak amplitude of their
    # k, i.e. a quarter of its peak power.
    spectrum = power_spectrum(e_x, dim="z")
    branch = fit_dispersion_branches(spectrum, n_branches=1, noise_level=0.5**2, order=10)[0]
    phase_velocity = float(branch.velocity)
    print(f"Measured phase velocity: {phase_velocity:.5f} (exact: 1.0)")

    figures = {
        "dispersion": spectrum.struphy.plotly.dispersion(
            branches={"light wave, c = 1": lambda k: k},
            fits=[branch],
            omega_max=float(spectrum.k.max()),
            title="Maxwell light-wave dispersion",
        ),
        # Waves travelling in both directions leave diagonal stripes, whose slope is the wave speed.
        "space-time": e_x.struphy.plotly.space_time(
            title="Maxwell light waves: electric field E(z, t)", colorbar_title="E_x"
        ),
    }
    results = {"measuredPhaseVelocity": phase_velocity, "exactPhaseVelocity": 1.0}
    return figures, results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the maxwell wave example.")
    argparser.add_argument(
        "--pproc-only",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    args = argparser.parse_args()

    simulation = create_simulation()
    if not args.pproc_only:
        simulation.run()
    figures, _ = pproc(simulation)
    for figure in figures.values():
        figure.show()
