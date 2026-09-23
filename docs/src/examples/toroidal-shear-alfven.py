"""A small toroidal LinearMHD run with the m=10,11 shear-Alfvén perturbation.

Run from the repository root with:
    .venv/bin/python cli.py run toroidal-shear-alfven

Requires compiled Struphy kernels and Plotly's PNG exporter (see README.md).
The default is an exploratory local run, not a converged ITPA TAE benchmark.
Increase NUM_ELEMENTS to (24, 96, 16), DEGREE to (3, 3, 3), and END_TIME
to 500.0 to recover the supplied spatial/time resolution. The equilibrium,
sector, mode numbers, Gaussian profiles and amplitudes are retained.
"""

from time import perf_counter

import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import map_coordinates

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.models import LinearMHD

STEM = "toroidal-shear-alfven"
NUM_ELEMENTS = (8, 48, 4)
DEGREE = (3, 3, 2)
END_TIME = 40.0
DT = 0.1
SAVE_STEP = 2



# These are logical H(div) / 2-form components, not physical unit-vector
# components. TorusModes uses phase 2*pi*(m*eta2 + n*eta3); n=-1 is one
# oscillation across the 1/6-torus sector, not a full-torus n=-1 mode.
modes = (10, 11)
amplitude = 1e-3

def create_simulation() -> Simulation:
    model = LinearMHD(base_units=BaseUnits())
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
    model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options()
    for field in (model.em_fields.b_field, model.mhd.density, model.mhd.velocity, model.mhd.pressure):
        field.save_data = True
    domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, pol_period=1, tor_period=6)
    equil = equils.AdhocTorus(
        a=1.0, R0=10.0, B0=3.0, q_kind=0, p_kind=1,
        q0=1.71, q1=1.87, p1=0.95, p2=0.05, beta=0.0018,
    )
    grid = grids.TensorProductGrid(num_elements=NUM_ELEMENTS)
    derham_opts = DerhamOptions(degree=DEGREE, bcs=(("dirichlet", "dirichlet"), None, None))
    model.mhd.velocity.add_perturbation(perturbations.TorusModesSin(
        ms=modes, ns=(-1, -1), amps=(amplitude, amplitude),
        pfuns=("exp", "exp"), pfun_params=([0.5, 0.1], [0.5, 0.1]),
        comp=0, given_in_basis="2",
    ))
    model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(
        ms=modes, ns=(-1, -1), amps=tuple(amplitude / (2 * np.pi * m) for m in modes),
        pfuns=("d_exp", "d_exp"), pfun_params=([0.5, 0.1], [0.5, 0.1]),
        comp=1, given_in_basis="2",
    ))
    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs", sim_folder="toroidal_shear_alfven",
        save_step=SAVE_STEP,
    )
    time_opts = Time(dt=DT, Tend=END_TIME)
    simulation = Simulation(
        model=model,
        name="Toroidal shear-Alfvén waves",
        description=(
            "A local preview of coupled poloidal shear-Alfvén perturbations in a circular tokamak, "
            "evolved with LinearMHD, including its magnetosonic propagator. "
            r"The hollow torus has :math:`0.1\le r\le1`, major radius :math:`R_0=10`, "
            r"and a periodic one-sixth toroidal sector. The AdhocTorus equilibrium uses "
            r":math:`B_0=3`, :math:`q_0=1.71`, :math:`q_1=1.87` and :math:`\beta=0.0018`. "
            r"The initial logical velocity combines :math:`m=10,11`, :math:`n=-1` sector modes "
            r"with amplitude parameter :math:`10^{-3}` and Gaussian profiles centered at "
            r":math:`\eta_1=0.5` with width :math:`0.1`; the poloidal component uses their radial derivatives. "
            "The coarse grid and short duration are intended for exploration, not a converged TAE benchmark. "
            "Plots show physical minor-radial, poloidal and toroidal velocity on the φ=0 slice; "
            "each component keeps its own fixed color range throughout the animation."
        ),
        params_path=__file__, env=env, time_opts=time_opts,
        domain=domain, equil=equil, grid=grid, derham_opts=derham_opts,
    )
    return simulation

def pproc(sim: Simulation):

    started = perf_counter()
    output = sim.output
    plot_results(output, perf_counter() - started)


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
