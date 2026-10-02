"""The ITPA toroidal Alfvén eigenmode (TAE) benchmark with the reduced ShearAlfven model.

The setup of Struphy's `examples/ShearAlfven/itpa_tae_benchmark`: an m = 10, 11 perturbation of
the velocity in one sixth of a circular tokamak, meant to excite the toroidal Alfvén eigenmode in the gap
between the shear-Alfvén continua of the two harmonics. ShearAlfven keeps only the shear-Alfvén part of LinearMHD (no
magnetosonic propagator and no density or pressure), so this is the same benchmark without
compressional coupling; compare with itpa-tae-linear-mhd.py. The measured frequencies are drawn against those
continua and the gap-centre estimate, with the radial eigenfunction of each harmonic.

Requires Struphy with compiled kernels (`struphy compile`) and plasma-plots with Plotly
(`pip install "plasma-plots[plotly]==0.1.1"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).

The parameters are those of the benchmark: 24 x 96 x 16 cells of degree 3 to t = 500, hours on a
cluster (`python scripts/submit_precomputed_run.py itpa-tae-shear-alfven --account ... --partition ...` from the
repository root submits it as a Slurm job, profiles it and packs its output). The website does not rerun it: its CI downloads the archived run listed in
`.github/precomputed-examples.json` and only post-processes it.
"""

import argparse

import numpy as np

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
from struphy.models import ShearAlfven
from plasma_plots import save_figure
from plasma_plots.theory.waves import alfven_continuum, tae_frequency

STEM = "itpa-tae-shear-alfven"
NUM_ELEMENTS = (24, 96, 16)
DEGREE = (3, 3, 3)
END_TIME = 500.0
DT = 0.1
SAVE_STEP = 10

# The torus is one sixth of a full one: the sector mode n = -1 is the full-torus mode n = -6.
TOR_PERIOD = 6
MODES = (10, 11)
SECTOR_MODE = -1
AMPLITUDE = 1e-3


def minor_radius(eta1):
    """The minor radius of the hollow torus, 0.1 <= r <= 1."""
    return 0.1 + 0.9 * eta1


def create_simulation() -> Simulation:
    model = ShearAlfven(base_units=BaseUnits())
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
    for field in (model.em_fields.b_field, model.mhd.velocity):
        field.save_data = True

    domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, pol_period=1, tor_period=TOR_PERIOD)
    equil = equils.AdhocTorus(
        a=1.0, R0=10.0, B0=3.0, q_kind=0, p_kind=1,
        q0=1.71, q1=1.87, p1=0.95, p2=0.05, beta=0.0018,
    )
    grid = grids.TensorProductGrid(num_elements=NUM_ELEMENTS)
    derham_opts = DerhamOptions(degree=DEGREE, bcs=(("dirichlet", "dirichlet"), None, None))

    # Radial (sin) and poloidal (cos) logical components of the velocity 2-form, the poloidal
    # one from the radial derivative of the Gaussian, so that the perturbation is nearly
    # divergence-free.
    model.mhd.velocity.add_perturbation(perturbations.TorusModesSin(
        ms=MODES, ns=(SECTOR_MODE,) * 2, amps=(AMPLITUDE,) * 2,
        pfuns=("exp", "exp"), pfun_params=([0.5, 0.1], [0.5, 0.1]),
        comp=0, given_in_basis="2",
    ))
    model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(
        ms=MODES, ns=(SECTOR_MODE,) * 2, amps=tuple(AMPLITUDE / (2 * np.pi * m) for m in MODES),
        pfuns=("d_exp", "d_exp"), pfun_params=([0.5, 0.1], [0.5, 0.1]),
        comp=1, given_in_basis="2",
    ))

    env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="itpa_tae_shear_alfven", save_step=SAVE_STEP)
    sim = Simulation(
        model=model,
        name="ITPA TAE benchmark (ShearAlfven)",
        description=(
            "The ideal-MHD part of the ITPA toroidal Alfvén eigenmode benchmark: a global Alfvén "
            "wave in the gap that toroidicity opens between the shear-Alfvén continua of two "
            "neighbouring poloidal harmonics, set up here for exploration. "
            r"A hollow torus :math:`0.1\le r\le 1`, :math:`R_0=10`, one sixth of the full torus, "
            r"carries the AdhocTorus equilibrium with :math:`B_0=3`, "
            r":math:`q(r)=1.71+0.16\,r^2` and :math:`\beta=0.0018`. "
            r"The continua of :math:`m=10` and :math:`m=11` at :math:`n=-6` cross where "
            r":math:`q=10.5/6=1.75`, at :math:`r=0.5`, near "
            r"$$\omega_\mathrm{TAE}=\frac{v_A}{2qR_0}\approx 0.096,\qquad v_A=\frac{B_0}{\sqrt{n(r)}}.$$ "
            "The velocity is seeded with these two harmonics, Gaussian in radius around r = 0.55, "
            "on 24 × 96 × 16 cells of degree 3, to t = 500 (about eight TAE periods). "
            "ShearAlfven evolves only the shear-Alfvén part of the linearized MHD equations: "
            "velocity and magnetic field, with no magnetosonic coupling, density or pressure "
            "perturbation. The figures post-process an archived run of this script; "
            "it takes hours on a cluster."
        ),
        params_path=__file__, env=env, time_opts=Time(dt=DT, Tend=END_TIME),
        domain=domain, equil=equil, grid=grid, derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation, show: bool = False):
    """Save (and show) the figures from an existing simulation."""
    output = sim.output
    equil = sim.equil
    R0 = sim.domain.params["R0"]

    # The physical velocity, rotated to the radial, poloidal and toroidal directions of the torus.
    output.pproc(physical=True, celldivide=(1, 1, 1), create_vtk=False)
    velocity = output.evaluate("mhd/velocity_xyz").plasma.analysis.toroidal_components(R0=R0)
    times = velocity.t.values
    if not np.isclose(times[-1], END_TIME):
        raise RuntimeError(f"Run stopped at t={times[-1]}, before requested t={END_TIME}.")
    if not np.isfinite(velocity.values).all():
        raise RuntimeError("Non-finite velocity: refusing to export the example.")
    u_r = velocity.sel(component="radial", drop=True).rename("u_r")
    u_r.attrs.update(label="$u_r$")

    # Theory: the shear-Alfvén continua of the seeded harmonics, and the gap frequency.
    n_full = SECTOR_MODE * TOR_PERIOD

    def v_A(r):
        return equil.params["B0"] / np.sqrt(equil.n_r(r))

    def continuum(r, m, n):
        omega = alfven_continuum(r, m, n, equil.q_r, major_radius=R0, alfven_speed=v_A)
        return {"shear Alfvén": omega.real}

    q_gap = (MODES[0] + 0.5) / abs(n_full)
    r_gap = float(np.sqrt((q_gap - equil.params["q0"]) / (equil.params["q1"] - equil.params["q0"])))
    omega_tae = float(tae_frequency(q_gap, major_radius=R0, alfven_speed=v_A(r_gap)))
    print(f"TAE gap: q = {q_gap:.3f} at r = {r_gap:.3f}, omega_TAE = {omega_tae:.4f}")

    # The measured frequency: the strongest peak of the radial velocity, refined between bins.
    peaks = u_r.plasma.analysis.spectral_peaks(n_peaks=2, window="hann")
    omega = float(peaks.omega_refined[0])
    measured = ", ".join(f"{w:.4f}" for w in peaks.omega_refined.values)
    print(f"Strongest frequencies of u_r: {measured} (resolution {2 * np.pi / (times[-1] - times[0]):.4f}); "
          f"omega_TAE = {omega_tae:.4f}")

    # Main figure: where each frequency lives in radius, against the continua.
    radial = u_r.plasma.plot.radial_power(
        window="hann", x_of=minor_radius, continuum=(continuum, [(m, n_full) for m in MODES]),
        omega_max=0.5, backend="plotly",
    )
    save_figure(radial, STEM, show=show)

    spectrum = u_r.plasma.plot.power_spectrum(
        window="hann", peaks=2, omega_max=0.5,
        frequencies={"ω_TAE (gap centre)": omega_tae}, title="Power spectrum of u_r", backend="plotly",
    )
    save_figure(spectrum, f"{STEM}-spectrum", show=show)

    profiles = u_r.plasma.plot.mode_profiles(omega, x_of=minor_radius, top=4, backend="plotly")
    save_figure(profiles, f"{STEM}-eigenfunction", show=show)

    amplitudes = u_r.plasma.plot.mode_amplitudes(top=4, backend="plotly")
    save_figure(amplitudes, f"{STEM}-mode-amplitudes", show=show)

    movie = u_r.plasma.plot.animation(
        coords="physical", plane="RZ", eta3=0, symmetric=True, cmap="RdBu_r", max_frames=40,
        title="u_r on the poloidal plane φ = 0", backend="plotly",
    )
    save_figure(movie, f"{STEM}-poloidal", frame=20, show=show)

    energies = output.plot.energies(backend="plotly")
    save_figure(energies, f"{STEM}-energy", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the ITPA TAE benchmark with ShearAlfven.")
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
