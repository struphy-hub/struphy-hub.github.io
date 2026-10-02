"""Electromagnetic waves along the magnetic field of a cold plasma, with Struphy's ColdPlasma model.

A cold electron fluid in a uniform magnetic field B0 = B0 e_z carries circularly polarized waves along
the field: the right-hand (R) wave with its whistler branch below the electron cyclotron frequency, and
the left-hand (L) wave. Both are cut off at low frequency, where the plasma reflects them. Broadband
noise in the transverse electric field excites all branches at once; the (k, omega) power spectrum of
E_x shows them, and is compared with the analytic cold-plasma dispersion relation.

Install dependencies and compile the kernels (Python 3.10 or newer):

    pip install "struphy[pproc]>=3.4.0"
    pip install "plasma-plots[plotly]>=0.1.1"
    struphy compile

PNG exports require Chrome or Chromium. If Chrome is not installed, run:

    kaleido_get_chrome

Save this file and run it from the directory where you want the output:

    python cold-plasma-waves.py

Figures are saved in the current directory; add --show to display them before saving.
"""

import argparse

import numpy as np

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, perturbations
from struphy.models import ColdPlasma
from plasma_plots import save_figure

# Plasma frequency equal to the cyclotron frequency (alpha = 1), and time in units of the inverse
# cyclotron frequency (epsilon = 1). The R cutoff is then at (1 + sqrt 5)/2 and the L cutoff at (sqrt 5 - 1)/2.
alpha, epsilon, n0, B0z = 1.0, 1.0, 1.0, 1.0
omega_R = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * alpha**2))
omega_L = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * alpha**2))


def parallel_branches(k):
    """Positive frequencies of the cold electron plasma for k along B0, in units of the cyclotron frequency.

    With c = 1, k^2 = omega^2 - omega_p^2 omega / (omega -+ Omega_c) for the R (upper sign) and L waves,
    i.e. the cubic omega^3 -+ Omega_c omega^2 - (omega_p^2 + k^2) omega +- k^2 Omega_c = 0. The R cubic has
    two positive roots (the whistler below Omega_c and the R wave above omega_R), the L cubic one.
    (The longitudinal plasma oscillation, omega = omega_p, lives in E_z, which the noise leaves at zero.)
    """
    omega_c, omega_p2 = 1.0, alpha**2
    whistler, r_wave, l_wave = [], [], []
    for kk in k:
        r_roots = np.sort(np.roots([1.0, -omega_c, -(omega_p2 + kk**2), kk**2 * omega_c]).real)
        l_roots = np.sort(np.roots([1.0, omega_c, -(omega_p2 + kk**2), -(kk**2) * omega_c]).real)
        whistler.append(r_roots[-2])
        r_wave.append(r_roots[-1])
        l_wave.append(l_roots[-1])
    return {
        "R-wave": np.array(r_wave),
        "L-wave": np.array(l_wave),
        "whistler": np.array(whistler),
    }


def create_simulation() -> Simulation:
    model = ColdPlasma(alpha=alpha, epsilon=epsilon)
    model.propagators.maxwell.options = model.propagators.maxwell.Options(algo="implicit")

    # Broadband noise in both transverse components of E excites the R and L waves together.
    for component in (0, 1):
        model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=component, seed=123))

    domain = domains.Cuboid(r3=40.0)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 128))
    derham_opts = DerhamOptions(degree=(1, 1, 3))
    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=B0z, n0=n0)
    time_opts = Time(dt=0.05, Tend=80.0)

    env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="cold_plasma_waves")
    sim = Simulation(
        model=model,
        name="Cold-plasma waves along a magnetic field",
        description=(
            "Broadband noise excites the right- and left-hand circularly polarized waves of a cold, magnetized "
            "electron plasma. The power spectrum of the transverse electric field shows the whistler branch below "
            "the cyclotron frequency and the two cutoffs, on top of the analytic cold-plasma dispersion relation."
            r" The background parameters are $$\mathbf{B}_0=\mathbf{e}_z,\qquad n_0=1,\qquad \alpha=\epsilon=1,$$"
            r" with transverse electric coefficient-noise amplitude :math:`0.1`, :math:`E_z(z,0)=0` and periodic length :math:`L_z=40`."
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
    output = sim.output
    output.pproc(physical=True)

    # The (k, omega) power spectrum of E_x along z, for omega, k >= 0.
    from plasma_plots.analysis import power_spectrum

    params = output.domain.params
    e_x = output.fields.em_fields.e_field.isel(component=0, eta1=0, eta2=0)
    e_x = e_x.assign_coords(eta3=e_x.eta3 * (params["r3"] - params["l3"]))  # physical z
    spectrum = power_spectrum(e_x, dim="eta3")
    quadrant = spectrum.sel(omega=spectrum.omega >= 0, k=spectrum.k >= 0)
    omega, k = quadrant.omega.values, quadrant.k.values
    power = quadrant.values / float(quadrant.max())

    # The analytic branches for propagation along B0.
    k_top, omega_top = 5.0, 4.0
    k_fine = np.linspace(1e-3, k_top, 400)
    branch_curves = parallel_branches(k_fine)

    # Read each branch off the spectrum: at every k, the frequency of the strongest power within a window
    # around the exact branch, and its relative distance from it.
    frequency_resolution = float(omega[1] - omega[0])
    errors = {}
    for name in branch_curves:
        deviations = []
        for column, kk in enumerate(k):
            if not 0.0 < kk <= k_top:
                continue
            target = float(np.interp(kk, k_fine, branch_curves[name]))
            window = np.flatnonzero(np.abs(omega - target) < 0.1 * target + 2.0 * frequency_resolution)
            if window.size and target <= omega_top:
                peak = omega[window[np.argmax(power[window, column])]]
                deviations.append(abs(peak / target - 1.0))
        errors[name] = float(np.median(deviations))
        print(f"{name}: median relative frequency error {errors[name]:.3f} over {len(deviations)} wavenumbers")
    if not all(np.isfinite(error) for error in errors.values()):
        raise RuntimeError("A wave branch could not be read off the spectrum")

    # The normalized power spectrum over 8 decades, with the analytic branches and the cutoffs.
    labels = {"R-wave": "R wave", "L-wave": "L wave", "whistler": "whistler (R, below Ω_c)"}
    figure = spectrum.plasma.plot.dispersion(
        kmin=0,
        kmax=k_top,
        omega_max=omega_top,
        dynamic_range=8,
        cmap="plasma",
        branches={label: (k_fine, branch_curves[name]) for name, label in labels.items()},
        frequencies={"Ω_c": 1.0, "ω_R cutoff": omega_R, "ω_L cutoff": omega_L},
        title="Cold-plasma waves along B₀: power spectrum of E_x",
        backend="plotly",
    )
    save_figure(figure, "cold-plasma-waves", height=700, show=show)

    # Energy channels: the noise starts purely electric, then shares its energy with the magnetic field
    # and the electron current, while the sum stays constant.
    energies = [output.scalars[name] for name in ("electric_energy", "magnetic_energy", "kinetic_energy", "total_energy")]
    relative_drift = float(energies[-1].plasma.analysis.relative_error().max())
    print(f"Maximum relative drift of the total energy: {relative_drift:.2e}")
    energy_figure = energies[0].plasma.plot.timeseries(
        *energies[1:], logy=False, title="Energy channels of the cold plasma", backend="plotly"
    )
    save_figure(energy_figure, "cold-plasma-waves-energy", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the cold plasma waves example.")
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
