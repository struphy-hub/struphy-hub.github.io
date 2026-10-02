"""Weibel instability: temperature anisotropy generates a magnetic field.

A plasma with a hotter perpendicular temperature than parallel temperature is
unstable to spontaneous magnetic field generation: small magnetic
perturbations grow exponentially, tapping the excess perpendicular thermal
energy, until they're strong enough to isotropize the distribution.

Adapted from Struphy's maintained example
(examples/VlasovMaxwellOneSpecies/weibel_instability), at reduced particle
count and run length to keep it a quick gallery run.

Requires Struphy 3.2 with compiled kernels (`struphy compile`) and plasma-plots with Plotly
(`pip install "plasma-plots[plotly]==0.1.1"`). Run as a script, it saves its figures in the current
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
from struphy.models import VlasovMaxwellOneSpecies
from plasma_plots import save_figure

wavenumber = 1.25

# A colder parallel (vth1) than perpendicular (vth2) thermal spread -- the
# temperature anisotropy that Weibel feeds on.
vth1 = 0.02 / np.sqrt(2)
vth2 = vth1 * np.sqrt(12)

# A tiny seed perturbation in B_z, needed to trigger the (otherwise exact) instability.
magnetic_perturbation_amplitude = -1e-4


def create_simulation() -> Simulation:
    model = VlasovMaxwellOneSpecies(alpha=1.0, epsilon=-1.0, measure_gauss_law=True)
    model.em_fields.e_field.save_data = True
    domain = domains.Cuboid(r1=2 * np.pi / wavenumber)
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    time_opts = Time(dt=0.1, Tend=200.0, split_algo="LieTrotter")
    # A binned (v1, v2) distribution at every step, to follow the temperature anisotropy.
    velocity_bins = BinningPlot(slice="v1_v2", n_bins=(48, 48), ranges=((-0.12, 0.12), (-0.3, 0.3)))
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(
            Np=20_000,
            set_zero_velocity=(False, False, True),
            moments=(0.0, 0.0, 0.0, vth1, vth2, 1.0),
            seed=1234,
        ),
        # The control-variate weighting violates Gauss's law for this setup, so it's disabled here.
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(binning_plots=(velocity_bins,)),
        bufsize=2.0,
    )

    model.propagators.maxwell.options = model.propagators.maxwell.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(vth1=(vth1, None), vth2=(vth2, None)))
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesCos(amps=(magnetic_perturbation_amplitude,), ls=(1,), comp=2),
    )

    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="weibel_instability",
    )
    sim = Simulation(
        model=model,
        name="Weibel instability",
        description=(
            "A temperature-anisotropic plasma spontaneously generates a magnetic "
            "field: a tiny seed perturbation grows exponentially, tapping the "
            "excess perpendicular thermal energy."
            r" The initial thermal speeds satisfy $$v_{\mathrm{th},x}=0.02/\sqrt{2},\qquad v_{\mathrm{th},y}=\sqrt{12}\,v_{\mathrm{th},x},$$"
            r" giving :math:`T_y/T_x=12`. A cosine seed in the third magnetic component has amplitude :math:`-10^{-4}` and wavenumber :math:`k=1.25`."
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

    output = sim.output.pproc(create_vtk=False)

    # Total magnetic (B3) field energy at each saved time, summed over the grid.
    b_field = output.fields.em_fields.b_field
    grid_shape = tuple(b_field.sizes[dim] for dim in ("eta1", "eta2", "eta3"))
    cell_volume = float(np.prod([1.0 / max(n - 1, 1) for n in grid_shape]))
    magnetic_energy = (b_field.isel(component=2) ** 2).sum(("eta1", "eta2", "eta3")) * cell_volume / 2
    magnetic_energy.attrs = {"label": "|B₃|² / 2"}
    time = np.asarray(magnetic_energy.t)

    # Fit the growth rate over the clean exponential window (roughly the
    # middle third of the run, before saturation).
    growth_window = (time > time[-1] / 5) & (time < 2 * time[-1] / 5)
    growth_rate = float(np.polyfit(time[growth_window], np.log(magnetic_energy.values[growth_window]), 1)[0] / 2)
    print(f"Measured growth rate (in |B3|, from the energy fit): {growth_rate:.5f}")

    figure = magnetic_energy.plasma.plot.timeseries(
        logy=True,
        title="Weibel instability: magnetic field energy",
        backend="plotly",
    )

    save_figure(figure, "weibel-instability", show=show)

    # The magnetic field along x, over time.
    magnetic_field = b_field.isel(component=2, eta2=0, eta3=0)  # (t, e1)
    space_time = magnetic_field.assign_coords(eta1=magnetic_field.eta1.values * domain.params["r1"]).plasma.plot.slice(
        x="eta1",
        y="t",
        symmetric=True,
        cmap="RdBu_r",
        title="Weibel instability: magnetic field B₃(x, t)",
        xlabel="x [a.u.]",
        ylabel="t [a.u.]",
        colorbar_label="B₃",
        backend="plotly",
    )

    # The temperature anisotropy from the binned (v1, v2) distribution: the ratio of the velocity
    # variances, which the instability reduces by heating the cold direction.
    f = output.evaluate("kinetic_ions/v1_v2_density/f")  # (t, v1, v2)
    moments = f.plasma.analysis.velocity_moments()
    anisotropy = moments.variance_v2 / moments.variance_v1
    anisotropy.attrs = {"label": "⟨(v₂ − u₂)²⟩ / ⟨(v₁ − u₁)²⟩"}
    anisotropy_figure = anisotropy.plasma.plot.timeseries(
        logy=False,
        title="Weibel instability: temperature anisotropy",
        backend="plotly",
    )

    save_figure(space_time, "weibel-instability-space-time", show=show)
    save_figure(anisotropy_figure, "weibel-instability-anisotropy", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the weibel instability example.")
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
