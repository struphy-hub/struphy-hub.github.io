"""Weibel instability: temperature anisotropy generates a magnetic field.

A plasma with a hotter perpendicular temperature than parallel temperature is
unstable to spontaneous magnetic field generation: small magnetic
perturbations grow exponentially, tapping the excess perpendicular thermal
energy, until they're strong enough to isotropize the distribution.

Adapted from Struphy's maintained example
(examples/VlasovMaxwellOneSpecies/weibel_instability), at reduced particle
count and run length to keep it a quick gallery run.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import argparse

import numpy as np
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
from struphy.models import VlasovMaxwellOneSpecies


wavenumber = 1.25

# A colder parallel (vth1) than perpendicular (vth2) thermal spread -- the
# temperature anisotropy that Weibel feeds on.
vth1 = 0.02 / np.sqrt(2)
vth2 = vth1 * np.sqrt(12)
# A binned (v1, v2) distribution at every step, to follow the temperature anisotropy.



# A tiny seed perturbation in B_z, needed to trigger the (otherwise exact) instability.
magnetic_perturbation_amplitude = -1e-4

def create_simulation() -> Simulation:
    model = VlasovMaxwellOneSpecies(alpha=1.0, epsilon=-1.0, measure_gauss_law=True)
    model.em_fields.e_field.save_data = True
    domain = domains.Cuboid(r1=2 * np.pi / wavenumber)
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    time_opts = Time(dt=0.1, Tend=200.0, split_algo="LieTrotter")
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
    simulation = Simulation(
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
    return simulation

def pproc(sim: Simulation):

    from _gallery import (
        export_profiling,
        merge_metadata,
        save_extra_figure,
        save_figure,
        space_time_figure,
    )

    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    output = sim.output.process(create_vtk=False)

    # Total magnetic (B3) field energy at each saved time, summed over the grid.
    b_field = output.fields.em_fields.b_field
    times = np.asarray(b_field.t)
    grid_shape = tuple(b_field.sizes[dim] for dim in ("e1", "e2", "e3"))
    cell_volume = float(np.prod([1.0 / max(n - 1, 1) for n in grid_shape]))
    magnetic_energy = np.asarray((b_field.isel(component=2) ** 2).sum(("e1", "e2", "e3"))) * cell_volume / 2
    time = np.asarray(times)

    # Fit the growth rate over the clean exponential window (roughly the
    # middle third of the run, before saturation).
    growth_window = (time > time[-1] / 5) & (time < 2 * time[-1] / 5)
    growth_rate = float(np.polyfit(time[growth_window], np.log(magnetic_energy[growth_window]), 1)[0] / 2)
    print(f"Measured growth rate (in |B3|, from the energy fit): {growth_rate:.5f}")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=time,
                y=magnetic_energy,
                mode="lines",
                name="|B₃|² / 2 (Struphy)",
                line={"color": "#168aad", "width": 3},
            ),
        ],
    )
    figure.update_layout(
        title="Weibel instability: magnetic field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="|B₃|² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    save_figure(figure, "weibel-instability")

    # The magnetic field along x, over time.
    magnetic_field = b_field.isel(component=2, e2=0, e3=0)  # (t, e1)
    space_time = space_time_figure(
        magnetic_field,
        space="e1",
        x_values=magnetic_field.e1.values * domain.params["r1"],
        title="Weibel instability: magnetic field B₃(x, t)",
        colorbar_title="B₃",
    )

    # The temperature anisotropy from the binned (v1, v2) distribution: the ratio of the velocity
    # variances, which the instability reduces by heating the cold direction.
    f = output.evaluate("kinetic_ions/v1_v2_density/f")  # (t, v1, v2)
    moments = f.struphy.analysis.velocity_moments()
    anisotropy = moments.variance_v2 / moments.variance_v1
    anisotropy_figure = go.Figure(
        go.Scatter(
            x=anisotropy.t.values,
            y=anisotropy.values,
            mode="lines",
            line={"color": "#168aad", "width": 3},
        ),
    )
    anisotropy_figure.update_layout(
        title="Weibel instability: temperature anisotropy",
        xaxis_title="t [a.u.]",
        yaxis_title="⟨(v₂ − u₂)²⟩ / ⟨(v₁ − u₁)²⟩",
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    figures = [
        save_extra_figure(
            space_time,
            "weibel-instability",
            "space-time",
            alt="Space-time map of the magnetic field of the Weibel instability",
            caption=(
                "The magnetic field B₃(x, t). The seeded cosine mode, one wavelength across the box, grows in place around fixed nodes. It remains small until t ≈ 100 and reaches |B₃| ≈ 0.1 by the end of the run."
            ),
        ),
        save_extra_figure(
            anisotropy_figure,
            "weibel-instability",
            "anisotropy",
            alt="Temperature anisotropy as a function of time",
            caption=(
                "The temperature anisotropy: the ratio of the velocity variances along v₂ and v₁, computed from the binned (v₁, v₂) distribution with velocity_moments. It starts near 12, the ratio set by the initial temperatures, stays there while the field is small, and falls to about 3 by t = 200 as the magnetic field grows."
            ),
        ),
    ]

    profiling = export_profiling(sim, "weibel-instability")

    merge_metadata(
        "weibel-instability",
        measuredGrowthRate=growth_rate,
        figures=figures,
        **profiling,
    )


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
