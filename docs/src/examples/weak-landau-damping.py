"""Weak Landau damping: the canonical Vlasov-Ampère verification case.

A tiny electrostatic perturbation in a uniform, collisionless plasma damps
exponentially as particles phase-mix with the self-consistent field --
Landau damping. This benchmark validates the coupled Vlasov-Ampère PIC
discretization against the analytically known damping rate.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/weak_Landau_damping).

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import (
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


# A periodic box of length 4*pi (k = 0.5), resolved by a single, low-degree element row.

# 1000 particles per cell, sorted into boxes for the control-variate weighting.


# A single small-amplitude cosine mode perturbs an otherwise uniform Maxwellian.
perturbation_amplitude = 0.001

def create_simulation() -> Simulation:
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
    model.em_fields.e_field.save_data = True
    domain = domains.Cuboid(r1=12.56)
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    time_opts = Time(dt=0.05, Tend=20.0, split_algo="LieTrotter")
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=1000),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=2.0,
    )
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")
    background = maxwellians.Maxwellian3D(n=(1.0, None))
    model.kinetic_ions.var.add_background(background)
    perturbation = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))
    env = EnvironmentOptions(
        out_folders="struphy_gallery_runs",
        sim_folder="weak_landau_damping",
    )
    simulation = Simulation(
        model=model,
        name="Weak Landau damping",
        description=(
            "A tiny electrostatic perturbation phase-mixes away in a collisionless "
            "plasma — the classic Landau-damping benchmark, compared here against "
            r"the linear damping rate :math:`\gamma \approx -0.1533` at "
            r"wavenumber :math:`k \approx 0.5` in normalized units."
            r" The initial Maxwellian has density $$n(x,0)=1+10^{-3}\cos(2\pi x/L),\qquad L=12.56,$$"
            r" zero mean velocity and thermal speed :math:`v_{\mathrm{th}}=1`."
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

    # The exact linear damping rate/frequency for k = 0.5 (Cuboid r1 = 4*pi),
    # from the Vlasov-Ampère dispersion relation (see the struphy verification test).
    def field_energy_exact(t):
        r, omega_r, omega_i, phi = 0.3677, 1.4156, -0.1533, 0.5362
        return (4 * perturbation_amplitude * r * np.exp(omega_i * t) * np.cos(omega_r * t - phi)) ** 2 * np.pi

    output = sim.output
    field_energy = output.evaluate("electric_energy")
    time = np.asarray(field_energy.t)

    # Fit the damping rate from the envelope maxima, for comparison with omega_i = -0.1533.
    # Only fit the clean exponential-decay region -- once the signal drops
    # below the discrete-particle noise floor (around t ~ 8 here), later
    # envelope maxima track PIC noise rather than the physical damping.
    measured_rate = field_energy.struphy.analysis.damping_rate(window=(None, 8.0), amplitude=True).rate
    print(f"Measured damping rate: {measured_rate:.4f} (exact: -0.1533)")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=time,
                y=np.asarray(field_energy),
                mode="lines",
                name="Struphy (PIC)",
                line={"color": "#168aad", "width": 3},
            ),
            go.Scatter(
                x=time,
                y=field_energy_exact(time),
                mode="lines",
                name="Exact envelope",
                line={"color": "#d62828", "width": 2, "dash": "dot"},
            ),
        ],
    )
    figure.update_layout(
        title="Weak Landau damping: electric field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="E² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        legend={
            "x": 0.98,
            "y": 0.98,
            "xanchor": "right",
            "bgcolor": "rgba(255,255,255,0.82)",
        },
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    save_figure(figure, "weak-landau-damping")

    # The electric field along x, over time.
    output.pproc()
    electric_field = output.evaluate("em_fields/e_field").isel(component=0, e2=0, e3=0)  # (t, e1)
    space_time = space_time_figure(
        electric_field,
        space="e1",
        x_values=electric_field.e1.values * domain.params["r1"],
        title="Weak Landau damping: electric field E(x, t)",
        colorbar_title="E_x",
    )
    figures = [
        save_extra_figure(
            space_time,
            "weak-landau-damping",
            "space-time",
            alt="Space-time map of the electric field of the damped Langmuir wave",
            caption=(
                "The electric field E(x, t) of the run above. The single cosine mode is a standing wave: its sign alternates in time with a period of about 4.4 (ω ≈ 1.42) around nodes that stay in place, while its amplitude decays."
            ),
        ),
    ]

    profiling = export_profiling(sim, "weak-landau-damping")
    merge_metadata(
        "weak-landau-damping",
        measuredDampingRate=measured_rate,
        exactDampingRate=-0.1533,
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
