"""Strong (nonlinear) Landau damping: particle trapping in a Vlasov-Ampère plasma.

A large-amplitude electrostatic perturbation drives the plasma into the
nonlinear regime: particles get trapped in the potential wells of the
self-consistent field, and the field energy no longer decays monotonically
like weak Landau damping -- it bounces as trapped particles slosh back and
forth, a signature of nonlinear kinetic trapping.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/strong_Landau_damping).

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

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

model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
model.em_fields.e_field.save_data = True

# A periodic box of length 4*pi (k = 0.5), resolved by a single, low-degree element row.
domain = domains.Cuboid(r1=12.56)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions()
time_opts = Time(dt=0.05, Tend=75.0, split_algo="LieTrotter")

# 1000 particles per cell, sorted into boxes for the control-variate weighting.
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

# A large-amplitude cosine mode, well past the linear (weak-damping) regime.
perturbation_amplitude = 0.5
background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)
perturbation = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="strong_landau_damping",
)
sim = Simulation(
    model=model,
    name="Strong Landau damping",
    description=(
        "A large-amplitude perturbation drives a Vlasov-Ampère plasma into "
        "the nonlinear regime — particles trap in the field's potential "
        "wells, and the field energy bounces instead of decaying smoothly."
        r" The initial Maxwellian has density $$n(x,0)=1+0.5\cos(2\pi x/L),\qquad L=12.56,$$"
        r" zero mean velocity and thermal speed :math:`v_{\mathrm{th}}=1`."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    from _gallery import export_profiling, merge_metadata, save_figure

    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    sim.run(profiling_activated=True)

    output = sim.output
    field_energy_array = output.evaluate("electric_energy")
    time = np.asarray(field_energy_array.t)
    field_energy = np.asarray(field_energy_array)

    # The bounce period of trapped particles shows up as the spacing between
    # local maxima in the field energy, once the initial (linear) damping
    # phase has given way to nonlinear trapping oscillations.
    maxima_t = np.asarray(output.envelope(field_energy_array).t)
    bounce_period = float(np.mean(np.diff(maxima_t))) if len(maxima_t) > 1 else float("nan")
    print(f"Estimated trapped-particle bounce period: {bounce_period:.2f}")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=field_energy, mode="lines", name="Struphy (PIC)", line={"color": "#168aad", "width": 3}),
        ],
    )
    figure.update_layout(
        title="Strong Landau damping: electric field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="E² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    save_figure(figure, "strong-landau-damping")

    profiling = export_profiling(sim, "strong-landau-damping")

    merge_metadata(
        "strong-landau-damping",
        bouncePeriod=bounce_period,
        **profiling,
    )
