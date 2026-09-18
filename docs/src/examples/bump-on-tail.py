"""Bump-on-tail instability: a minority beam drives Langmuir waves.

A small ("bump") population of fast particles riding on the tail of an
otherwise Maxwellian distribution is a classic source of free energy: it
drives Langmuir waves unstable, transferring energy from the hot minority
population to the growing field until particle trapping saturates it.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/bump_on).

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

domain = domains.Cuboid(r1=62.83)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.1, Tend=60.0, split_algo="LieTrotter")

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=1000, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0)),
    weights_params=WeightsParameters(control_variate=True),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
    saving_params=SavingParameters(),
    bufsize=0.4,
)

model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# A 90% bulk Maxwellian plus a 10% "bump" population drifting at u1 = -4.5.
perturbation_amplitude = 0.05
perturbation = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
bulk = maxwellians.Maxwellian3D(n=(0.9, None), u1=(3.0, None))
bump = maxwellians.Maxwellian3D(n=(0.1, None), u1=(-4.5, None), vth1=(0.5, None))
model.kinetic_ions.var.add_background(bulk + bump)
init_bump = maxwellians.Maxwellian3D(n=(0.1, perturbation), u1=(-4.5, None), vth1=(0.5, None))
model.kinetic_ions.var.add_initial_condition(bulk + init_bump)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="bump_on_tail",
)
sim = Simulation(
    model=model,
    name="Bump-on-tail instability",
    description=(
        "A minority “bump” of fast particles on the tail of an otherwise "
        "Maxwellian distribution drives Langmuir waves unstable, feeding "
        "energy into the field until particle trapping saturates it."
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
    time = np.asarray(output.time)
    field_energy = np.asarray(output.scalars["electric_energy"])

    # Fit the exponential growth rate over the clean linear-growth window.
    linear = (time > 5.0) & (time < 25.0)
    growth_rate = float(np.polyfit(time[linear], np.log(field_energy[linear]), 1)[0] / 2)
    print(f"Measured growth rate: {growth_rate:.4f}")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=field_energy, mode="lines", name="Struphy (PIC)", line={"color": "#168aad", "width": 3}),
        ],
    )
    figure.update_layout(
        title="Bump-on-tail instability: electric field energy",
        xaxis_title="t [a.u.]",
        yaxis_title="E² / 2 [a.u.]",
        yaxis={"type": "log"},
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    save_figure(figure, "bump-on-tail")

    profiling = export_profiling(sim, "bump-on-tail")

    merge_metadata(
        "bump-on-tail",
        measuredGrowthRate=growth_rate,
        **profiling,
    )
