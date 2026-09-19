"""Landau damping followed to small amplitudes with a delta-f particle method.

The benchmark of the weak Landau damping example again: a Langmuir wave with wavenumber k = 0.5 in a
Maxwellian plasma, damped at the rate 0.1533. A full-f particle-in-cell run has to resolve a perturbation
of 0.1% of the density with markers that sample the whole Maxwellian, so its noise hides the wave after
about two e-foldings. Struphy's `LinearVlasovAmpereOneSpecies` evolves only the perturbation delta f around
the Maxwellian background. The markers carry weights for delta f alone, and the noise is smaller by the
size of the perturbation. The electric energy can be followed over four orders of magnitude, with the
same number of markers.

Requires Struphy 3.3 with compiled kernels (`struphy compile`).
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
from struphy.models import LinearVlasovAmpereOneSpecies

perturbation_amplitude = 0.001
# The exact linear solution for k = 0.5: the roots of the Vlasov-Ampère dispersion relation give the
# frequency 1.4156 and the damping rate 0.1533, and the residue and phase of the electric field.
residue, frequency, damping_rate, phase = 0.3677, 1.4156, 0.1533, 0.5362

model = LinearVlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False, with_E0=False)
model.em_fields.e_field.save_data = True

# A periodic box of length 4 pi (k = 0.5), resolved by a single, low-degree element row.
domain = domains.Cuboid(r1=12.56)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.05, Tend=30.0, split_algo="LieTrotter")

model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=200),
    weights_params=WeightsParameters(),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(),
    saving_params=SavingParameters(),
)
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.coupling_Eweights.options = model.propagators.coupling_Eweights.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# The background is a uniform Maxwellian; the initial condition adds the cosine mode, so that delta f is
# the difference between the two.
model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
mode = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, mode)))

env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="delta_f_landau_damping")
sim = Simulation(
    model=model,
    name="Delta-f Landau damping",
    description=(
        "Landau damping followed over four orders of magnitude of the electric energy with a delta-f "
        "particle method, which evolves only the perturbation of a Maxwellian and so has far less noise "
        "than a full-f particle-in-cell run."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
    derham_opts=derham_opts,
)


def field_energy_exact(t):
    """The exact electric energy of the mode, E^2 / 2 integrated over the box of length 4 pi."""
    return (4 * perturbation_amplitude * residue * np.exp(-damping_rate * t) * np.cos(frequency * t - phase)) ** 2 * np.pi


if __name__ == "__main__":
    from _gallery import export_profiling, is_root, merge_metadata, save_figure

    output = sim.run(profiling_activated=True)
    print(list(output.scalars))
