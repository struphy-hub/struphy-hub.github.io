"""Vortex merger: two like-signed blobs of charge co-rotate and merge.

In a strong magnetic field, a cloud of charge drifts with the E×B velocity of its own electric
field. Two blobs of the same sign therefore circle around each other, and when they start close
enough they merge into one: the guiding-centre analogue of the merger of two like-signed vortices
in a two-dimensional incompressible fluid.

Follows the setup of Struphy's diocotron example (examples/ToyGyrokinetic/diocotron_instability):
an annulus with grounded walls, a uniform background field, and the ToyDrift model.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import numpy as np
import plotly.graph_objects as go

from struphy import (
    BaseUnits,
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
    equils,
    grids,
    maxwellians,
)
from struphy.models import ToyDrift

model = ToyDrift(epsilon=1.0, alpha=1.0, base_units=BaseUnits(kBT=1.0))

# An annulus between r = 1 and r = 6 with grounded walls and a uniform background field.
domain = domains.HollowCylinder(a1=1.0, a2=6.0, Lz=10.0)
equil = equils.HomogenSlab()
grid = grids.TensorProductGrid(num_elements=(32, 64, 1), mpi_dims_mask=(False, True, False))
derham_opts = DerhamOptions(degree=(3, 3, 1), bcs=(("dirichlet", "dirichlet"), None, None))
time_opts = Time(dt=0.02, Tend=20.0, split_algo="LieTrotter")

# A binned radial-angular density snapshot at every step.
density_bins = BinningPlot(slice="e1_e2", n_bins=(48, 96), ranges=((0.0, 1.0), (0.0, 1.0)))
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=60, loading="sobol_standard", spatial="disc"),
    # The markers are loaded over the whole annulus; those outside the blobs carry no weight and are removed.
    weights_params=WeightsParameters(control_variate=True, reject_weights=True, threshold=0.0001),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(8, 8, 1), do_sort=True, sorting_frequency=5),
    saving_params=SavingParameters(binning_plots=(density_bins,)),
    bufsize=2.0,
)

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options()
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
    algo="discrete_gradient_1st_order_newton",
    evaluate_e_field=True,
)

# Two Gaussian blobs of peak density `peak` and width `width`, on the circle of radius `ring_radius`,
# with their centres `separation` apart.
peak, width, ring_radius, separation = 3.0, 0.55, 3.5, 2.0
half_angle = np.arcsin(separation / (2.0 * ring_radius))
a1, a2 = domain.params["a1"], domain.params["a2"]
centres = [(ring_radius * np.cos(sign * half_angle), ring_radius * np.sin(sign * half_angle)) for sign in (1, -1)]


def two_blobs(etas):
    radius = a1 + (a2 - a1) * etas[:, 0]
    angle = 2.0 * np.pi * etas[:, 1]
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    return sum(peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * width**2)) for cx, cy in centres)


model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(0.0, None)))
model.kinetic_ions.var.add_initial_condition(maxwellians.GyroMaxwellian2D(n=(two_blobs, None)))

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="vortex_merger",
)
sim = Simulation(
    model=model,
    name="Vortex merger",
    description=(
        "Two blobs of charge of the same sign circle each other under E×B drift "
        "and merge into one, the guiding-centre analogue of the merger of two "
        "like-signed vortices."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    output = sim.run(profiling_activated=True)
    output.pproc()
    print("scalars:", list(output.scalars.data_vars))
    f = output.evaluate("kinetic_ions/e1_e2_density/f")
    print(f.dims, f.shape, float(f.max()), float(f.isel(t=0).max()))
