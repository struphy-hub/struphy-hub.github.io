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

import json
import os
from pathlib import Path

import h5py
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
from struphy.models import VlasovMaxwellOneSpecies

model = VlasovMaxwellOneSpecies(alpha=1.0, epsilon=-1.0, measure_gauss_law=True)
model.em_fields.e_field.save_data = True

wavenumber = 1.25
domain = domains.Cuboid(r1=2 * np.pi / wavenumber)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.1, Tend=200.0, split_algo="LieTrotter")

# A colder parallel (vth1) than perpendicular (vth2) thermal spread -- the
# temperature anisotropy that Weibel feeds on.
vth1 = 0.02 / np.sqrt(2)
vth2 = vth1 * np.sqrt(12)
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
    saving_params=SavingParameters(),
    bufsize=2.0,
)

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(vth1=(vth1, None), vth2=(vth2, None)))

# A tiny seed perturbation in B_z, needed to trigger the (otherwise exact) instability.
magnetic_perturbation_amplitude = -1e-4
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
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    sim.run()
    sim.pproc(create_vtk=False)
    sim.load_plotting_data()

    # Total magnetic (B3) field energy at each saved time, summed over the grid.
    b_field = sim.spline_values.em_fields.b_field_log.data
    times = sorted(b_field.keys())
    grid_shape = sim.grids_phy[0].shape
    cell_volume = float(np.prod([1.0 / max(n - 1, 1) for n in grid_shape]))
    magnetic_energy = np.array([float(np.sum(np.asarray(b_field[t][2]) ** 2)) * cell_volume / 2 for t in times])
    time = np.asarray(times)

    # Fit the growth rate over the clean exponential window (roughly the
    # middle third of the run, before saturation).
    growth_window = (time > time[-1] / 5) & (time < 2 * time[-1] / 5)
    growth_rate = float(np.polyfit(time[growth_window], np.log(magnetic_energy[growth_window]), 1)[0] / 2)
    print(f"Measured growth rate (in |B3|, from the energy fit): {growth_rate:.5f}")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=magnetic_energy, mode="lines", name="|B₃|² / 2 (Struphy)", line={"color": "#168aad", "width": 3}),
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

    png_path = Path("weibel-instability.png")
    html_path = Path("weibel-instability.html")
    figure.write_image(png_path, width=1100, height=650, scale=2)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {html_path.resolve()}")

    metadata_path = Path("weibel-instability.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredGrowthRate"] = growth_rate
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
