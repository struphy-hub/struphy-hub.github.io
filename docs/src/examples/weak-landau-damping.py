"""Weak Landau damping: the canonical Vlasov-Ampère verification case.

A tiny electrostatic perturbation in a uniform, collisionless plasma damps
exponentially as particles phase-mix with the self-consistent field --
Landau damping. This benchmark validates the coupled Vlasov-Ampère PIC
discretization against the analytically known damping rate.

Adapted from Struphy's maintained example (examples/VlasovAmpereOneSpecies/weak_Landau_damping).

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
from struphy.models import VlasovAmpereOneSpecies

model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
model.em_fields.e_field.save_data = True

# A periodic box of length 4*pi (k = 0.5), resolved by a single, low-degree element row.
domain = domains.Cuboid(r1=12.56)
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))
time_opts = Time(dt=0.05, Tend=20.0, split_algo="LieTrotter")

# 1000 particles per cell, sorted into boxes for the control-variate weighting.
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=1000),
    weights_params=WeightsParameters(control_variate=True),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
    saving_params=SavingParameters(),
    bufsize=0.4,
)

model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# A single small-amplitude cosine mode perturbs an otherwise uniform Maxwellian.
perturbation_amplitude = 0.001
background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)
perturbation = perturbations.ModesCos(amps=(perturbation_amplitude,), ls=(1,))
model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="weak_landau_damping",
)
sim = Simulation(
    model=model,
    name="Weak Landau damping",
    description=(
        "A tiny electrostatic perturbation phase-mixes away in a collisionless "
        "plasma — the classic Landau-damping benchmark, compared here against "
        "its analytically known damping rate."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    sim.run()

    # The exact linear damping rate/frequency for k = 0.5 (Cuboid r1 = 4*pi),
    # from the Vlasov-Ampère dispersion relation (see the struphy verification test).
    def field_energy_exact(t):
        r, omega_r, omega_i, phi = 0.3677, 1.4156, -0.1533, 0.5362
        return (4 * perturbation_amplitude * r * np.exp(omega_i * t) * np.cos(omega_r * t - phi)) ** 2 * np.pi

    with h5py.File(os.path.join(env.path_out, "data", "data_proc0.hdf5"), "r") as f:
        time = np.asarray(f["time"]["value"])
        field_energy = np.asarray(f["scalar"]["electric_energy"])

    # Fit the damping rate from the envelope maxima, for comparison with omega_i = -0.1533.
    log_energy = np.log(field_energy)
    d_log = (np.roll(log_energy, -1) - np.roll(log_energy, 1))[1:-1] / (2 * time_opts.dt)
    is_maximum = (d_log[:-1] > 0) & (d_log[1:] < 0)
    maxima_t = time[1:-1][:-1][is_maximum]
    maxima_e = log_energy[1:-1][:-1][is_maximum]
    # Only fit the clean exponential-decay region -- once the signal drops
    # below the discrete-particle noise floor (around t ~ 8 here), later
    # envelope maxima track PIC noise rather than the physical damping.
    clean = maxima_t < 8.0
    measured_rate = float(np.polyfit(maxima_t[clean], maxima_e[clean], 1)[0] / 2)
    print(f"Measured damping rate: {measured_rate:.4f} (exact: -0.1533)")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=field_energy, mode="lines", name="Struphy (PIC)", line={"color": "#168aad", "width": 3}),
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
        legend={"x": 0.98, "y": 0.98, "xanchor": "right", "bgcolor": "rgba(255,255,255,0.82)"},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )

    png_path = Path("weak-landau-damping.png")
    html_path = Path("weak-landau-damping.html")
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

    metadata_path = Path("weak-landau-damping.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredDampingRate"] = measured_rate
    metadata["exactDampingRate"] = -0.1533
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
