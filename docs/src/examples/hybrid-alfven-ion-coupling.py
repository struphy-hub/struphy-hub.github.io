"""Hybrid MHD-kinetic energy exchange with Struphy's LinearMHDVlasovPC model.

A population of energetic ions, drifting close to the Alfvén speed along a
uniform background field, is evolved together with a linear-MHD shear-Alfvén
wave via the pressure-coupling scheme. The two subsystems continuously trade
energy back and forth through the coupling term while the discretization
conserves the total -- the defining property of a structure-preserving hybrid
scheme.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import numpy as np
import plotly.graph_objects as go

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
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
    perturbations,
)
from struphy.models import LinearMHDVlasovPC

model = LinearMHDVlasovPC()
model.em_fields.b_field.save_data = True

# A periodic domain aligned with the background field, so energetic ions can
# free-stream along B0 and resonate with the Alfvén wave.
domain = domains.Cuboid(r3=20.0)
equil = equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.1)
grid = grids.TensorProductGrid(num_elements=(1, 1, 32))
derham_opts = DerhamOptions(degree=(1, 1, 3))
time_opts = Time(dt=0.02, Tend=40.0)

# A modest population of energetic ions, sorted into boxes for efficient
# accumulation of the pressure-coupling term onto the field grid.
model.energetic_ions.set_markers(
    loading_params=LoadingParameters(Np=8000, seed=1),
    weights_params=WeightsParameters(),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(1, 1, 8), do_sort=True),
    saving_params=SavingParameters(),
)

model.propagators.push_eta_pc.options = model.propagators.push_eta_pc.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.pc6d.options = model.propagators.pc6d.Options()
model.propagators.shearalfven.options = model.propagators.shearalfven.Options()
model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options()

# A small-amplitude shear-Alfvén seed, and energetic ions drifting near the
# resonant parallel velocity u3 = v_A = 1 with a finite thermal spread.
model.mhd.velocity.add_background(FieldsBackground())
model.mhd.velocity.add_perturbation(perturbations.Noise(amp=1e-3, comp=0, seed=42))
model.energetic_ions.var.add_background(
    maxwellians.Maxwellian3D(n=(0.1, None), u3=(1.0, None), vth3=(0.3, None)),
)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="hybrid_alfven_ion_coupling",
)
sim = Simulation(
    model=model,
    name="Hybrid MHD-kinetic energy exchange",
    description=(
        "Energetic ions drifting near the Alfvén speed exchange energy with a "
        "shear-Alfvén wave through Struphy’s pressure-coupling scheme, while "
        "the total energy stays conserved."
        r" The energetic-ion Maxwellian is specified by $$n_h=0.1,\qquad u_{z,h}=1,\qquad v_{\mathrm{th},z,h}=0.3,$$"
        r" with transverse thermal speeds :math:`v_{\mathrm{th},x,h}=v_{\mathrm{th},y,h}=1`."
        r" The bulk has :math:`\mathbf{B}_0=\mathbf{e}_z`, :math:`n_0=1` and transverse velocity coefficient-noise amplitude :math:`10^{-3}`."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    from _gallery import export_profiling, merge_metadata, save_figure

    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    output = sim.run(profiling_activated=True)
    output.pproc()  # on every rank; under MPI the output is not processed implicitly

    # LinearMHDVlasovPC tracks each subsystem's energy as a scalar every step:
    # en_B/en_U (field/fluid), en_f (kinetic energetic ions), en_tot (total).
    time = np.asarray(output.time)
    en_B = np.asarray(output.scalars["en_B"])
    en_U = np.asarray(output.scalars["en_U"])
    en_f = np.asarray(output.scalars["en_f"])
    en_tot = np.asarray(output.scalars["en_tot"])

    relative_drift = float(np.max(np.abs(en_tot - en_tot[0]) / en_tot[0]))
    print(f"Max relative drift in total energy (should be ~0): {relative_drift:.2e}")

    figure = go.Figure(
        data=[
            go.Scatter(x=time, y=en_B, mode="lines", name="Field energy (en_B)", line={"color": "#168aad", "width": 2.5}),
            go.Scatter(x=time, y=en_U, mode="lines", name="Fluid kinetic energy (en_U)", line={"color": "#f77f00", "width": 2.5}),
            go.Scatter(x=time, y=en_f - en_f[0], mode="lines", name="Energetic-ion energy change (en_f − en_f₀)", line={"color": "#d62828", "width": 2.5}),
            go.Scatter(
                x=time,
                y=(en_tot - en_tot[0]) / en_tot[0],
                mode="lines",
                name="Total energy drift (relative)",
                line={"color": "#6a4c93", "width": 2, "dash": "dot"},
                yaxis="y2",
            ),
        ],
    )
    figure.update_layout(
        title="Hybrid MHD-kinetic energy exchange",
        xaxis_title="t [a.u.]",
        yaxis_title="Energy [a.u.]",
        yaxis2={
            "title": "Relative total-energy drift",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        template="plotly_white",
        autosize=True,
        legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.82)"},
        margin={"l": 70, "r": 70, "t": 80, "b": 60},
    )

    save_figure(figure, "hybrid-alfven-ion-coupling")

    profiling = export_profiling(sim, "hybrid-alfven-ion-coupling")

    merge_metadata(
        "hybrid-alfven-ion-coupling",
        relativeTotalEnergyDrift=relative_drift,
        **profiling,
    )
