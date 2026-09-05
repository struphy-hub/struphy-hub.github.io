"""Diocotron instability: a sheared E×B ring develops rippled edges.

An annular ring of charge, confined by a magnetic field, has a sheared E×B
rotation profile at its inner and outer edges. That shear is unstable: a
tiny azimuthal (mode number m) perturbation grows, rippling the ring's edges
-- the onset of the diocotron instability, a non-neutral-plasma analogue of
the Kelvin-Helmholtz instability, which given enough time rolls those
ripples up into a rotating pattern of discrete vortices.

Adapted from Struphy's maintained example
(examples/ToyGyrokinetic/diocotron_instability), at reduced resolution and
run length to keep it a quick gallery run. Parameters follow Crouseilles,
Mehrenberger & Vecil (2014), https://doi.org/10.1140/epjd/e2014-50180-9.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import json
from pathlib import Path

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
    perturbations,
)
from struphy.models import ToyDrift

model = ToyDrift(epsilon=1.0, alpha=1.0, base_units=BaseUnits(kBT=1.0))

# An annular ring, r in [1, 10], with a uniform background field.
domain = domains.HollowCylinder(a1=1.0, a2=10.0, Lz=10.0)
equil = equils.HomogenSlab()
grid = grids.TensorProductGrid(num_elements=(32, 64, 1), mpi_dims_mask=(False, True, False))
derham_opts = DerhamOptions(degree=(3, 3, 1), bcs=(("dirichlet", "dirichlet"), None, None))
time_opts = Time(dt=0.02, Tend=25.0, split_algo="LieTrotter")

# A binned e1-e2 (radial-angular) density snapshot at every step, for the animation.
density_bins = BinningPlot(slice="e1_e2", n_bins=(64, 64), ranges=((0.0, 1.0), (0.0, 1.0)))
model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=20, loading="sobol_standard", spatial="disc"),
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

# A uniform-density ring between r = 4 and r = 5, seeded with a tiny m = 4 azimuthal mode.
r_minus, r_plus, mode_number = 4.0, 5.0, 4
a1, a2 = domain.params["a1"], domain.params["a2"]
eta_minus, eta_plus = (r_minus - a1) / (a2 - a1), (r_plus - a1) / (a2 - a1)


def ring_density(etas, r_minus=r_minus, r_plus=r_plus):
    radial = a1 + (a2 - a1) * etas[:, 0]
    return 1.0 * ((r_minus <= radial) & (radial < r_plus))


model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(0.0, None)))
perturbation = perturbations.ModesCos(amps=(1e-6,), ms=(mode_number,), perb_domain=((eta_minus, eta_plus), None, None))
model.kinetic_ions.var.add_initial_condition(maxwellians.GyroMaxwellian2D(n=(ring_density, perturbation)))

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="diocotron_instability",
)
sim = Simulation(
    model=model,
    name="Diocotron instability",
    description=(
        "A sheared E×B ring of charge is unstable: a tiny azimuthal "
        "perturbation grows, rippling the ring's edges — the onset of a "
        "rotating pattern of vortices."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    sim.run()
    sim.pproc(create_vtk=False)
    sim.load_plotting_data()

    density = sim.f.kinetic_ions.e1_e2_density
    radius = a1 + (a2 - a1) * density.grid_e1
    angle_deg = 360.0 * density.grid_e2
    frames_data = density.f_binned  # (n_saved_times, n_radius_bins, n_angle_bins)
    times = np.linspace(0.0, time_opts.Tend, len(frames_data))

    # A simple measure of how far the ring has departed from its initial,
    # axisymmetric shape: the standard deviation of density around each
    # radius, averaged over the ring -- ~0 initially, growing as vortices form.
    asymmetry = np.array([float(np.mean(np.std(frame, axis=1))) for frame in frames_data])
    growth_window = (times > 5.0) & (times < 15.0)
    growth_rate = float(np.polyfit(times[growth_window], np.log(asymmetry[growth_window] + 1e-12), 1)[0])
    print(f"Measured asymmetry growth rate: {growth_rate:.4f}")

    frames = [
        go.Frame(name=f"{t:.1f}", data=[go.Heatmap(z=frame, x=angle_deg, y=radius, zmin=0, zmax=1.2, colorscale="Viridis")])
        for t, frame in zip(times, frames_data)
    ]
    # Default to the final (most visibly rippled) frame -- both for the
    # interactive page's initial view and for the static PNG export, which
    # can only ever capture the base `data`, not the animation frames.
    figure = go.Figure(
        data=[go.Heatmap(z=frames_data[-1], x=angle_deg, y=radius, zmin=0, zmax=1.2, colorscale="Viridis", colorbar={"title": "density"})],
        frames=frames,
    )
    figure.update_layout(
        title="Diocotron instability: ring density n(r, θ)",
        xaxis_title="θ [deg]",
        yaxis_title="r [a.u.]",
        template="plotly_white",
        autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.28,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {"args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": frame.name, "method": "animate"}
                    for frame in frames
                ],
                "active": len(frames) - 1,
                "x": 0.12,
                "len": 0.88,
                "y": -0.18,
                "currentvalue": {"prefix": "t = "},
            },
        ],
    )

    png_path = Path("diocotron-instability.png")
    html_path = Path("diocotron-instability.html")
    figure.write_image(png_path, width=1100, height=750, scale=2)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {html_path.resolve()}")

    metadata_path = Path("diocotron-instability.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredGrowthRate"] = growth_rate
    metadata["modeNumber"] = mode_number
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
