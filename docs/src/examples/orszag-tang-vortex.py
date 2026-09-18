"""The Orszag--Tang vortex: a nonlinear 2D MHD turbulence benchmark.

Two crossed, periodic velocity and magnetic-field vortices are evolved with
Struphy's full visco-resistive MHD model. Their interaction quickly produces
current sheets and magnetic islands, making this a compact test of nonlinear
field--fluid coupling and the divergence-preserving magnetic discretization.

Requires Struphy 3.2 with compiled kernels (``struphy compile``).
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from struphy import DerhamOptions, EnvironmentOptions, FieldsBackground, Simulation, Time, domains, grids, perturbations
from struphy.models import ViscoResistiveMHD

# The standard periodic Orszag--Tang initial condition on [0, 2π]².
model = ViscoResistiveMHD(with_viscosity=True, with_resistivity=True)
model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="full")
model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(mu=1e-3, mu_a=2e-3)
model.propagators.variat_resist.options = model.propagators.variat_resist.Options(eta=1e-3, eta_a=2e-3)

domain = domains.Cuboid(r1=2 * np.pi, r2=2 * np.pi, r3=1.0)
grid = grids.TensorProductGrid(num_elements=(64, 64, 1))
derham_opts = DerhamOptions(degree=(2, 2, 1))
time_opts = Time(dt=0.005, Tend=0.5, split_algo="LieTrotter")

# Uniform density/entropy and zero mean fields, then the two solenoidal
# sine-mode vortices: u = (-sin y, sin x, 0), B = (-sin y, sin 2x, 0).
model.mhd.density.add_background(FieldsBackground(values=(1.0,)))
model.mhd.entropy.add_background(FieldsBackground(values=(0.6,)))
model.mhd.velocity.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
model.em_fields.b_field.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
model.mhd.velocity.add_perturbation(perturbations.ModesSin(ms=(1,), amps=(-1.0,), Ly=2 * np.pi, comp=0))
model.mhd.velocity.add_perturbation(perturbations.ModesSin(ls=(1,), amps=(1.0,), Lx=2 * np.pi, comp=1))
model.em_fields.b_field.add_perturbation(perturbations.ModesSin(ms=(1,), amps=(-1.0,), Ly=2 * np.pi, comp=0))
model.em_fields.b_field.add_perturbation(perturbations.ModesSin(ls=(2,), amps=(1.0,), Lx=2 * np.pi, comp=1))

env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="orszag_tang_vortex")
sim = Simulation(
    model=model,
    name="Orszag–Tang vortex",
    description="Crossed velocity and magnetic vortices cascade into current sheets and magnetic islands — a classic nonlinear, two-dimensional MHD benchmark.",
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

    b_field = sim.spline_values.em_fields.b_field_log.data
    times = sorted(b_field.keys())
    x, y = sim.grids_phy[0][..., 0], sim.grids_phy[1][..., 0]

    def magnetic_pressure(t):
        bx, by = np.asarray(b_field[t][0])[..., 0], np.asarray(b_field[t][1])[..., 0]
        return 0.5 * (bx**2 + by**2)

    frame_indices = np.linspace(0, len(times) - 1, min(80, len(times)), dtype=int)
    frames = [
        go.Frame(name=f"{times[i]:.3f}", data=[go.Heatmap(x=x[:, 0], y=y[0, :], z=magnetic_pressure(times[i]).T, colorscale="Turbo")])
        for i in frame_indices
    ]
    figure = go.Figure(data=[go.Heatmap(x=x[:, 0], y=y[0, :], z=magnetic_pressure(times[-1]).T, colorscale="Turbo", colorbar={"title": "B² / 2"})], frames=frames)
    figure.update_layout(
        title="Orszag–Tang vortex: magnetic pressure",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        margin={"l": 70, "r": 35, "t": 80, "b": 135},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.25, "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True}]}]}],
        sliders=[{"active": len(frames) - 1, "x": 0.12, "len": 0.88, "y": -0.17, "currentvalue": {"prefix": "t = "}, "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": frame.name, "method": "animate"} for frame in frames]}],
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.write_image(Path("orszag-tang-vortex.png"), width=900, height=780, scale=2)
    figure.write_html(Path("orszag-tang-vortex.html"), include_plotlyjs="cdn", default_width="100%", default_height="100%", config={"responsive": True, "displaylogo": False})

    metadata_path = Path("orszag-tang-vortex.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["finalTime"] = time_opts.Tend
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
