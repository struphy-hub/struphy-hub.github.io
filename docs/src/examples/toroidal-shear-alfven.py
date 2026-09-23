"""A small toroidal LinearMHD run with the m=10,11 shear-Alfvén perturbation.

Run from the repository root with:
    .venv/bin/python cli.py run toroidal-shear-alfven

Requires compiled Struphy kernels and Plotly's PNG exporter (see README.md).
The default is an exploratory local run, not a converged ITPA TAE benchmark.
Increase NUM_ELEMENTS to (24, 96, 16), DEGREE to (3, 3, 3), and END_TIME
to 500.0 to recover the supplied spatial/time resolution. The equilibrium,
sector, mode numbers, Gaussian profiles and amplitudes are retained.
"""

from time import perf_counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import map_coordinates

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.models import LinearMHD

STEM = "toroidal-shear-alfven"
NUM_ELEMENTS = (8, 48, 4)
DEGREE = (3, 3, 2)
END_TIME = 20.0
DT = 0.5
SAVE_STEP = 2

model = LinearMHD(base_units=BaseUnits())
model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options()
for field in (model.em_fields.b_field, model.mhd.density, model.mhd.velocity, model.mhd.pressure):
    field.save_data = True

domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, pol_period=1, tor_period=6)
equil = equils.AdhocTorus(
    a=1.0, R0=10.0, B0=3.0, q_kind=0, p_kind=1,
    q0=1.71, q1=1.87, p1=0.95, p2=0.05, beta=0.0018,
)
grid = grids.TensorProductGrid(num_elements=NUM_ELEMENTS)
derham_opts = DerhamOptions(degree=DEGREE, bcs=(("dirichlet", "dirichlet"), None, None))

# These are logical H(div) / 2-form components, not physical unit-vector
# components. TorusModes uses phase 2*pi*(m*eta2 + n*eta3); n=-1 is one
# oscillation across the 1/6-torus sector, not a full-torus n=-1 mode.
modes = (10, 11)
amplitude = 1e-3
model.mhd.velocity.add_perturbation(perturbations.TorusModesSin(
    ms=modes, ns=(-1, -1), amps=(amplitude, amplitude),
    pfuns=("exp", "exp"), pfun_params=([0.5, 0.1], [0.5, 0.1]),
    comp=0, given_in_basis="2",
))
model.mhd.velocity.add_perturbation(perturbations.TorusModesCos(
    ms=modes, ns=(-1, -1), amps=tuple(amplitude / (2 * np.pi * m) for m in modes),
    pfuns=("d_exp", "d_exp"), pfun_params=([0.5, 0.1], [0.5, 0.1]),
    comp=1, given_in_basis="2",
))

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs", sim_folder="toroidal_shear_alfven",
    save_step=SAVE_STEP,
)
time_opts = Time(dt=DT, Tend=END_TIME)
sim = Simulation(
    model=model,
    name="Toroidal shear-Alfvén waves",
    description=(
        "A local preview of coupled poloidal shear-Alfvén perturbations in a circular tokamak, "
        "evolved with LinearMHD, including its magnetosonic propagator. "
        r"The hollow torus has :math:`0.1\le r\le1`, major radius :math:`R_0=10`, "
        r"and a periodic one-sixth toroidal sector. The AdhocTorus equilibrium uses "
        r":math:`B_0=3`, :math:`q_0=1.71`, :math:`q_1=1.87` and :math:`\beta=0.0018`. "
        r"The initial logical velocity combines :math:`m=10,11`, :math:`n=-1` sector modes "
        r"with amplitude parameter :math:`10^{-3}` and Gaussian profiles centered at "
        r":math:`\eta_1=0.5` with width :math:`0.1`; the poloidal component uses their radial derivatives. "
        "The coarse grid and short duration are intended for exploration, not a converged TAE benchmark. "
        "Plots show physical minor-radial, poloidal and toroidal velocity on the φ=0 slice; "
        "each component keeps its own fixed color range throughout the animation."
    ),
    params_path=__file__, env=env, time_opts=time_opts,
    domain=domain, equil=equil, grid=grid, derham_opts=derham_opts,
)

def plot_results(output, simulation_seconds):
    """Export the gallery figures; also usable with an existing Struphy Output."""
    from _gallery import export_profiling, merge_metadata, save_extra_figure, save_figure

    started = perf_counter()
    # More samples improve the display of the spline, not the simulation resolution.
    output.pproc(physical=True, celldivide=(3, 3, 1), create_vtk=False)
    velocity = output.evaluate("mhd/velocity_xyz").isel(e3=0).transpose("t", "component", "e1", "e2")
    times = velocity.t.values
    if not np.isclose(times[-1], END_TIME):
        raise RuntimeError(f"Run stopped at t={times[-1]}, before requested t={END_TIME}.")
    if not np.isfinite(velocity.values).all():
        raise RuntimeError("Non-finite velocity: refusing to export the example.")

    # At phi=0: e_R=e_x, e_phi=e_y, e_Z=e_z. Rotate the Cartesian
    # push-forward into orthonormal minor-radial and poloidal directions.
    theta = 2 * np.pi * velocity.e2.values
    ux, uy, uz = (velocity.isel(component=c).values for c in range(3))
    components = np.stack((
        ux * np.cos(theta) + uz * np.sin(theta),
        -ux * np.sin(theta) + uz * np.cos(theta),
        uy,
    ), axis=1)
    labels = ("u_r", "u_θ", "u_φ")
    titles = ("Minor-radial velocity", "Poloidal velocity", "Toroidal velocity")
    limits = np.maximum(np.abs(components).max(axis=(0, 2, 3)), 1e-16)

    # Resample the evaluated annulus onto pixels in the physical (R,Z) plane.
    # Use the actual evaluation coordinates; close the periodic angular seam.
    axis = np.linspace(-domain.params["a2"], domain.params["a2"], 112)
    pixel_r, pixel_z = np.meshgrid(axis, axis)
    radius = np.hypot(pixel_r, pixel_z)
    angle = np.mod(np.arctan2(pixel_z, pixel_r), 2 * np.pi) / (2 * np.pi)
    outside = (radius < domain.params["a1"]) | (radius > domain.params["a2"])
    eta1 = (radius - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])
    radial_index = np.interp(eta1, velocity.e1.values, np.arange(velocity.sizes["e1"]))
    periodic = velocity.e2.values < 1.0 - 1e-12
    angular_grid = np.append(velocity.e2.values[periodic], 1.0)
    angular_index = np.interp(angle, angular_grid, np.arange(len(angular_grid)))

    def traces(index):
        result = []
        for component, label in enumerate(labels):
            values = components[index, component][:, periodic]
            values = np.concatenate((values, values[:, :1]), axis=1)
            pixels = map_coordinates(values, [radial_index, angular_index], order=1, mode="nearest")
            pixels[outside] = np.nan
            result.append(go.Heatmap(
                x=axis + domain.params["R0"], y=axis, z=pixels.astype(np.float32),
                zmin=-limits[component], zmax=limits[component], colorscale="RdBu",
                colorbar={
                    "title": label, "x": (0.28, 0.64, 1.0)[component], "len": 0.55,
                    "thickness": 10, "tickformat": ".1e", "tickfont": {"size": 10},
                },
                hovertemplate=f"R=%{{x:.3f}}<br>Z=%{{y:.3f}}<br>{label}=%{{z:.3e}}<extra></extra>",
                xaxis=f"x{component + 1}" if component else "x",
                yaxis=f"y{component + 1}" if component else "y",
            ))
        return result

    figure = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.12)
    for column, trace in enumerate(traces(0), 1):
        figure.add_trace(trace, row=1, col=column)
    frame_indices = np.unique(np.linspace(0, len(times) - 1, min(26, len(times)), dtype=int))
    figure.frames = [go.Frame(name=f"{times[i]:g}", data=traces(i), traces=[0, 1, 2]) for i in frame_indices]
    figure.update_xaxes(
        title_text="R", range=[domain.params["R0"] - domain.params["a2"], domain.params["R0"] + domain.params["a2"]],
        constrain="domain", showgrid=False, zeroline=False,
    )
    for column in range(1, 4):
        figure.update_yaxes(
            title_text="Z" if column == 1 else None, range=[-domain.params["a2"], domain.params["a2"]],
            constrain="domain", showticklabels=column == 1, showgrid=False, zeroline=False,
            scaleanchor="x" if column == 1 else f"x{column}", scaleratio=1, row=1, col=column,
        )
    figure.update_layout(
        title="Toroidal shear-Alfvén waves · physical velocity at φ = 0",
        template="plotly_white", autosize=True, margin={"l": 55, "r": 95, "t": 95, "b": 130},
        sliders=[{
            "active": 0, "x": 0.12, "len": 0.86, "y": -0.12,
            "currentvalue": {"prefix": "t = "},
            "steps": [{"label": f.name, "method": "animate", "args": [[f.name], {
                "mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0},
            }]} for f in figure.frames],
        }],
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0, "y": -0.12, "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {
                "fromcurrent": True, "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0},
            }]},
            {"label": "Pause", "method": "animate", "args": [[None], {
                "mode": "immediate", "frame": {"duration": 0, "redraw": False},
            }]},
        ]}],
    )
    # A later snapshot also shows the generated toroidal component in the thumbnail.
    still = len(frame_indices) // 2
    save_figure(figure, STEM, static_data=traces(frame_indices[still]), static_active=still)

    radial_probe = int(np.abs(velocity.e1.values - 0.5).argmin())
    probe_radius = domain.params["a1"] + (domain.params["a2"] - domain.params["a1"]) * float(velocity.e1.values[radial_probe])
    history = make_subplots(rows=1, cols=3, subplot_titles=labels, horizontal_spacing=0.12)
    for component in range(3):
        history.add_trace(go.Heatmap(
            x=theta / (2 * np.pi), y=times, z=components[:, component, radial_probe],
            zmin=-limits[component], zmax=limits[component], colorscale="RdBu",
            colorbar={
                "title": labels[component], "x": (0.28, 0.64, 1.0)[component], "len": 0.8,
                "thickness": 10, "tickformat": ".1e", "tickfont": {"size": 10},
            },
        ), row=1, col=component + 1)
        history.update_yaxes(showticklabels=component == 0, range=[times[0], times[-1]], row=1, col=component + 1)
    history.update_xaxes(title_text="θ / 2π")
    history.update_yaxes(title_text="t", row=1, col=1)
    history.update_layout(
        title=f"Velocity around the ring r = {probe_radius:.3f}, φ = 0",
        template="plotly_white", margin={"l": 60, "r": 95, "t": 90, "b": 70},
    )

    energy = go.Figure()
    for key, label in (("en_U", "Kinetic"), ("en_B", "Magnetic"), ("en_thermal", "Compressional")):
        values = output.evaluate(key)
        if not np.isfinite(values.values).all():
            raise RuntimeError(f"Non-finite energy diagnostic: {key}")
        energy.add_scatter(x=values.t.values, y=values.values, mode="lines", name=label)
    energy.update_layout(
        title="Perturbation energies", xaxis_title="t", yaxis_title="Energy [normalized units]",
        template="plotly_white", margin={"l": 80, "r": 30, "t": 80, "b": 65},
    )
    figures = [
        save_extra_figure(history, STEM, "velocity-history",
            alt="Time histories of three physical velocity components around a poloidal ring",
            caption=f"Physical velocity at r={probe_radius:.3f}, φ=0, over the complete run. "
            "The angular structure starts with the m=10,11 perturbations. Each panel uses the same "
            "component color range as the slice animation; interpolated display pixels do not add simulation resolution."),
        save_extra_figure(energy, STEM, "energy",
            alt="Kinetic, magnetic and compressional perturbation energies over time",
            caption="Volume-integrated quadratic perturbation energies from LinearMHD. "
            "These exclude the equilibrium magnetic and thermal energies. They show the response of "
            "both the shear-Alfvén and magnetosonic propagators in the nonuniform toroidal equilibrium; "
            "this short coarse run does not establish a converged TAE frequency or growth rate."),
    ]
    merge_metadata(
        STEM, figures=figures, simulationSeconds=simulation_seconds,
        finalTime=float(times[-1]), savedFrames=len(times), **export_profiling(sim, STEM),
    )
    print(f"Simulation: {simulation_seconds:.1f} s; run and plots: {simulation_seconds + perf_counter() - started:.1f} s")


if __name__ == "__main__":
    started = perf_counter()
    output = sim.run(profiling_activated=True)
    plot_results(output, perf_counter() - started)
