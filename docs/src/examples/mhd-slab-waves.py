"""The three MHD wave branches of a magnetized slab, with Struphy's LinearMHD model.

A uniform plasma in an oblique magnetic field carries three waves along z: the shear Alfvén wave, and
the slow and fast magnetosonic waves. Broadband noise in the velocity excites all of them at once. The
(k, omega) power spectrum of the velocity shows the Alfvén branch, that of the pressure the two
magnetosonic branches, and the speeds fitted to them are compared with the exact ideal-MHD values.

Adapted from Struphy's tutorial (tutorials/tutorial_linear_mhd_slab_waves_1d.ipynb).

Requires Struphy 3.3 with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, perturbations
from struphy.models import LinearMHD
from struphy_plots.theory.waves import magnetosonic_speeds

# The background: B0 = (0, 1, 1), density 0.7 and a plasma beta of 3 (thermal over magnetic pressure).
B0x, B0y, B0z = 0.0, 1.0, 1.0
n0, beta, gamma = 0.7, 3.0, 5.0 / 3.0
B_squared = B0x**2 + B0y**2 + B0z**2
p0 = beta * B_squared / 2.0

# Ideal-MHD wave speeds along z, at the angle theta between z and B0: the shear Alfvén wave, and the slow
# and fast magnetosonic waves.
alfven_speed = np.sqrt(B_squared / n0)
sound_speed = np.sqrt(gamma * p0 / n0)
speeds = magnetosonic_speeds(np.arccos(B0z / np.sqrt(B_squared)), alfven_speed=alfven_speed, sound_speed=sound_speed)
exact_speeds = {"alfven": float(speeds["shear Alfvén"]), "slow": float(speeds["slow"]), "fast": float(speeds["fast"])}


def create_simulation() -> Simulation:
    model = LinearMHD()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(algo="implicit")

    # Broadband noise in all three velocity components, so that every branch is excited.
    for component in range(3):
        model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=component, seed=123))

    domain = domains.Cuboid(r3=60.0)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 64))
    derham_opts = DerhamOptions(degree=(1, 1, 3))
    equil = equils.HomogenSlab(B0x=B0x, B0y=B0y, B0z=B0z, beta=beta, n0=n0)
    time_opts = Time(dt=0.15, Tend=180.0)

    env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="mhd_slab_waves")
    sim = Simulation(
        model=model,
        name="MHD waves in a magnetized slab",
        description=(
            "Broadband noise excites the shear Alfvén wave and the slow and fast magnetosonic waves of a "
            "uniform, obliquely magnetized plasma. The power spectra of the velocity and the pressure show "
            "the three branches, and their fitted speeds are compared with the exact ideal-MHD values."
            r" The equilibrium parameters are $$\mathbf{B}_0=(0,1,1),\qquad n_0=0.7,\qquad p_0=\frac{\beta|\mathbf{B}_0|^2}{2}=3,$$"
            r" with :math:`\beta=3` and :math:`\gamma=5/3`. All three velocity components receive coefficient noise of amplitude :math:`0.1` in a periodic interval :math:`L_z=60`."
        ),
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def save(figure, name: str, *, show: bool = False, frame: int | None = None, still=None, width=1100, height=650):
    """Save a Plotly figure as ``<name>.html``, ``<name>.png`` and ``<name>.plotly.json``.

    ``figure`` is a plot of struphy-plots drawn with ``backend="plotly"``, or a
    ``plotly.graph_objects.Figure``. ``show`` shows it first. For an animation, the PNG shows
    ``frame`` (default: the first), or the figure ``still`` instead. Under MPI only rank 0 writes.
    """
    import struphy_plots
    from struphy_plots.plotting import PlotResult

    if not struphy_plots.is_plotting_rank():
        return
    result = figure if isinstance(figure, PlotResult) else PlotResult(figure, None)
    if show:
        result.show()
    result.save(f"{name}.html")
    image = PlotResult(still, None) if still is not None else result
    image.save(f"{name}.png", frame=frame, width=width, height=height, scale=2)
    result.save(f"{name}.plotly.json")


def pproc(sim: Simulation, show: bool = False):
    from plotly.subplots import make_subplots

    output = sim.output
    output.pproc(physical=True)

    from struphy_plots.analysis import fit_dispersion_branches, power_spectrum

    length = output.domain.params["r3"] - output.domain.params["l3"]

    def spectrum_along_z(field, n_branches, noise_level):
        """omega >= 0, k >= 0, the Fourier amplitude of a field along z, and fits of its straight branches.

        Peaks count above `noise_level` times the column's peak amplitude, i.e. its square times the peak power.
        """
        line = field.isel(eta1=0, eta2=0)
        line = line.isel(component=0) if "component" in line.dims else line
        spectrum = power_spectrum(line.assign_coords(eta3=line.eta3 * length), dim="eta3")  # physical z
        fits = fit_dispersion_branches(spectrum, n_branches=n_branches, noise_level=noise_level**2, order=10)
        quadrant = spectrum.sel(omega=spectrum.omega >= 0, k=spectrum.k >= 0)
        return quadrant.omega.values, quadrant.k.values, np.sqrt(quadrant.values), fits

    omega_u, k_u, spectrum_u, fit_u = spectrum_along_z(output.fields.mhd.velocity, 1, 0.5)
    omega_p, k_p, spectrum_p, fit_p = spectrum_along_z(output.fields.mhd.pressure, 2, 0.4)
    measured_speeds = {
        "alfven": float(fit_u[0].velocity),
        "slow": float(fit_p[0].velocity),
        "fast": float(fit_p[1].velocity),
    }
    for branch, exact in exact_speeds.items():
        print(f"{branch}: measured {measured_speeds[branch]:.4f}, exact {exact:.4f}")
    if not all(np.isfinite(v) for v in measured_speeds.values()):
        raise RuntimeError("A wave branch could not be fitted")

    def log_power(spectrum):
        power = np.asarray(spectrum) ** 2
        return np.log10(np.clip(power / power.max(), 1e-15, None))

    k_top = float(min(k_u[-1], k_p[-1]))
    colors = {"alfven": "#168aad", "slow": "#f4a261", "fast": "#d62828"}
    labels = {"alfven": "shear Alfvén", "slow": "slow magnetosonic", "fast": "fast magnetosonic"}
    panels = (
        ("Velocity u₁", omega_u, k_u, spectrum_u, ("alfven",)),
        ("Pressure", omega_p, k_p, spectrum_p, ("slow", "fast")),
    )
    figure = make_subplots(rows=1, cols=2, subplot_titles=[panel[0] for panel in panels], horizontal_spacing=0.12)
    for column, (_, omega, k, spectrum, branches) in enumerate(panels, start=1):
        figure.add_trace(
            go.Heatmap(
                x=np.asarray(k), y=np.asarray(omega), z=log_power(spectrum), zmin=-12, zmax=-1,
                colorscale="Plasma", showscale=column == 2,
                colorbar={"title": {"text": "log₁₀ P"}, "len": 0.9},
                hovertemplate="k=%{x:.3f}<br>ω=%{y:.3f}<br>log₁₀ P=%{z:.2f}<extra></extra>",
            ),
            row=1, col=column,
        )
        for branch in branches:
            figure.add_trace(
                go.Scatter(
                    x=[0, k_top], y=[0, exact_speeds[branch] * k_top], mode="lines",
                    name=f"{labels[branch]}: v = {exact_speeds[branch]:.3f} (exact), "
                         f"{measured_speeds[branch]:.3f} (fit)",
                    line={"color": colors[branch], "width": 3, "dash": "dash"}, legendgroup=branch,
                ),
                row=1, col=column,
            )
        figure.update_xaxes(title_text="k", range=[0, k_top], row=1, col=column)
        figure.update_yaxes(title_text="ω", range=[0, exact_speeds["fast"] * k_top], row=1, col=column)
    figure.update_layout(
        title="Three MHD wave branches in a magnetized slab", template="plotly_white",
        legend={"orientation": "h", "y": -0.2}, margin={"l": 70, "r": 40, "t": 90, "b": 110},
    )
    save(figure, "mhd-slab-waves", width=1300, height=650, show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the mhd slab waves example.")
    argparser.add_argument(
        "--pproc-only",
        action="store_true",
        help="Run post-processing on an existing simulation instead of running a new one.",
    )
    argparser.add_argument("--show", action="store_true", help="Show the figures before saving them.")
    args = argparser.parse_args()

    simulation = create_simulation()
    if not args.pproc_only:
        simulation.run()
    pproc(simulation, show=args.show)
