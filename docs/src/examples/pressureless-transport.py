"""Exact density transport with VariationalPressurelessFluid.

Without pressure, a uniform velocity transports any smooth density profile
unchanged. Here rho = 1 + A cos(x - U t) makes one circuit of a periodic box.
This is an exact nonlinear solution, without characteristic crossing.

Requires Struphy with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np

from struphy import (
    DerhamOptions, EnvironmentOptions, FieldsBackground, Simulation, Time,
    domains, equils, grids, perturbations,
)
from struphy.models import VariationalPressurelessFluid

stem = "pressureless-transport"
length, speed, amplitude = 2.0 * np.pi, 0.5, 0.2
crossing_time = length / speed


def create_simulation() -> Simulation:
    model = VariationalPressurelessFluid()
    model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="pressureless")
    # The density 3-form and contravariant velocity include the coordinate mapping.
    model.fluid.density.add_background(FieldsBackground(values=(length,)))
    model.fluid.velocity.add_background(FieldsBackground(values=(speed / length, 0.0, 0.0)))
    model.fluid.density.add_perturbation(
        perturbations.ModesCos(ls=(1,), amps=(amplitude,), Lx=length, given_in_basis="physical"),
    )
    model.fluid.density.save_data = True
    model.fluid.velocity.save_data = True
    domain = domains.Cuboid(r1=length)
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))
    time_opts = Time(dt=crossing_time / 800, Tend=crossing_time, split_algo="Strang")
    sim = Simulation(
        model=model,
        name="Pressureless density transport",
        description=(
            "A density ripple travels once around a periodic box at constant speed. "
            "VariationalPressurelessFluid has no pressure force, so the exact nonlinear "
            "solution keeps the profile unchanged. Compare its shape, mass and kinetic energy."
            r" The initial data are $$\rho(x,0)=1+0.2\cos x,\qquad \mathbf{u}(x,0)=(0.5,0,0),$$"
            r" giving :math:`\rho(x,t)=1+0.2\cos(x-0.5t)` on :math:`0\leq x<2\pi`."
        ),
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="pressureless_transport"),
        time_opts=time_opts, domain=domain, grid=grid, derham_opts=derham_opts,
        equil=equils.HomogenSlab(B0z=1.0, n0=1.0),
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
    time_opts = sim.time_opts
    from plotly.subplots import make_subplots

    output = sim.output
    output.pproc(physical=True)
    rho = output.evaluate("fluid/density_xyz").isel(eta2=0, eta3=0)
    velocity = output.evaluate("fluid/velocity_xyz").isel(component=0, eta2=0, eta3=0)
    energy = output.scalars["kinetic_energy"]
    times, x, density = rho.t.values, rho.eta1.values * length, rho.values
    exact = 1.0 + amplitude * np.cos(x[None, :] - speed * times[:, None])
    if not all(np.isfinite(a).all() for a in (density, velocity.values, energy.values)):
        raise RuntimeError("The transport run produced non-finite diagnostics")
    if not np.isclose(times[-1], time_opts.Tend) or np.min(density) <= 0:
        raise RuntimeError("The transport run is incomplete or has non-positive density")
    errors = np.max(np.abs(density - exact), axis=1) / amplitude
    velocity_error = float(np.max(np.abs(velocity.values - speed)) / speed)
    energy_drift = float(np.max(np.abs(energy.values / energy.values[0] - 1.0)))
    # Sampled mass uses the periodic evaluation grid, dropping its repeated endpoint.
    mass = length * np.mean(density[:, :-1], axis=1)
    mass_drift = float(np.max(np.abs(mass / mass[0] - 1.0)))
    print(f"Maximum profile error against exact transport: {errors.max():.2e} (relative to A)")
    if errors.max() > 0.05 or velocity_error > 0.01 or max(energy_drift, mass_drift) > 1e-3:
        raise RuntimeError("Pressureless transport failed its profile or conservation checks")

    figure = make_subplots(rows=2, cols=1, vertical_spacing=0.18,
                           subplot_titles=("The ripple moving around the box", "Error and conservation"))
    for fraction, color in ((0.0, "#168aad"), (0.25, "#f77f00"), (0.5, "#d62828"), (1.0, "#6a4c93")):
        i = int(np.argmin(np.abs(times - fraction * crossing_time)))
        figure.add_scatter(x=x, y=density[i], name=f"t = {times[i]:.2f}", line={"color": color}, row=1, col=1)
        figure.add_scatter(x=x[::4], y=exact[i, ::4], mode="markers", name="Exact transport",
                           showlegend=fraction == 0.0, marker={"color": color, "symbol": "circle-open"}, row=1, col=1)
    for t, values, label in ((times, errors, "Profile error / A"),
                              (times, np.abs(mass / mass[0] - 1.0), "Relative sampled-mass drift"),
                              (energy.t.values, np.abs(energy.values / energy.values[0] - 1.0), "Relative kinetic-energy drift")):
        figure.add_scatter(x=t, y=values, name=label, row=2, col=1)
    figure.update_xaxes(title_text="x", row=1, col=1)
    figure.update_yaxes(title_text="ρ", row=1, col=1)
    figure.update_xaxes(title_text="t", row=2, col=1)
    figure.update_yaxes(title_text="absolute relative error", row=2, col=1)
    figure.update_layout(title="Pressureless transport at constant velocity", template="plotly_white",
                         legend={"orientation": "h", "y": -0.18}, margin={"l": 85, "r": 30, "t": 90, "b": 140})
    save(figure, stem, height=800, show=show)
    movie = rho.assign_coords(eta1=x).struphy.plot.slice(
        x="eta1", y="t", symmetric=True, cmap="RdBu_r", title="Density transported around a periodic box",
        xlabel="x", ylabel="t [a.u.]", colorbar_label="ρ", backend="plotly",
    )
    save(movie, f"{stem}-space-time", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the pressureless transport example.")
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
