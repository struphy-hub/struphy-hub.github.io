"""A damped standing wave in ViscoResistiveLinearMHD.

Equal kinematic viscosity and magnetic diffusivity give the exact solution
u_x = A exp(-nu*k**2*t) sin(k*z) cos(k*t),
b_x = A exp(-nu*k**2*t) cos(k*z) sin(k*t), for B0 = rho0 = 1.
The linear model dissipates the quadratic wave energy; heating is second order
and is not part of this linear perturbation system.

Requires Struphy with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np

from struphy import (
    DerhamOptions, EnvironmentOptions, Simulation, Time,
    domains, equils, grids, perturbations,
)
from struphy.models import ViscoResistiveLinearMHD

stem = "linear-dissipative-alfven-wave"
length, amplitude, diffusivity = 2.0 * np.pi, 0.1, 0.1


def create_simulation() -> Simulation:
    model = ViscoResistiveLinearMHD()
    model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="linear")
    model.propagators.variat_pb.options = model.propagators.variat_pb.Options(model="linear")
    model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(model="linear_p", mu=diffusivity)
    model.propagators.variat_resist.options = model.propagators.variat_resist.Options(model="linear_p", eta=diffusivity)
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(ns=(1,), amps=(amplitude,), comp=0, Lz=length, given_in_basis="physical"),
    )
    model.mhd.velocity.save_data = True
    model.em_fields.b_field.save_data = True
    domain = domains.Cuboid(r3=length)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 32))
    derham_opts = DerhamOptions(degree=(1, 1, 3))
    time_opts = Time(dt=0.025, Tend=2.0 * length, split_algo="Strang")
    sim = Simulation(
        model=model,
        name="Viscous and resistive linear Alfvén wave",
        description=(
            "A standing Alfvén wave loses energy through both viscosity and resistivity in "
            "ViscoResistiveLinearMHD. Equal diffusivities give a simple exponential envelope "
            "for the oscillating velocity and magnetic field."
            r" Initially $$u_x(z,0)=0.1\sin z,\qquad\delta\mathbf{B}(z,0)=0,$$"
            r" with :math:`\rho_0=B_0=1` on :math:`0\leq z<2\pi`."
            r" Both diffusivities are :math:`\nu=\eta=0.1`; the wave energy decays as :math:`e^{-0.2t}`."
        ),
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="linear_dissipative_alfven_wave"),
        time_opts=time_opts, domain=domain, grid=grid, derham_opts=derham_opts,
        equil=equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.1),
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
    from struphy_plots.theory.waves import dissipative_alfven

    output = sim.output
    output.pproc(physical=True)
    velocity = output.evaluate("mhd/velocity_xyz").isel(component=0, eta1=0, eta2=0)
    magnetic = output.evaluate("em_fields/b_field_xyz").isel(component=0, eta1=0, eta2=0)
    kinetic = output.scalars["en_U"]
    magnetic_energy = output.scalars["en_mag_1"]
    times, z = velocity.t.values, velocity.eta3.values * length
    if not all(np.isfinite(a.values).all() for a in (velocity, magnetic, kinetic, magnetic_energy)):
        raise RuntimeError("The linear MHD run produced non-finite diagnostics")
    if abs(times[-1] - time_opts.Tend) > time_opts.dt:
        raise RuntimeError("The linear MHD run did not reach the requested end time")
    # The k = 1 visco-resistive Alfvén wave: omega = 1 - 0.1i for v_A = 1 and nu = eta = 0.1.
    omega = dissipative_alfven(1.0, resistivity=diffusivity, viscosity=diffusivity)["forward"]
    frequency, decay = float(omega.real), -float(omega.imag)
    envelope = amplitude * np.exp(-decay * times[:, None])
    exact_u = envelope * np.cos(frequency * times[:, None]) * np.sin(z[None, :])
    exact_b = envelope * np.sin(frequency * times[:, None]) * np.cos(z[None, :])
    error = max(float(field.struphy.analysis.error(exact, norm="max", dims=("t", "eta3")))
                for field, exact in ((velocity, exact_u), (magnetic, exact_b))) / amplitude
    energy_time = kinetic.t.values
    wave_energy = kinetic.values + magnetic_energy.values
    exact_energy = np.exp(-2.0 * decay * energy_time)
    energy_error = float(np.max(np.abs(wave_energy / wave_energy[0] - exact_energy)))
    print(f"Maximum relative error against the exact solution: field {error:.2e}, energy {energy_error:.2e}")
    if max(error, energy_error) > 0.03:
        raise RuntimeError(f"The dissipative Alfvén wave differs from its exact solution: field={error:.3g}, energy={energy_error:.3g}")
    # sin(z) and cos(z) are mode 1 along eta3; project_mode drops the repeated periodic endpoint.
    u_mode = velocity.struphy.analysis.project_mode(dim="eta3", number=1, kind="sin").values / amplitude
    b_mode = magnetic.struphy.analysis.project_mode(dim="eta3", number=1, kind="cos").values / amplitude
    figure = make_subplots(rows=2, cols=1, vertical_spacing=0.18,
                           subplot_titles=("Damped velocity and magnetic modes", "Quadratic wave energy"))
    for values, label, color in ((u_mode, "Velocity mode", "#168aad"), (b_mode, "Magnetic mode", "#d62828")):
        figure.add_scatter(x=times, y=values, name=label, line={"color": color}, row=1, col=1)
    for sign in (-1, 1):
        figure.add_scatter(x=times, y=sign * np.exp(-decay * times), name="Exact envelope",
                           showlegend=sign == 1, line={"color": "#222", "dash": "dot"}, row=1, col=1)
    for values, label, color in ((kinetic.values, "Kinetic", "#168aad"),
                                 (magnetic_energy.values, "Magnetic", "#d62828"), (wave_energy, "Wave total", "#222")):
        figure.add_scatter(x=energy_time, y=values / wave_energy[0], name=label, line={"color": color}, row=2, col=1)
    figure.add_scatter(x=energy_time[::20], y=exact_energy[::20], mode="markers", name="Exact energy decay",
                       marker={"color": "#222", "symbol": "circle-open"}, row=2, col=1)
    figure.update_xaxes(title_text="t")
    figure.update_yaxes(title_text="mode amplitude / A", row=1, col=1)
    figure.update_yaxes(title_text="energy / initial wave energy", row=2, col=1)
    figure.update_layout(title="Viscous and resistive damping of a linear Alfvén wave", template="plotly_white",
                         legend={"orientation": "h", "y": -0.18}, margin={"l": 85, "r": 30, "t": 90, "b": 140})
    save(figure, stem, height=800, show=show)
    space_time = velocity.assign_coords(eta3=z).struphy.plot.slice(
        x="eta3", y="t", symmetric=True, cmap="RdBu_r", title="A standing Alfvén wave with a fading amplitude",
        xlabel="z", ylabel="t [a.u.]", colorbar_label="u_x", backend="plotly",
    )
    save(space_time, f"{stem}-space-time", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the linear dissipative alfven wave example.")
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
