"""The Orszag--Tang vortex: a nonlinear 2D MHD turbulence benchmark.

Two periodic velocity and magnetic-field vortices are evolved to t = 0.5.
The example shows early nonlinear compression and field deformation, with
energy and discrete magnetic-divergence diagnostics.

Requires Struphy 3.3 with compiled kernels (``struphy compile``).
"""

import numpy as np
import plotly.graph_objects as go

from struphy import DerhamOptions, EnvironmentOptions, FieldsBackground, Simulation, Time, domains, equils, grids, perturbations
from struphy.models import ViscoResistiveMHD

# The standard periodic Orszag--Tang initial condition on [0, 2π]².
model = ViscoResistiveMHD(with_viscosity=False, with_resistivity=False)
model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model="full")

domain = domains.Cuboid(r1=2 * np.pi, r2=2 * np.pi, r3=1.0)
# A compact grid resolves the early nonlinear evolution at gallery scale.
grid = grids.TensorProductGrid(num_elements=(16, 16, 1))
derham_opts = DerhamOptions(degree=(2, 2, 1))
time_opts = Time(dt=0.005, Tend=0.5, split_algo="LieTrotter")
# The equilibrium supplies normalization; the evolved magnetic field has no guide component.
equil = equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.1)

# Logical volume forms include det(DF) = 4*pi**2, giving physical rho = 1, s = 0.6.
# Vector perturbations are explicitly specified in the physical basis.
# Uniform density/entropy and zero mean fields, then the two solenoidal
# sine-mode vortices: u = (-sin y, sin x, 0), B = (-sin y, sin 2x, 0).
model.mhd.density.add_background(FieldsBackground(values=(4 * np.pi**2,)))
model.mhd.entropy.add_background(FieldsBackground(values=(0.6 * 4 * np.pi**2,)))
model.mhd.velocity.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
model.em_fields.b_field.add_background(FieldsBackground(values=(0.0, 0.0, 0.0)))
model.mhd.velocity.add_perturbation(perturbations.ModesSin(ms=(1,), amps=(-1.0,), Ly=2 * np.pi, comp=0, given_in_basis="physical"))
model.mhd.velocity.add_perturbation(perturbations.ModesSin(ls=(1,), amps=(1.0,), Lx=2 * np.pi, comp=1, given_in_basis="physical"))
model.em_fields.b_field.add_perturbation(perturbations.ModesSin(ms=(1,), amps=(-1.0,), Ly=2 * np.pi, comp=0, given_in_basis="physical"))
model.em_fields.b_field.add_perturbation(perturbations.ModesSin(ls=(2,), amps=(1.0,), Lx=2 * np.pi, comp=1, given_in_basis="physical"))

env = EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="orszag_tang_vortex", max_runtime=3600)
sim = Simulation(
    model=model,
    name="Orszag–Tang vortex",
    description="Early nonlinear evolution of crossed velocity and magnetic vortices in ideal MHD, with density, magnetic field lines, pressure and conservation diagnostics.",
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)


if __name__ == "__main__":
    from plotly.subplots import make_subplots
    from _gallery import export_profiling, merge_metadata, save_extra_figure, save_figure

    output = sim.run(profiling_activated=True)
    output.pproc(physical=True, celldivide=2)
    if not all(bool(np.isfinite(value).all()) for value in output.scalars.values()):
        raise RuntimeError("Non-finite MHD diagnostics: refusing to publish the run")
    b = output.evaluate("em_fields/b_field_xyz").isel(e3=0)
    rho = output.evaluate("mhd/density_xyz").isel(e3=0).squeeze(drop=True)
    entropy = output.evaluate("mhd/entropy_xyz").isel(e3=0).squeeze(drop=True)
    times = b.t.values
    if times[-1] < time_opts.Tend - 0.5 * time_opts.dt or float(rho.min()) <= 0:
        raise RuntimeError("Incomplete MHD evolution or non-positive density")
    x, y = np.asarray(b.X)[:, 0], np.asarray(b.Y)[0, :]
    bx, by = (b.isel(component=i).transpose("t", "e1", "e2").values for i in (0, 1))
    rho_values = rho.transpose("t", "e1", "e2").values
    entropy_values = entropy.transpose("t", "e1", "e2").values
    gamma = model.propagators.variat_dens.options.gamma
    pressure = (gamma - 1) * rho_values**gamma * np.exp(entropy_values / rho_values)

    # Field lines of the in-plane B are contours of the flux function A_z, with Bx = dA/dy and By = -dA/dx.
    # A_z is recovered spectrally (the mean field vanishes); a duplicated periodic end point is dropped first.
    period = 2 * np.pi
    nx, ny = (n - 1 if abs(c[-1] - c[0] - period) < 1e-6 * period else n for n, c in ((len(x), x), (len(y), y)))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=period / nx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=period / ny)[None, :]
    k2 = np.where((kx == 0) & (ky == 0), 1.0, kx**2 + ky**2)
    bx_hat, by_hat = (np.fft.fft2(field[:, :nx, :ny], axes=(1, 2)) for field in (bx, by))
    flux = np.fft.ifft2(1j * (kx * by_hat - ky * bx_hat) / k2, axes=(1, 2)).real
    flux = np.pad(flux, ((0, 0), (0, len(x) - nx), (0, len(y) - ny)), mode="wrap")

    # The classic picture: the density in a jet colour scale with the magnetic field lines on top.
    if not all(np.isfinite(field).all() for field in (rho_values, pressure, flux)):
        raise RuntimeError("Non-finite field diagnostic")
    # Fixed levels over all times, so the lines follow the same flux surfaces as the field evolves.
    flux_levels = {"start": float(flux.min()), "end": float(flux.max()), "size": float(np.ptp(flux)) / 14}
    limits = (float(rho_values.min()), float(rho_values.max()))

    def traces(index):
        return [
            go.Heatmap(x=x, y=y, z=rho_values[index].T, zmin=limits[0], zmax=limits[1], colorscale="Jet",
                       colorbar={"title": "Density ρ"}),
            go.Contour(x=x, y=y, z=flux[index].T, contours={"coloring": "none", **flux_levels},
                       line={"color": "white", "width": 1}, showscale=False, hoverinfo="skip"),
        ]

    picks = np.unique(np.linspace(0, len(times) - 1, min(51, len(times)), dtype=int))
    figure = go.Figure(data=traces(0), frames=[go.Frame(name=f"{times[i]:.3f}", data=traces(i), traces=[0, 1]) for i in picks])
    figure.update_layout(
        title="Orszag–Tang vortex: density and magnetic field lines", template="plotly_white",
        margin={"l": 70, "r": 30, "t": 80, "b": 130},
        xaxis={"title": "x", "range": [0, period], "constrain": "domain"},
        yaxis={"title": "y", "range": [0, period], "scaleanchor": "x", "scaleratio": 1},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.12,
                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 60, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]}]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.07, "currentvalue": {"prefix": "t = "},
                  "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}], "label": frame.name, "method": "animate"} for frame in figure.frames]}],
    )
    save_figure(figure, "orszag-tang-vortex", width=900, height=850, static_data=traces(len(times)-1), static_active=len(picks)-1)

    scalars = output.scalars
    energy = scalars.en_tot
    drift = np.abs(energy / energy.isel(t=0) - 1)
    divergence = np.sqrt(np.maximum(scalars.tot_div_B, 0))
    diagnostics = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Energy channels", "Conservation diagnostics"), vertical_spacing=0.18)
    for key, label in (("en_U", "kinetic"), ("en_mag", "magnetic"), ("en_thermo", "internal"), ("en_tot", "total")):
        diagnostics.add_scatter(x=scalars.t.values, y=scalars[key].values, name=label, row=1, col=1)
    diagnostics.add_scatter(x=scalars.t.values, y=np.maximum(drift.values, 1e-16), name="|ΔW/W₀|", row=2, col=1)
    diagnostics.add_scatter(x=scalars.t.values, y=np.maximum(divergence.values, 1e-16), name="‖div B‖ L2", row=2, col=1)
    diagnostics.update_yaxes(title_text="energy", row=1, col=1)
    diagnostics.update_yaxes(type="log", exponentformat="power", row=2, col=1)
    diagnostics.update_xaxes(title_text="t", row=2, col=1)
    diagnostics.update_layout(template="plotly_white", margin={"l": 70, "r": 30, "t": 70, "b": 60})

    cut_index = int(np.argmin(np.abs(y - np.pi)))
    cut = go.Figure()
    for index, label in ((0, "initial"), (-1, "final")):
        cut.add_scatter(x=x, y=pressure[index, :, cut_index], name=label, mode="lines")
    cut.update_layout(title="Gas pressure along y = π", xaxis_title="x", yaxis_title="p", template="plotly_white")
    figures = [
        save_extra_figure(diagnostics, "orszag-tang-vortex", "conservation",
            alt="MHD energy channels and conservation diagnostics over time",
            caption="Kinetic, magnetic, internal and total energy, followed by the relative total-energy change and the L2 norm of the discrete magnetic divergence. The latter is the square root of tot_div_B, which stores the squared norm. Values below 1e-16 are clipped for display. Finite solver tolerances and time splitting affect energy conservation."),
        save_extra_figure(cut, "orszag-tang-vortex", "pressure-cut",
            alt="Initial and final gas pressure along the midplane",
            caption="Gas pressure along y = π at the beginning and end of this run. This is a diagnostic of the coarse-grid evolution, not a comparison with reference data or a convergence result."),
    ]
    merge_metadata("orszag-tang-vortex", finalTime=float(times[-1]), maxEnergyDrift=float(drift.max()),
                   maxDivB=float(divergence.max()), minDensity=float(rho.min()), figures=figures,
                   **export_profiling(sim, "orszag-tang-vortex"))
