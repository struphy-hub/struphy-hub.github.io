"""Wave packets in a cold magnetized plasma, with Struphy's ColdPlasma model.

A cold electron fluid in a uniform field B0 e_z carries two circularly polarized waves along the field: the right-hand R wave, with its
whistler branch below the electron cyclotron frequency and an upper branch above the R cutoff, and the left-hand L wave. Their dispersion
relations differ, and so do their group velocities d omega / d k. A Gaussian packet of E_x = env cos(k0 z), E_y = -env sin(k0 z), with no
magnetic field and no current, is launched from the middle of the box. Its two circular components, e^(+i k0 z) and e^(-i k0 z), have opposite
handedness relative to their direction of propagation, so the packet splits into an L-wave packet running one way and an R-type packet running the
other way: the L packet is fast, the R packet slow and spreading, since the whistler frequency depends strongly on k. The measured speed of
each packet is compared with the analytic group velocity d omega / d k of the cold-plasma dispersion relation.

Requires Struphy with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids
from struphy.initial.base import GenericPerturbation
from struphy.models import ColdPlasma
from struphy_plots import save_figure

# Plasma frequency equal to the cyclotron frequency, time in units of the inverse cyclotron frequency, c = 1.
alpha, epsilon, n0, B0z = 1.0, 1.0, 1.0, 1.0
length = 80.0
center = 0.5 * length
mode_number = 14
k0 = 2 * np.pi * mode_number / length  # the carrier wavenumber
width = 7.0  # of the Gaussian envelope
amplitude = 0.05


def envelope(z):
    return amplitude * np.exp(-0.5 * ((z - center) / width) ** 2)


def packet_energy(run):
    """Time, position and the transverse electric energy density |E_perp|^2(t, z) of a post-processed run."""
    e_field = run.evaluate("em_fields/e_field_xyz").isel(eta1=0, eta2=0)
    density = e_field.isel(component=0).values ** 2 + e_field.isel(component=1).values ** 2
    return e_field.t.values, e_field.eta3.values * length, density


def create_simulation() -> Simulation:
    time_opts = Time(dt=0.05, Tend=40.0)
    grid = grids.TensorProductGrid(num_elements=(1, 1, 256))
    derham_opts = DerhamOptions(degree=(1, 1, 3))
    domain = domains.Cuboid(r3=length)
    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=B0z, n0=n0)

    model = ColdPlasma(alpha=alpha, epsilon=epsilon)
    model.propagators.maxwell.options = model.propagators.maxwell.Options(algo="implicit")
    # The carrier of the packet: E_x = env cos(k0 z), E_y = -env sin(k0 z).
    model.em_fields.e_field.add_perturbation(
        GenericPerturbation(lambda x, y, z: envelope(z) * np.cos(k0 * z), given_in_basis="physical", comp=0)
    )
    model.em_fields.e_field.add_perturbation(
        GenericPerturbation(lambda x, y, z: -envelope(z) * np.sin(k0 * z), given_in_basis="physical", comp=1)
    )

    sim = Simulation(
        model=model,
        name="Cold-plasma wave packets",
        description=(
            "A Gaussian packet of circularly polarized electric field splits, in a magnetized cold plasma, into a fast L-wave packet "
            "and a slow, spreading R-wave packet that leave the launch point in opposite directions. Their speeds are the group "
            "velocities of the cold-plasma dispersion relation."
            r" The launch field is $$\mathbf{E}(z,0)=0.05e^{-(z-40)^2/(2\cdot7^2)}(\cos(k_0z),-\sin(k_0z),0),$$"
            r" with :math:`k_0=2\pi\cdot14/80`, :math:`\mathbf{B}_0=\mathbf{e}_z` and :math:`n_0=1`."
        ),
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="cold_plasma_wave_packet"),
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )
    return sim


def pproc(sim: Simulation, show: bool = False):
    from struphy_plots.theory.waves import Species, cold_plasma_waves, group_velocity

    time_opts = sim.time_opts

    output = sim.output
    output.pproc(physical=True)

    times, z, density = packet_energy(output)
    waves = ("whistler", "R wave", "L wave")
    tracked = ("whistler", "L wave")  # the slow packet holds the whistler and upper R branches, of almost equal speed
    # The branches along B0 (theta = 0) of the cold electron plasma. At k0 they are, by ascending frequency, the whistler,
    # the plasma oscillation (omega = omega_p), the L wave and the upper R wave.
    electrons = Species(plasma_frequency=alpha, cyclotron_frequency=-1.0)
    branch = {"whistler": "branch 1", "L wave": "branch 3", "R wave": "branch 4"}
    frequencies = cold_plasma_waves(k0, 0.0, electrons)
    speeds = group_velocity(lambda k: cold_plasma_waves(k, 0.0, electrons), k0, step=1e-4)
    exact_speed = {wave: float(speeds[branch[wave]].real) for wave in waves}
    exact_frequency = {wave: float(frequencies[branch[wave]].real) for wave in waves}

    def track_peak(side, speed):
        """Position of the energy peak of the packet on one side of the launch point, in a window around its characteristic."""
        positions = np.full(len(times), np.nan)
        for index, t in enumerate(times):
            window = np.abs(z - center - side * speed * t) < 4.0
            window &= (z > center) if side > 0 else (z < center)
            if t > 1.0 and window.any():
                positions[index] = z[window][np.argmax(density[index, window])]
        return positions

    # Which side holds which wave: the L packet is the faster one. The right-hand-side packet is the fast one if it stays
    # closer to the fast characteristic than to the slow one.
    def miss(side, speed):
        positions = track_peak(side, speed)
        good = (times > 4.0) & (times < 0.75 * center / speed) & np.isfinite(positions)
        return abs(np.polyfit(times[good], side * (positions[good] - center), 1)[0] - speed)

    fast_side = +1 if miss(+1, exact_speed["L wave"]) < miss(-1, exact_speed["L wave"]) else -1
    side_of = {"L wave": fast_side, "whistler": -fast_side, "R wave": -fast_side}
    measured, tracks = {}, {}
    for wave in tracked:
        side = side_of[wave]
        positions = track_peak(side, exact_speed[wave])
        # Fit while the packet is well separated from the launch region and still inside the box.
        reach = (center - 4.0) / exact_speed[wave]
        good = (times > 8.0) & (times < min(reach, time_opts.Tend)) & np.isfinite(positions)
        measured[wave] = float(np.polyfit(times[good], side * (positions[good] - center), 1)[0])
        tracks[wave] = side * (positions - center)
    if not all(np.isfinite(list(measured.values()))):
        raise RuntimeError("A packet speed could not be fitted")
    for wave in tracked:
        print(f"{wave}: group velocity {measured[wave]:.4f} (exact {exact_speed[wave]:.4f}), omega = {exact_frequency[wave]:.4f}")

    # The energy density in space and time, with the analytic characteristics on top.
    figure = go.Figure(go.Heatmap(z=np.sqrt(density / density.max()), x=z, y=times, colorscale="Plasma", zmin=0.0, zmax=1.0,
                                  colorbar={"title": "|E⊥| / peak"},
                                  hovertemplate="z=%{x:.2f}<br>t=%{y:.2f}<br>|E⊥|/peak=%{z:.3f}<extra></extra>"))
    line_colors = {"whistler": "#90e0ef", "R wave": "#b7e4c7", "L wave": "#ffd166"}
    for wave in waves:
        figure.add_scatter(x=center + side_of[wave] * exact_speed[wave] * times, y=times, mode="lines",
                           name=f"{wave}: v_g = {exact_speed[wave]:.3f}", line={"color": line_colors[wave], "width": 2, "dash": "dash"})
    figure.update_layout(
        title=f"A wave packet of carrier wavenumber k₀ = {k0:.3f} splits into an L and an R packet", template="plotly_white",
        autosize=True, xaxis_title="z [c/Ω_c]", yaxis_title="t [1/Ω_c]", legend={"orientation": "h", "y": -0.18},
        margin={"l": 75, "r": 30, "t": 80, "b": 110},
    )
    figure.update_xaxes(range=[0.0, length])
    figure.update_yaxes(range=[0.0, time_opts.Tend])
    save_figure(figure, "cold-plasma-wave-packet", width=1100, height=750, show=show)

    centroids = go.Figure()
    colors = {"whistler": "#168aad", "R wave": "#2a9d8f", "L wave": "#d62828"}
    for wave in ("L wave", "whistler"):
        packet = "L-wave packet" if wave == "L wave" else "R-type packet"
        centroids.add_scatter(x=times, y=tracks[wave], mode="markers", name=f"{packet}, Struphy",
                              marker={"color": colors[wave], "size": 5})
        centroids.add_scatter(x=times, y=exact_speed[wave] * times, mode="lines", name=f"{wave}: v_g t",
                              line={"color": colors[wave], "width": 1.5, "dash": "dash"})
    centroids.update_layout(
        title="Position of the two packets", template="plotly_white", autosize=True,
        xaxis_title="t [1/Ω_c]", yaxis_title="distance travelled from the launch point [c/Ω_c]",
        margin={"l": 75, "r": 30, "t": 80, "b": 60},
    )
    centroids.update_xaxes(range=[0.0, time_opts.Tend])
    centroids.update_yaxes(range=[0.0, center])
    save_figure(centroids, "cold-plasma-wave-packet-centroid", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the cold plasma wave packet example.")
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
