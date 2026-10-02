"""Viscous decay of a velocity mode with smoothed-particle hydrodynamics.

The pressure-free `ViscousEulerSPH` model starts with a sinusoidal longitudinal
velocity, u(x, 0) = A sin(2 pi x / L), on a periodic interval. Viscosity alone
then damps that mode at 4 mu k^2 / 3, with k = 2 pi / L. The binned SPH current
makes a small, fast verification case with an analytic answer.

Requires Struphy >=3.4.0 with compiled kernels (`struphy compile`) and plasma-plots with Plotly
(`pip install "plasma-plots[plotly]==0.1.1"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np

from struphy import (
    BinningPlot,
    BoundaryParameters,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    perturbations,
)
from struphy.models import ViscousEulerSPH
from struphy.ode.utils import ButcherTableau
from plasma_plots import save_figure

length = 1.0
viscosity = 0.05
initial_amplitude = 0.5
boxes = 16
markers_per_box = 64
bins = 32
wavenumber = 2 * np.pi / length
exact_decay_rate = viscosity * 4 / 3 * wavenumber**2


def create_simulation() -> Simulation:
    # With pressure disabled, this is the particle discretisation of velocity diffusion.
    model = ViscousEulerSPH(with_B0=False, with_p=False, with_viscosity=True)
    model.propagators.push_eta.options = model.propagators.push_eta.Options(
        butcher=ButcherTableau(algo="forward_euler")
    )
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(
        kernel_type="gaussian_1d", mu=viscosity
    )

    domain = domains.Cuboid(r1=length)
    model.euler_fluid.set_markers(
        loading_params=LoadingParameters(ppb=markers_per_box, loading="tesselation"),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(boxes, 1, 1), dims_mask=(True, False, False)),
        saving_params=SavingParameters(
            binning_plots=(
                BinningPlot(slice="e1", n_bins=(bins,), ranges=(0.0, 1.0), output_quantity="current_1"),
            ),
        ),
    )
    model.euler_fluid.var.add_background(equils.ConstantVelocity())
    model.euler_fluid.var.add_perturbation(
        del_u1=perturbations.ModesSin(ls=(1,), amps=(initial_amplitude,))
    )

    sim = Simulation(
        model=model,
        name="SPH velocity diffusion",
        description=(
            "A sinusoidal velocity field diffuses on a periodic interval. The binned SPH current "
            "is compared with the exact viscous decay of the mode."
            r" The initial state is $$u_x(x,0)=0.5\sin(2\pi x),\qquad n(x,0)=1,$$"
            r" on :math:`0\le x<1`, with viscosity :math:`\mu=0.05`. The reference decay rate is :math:`\Gamma=\tfrac43\mu(2\pi)^2`."
        ),
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder="sph_velocity_diffusion"),
        time_opts=Time(dt=0.0025, Tend=0.3, split_algo="Strang"),
        domain=domain,
        grid=None,
        derham_opts=None,
    )
    return sim


def pproc(sim: Simulation, show: bool = False):
    output = sim.output
    output.pproc()
    velocity = output.evaluate("euler_fluid/e1_current_1/f")
    times = np.asarray(velocity.t.values)
    values = np.asarray(velocity.values)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite SPH velocity: refusing to publish the run")

    # The sine amplitude of the mode; a finite bin records the average of a sine wave, rather than its point value.
    mode_amplitude = velocity.plasma.analysis.project_mode(dim="eta1", number=1, bin_correction=True)
    exact_amplitude = initial_amplitude * np.exp(-exact_decay_rate * times)
    measured_decay_rate = -mode_amplitude.plasma.analysis.growth_rate().rate
    rms_error = float(mode_amplitude.plasma.analysis.error(exact_amplitude, norm="rms", dims="t"))
    print(
        f"decay rate {measured_decay_rate:.4f} (exact {exact_decay_rate:.4f}); "
        f"amplitude RMS error {rms_error:.4g}"
    )

    # The binned profile against the exact decaying mode, in at most 80 frames.
    figure = velocity.assign_attrs(label="SPH velocity u").plasma.plot.line_animation(
        x_of=lambda eta1: length * eta1,
        reference={"exact": lambda x, t: initial_amplitude * np.exp(-exact_decay_rate * t) * np.sin(wavenumber * x)},
        ylim=(-1.1 * initial_amplitude, 1.1 * initial_amplitude),
        step=-(-len(times) // 80),
        title="SPH velocity diffusion: sinusoidal mode against its exact decay",
        backend="plotly",
    )
    save_figure(figure, "sph-velocity-diffusion", show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the sph velocity diffusion example.")
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
