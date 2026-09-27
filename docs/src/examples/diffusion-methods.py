"""Two particle methods for the diffusion equation: random walks and a deterministic drift.

A periodic density u(x, 0) = 1 + a cos(2 pi x) relaxes to the uniform state by diffusion,
du/dt = D u'', and the amplitude of the mode decays as a exp(-D k^2 t) with k = 2 pi. Struphy solves this
with particles in two ways:

* `RandomParticleDiffusion` moves every marker by a Wiener process, dx = sqrt(2 D) dB. The density is
  that of the markers, so the result is noisy, and the noise falls only as 1 / sqrt(N).
* `DeterministicParticleDiffusion` moves every marker with the velocity -D u'/u, where the density u is a
  finite element projection of the markers' own weights. There is no random noise, but the density
  estimate limits the accuracy.

Both start from the same markers and are compared with the exact decay.

Requires Struphy 3.3 with compiled kernels (`struphy compile`) and struphy-plots with Plotly
(`pip install "struphy-plots[plotly]"`). Run as a script, it saves its figures in the current
directory (`--show` shows them first).
"""

import argparse

import numpy as np
import plotly.graph_objects as go

from struphy import (
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import DeterministicParticleDiffusion, RandomParticleDiffusion
from struphy.ode.utils import ButcherTableau

diffusion = 0.05  # D
amplitude = 0.5  # of the initial density perturbation, relative to the uniform density 1
wavenumber = 2 * np.pi
decay_rate = diffusion * wavenumber**2  # the exact decay rate of the mode
markers = 100_000
bins = 32


def create_simulation(method="Random walk") -> Simulation:
    """Build either of the two hardcoded diffusion comparisons."""
    time_opts = Time(dt=0.005, Tend=1.0, split_algo="LieTrotter")
    domain = domains.Cuboid(r1=1.0)
    if method == "Random walk":
        model = RandomParticleDiffusion()
        propagator = model.propagators.rand_diff
        # The random walk supports forward Euler only: other tableaux would
        # add the noise once per stage and multiply the diffusion coefficient.
        propagator.options = propagator.Options(
            diff_coeff=diffusion, butcher=ButcherTableau(algo="forward_euler")
        )
        folder = "diffusion_random"
        grid = None
        derham_opts = None
    else:
        model = DeterministicParticleDiffusion()
        propagator = model.propagators.det_diff
        propagator.options = propagator.Options(diff_coeff=diffusion)
        folder = "diffusion_deterministic"
        # Only the deterministic method needs a finite element density estimate.
        grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
        derham_opts = DerhamOptions(degree=(3, 1, 1))

    model.hydrogen.set_markers(
        loading_params=LoadingParameters(Np=markers, loading="pseudo_random", seed=1608),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        saving_params=SavingParameters(
            binning_plots=(BinningPlot(slice="e1", n_bins=(bins,), ranges=(0.0, 1.0)),),
        ),
    )
    # A uniform background, and the cosine mode as the initial condition.
    model.hydrogen.var.add_background(maxwellians.ColdPlasma(n=(1.0, None)))
    mode = perturbations.ModesCos(amps=(amplitude,), ls=(1,))
    model.hydrogen.var.add_initial_condition(maxwellians.ColdPlasma(n=(1.0, mode)))
    return Simulation(
        model=model,
        name="Random and deterministic particle diffusion",
        description=(
            "A cosine density relaxes by diffusion. Struphy's random-walk and deterministic particle methods "
            "both follow it, and their density and decay of the mode are compared with the exact solution."
            r" Both methods use $$n(x,0)=1+0.5\cos(2\pi x),\qquad D=0.05,$$"
            r" on :math:`0\le x<1`. The reference mode amplitude is :math:`A(t)=0.5e^{-D(2\pi)^2t}`."
        ),
        env=EnvironmentOptions(out_folders="struphy_gallery_runs", sim_folder=folder),
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )


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

    runs = {
        "Random walk": sim.output,
        "Deterministic": create_simulation("Deterministic").output,
    }
    for run in runs.values():
        run.pproc()

    # The binned marker density of each method, (t, e1), and the exact solution on the same bins.
    density = {name: run.evaluate("hydrogen/e1_density/f") for name, run in runs.items()}
    times = density["Random walk"].t.values
    x = density["Random walk"].eta1.values
    exact_amplitude = amplitude * np.exp(-decay_rate * times)
    profile_exact = 1.0 + exact_amplitude[:, None] * np.cos(wavenumber * x)

    # The amplitude of the cosine mode relative to the mean. A bin average lowers a mode by sinc(k dx / 2).
    measured = {
        name: values.struphy.analysis.project_mode(dim="eta1", number=1, kind="cos", bin_correction=True)
        / values.mean("eta1")
        for name, values in density.items()
    }
    fitted_rate = {name: -abs(amp).struphy.analysis.growth_rate().rate for name, amp in measured.items()}
    rms_error = {
        name: float(values.struphy.analysis.error(profile_exact, dims=("t", "eta1"))) for name, values in density.items()
    }
    if not all(np.isfinite(rate) for rate in fitted_rate.values()):
        raise RuntimeError("A decay rate could not be fitted")
    for name in runs:
        print(f"{name}: decay rate {fitted_rate[name]:.3f} (exact {decay_rate:.3f}), density rms error {rms_error[name]:.4f}")

    colors = {"Random walk": "#d62828", "Deterministic": "#168aad"}
    dense_x = np.linspace(0.0, 1.0, 401)

    def traces(index):
        result = [
            go.Scatter(x=dense_x, y=1.0 + exact_amplitude[index] * np.cos(wavenumber * dense_x), mode="lines",
                       name="exact", line={"color": "#111", "width": 2, "dash": "dash"})
        ]
        for name, values in density.items():
            result.append(go.Scatter(x=x, y=values.values[index], mode="lines+markers", name=name,
                                     line={"color": colors[name], "width": 2}, marker={"size": 5}))
        return result

    picks = np.unique(np.linspace(0, len(times) - 1, min(80, len(times)), dtype=int))
    figure = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                           subplot_titles=("Density", "Amplitude of the cosine mode"))
    for trace in traces(0):
        figure.add_trace(trace, row=1, col=1)
    figure.add_scatter(x=times, y=exact_amplitude, mode="lines", name="exact decay", showlegend=False,
                       line={"color": "#111", "width": 2, "dash": "dash"}, row=2, col=1)
    for name, amp in measured.items():
        figure.add_scatter(x=times, y=np.abs(amp.values), mode="lines", name=name, showlegend=False,
                           line={"color": colors[name], "width": 2}, row=2, col=1)
    figure.frames = [go.Frame(name=f"{times[i]:.3f}", data=traces(i), traces=[0, 1, 2]) for i in picks]
    figure.update_layout(
        title="Diffusion of a density mode by two particle methods", template="plotly_white",
        margin={"l": 70, "r": 30, "t": 90, "b": 140}, legend={"orientation": "h", "y": 1.08},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.12,
                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 60, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}]}]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.07, "currentvalue": {"prefix": "t = "},
                  "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}], "label": frame.name, "method": "animate"} for frame in figure.frames]}],
    )
    figure.update_xaxes(title_text="x", range=[0, 1], row=1, col=1)
    figure.update_yaxes(title_text="density", range=[0.4, 1.6], row=1, col=1)
    figure.update_xaxes(title_text="t", row=2, col=1)
    figure.update_yaxes(title_text="amplitude", type="log", row=2, col=1)
    save(figure, "diffusion-methods", width=900, height=850, show=show)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the diffusion methods example.")
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
        create_simulation("Deterministic").run()
    pproc(simulation, show=args.show)
