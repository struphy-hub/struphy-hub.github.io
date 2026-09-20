"""Convergence of Struphy's FEEC Poisson solver: the error against resolution, for three spline degrees.

The Poisson equation -div grad phi = rho is solved on a periodic unit square for the manufactured source
rho = A cos(k_x x + k_y y), whose exact solution is phi = A / (k_x^2 + k_y^2) cos(k_x x + k_y y). Running
the same problem on a sequence of grids and for spline degrees 1, 2 and 3 gives the convergence rates of
the discretization: the L2 error of a degree-p spline approximation falls as h^(p+1), and the fitted
slopes are compared with that expectation.

Every run is a single linear solve, so the whole study is cheap. All grids are compared on the same
evaluation points, by dividing each cell into as many points as the coarsest grid needs.

Requires Struphy 3.3 with compiled kernels (`struphy compile`).
"""

import numpy as np
import plotly.graph_objects as go

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, grids, perturbations
from struphy.models import Poisson

amplitude = 0.1
mode_x, mode_y = 1, 1  # wavelengths of the source per box length
degrees = (1, 2, 3)
resolutions = (8, 16, 32, 64)
evaluation_points = 64  # per direction, the same for every run

k_x = 2.0 * np.pi * mode_x
k_y = 2.0 * np.pi * mode_y
potential_amplitude = amplitude / (k_x**2 + k_y**2)

domain = domains.Cuboid(r1=1.0, r2=1.0)
time_opts = Time(dt=1.0, Tend=1.0)  # the solve is static; one step is one solve


def build_simulation(degree, num_elements, **extra):
    """The same Poisson problem at one resolution and spline degree."""
    model = Poisson()
    model.em_fields.source.add_perturbation(
        perturbations.ModesCos(ls=(mode_x,), ms=(mode_y,), amps=(amplitude,)),
    )
    return Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders="struphy_gallery_runs",
            sim_folder=f"poisson_convergence_p{degree}_n{num_elements}",
        ),
        time_opts=time_opts,
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=(num_elements, num_elements, 1)),
        derham_opts=DerhamOptions(degree=(degree, degree, 1)),
        **extra,
    )


sim = build_simulation(
    degrees[-1],
    resolutions[1],
    name="Poisson convergence study",
    description=(
        "A manufactured cosine source on a periodic square, solved on four grids for spline degrees 1, 2 "
        "and 3. The L2 error falls as h^(p+1), the rate expected of a degree-p spline space, and the "
        "fitted slopes are compared with it."
    ),
)


if __name__ == "__main__":
    from _gallery import export_profiling, merge_metadata, save_extra_figure, save_figure

    def exact_potential(x, y):
        return potential_amplitude * np.cos(k_x * x + k_y * y)

    def l2_error(degree, num_elements, run=None):
        """Run one case and return its L2 error against the exact potential, on the common grid."""
        simulation = run or build_simulation(degree, num_elements)
        output = simulation.run()
        # Each cell is divided so that every resolution lands on the same evaluation points.
        output.pproc(celldivide=(evaluation_points // num_elements, evaluation_points // num_elements, 1))
        phi = output.evaluate("em_fields/phi").isel(t=-1, e3=0)
        x = phi.e1.values
        y = phi.e2.values
        error = phi.values - exact_potential(x[:, None], y[None, :])
        return float(np.sqrt(np.mean(error**2))), simulation

    errors = {}
    profiling = None
    for degree in degrees:
        errors[degree] = []
        for num_elements in resolutions:
            # The reference simulation of the page is one of the cases; it carries the profiling data.
            reference = sim if (degree, num_elements) == (degrees[-1], resolutions[1]) else None
            if reference is not None:
                reference.profiling_activated = True
            error, simulation = l2_error(degree, num_elements, run=reference)
            errors[degree].append(error)
            print(f"degree {degree}, {num_elements:3d} elements: L2 error {error:.3e}")

    cell_sizes = 1.0 / np.array(resolutions, dtype=float)
    slopes = {}
    for degree in degrees:
        values = np.array(errors[degree])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise RuntimeError(f"Degree {degree} produced a non-positive or non-finite error")
        slopes[degree] = float(np.polyfit(np.log(cell_sizes), np.log(values), 1)[0])
        print(f"degree {degree}: fitted slope {slopes[degree]:.2f} (expected {degree + 1})")

    colors = {1: "#168aad", 2: "#f4a261", 3: "#d62828"}
    figure = go.Figure()
    for degree in degrees:
        figure.add_scatter(
            x=resolutions, y=errors[degree], mode="lines+markers",
            name=f"degree {degree}: slope {slopes[degree]:.2f} (expected {degree + 1})",
            line={"color": colors[degree], "width": 2.5}, marker={"size": 9},
        )
        # The expected rate, anchored at the coarsest grid of this degree.
        expected = errors[degree][0] * (cell_sizes / cell_sizes[0]) ** (degree + 1)
        figure.add_scatter(
            x=resolutions, y=expected, mode="lines", showlegend=False,
            line={"color": colors[degree], "width": 1, "dash": "dot"},
            hovertemplate=f"h^{degree + 1}<extra></extra>",
        )
    figure.update_layout(
        title="Poisson solver: L2 error against grid resolution",
        xaxis_title="elements per direction", yaxis_title="L2 error of φ",
        template="plotly_white", autosize=True,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 80, "r": 30, "t": 80, "b": 100},
    )
    figure.update_xaxes(type="log", tickvals=resolutions, ticktext=[str(n) for n in resolutions])
    figure.update_yaxes(type="log", exponentformat="power")
    save_figure(figure, "poisson-convergence")

    # The solution itself, from the finest degree-3 run, next to its error.
    phi = sim.output.evaluate("em_fields/phi").isel(t=-1, e3=0)
    x, y = phi.e1.values, phi.e2.values
    error_map = phi.values - exact_potential(x[:, None], y[None, :])
    solution_figure = go.Figure(
        go.Heatmap(x=x, y=y, z=phi.values.T, colorscale="RdBu", zmid=0.0,
                   colorbar={"title": {"text": "φ"}})
    )
    solution_figure.update_layout(
        title=f"Potential of the degree-{degrees[-1]}, {resolutions[1]}-element solve",
        xaxis_title="x", yaxis_title="y", template="plotly_white", autosize=True,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    solution_figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figures = [
        save_extra_figure(
            solution_figure, "poisson-convergence", "solution",
            alt="The FEEC potential of the manufactured cosine source on the periodic square",
            caption=(
                "The potential of the manufactured source, from the degree-"
                f"{degrees[-1]} solve on {resolutions[1]}×{resolutions[1]} elements. Its largest deviation "
                f"from the exact solution is {np.max(np.abs(error_map)):.1e}, against a potential amplitude "
                f"of {potential_amplitude:.3f}."
            ),
        ),
    ]

    merge_metadata(
        "poisson-convergence",
        **{f"slopeDegree{degree}": slopes[degree] for degree in degrees},
        **{f"finestErrorDegree{degree}": errors[degree][-1] for degree in degrees},
        figures=figures,
        **export_profiling(sim, "poisson-convergence"),
    )
