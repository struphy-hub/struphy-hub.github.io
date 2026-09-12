"""Verify shear-Alfvén wave dispersion with Struphy's ShearAlfven model.

A broadband transverse velocity perturbation excites shear-Alfvén waves
propagating along a uniform background magnetic field. Struphy evolves the
linearized MHD induction/momentum equations with FEEC, and the numerical
dispersion relation is compared against the exact Alfvén speed
v_A = B0 / sqrt(n0) = 1.

Requires Struphy 3.2 with compiled kernels (`struphy compile`).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.diagnostics.diagn_tools import power_spectrum_2d
from struphy.models import ShearAlfven

# Model and its single (linear) propagator.
model = ShearAlfven()
model.propagators.shear_alf.options = model.propagators.shear_alf.Options()

# A periodic one-dimensional domain embedded in 3D, aligned with the
# background field so waves can propagate along B0.
domain = domains.Cuboid(r3=20.0)
equil = equils.HomogenSlab(B0z=1.0, n0=1.0, beta=0.1)
grid = grids.TensorProductGrid(num_elements=(1, 1, 128))
derham_opts = DerhamOptions(degree=(1, 1, 3))
time_opts = Time(dt=0.05, Tend=50.0)

# Broadband noise in the two components transverse to B0 excites several
# shear-Alfvén modes at once, same trick as the Maxwell light-wave example.
model.mhd.velocity.add_background(FieldsBackground())
model.mhd.velocity.add_perturbation(
    perturbations.Noise(amp=0.05, comp=0, seed=123),
)
model.mhd.velocity.add_perturbation(
    perturbations.Noise(amp=0.05, comp=1, seed=123),
)

env = EnvironmentOptions(
    out_folders="struphy_gallery_runs",
    sim_folder="shear_alfven_wave",
)
sim = Simulation(
    model=model,
    name="Shear-Alfvén wave dispersion",
    description=(
        "Excite a broadband transverse velocity perturbation and recover the "
        "shear-Alfvén dispersion relation ω = v_A k with Struphy’s linearized "
        "MHD solver."
    ),
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)

if __name__ == "__main__":
    # Run, evaluate the FEEC fields on a grid, and load the result.
    # scope-profiler is built into Struphy: this instruments every propagator,
    # pusher and solver call during the run and writes a timing HDF5 file.
    sim.run(profiling_activated=True)
    sim.pproc(create_vtk=False)
    sim.load_plotting_data()

    # Struphy's diagnostic computes the (k, omega) spectrum and fits its branch.
    velocity = sim.spline_values.mhd.velocity_log.data
    omega, kvec, dispersion, coefficients = power_spectrum_2d(
        velocity,
        "velocity_log",
        grids=sim.grids_log,
        grids_mapped=sim.grids_phy,
        component=0,
        slice_at=[0, 0, None],
        do_plot=False,
        fit_branches=1,
        noise_level=0.5,
        extr_order=10,
        fit_degree=(1,),
    )

    phase_velocity = float(coefficients[0][0])
    print(f"Measured Alfvén speed: {phase_velocity:.5f} (exact: 1.0)")

    # Build an interactive Plotly view of the normalized power spectrum.
    omega = np.asarray(omega)
    kvec = np.asarray(kvec)
    power = np.asarray(dispersion) ** 2
    power /= power.max()
    log_power = np.log10(np.clip(power, 1e-15, None))
    fit = np.polyval(np.asarray(coefficients[0]), kvec)

    figure = go.Figure(
        go.Heatmap(
            x=kvec,
            y=omega,
            z=log_power,
            zmin=-15,
            zmax=-1,
            colorscale="Plasma",
            colorbar={
                "title": {"text": "log₁₀ P"},
                "tickvals": [-15, -12, -9, -6, -3],
                "ticktext": ["10⁻¹⁵", "10⁻¹²", "10⁻⁹", "10⁻⁶", "10⁻³"],
            },
            hovertemplate="k=%{x:.3f}<br>ω=%{y:.3f}<br>log₁₀ P=%{z:.2f}<extra></extra>",
        ),
    )
    figure.add_scatter(
        x=kvec,
        y=kvec,
        mode="lines",
        name="Alfvén wave, v_A = 1",
        line={"color": "#168aad", "width": 3, "dash": "dash"},
    )
    figure.add_scatter(
        x=kvec,
        y=fit,
        mode="lines",
        name=f"Struphy fit, v_A = {phase_velocity:.5f}",
        line={"color": "#d62828", "width": 3, "dash": "dot"},
    )
    figure.update_layout(
        title="Shear-Alfvén wave dispersion",
        xaxis_title="k [a.u.]",
        yaxis_title="ω [a.u.]",
        template="plotly_white",
        autosize=True,
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(44,62,80,0.25)",
            "borderwidth": 1,
        },
        margin={"l": 75, "r": 45, "t": 80, "b": 70},
    )
    figure.update_xaxes(range=[0, float(kvec[-1])])
    figure.update_yaxes(range=[0, float(kvec[-1])])

    png_path = Path("shear-alfven-wave.png")
    html_path = Path("shear-alfven-wave.html")
    figure.write_image(png_path, width=1100, height=650, scale=2)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {html_path.resolve()}")

    # Fold this run's measured result into the page metadata that
    # generate_examples.py already wrote for this script's static setup.
    # Export scope-profiler's plot-data JSON (durations, gantt, region
    # statistics) via its Python API, so the example page can render native
    # Plotly figures from the real run above -- not a separate report.
    from scope_profiler import plot_durations, plot_gantt, read_h5, write_region_statistics_json

    profile_reader = read_h5(sim.profiling_filepath)
    profile_h5_path = Path("shear-alfven-wave-profile.h5")
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    # `stack_children` splits each bar into the region's own time plus one
    # segment per region it calls, which is what the page's durations chart
    # stacks; it only decomposes total/avg, so those are the metrics exported.
    # `sort_by` fixes the region order the chart draws, biggest total first.
    durations_path = Path("shear-alfven-wave-durations.json")
    plot_durations(
        [profile_reader],
        ranks=[0],
        metrics=("total", "avg"),
        sort_by="total",
        stack_children=True,
        data_filepath=durations_path,
        data_format="json",
        verbose=False,
    )

    # The stacked export is a dense region x segment grid, but a call graph is
    # sparse -- roughly 90% of the pairs are zeros for regions that never call
    # each other. The chart reads a missing pair as null and skips it, so
    # dropping them cuts the file ~10x with no change on the page.
    durations_payload = json.loads(durations_path.read_text())
    durations_payload["bars"] = [bar for bar in durations_payload["bars"] if bar["value_seconds"]]
    durations_path.write_text(json.dumps(durations_payload))

    gantt_path = Path("shear-alfven-wave-gantt.json")
    region_stats_path = Path("shear-alfven-wave-region-stats.json")
    plot_gantt([profile_reader], ranks=[0], data_filepath=gantt_path, data_format="json", verbose=False)
    write_region_statistics_json([profile_reader], region_stats_path, ranks=[0])

    # A gantt bar is one call, not an aggregate, and a run with thousands of
    # steps draws tens of thousands of near-identical bars -- heavy enough to
    # hang the tab. Keeping the first GANTT_MAX_INTERVALS in time order leaves
    # the setup phase plus several complete iterations of the step loop.
    GANTT_MAX_INTERVALS = 5000
    gantt_payload = json.loads(gantt_path.read_text())
    if len(gantt_payload["intervals"]) > GANTT_MAX_INTERVALS:
        gantt_payload["intervals"] = sorted(gantt_payload["intervals"], key=lambda c: c["start_seconds"])[:GANTT_MAX_INTERVALS]
        gantt_path.write_text(json.dumps(gantt_payload))
    print(f"Saved {profile_h5_path.resolve()}")
    print(f"Saved {durations_path.resolve()}, {gantt_path.resolve()}, {region_stats_path.resolve()}")

    metadata_path = Path("shear-alfven-wave.metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["measuredAlfvenSpeed"] = phase_velocity
    metadata["exactAlfvenSpeed"] = 1.0
    metadata["profilingData"] = "/examples/shear-alfven-wave-profile.h5"
    metadata["profilingDurations"] = "/examples/shear-alfven-wave-durations.json"
    metadata["profilingGantt"] = "/examples/shear-alfven-wave-gantt.json"
    metadata["profilingRegionStats"] = "/examples/shear-alfven-wave-region-stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {metadata_path.resolve()}")
