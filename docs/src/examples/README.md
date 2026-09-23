# Adding an example

Every file here is a complete, runnable Struphy simulation shown on `/examples/`. Several
scripts exist today — use whichever is closest to your model and diagnostic as a template:

- `poisson-source.py` / `weak-landau-damping.py` — a field-vs-exact-solution comparison (1D line plot / log-scale energy plot)
- `vlasov-tokamak.py` — kinetic/PIC 3D particle trajectories (`Scatter3d`)
- `diocotron-instability.py` — a 2D binned density animated over time (`Heatmap` + frames)
- `hasegawa-wakatani.py` — 2D drift-wave turbulence, with animated vorticity/density heatmaps and a zonal-flow energy split
- `dam-break.py` — SPH markers (no grid): a two-panel animation of the markers and their kernel density estimate, built with `make_subplots` and a frame per saved step
- `beltrami-sph.py` — pressureless SPH markers circulating in a prescribed stationary Beltrami flow, checked against the exact velocity field and particle Hamiltonian
- `gas-expansion.py` — isothermal SPH rarefaction, with an analytic reference, similarity profiles and error diagnostics
- `coaxial-waveguide.py` — rotating Maxwell eigenmode, physical-plane field/error animation, probe frequency and energy
- `guiding-center-orbits.py` — passing and trapped orbits, parallel velocity and per-particle invariants
- `vortex-merger.py` — physical-plane density animation of two charge blobs and electrostatic-energy drift
- `orszag-tang-vortex.py` — nonlinear MHD to t = 1, density with magnetic field lines, energy/divergence diagnostics and a pressure cut
- `resistive-x-point.py` — nonlinear visco-resistive MHD at a driven magnetic null, with current-density/flux animation, reconnection rate and conservation diagnostics
- `mhd-slab-waves.py` — the shear Alfvén and the slow and fast magnetosonic waves of `LinearMHD`, from the (k, ω) spectra of the velocity and the pressure, with fitted against exact speeds
- `toroidal-shear-alfven.py` — a small `LinearMHD` tokamak run with the m=10,11 perturbations, animated physical velocity components on a poloidal slice, ring histories, radial profiles, radius–time RMS maps, poloidal and temporal FFT spectra, frequency–radius maps, dominant-band reconstruction, and perturbation energies. Defaults to 8 × 48 × 4 cells, degree (3,3,2), dt=0.5 and t=20; edit the constants at the top for longer, finer runs. The short default record has only 21 samples and Δω≈0.299; this is an exploratory preview, not a converged TAE frequency measurement. Fourier diagnostics use the submodule's current `Output.fft`, `Output.time_fft` and `Output.filter_time` API.
- `zeldovich-caustic.py` — pressureless SPH collapse to a caustic and multi-stream flow, with the exact density from the Lagrangian map (animated density and phase space)
- `diffusion-methods.py` — the random-walk and the deterministic particle methods for the diffusion equation, compared with the exact decay (two simulations in one script)
- `incompressible-shear-relaxation.py` — incompressible SPH between no-slip walls: the pressure projection removes a compressive wave, and the shear mode decays at the exact viscous rate
- `gvec-equilibrium.py` — runs GVEC to create a converged five-field-period stellarator, then follows Struphy guiding-center orbits in its curved geometry and magnetic field (requires the `phys` extra)
- `strong-landau-damping.py`, `two-stream-instability.py`, `bump-on-tail.py`, `weibel-instability.py` — other Vlasov-Ampère/Maxwell instability benchmarks, all sharing the same field-energy-vs-time diagnostic pattern
- `cold-plasma-oscillation.py` — a cold electron fluid rings at the plasma frequency: field and flow energy trade places as cos² and sin², and a scan over the density confirms omega = sqrt(n0)
- `cold-plasma-wave-packet.py` — a Gaussian packet of circularly polarized field splits into a fast L-wave packet and a slow R-wave packet, whose speeds are compared with the analytic group velocities
- `maxwell-cavity-resonances.py` — noise in E_z in a rectangular periodic box, whose power spectrum has one peak at each exact resonance of the box
- `maxwell-curved-mesh.py` — a pulse of E_z on the distorted Colella mesh, against its exact Fourier-series solution, with the energy conservation
- `poisson-convergence.py` — error of the Poisson potential against resolution at spline degrees 1 to 3, on a straight and a distorted mesh, with the measured slopes
- `gyromotion.py` — four test particles in a uniform field on helices of different Larmor radii, against the exact orbits (Strang splitting; the default Lie-Trotter splitting is only first order in the position)
- `grad-b-drift.py` — three full-orbit test ions in a straight, periodic magnetic-field gradient, with the transverse drift compared against the guiding-center prediction and its perpendicular-energy scaling (run on one rank)
- `ordinary-mode-dispersion.py` — four ordinary electromagnetic modes in `ColdPlasma`, with independently measured frequencies, exact oscillations, and the plasma-frequency cutoff of the dispersion curve
- `faraday-rotation.py` — two circular cold-plasma eigenmodes at the same frequency form a wave train whose linear polarization rotates in space, with an animated field, measured polarization angle, and local polarization traces
- `langmuir-wave-dispersion.py` — oscillation frequency and Landau damping of Langmuir waves at four wavenumbers, against the root of the kinetic dispersion relation and the fluid Bohm-Gross estimate
- `resistive-diffusion.py` — resistive decay of a sinusoidal magnetic field with its Ohmic heating, for three resistivities (`ViscoResistiveMHD`)
- `damped-alfven-wave.py` — a standing Alfvén wave in resistive MHD, at the Alfvén frequency and damped at the rate eta k^2 / 2, for three resistivities
- `acoustic-pulse.py` — a Gaussian density pulse in `VariationalCompressibleFluid` splitting into two pulses at the speed of sound, against d'Alembert's solution and with the energy exchange
- `linear-dissipative-alfven-wave.py` — a standing wave in `ViscoResistiveLinearMHD`, with equal viscosity and resistivity, compared with the exact damped field profiles and wave-energy decay
- `pressureless-transport.py` — a density ripple carried around a periodic box by `VariationalPressurelessFluid`, compared with exact nonlinear transport and checked for profile, velocity, sampled-mass and kinetic-energy errors
- `hybrid-current-coupling.py` — an Alfvén wave coupled to full-orbit energetic ions with `LinearMHDVlasovCC`, showing energy exchange, conservation error on the wave-energy scale, and a velocity space-time map

To build the assets of existing examples, use the CLI at the repository root (standard library only):

```sh
python cli.py list                    # examples, and which have generated figures
python cli.py run orszag-tang-vortex  # metadata + run + clean-up, then prints every generated file
python cli.py show orszag             # paths of an earlier run (unique prefixes work; --open shows its page)
python cli.py clean --all             # remove generated figures, profiling data and scratch output
```

`run` does what steps 2 and 3 below do by hand, and what CI does per example. Use `python cli.py run --help` for
`--open`, `--quiet`, `--mpi`, `--keep-scratch` and `--keep-going`.

This walks through adding a new one, using `poisson-source.py` as the worked reference.

The one rule that ties everything together: **an example's page lives at
`docs/src/pages/examples/<script-stem>/`, matching its `<script-stem>.metadata.json`.**
Nothing else needs to know the mapping — `generate-examples-index.mjs` derives it from that
filename.

## 1. Write the script: `docs/src/examples/<script-stem>.py`

- Keep hardcoded problem parameters and reusable setup helpers at module scope, but construct
  the `Simulation(...)` inside `create_simulation()`. `generate_examples.py` imports the script
  and calls that factory without running the simulation, so construction must be possible
  without a compiled Struphy install.
- Expose `create_simulation() -> Simulation` and `pproc(sim: Simulation)` functions. The
  entrypoint should accept only the `--pproc` flag: without it, run the hardcoded simulation
  and then call `pproc`; with it, construct the same simulation and post-process its existing
  output without running time integration. Do not add command-line options for model or run
  parameters.
- Pass `name=` and `description=` to `Simulation(...)`. These become the page's title and
  intro text — don't duplicate them anywhere else.
- Descriptions support inline LaTeX with `` :math:`\gamma \approx -0.1533` `` and
  display equations with `$$...$$` or an indented `.. math::` block. Use Python raw
  strings (`r"..."`) to preserve LaTeX backslashes; see `weak-landau-damping.py`.
  Equations render on the example page and inline in gallery/model cards. Ordinary
  text is escaped, so descriptions do not accept raw HTML.
- Keep the command-line dispatch behind `if __name__ == "__main__":`; simulation execution,
  post-processing, plotting, and file output belong in the two functions above. The normal
  branch is the part that needs `struphy compile` and actually takes time to run.
- Plot with Plotly (`plotly.graph_objects`), not matplotlib. The shared gallery helper
  serializes the completed figure as Plotly JSON; the site renders it with its single shared
  Plotly runtime. See an existing script for layout conventions (margins, slider/button
  placement if animated).
- Save output with the helpers in `_gallery.py` (import them inside the `__main__` block, since
  `generate_examples.py` runs the module without the examples directory on `sys.path`):
  `save_figure(figure, "<script-stem>")` writes `<script-stem>.png`, `.plotly.json`, and `.html`;
  `export_profiling(sim, "<script-stem>")` writes the profiling files and returns their metadata
  fields; `merge_metadata("<script-stem>", **fields)` adds result fields to the metadata JSON.
  The script is run from inside `docs/public/examples/`, so these land there directly.
- Additional figures go below the main one. `heatmap_figure`, `space_time_figure` (a field over
  space and time) and `heatmap_movie` (an animation over one dimension) build them from xarray
  arrays such as `output.evaluate("kinetic_ions/e1_v1_density/f")`, and
  `save_extra_figure(figure, "<script-stem>", "<key>", alt=..., caption=...)` saves one (its PNG is also copied
  to `docs/public/images/examples/`, like the main figure's) and returns its entry for
  `merge_metadata("<script-stem>", figures=[...])`. The page shows every entry of `figures`
  through `ExtraFigures.astro`, so no page edit is needed beyond adding that component once.
  Reductions such as `f.struphy.analysis.spatial_average()` and `.velocity_moments()` turn a
  binned distribution into f(v, t) or the velocity variance.
- Analyze with the `Output` returned by the run (`sim.output`), e.g.
  `sim.output.evaluate("electric_energy").struphy.analysis.damping_rate(window=(None, 8.0), amplitude=True)`;
  see `weak-landau-damping.py`.
- If the run produces a result worth reporting (a measured value, an error norm, ...), pass it
  to `merge_metadata`. This is additive — `generate_examples.py` (next step) only ever
  adds/overwrites the *structural* fields (name, description, equations, config summary), never
  this one.

## 2. Generate the structural metadata

From the repo root, with Struphy installed (`pip install ./submodules/struphy` — no
compiled kernels needed for this step):

```sh
python generate_examples.py
```

This writes `docs/public/examples/<script-stem>.metadata.json` for every script in this
directory, with `name`, `description`, `model`, `equationsMarkdown`, `domain`, `grid`,
`degree`, `steps`, and (when applicable) `integrator` and `visualization` — all read
straight off your `sim`/`model` objects. It's safe to re-run any time; it preserves fields
it doesn't know about (like the result fields from step 1).

## 3. Run it for real

```sh
cd docs/public/examples
struphy compile   # once, if you haven't already
python ../../src/examples/<script-stem>.py
```

This produces `<script-stem>.png`, `<script-stem>.plotly.json`, and `<script-stem>.html` in `docs/public/examples/`, copies each PNG
to `docs/public/images/examples/` (the gallery thumbnail reads it there, and so does the example page
on phones and without JavaScript, where it shows the PNG instead of the interactive plot)
and folds any result field into the metadata JSON (step 1). Everything is named after the script:
the example page refers to `/examples/<script-stem>.plotly.json` and `/images/examples/<script-stem>.png`,
and the gallery finds the thumbnail by that name. Nothing has to be wired up by hand.

Clean up `docs/public/examples/struphy_gallery_runs/` (or wherever `EnvironmentOptions`
pointed) and any `struphy.log` left behind in that directory — they're simulation
scratch output, not part of the site.

None of these files is committed: `.gitignore` excludes the metadata JSON, the figures, the thumbnails,
the profiling exports and `docs/src/data/examples-index.json`. `.github/workflows/build-site.yml` reruns
every script from scratch before each site build. A fresh clone therefore has none of them, and
`npm run dev` stops with instructions to generate them: `python cli.py metadata --all` is enough for
the pages to build (Struphy needed, compiled kernels not), and `python cli.py run <example>` also
makes that example's figures.

### Running on several MPI ranks

GitHub-hosted runners execute each example directly with Python in a single process
(see `run-example` in `.github/workflows/build-site.yml`).
For optional local MPI runs: `python cli.py run <example> --mpi 4`, or
`mpirun -n 4 python ../../src/examples/<script-stem>.py`.

- The simulation, `output.pproc(...)` and the analysis run on every rank; the `_gallery.py` helpers
  (`save_figure`, `save_extra_figure`, `merge_metadata`, `export_profiling`) write on rank 0 only. Write
  files only through them, or guard the code with `_gallery.is_root()`.
- The grid must split over the ranks: with four ranks, keep at least a few cells per rank and direction
  (Orszag–Tang uses 32 × 32 × 1, i.e. 16 × 16 cells per rank).
- Particle results depend on the rank count (each rank draws its own markers), so measured rates move a little.
- Saved orbits of tracked markers (`SavingParameters(n_markers=...)`) are empty after the first time step on several ranks, so run `guiding-center-orbits` on one rank.
- The SPH examples (`beltrami-sph`, `dam-break`, `gas-expansion`, `zeldovich-caustic`, `incompressible-shear-relaxation`) hang under MPI in Struphy, so run them on one rank.

## 4. Add the download route: `docs/src/pages/examples/<script-stem>.py.ts`

```ts
import scriptSource from '../../examples/<script-stem>.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="<script-stem>.py"',
    },
  });
}
```

## 5. Register the page presentation

All example detail pages use the shared dynamic route
`docs/src/pages/examples/[slug]/index.astro` and renderer
`docs/src/components/ExamplePage.astro`. Add the example's short presentation details
(category, setup heading, plot title and alt text) to
`docs/src/data/example-config.ts`. The metadata, source code, equations, figures and
profiling data are discovered automatically from the script stem; no copied page or
repeated CSS is needed.

## 6. Rebuild

```sh
cd docs
npm run build   # or `npm run dev`
```

The `pre*` npm hooks run `generate-examples-index.mjs`, which scans every
`docs/public/examples/*.metadata.json` and writes `docs/src/data/examples-index.json` — this
is what makes the new example show up on `/examples/` **and** get listed automatically under
its model's "Runnable examples" section on `/models/<slug>/`, purely from the `model` field
in its metadata. No list to edit by hand.

## Checklist

- [ ] `docs/src/examples/<script-stem>.py` — construct `Simulation(name=..., description=...)`
      and all Struphy setup objects inside `create_simulation()`, with heavy work behind
      `if __name__ == "__main__":`, Plotly output
- [ ] Name every file the script writes after the script: `<script-stem>.png`, `<script-stem>.plotly.json`, `<script-stem>.html`
      (and `<script-stem>-<key>.*` for extra figures). The metadata JSON, figures and thumbnails are
      generated and gitignored; check them locally with `python cli.py run <script-stem>`
- [ ] `docs/src/pages/examples/<script-stem>.py.ts` — download route
- [ ] Add the `<script-stem>` presentation entry to `docs/src/data/example-config.ts`
- [ ] `npm run build` (or `dev`) — regenerates `examples-index.json` and confirms it builds

## Reproducing the completed examples

Use the Struphy revision pinned by this repository, including local submodule changes while developing.
The full Orszag–Tang run needs the fix in `struphy/feec/mass.py` that preserves geometric weights
between density-weighted matrix assemblies. Commit that fix and its regression test in Struphy,
then update the website's submodule pointer before publishing. A website-only commit cannot
reproduce this example in CI.

The Orszag–Tang script checks for non-finite diagnostics, non-positive density and incomplete
evolution before publishing figures. The plotted current uses differences of sampled physical
fields; `sqrt(tot_div_B)` is the independent FEEC divergence norm. These are distinct diagnostics.
