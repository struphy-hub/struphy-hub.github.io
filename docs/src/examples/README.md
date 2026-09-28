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
- `toroidal-shear-alfven.py` — a small `LinearMHD` tokamak run with the m=10,11 perturbations, animated physical velocity components on a poloidal slice, ring histories, radial profiles, radius–time RMS maps, poloidal and temporal FFT spectra, frequency–radius maps, dominant-band reconstruction, and perturbation energies. Defaults to 8 × 48 × 4 cells, degree (3,3,2), dt=0.5 and t=20; edit the constants at the top for longer, finer runs. The short default record has only 21 samples and Δω≈0.299; this is an exploratory preview, not a converged TAE frequency measurement. Fourier diagnostics use plasma-plots' `out.analysis.fft`, `out.analysis.time_fft` and `out.analysis.filter_time`.
- `itpa-tae-linear-mhd.py`, `itpa-tae-shear-alfven.py` — the ITPA toroidal Alfvén eigenmode benchmark (m=10,11, n=−6 in a sixth of a tokamak), with `LinearMHD` and with the reduced `ShearAlfven` model: radial power against the shear-Alfvén continua, the power spectrum against ω_TAE, the eigenfunction of each harmonic, mode amplitudes, a matrix-pencil fit and the poloidal-plane animation, all drawn by plasma-plots. Defaults to 8 × 48 × 4 cells and t=150 (about two TAE periods); the benchmark resolution is noted at the top of each script.
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
python cli.py pproc orszag-tang-vortex # regenerate figures from existing simulation output
python cli.py show orszag             # paths of an earlier run (unique prefixes work; --open shows its page)
python cli.py clean --all             # remove generated figures, profiling data and scratch output
```

`run` does what steps 2 and 3 below do by hand, and what CI does per example. Use `python cli.py run --help` for
`--open`, `--quiet`, `--mpi`, `--keep-scratch` and `--keep-going`.
Use `python cli.py pproc --help` for the corresponding post-processing options. `pproc` preserves the existing
raw simulation output so it can be rerun without another simulation.

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
  entrypoint should accept only the `--pproc-only` flag: without it, run the hardcoded simulation
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
- Keep the command-line dispatch behind `if __name__ == "__main__":`; simulation setup and
  post-processing belong in the two functions above. The script must work on its own, without
  this repository: `python <script-stem>.py` simulates, post-processes and saves its figures in
  the current directory, and `--show` shows them first. See `maxwell-wave.py`.
- `pproc(sim, show=False)` contains only the physics. Draw the figures with plasma-plots and
  `backend="plotly"`, e.g. `e_x.plasma.plot.slice(x="z", y="t", symmetric=True, backend="plotly")`
  or `spectrum.plasma.plot.dispersion(kmin=0, branches=..., fits=..., backend="plotly")` (or with
  `plotly.graph_objects` for anything plasma-plots does not draw), and save each with
  `plasma_plots.save_figure(figure, name, show=show)`, which writes `name.html`, `.png` and
  `.plotly.json` on MPI rank 0 (`frame=` or `still=` choose the image of an animation). Name the
  page's main figure `<script-stem>` and the others `<script-stem>-<key>`.
- The website's part is `run_example.py`, which `python cli.py run` and CI call: it runs the script
  in `docs/public/examples/`, profiles its simulation, exports the profiling data into
  `<script-stem>.metadata.json` and copies the PNGs to `docs/public/images/examples/`.
- Page texts live in `docs/src/data/example-config.ts`: the main figure's title and alt text, and
  under `figures` each other figure's key, in page order, with its alt text and caption.
- Analyze with the `Output` of the run (`sim.output`), e.g.
  `sim.output.scalars["electric_energy"].plasma.analysis.damping_rate(window=(None, 8.0), amplitude=True)`;
  see `weak-landau-damping.py`. Print the measured results; the page shows the figures.

## 2. Generate the structural metadata

From the repo root, with Struphy installed (`pip install ./submodules/plasma-plots/struphy` — no
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

GitHub-hosted runners execute each example with `run_example.py` in a single process
(see `run-example` in `.github/workflows/build-site.yml`).
For optional local MPI runs: `python cli.py run <example> --mpi 4`, or, from the repository root,
`mpirun -n 4 python run_example.py <script-stem>`.

- The simulation, `output.pproc(...)` and the analysis run on every rank; `save_figure`
  writes on rank 0 only (plasma-plots draws nothing on the other ranks), as does `run_example.py`.
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
then bump the `struphy` pointer in plasma-plots and the website's `submodules/plasma-plots` pointer before publishing. A website-only commit cannot
reproduce this example in CI.

The Orszag–Tang script checks for non-finite diagnostics, non-positive density and incomplete
evolution before publishing figures. The plotted current uses differences of sampled physical
fields; `sqrt(tot_div_B)` is the independent FEEC divergence norm. These are distinct diagnostics.
