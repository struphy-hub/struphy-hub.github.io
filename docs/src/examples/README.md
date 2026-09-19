# Adding an example

Every file here is a complete, runnable Struphy simulation shown on `/examples/`. Several
scripts exist today — use whichever is closest to your model and diagnostic as a template:

- `poisson-source.py` / `weak-landau-damping.py` — a field-vs-exact-solution comparison (1D line plot / log-scale energy plot)
- `vlasov-tokamak.py` — kinetic/PIC 3D particle trajectories (`Scatter3d`)
- `diocotron-instability.py` — a 2D binned density animated over time (`Heatmap` + frames)
- `dam-break.py` — SPH markers (no grid): a two-panel animation of the markers and their kernel density estimate, built with `make_subplots` and a frame per saved step
- `gas-expansion.py` — isothermal SPH rarefaction, with an analytic reference, similarity profiles and error diagnostics
- `coaxial-waveguide.py` — rotating Maxwell eigenmode, physical-plane field/error animation, probe frequency and energy
- `guiding-center-orbits.py` — passing and trapped orbits, parallel velocity and per-particle invariants
- `vortex-merger.py` — physical-plane density animation of two charge blobs and electrostatic-energy drift
- `orszag-tang-vortex.py` — early nonlinear MHD to t = 0.5, density with magnetic field lines, energy/divergence diagnostics and a pressure cut
- `strong-landau-damping.py`, `two-stream-instability.py`, `bump-on-tail.py`, `weibel-instability.py` — other Vlasov-Ampère/Maxwell instability benchmarks, all sharing the same field-energy-vs-time diagnostic pattern

This walks through adding a new one, using `poisson-source.py` as the worked reference.

The one rule that ties everything together: **an example's page lives at
`docs/src/pages/examples/<script-stem>/`, matching its `<script-stem>.metadata.json`.**
Nothing else needs to know the mapping — `generate-examples-index.mjs` derives it from that
filename.

## 1. Write the script: `docs/src/examples/<script-stem>.py`

- Build the model, domain, grid, and `Simulation(...)` at **module scope** — not inside
  `if __name__ == "__main__":`. `generate_examples.py` imports the script without running it
  to read this setup, so it must be constructible without a compiled Struphy install.
- Pass `name=` and `description=` to `Simulation(...)`. These become the page's title and
  intro text — don't duplicate them anywhere else.
- Put `sim.run()`, post-processing, plotting, and file output behind
  `if __name__ == "__main__":`. That's the part that needs `struphy compile` and actually
  takes time to run.
- Plot with Plotly (`plotly.graph_objects`), not matplotlib — the site embeds the
  `write_html(...)` output directly for a live, interactive figure. See either existing
  script for layout conventions (margins, slider/button placement if animated).
- Save output with the helpers in `_gallery.py` (import them inside the `__main__` block, since
  `generate_examples.py` runs the module without the examples directory on `sys.path`):
  `save_figure(figure, "<script-stem>")` writes `<script-stem>.png` and `.html`;
  `export_profiling(sim, "<script-stem>")` writes the profiling files and returns their metadata
  fields; `merge_metadata("<script-stem>", **fields)` adds result fields to the metadata JSON.
  The script is run from inside `docs/public/examples/`, so these land there directly.
- Additional figures go below the main one. `heatmap_figure`, `space_time_figure` (a field over
  space and time) and `heatmap_movie` (an animation over one dimension) build them from xarray
  arrays such as `output.evaluate("kinetic_ions/e1_v1_density/f")`, and
  `save_extra_figure(figure, "<script-stem>", "<key>", alt=..., caption=...)` saves one (plus its
  committed thumbnail in `docs/public/images/examples/`) and returns its entry for
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
it doesn't know about (like the result field from step 1, and `thumbnail`/`interactive`
from step 3).

## 3. Run it for real, and wire up the figure

```sh
cd docs/public/examples
struphy compile   # once, if you haven't already
python ../../src/examples/<script-stem>.py
```

This produces `<script-stem>.png` and `<script-stem>.html` in `docs/public/examples/`, and
folds any result field into the metadata JSON (step 1). Then:

- Copy the PNG to `docs/public/images/examples/<script-stem>.png` (the `<noscript>`
  fallback and the gallery thumbnail use this path, separate from the interactive HTML).
- Add two fields to `docs/public/examples/<script-stem>.metadata.json` by hand — these are
  run outputs `generate_examples.py` can't know about:

  ```json
  "thumbnail": "/images/examples/<script-stem>.png",
  "interactive": "/examples/<script-stem>.html"
  ```

Clean up `docs/public/examples/struphy_gallery_runs/` (or wherever `EnvironmentOptions`
pointed) and any `struphy.log` left behind in that directory — they're simulation
scratch output, not part of the site.

Only `<script-stem>.metadata.json` and `docs/public/images/examples/<script-stem>.png` are
committed to git — `.gitignore` excludes everything else this step produces
(`<script-stem>.html`, the `docs/public/examples/<script-stem>.png` copy, and the profiling
JSON/HDF5). `.github/workflows/deploy.yml` reruns every script here from scratch before each
site build, so those files never need to be pushed by hand; running the script locally is
only for wiring up the figure once and for local `npm run dev` previews.

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

## 5. Add the detail page: `docs/src/pages/examples/<script-stem>/index.astro`

Copy `docs/src/pages/examples/poisson-source/index.astro` (or `maxwell-wave/index.astro`)
to `docs/src/pages/examples/<script-stem>/index.astro` and adapt:

- The three imports at the top (`<script-stem>.py?raw`, the metadata JSON, and anything
  else) to point at your new files.
- The `<iframe src="/examples/<script-stem>.html">` and `<noscript><img
  src="/images/examples/<script-stem>.png">` in the `.result` figure.
- The hand-written "Physical problem" paragraph and the `<figcaption>` (the only prose that
  isn't pulled from metadata — everything else in that section, including the equations and
  the `<dl>` config summary, renders from `data.*` automatically).
- The `<dl>` rows: only include `{data.integrator && ...}` if your model's propagators
  actually expose one (check the generated metadata JSON — `generate_examples.py` omits
  fields it can't derive).
- The provenance link at the bottom, if adapted from a specific Struphy test.

The `<p class="eyebrow">` model link, the `<dl>` "Model" row link (both
`/models/${toSlug(data.model)}/`), the `<dl>` "Domain" row link
(`/domains/#${data.domain.split(' ')[0]}` — the domain viewer deep-links by class name via a
URL hash), and the shiki/KaTeX rendering can be copied as-is — don't hardcode the model or
domain name as plain text anywhere; always link them via `data.model` / `data.domain`.

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

- [ ] `docs/src/examples/<script-stem>.py` — `Simulation(name=..., description=...)` at
      module scope, heavy work behind `if __name__ == "__main__":`, Plotly output
- [ ] `docs/public/examples/<script-stem>.metadata.json` — from `generate_examples.py`, plus
      hand-added `thumbnail` / `interactive` (and any result field from the script itself)
- [ ] `docs/public/examples/<script-stem>.png` / `.html` — from actually running the script
      once locally (gitignored; CI regenerates these on every deploy)
- [ ] `docs/public/images/examples/<script-stem>.png` — copy of the PNG (the one example
      artifact that *is* committed, as a fallback for local dev without a compiled Struphy)
- [ ] `docs/src/pages/examples/<script-stem>.py.ts` — download route
- [ ] `docs/src/pages/examples/<script-stem>/index.astro` — detail page
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
