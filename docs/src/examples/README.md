# Adding an example

Every file here is a complete, runnable Struphy simulation shown on `/examples/`. Two
scripts exist today (`maxwell-wave.py`, `poisson-source.py`) — use whichever is closer to
your model as a template. This walks through adding a new one, using `poisson-source.py` as
the worked reference.

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
- Save output as `<script-stem>.png` and `<script-stem>.html` via `write_image(...)` /
  `write_html(...)`, using relative paths (`Path("<script-stem>.png")`) — the script is run
  from inside `docs/public/examples/`, so these land there directly.
- If the run produces a result worth reporting (a measured value, an error norm, ...), merge
  it into the script's own metadata JSON at the end, e.g.:

  ```python
  metadata_path = Path("<script-stem>.metadata.json")
  metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
  metadata["someResult"] = value
  metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
  ```

  This is additive — `generate_examples.py` (next step) only ever adds/overwrites the
  *structural* fields (name, description, equations, config summary), never this one.

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
- [ ] `docs/public/images/examples/<script-stem>.png` — copy of the PNG
- [ ] `docs/src/pages/examples/<script-stem>.py.ts` — download route
- [ ] `docs/src/pages/examples/<script-stem>/index.astro` — detail page
- [ ] `npm run build` (or `dev`) — regenerates `examples-index.json` and confirms it builds
