# struphy-hub.github.io

The Struphy documentation site: an [Astro](https://astro.build) site in `docs/`, backed by data
generated from the [Struphy](https://github.com/struphy-hub/struphy) Python package (checked out
here as a git submodule).

## Building the site locally

### 1. Get the code, including the Struphy submodule

```sh
git clone --recurse-submodules git@github.com:struphy-hub/struphy-hub.github.io.git
cd struphy-hub.github.io
```

If you already cloned without `--recurse-submodules`:

```sh
git submodule update --init
```

### 2. Install Struphy from the submodule

Requires Python 3.12. From the repo root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install ./submodules/struphy
```

### 3. Generate the site's data

These scripts (repo root) import Struphy classes and export the JSON/VTK/image data the site
pages are built from:

```sh
python generate_all.py
```

which just runs, in order:

```sh
python generate_models.py
python generate_domains.py docs/public/domains
python generate_equilibrium_slices.py
python generate_perturbations.py
```

Each one prints what it generated (and skips) as it runs. Their output lives under `docs/src/data/`
and `docs/public/domains/`.

### 4. Build the Astro site

Requires Node.js >= 22.12. From `docs/`:

```sh
cd docs
npm ci
npm run build
```

The built site is written to `docs/dist/`. Use `npm run dev` instead for a local dev server
(both `dev` and `build` automatically regenerate `docs/src/data/catalogue-index.json` /
`catalogue-details.json` from the data produced in step 3 via their `pre*` npm hooks).

The full pipeline (including the CI-specific steps) is defined in
`.github/workflows/deploy.yml`, which is the source of truth if these instructions drift.

### Example gallery scripts (optional)

`docs/src/examples/*.py` are full, runnable Struphy simulations shown on the `/examples/` pages,
separate from the data pipeline above. They need `struphy compile` and a heavier install
(`pip install './submodules/struphy[mpi]'`, or whatever extras the example needs) and aren't
required to build the site. `generate_examples.py` (repo root) generates their page metadata
(name, description, equations, config summary) by importing each script without running its
simulation — see `docs/public/examples/README.md` for details.
