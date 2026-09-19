Everything in this directory except this file is generated and gitignored: the interactive figures and
`*.metadata.json` sidecars, made by the scripts in `src/examples/` and `generate_examples.py`
(which imports each script without running its simulation).

Generate them from the repository root with `python cli.py metadata --all` (metadata only) or
`python cli.py run <example>` (metadata and figures).

See `docs/src/examples/README.md` for the full step-by-step guide to adding a new example.
