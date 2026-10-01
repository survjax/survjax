# Chronox documentation

Chronox is a JAX library for survival analysis. The current implementation
provides `SurvivalData` construction, normalization, dtype handling and data
diagnostics. Indexing, likelihoods, distributions and evaluation are planned;
the Slice A completion gate has not yet been met.

The [SurvivalData guide](survival-data.md) walks a synthetic cohort through
both constructors, diagnostics and R input conventions. The [playground](playground.ipynb) uses
only synthetic observations and the current public API. Runnable examples
execute afresh when the site is built.

```{toctree}
:maxdepth: 1

playground
survival-data
```

## Local commands

From the repository root, run `scripts/build_docs.sh` to build the site and
execute its notebook separately with JAX float32 and float64. The generated
pages are in `docs/_build/html-x32/` and `docs/_build/html-x64/`. The build
fails on warnings and notebook errors. The examples need no network access,
R installation or external data; dependency installation uses the lockfile.

To view the rendered site on macOS, run `open
docs/_build/html-x64/index.html` (or use `html-x32` for float32). On other
systems, open that `index.html` file in a browser.

For an editable notebook, run `uv run --locked --group docs jupyter lab
docs/playground.ipynb`. JupyterLab uses the same locked environment as the
build. Notebook outputs and generated pages are local artifacts.
