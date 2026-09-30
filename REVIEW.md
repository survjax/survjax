# Review: DOC-1 local documentation process

## Read first

1. `scripts/build_docs.sh` and `docs/conf.py`: two fresh, warning-strict builds.
2. `docs/index.md` and `docs/playground.ipynb`: current scope, commands and the synthetic example.
3. `pyproject.toml`, `uv.lock` and `.gitignore`: optional locked tools and excluded outputs.
4. `PLAN.md`, DOC-1: approved scope. DOC-2 remains a separate step.

## Invariants and scope

- The notebook uses only the public `SurvivalData` constructor and `data.issues`.
  It asserts exact flags, stored lower times, validity and the JAX dtype mode.
- Every build starts a separate process per dtype and forces notebook execution.
  Sphinx treats warnings and execution errors as build failures.
- The committed notebook has no outputs. Generated HTML and execution artifacts
  are ignored. The optional docs group does not change core dependencies.
- `docs/simulation-guide.md` is excluded from the site: it is explicitly
  unverified, covers unimplemented likelihood workflows and currently has an
  unresolved cross-source specification link under warning-strict Sphinx.
- The lockfile grows by about 1,900 lines because of Sphinx and JupyterLab's
  transitive dependencies. Splitting it from the build would leave DOC-1
  unusable; the hand-written change remains small.

## Evidence

- `UV_CACHE_DIR=/private/tmp/chronox-docs-uv-cache scripts/build_docs.sh`:
  exit 0. Both float32 and float64 HTML builds succeeded; the notebook executed
  in each. Sphinx reported no documentation warnings. Jupyter printed its
  standard local TCP transport warning, which is not a Sphinx build warning.
- Rendered HTML inspection: both index pages link to the playground; the
  playground shows `JAX float64 enabled: False` in x32 and `True` in x64, plus
  `Exact flags: [True, False]` and `Lower times: [2.0, 3.0]` in each.
  The source notebook was checked to have null execution count and no outputs.
- `UV_CACHE_DIR=/private/tmp/chronox-docs-uv-cache scripts/check.sh --through A2`:
  exit 0. Ruff format: 24 files formatted; Ruff lint: all checks passed;
  mypy: no issues in 4 files. Own tests: x64 77 passed, 3 skipped; x32 80 passed.
  Independent A2 tests: x64 178 passed, 13 skipped, 196 deselected; x32
  186 passed, 5 skipped, 196 deselected.
- `git diff --check`: exit 0, no output.

## Remaining work

DOC-2 will document the A1–A2 constructors and diagnostics. The site does not
claim Slice A completion or numerical agreement with R. JupyterLab was provided
through the locked group, but an interactive session was not used as evidence.
