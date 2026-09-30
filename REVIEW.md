# Review: DOC-2 A1–A2 documentation

## Read first

1. `docs/survival-data.md`: both input forms, diagnostics, dtype policy and R conventions.
2. `docs/index.md`: navigation and commands to build and open rendered HTML.
3. `docs/playground.ipynb`: synthetic interactive path, including the maintainer's
   exploratory `data` cell with saved outputs cleared.
4. `PLAN.md`, DOC-2 amendment: authorized scope and preservation of local edits.

## Invariants and coverage

- Documentation describes the implemented A1–A2 `SurvivalData` API only.
  A3 indexing/static metadata and all likelihood methods are labelled planned.
- Every guide code cell checks its stated result. The status and mixed-flag
  cells check normalization (spec 3.1–3.2). The bounds cell checks exactness,
  unbounded endpoints and truncation (3.1–3.3). The diagnostic cell checks
  `uninformative`, `invalid_rows`, `ok` and `InvalidDataError` (2, 6.2).
  The misuse/dtype cell checks `TypeError` and float32/float64 behavior (3.3,
  6.2). The compiled cell checks construction and issue access inside `jit`,
  with explicit host raising afterward (6.2).
- The notebook preserves the maintainer's `data` code cell. All source notebook
  outputs and execution counts are cleared; the generated pages contain results
  from fresh execution.
- R comparison describes input conventions only. Its source is the official
  `survival::Surv` help page:
  https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html
  There is no claim of numerical agreement or R fixture verification.

## Checks and rendered-page inspection

- `UV_CACHE_DIR=/private/tmp/chronox-docs-uv-cache scripts/build_docs.sh`: exit 0.
  Both x32 and x64 builds executed the playground and guide without Sphinx
  warnings or execution errors. Jupyter printed a local TCP transport warning.
- Generated HTML inspection: both index pages link to the guide; both playground
  pages show `Bounds: [2.0, 0.0], [2.0, 3.0]`; all source notebook cells have
  empty outputs and null execution counts. The x64 index shows the macOS
  `open docs/_build/html-x64/index.html` command.
- Rendered guide inspected with headless Chrome at 1280x900: title, navigation,
  introductory text and first code cell are visible and readable. The screenshot
  is a local temporary artifact, not committed.
- `UV_CACHE_DIR=/private/tmp/chronox-docs-uv-cache scripts/check.sh --through A2`:
  exit 0. Ruff format: 25 files formatted; Ruff lint: all checks passed; mypy:
  no issues in 4 source files. Own tests: x64 77 passed, 3 skipped; x32 80
  passed. Independent A2 tests: x64 178 passed, 13 skipped, 196 deselected;
  x32 186 passed, 5 skipped, 196 deselected.
- `git diff --check`: exit 0, no output.

## Limits

The existing illustrative simulation guide remains outside the rendered site.
The guide documents representation of left/interval censoring and truncation;
it does not imply those likelihood modes are implemented. No R runtime or
reference fixture was used in this step.
