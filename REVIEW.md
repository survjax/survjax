# A6.4b2 nit N3 — Name the current-field check

Status: implemented on `A6`, based on `1ff4ced`, as explicitly requested.

## Read first

1. `src/survjax/_data_validation.py::current_field_issues`: renamed to match
   its docstring and its inspection of current normalized field values.
2. `src/survjax/likelihood.py::_unusable_data_rows`: updated import and call;
   the caller still combines current-field checks with stored issue bits.

## Invariants and coverage

Only the private identifier changed: three source lines. Field rules,
construction diagnostics, numerical behavior and public API are unchanged.
Existing core tests cover reconstructed-field validation and preservation of
stored issues (spec 3.3/6.2/7.1), plus likelihood values and transformations.
No new test is needed for the rename. D-026 remains open; no spec change.

## Checks actually run

All uv commands use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- `uv run ruff format --check .`: exit 0, 64 files already formatted.
- `uv run ruff check .`: exit 0, All checks passed!
- `uv run mypy src`: exit 0, no issues found in 17 source files.
- `uv run --locked --group test pytest tests/unit/test_likelihood_core.py
  -q --tb=short`, separate processes with `JAX_ENABLE_X64=1` and `0`:
  both exit 0, 27 passed (9.93s/9.46s).
- `git diff --check`: exit 0, no output.
- Search confirms no references to the previous helper name remain.

## Limits

The full through-A5 gate was not rerun for this identifier-only change; its
last passing run is recorded in commit `1ff4ced`. No docs, spec-test or spec
edits, no new public API and no A6.4c work. Stop after this fix-up commit.
