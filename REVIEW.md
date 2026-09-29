# A2a docstring review fix-up

Branch: `a2-constructors`. Stop after this documentation fix-up for maintainer
review before A2b.

## Read in this order

1. The “A2a docstring review fix-up” section at the top of `PLAN.md`.
2. The class, flag-constructor, and `from_bounds` docstrings in
   `src/chronox_jax/data.py`.

## Invariants and requirement map

- The class docstring describes the normalized five fields, the sixth private
  issue leaf, and observation shape (spec 3.2). It no longer compares storage
  with R.
- The flag-constructor docstring explains Chronox's exact/right `event`
  coding, when to use censoring flags, active `time2`, truncation defaults,
  dtype, and broadcasting before comparing with R (spec 3.1–3.2). The
  maintainer's clarification that flags cover mixtures with left or interval
  censoring is preserved.
- The bounds-constructor docstring explains its censored interval, explicit
  exact mask, sentinel endpoints, shape, truncation, and dtype before its R
  comparison (spec 3.1–3.2). All three `WA` notes were addressed and removed.
- Only docstrings changed in source. Constructor signatures, normalization,
  validation, issue bits, and the public export are unchanged. Existing tests
  and the independent A2 gate check the runtime behavior.

## Checks run

Commands used `UV_CACHE_DIR=/private/tmp/uv-cache` because the sandbox cannot
write the default uv cache.

- `uv run ruff format --check .`: `17 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 3 source files`.
- `scripts/check.sh --through A2`: guard OK; own x64 `47 passed, 3 skipped`,
  own x32 `50 passed`; independent x64 `179 passed, 13 skipped, 109 deselected`,
  independent x32 `187 passed, 5 skipped, 109 deselected`; `check: all gates
  passed through A2`.

## Open items

No new spec ambiguity arose. D-001 remains formally open in `DEVIATIONS.md`.
`DataIssues` and `SurvivalData.with_issues` remain the separate A2b step.
