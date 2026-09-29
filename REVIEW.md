# A2b documentation fix-up

Branch: `a2-constructors`. Stop for maintainer review before A3.

## Read in this order

1. The “A2b documentation fix-up” section near the top of `PLAN.md`.
2. The `row_issues` and `flag_fields` docstrings in
   `src/chronox_jax/_data_validation.py`.

## Invariants and evidence

- `flag_fields` normalizes flag-constructor inputs, builds the five stored
  fields and input-fault masks, and calls `row_issues`. Both eager construction
  and `with_issues` use its result (spec 3.1–3.2, 6.2).
- `row_issues` adds interval, truncation, and observation-window faults and
  packs the ten rule masks into the int32 bitfield. `from_bounds` calls it
  after its own normalization (spec 6.2).
- Only these docstrings changed in source. The existing unit and independent
  tests continue to cover their runtime behavior; no new test was added for
  documentation text.

## Checks run

All commands used `UV_CACHE_DIR=/private/tmp/uv-cache` because the sandbox
cannot write the default uv cache.

- `uv run ruff format --check .`: `19 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 4 source files`.
- `scripts/check.sh --through A2`: guard OK; own x64 `59 passed, 3 skipped`,
  own x32 `62 passed`; independent x64 `179 passed, 13 skipped, 109 deselected`,
  independent x32 `187 passed, 5 skipped, 109 deselected`; `check: all gates
  passed through A2`.

## Open items

No new spec ambiguity arose. D-001 remains formally open in `DEVIATIONS.md`.
A3 indexing and static metadata remain for a later step.
