# A2a review fix-up — direct `row_issues` tests

Branch: `a2-constructors`. This is a test-only fix-up to the A2a commit. Stop
here for maintainer review before A2b.

## Read in this order

1. The fix-up section at the top of `PLAN.md`.
2. `tests/unit/test_data_validation.py`: eight direct tests of the private
   `row_issues` helper.
3. `src/chronox_jax/_data_validation.py`: the unchanged implementation under
   test, especially the mask ordering and derived interval/window rules.

## Invariants and test map

- `row_issues` receives raw-input fault masks from its caller and keeps each
  distinct. `test_row_issues_preserves_each_raw_input_mask` checks the five
  supplied masks and their int32 bit values.
- Endpoint presence and domains precede interval order.
  `test_row_issues_checks_interval_order_only_after_endpoint_domains` checks
  equal/reversed intervals and suppression of spurious order faults.
- `0 <= L < U` and exact `L < t <= U` hold row-wise. The truncation and exact
  membership tests check each boundary directly.
- A right-censored `c=L` is valid; `c<L` is invalid. A censored set with
  `upper<=L` or `lower>=U` has an empty intersection. The right-censoring and
  empty-intersection tests check these inequalities and full-window cases.
- Independent faults can overlap without changing the observation shape or
  int32 dtype. The overlap test checks a `(2,2)` bitfield. The transformation
  test checks the same per-row results through eager `vmap` and `jit(vmap)`.

The expected bit values are fixed in the tests from the ten named rules in
spec 6.2 and the A2a bit assignment. The expected valid/invalid rows follow
the explicit observation and truncation intervals shown in test docstrings and
comments; none are computed by calling a constructor as an oracle.

## Checks run

All commands used `UV_CACHE_DIR=/private/tmp/uv-cache` because the sandbox
cannot write the default uv cache.

- `JAX_ENABLE_X64=1 uv run pytest tests/unit/test_data_validation.py -q`:
  `8 passed`.
- `JAX_ENABLE_X64=0 uv run pytest tests/unit/test_data_validation.py -q`:
  `8 passed`.
- `uv run ruff format --check .`: `17 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 3 source files`.
- `scripts/run_tests.sh`: x64 `47 passed, 3 skipped`; x32 `50 passed`.
- `scripts/check.sh --through A2`: guard OK; same static and own-test results;
  independent x64 `179 passed, 13 skipped, 109 deselected`; independent x32
  `187 passed, 5 skipped, 109 deselected`; `check: all gates passed through A2`.

## Open items

No implementation defect surfaced in these tests, so no source code changed.
`DataIssues` and `SurvivalData.with_issues` remain the separate A2b step.
D-001 remains formally open in `DEVIATIONS.md`.
