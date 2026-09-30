# A2 pre-merge bookkeeping review

## Read first

1. `src/chronox_jax/data.py`: the `from_bounds` docstring now explains equal
   endpoints when `exact` is false or omitted.
2. `DEVIATIONS.md`: the branch's duplicate D-006 is now D-011; D-011 and
   D-007–D-009 are marked superseded by revision 9, still awaiting maintainer
   closure. The main-line D-006 cited by spec section 13 stays open.
3. `src/chronox_jax/_data_validation.py` and
   `tests/unit/test_data_construction.py`: historical references now use D-011.

## Invariants and test evidence

Equal lower and upper bounds without `exact=True` remain a non-exact interval
with `invalid_interval=True`; construction returns data and only an explicit
host checkpoint raises `InvalidDataError`. No runtime behavior or public API
changed in this step. The independent A2 tests
`test_equal_endpoints_on_non_exact_row_set_issue` and
`test_equal_endpoints_without_exact_mask_set_issue` cover both forms; the own
`test_row_issues_checks_interval_order_only_after_endpoint_domains` covers the
exact `invalid_interval` bit for equal endpoints.

## Checks and uncertainties

`scripts/check.sh --through A2` passed with `UV_CACHE_DIR` set to the task's
temporary cache: guard passed; Ruff format reported `20 files already
formatted`; Ruff lint reported `All checks passed!`; mypy reported `Success: no
issues found in 4 source files`; own tests were `77 passed, 3 skipped` in
float64 and `80 passed` in float32; independent A2 tests were `178 passed, 13
skipped, 189 deselected` in float64 and `186 passed, 5 skipped, 189 deselected`
in float32. `git diff --check` passed with no output.

The four superseded branch entries remain open for maintainer closure. D-006
legacy diagnostic compatibility and D-010 A3 scalar-`ok` test mismatch remain
open and untouched. No A3 gate result is claimed by this bookkeeping step.
