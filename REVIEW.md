# A8 method-only likelihood compatibility

Approved by "ok do it" after the integration audit. The maintainer-started
merge of main be1a5b3 was completed as f4e0eae with the resolved records.
This is the separate A8 compatibility test step, then a review stop.

## Reading order

1. tests/reference/test_exponential_r.py: retain prepared/raw method tests;
   remove their obsolete function duplicates; use the method for R totals.
2. tests/reference/test_weibull_bc_like_r.py: remove the function case axis
   and branch; all method prepared/raw eager/jit/vmap checks remain.
3. PLAN.md: current approved step, separate docs correction and A8 closure.

## Invariants and requirement mapping

- Spec 6.1/7.1/10.2, revision 35+: the sole public likelihood entry point is
  dist.log_likelihood. No alias is restored and no library code is changed.
- Spec 3.3/12.1: fixture loading, active-dtype inputs, host float64 expected
  R values, all shape/dtype assertions and central tolerances are unchanged.
- Spec 6.1/12.1: both raw and prepared method calls still compare each ordered
  likelihood row with both sources for every exponential rate and Weibull AFT
  vector. Eager/jit/vmap comparisons keep dynamic distribution arguments.
- Spec 4.6/5/11.1/12.1: observed-time predictions and time-first stored-time
  grids are unchanged, including both Weibull constructors and all groups.
- Separately stored R totals remain supplemental checks. Exponential totals
  now call the method; their expected numbers and tolerances are unchanged.

Only tests/reference plus PLAN.md/REVIEW.md change in this step. No fixture,
loader, tolerance, dependency, source, protected-test, spec or docs/ changes.
R is not run; all expectations remain committed R outputs. The related R
package paths remain limited evidence, not independent numerical oracles.

## Focused validation

With UV_CACHE_DIR=/private/tmp/survjax-uv-cache and JAX_ENABLE_X64=<1|0>:
`uv run --group test pytest -q tests/reference/test_exponential_r.py
 tests/reference/test_weibull_bc_like_r.py`:

- float64: 111 passed in 9.05s.
- float32: 111 passed in 8.54s.

Logs: /private/tmp/survjax-a8compat-focused-x64.log and
/private/tmp/survjax-a8compat-focused-x32.log. The retained counts are 33
exponential and 78 Weibull cases per dtype; 36 former function-only duplicate
cases were removed, not numerical coverage of the remaining public method.

Sensitivity plugin /private/tmp/survjax_a8compat_control.py temporarily rolls
method likelihood output rows by one and reverses Weibull survival-grid
columns. Committed source/fixtures are untouched. Both suites run with
`-p survjax_a8compat_control` and PYTHONPATH=/private/tmp, separately per dtype:
48 expected assertion failures / 63 passes in 4.54s x64 and 4.65s x32.
All 36 ordered-row checks reject the roll; all 12 grid checks reject swapped
columns. Totals retain their sums and pass. Logs:
/private/tmp/survjax-a8compat-control-x64.log and
/private/tmp/survjax-a8compat-control-x32.log. These expected child exit-1
runs are corruption-detection evidence, not unchanged-suite failures.

## Gate and remaining limits

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
uses main's incoming default six pytest-xdist workers. Full log:
/private/tmp/survjax-a8compat-gate.log. Exact final outcome recorded below.

The protected acceptance fixture remains absent. No through-A8 or Slice A
completion claim. No new deviations. No docs build or R regeneration in this
code step. The known obsolete function/method coverage sentence in
builtin-families.md is queued for the separate planned docs correction.
Stop after this compatibility commit; do not begin that docs step yet.

Final gate: exit 0, `check: all gates passed through A7`.
- Guard OK; Ruff: 78 files already formatted / All checks passed!
- mypy: Success: no issues found in 17 source files.
- Own float64: 943 passed, 7 skipped in 47.23s.
- Own float32: 949 passed, 1 skipped in 47.02s.
- Source doctests: 4 passed per dtype in 1.15s / 1.09s.
- Independent through A7 float64: 2114 passed, 28 skipped in 46.76s.
- Independent through A7 float32: 2118 passed, 24 skipped in 46.07s.

The gate's xdist summaries do not print deselection counts. No exclusions,
new skips or protected-test changes were introduced in this step. Final Ruff
format/lint and git diff --check pass after recording results. The two inherited
blank-line warnings in protected tests belong to incoming main and are not
part of this compatibility diff; they remain untouched.
