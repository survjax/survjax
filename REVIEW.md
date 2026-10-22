# A6.5 — Remaining likelihood derivative verification

Status: implemented, ready for review on `A6`, based on `dca2ee8`,
authorized by "ok get to it"
after the remaining-gap explanation. Spec revision 32, marker `cf4fef9`.
One test-only commit, then stop. Tutorial docs follow separate review steps.

## Read first

1. `tests/unit/test_likelihood_transformations.py`: two explicit finite-
   difference tests for Exponential rate and joint Weibull shape/scale.
2. `tests/impl_tolerances.py`: documented step sizes; existing derivative
   absolute/relative tolerances are unchanged.
3. PLAN.md's current A6.5 entry: narrowed scope, retained coverage and the
   maintainer's tutorial-writing instruction for subsequent docs.

## Invariants and test/spec mapping

- Spec 7.3/12.1: check_grads(order=2, modes=("fwd", "rev")) compares AD
  derivatives with central finite differences. Both public entry points,
  raw/prepared forms and eager/jit calls are named pytest cases: 8 per family,
  16 total. This adds the missing finite-difference evidence, without copying
  the existing analytical Jacobian/Hessian or invalid-row isolation matrices.
- Times [0.5,1,1.5] and event indicators [True,False,True] give a small mixed
  exact/right batch strictly inside the supported domain. Rate=1.5 and
  shape/scale=[1.5,2] are smooth positive interior points. The objective is
  the caller's mean of the unsummed contributions, keeping it O(1).
- Only the rate or joint shape/scale vector is perturbed. Observation times
  and event indicators are held fixed. No data derivatives, boundary
  derivatives, invalid-row derivatives or orders above two are claimed.
- Compiled functions receive time/event as arguments. The raw entry point or
  explicit prepared-data construction therefore sees traced observation
  inputs inside jit; both use the ordinary checked constructor.
- Finite-difference steps were selected and documented before the first run:
  h=1e-4 in x64, h=1e-3 in x32. For these smooth O(1) objectives/derivatives,
  O(h**2) truncation and O(eps/h) rounding scales fit the existing 1e-7/1e-3
  derivative targets. No tolerances were loosened and no points were changed
  in response to a failure; all 16 cases passed on the first run in each mode.
- Existing evidence retained: public analytical derivatives in
  test_likelihood_entry_points.py; invalid/unsupported-row and nested masking
  in the own core/parameter suites and independent A6 suites; compilation
  reuse and transfer guards in the independent likelihood entry-point suite;
  mixed valid eager debug_nans checks in test_likelihood_core.py.
- No src/, public API, docs/, spec, independent-test or dependency changes.
  D-026 remains open. The base.py cleanup remains deferred.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Focused tests in separate JAX_ENABLE_X64=1/0 processes:
  `uv run --locked --group test pytest
  tests/unit/test_likelihood_transformations.py -q --tb=short`.
  Both exit 0: 16 passed (13.74s each).
  Logs: `/private/tmp/survjax-a65-focused-x64.log` and `...-x32.log`.
- Focused Ruff format: 2 files left unchanged. Ruff lint: All checks passed!
  mypy src: Success: no issues found in 17 source files.
- `scripts/check.sh --through A6`: exit 0, final output
  `check: all gates passed through A6`. Guard: `ok (A6)`; Ruff:
  `66 files already formatted`, `All checks passed!`; mypy:
  `Success: no issues found in 17 source files`.
  Own x64: 722 passed, 7 skipped (135.91s); x32: 728 passed, 1 skipped
  (136.74s). Source doctests: 4 passed per mode (0.90s/0.89s).
  Independent through-A6 x64: 2023 passed, 28 skipped (96.00s);
  x32: 2027 passed, 24 skipped (91.82s). Existing skip counts are unchanged;
  no independent tests were edited, skipped or xfailed by this step.
  Full log: `/private/tmp/survjax-a65-check.log`.
- `../scripts/spec-since.sh dev`: exit 0, unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits and next step

Passing these representative directional finite-difference checks is not a
claim about all parameter values or every Hessian entry; existing analytical
and full-Jacobian/Hessian checks provide complementary evidence. The tests do
not add family behavior or features. Later Slice A tail, reference, install,
example and performance evidence remains separate.

No docs build was run for this test-only step. Subsequent docs should teach
one idea at a time: mathematics, code applying it, then interpretation of the
output, using coherent synthetic examples. Parameter diagnostics precede
likelihood usage; D-026 needs a wording decision for the triage guide. No
unit-test-style assertion sequences in the tutorial. Stop after this commit.

Test/comment diff: 87 added lines; the new test file has 80 lines.
