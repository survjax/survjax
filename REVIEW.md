# A6.4c review fix — Validate the distribution argument

Status: implemented, ready for review on `A6`, based on `b1edf82`, in response to
`../review/slices/A6.4c.md` and "look at the review and implement".
Spec revision 32, marker `cf4fef9`. One fix-up commit, then stop.

## Read first

1. `src/survjax/likelihood.py::log_likelihood`: the two-line S1 guard at the
   public entry point, before resolving data or calling distribution methods.
2. `tests/unit/test_likelihood_entry_points.py`:
   `test_non_distribution_argument_raises_clear_type_error` and the N1 citation.

## Findings, invariants and coverage

- S1: None, an unrelated object and a distribution class now raise TypeError
  with `dist must be a SurvivalDistribution instance`. This fulfills the
  public docstring's existing invalid-input-type promise (spec 6.1).
- The regression has named eager/jit cases for each invalid type. Jit closes
  over the invalid Python object so the guard itself runs during tracing;
  JAX does not reject an unsupported argument before reaching the function.
- Valid built-in/custom instances retain the same resolver and numerical core.
  The guard inspects only Python type, never array values, tracers or issue
  masks. Signatures, exports, dtype behavior and row isolation are unchanged.
  Focused own/independent API suites exercise valid calls and delegation.
- N1: the keyword-only test now cites spec 6.1. N2 explicitly requests no
  action here; a later cleanup can split the existing large base.py module.
- D-026 is unchanged and open. No spec, independent-test, docs/, dependency,
  direct-method or numerical change. No A6.5 work.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Before adding the guard, separate JAX_ENABLE_X64=1/0 runs:
  `uv run --locked --group test pytest
  tests/unit/test_likelihood_entry_points.py -k non_distribution -q --tb=short`.
  Both exit 1 as expected: 6 failed, 52 deselected (1.18s each). None/object
  raise AttributeError; class gives an internal JAX TypeError with the wrong
  message. Logs: `/private/tmp/survjax-a64c-s1-red-x64.log` and `...-x32.log`.
- After the fix, separate JAX_ENABLE_X64=1/0 and SPEC_THROUGH=A6 runs:
  `uv run --locked --group test pytest
  tests/unit/test_likelihood_entry_points.py
  tests/spec/test_a6_likelihood_entry_points.py
  tests/spec/test_a6_likelihood_method.py -q --tb=short`.
  Both exit 0: 113 passed (8.77s/8.74s).
  Logs: `/private/tmp/survjax-a64c-s1-focused-x64.log` and `...-x32.log`.
- Focused Ruff formatting: 2 files left unchanged. Ruff lint:
  `All checks passed!`. mypy src: no issues found in 17 source files.
- `scripts/check.sh --through A6`: exit 0, final output
  `check: all gates passed through A6`. Guard: `ok (A6)`; Ruff:
  `65 files already formatted`, `All checks passed!`; mypy:
  `Success: no issues found in 17 source files`.
  Own x64: 706 passed, 7 skipped (120.82s); x32: 712 passed, 1 skipped
  (122.64s). Source doctests: 4 passed per mode (0.92s/0.89s).
  Independent through-A6 x64: 2023 passed, 28 skipped (92.44s);
  x32: 2027 passed, 24 skipped (88.66s). Existing skip counts are unchanged;
  no independent tests were edited, skipped or xfailed by this follow-up.
  Full log: `/private/tmp/survjax-a64c-s1-check.log`.
- `../scripts/spec-since.sh dev`: exit 0, unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits

No docs build was rerun: public docstrings and generated entries did not
change. R fixtures, wheel installation and performance checks remain outside
this follow-up. The full Slice A completion gate is not claimed. The own
regression targets the public function because a non-distribution receiver
has no inherited likelihood method. Source/test diff: 19 changed lines.
