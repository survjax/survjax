# Review: A7.1 — Likelihood tails and overflow

## What to read, in order

1. `tests/unit/test_likelihood_tails.py`: local reference formulas, then
   scalar tail cases, mixed transformations, analytical gradients and overflow.
2. `tests/impl_tolerances.py`: three added comment lines explaining why the
   existing relative-only tail targets also apply to these likelihood cases.
3. The A7 entry at the top of `PLAN.md`: approved scope and later review stops.

This step adds 36 test cases in 231 lines, plus three tolerance-comment lines.
There are no source, API, numerical tolerance, dependency, docs or spec-test
changes. No spec ambiguity or new deviation was found.

## Invariants and expected-value origins

- Tests exercise only the supported exact/right subset without truncation.
  Exponential uses the public function with raw inputs; Weibull uses the public
  method with prepared data. A6 already covers their full equivalence matrix.
- mpmath uses 100 decimal digits and the concrete inputs after JAX dtype
  rounding. Expected values come from `log(r)-r*t`, `-r*t`, and Weibull
  `event*(log(k)-log(b)+(k-1)*log(t/b))-(t/b)**k`, not library outputs.
- Parameter derivatives use the hand derivatives written in the tests,
  evaluated with mpmath for the small terms. Both AD modes are checked.
- `TAIL_ATOL=0` and the unchanged `TAIL_RTOL` prevent tiny normal nonzero
  values or derivatives from passing as zero. O(1) anchors and analytical
  derivative zeros use the unchanged ordinary tolerances. Infinite values are
  compared exactly. References do not depend on subnormal behavior.
- When probabilities underflow but log contributions are representable,
  contributions must remain finite. At Weibull `k=50,b=1,t=1e10`, the true
  log contribution is about `-1e500`: revision 34 requires `-inf` with clear
  data/parameter diagnostics. Overflow tests run outside NaN debug mode.
- The finite neighbour's shared-parameter forward derivative and per-row
  reverse derivative are checked. Derivatives of the infinite row and shared
  reverse derivatives next to it are outside spec 7.3 and are not asserted.
- JAX's process dtype mode is supplied by the runner; tests never change it.
  NumPy and mpmath conversions are host-side reference work on concrete inputs.

## Requirement-to-test map

| Spec | Tests in the new module | Evidence |
| --- | --- | --- |
| 2, 4.1, 7.1, 12.1 | `test_exponential_tail_contribution_matches_mpmath`, `test_weibull_tail_contribution_matches_mpmath` | 26 named exact/right cases: scale changes, shapes below/at/above one, early events, tiny normal H, probability underflow and finite H around 1e32 |
| 5, 7.2–7.3 | `test_mixed_tail_rows_match_mpmath_under_transformations` | Mixed tiny/interior/upper-tail vectors match independent references eagerly, under jit and row-wise vmap; both public entry points represented |
| 7.3, 12.1 | `test_exponential_tail_rate_derivative_matches_algebra`, `test_weibull_tail_parameter_derivatives_match_algebra` | Hand-derived first derivatives at tiny-H and finite upper-tail points, forward and reverse mode, zero absolute tolerance |
| 7.1, revision 34 | `test_overflowing_log_contribution_is_negative_infinity` | Overflowing exact/right contributions are -inf eagerly/jit/vmap, without diagnostic issues; a finite neighbour keeps its mathematical value |
| 7.3, revision 34 | `test_finite_neighbour_forward_derivative_survives_overflow`, `test_finite_neighbour_reverse_derivative_survives_per_row_overflow` | Only the derivatives still guaranteed next to an infinite row |

Existing direct-method evidence remains in `test_log_cdf_tails.py`,
`test_exponential.py`, `test_weibull.py`, `test_family_boundaries.py`,
`test_family_infinity.py` and `test_family_mixed_infinity.py`. The existing
`test_likelihood_transformations.py` supplies second-order interior finite
differences; the A6 parameter suites supply declared-violation isolation.

## Checks run

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1,
pytest 9.1.1; device `[CpuDevice(id=0)]`.

- `.venv/bin/ruff format tests/unit/test_likelihood_tails.py`: one file
  reformatted; `.venv/bin/ruff check tests/unit/test_likelihood_tails.py`:
  `All checks passed!`.
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A7 .venv/bin/python -m pytest -q -ra
  tests/unit/test_likelihood_tails.py tests/spec/test_a7_likelihood_mpmath.py
  tests/spec/test_a7_likelihood_tails.py`: **91 passed in 7.93s**.
  Same command with `JAX_ENABLE_X64=0`: **91 passed in 7.94s**. These include
  all 36 new cases; neither process skipped a focused test.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
  passed. Protected-path guard `ok (A7)`; Ruff
  `68 files already formatted` and `All checks passed!`; mypy
  `Success: no issues found in 17 source files`; own float64 suite
  **758 passed, 7 skipped in 140.27s**, own float32 suite
  **764 passed, 1 skipped in 141.61s**; source doctests **4 passed in 0.87s**
  (float64) and **4 passed in 0.88s** (float32). Independent suites through A7:
  **2123 passed, 28 skipped, 44 deselected in 94.88s** (float64),
  **2127 passed, 24 skipped, 44 deselected in 94.29s** (float32).
  Final output: `check: all gates passed through A7`. The 44 later-step tests
  were deselected by the maintainer's gate. No skips or exclusions were added.
- `git diff --check`: clean.

## Remaining work and limits

A7.2 boundary/inactive-branch tests and the separate A7 numerical documentation
step have not started. The gate also runs the independent boundary/debug tests,
but the planned implementer tests remain A7.2 work. No R fixture generation,
docs build, GPU check, installation smoke test or performance measurement was
run for this test-only step. No claim of A7 or Slice A completion is made.
Stop after this commit for maintainer review.
