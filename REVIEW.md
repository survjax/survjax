# A6.4b1 review fix-up — Two nested levels and clearer failure messages

Status: requested review fixes implemented on `A6`, based on `2e7ae42`.
Spec revision 32, marker `cf4fef9`; independent snapshot `08d97a6`.
No library source changed. Stop after this fix-up commit; no A6.4b2 work.

## Read first

1. `tests/unit/test_likelihood_parameters_depth.py`: a wrapper nested twice
   around a Weibull, with a deepest NaN scale and valid shared parameters in
   both wrapper levels. Transform names and eager/jit appear in pytest ids.
2. `tests/unit/test_likelihood_parameters.py`: existing transform loops now
   name failures in finite-value assertions and allclose messages, including
   jit-vmap. Existing cases and mathematical coverage are unchanged.
3. N1 below: the explicit broadcast remains, and the earlier test claim is
   corrected. PLAN.md records the authorized fix-up scope.

## Invariants and test/spec mapping

- Spec 7.1/7.3: the new test compares valid-row values, jacfwd/jacrev and all
  four second-derivative compositions, eagerly and under jit, with the
  isolated analytical formula `-exp(a+b)*(t/scale)**1.5`. Both wrapper
  parameters and all three deepest scales are coordinates of the same input
  vector. The bad scale is unused by the expected formula, so valid-row
  cross derivatives with respect to it must remain finite and zero.
- A nonlinear Weibull baseline makes omitting the deepest masking observable
  in reverse derivatives. The test differentiates selected valid contributions
  only; no invalid-row likelihood derivative is asserted. It adds 14 cases.
- N1: keep `_likelihood_parameters.py`'s explicit broadcast, matching the
  specification and making shapes visible to readers. Deleting only that
  line is behaviorally equivalent because where broadcasts its operands.
  The structural tests catch skipped masking of rebuilt leaves, not removal
  of this redundant broadcast. No test is claimed to distinguish equivalent
  implementations. The reviewer's M1 evidence is accepted, not rerun here.
- N2: transform names identify which existing comparison fails. The case
  builder and existing assertions remain intact; no broader test refactor.
- No public API, parameter policy, tolerance, dependency, docs, spec or
  independent-test change. New coverage is helper-level, before integration
  with the likelihood evaluator in A6.4b2/c.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Final Weibull depth suite, separate JAX_ENABLE_X64=1/0:
  `uv run --locked --group test pytest
  tests/unit/test_likelihood_parameters_depth.py -q --tb=short`.
  Both exit 0: 14 passed in 4.90s per mode.
- An initial linear Exponential variant also passed (14 per mode, 4.64s);
  replaced it with Weibull to expose a missed deepest mask.
- In-memory mutation only, separate dtype processes:
  `uv run --locked --group test python
  /private/tmp/survjax-a64b1-depth-mutation.py`.
  It skips parameter masking at depth two, retaining masks above it, then
  runs the depth suite with `-k 'eager and rev'`. Each pytest run returns 1:
  4 failed, 10 deselected (2.14s/2.13s), all at the finite-derivative assertion.
  The harness exits 0 because these failures are expected. It changes no
  repository files; this is evidence that the new regression detects the gap.
- Focused Ruff format/lint checks pass; both test files are formatted.
- `scripts/check.sh --through A5`: exit 0, final output
  `check: all gates passed through A5`. Guard: `ok (A6)`; Ruff:
  `62 files already formatted`, `All checks passed!`; mypy:
  `Success: no issues found in 16 source files`.
  Own x64: 586 passed, 7 skipped (84.71s); x32: 592 passed, 1 skipped
  (89.04s). Source doctests: 4 passed per mode (0.92s/0.98s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 769 deselected
  (24.42s); x32: 1261 passed, 21 skipped, 769 deselected (20.85s).
  Full log: `/private/tmp/survjax-a64b1-review-check.log`.
- `git diff --exit-code 2e7ae42 -- src`: exit 0, no output.
- `../scripts/spec-since.sh dev`: exit 0, spec unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits and uncertainties

No new ambiguity or deviation. Public A6 likelihood acceptance tests remain
for the public integration step; helper coverage does not establish complete
likelihood conformance. No docs build, R fixture, install or benchmark run.
