# A4.5 reviewer fix-up — F-1 and N-1 through N-4

Implemented on `A4.5` after `fe9aa8e`, against specification revision 21.
Authorized by the maintainer request to address `../review/slices/A4.5.md`.
Commit this fix-up, then stop for review. No public API or A5 changes.

## Reading order

1. `tests/unit/test_family_density.py`: the Quadratic primitive, the two new
   zero-time hazard gradient tests, strengthened override test and reference
   call-count test.
2. `src/survjax/distributions/_survival_methods.py`: `_safe_log`, simplified
   density/JVP defaults and raw log-hazard dispatch in `hazard`.
3. `PLAN.md`: the authorized review checklist and stopping point.

## Findings addressed

- **F-1:** added `Quadratic` with `log S=-r*t**2`, hence `h=2*r*t`.
  `test_zero_time_hazard_has_zero_parameter_gradient` asserts a finite
  derivative exactly 0. `test_zero_time_hazard_gradient_is_finite_in_mixed_batch`
  asserts a finite derivative exactly 2 for the sum at times `[0,1]`.
  Hand derivations are in the docstrings. Both detect the missing guard.
- **Corrected earlier evidence:** the previous REVIEW.md overstated
  `test_zero_hazard_does_not_poison_active_derivatives`. Its zero slot is
  independent of the differentiated parameter and it differentiates only the
  active output. It checks values/active derivatives but does not demonstrate
  that the zero-hazard logarithm is protected. The new Quadratic tests do.
- **N-1:** the override in `test_inherited_overrides_dispatch_and_policy` now
  deliberately uses `log(2*rate)`. With rate 2, inherited hazard is 4 at finite
  time and infinity, while generic density at time 1 is still `log(2)-2`.
  Expected values are hand-derived constants. Density and hazard dispatch are
  now distinguishable by this test.
- **N-2:** removed the duplicate positive-infinity guard in generic density
  and the integer cast in the JVP helper. No test failures or pass/skip-count
  reductions occurred: the wrapper supplies floating safe times and owns
  the density infinity rule. Neither redundant path was retained.
- **N-3:** hazard resolves the raw log-hazard method via `_survjax_raw`,
  honoring overrides without running the policy wrapper twice. The new
  reference-time call-count test checks exactly one call for each of the seven
  direct methods. The independent override-at-infinity test still passes.
- **N-4:** renamed the helper to `_safe_log(x)`, its safe operand to `safe_x`,
  and updated its docstring and both call sites. It logs a nonnegative value,
  mapping zero to minus infinity with a protected gradient.

## Invariants and spec-to-test mapping

- Spec 4.2/5: forward-mode JVP still differentiates the raw primitive on
  already-substituted times, with a ones tangent and ordinary pointwise
  broadcasting. Existing integer-time, paired/outer-shape and component/nested
  tests pass after removing the unreachable cast.
- Spec 4.2: generic density uses the primitive derivative independently of
  log-hazard overrides. The deliberately different override test proves this;
  hazard honors that override through raw-method dispatch.
- Spec 4.5: the outer wrapper remains responsible for time policy, including
  density at positive infinity. All independent value-policy and hazard-limit
  tests pass. No guarantees are added for infinite-density gradients.
- Spec 7.2: `_safe_log` substitutes before logarithms and stops only the
  placeholder. The new zero-time tests prove finite boundary gradients and
  mixed-batch isolation when the zero slot depends on the real parameter.
  Existing safe-reference and NaN-precedence tests remain unchanged.
- Spec 7.3: existing interior first/second parameter-derivative tests still pass
  under jit/vmap and forward/reverse checks. Tolerances are unchanged.

## D1 mutation evidence

Temporarily replaced `jnp.log(safe_x)` with `jnp.log(x)` in `_safe_log`,
leaving the final zero selection intact. Ran
`uv run pytest tests/unit/test_family_density.py -q -k zero_time_hazard`
in separate processes with `JAX_ENABLE_X64=1` and `0`:

- x64: **2 failed, 22 deselected** (0.70s), exit 1; both derivatives were NaN.
- x32: **2 failed, 22 deselected** (0.71s), exit 1; both derivatives were NaN.

The implementation was restored in a finally block before final validation.
The unmutated focused suite had **24 passed** in each mode (x64 3.45s,
x32 2.92s). The full gate below was run after restoration and also passes the
new tests. This establishes that F-1 catches the specific missing safe operand.

## Exact final validation

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.
`scripts/check.sh --through A4` exited 0:

- `guard: ok (A4.5)`.
- Ruff format: `38 files already formatted`; lint: `All checks passed!`.
- mypy: `Success: no issues found in 9 source files`.
- Own x64: **291 passed, 6 skipped** (11.59s).
- Own x32: **296 passed, 1 skipped** (10.89s).
- Independent x64: **676 passed, 25 skipped** (16.50s).
- Independent x32: **690 passed, 11 skipped** (13.37s).
- Final output: `check: all gates passed through A4`.

Nine new own cases were added per mode: two boundary-gradient tests and seven
call-count cases. All existing tests retain their outcomes. Existing skips are
mode-specific conversion/promotion cases, optional pandas and independent
finite-difference density checks restricted to x64. No protected tests were
changed, skipped or xfailed.

Whole-word search for the old logarithm helper name across `src/`, `tests/`
and `docs/` returned no matches (rg exit 1). `git diff --check` passed with no
output. No protected files, dependencies, exports or tolerance files changed.

Combined own/spec sessions, source doctests and documentation builds were not
repeated in this fix-up; their prior A4.5 evidence is in `e2c08af`. The review
checklist requires the full A4 gate, which passed on the final restored source.

## Deferred observations and stopping point

D-016 remains open for A5; no deviation status was changed. Carry forward the
review observations about `log_prob(+inf)` gradients and built-in Weibull
`log_prob(0)` parameter gradients at shape 1 (shape and scale), alongside the
existing survival-at-zero case. No boundary fix or built-in work was performed.
No new contract ambiguity was identified. No merge, publishing or A5 work.
