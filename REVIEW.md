# B1c.1 — Weibull interval override

Independent review: round 3/3

Read PLAN.md, spec 4.2 and 7.1–7.3, distributions/_weibull_intervals.py,
its binding in weibull.py, then tests/unit/test_weibull_intervals.py.
Start / previous implementation: 6fab866; original worktree base: cc0e30f.
Plan archive: aafe3a5. Maintainer explicitly approved the new private module.

Invariants: the private increment helper receives finite 0<a<b and valid
parameters. For nearby endpoints, log(b/a)=log1p((b-a)/a); for separated ones,
log(b)-log(a) avoids an overflowing ratio. Inactive branches substitute before
operating. log(delta_H)=log H(b)+log(1-exp(-k*log(b/a))) reuses the guarded
log-CDF-from-log-hazard primitive, never subtracting nearly equal powers or
forming an underflowing k*log(b/a). The interval result is
-H(a)+log(1-exp(-delta_H)). Zero/infinite endpoints use existing direct methods;
invalid or empty inputs never reach the finite increment helper unchanged.
Parameters of that inactive piece are substituted too; real active parameters
retain their derivatives. No likelihood or conditional override is changed.

Test map:
- Spec 7.2: 400-digit mpmath powers for narrow, tiny-mass, wide ratio, far-tail,
  decreasing-hazard and tiny-shape inputs. The final case resolves an O(1e-312)
  power difference in x64. Eager/jit/vmap agree in both dtypes.
- Spec 7.3: 100-digit mpmath natural shape/scale gradient and full Hessian on a
  narrow interval; all four AD compositions; check_grads order 2 both modes.
- Spec 4.1/4.2: shape-one Exponential identity with two-axis broadcasting and
  Python-list endpoints.
- Spec 7.1: equal/reversed/negative/NaN/infinite endpoints and invalid parameters.
- Spec 7.2/7.3: mixed full/right/empty/infinite batch under jax_debug_nans,
  analytic value/gradient/Hessian at k=1 and scale=2.

Checks:
- Final touched file, pytest -n 6, x64 and x32: 5 passed each.
- Ruff format/check, mypy src (23 source files), git diff --check: passed.
- An initial Python-list probe failed in result_type; fixed array conversion
  before dtype selection and added list coverage, then reran both dtype suites.
- scripts/update_slow_tests.sh --through A10: completed, 397 entries; generated
  timing churn committed separately. Observed x64 full duration results:
  implementer 1337 passed/8 skipped; spec 2332/28. Final x32 logs not retained,
  so script completion is not claimed as proof of those full test results.
- scripts/check.sh --fast --through A10: all gates passed. Implementer
  x64 1133 passed/8 skipped, x32 1138/3. Doctests: 4 per mode. Spec:
  x64 2139 passed/28 skipped, x32 2146/21.
- No R fixtures regenerated, docs built, or performance claims made.

Remaining: B1c.2 conditional Weibull; B2 likelihood integration; B3 full
reference/numerical matrix; B4 measured metadata pruning and docs. The generic
conditional method still subtracts absolute log probabilities and is not
claimed to handle far-tail cancellation until B1c.2.

## Authorized review retry

Maintainer requested another reviewer attempt. Round 1 at e0fc2f1 performed
no review because checkout permissions blocked it; no findings were issued.
Implementation and tests are unchanged. Retry uses the recorded previous SHA
and finding IDs `none`.

## Round 2 fix-up: F-1

Reviewed 3f5742b01bf9635e9d1e61b98c059564e12c7532; verdict fix-first.
Added a narrow interval starting at 3 to the existing high-precision value
matrix. Unlike a=1, its log(a) is nonzero, exposing cancellation if the nearby
log1p path is removed. Widths: 1e-12 (x64), 1e-5 (x32), with expectations
computed at represented endpoints. Existing tolerances unchanged. No library
code or new test function added; duration list unchanged.
Validation: touched file 5 passed in each dtype. Fast A10 gate passed:
implementer 1133/8 (x64), 1138/3 (x32); doctests 4 each; spec 2139/28 and
2146/21. Ruff, mypy and whitespace checks passed. Full gate not rerun.
