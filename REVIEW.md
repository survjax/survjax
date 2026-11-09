# B1c.2 — Conditional Weibull interval override

Independent review: round 1/3

Read PLAN.md B1c.2, spec 4.2 and 7.1–7.3, the new boundary-aware increment
wrapper and conditional method in distributions/_weibull_intervals.py, its
binding in weibull.py, then tests/unit/test_weibull_conditional.py.
Start / previous step: 68ba259; original worktree base: cc0e30f.

Invariants: valid finite positive Weibull parameters have positive mass on
any valid nonempty truncation window. Intersect before evaluation; inactive
numerators use the window, without re-intersection. The boundary wrapper
receives ordered nonnegative endpoints with finite lower endpoint; it delegates
finite positive interiors to the already-reviewed stable increment helper and
uses log H for a zero origin. Equal endpoints give log increment -inf; infinite
upper endpoint gives +inf. Substitution occurs before inactive operations.
The conditional expression cancels H(L) before arithmetic, then exponentiates
only the offset increment H(a)-H(L). Numerator and denominator remain in log
space through the existing log-CDF-from-log-hazard helper. Full coverage is
structurally zero and empty intersections are -inf; invalid inputs give NaN.
No new module, public API, dependency or likelihood dispatch in this step.

Test map:
- Spec 4.2/7.2/7.3: far-tail conditional values and natural-parameter gradients
  and all four Hessian mode combinations against 100-digit mpmath; eager/jit/vmap.
- Spec 7.1/7.2: k=2 tiny window with underflowing linear mass against mpmath.
- Spec 4.1/7.1: shape-one Exponential identity, finite/infinite windows, partial,
  full and empty intersections, lists and trailing-axis broadcasting.
- Spec 7.1: invalid endpoints, windows and parameters remain NaN.
- Spec 7.2/7.3: mixed finite-window full/partial/empty rows under jax_debug_nans;
  values, gradients and Hessian checked against direct analytical survival
  ratios at moderate inputs; check_grads order 2, fwd/rev on the same interior.

Checks:
- Touched Weibull interval suites, pytest -n 6: 10 passed per dtype.
- Ruff format/check, mypy src (23 files), git diff --check: passed.
- scripts/update_slow_tests.sh --through A10: completed, 439 entries; generated
  churn committed separately. Observed x64 duration suites: implementer
  1342 passed/8 skipped; spec 2332/28. Full x32 summaries not retained; script
  completion alone is not claimed as proof of those full test results.
- scripts/check.sh --fast --through A10: all gates passed. Implementer
  x64 1081 passed/8 skipped, x32 1086/3; doctests 4 each; spec x64 2154/28,
  x32 2158/24.
- No full non-fast gate, R regeneration, docs build or benchmark claimed.

Remaining: B2 likelihood integration; B3 full numerical/reference matrix;
B4 measured metadata pruning and docs. Direct invalid-parameter gradient
isolation is not added; likelihood preparation retains ownership of that rule.
