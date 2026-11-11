# B3.5 — PH/AFT likelihood derivatives across censoring and truncation

Base / previous step: 40c022f78160f7dc1224e966bdf8e0c4fe5a2b95
([B3.4](../B3.4/PLAN.md), independent verdict: merge).
Original worktree base: cc0e30f. Dependencies: [B3.3 audit](../B3.3/REVIEW.md),
B2 likelihood integration, merged PH/AFT and four built-in families.
Cumulative gate: A12, existing observation feature flags unchanged.

Address B3.3 gap 2 in one test-only topic. Create
tests/unit/test_wrapper_censoring_derivatives.py (target below 250 lines),
with independent 80-digit mpmath references. Use section 4.4's PH survival
power/density and AFT time rescaling/density Jacobian, then section 2's
intersection and conditioning formulas. Cover PH and AFT over Exponential,
Weibull, LogNormal and LogLogistic in an ordinary finite parameter regime.

Each mixed batch includes all four kinds under absent, left, right and two-sided
truncation, partial overlaps at L/U, and whole-window censoring. Compare per-row
values, baseline/eta Jacobians and full mixed Hessians, using all four AD
compositions, eager and jit(vmap) at two distinct parameter vectors. Include
eager debug_nans, exact zeros for whole-window rows and an order-two check_grads
test of a compiled interior objective. Use existing tolerances without changing
values; add a bounded-domain rationale in tests/impl_tolerances.py if needed.
No claim about the PH extreme-scaling exception (spec 7.3), invalid-parameter
isolation or arbitrary custom baselines beyond the existing tests.

No src/, public API, dependency, docs/, protected path or feature-selection
changes. Run the complete new file in each dtype mode with six workers and
durations. If new tests exceed one second, run scripts/update_slow_tests.sh and
keep its mechanical refresh in a separate commit as in B3.2/B3.4. Run the fast
A12 gate, write REVIEW.md, commit and independently review, then stop.
Spec coverage: 2, 4.4, 7.1–7.3, 12.1–12.2.

## Docs to-do

The later likelihood docs can cite moderate-range wrapper derivatives across
the supported observation modes after validation. Remaining: B4 static-metadata
pruning and measurements, family-feature activation decisions, likelihood docs.
No Slice B completion claim. Reviewer-workflow investigation remains deferred.
