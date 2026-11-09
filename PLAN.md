# B1c.1 — Weibull interval probability override (approved module split)

Start / previous step: 6fab866; original worktree base: cc0e30f.
Maintainer explicitly approved the proposed private module in this conversation.
Plan archive commit: aafe3a5.
weibull.py is already 330 lines; B1b.2 review N-1 requests raising a family-local
split before the Weibull overrides.

Files:
- src/survjax/distributions/_weibull_intervals.py (new): stable log cumulative-
  hazard increment and interval-method implementation, typed private functions.
- src/survjax/distributions/weibull.py: bind the spec-named interval override.
- tests/unit/test_weibull_intervals.py: mpmath values and derivatives,
  narrow/wide/tail cases, shape-one exponential identity, zero/infinite/empty
  endpoints, invalid inputs, jit/vmap/broadcasting and inactive-gradient safety.
- PLAN.md, REVIEW.md, tests/slow_tests.txt if timings require regeneration.
No new public API or dependency. No protected path or docs edit.

Spec 4.2/7.2: log interval mass = -H(a) + log(1-exp(-delta_H)).
Use delta_H = H(b)*(1-exp(-k*log(b/a))), evaluated in log space; use
log1p((b-a)/a) for nearby endpoints, log(b)-log(a) for separated endpoints
without forming overflowing ratios. Handle zero/infinite endpoints through
safe substitutions and the existing direct-method limits. Reuse the stable
log-CDF-from-log-hazard primitive, retaining positive underflowing increments.
Never subtract nearly equal powers or approximate an interval by a density.
Keep implementation/tests about 300 changed lines; conditional override is a
separate B1c.2 step. Run touched suites both dtypes, fast A10 gate, and independent
review before stopping. Generated timing-only churn may be a separate commit.

Remaining: B1c.2 conditional Weibull; B2 likelihood integration; B3 complete
reference/numerical matrix; B4 pruning measurements and docs.
Docs to-do: built-in interval behavior and expanded observation support after B2.
