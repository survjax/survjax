# B1b.1 — Exponential interval probability override

Start / previous step: 8f9eb95; original worktree base: cc0e30f.
Maintainer authorized next implementation step; split built-in formulas into
bounded steps. This step adds only the Exponential log_interval_prob override.
Files: src/survjax/distributions/exponential.py,
tests/unit/test_interval_defaults.py, tests/unit/test_exponential_intervals.py,
tests/slow_tests.txt if timings require refresh, PLAN.md, REVIEW.md.
No new public API/module/dependency, protected path or docs changes.
Spec 4.2, 7.1–7.3: log P(a<T<=b)=-r*a+log(1-exp(-r*(b-a))).
Reuse the existing stable log-CDF calculation for the width, preserving tiny
positive mass without forming an underflowing rate*width. Substitute endpoints
before arithmetic; preserve equal/invalid/infinite endpoint policies.
Tests: keep generic coverage with a test subclass explicitly using base hooks;
add analytical/mpmath values, narrow-interval first/second derivatives,
broadcast/jit/vmap, endpoint and safe inactive-gradient tests in both dtypes.
Run full touched files in both modes, duration refresh if needed, fast A10 gate.
Keep the generated timing-only list in a separate commit if its churn is large.
Write REVIEW.md, commit, independent review (up to three rounds), then stop.

Remaining: B1b.2 Exponential conditional override; B1c Weibull interval and
conditional overrides; B2 likelihood integration; B3 full reference/numerical
matrix; B4 metadata pruning measurements and docs.
Docs to-do: explain built-in stable interval methods with expanded observation
support after likelihood integration.
