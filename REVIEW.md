# B1b.2 — Conditional Exponential interval override

Independent review: round 1/3

Read PLAN.md (B1b.2), spec 4.2 and 7.1–7.3, Exponential's conditional
interval method, then tests/unit/test_exponential_conditional.py.
Start / previous step: 6a0e342; original worktree base: cc0e30f.

Invariants: finite positive exponential rates have positive mass on every
nonempty valid truncation window. Intersect endpoints first, then substitute
the window for inactive numerators. Cancel exp(-rate*L) analytically, before
multiplication: -rate*(a-L) + log F(b-a) - log F(U-L). The existing stable
log-CDF handles zero/infinity and products too tiny for linear representation.
Full coverage is structurally zero; empty intersection is -inf. Invalid
endpoints, windows and parameters give NaN. No likelihood dispatch changed.

Test map:
- Spec 4.2/7.2: far-tail conditional probability at L=2**40 (x64), 2**20
  (x32), versus independent mpmath survival-ratio formula after memorylessness.
- Spec 7.3: same case verifies eager/jit/vmap, analytic first and second
  derivatives via mpmath, all four AD mode compositions, and check_grads order 2.
- Spec 7.1/7.2: tiny rate/window product that underflows in linear space,
  checked against mpmath expm1 ratio without treating the denominator as zero.
- Spec 7.1: partial/full/empty intersections; exact window boundaries;
  invalid and infinite endpoints and windows; invalid rates; batch broadcasting.
- Spec 7.2/7.3: mixed full/right/empty/infinite rows with eager jax_debug_nans,
  objective -r, gradient -1 and Hessian 0.

Checks run:
- Three interval test files, pytest -n 6, both dtype modes: 20 passed each.
- Ruff format/check, mypy src, git diff --check: passed.
- scripts/update_slow_tests.sh --through A10: completed, 358 slow entries.
  Observed full x64 duration suites: implementer 1332 passed/8 skipped;
  spec 2332/28. Full x32 summaries not retained; script completion alone is
  not claimed as proof of that full suite (script tolerates test failures).
- scripts/check.sh --fast --through A10: all gates passed. Implementer:
  x64 1150 passed/8 skipped, x32 1155/3. Doctests: 4 passed each mode.
  Spec: x64 2156 passed/28 skipped, x32 2160/24.
- No docs build, R regeneration, or performance benchmark for this code step.

Remaining: B1c Weibull interval/conditional overrides; B2 likelihood integration;
B3 full numerical/reference matrix; B4 measured pruning and docs. No expanded
invalid-parameter derivative-isolation guarantee for direct hooks; likelihood
parameter preparation still owns that guarantee. Source file is now 332 lines,
slightly above the approximate 300-line guideline, with one family responsibility
and no new helper module or abstraction introduced solely to move lines.
