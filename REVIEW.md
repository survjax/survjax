# B1b.1 — Exponential interval probability override

Independent review: round 1/3

Read PLAN.md, spec 4.2 and 7.1–7.3, the Exponential.log_interval_prob
method, tests/unit/test_exponential_intervals.py, then the generic-test subclass
in tests/unit/test_interval_defaults.py.

Start / previous step: 8f9eb95; original worktree base: cc0e30f.
Plan archive commit: f0e15cc.

Invariants: valid rates are finite and positive. Empty/invalid intervals use
finite substitute endpoints before subtraction, so inf-inf is never evaluated.
The active formula is -rate*lower + log_cdf(upper-lower). The existing log-CDF
implementation calculates from log cumulative hazard, retaining log mass even
when rate*width underflows. No floors, clipping or density approximation.
Generic tests explicitly select the base hooks on a registered test subclass,
so adding the built-in override does not silently remove default-hook coverage.

Requirement / test map:
- Spec 4.2/7.2: high-precision tests at represented endpoints cover narrow,
  finite far-tail, product-underflow and infinite intervals; eager/jit/vmap.
- Spec 7.3: mpmath first/second derivatives on a narrow interval, all four
  forward/reverse second-order combinations; separate check_grads interior.
- Spec 7.1: equal, reversed, negative, NaN and infinite endpoints; invalid rates.
- Spec 7.2: full-support/right-tail/empty batch gradients and Hessian under
  eager jax_debug_nans, with analytically known -2r objective.
- Spec 4.2: explicit two-axis broadcasting check.

Checks run:
- Touched files, pytest -n 6 in each dtype: 15 passed per mode.
- Ruff formatting/lint, mypy src, git diff --check: passed.
- scripts/update_slow_tests.sh --through A10: completed, 359 entries. Its
  timing-only list is committed separately because regenerated churn exceeds
  the numerical step size. Observed full x64 results: implementer 1327 passed,
  8 skipped; spec 2332 passed, 28 skipped. Full x32 summaries were removed by
  the script; do not infer their outcome from its exit status alone.
- scripts/check.sh --fast --through A10 with refreshed list: all gates passed.
  Captured counts include x64 implementer 1145/8 and x32 spec 2159/24;
  remaining summaries were truncated in tool output, but final gate exit is 0.
- No R regeneration, docs build, or performance claim in this code-only step.

Remaining: conditional Exponential override (B1b.2), Weibull overrides (B1c),
likelihood integration (B2), full numerical/reference matrix (B3), pruning and
docs (B4). The generic conditional method benefits from this interval override
but still subtracts absolute log masses; its far-tail cancellation is not fixed
by this step. Mixed invalid-parameter derivative isolation remains owned by the
likelihood evaluator; no expanded direct-method guarantee is claimed here.
