# A5.6 review fix-up — F-1/F-2, N-1/N-2 (2026-10-19)

Addresses `../review/slices/A5.6.md`, including the maintainer's F-2 decision,
on the existing `A5.6` branch after `a4bea80`. Narrative docs remain step 2.
The spec, independent tests and deviation statuses are unchanged.

## Read first, in order

1. `_survival_methods.py:apply_policy`: hook-NaN precedence now applies only to
   the four base-boundary methods and log density. Every method still preserves
   its own NaN output. This follows section 4.5 and replaces the old approved
   plan's broader rule; no spec change is needed.
2. `test_family_mixed_infinity.py:test_nan_hook_preserves_analytic_hazard_limit`:
   an analytic override gives log(rate)/rate at infinity despite a NaN hook,
   in eager/jit/vmap. The same family's log survival/log density stay NaN.
3. `test_family_infinity.py:test_tiny_cure_fraction_keeps_log_cdf_infinity_relative_precision`:
   fraction=1e-12 gives log1p(-fraction), not zero, with existing zero-atol tail
   tolerances. Eager/jit/vmap are covered; M4 evidence is below.
4. Cure and NaNLimit fixture changes in those two modules: removed their dead
   infinity branches so primitive formulas rely on the base and limit hooks.
   `base.py` documents hook ownership and that the default does not raise for
   parameter values (N-2, prose rather than an invented exception type).

## Invariants and test-to-spec mapping

- F-1 (4.2/4.5/7.2/12.1): derive log F(inf) from log S(inf) with safe
  log1mexp. The expected value is the analytical identity log(1-p), evaluated
  by NumPy log1p; its expansion is -p-p**2/2-... . Zero absolute tolerance
  rejects an erroneous zero; no tolerance was added or loosened.
- F-2 (4.5): the hook's NaN governs base-supplied infinity values, including
  log_prob(inf). Analytic hazard limits remain family-supplied. Generic hazard
  defaults remain NaN at infinity, with or without a NaN hook. Existing
  NaN-hook density tests and boundary acceptance cases remain regressions.
- Safe time substitution, active hook derivatives, batching/dtypes, reference
  reuse and primitive raw-method unwrapping are unchanged. No floor, clipping,
  epsilon, parameter validation or new stop_gradient was introduced.
- N-1's naive cure primitives now run the existing value/first-derivative and
  mixed-entry Hessian suites without their former infinity guards. The base
  must provide safe inputs; the primitives no longer hide an unsafe call.

## Checks actually run

All commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Focused pytest:
`uv run --locked --group test pytest tests/unit/test_family_infinity.py tests/unit/test_family_mixed_infinity.py -q --tb=short`,
in separate processes with JAX_ENABLE_X64=1 and 0.

- Before the F-2 implementation fix: both modes exited 1, **2 failed,
  47 passed** (6.63s x64, 6.77s x32). Exactly the new analytic log-hazard and
  hazard cases failed: infinity became NaN while finite times were correct.
- Reviewer mutant M4 temporarily replaced only the boundary safe log1mexp with
  log(1-exp). The new F-1 test exited 1 in both modes, one failure each:
  x64 relative error **2.21217201e-5** (atol 0, rtol 1e-8; 0.62s),
  x32 actual **0** versus approximately **-1e-12** (atol 0, rtol 1e-4; 0.63s).
  A finally block restored the source before any subsequent checks.
- After the fix and fixture cleanup: focused modules exited 0 in both modes,
  **49 passed** (5.85s x64, 6.11s x32).
- `uv run ruff format` on the four source/test files: exit 0, one file
  reformatted, three unchanged. `git diff --check`: exit 0, no output.

- `scripts/check.sh --through A5`: exit 0, `check: all gates passed through A5`.
  Guard passed on A5.6; Ruff reported 53 files already formatted and all checks
  passed; mypy reported no issues in 13 source files. Exact dtype results:

  | Suite | x64 | x32 |
  | --- | --- | --- |
  | Own tests | 466 passed, 7 skipped (36.79s) | 472 passed, 1 skipped (37.63s) |
  | Source doctests | 4 passed (0.78s) | 4 passed (0.85s) |
  | Independent through A5 | 1256 passed, 26 skipped (24.81s) | 1261 passed, 21 skipped (20.65s) |

  Existing dtype-specific skips remain; no test was skipped or xfailed to
  satisfy this review. No source/test changes followed the full gate.

## Remaining work

No narrative docs build was run: step 2 still needs its guide/API refresh and
build. No new spec ambiguity was identified; Q-1 was settled by the maintainer
in the review. Leave deviation closure to the maintainer and stop after this
fix-up commit.
