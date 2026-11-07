# AFT.1 — Registered wrapper and direct methods

Independent review: round 1/3

Read first: PLAN.md AFT.1; spec 4.4–4.5 and 7.1–7.3;
wrappers.py AcceleratedFailureTime; exports; tests/unit/test_aft.py;
then the import and installed-wheel export-list updates.
Start: 7b5b70fa1f743d2656699162fc591878f75dc1a0. Previous completed work
and original worktree base: cc0e30f. No spec/protected/dependency changes. Maintainer explicitly authorized the
single API autodoc entry during this code step and the 310-line generated
slow-test list diff in this one commit (in addition to ~300 implementation
and test lines).

Invariants:
- Existing registration broadcasts nested leaves without normalizing their
  dtypes and prefixes baseline parameter diagnostics; eta declares real.
- Rescale positive finite times through exp(log(t)-eta), guarding overflow
  before exponentiation. Zero and infinity reach baseline boundary overrides
  without 0*inf. Base registration supplies the outer time-value policy.
- Direct density/hazard use baseline overrides minus eta, rather than
  differentiating the AFT survival. Infinity survival keeps the baseline hook.
- Reference times stay finite on the baseline scale instead of forming
  exp(eta)*reference; the wider baseline/predictor dtype is retained.
- No RMST override or summary API work; inherited unsupported behavior remains.

Tests and requirements:
- Construction/pytree/indexing/dtypes and misuse: 3.3, 4.1–4.4.
- Nested diagnostics eager/jit/vmap: 4.7.
- Weibull equivalent scale b*exp(eta), three shapes/all seven methods,
  eager/jit grids/vmap including negative/zero/infinity/NaN times: 4.4–4.5, 5.
- Exponential equivalent rate r*exp(-eta): 4.4.
- Smooth interior direct values, both Jacobians and Hessian match an explicit
  algebraic log-density identity; order-2 check_grads: 7.3.
- Invalid likelihood row leaves valid gradient/Hessian intact: 7.1–7.3.
- Explicit unsupported RMST retained; import/install lists include the export.

Validation so far:
- Tier 0 tests/unit/test_aft.py: x64 34 passed (19.23s), x32 34 passed (11.37s).
- Expanded Tier 0 tests/unit/test_aft.py tests/unit/test_import.py
  tests/install/test_wheel.py, -n 6, both dtype modes: x64 41 passed (49.46s),
  x32 41 passed (43.93s). Installed wheel tests use isolated environments.
- uv run ruff format --check .: 194 files already formatted;
  uv run ruff check .: All checks passed;
  uv run mypy src: Success: no issues found in 22 source files.
- Initial scripts/check.sh --fast --through A10: failed implementation x64
  (3 failed, 1103 passed, 8 skipped). Two public-export lists subsequently
  updated and verified above. Third failure: API reference missing AFT entry;
  maintainer authorized the single autodoc entry, now added. Explanatory docs
  remain AFT.3. No assertions weakened or tests manually skipped.
- Final scripts/check.sh --fast --through A10: all gates passed. Implementation
  x64 1106 passed/8 skipped (67.34s), x32 1111 passed/3 skipped (52.79s).
  Source doctests 4 passed per dtype. Spec x64 2149 passed/27 skipped (25.12s),
  x32 2156 passed/20 skipped (25.79s). Used the original slow list, so all new
  AFT tests ran. No numerical edits followed this gate.
- scripts/update_slow_tests.sh --through A10 completed: 656 slow tests written.
  This ran concurrently with validation; scheduling affects timing metadata.
  It measures durations and does not certify suite success (its script masks
  pytest exit statuses). Only the focused tests and gate above certify results.
- Direct scripts/build_docs.sh: build succeeded for both modes, verified both
  dtypes with success identity. Parsed both rendered API pages to confirm the
  keyword-only AFT signature, sign convention and four method entries.
- git diff --check passed. Tier 2 full gate not run; maintainer/CI tier.

Remaining verification is AFT.2: extreme rescaling/tails, custom improper
baseline, broader integration, likelihood identities and mixed derivatives.
AFT.3 covers explanatory documentation. Current tests do not establish these
broader claims; no PH derivative exception is claimed for AFT.
