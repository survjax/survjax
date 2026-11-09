# AFT.1 — Registered wrapper and direct methods

Independent review: round 3/3

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


Round 1 (b969eab): VERDICT fix-first, F-1/F-2 should-fix; N-1/N-2/N-3 nits.
F-1: detect underflow/overflow of positive interior rescaled time and return
NaN as numerical failure (7.1), rather than borrowing baseline boundaries.
This is detection, not a claim of full log-time extreme-tail evaluation.
F-2: add zero/negative/infinity survival parameter Jacobian/Hessian tests and
helper-level zero/infinity predictor Jacobians so baseline masking cannot hide
unsafe rescaling. N-1 comment explains NaN idiom; N-2 cites value policy.
N-3 left unchanged: direct values for invalid infinite predictors follow
formulas, not an inferred validity mask, per 4.5. No new invalid-value policy.
The review's approximate F-1 expected number is incorrect: for k=.5,b=t=1,
log f = log(.5)-.5*eta-exp(-.5*eta), about -400.693 at eta=800.
Its diagnosis that the current +inf is incorrect remains valid.


Fix-up validation: full touched tests/unit/test_aft.py in both dtype modes,
-n 6: 44 passed x64 (5.21s), 44 passed x32 (5.60s). New boundary derivative
cases exceed 1s, so the required timing script is rerunning.
scripts/check.sh --fast --through A10 passed after the fixes: implementation
x64 1040 passed/8 skipped (21.47s), x32 1045 passed/3 skipped (15.81s);
source doctests 4 passed per mode; spec x64 1993 passed/27 skipped (28.19s),
x32 2007 passed/13 skipped (33.23s). All formatting/lint/mypy gates passed.
Direct scripts/build_docs.sh passed again, both strict sites with success
identity. Both rendered API entries contain the numerical-failure wording.

Timing regeneration finished: 793 slow tests written, including all three new
boundary derivative cases. Generated-list diff is 269 lines in this fix-up;
code/tests are 57 changed lines. During timing I inspected completed logs:
implementation x64 1356 passed/8 skipped (311.03s), x32 1361 passed/3 skipped
(344.87s), spec x64 2333 passed/27 skipped (247.16s). The script discarded
its temporary logs before I captured the final full x32 spec summary, so no
full x32 spec count or Tier 2 gate certification is claimed here.


Round 2 (e01c42d): VERDICT maintainer. F-1/F-2 addressed on reading,
N-1/N-2 fixed, N-3 reasoning accepted; no new findings. Reviewer could not
execute checkout, tests or mutants because of headless permission restrictions.
Maintainer explicitly requested a retry on 2026-11-09. Round 3 retries execution
verification of the same implementation; only this review record changed.


Round 3 (1844b31): VERDICT maintainer. Gate and AFT tests passed; F-1
verified fixed by mutation tests. F-2 remained a deferrable coverage question:
removing the pre-log boundary substitution still passed all 44 AFT tests.
Maintainer decision (2026-11-09): "remove the defensive boundary as no
 demonstrated benefit". Remove just those time/predictor substitutions;
preserve the other numerical guards and all test assertions. Correct the
helper test explanation so it describes the observable result rather than
claiming to prove the removed substitution. This is an AFT.1 follow-up from
1844b31; original step start 7b5b70f, previous work/base cc0e30f.
All three review rounds have been used; do not start an unauthorized fourth.

Follow-up verification (2026-11-09):
- Removed only pre-log time/predictor substitution in _rescale_time. Restored
  boundaries, guarded exp and underflow/overflow failure detection remain.
- All 44 AFT tests pass in each dtype: x64 5.88s, x32 5.19s, using
  JAX_ENABLE_X64=1/0 uv run --group test pytest -n 6 tests/unit/test_aft.py.
  Every test assertion retained. The helper test explanation no longer claims
  it establishes the removed boundary substitution's necessity.
- scripts/check.sh --fast --through A10: all gates passed. Implementation
  x64 945 passed/8 skipped (6.91s), x32 950 passed/3 skipped (6.70s);
  source doctests 4 passed per mode; spec x64 1951 passed/27 skipped (8.12s),
  x32 1965 passed/13 skipped (8.11s).
- Ruff format: 194 files already formatted; Ruff check: All checks passed;
  mypy: Success: no issues found in 22 source files; git diff --check passed.
- No new tests, docstrings, docs or timing-list edits; no timing regeneration
  or docs rebuild needed for this deletion and test-comment correction.
- No fourth independent review: all three allowed rounds are used. Last
  reviewer verdict remains maintainer; this commit executes that maintainer's
  explicit F-2 decision. Remaining sub-steps: AFT.2 and AFT.3.
