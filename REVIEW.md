Independent review: round 3/3

# RMST.3 — Full-range Weibull RMST derivative and isolation evidence

Read PLAN.md RMST.3, spec 8.7/12.1 revision 38, then
appendices/weibull-rmst.md; tests/unit/test_weibull_rmst_derivatives.py,
tests/impl_tolerances.py and the expanded zero test in test_weibull_rmst.py.
At ab7afb1 only tests/tolerances changed (about 220 numerical lines).
The follow-up below also guards an inactive exponential after a gamma
precision failure. No public API, wrapper, export or dependency changed.

Invariants/reference origins:
- 142 cases: six shapes × seven cutoff ratios × three scales (126), plus
  16 points straddling both switches. Actual dtype-rounded inputs enter
  independent mpmath quadrature at 60 decimal digits. The reference integrates
  survival and its shape/scale derivative kernels directly, without gamma
  functions. The scale kernel avoids cancellation in m-r*S. Cutoff reference
  is exp(-(tau/b)**k), by the fundamental theorem of calculus.
- Natural Jacobians are compared in log-parameter units k*dM/dk and b*dM/db
  with absolute errors scaled by independent |M|; cutoff uses absolute scale
  1, all with rtol=0. New factors were fixed from pre-test full-range probes,
  not loosened after failures. Existing tolerances are unchanged. Max measured
  parameter/cutoff errors: 1.72e-14/1.56e-15 x64; 1.29e-5/6.45e-7 x32.
- Supported Hessians hold shape outside the differentiated argument. Packing
  a constant shape and differentiable scale into one array can materialize a
  zero shape tangent and make JAX invoke unsupported igamma_grad_a AD. The
  scale-only check therefore uses separate k,b arguments with argnums=1;
  no stop_gradient, custom rule or new higher shape guarantee is introduced.
- Own derivatives of NaN results are unclaimed, explicitly approved by the
  maintainer. Valid entries must remain isolated through shared parameters.
  Weibull zero tests assert only value and first derivatives.

Spec evidence map (test names in the new file unless noted):
- first_derivatives_full_range: forward/reverse, batched eager, jit(vmap),
  vmap(jit), representative scalar eager; lower/upper tails, scale changes,
  k=1 and switch neighbors against independent quadrature (8.7/12.1).
- cutoff_derivative_equals_survival: dM/dtau=S(tau) over the same range.
- supported_hessians_are_finite: both modes across all 142 cases under
  debug_nans. supported_hessian_matches_calculus: independent analytical
  bb/btau/tautau entries, series and gamma cases, all four AD compositions;
  zero absolute tolerance prevents tiny expected Hessians accepting zero.
- shape_hessian_raises_in_both_modes: all four AD compositions, gamma middle.
- shared_parameters_isolate_valid_entries: valid rows beside zero, negative,
  NaN and infinite parameter issues or invalid cutoffs; both first modes and
  supported shared-scale Hessians equal the isolated valid call (7.2/8.7).
- test_weibull_rmst_zero_derivatives (existing file): 12 shape/scale pairs,
  eager/jit/vmap first derivatives under debug_nans are exactly (0,0,1).
  Existing tests retain all staged surface, broadcasting and value policies.

Validation so far:
- JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_rmst.py
  tests/unit/test_weibull_rmst.py tests/unit/test_weibull_rmst_derivatives.py -q:
  31 passed in 65.23s. Identical command in a separate X64=0 process:
  31 passed in 65.91s. Ruff passed; mypy: no issues in 21 source files.
- Initial test-authoring failures (four per mode) were corrected: NumPy 2.5
  assert_allclose cannot format an array-valued atol, so scalar normalized
  comparisons replace it; the scale-Hessian fixture must avoid packed shape
  tangents as described above. No production changes or tolerance changes.
- scripts/check_wheel.sh --through A10 exited 0: all gates passed through
  A10. Own tests: 1079 passed / 7 skipped x64; 1085 passed / 1 skipped x32.
  Source doctests: 4 passed per mode. Independent acceptance: 2332 passed /
  28 skipped x64; 2322 passed / 38 skipped x32. Fresh wheel retained at
  /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.Jp77jX/
  survjax-0.0.1.dev0-py3-none-any.whl. Final static checks: 117 files already
  formatted, Ruff all checks passed, mypy clean (21 source files),
  git diff --check clean.
- Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1,
  pytest 9.1.1; CPU only. No new dependency/version claims or R regeneration.

Appendix review round 2 at dc48b24 returned VERDICT: merge: F-1/F-2/N-1/N-2
fixed, S-1a withdrawn, S-1b answered by maintainer; S-1c is integration only.
The maintainer still owns D-028 status and integration of design revision 38
(e4c1d50). This worktree does not merge either repository.

Handoff after this core step is reviewed and integrated: PH owns wrappers.py
and exports, and may then edit the built-in private closure hooks. PH must
preserve the zero/invalid policies and native shape-derivative restriction.
Its separate step supplies PH RMST value/derivative/broadcasting/unsupported-
family evidence. Executable RMST user docs and the final combined release gate
remain later work; this step makes no PH or complete-release claim.

## Implementer-found precision-failure isolation regression

After ab7afb1 was committed and round 1 started, a targeted additional probe
found that valid entries can receive NaN reverse derivatives from a failed
NaN row outside the value-accuracy range. In x32 use k=[2,0.01], b=1.25
shared, tau=[0.75,0.01]. In x64 use k=[2,0.005]. The valid row has a finite
value/forward derivative, but its reverse scale derivative is NaN. Both
pre-fix failures were reproduced against the committed helper. The failed
row's gamma P underflows while Gamma(1+1/k) overflows; substituting P=1
before log is insufficient because exp(log_value) then overflows in the
inactive expression and contributes 0*inf in reverse AD.

The fix substitutes a finite log operand before exponentiation when P is
lost. It preserves the failed result as NaN and does not floor probabilities
or alter valid active calculations. New failed_gamma_entry_isolates_shared_scale
covers first and second scale derivatives in both modes/dtypes. This enforces
the retained valid-entry isolation requirement; it adds no guarantee about
the failed result's own tangent. Targeted first/second-order tests passed in both separate dtype processes:
1 passed / 10 deselected, 11.01s x64 and 10.62s x32. The final unchanged
scripts/check_wheel.sh --through A10 exited 0: own 1080 passed / 7 skipped
x64 and 1086 passed / 1 skipped x32; doctests four per mode; independent
2332 passed / 28 skipped x64 and 2322 passed / 38 skipped x32. Ruff format
117 files, Ruff check and mypy (21 source files) passed. Final wheel is under
/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.AxC1Lt/.
Numerical source/test total remains below about 300 changed lines. Round 1
is reviewing ab7afb1; request round 2 on this meaningful fix-up only after
its report is written. This is not a rerun of an unchanged snapshot.


## Round 1 outcome and round 2 request

Round 1 (ab7afb1) returned VERDICT: merge. Reviewer reproduced the measured
error maxima; all derivative-breaking mutants were caught in both dtypes.
S-1 is recorded as D-029 and the maintainer was asked to confirm the
log-parameter normalization and clarify spec 12.1. No spec action pending
that answer. The new failed-entry guard is independent of normalization.
N-1: the old Weibull masked-invalid-cutoff test now asserts only valid cutoff
positions. Applied the same correction to its Exponential counterpart. This
implements the maintainer's already confirmed NaN-own-tangent decision,
without removing valid-entry isolation coverage. Final focused checks for
both cutoff-isolation tests and the failed-gamma regression passed in separate
processes: 3 passed / 29 deselected per dtype (3.83s x64, 3.88s x32).
Ruff and mypy passed; git diff --check passed. The fresh-wheel cumulative
gate above covers the final production code/regression before narrowing these
two old test assertions. No production change followed that gate.

Root's final fresh-wheel gate after the source fix passes both modes (above).
Round 1 reviewer's own gate had only the five known missing-wheel setup
errors; its separately run x32 spec tests (excluding wheel setup) passed
2317/38. Mutation worktrees requested for cleanup by the reviewer are
maintainer-owned; no deletion or edit outside the allowed review paths.
Round 2 must review the actual source guard and its first/second-order
failed-entry isolation regression, not just the initial test-only snapshot.


## Round 2 outcome; approved contract clarification for round 3

Round 2 (33a763b) returned VERDICT: maintainer solely because D-029 awaited
a decision. It reports no code finding and says the code is ready to merge.
N-1 is fixed in both families. Independent probes verified failed-row
isolation in all first directions and nonshape Hessians in both dtypes.
Mutants reverting the exp guard or safe probability were caught by the new
regression in both modes. The role's cumulative check again had only the
known five missing-wheel setup errors; root's fresh-wheel gate above passes.

The maintainer has now explicitly answered “Confirm log-parameter scaling”
to the question naming k*dM/dk and b*dM/db at |RMST| scale. Spec revision 39
clarifies 12.1, also spelling out the equivalent natural-derivative scales
|M|/k, |M|/b and |M|/r for Exponential, with cutoff scale 1 unchanged.
Design commit 1389af4da8cd23f7c0d38da047ba14b5ac0801f2 is separate on
release-rmst; .spec-seen marks that exact revision in this checkout.
D-029 now quotes the approval, records the 5x raw-k tradeoff at k=0.2, and
avoids overstating the reviewer's earlier endorsement. Status stays open
for the maintainer, as does D-028. Neither spec revision 38 nor 39 is merged.

This round changes contract/decision records and .spec-seen only. No source,
reference, tolerance or test changes follow the passing gate and final focused
checks above. Static and whitespace checks validate the final snapshot;
repeat numerical regression is not needed for a wording-only clarification.
All RMST implementation findings/questions are addressed; no new step is
started. Round 3 is the last permitted review for RMST.3, then stop/report.
