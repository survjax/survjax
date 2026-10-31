# PH.3a — Built-in PH closure and delegated RMST

Independent review: round 1/3

Base: 2e6b55a9610e6733bf539d420d7a2f6d52ee355e, clean release-ph after
completing the maintainer-started RMST merge. That prerequisite merge preserves
both parent plans/reviews and the original imported RMST code/tests. It was
requested by the maintainer before “get on with the PH implementation”.
This implementation step is separate: 62 production additions + 238 test
lines = exactly 300 implementation lines. Plan/review/deviation/spec-seen
records are excluded under the updated agreement.

## Read first

1. PLAN.md PH.3a; spec 4.4, 4.5, 7.2–7.3, 8.7 and 12.1–12.2.
2. distributions/base.py _ph_equivalent default, then built-in overrides.
3. distributions/wrappers.py rmst: original diagnostics, promotion, closure,
   failed-zero substitution, delegation and final NaN selection.
4. tests/unit/test_proportional_hazards_rmst.py: independent integral/gradient,
   broadcast identity, zero boundary, native shape restriction, invalid-row
   isolation, then custom-baseline unsupported behavior.
5. DEVIATIONS.md D-031 and the unchanged PH.2c independent report for the
   separate, explicitly deferred extreme direct-method/likelihood derivatives.

## Spec decision and remaining scope

The designer completed revision 40 while this implementation was running:
765c65bba9efb3af8ef4a13c1ecb8e302a47f346. Read its full SPECIFICATION.md diff
from 1389af4 and marked this worktree with
`../scripts/spec-since.sh dev-release-ph --mark`. No design file was edited.
D-031 accurately records PH.2c F-1/F-2/S-1, the maintainer's deferral decision
and alignment; its status remains open/maintainer-owned. The original
“rebuilt-vector limitation” wording was not a numerical fix: actual finite
natural-parameter derivatives fail. The old review is preserved in Git.

Revision 40's exception covers extreme PH direct/likelihood derivatives;
staged RMST is unchanged. This step does NOT claim an RMST derivative
exception, full range completion, closure values beyond representability,
or a completed PH release gate. Broader PH RMST range/chain-rule evidence,
supported-Hessian regression, remaining integration evidence, exports and
executable docs stay separate (PH.3b/export/docs). No existing test or
assertion was weakened. The new draft was split before committing to meet
the code/test budget. API-reference/exact-export tests remain unchanged;
public export integration still needs its narrow docs/api.md exception.

## Invariants and numerical reasoning

The hook is private, not part of the extension contract. Base returns None;
only the two supported built-ins supply closures. Unsupported baselines,
including component-valued custom families, never enter numerical family code
or a generic RMST path. No new module/config/result type is introduced.

RMST alone uses the closure. All direct methods, grids and likelihood keep
the already reviewed PH formulas. Equivalent natural parameters follow
r_PH=exp(log(r)+eta), k_PH=k, b_PH=exp(log(b)-eta/k). Adding logs before exp
avoids premature overflow of exp(eta) when the final parameter is finite.
Original baseline/eta leaves are copied/promoted with the cutoff dtype before
closure arithmetic; original stored leaves and diagnostics are unchanged.

The original wrapper issue union and invalid cutoffs are evaluated FIRST.
The evaluated mask shape E includes explicit leading cutoff axes. Built-in
scalar fields broadcast from batch shape B to E, with neutral 1 for invalid
rate/shape/scale and 0 for invalid eta. Those substitutions precede log,
eta/k and exp, including on rows discarded later; original NaN/zero/negative/
infinite parameters never reach unsafe active arithmetic. No real valid
parameter tangent is stopped. Cutoff violations receive safe zero, and final
original invalid rows are NaN even if their neutral closure is finite. Shared
parameter derivatives of valid entries stay isolated from invalid neighbors.

Overflowing log-rate/log-scale inputs are replaced by zero BEFORE exp, then
infinity is restored. This is not clipping to a finite cap. A closure with
unrepresentable natural parameters can fail for positive cutoffs (NaN from
its staged family); no inaccurate finite value is invented. At zero, section
8.7's exact value/first derivatives still apply: if a closure parameter is
invalid due to representability, replace that closure's leaves by neutral
ones BEFORE delegating. Native RMST at active tau=0 then returns 0 with
parameter tangents 0 and cutoff tangent 1, without an intermediate NaN.
Ordinary representable closures are not substituted at zero, preserving their
native boundary behavior. No higher-order extreme boundary claim is made.

## Expected values and tolerances

Independent 60-digit mpmath quadrature integrates the physical PH survival:
Exponential exp(-exp(eta)*r*t), Weibull exp(-exp(eta)*(t/b)**k).
mp.diff differentiates that independent integral in every baseline field,
eta and cutoff at actual dtype-rounded points; no library method provides
those expected values or gradients. Two interior parameter vectors per family
check dynamic values. The broadcast/grid reference uses the spec's analytical
closure family, whose RMST primitive has its own independent quadrature suite.

Existing ordinary RTOL (zero atol on values), and existing RMST log-parameter
and absolute-cutoff derivative factors are unchanged. Multiply natural
parameter errors by r/k/b, divide by M; eta is already a log ratio (unit 1).
For the selected Weibull k=1.4, the eta chain factor 1/k is below one, so the
existing measured native derivative bound is conservative for these points.
The scalar eager derivative cross-check uses ordinary ATOL/RTOL on O(1)
entries. Every allclose uses equal_nan=False; invalid outputs are checked as
NaN explicitly and their own tangents are not asserted.

## Requirement-to-test map (19 cases)

- test_rmst_values_and_first_derivatives_match_independent_integrals (2):
  spec 4.4/8.7/12.1 values and every first parameter/cutoff derivative,
  eager AD, vmap/jit(vmap), both modes; valid values under debug_nans.
- test_rmst_broadcast_closure_identity_and_explicit_cutoff_grid (2):
  spec 4.4/5/8.7 baseline (3,1) times eta (3,2), scalar and explicit (3,1,1)
  cutoff grids (m=S), eager/jit/vmap, output (3,3,2). Baseline float32 with
  eta/cutoff float64 in x64 mode additionally tests promotion before log.
- test_rmst_zero_value_and_first_derivatives (6): spec 8.7 value 0 and
  Jacobian [0,...,0,1], eager/jit values and both compiled first AD modes,
  debug_nans; eta=0.2 and ±2000 (unrepresentable closure in both dtypes).
- test_weibull_shape_hessian_remains_explicitly_unsupported (1): spec 8.7
  four second-order forward/reverse compositions still raise the native
  NotImplementedError in the gamma interior branch.
- test_invalid_rows_preserve_shared_parameter_derivatives (7): original
  Exponential rate, Weibull shape/scale, eta and cutoff violations, including
  0/-1/NaN/±inf for positive parameters; NaN/±inf for eta; negative/NaN/±inf
  cutoffs. Only valid output differentiated; shared healthy eta or baseline
  survives invalid neighbors, eager/jit values and compiled fwd/rev first AD.
- test_custom_baseline_rmst_has_no_generic_fallback (1): spec 8.7 unsupported
  type gives NotImplementedError, eager/jit, for 0,1,inf,NaN cutoffs. Custom
  component axes show that the default hook never numerically masks leaves.

## Validation and development results

Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1 (same environment).

The first formatted draft was 337 code/test lines; its 15 focused cases
(including independent supported Hessians but without the broadcast test)
passed both modes (17.87s/17.60s). That full supported-Hessian test is deferred
to PH.3b; it was not removed to address a failure. Broadcast/grid checks fit
again after splitting that larger Hessian test and remain in this commit.

A direct probe then found Exponential PH rmst(0) was NaN at eta=±1000 due to
an unrepresentable natural rate. Added guarded closure exp and a failed-zero
fallback. The first fallback selected zero AFTER calling invalid closure RMST;
debug_nans caught that intermediate NaN (4 passed, 1 failed per mode, 8.94s /
8.41s). Corrected by neutralizing failed-zero closure leaves BEFORE delegation.
The 19-case suite then passed both modes (14.80s/14.92s). Promoted copied
leaves before log/exp and added mixed-dtype coverage; final numerical suite:

- JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_proportional_hazards_rmst.py -x:
  **19 passed in 15.95s**.
- Separate JAX_ENABLE_X64=0, same command: **19 passed in 16.01s**.
- Initial format check flagged the long new closure-call line; split through
  an explicit eta local. Final `uv run ruff format --check .`: 133 formatted.
- `uv run ruff check .`: all checks passed.
- `uv run mypy src`: no issues found in 22 source files.
- `git diff --check`: clean.
- `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed**.
  Own suites: **1261 passed, 8 skipped** x64 (104.01s), **1266 passed,
  3 skipped** float32 (134.73s). Source doctests: **4 passed** each mode
  (1.59s/1.23s). Independent acceptance: **2332 passed, 28 skipped** x64
  (98.90s), **2322 passed, 38 skipped** float32 (82.75s).
- Wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.9iPs7x/survjax-0.0.1.dev0-py3-none-any.whl`;
  SHA256 `3f66704616570199d9ce430a02aa896afcaf3b3ecebb2cf22c795d8d8d1a838f`.
  Full gate log: `/private/tmp/ph3a-wheel-gate.log`.
- Final scope audit: four existing source files plus one 238-line test module;
  source changes are closure hooks / rmst only. Existing direct/likelihood
  expressions, RMST numerical helpers/tolerances, exports, API-reference and
  protected acceptance files are unchanged. .spec-seen and D-031 are records.

No protected paths, dependencies, exports, tolerance constants, ordinary docs,
PDF source/artifact or RMST numerical expressions changed. Base/built-ins are
edited only for the PH-owned private closure hooks after the core handoff.
The numerical scope is 300 additions, with a staged release marker and process
records separately. Commit this step, run independent PH.3a review, and stop.
