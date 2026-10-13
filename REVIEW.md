# A5.4 — Weibull review handoff

Implemented on the maintainer-selected A5.4 branch from cd3f213, after approval
of PLAN.md. Specification revision 23; independent numerical tests remain
revision 22. Code/tests/docstrings: 145 family lines, 150 own-test lines and
5 export/import-test additions, **300 changed lines**. Planning, deviation and
review records are additional bookkeeping.

## Read first, in order

1. DEVIATIONS.md, D-020: one independent density expectation appears to omit
   log(shape). Acceptance is incomplete; the implementation retains the density
   derived from the specified survival function. The independent author must
   review/correct the expectation; no test was edited or excluded.
2. src/survjax/distributions/weibull.py: natural shape/scale leaves, common-dtype
   from_log, scale reference, log-space formulas, safe zero/infinity operands
   and analytic hazard limits. The class docstring includes all 4.6 mappings
   and a short survreg conversion example.
3. tests/unit/test_weibull.py: mixed-boundary cases also establish the full
   interior shape/scale Hessian, including cross derivatives, for k<1, k=1
   and k>1. All expected values come from stated algebra/identities.
4. Both initializers and tests/unit/test_import.py: exactly one new public
   export, Weibull. PLAN.md records scope, accounting and deferred docs.

## Invariants

- Registration supplies immutable PyTree construction, keyword requirements,
  batch indexing and one common floating dtype; no value validation is added.
  from_log normalizes before exp and stores only natural leaves, even on
  overflow. Reference time is scale, exactly batch_shape and parameter dtype.
- Interior H=exp(k*(log(t)-log(b))) avoids premature division underflow or
  overflow. Log hazard is analytic, without subtracting log density and log
  survival. No floor, epsilon, clipping, dependency, log storage or AD rule.
- Zero/infinite times substitute constants before log and multiplication.
  The scale operand is substituted too, keeping extreme-scale inactive
  derivatives out of the interior formula. No actual parameter/result has
  stop_gradient applied. Finite shape-one boundary values retain -log(b).
- Explicit boundary limits preserve NaN parameters via formula-undefined
  checks; there is no blanket parameter validity mask. Invalid-parameter
  derivatives are not guaranteed. Negative/NaN times and density at infinity
  use the existing base policy; density evaluation at infinity is substituted
  by that wrapper before the raw density is called.
- Public overrides share private numerical helpers, avoiding nested policy
  calls. Inherited survival/log-CDF/hazard honor the primitive and analytic
  log-hazard. Base/registration modules are unchanged.
- Mixed zero/finite/infinity cases differentiate only finite components.
  No guarantee is asserted for nonfinite outputs, discontinuous k=1 density/
  hazard shape limits, or finite-time precision loss.

## Spec-to-test evidence

- 3.3, 4.1–4.2: own test_log_inputs_normalize_before_exp_and_obey_chain_rule,
  test_nonreal_inputs_are_rejected_before_exp and
  test_extreme_ratios_and_reference_scales; independent construction/from_log
  cover exports, required keywords, two natural leaves, batch broadcasting/
  indexing and traced construction.
- 4.1, 4.5, 5, 12.1: independent closed-forms/boundaries cover all seven methods,
  shape-one Exponential identity, small H=1e-20, lower/upper tails, paired
  batches, policy rows and transformed boundary values. Own
  test_nan_parameters_are_isolated_and_raw_methods_broadcast adds NaN scale
  isolation and raw scalar/multiaxis broadcasting.
- 4.5, 7.2–7.3: own test_constant_boundary_parameter_derivatives covers
  constant finite limits in both parameters at extreme scales;
  test_shape_one_boundary_hazard_scale_derivative pins -1/b**2 at zero/inf.
  Independent boundaries pin -1/b for shape-one log density/log hazard.
- 7.2–7.3, 12.1: own test_mixed_boundary_full_parameter_derivatives checks
  shared/per-entry Jacobians and full Hessians for log survival, H, log density,
  log hazard and hazard, composed with jit/vmap in both differentiation orders.
  Independent derivatives supply x64 order-two forward/reverse check_grads.
- 4.6: inspect the class docstring for every required convention; narrative
  executable conversions remain an A5 docs task.

## Checks actually run

Focused own tests used separate processes:

    JAX_ENABLE_X64=1 .venv/bin/python -B -m pytest tests/unit/test_weibull.py -q -p no:cacheprovider

Repeated with mode 0: **18 passed** each (x64 9.15s; x32 9.47s), no skips.
An earlier version gave 17 passed each (4.04s/4.15s); the final suite consolidates
overlapping Hessian cases and expands mixed-boundary checks to three shapes.

The gate command was:

    UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4

Exit 0: **guard: ok (A5.4)**; Ruff **44 files already formatted**, **All
checks passed!**; mypy **no issues in 11 source files**. Own x64:
**369 passed, 7 skipped** (26.74s); x32: **375 passed, 1 skipped** (25.47s).
Source doctests: **1 passed** each (0.51s/0.53s). Independent through-A4:
x64 **679 passed, 25 skipped, 370 deselected** (17.59s); x32 **693 passed,
11 skipped, 370 deselected** (14.73s). Final output:
**check: all gates passed through A4**. Existing dtype skips remain unchanged.

All five non-grid A5 modules were run without a keyword filter in each mode:

    JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py tests/spec/test_a5_derivatives.py -q -p no:cacheprovider

Each exited 1 solely on test_weibull_policy_applies_elementwise_in_one_call:
x64 **1 failed, 321 passed** (9.33s); x32 **1 failed, 311 passed, 10 skipped**
(5.81s). The x32 skips are one float64-promotion case and nine x64-only
finite-difference cases. The implementation gives -0.9431471805599453;
the test expects -1.6362943611198906, a discrepancy of log(2), explained in
D-020. These runs are not a passing A5 acceptance gate. git diff --check
passed with no output.

## Uncertainty, deferred work and stop

D-020 awaits maintainer/independent-author review; other deviations stay open
for maintainer closure. No spec, independent test, alignment marker, tolerance,
base wrapper, dependency or docs page changed. No docs build, mutation check,
R fixture, wheel check, likelihood debug-NaN run or benchmark was performed;
those are not evidence for this step. No runtime-cost claim is made.

The 48 grid cases remain outside this selection and belong to A5.5. Narrative
docs/builds follow the last A5 code merge as planned. Commit this implementation
and stop for review; do not begin grids, docs or a later step.
