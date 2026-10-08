# A4.5 — Autodiff density and hazard defaults

Implemented on `A4.5` from `9d1d08d`, against specification revision 21
(2026-10-08). The maintainer authorized the completion commit on 2026-10-08;
stop here for maintainer review after committing. The maintainer
approved keeping this roughly 375-line source/test/doc change together.
No A5 implementation, dependency changes, protected-file edits or new exports.

## Reading order

1. `src/survjax/distributions/_survival_methods.py`: the JVP helper,
   zero-rate logarithm guard, three defaults and positive-infinity density policy.
2. `src/survjax/distributions/base.py`: explicit default bindings and seven
   registration-time policy wrappers.
3. `tests/unit/test_family_density.py`: hand-derived formulas, broadcasting,
   compilation reuse, support ends, overrides and parameter derivatives.
4. `tests/unit/test_family_direct.py`: added mixed-time policy rows, safe-reference
   derivatives, dtype rejection and mixed-dtype promotion cases.
5. `docs/distributions.md` and `docs/index.md`: executed examples and limitations.

## Invariants and numerical decisions

- Pointwise output shapes are the ordinary broadcast of time and batch shapes.
  The all-ones JVP tangent obtains elementwise time derivatives from a pointwise
  primitive without a full Jacobian or flattened batch/component axes.
- Defaults differentiate the raw primitive after outer time substitution.
  Integer inputs are converted to the promoted floating calculation dtype before
  JVP when necessary; floating inputs retain JAX promotion and weak typing.
- Generic density is `log(-d log S/dt) + log S`, independently of log-hazard
  overrides. Generic log hazard uses the derivative directly, avoiding large-H
  subtraction. Hazard calls the log-hazard method, including family overrides.
- A zero derivative has a safe logarithm operand before selecting minus infinity.
  Negative and NaN rates retain NaN; no floors, clipping, epsilons or inferred
  parameter validity domain. Exactly minus-infinite or NaN log survival makes
  generic log hazard NaN; finite log survival remains eligible after linear
  survival underflow.
- Negative/NaN time substitutions stop only the substituted reference input.
  Active parameters/calculations retain their gradients. NaN family values retain
  precedence at negative/NaN times. Zero and positive infinity reach family code.
- The default substitutes safe operands before an inactive infinite-time density
  addition. The outer density policy also enforces minus infinity for overrides
  at positive infinity, even if their result was NaN. No claim is made about
  gradients of infinite densities or invalid inputs.
- Inheritance retains raw methods without stacking wrappers. Construction,
  PyTree reconstruction, batch indexing and static metadata are unchanged.

## Spec-to-test mapping

| Requirement | Own evidence | Independent evidence (read-only) |
| --- | --- | --- |
| 4.2: autodiff density/hazard identities | density suite: constant and power formulas, large-H test | `test_a4_log_prob_default.py`, `test_a4_generic_defaults.py` |
| 4.1/5: pointwise shapes and components | constant paired/outer broadcasting, integer times, shape TypeError | density default tests: shapes, components, nested families |
| 3.3/4.5/6.2: time dtype rejection/promotion | direct dtype misuse/tracer/promotion tests, including added methods | `test_a4_time_dtypes.py` |
| 4.5: negative/zero/infinite/NaN values | direct mixed-time test; power boundary and bounded-support tests | `test_a4_value_policy.py`, `test_a4_policy_enforcement.py`, `test_a4_nan_precedence.py` |
| 4.5: generic zero-survival behavior | bounded-support and underflow test | `test_a4_hazard_at_zero_survival.py` |
| 4.2/4.5: overrides/inheritance/purity | inherited override dispatch test; raw markers and leaf identity | override cases in density, hazard and policy suites |
| 7.2: safe inputs and zero-rate logs | direct Root safe-reference tests; density zero-hazard active-derivative test | negative-time parameter-gradient tests |
| 7.2: reuse compilation | two successful shape traces across changed rates | existing registration/static-configuration tests |
| 7.3/12.1: second derivatives and transformations | density gradient/Hessian cases with jit/vmap, order-two forward/reverse check_grads using unchanged justified tolerances | analytical gradients/Hessians and x64 finite-difference checks in density suite |
| 10.5: executable docs | both documentation builds and source doctests | not claimed as complete Slice A evidence |

Expected values come from the documented constant-hazard and power identities
or uniform survival `S(t)=1-t/end`. No expected values come from library outputs.
The vmap checks additionally compare transformation agreement, not mathematical
correctness. Tolerances in `tests/impl_tolerances.py` are unchanged.

## Exact verification results

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Environment:
Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, pytest 9.1.1. CPU; no versions changed.

`scripts/check.sh --through A4` exited 0:

- `guard: ok (A4.5)`.
- Ruff format: `38 files already formatted`; lint: `All checks passed!`.
- mypy: `Success: no issues found in 9 source files`.
- Own tests, x64: **282 passed, 6 skipped** (10.64s).
- Own tests, x32: **287 passed, 1 skipped** (10.48s).
- Independent tests through A4, x64: **676 passed, 25 skipped** (15.45s).
- Independent tests through A4, x32: **690 passed, 11 skipped** (13.05s).
- Final output: `check: all gates passed through A4`.

Combined `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A4 uv run pytest tests -q`:

- x64: **958 passed, 31 skipped** (24.03s), exit 0.
- x32: **977 passed, 12 skipped** (21.35s), exit 0.

Own skips are existing conversion-warning/promotion cases specific to one mode.
Independent skips are existing dtype-specific cases, optional pandas (not a
runtime dependency), and the two second-order finite-difference density checks
restricted to x64. No tests were edited, skipped or xfailed in `tests/spec`.

`JAX_ENABLE_X64=<mode> uv run python -m doctest -v
src/survjax/distributions/base.py
src/survjax/distributions/_survival_methods.py`, in each mode: **13 passed,
0 failed** in base.py; **0 examples** in _survival_methods.py; exit 0.

`scripts/build_docs.sh`: **build succeeded** in both x32 and x64, exit 0.
All three executable pages ran afresh, including the added density/hazard cell.
The kernel printed its existing TCP-without-encryption notice; Sphinx emitted
no build warnings. Rendered pages were not inspected visually in this step.

`git diff --check`: exit 0, no output.

Development checks before the final gate: Ruff initially found two lambda-style
violations and a loop-variable binding violation in the new tests; fixed them.
The first focused x64 density run had **3 failed, 12 passed**: the compilation
counter also counted deliberately failed shape traces. Moved shape misuse after
successful-trace assertions; no tolerance changed. A subsequent focused x32
run of density/direct tests had **76 passed**, before adding the last dtype cases.
The final full gate and combined sessions above include every final code/test edit.

## Limits, uncertainties and A5 handoff

- D-016 remains open and explicitly deferred to A5: generic survival/log-CDF
  values at positive infinity can have NaN parameter gradients. This step does
  not change those methods or claim the issue is fixed.
- Autodiff defaults retain the documented representable-linear-hazard limit;
  custom families still need correct primitive boundary values/derivatives.
  No parameter diagnostics, built-ins, shared grids or likelihoods were added.
- No new spec ambiguity was identified. Existing deviation statuses are untouched.
- R fixtures, wheel checks, benchmarks and the complete Slice A gate were not
  run; they are outside A4.5. No publication or A5 work was performed.
