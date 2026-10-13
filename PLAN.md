# A5 — Exponential, Weibull and time-first grids

Status: A5.1 (`3f50962`) was approved by the reviewer and merged into main by
the maintainer. A5.2 (`a58c57c`) received a merge verdict on 2026-10-12.
The F-2/F-3 review nits are corrected; A5.2b (`71403c3`) and its review fix-up
(`63874e8`) are merged into main. A5.3–A5.5 and A5 docs remain future steps.
The original planning started from a
clean `A5` checkout at `3f50962`, then the main tip. No branch was created
or switched by this planning request; implementation uses the maintainer's
selected step branch after approval. Based on
specification revision 22, design commit `9323cb5`, and the independent tester
commit `7f1d6df`. Implementation starts on the maintainer-selected branch `A5`
at `89cb9f4`, after the maintainer aligned the marker and synchronized tests.
Commit this approved planning/deviation record separately from the A5.1
implementation so each diff remains reviewable. The A4 plans below are history.

## What changed after the spec/test review

Read the complete revision-21-to-22 diff, the six new A5 test modules,
Q-027–Q-033 and their TESTMAP entries, plus the three new A4 NaN-precedence
cases. Revision 22 replaces the earlier log-leaf proposal: `from_log` simply
exponentiates inputs and stores the resulting natural parameters. The previous
requirement to retain extreme log inputs is removed. No log storage, private
representation flag or registration redesign is needed (D-019).

Other settled requirements:

- Built-ins are registered frozen subclasses of `SurvivalDistribution`, with
  natural `rate` or `shape`/`scale` as their only dynamic leaves (4.1–4.2).
- `reference_time()` returns an array of exactly `batch_shape`, in the parameter
  dtype, positive and finite for valid parameters; invalid cases are unspecified.
- `log_prob(+inf)` is NaN when the family's raw `log_survival(+inf)` is NaN;
  otherwise it is -inf, including when a density formula yields inf-inf.
  Testing the raw density for NaN alone cannot distinguish these cases (4.5).
- At zero and infinity, finite built-in values have the actual parameter
  derivatives of their limits. Derivatives across a discontinuity are excluded:
  do not require the Weibull shape derivative of log density at zero and k=1.
  The scale derivative there is required. Custom-family guarantees depend on
  the author's primitive; the extension recipe must explain this (D-016/4.5).
- The base supplies all three methods `survival_grid(t)`,
  `cumulative_hazard_grid(t)` and `hazard_grid(t)`, for every registered family.
  The first positional argument is named `t` and also accepts `t=...`; it must
  be one-dimensional and follows the direct-time dtype rules (5/8.1).
- The zero-time grid example is self-contained: Weibull shape 1, scale 10.
  The section 11.1 fixed vector additionally gives group scales
  `10 * exp([0, -0.5, -1])`; it requires no R fixture.

These decisions supersede the old pending choices; no new spec ambiguity was
identified in this review. Record revision-22 evidence in DEVIATIONS.md and
leave formal deviation closure to the maintainer.

## Scope, staging and independent-test availability

Public additions: `survjax.Exponential(rate=...)`,
`survjax.Weibull(shape=..., scale=...)`, keyword-only `from_log` classmethods,
all seven direct methods on both families, and the three grid base methods.
Covered spec: 3.3, 4.1–4.2, 4.5–4.6, 5, 7.2–7.3, 8.1 (grids only),
10.2–10.5 and applicable family/boundary requirements of 12.1–12.2.
No likelihoods, interval/truncation methods, parameter diagnostics (D-013),
summaries, sampling, wrappers, `check_family` or new dependencies. R fixtures,
installation checks and the CPU likelihood baseline remain later Slice A work.

There are five commit-sized steps below, separating the newly specified policy
correction from the generic gradient work. Aim for roughly 300 changed
source/test/docstring lines per code step, at most 300; split before expanding
if necessary. Each step requires its own plan approval, tests and public API
docstrings with the code, one commit and an
updated REVIEW.md, then a stop. This proposal does not authorize proceeding
through all five steps. Use the maintainer-selected branch; never commit on main.

From A5.2b onward, narrative docs follow specification revision 23 (decision
0038): code steps don't edit `docs/`; record their documentation needs under
"A5 docs" below. Docs steps start after the last A5 code step merges, and A6
starts only after the last A5 docs step merges. Earlier A5 docs are drafts;
the A5.1/A5.2 instructions and verification evidence below retain their history.

The maintainer aligned `.spec-seen` to `9323cb5` in `96fc32d` and synchronized
the independent suite from `7f1d6df` in `89cb9f4` before implementation. The
local through-A4 gate now includes the three revision-22 regressions, and
through-A5 includes all 370 A5 cases. Both independent suites and the alignment
marker remain untouched by the implementation agent. Never edit, skip or xfail
acceptance tests to obtain a pass.

## A5.1 — Correct NaN precedence for density at infinity

- Change `src/survjax/distributions/_survival_methods.py`'s density policy to
  consult the raw primitive at +inf. Return NaN iff that primitive value is NaN;
  otherwise preserve the unconditional -inf density limit. Leave negative/NaN
  time handling, normal broadcasting, override dispatch and non-density methods
  unchanged. No exports, signature changes or generic validity-domain inference.
- Add `tests/unit/test_family_infinity.py`: constant-hazard and power custom
  families with NaN parameters; mixed NaN/valid entries where the valid generic
  density formula is NaN at infinity; an explicit density override whose result
  differs from the primitive's NaN status. Exercise eager/jit/vmap. Expected
  constants follow the revision-22 policy, not the implementation.
- Preserve the existing one-reference-call invariant. Keep any extra primitive
  evaluation restricted to the density rule; do not double-wrap methods.
- Update the infinity rule in `docs/distributions.md` and relevant base-method
  docstrings. Cover 4.2/4.5/7.2. Run the three new acceptance tests in
  `test_a4_nan_precedence.py`, then the complete through-A4 regression gate
  once the suite is synced. Stop with REVIEW.md and the step commit.

## A5.2 — Generic finite boundary gradients and extension recipe (D-016)

Outcome: generic survival and log-CDF give their correct finite values and
parameter gradients at +inf for proper primitives such as `log S=-r*t`, while
preserving finite, parameter-dependent limits of custom cure families. Public
API, exports, constructors, dtypes and broadcasting are unchanged. Covered spec:
4.2, 4.5, 5, 7.2–7.3, 10.5 and relevant transformations in 12.1. No built-ins,
grids, density/hazard changes, dependency additions or custom derivative rules.

### Numerical change — `src/survjax/distributions/_survival_methods.py`

- Integrate the guard in the existing policy wrapper, where its already-computed
  reference input can be reused. This refines the earlier defaults-local sketch:
  calling `reference_time()` again inside each default would break the existing
  one-reference-call invariant. Keep the change in this module; registration
  and `base.py` need no change.
- After resolving the existing raw-method marker, recognize only the generic
  `survival` and `log_cdf` functions using function identity, a static Python
  decision. Explicit and inherited family overrides retain their dispatch.
  No new marker protocol, cache, parameter specialization or wrapper stack.
- Retain the single stopped reference input in a named local. Evaluate raw
  `log_survival` on the current safe times to obtain the true limit. The guard
  is elementwise: original time is positive infinity AND raw log survival is
  negative infinity. Use this result only as a Boolean selection condition.
- Before the default's numerical operations, replace those guarded times with
  the stopped reference input. Evaluate the default on these safe times, then
  select the exact finite boundary value 0 for survival or log-CDF. The raw
  primitive has still been called at the actual infinity to establish its limit.
  Stop only the substituted reference, never real parameters or active results.
- Leave finite raw limits at +inf active: if S(+inf)=p, gradients in p must
  survive. NaN raw limits must remain NaN. The guard must not activate at finite
  times, even when linear survival underflows or log survival overflows to -inf;
  no precision loss is reclassified as zero mass. Preserve zero-time behavior,
  A5.1's density rule and negative/NaN-time policy.
- A second primitive evaluation is expected for these generic methods, including
  finite calls. Record that cost without claiming a measured speedup or adding
  an optimization layer. Custom authors remain responsible for primitive
  differentiability and safe behavior at their chosen reference inputs.

### Own tests — `tests/unit/test_family_infinity.py`

Extend the existing 111-line module, keeping it below roughly 300 lines.
Expected values/derivatives come from algebra written in each test; run both
dtype modes with `tests/impl_tolerances.py`'s existing appropriate tolerances.
Add a scale only with a separate numerical justification in that file.

- Existing Constant and Quadratic primitives: survival(+inf)=0 and
  log-CDF(+inf)=0; first parameter derivatives are exactly 0 in forward and
  reverse modes, eagerly and composed with jit/vmap.
- Scalar rate with times [1,+inf]: sum survival=e**(-r), derivative=-e**(-r)
  and second derivative=e**(-r). Sum log-CDF=log(1-e**(-r)), derivative
  1/expm1(r) and second derivative=-e**r/expm1(r)**2. Cover per-entry parameter
  batches as well, so an inactive infinity entry cannot poison active entries.
- A guarded cure primitive S(t)=p+(1-p)*e**(-r*t), with 0<p<1 and r>0:
  at infinity survival=p and log-CDF=log(1-p). Their p derivatives are 1 and
  -1/(1-p), and their r derivatives are 0. Preserve these under jit/vmap and
  check a finite-time case; the primitive itself guards its infinity operands.
- NaN parameter entries beside valid entries at +inf retain NaNs only in their
  positions; finite-time underflow and active lower-tail values retain current
  behavior. Do not assert derivative guarantees for invalid parameters.
- Explicit/inherited survival and log-CDF overrides remain honored. Extend
  reference-call coverage to these methods at infinity, checking one call.
  Existing finite/negative-time, zero-CDF, primitive time-derivative, density
  and second-order interior tests remain regressions, not targets for alteration.
- Demonstrate before/after evidence with the new infinity-gradient cases.
  After the fix, temporarily remove only the safe-time substitution: the
  constant-family gradient cases must detect NaN despite final output masking.
  Restore the source before final checks; never mutate independent tests.

### Documentation — `docs/distributions.md`

Preserve the maintainer's tutorial. Add a short executable guarded-boundary
extension recipe and parameter-gradient assertions, explain the proper-family
zero limit and the parameter-dependent cure limit, and state the author's
responsibility for safe primitive operands/NaN propagation. Update method
docstrings with the same scope. Amend PiecewiseHazard's `reference_time()` to
return exactly `batch_shape` in `rates.dtype`, and revise the checklist's
current "broadcastable" promise to the revision-22 exact-shape contract.
No new page, notebook or later-slice feature is needed.

The separately requested Gompertz notation correction is applied during this
planning turn: display `-(b/c)*(e**(c*t)-1)` mathematically, with `jnp.expm1`
still explained in prose and used by executable code. It changes no computation.
Planning verification: `UV_CACHE_DIR=/private/tmp/survjax-uv-cache
scripts/build_docs.sh` exited 0, reporting `build succeeded` for x32 and x64
and executing the existing examples afresh. `git diff --check` passed with no
output. No numerical implementation tests were run during this planning turn.

### Validation, commit and stop

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run the focused own suite in
separate x64/x32 processes, then `scripts/check.sh --through A4`,
`scripts/build_docs.sh` (both modes), and `git diff --check`. The independent
author does not guarantee arbitrary custom-family boundary gradients; the new
own tests supply that evidence. All 370 A5 cases remain outside this step's
gate, since built-ins/grids are not yet implemented. Do not edit or skip them.

Aim for roughly 300 changed source/test/doc lines; split before expanding if
necessary. Overwrite REVIEW.md with reading order, safe-input/dispatch invariants,
spec-to-test mapping, mutation evidence, exact results/skips and the unmeasured
extra-evaluation cost. Leave D-016 open for maintainer closure. Commit the
approved step on the maintainer-selected branch, then stop before A5.3.

## A5.2b — Isolate finite-entry derivatives from infinity entries (review F-1)

### Review fix-up (2026-10-13)

Authorized by the maintainer's request to implement the A5.2b review comments.
Same step and branch; no new public API or numerical policy.

- `tests/unit/test_family_mixed_infinity.py`: add a density override to the
  existing cure family whose formula is -r*t. Pin [1,+inf] values [-2,-inf]
  at r=2, p=0.25, and the finite-component rate derivative -1 in forward and
  reverse modes, including jit/vmap. Spec 4.5/7.2; expected numbers are algebra.
- `_survival_methods.py`: consolidate each generic method's guard flag and
  boundary value in one static conditional chain; retain the hazard check
  against its underlying log-hazard and rename that local `log_hazard_method`.
  No registration-table redesign or new configuration layer is needed.
- F-3 remains a structural argument: `isposinf(time)` explicitly excludes
  finite times. Do not assert unsupported NaN-gradient behavior to test it.
- Demonstrate that reviewer mutant M1 (density substitutes only zero_limit)
  fails the new test in both dtype modes; restore before the full through-A4
  gate. Run focused tests, Ruff/mypy through that gate and `git diff --check`.
  Overwrite REVIEW.md, commit one review fix-up, then stop. Full docs stay
  deferred to the end of A5; spec and independent tests remain read-only.

Maintainer update (2026-10-13): code steps include tests and public API
docstrings, with no `docs/` edits. Narrative docs and their builds are separate
A5.d1, A5.d2, … steps after the last A5 code step merges; see "A5 docs" below.

Status: implemented, reviewed and merged with its fix-up on 2026-10-13.
This is a separate numerical
step before A5.3, on the maintainer-selected branch. It addresses the review's
shared-parameter mixed-batch leak; it does not promise derivatives of infinite
or NaN outputs. Covered spec: 4.2, 4.5 and 7.2–7.3; the extension recipe in
10.5 is updated and verified at the end of A5.
No public API, exports, dependencies, built-ins, grids or spec-test changes.

Files and numerical approach:

- `src/survjax/distributions/_survival_methods.py`: retain the single stopped
  reference and probe the primitive at the original policy-safe times before
  any infinity substitution. Use probe results only as Boolean conditions.
  For density, substitute all positive-infinity times before running the raw
  density, including overrides. Select -inf or NaN using the original probe's
  NaN status; a substituted reference must not determine that status.
- For the policy-wrapped log-survival primitive and the generic cumulative
  hazard, extend the existing zero-limit guard (positive infinite time AND
  negative infinite raw log survival); substitute before evaluation and select
  -inf/+inf respectively. Finite cure limits stay active. Cumulative-hazard
  overrides retain their own dispatch.
- Guard generic log-hazard with the same zero-limit condition and select its
  existing NaN boundary value. Guard generic hazard only when its underlying
  log-hazard is also generic. Analytic log-hazard/hazard overrides, including
  inherited overrides, must preserve their infinity values and derivatives.
  This prevents the generic hazard default from reintroducing the same leak.
- `src/survjax/distributions/base.py`: if needed, pass an explicit private
  primitive flag from registration to the existing policy wrapper. The
  log-survival primitive has no generic function identity; do not infer its
  role from an author's function name or introduce a new marker protocol.
  Other generic defaults continue to use static raw-function identity.
- Preserve NaN precedence, negative/NaN and zero-time behavior, finite tails,
  and reference-call count. Stop only substituted inputs, never active values
  or parameters. Reuse the probe for density instead of probing at the newly
  substituted time. Record extra evaluation cost without optimizing it here.

Tests: create `tests/unit/test_family_mixed_infinity.py` to keep the existing
infinity module below 300 lines. Use small explicit constant/quadratic/cure
families and hand-derived expectations, with existing test tolerances.

- Shared scalar rate r=2 at times [1,+inf]: finite-entry derivatives are -1
  for log survival, +1 for cumulative hazard, 1/r-1 for log density, 1/r for
  log hazard and 1 (constant) or 2 (quadratic) for hazard. Check the finite
  output component with jacfwd/jacrev, eager/jit and jit/vmap compositions;
  do not differentiate a sum containing a nonfinite output.
- Check second derivatives of the finite output component against algebra:
  log density/log hazard -1/r**2, other listed methods zero. Independent
  parameter batches must keep finite-entry derivatives separate too.
- Preserve A5.1 density values for mixed NaN/valid parameters, including
  finite and NaN overrides. Preserve nonzero fraction derivatives of finite
  log-survival/cumulative-hazard cure limits and analytic hazard overrides.
- Pin one reference call and unchanged finite values. Demonstrate pre-fix
  failures in both dtype modes, then remove only the new input substitutions
  as a mutation: finite-component reverse derivatives must fail again.

Update function docstrings with mixed-entry isolation and its limits, including
that derivatives of nonfinite outputs and arbitrary unsafe custom overrides
are not guaranteed. Do not update full docs or run `scripts/build_docs.sh`
during A5.2b. Record this behavior under "A5 docs" for the docs steps to cover
and verify with the full documentation build in both dtype modes.
`DEVIATIONS.md`: extend D-016 evidence without resolving it.
Overwrite `REVIEW.md` with invariants, test/spec mapping and exact check results.
Run focused tests in both processes, `scripts/check.sh --through A4` and
`git diff --check`. Record the deferred documentation work in REVIEW.md.
Aim for about 300 source/test/docstring lines, split if needed, commit one
numerical step, then stop before A5.3.

## A5.3 — Exponential

### Reviewer broadcasting fix-up

Authorized by the maintainer's request to implement the reviewer suggestion.
On the existing `a5.3-exponential` branch, update `log_hazard` and `hazard`
in `src/survjax/distributions/exponential.py` to compute the broadcast shape
of `t` and `self.batch_shape` explicitly before broadcasting their values.
No public API or time-policy change; this removes the raw methods' dependency
on the registration wrapper pre-broadcasting time. Covers spec 4.1/4.5/5.

Add an own regression in `tests/unit/test_exponential.py` for raw and wrapped
hazard methods with scalar time and singleton-axis time arrays against batched
rates, eagerly and with jit/vmap. Expected values are rate and log(rate).
Demonstrate the new regression fails before the source change. Run the focused
suite in both dtype modes, the through-A4 gate, the existing Exponential-only
A5 acceptance selection, and `git diff --check`. Overwrite REVIEW.md, commit
one fix-up and stop. Full docs remain deferred to the end of A5.

Status: approved and implemented on 2026-10-13 from main at `7a2fef6`, on
`a5.3-exponential`; awaiting review. This is the code step after merged A5.2b,
before Weibull (A5.4) and grids (A5.5). Approved planning and spec-marker
alignment were committed separately in `9337969` before implementation.

Spec: 3.3, 4.1–4.2, 4.5–4.6, 5 (pointwise broadcasting only), 7.2–7.3,
10.2–10.5 and the applicable 12.1 checks. Revision 23 changes documentation
scheduling only; the synced revision-22 numerical acceptance tests remain
applicable. `.spec-seen` was aligned from `9323cb5` to `a212ddd` after reading
the diff, in the separate bookkeeping commit. Spec tests and their revision
labels remain unchanged.

### Files and public API

- Create `src/survjax/distributions/exponential.py`: registered frozen dataclass
  subclass of `SurvivalDistribution`, with `rate: jax.Array = param()` as its
  only leaf. Constructor is keyword-only `Exponential(rate=...)`; no default,
  dtype keyword, value validation, log-storage flag or new dependency.
- Add `Exponential` imports/exports in `src/survjax/distributions/__init__.py`
  and `src/survjax/__init__.py`; update the exact public export assertion in
  `tests/unit/test_import.py`.
- Add typed `from_log(*, log_rate)` returning the same class. Reuse
  `_dtypes.normalize_parameters` before exponentiation so boolean/complex/
  string/object inputs cannot become valid parameters through `exp`. Keep
  weak/default dtype promotion and the float64-to-float32 warning policy.
  Store only `exp(log_rate)`; overflow yields an infinite natural rate.
- Implement `log_survival=-r*t`, `cumulative_hazard=r*t`,
  `log_prob=log(r)-r*t`, `log_hazard=log(r)` and `hazard=r` analytically.
  Inherit `survival` and stable `log_cdf` from the existing base. Preserve
  pointwise output broadcasting even for time-independent hazard methods.
  Use the existing policy wrappers; no registration or generic-wrapper changes.
- `reference_time()` returns exactly `batch_shape` in `rate.dtype`, positive
  and finite for valid rates. Use `1/r` where representable; substitute safe
  operands before division and choose a finite positive reference if reciprocal
  overflow or underflow would make it unusable. Invalid-rate references have
  no value guarantee. This fallback never changes active rates or probabilities.

### Numerical invariants and docstrings

- Negative/NaN times use the base policy. At zero: log S=H=0, S=1,
  log F=-inf, log density/log hazard=log(r), hazard=r. At +inf: log S=-inf,
  H=+inf, S=0, log F=0, log density=-inf, log hazard=log(r), hazard=r.
- Substitute safe times before inactive infinity arithmetic in analytic
  overrides where needed. Retain NaN-rate propagation at every time, including
  negative/zero/infinite times; don't hide it behind boundary constants.
  Invalid finite rates are accepted and follow the formula; add no domain mask.
- At finite-valued boundaries, derivatives are the actual limit derivatives:
  S(0), H(0), S(+inf) and log F(+inf) have zero rate derivative;
  log density/log hazard at zero and log hazard at infinity have derivative
  1/r; hazard at infinity has derivative 1. Mixed finite/infinite batches
  must preserve finite-output derivatives for shared and per-entry rates.
- Class and method docstrings state purpose, arguments, returns/shapes and
  dtype/shape misuse errors. Include defining formulas, positive finite rate
  domain, inverse-time units, rate=1/mean, log-rate/log-mean conversions and a
  short conversion example. `from_log` states natural leaves and overflow
  limits. No narrative `docs/` edits or documentation build in this code step;
  tutorial/navigation work remains in "A5 docs" below.

### Own tests and diff budget

Create `tests/unit/test_exponential.py`, with algebra or independently evaluated
NumPy formulas as expectations and existing `tests/impl_tolerances.py` values.

- Seven-method values, paired/multiaxis broadcasting, all time-policy rows,
  NaN isolation, large-H analytic hazard and underflowed survival/log density.
  At H=2e-30 assert relative precision with zero atol; check the lower-tail
  log CDF against log(2e-30), without accepting a zero-H result.
- Only natural rate as a PyTree leaf; keyword requirements, no value checks,
  no `.shape` alias, batch indexing/tree slicing and construction inside jit/vmap.
  Test ordinary/from_log integer/scalar/array dtype behavior and rejected kinds,
  plus reference shape/dtype and finite positive extreme-rate references.
- Finite log-input values and gradients against exp-parameterized algebra:
  d log f/d log r = 1-r*t. Test from_log keyword-only misuse and natural-rate
  overflow without promising a specific invalid-rate method result.
- Eager/jit/vmap agreement; interior forward/reverse derivative 1/r-t and
  Hessian -1/r**2, finite boundary derivatives and mixed-batch finite-component
  Jacobians. Use x64 order-two `check_grads`; use algebra in both modes.

Budget: about 90–110 source/docstring lines, 150–170 own-test lines and under
15 export/import-test lines; keep code/tests/docstrings at most 300 changed
lines. Reuse existing generic registration/policy tests and the independent
suite rather than copying their full matrices. Split before exceeding the
limit; don't defer correctness tests just to fit the diff.

Implementation accounting: 103 family lines, 183 own-test lines, 13 changed
export/import-test lines: **299 changed lines**. Complete boundary matrices,
registration behavior and x64 order-two finite differences reuse the independent
suite. Own tests focus on additional extreme-reference, input-dtype and
mixed-infinity cases. No coverage is deferred beyond the stated step scope.

### Checks, evidence and stop

After implementation, run focused own tests in separate x64/x32 processes,
then `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
and `git diff --check`. Run the following read-only A5 selection in each mode
with `JAX_ENABLE_X64=1` and `0`, `SPEC_THROUGH=A5`:

```bash
.venv/bin/python -B -m pytest \
  tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py \
  tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py \
  tests/spec/test_a5_derivatives.py \
  -k '(exponential and not weibull) or nan_rate or boundary_values_agree_under_vmap_over_time' \
  -q -p no:cacheprovider
```

Planning evidence: this selection collected successfully with `--collect-only
-qq` in both modes (exit 0): construction 10, from_log 4, closed forms 13,
boundaries 54, derivatives 14 — **95 cases**. No implementation tests or full
gate ran during planning. The broader `-k exponential` selection also includes
seven Weibull identity cases, so it is not the A5.3 selection. Mixed-family
tests, Weibull identities and grid tests wait for their implementing steps;
report these exclusions and the existing x32 finite-difference skips explicitly.

After code validation, overwrite `REVIEW.md` with reading order, invariants,
spec/test mapping, exact commands/results and deferred docs. Commit this one
approved code step on the selected branch and stop before A5.4. Don't merge,
publish, edit the spec/independent tests or resolve existing deviations.

## A5.4 — Weibull

- Create `src/survjax/distributions/weibull.py`, a registered frozen dataclass
  with typed natural `shape` and `scale` parameters as its only two leaves.
  Export it in both initializers; update the explicit import expectation.
- Implement `from_log(*, log_shape, log_scale)` with common input dtype
  normalization before exponentiation, then ordinary construction/broadcasting.
  Return the same class and preserve no log leaves; overflow becomes infinite
  natural parameters with the section 4.5 formula-driven behavior.
- Reference time is scale b, exactly batch_shape and parameter dtype. Implement
  log S=-(t/b)**k, H=(t/b)**k, analytic log h=log(k)-log(b)+(k-1)*log(t/b),
  and log density=log h-H. Work in log space where needed; guard xlogy-style
  operands before operations. Avoid inactive log(0), 0*inf and inf-inf.
- At zero S=1 and H=0; log density/log hazard are +inf, -log(b), -inf for
  k<1, k=1, k>1. At infinity survival=0, log-CDF=0, density=0; hazard
  limits are 0, 1/b, +inf. Preserve NaN parameters rather than hiding them
  behind boundary constants. No floors, epsilons or clipping.
- Add `tests/unit/test_weibull.py`: analytical seven-method values in all
  shape regimes, shape-one Exponential identity, natural/from_log values and
  chain-rule derivatives, batching/PyTree/indexing/dtypes, eager/jit/vmap,
  H=1e-20 relative precision, early decreasing-hazard density and large-H
  tails. Check gradients/full Hessian against algebra on positive-time
  interior cases, including mixed shape/scale derivative entries.
- Pin zero gradients of S(0), log S(0), H(0) in k/b and of survival/log-CDF
  at +inf, including transformed/mixed batches. At k=1, log density/log hazard
  at zero and log hazard at infinity have scale derivative -1/b. Do not test
  a shape derivative across these discontinuous boundary limits as a guarantee.
- Public API docstrings state the shape/scale convention and parameter domain.
  The 4.6 table and conversion examples belong in "A5 docs". Cover
  3.3/4.1/4.2/4.5/4.6/7.2/7.3/12.1. If code/tests/docstrings exceed 300 changed
  lines, split before implementation.

## A5.5 — Three time-first grid methods

- Bind typed `survival_grid(t)`, `cumulative_hazard_grid(t)`, `hazard_grid(t)`
  in `src/survjax/distributions/base.py`. Keep thin implementations in
  `_survival_methods.py`, reshaping a 1-D time vector to
  `(K,) + (1,)*len(batch_shape)` and calling the corresponding public direct
  method, honoring overrides and the complete time policy. No grid registration
  wrapper is needed. Keep modules below roughly 300 lines; if needed, put grid
  functions in one private `_grid_methods.py` with only this responsibility.
- Add `tests/unit/test_survival_grid.py` for all three methods: positional and
  keyword t; scalar/1-D/2-D distribution batches, K=N still shared rather than
  paired, more batch axes/empty grids, custom component families without
  component axes in outputs, dtype/rank misuse eagerly and during tracing,
  integer grids and silent float64-time conversion under x32. Include override
  dispatch and analytical entries, not just comparisons between library calls.
- Check eager/jit/vmap agreement, survival-grid parameter gradients and the
  zero-time shape-one survival/hazard/density example, including the three
  synthetic group scales of section 11.1. Hazards at infinity use analytic
  limits; NaN times and parameters stay isolated by entry.
- Public API docstrings state the time-first shape convention and rank/dtype
  requirements. Record tutorial, navigation and playground needs under
  "A5 docs". Cover 4.5/5/8.1/10.5/12.1; this ends the planned code steps.

### A5 docs

Docs to-do:

- Audit all `docs/` against merged A5 code: mark stale, missing or fine.
  A5.1 commit `3f50962` changed `docs/distributions.md` (12 insertions,
  1 deletion); treat it as a draft. Include the earlier tutorial rewrite
  `60bee93` and its distributions, SurvivalData, index and config changes.
- Custom families: guarded-boundary extension recipe, proper-family zero
  limits and parameter-dependent cure limits; primitive safe operands/NaN
  propagation; mixed-entry derivative isolation and its limits. Correct
  PiecewiseHazard's reference to exactly `batch_shape` in `rates.dtype`.
- Built-in families: Exponential units/domain, rate=1/mean, log-rate/log-mean
  conversions and from_log overflow limits; Weibull's complete 4.6 table and
  conversions for R/flexsurv, NumPyro, survreg mu=log(b), sigma=1/k,
  WeibullPH m=b**(-k), inverse scale and log inputs; common-shape PH gamma=-k*beta.
- Direct methods and grids: executable synthetic examples, paired-vector
  pitfalls and posterior batch `shape=k[:, None]`, `scale=jnp.exp(beta @ X.T)`.
- Update `docs/index.md` navigation/current coverage and `docs/playground.ipynb`
  for the implemented APIs. Preserve the tutorial; claim no unverified R
  agreement or full-Slice-A completion. Inspect rendered pages and execute
  examples/notebooks in both dtype modes.
- Code steps append any additional documentation needs here.

Docs steps:

- A5.d1: audit `docs/` against merged A5 code (A5.1 drafts included); list
  stale, missing, fine; split the rest into steps.
- A5.d2+: one per page or topic, e.g. SurvivalData interface, built-in families,
  custom families, grids, navigation or playground, as determined by the audit.
  Each is at most 300 changed lines; `scripts/build_docs.sh` builds clean and
  examples/notebooks execute. Review against the spec and merged code, covering
  every Docs to-do. A5 is done after its last docs step merges, before A6 starts.

## Verification and acceptance mapping

Independent A5 modules: construction/reference arrays -> A5.3/A5.4;
from_log -> A5.3/A5.4; closed forms/tails -> A5.3/A5.4;
boundaries -> A5.1/A5.2/A5.3/A5.4; derivatives -> A5.3/A5.4;
survival grids (all three methods) -> A5.5. The three revision-22 A4 additions
are A5.1 regressions. Tests purposely do not define finite-invalid-parameter
values, overflowing-from_log method values, discontinuous shape gradients,
arbitrary custom boundary derivatives or a particular reference-time formula.

- Own value tolerances remain in `tests/impl_tolerances.py`. For tiny H use
  relative-only comparisons at the existing rtol, with zero atol; do not accept
  an absolute tolerance that makes zero pass. New derivative/tail scales need
  justification in that same file. Expected values have stated algebra/identity
  origins. Never loosen tolerances to make failures pass.
- Run interior order-two forward/reverse `check_grads` with justified tolerances;
  independent finite-difference cases are x64-only, while hand-derived first/
  second derivatives run in both modes. Own x32 finite differences need their
  own scale justification. Do not toggle JAX x64 inside a process.
- Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Each code step runs
  `scripts/check.sh --through A4` and
  `git diff --check`, plus read-only independent tests for completed behavior
  in separate x64/x32 processes. At A5.5 run the full
  `scripts/check.sh --through A5` against the synced revision-22 suite.
  Report focused selection and still-unimplemented cases explicitly; never
  present an unsynced local gate as revision-22 acceptance.
  Docs steps run `scripts/build_docs.sh` in both dtype modes and inspect the
  rendered pages and outputs against the merged A5 implementation.
- REVIEW.md records reading order, numerical/storage invariants, spec-to-test
  mapping, exact outputs/skips and open deviations; docs steps also record
  executable docs and rendered-page review. One commit,
  then stop. No merge, publishing, remote setup or next-step work without approval.

## Evidence gathered while updating this plan

Read-only execution used this checkout's `.venv/bin/python -B`, with
`-p no:cacheprovider`, against `../spectest` at `7f1d6df` (no sync or writes):

- x64: `JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  ../spectest/tests/spec --collect-only -q -k test_a5 -p no:cacheprovider`
  exited 0: **370/1074 tests collected (704 deselected)**, 0.44s.
- x32: `JAX_ENABLE_X64=0 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  ../spectest/tests/spec --collect-only -qq -k test_a5 -p no:cacheprovider`
  exited 0: boundaries 186, closed forms 58, construction 29, derivatives 39,
  from_log 10, grids 48 (**370 total**).
- Ran `JAX_ENABLE_X64=1 SPEC_THROUGH=A4 .venv/bin/python -B -m pytest
  ../spectest/tests/spec/test_a4_nan_precedence.py -q -k infinity
  -p no:cacheprovider`, then the identical command with `JAX_ENABLE_X64=0`.
  Both exited 1:
  **3 failed, 7 passed, 24 deselected** (x64 0.66s; x32 0.46s).
  The three new NaN-parameter cases get -inf rather than NaN from the current
  density policy. This is an implementation gap against the changed contract,
  not a suspected spec-test error; no new deviation is needed.
- A5 tests were collected/read, not executed. The full gate, docs build,
  numerical A5 implementation checks and R work were not run for this plan.
- `git diff --check` exited 0 with no output after the planning edits.

## Remaining sub-slices: docs planning

These entries reserve the final docs steps; code-step plans fill in the page
and topic details. A1–A4 and the older plans below are completed history;
the A5 transition does not reopen them. For each remaining sub-slice, docs
steps start after its last code step merges, cover one page or topic and at
most 300 changed lines each, and pass `scripts/build_docs.sh` with examples
and notebooks executing. The next sub-slice waits for the last docs merge.

### A6 docs

Docs to-do:

- Exact/right-censored likelihoods, result containers, issue propagation and
  explicit failure for unsupported modes; refine during A6 code steps.

Docs steps:

- A6.d1+: one page or topic per step against merged A6 code; cover all Docs to-do.

### A7 docs

Docs to-do:

- Verified tail/boundary behavior, numerical limits and supported diagnostics;
  fill in any documentation needs found by A7 code/test steps.

Docs steps:

- A7.d1+: one page or topic per step against merged A7 code; cover all Docs to-do.

### A8 docs

Docs to-do:

- Reference-fixture provenance and verified R comparison claims; fill in from
  committed fixtures and A8 check results.

Docs steps:

- A8.d1+: one page or topic per step against merged A8 code; cover all Docs to-do.

### A9 docs

Docs to-do:

- Tiny synthetic example, supported Slice A workflows and limitations, and
  verified installation/transformation behavior; refine during A9 code steps.

Docs steps:

- A9.d1+: one page or topic per step against merged A9 code; cover all Docs to-do
  and record the final Slice A rendered-documentation review.

# A4.5 reviewer fix-up — F-1 and N-1 through N-4

Status: implemented and verified; see REVIEW.md for the successful A4 gate and
D1 mutation evidence. Authorized by the maintainer's request to address
`../review/slices/A4.5.md`.
Continue on `A4.5` after `fe9aa8e`; implement the review checklist in one
fix-up commit, then stop. No public API, protected-file or A5 changes.

- `_survival_methods.py`: rename the guarded logarithm to `_safe_log(x)`;
  remove the default density's duplicate +inf guard and the unreachable
  integer cast, retaining either only if a test result changes; have hazard
  resolve the raw log-hazard method via its marker, honoring overrides.
- `tests/unit/test_family_density.py`: add the quadratic-survival zero-time
  hazard gradients (single time and mixed batch, hand-derived 0 and 2);
  make the override deliberately different from the primitive hazard; count
  reference-time calls for every direct method. Covers spec 4.2/4.5/7.2/7.3.
- Temporarily remove the safe logarithm operand and show that the new F-1
  tests fail with NaN in both dtype modes; restore the implementation before
  final validation. No tolerance or independent-test changes.
- Run the full A4 gate, whole-word stale-name search and git diff checks.
  Overwrite REVIEW.md with findings addressed, corrected evidence, mutation
  results, exact gate output and the unchanged A5/D-016 handoff.
- Commit the review checklist as one fix-up with an imperative subject, then stop.

# A4.5 review fixes — WA naming comments

Status: implemented and verified; see REVIEW.md. Authorized by the maintainer's
request to address the inline WA comments. Continue on `A4.5` after `e2c08af`;
one naming-only fix-up commit, then stop.

- `src/survjax/distributions/_survival_methods.py`: rename the JVP helper to
  `_log_survival_and_derivative`, matching its actual return values. Rename
  `_log_rate` to `_log_hazard`, name its argument and safe operands for hazard,
  and document that it takes the logarithm of a linear hazard value. Update
  both defaults' calls and remove the three addressed WA comments.
- Public API and numerical behavior: unchanged. No new tests; existing density,
  hazard, zero-hazard derivative and transformation cases cover spec 4.2/7.2.
- Run `scripts/check.sh --through A4` with the existing temporary uv cache,
  check for remaining WA/stale helper names and run `git diff --check`.
- Overwrite REVIEW.md with the fixes, invariants and exact results. Commit this
  review fix-up on the same branch under the working agreement, then stop.

# A4.5 — Autodiff density and hazard defaults

Status: implemented and verified on maintainer-selected branch `A4.5`,
starting at `9d1d08d`, against specification revision 21 (2026-10-08).
The maintainer approved implementation and keeping density/hazard together
despite the roughly 375-line source/test/doc diff. See REVIEW.md for exact
evidence. The maintainer authorized the completion commit on 2026-10-08;
commit this step, then stop for review.
This is the active plan; the A4.4 entries below are completed history.

## Outcome and scope

Complete the Slice A distribution base's pointwise methods by adding typed
`SurvivalDistribution.log_prob(t)`, `log_hazard(t)` and `hazard(t)` defaults,
with the section 4.5 policy enforced for defaults and family overrides.
No new top-level exports. Their output shape is the ordinary broadcast of
`t.shape` and `batch_shape`; equal-length vectors pair entries, and singleton
axes request all pairs. These are not shared time-first grid methods.

Covered spec: 3.3, 4.1–4.2, 4.5, 5, 6.2, 7.2–7.3, 8.1 and 12.1–12.2.
Built-in Exponential/Weibull families and survival grids remain A5; likelihoods
and `parameter_issues()` remain A6. The review's D-016 positive-infinity
survival/log-CDF gradient fix stays explicitly deferred to A5. No dependencies,
support API, custom derivative rules, parameter-domain validator or new cache.

## Files and numerical implementation

1. Extend `src/survjax/distributions/_survival_methods.py`; keep numerical
   additions here, as directed by the A4.4 review, rather than growing base.py.
   Add a small typed private helper returning raw log survival and its pointwise
   time derivative using forward-mode `jax.jvp` with an all-ones time tangent.
   Reuse `_raw_log_survival`; differentiate the family primitive on the times
   already substituted by the outer policy, not the policy constants.
   Do not build a full time Jacobian or flatten distribution batch/component
   axes. Cast integer times to the common promoted floating calculation dtype
   before JVP; preserve normal weak typing and time/parameter promotion.
2. `log_prob(t)` uses `log(-d log S/dt) + log S`; it is independent of any
   `log_hazard` override. `log_hazard(t)` uses the derivative directly, avoiding
   cancellation from `log_prob - log_survival` at large cumulative hazard.
   Its generic result is NaN where raw log survival is exactly `-inf`, including
   at finite support ends; finite log survival remains eligible even when
   exp(log survival) underflows to zero. `hazard(t)` is exp(log hazard) and
   honors the family's `log_hazard` override, including its analytic +inf limit.
3. Substitute safe operands before inactive logarithms/additions; use the
   double-where pattern, stopping only placeholders/substituted inputs.
   True zero hazard gives log hazard/log density `-inf` where the formulas
   apply. Invalid negative/NaN derivatives stay NaN; never clip, floor, add an
   epsilon or infer a custom family's parameter validity domain.
4. Extend the existing policy wrapper only as needed for density at +inf.
   Negative times give log density/log hazard `-inf`, hazard zero; NaN times
   give NaN. Preserve the revision-21 NaN precedence at negative/NaN times.
   Zero remains active and obtains boundary values from the primitive or
   override. Enforce `log_prob(+inf)=-inf` for defaults and overrides, including
   formulas that otherwise yield `+inf - inf`. Keep family evaluation at zero
   and +inf consistent with 4.5, and do not change existing survival/CDF policy.
5. `src/survjax/distributions/base.py`: bind the three defaults explicitly and
   add them to registration's policy list. Keep the raw-method marker behavior
   so decorated subclasses preserve overrides without stacking wrappers.
   No construction, PyTree, static metadata or public diagnostic changes.

## Own tests, with the implementation

Create `tests/unit/test_family_density.py` (aim below 200 lines), using small
custom frozen dataclass families and hand-derived expectations. Extend the
existing method-name cases in `test_family_direct.py` only where they give
useful dtype/policy coverage for the three added methods. Use shared value and
existing justified derivative tolerances from `tests/impl_tolerances.py`;
change that file only if a distinct numerical scale needs justification.

- Constant hazard: log density `log r-r*t`, log hazard `log r`, hazard `r`;
  scalar, paired vectors, explicit outer broadcasting, integer times and
  mixed time/parameter dtypes. Exercise jit/vmap, shape-mismatch TypeError,
  and repeat compiled calls with changed parameter values.
- Power survival `log S=-(t/b)**k`: hand-derived log density/hazard across
  decreasing, constant and increasing hazards, zero-time values for k below,
  equal to and above one, and large H with representable hazard. Pin density
  at +inf to -inf and distinguish generic hazards from analytic overrides.
- A bounded-support primitive: generic hazards become NaN at/beyond the
  support end; finite very negative log survival keeps a valid hazard despite
  linear survival underflow. Include a locally zero hazard without log(0)
  contaminating an otherwise active batch's finite derivatives.
- Explicit/inherited density and log-hazard overrides: verify dispatch,
  hazard's use of the log-hazard override, policy at negative/NaN/+inf times,
  no mutation, and no stacked wrappers after subclass registration.
- Unguarded primitive and parameter-dependent reference: negative/NaN time
  slots see safe inputs and retain zero parameter gradients with valid
  parameters. Mixed valid/NaN parameters keep NaNs isolated by entry.
- Smooth positive-time interior: hand-derived gradient and Hessian (constant
  density has `d/dr=1/r-t`, `d2/dr2=-1/r**2`); check first/second parameter
  derivatives forward/reverse, composed with jit/vmap. Run order-two
  `check_grads` with the established tolerances. Do not promise parameter
  derivatives at invalid inputs, support changes or infinite log densities.

## Documentation, verification and stopping point

- Update `docs/distributions.md` with an executable concrete density/hazard
  example, the complete negative-time policy, generic zero-survival behavior,
  override dispatch, and the representable-linear-hazard limitation of the
  autodiff defaults. Update `docs/index.md` to reflect the completed methods.
- Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run the full maintainer gate
  `scripts/check.sh --through A4`: Ruff, mypy, own tests and all independent A4
  tests in separate x64/x32 processes. The former focused subset is no longer
  sufficient; this step completes the remaining A4 acceptance surface.
- Run combined own/independent sessions with `SPEC_THROUGH=A4` in both modes
  to catch collection/import conflicts, source doctests on base.py and
  _survival_methods.py in both modes, executable docs in both modes, and
  `git diff --check`. No network, R fixture generation or wheel work here.
- Keep spec, acceptance tests, gate scripts and alignment markers read-only.
  On a genuine contract ambiguity or apparent acceptance-test error, record
  an open deviation and ask rather than silently choosing or editing tests.
- Aim for roughly 300 changed source/test/doc lines. If it grows materially
  beyond that, propose a split before expanding the implementation.
- Overwrite REVIEW.md with reading order, numerical/dispatch invariants,
  spec-to-test mapping, exact results/skips and D-016's A5 handoff. Update this
  status, then stop for review. Do not begin A5 or commit until instructed.

# A4.4 completion — specification revision 21 (2026-10-08)

Status: implemented and verified; the maintainer authorized the completion
commit on 2026-10-08. See REVIEW.md for current results and the A5 handoff.

Approved by the maintainer's request to finish A4.4 after spec/acceptance updates.
This continues the existing approved step on `A4.4` at `d0b5538`; no A4.5 work.
The earlier implementation was committed by the maintainer as `8326b45`.
Two new untracked files under tests/spec are read-only inputs; leave them alone.
The maintainer's 2026-10-08 commit request supersedes the earlier hold.

- Rename `distributions/_direct.py` to `_survival_methods.py` and update the
  base import/bindings/wrapping calls, as directed by review A4.4 item 3.
- In that module, reject nonreal/non-numeric times with field-independent
  TypeError using dtype only, including under jit/vmap. Preserve JAX conversion
  and promotion without float64-time warnings (spec 4.5 revision 21).
- Keep numerical NaN precedence; replace its D-013 comment with the now-settled
  spec 4.5 citation. Consolidate repeated raw-primitive lookups into the
  review-requested typed `_raw_log_survival` helper. No public API additions.
- Add own dtype rejection/transformation and no-warning/promotion tests in
  `tests/unit/test_family_direct.py`, plus the exact hand-derived time derivative
  at zero (`-rate = -2`). Keep existing tolerances unchanged.
- Add the review's D-016/D-017/D-018 entries to DEVIATIONS.md. D-016's fix is
  explicitly deferred to A5. D-017/D-018 now have normative revision-21 text;
  annotate that evidence but leave all statuses open for maintainer closure.
- Update docs/distributions.md for direct-time dtypes, NaN precedence and
  finite-invalid-parameter behavior. Update the active plan/status and overwrite
  REVIEW.md with current invariants and exact evidence. No protected edits.

Coverage: spec 3.3, 4.2, 4.5, 5, 6.2, 7.2–7.3, 12.1. Validation: through-A3
maintainer gate; prior 112 A4 nodes plus new acceptance nodes using only the
four A4.4 methods, in both dtype modes; combined own/spec through-A3 sessions;
docs build and source doctests in both modes; git diff --check. Do not run or
claim the full A4 gate until A4.5. Keep this fix-up under roughly 300 lines.
Commit this completion as authorized, then stop before A4.5.

# A4.4 — Direct-method policy and survival/CDF defaults

Historical plan: approved by the maintainer ("go for it and implement") and
implemented on `A4.4`, starting at `6dc4747` against specification revision 19.
The maintainer committed that implementation as `8326b45`. The revision-21
completion above is the active plan; this and earlier plans are history.

## Outcome and public API

Registered custom families gain `survival(t)`, `log_cdf(t)` and
`cumulative_hazard(t)` defaults. Registration enforces the section 4.5 time
policy for these methods and `log_survival(t)`, including family overrides.
The primitive family implementation sees only nonnegative, non-NaN times;
zero and positive infinity still reach it. Pointwise shapes follow ordinary
trailing-axis broadcasting, with TypeError for static time/batch mismatches.

No new top-level exports. Density and hazard defaults/policy stay A4.5;
built-ins and time-first grids stay A5. D-013 remains an A6 handoff: do not
invent a parameter diagnostic contract or custom parameter validity domain.

## Files and implementation

- Create `src/survjax/distributions/_direct.py`: typed ordinary functions for
  the three generic defaults and a single registration-time policy wrapper.
  Keep this numerical responsibility separate from construction/PyTrees.
  Use TYPE_CHECKING imports for the base type to avoid import cycles.
- `src/survjax/distributions/base.py`: explicitly bind the three default
  functions as base methods and integrate policy wrapping at registration.
  Preserve the concrete nested-family docstring and all A4.3 behavior.
  Avoid a large mechanical registration refactor just to add these methods.
- Defaults are `exp(log_survival(t))`, `-log_survival(t)`, and
  `log(-expm1(log_survival(t)))`. Guard zero CDF with safe inputs before the
  logarithm, then return -inf for true zero; no floors, epsilons or clipping.
  NaNs remain NaNs. Document the generic CDF lower-tail underflow limit.
- The policy wrapper first broadcasts times against `batch_shape`. For
  negative/NaN times, substitute the family's reference time before executing
  family code and stop gradients only through the substituted input. Apply
  negative-time constants (0, 1, -inf, 0 respectively) and NaN-time outputs
  afterwards. Preserve NaN family outputs; do not hide invalid parameters
  behind a negative-time constant or add host-side parameter inspection.
- Wrap both inherited defaults and overrides of the four in-scope methods.
  Retain raw implementations through a transparent wrapping convention, so
  registration of explicitly decorated subclasses does not stack policy
  wrappers or introduce recursion. Cover inherited overrides in own tests.
  This is registration-time setup, not a mutable evaluation cache.
- `tests/unit/test_family_direct.py`: independent custom primitive and override
  families, one behavior per test. Extend `tests/impl_tolerances.py` with the
  mandated float32/float64 value tolerances and any justified tail-specific
  tolerance; retain existing derivative tolerances.
- Add the required reference hook to the existing batching-test Wrapper,
  whose numerical method now goes through the policy.
- `docs/distributions.md` and `docs/index.md`: describe the now-available
  methods, boundary policy and shape convention. Add an executable primitive
  family example with actual output values and positive-time/negative-time
  evaluation; mark density/hazard defaults as later A4 scope.
- Overwrite REVIEW.md after implementation: reading order, raw/policy method
  dispatch invariants, spec-to-test mapping, exact checks/results and limits.

Spec coverage: 4.2, 4.5, 5, 7.1–7.3, 10.2–10.5 and 12.1.

## Tests and evidence

- Hand-derived constant-hazard values: `log S=-r*t`, `S=exp(-r*t)`,
  `H=r*t`, `log F=log(-expm1(-r*t))`. Check scalar, paired vectors and
  explicit `(D,1)` by `(N,)` evaluation; incompatible shapes raise TypeError.
- All four methods: negative infinity/negative/zero/interior/positive infinity/
  NaN time values, eagerly and under jit/vmap, including mixed batches.
  Include invalid parameter NaNs beside valid entries and verify isolation.
- Independently override each derived method and verify dispatch plus policy
  enforcement; register a decorated subclass inheriting an override and
  verify it is not wrapped twice. Methods do not mutate PyTree leaves.
- A deliberately unguarded positive-time primitive verifies safe substituted
  times and finite zero parameter gradients for negative policy values.
  Test parameter-dependent reference times without stopping active parameter
  calculations; test second-order forward/reverse gradients on valid smooth
  interior cases using existing justified derivative tolerances.
- Generic log-CDF tails: tiny representable mass compared in log space against
  high-precision mpmath; zero mass remains -inf. No expected value is derived
  from the implementation under test. Include zero-boundary inputs where
  inactive operations must not produce spurious gradient NaNs.

## Validation and stopping point

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` throughout:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests in separate dtype processes.
2. Repeat the existing 89 focused A4 nodes in x64/x32. Additionally select
   from `test_a4_generic_defaults.py` every test except the seven tests using
   `log_hazard`/`hazard`; from `test_a4_policy_enforcement.py`, select
   `test_unguarded_log_survival_is_zero_for_negative_time`,
   `test_unguarded_survival_is_one_for_negative_time`,
   `test_unguarded_cumulative_hazard_is_zero_for_negative_time`,
   `test_unguarded_log_cdf_is_minus_infinity_for_negative_time`,
   `test_unguarded_infinite_time_log_survival_comes_from_the_family`,
   `test_negative_time_log_survival_has_zero_parameter_gradients` and
   `test_negative_time_survival_has_zero_parameter_gradients`.
   Inspect dependencies and record exact selectors/results in REVIEW.md.
   Do not edit, skip or xfail protected acceptance tests. The full A4 gate
   remains required at A4.5.
3. Combined own/independent sessions with `SPEC_THROUGH=A3`, in both dtype
   modes, preserving the unique implementer tolerance-module name.
4. `scripts/build_docs.sh` (approved kernel-port escalation if needed), plus
   the existing concrete nested-field doctest in both modes.
5. `git diff --check`.

Aim for about 300 changed code/test lines. If implementation requires a
larger refactor or split, propose it before exceeding the approved step.
Do not change dependencies, the specification, protected scripts or tests/spec.
Log newly uncovered spec ambiguities as open deviations and ask before choosing.
The earlier no-commit instruction remains in effect: implement and verify after
approval, leave the result uncommitted, then stop for review.

Planning evidence: inspected branch/status, current source, relevant spec,
existing plans/deviations, docs and read-only generic-default/policy acceptance
sources. No runtime, lint, type or documentation checks ran in this planning
turn. `git diff --check` passed with no output.

# A4.3 review fixes — approved, uncommitted

The maintainer requested implementation of `../review/slices/A4.3.md`.
Preserve the prior concrete docstring change; leave all fixes uncommitted
until further review, overriding the review file's suggested commit.

- `src/survjax/distributions/base.py`: F-1, mark each explicitly registered
  class and reject other nested values with field-named TypeError. Do not
  inherit registration from a parent. F-2, use the short tree-map indexing
  form and a spec 4.1 shared-leading-batch-rank comment; delegate index errors
  to JAX. Wrap the long declaration error (N-3). No new public API.
- `tests/unit/test_family_batching.py`: test array/None/unregistered nested
  values and a subclass that inherits a registered parent's marker; update
  the derivative-tolerance import (N-1). Existing indexing tests cover F-2.
- Rename `tests/tolerances.py` to `tests/impl_tolerances.py` as decided by the
  maintainer (N-1). Tighten float32 derivative tolerances to 1e-3 and justify
  the rounding scale (N-2). Do not change the independent tolerance module.
- `docs/index.md`: wrap the flagged line. `docs/distributions.md`: document
  JAX's out-of-bounds integer clamping, as decided by the maintainer.
- Record the review-directed nested-field TypeError policy as open D-015 in
  DEVIATIONS.md; no approval question is needed because the maintainer
  explicitly requested these fixes. Overwrite REVIEW.md with current reading
  order, invariants and exact check results. Spec coverage: 3.3, 4.1–4.2, 5, 6.2, 7.3, 10.5, 12.1.

Run `scripts/check.sh --through A3`, the previous 89 focused A4 nodes in both
modes, `scripts/build_docs.sh`, combined `SPEC_THROUGH=A3` unit/spec pytest
sessions in both modes, the concrete docstring doctest in both modes, and
`git diff --check`. No new dependencies, spec or protected-file edits.

# A4.3 review follow-up — concrete nested-field example

Approved by the maintainer's request to do this edit and leave it uncommitted.
Change only the `distribution()` docstring in
`src/survjax/distributions/base.py`, plus this plan and the review note.
Show complete custom constant-hazard and scaled-hazard families, then construct
and index an actual nested instance with numerical outputs. Use the existing
A4.3 API; no built-in families, numerical defaults or new public API.
Covers spec 4.2, 5 and 10.5. Execute the docstring example in both dtype modes,
run focused Ruff and mypy checks, and record results in REVIEW.md.
No new test file or commit; stop for further review.

# A4.3 — Nested distributions and batch indexing (2026-10-02)

Status: approved by the maintainer ("ok implement it") and implemented on
`A4.3`; see REVIEW.md for evidence. This is the active step plan; earlier plans below are history. Work on the maintainer-created
`A4.3` branch, starting at `138db08` (A4.2 review fix-up), against specification
revision 19. The working tree was clean before this plan update.

## Outcome and public API

Registered frozen families can declare a required nested distribution with
`survjax.distribution()` and select batch entries with `dist[idx]`. Own
parameters and all nested distributions share the outer batch shape after
construction. Nested component axes, dtypes, family types and static
configuration survive broadcasting, indexing and JAX transformations.

Export `distribution()` from both `survjax` and `survjax.distributions`.
Add typed `SurvivalDistribution.__getitem__`; extend the existing
`batch_shape` property and `register_family` behavior. No other public names.
Spec coverage: 3.3 (independent nested dtypes), 4.1–4.2 (representation,
registration and indexing), 5 (batch shapes), 7.2–7.3 (compilation reuse and
parameter transformations), 10.2–10.5 and 12.1 (engineering/docs/tests).

## Files and implementation

- `src/survjax/distributions/base.py`: add the no-argument, no-default
  `distribution()` dataclass field helper. Recognize nested fields alongside
  `param()` and `static()` in declaration checks and document all three roles.
  Include nested objects as dynamic children, so their leaves and static
  auxiliary data participate in the outer PyTree. Reconstruction continues
  to bypass public constructors, normalization and value inspection.
- Extend construction to broadcast own parameter batch shapes together with
  nested `batch_shape` values. Normalize only own parameters, using the existing
  helper and preserving warning attribution. Recursively broadcast nested
  parameter leaves to the common batch shape plus their declared component
  shapes, retaining their original dtypes and static values. Rebuild nested
  objects without public constructors; leave the supplied baseline immutable.
  Incompatible batch shapes raise TypeError. Support wrappers whose dynamic
  fields are exclusively nested distributions.
- Derive `batch_shape` from current own and nested leaves, excluding component
  axes. Store no cached batch shape or extra static shape metadata.
- Implement batch-only indexing for integers, slices, integer arrays and
  tuples, including ellipsis and inserted singleton axes. Resolve indexing
  against the batch rank before appending untouched component-axis slices;
  ellipsis must never consume component axes. Recursively index nested
  distributions and reconstruct without constructors. Follow ordinary JAX
  indexing constraints, including traced integer indices under jit and the
  restrictions on dynamic-size indexing. Do not add custom bounds semantics.
- Keep `base.py` under about 300 lines. Only if necessary, move registration
  internals to `src/survjax/distributions/_registration.py`, preserving public
  imports and caller-site warning behavior; no general helper framework.
- `src/survjax/distributions/__init__.py`, `src/survjax/__init__.py` and
  `tests/unit/test_import.py`: add and check the new export.
- Create `tests/unit/test_family_batching.py` with independently defined custom
  families and small behavior-specific tests. Use existing tolerances;
  derivative-specific additions belong only in `tests/tolerances.py` with a
  scale justification.
- `docs/distributions.md`: add executable nested-field and batch-selection
  examples, explain component axes and independent nested dtypes, and show
  paired `(N,)` versus explicit `(D,1)` evaluation using a simple custom
  `log_survival` implementation at valid positive times.
  `docs/index.md`: update the current-feature description.
- Overwrite `REVIEW.md` after implementation with reading order, invariants,
  spec-to-test mapping, exact checks/results/skips and remaining questions.

## Tests and evidence

- Spec 4.2: helper takes no arguments; nested fields are required; round trips
  retain nested types, static configuration and supplied sentinel leaf identity.
- Spec 4.1–4.2/5: scalar baseline/vector wrapper, vector baseline/scalar wrapper,
  `(2,1)` with `(3,)`, nested-only wrapper and at least two nesting levels;
  incompatible batches fail. Verify stored leaf shapes and explicit values,
  nested components (including two component axes), and unchanged input objects.
- Spec 3.3: x64-only mixed float32/float64 nesting preserves each family's dtype
  through broadcasting and indexing; x32 covers available dtypes without
  changing global configuration or claiming float64 preservation.
- Spec 4.1–4.2: integer, slice, array and multi-axis tuple selection preserves
  components/static fields; ellipsis and singleton insertion operate on batch
  axes only. Compare indexed leaves to independently indexed input arrays.
  Tree-map slicing agrees with batch selection and updates outer/nested shapes.
- Spec 7.2–7.3: eager/jit/vmap construction and selection agree; jit supports
  a traced integer index; vmap sees per-entry nested batch shapes and full
  component vectors. Compile-count tests vary outer and nested dynamic values
  at fixed shapes/dtypes/structure and require reuse; changed nested static
  configuration specializes separately. Use a smooth custom log-survival
  formula with hand-derived derivatives and second-order forward/reverse
  `check_grads` to verify gradients through nested construction and selection.

Run with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests, separately in both dtype modes.
2. Focused read-only A4 acceptance nodes with `JAX_ENABLE_X64=1` and `0`,
   `SPEC_THROUGH=A4`, and `uv run --locked --offline pytest -q`:
   - `test_a4_registration.py`: all except `test_methods_do_not_mutate_parameters`.
   - `test_a4_parameters.py`: all except
     `test_construction_with_traced_parameters_inside_jit`,
     `test_construction_inside_vmap` and `test_gradient_through_construction`.
   - `test_a4_indexing.py`: all except
     `test_indexed_distribution_matches_indexed_values`,
     `test_vmap_over_distribution_matches_batched_call` and
     `test_vmap_over_distribution_and_time_pairs_entries`.
   - `test_a4_nested.py`: all except
     `test_vmap_over_outer_distribution_with_broadcast_base` and
     `test_broadcast_nested_values_in_log_prob`.
   These excluded nodes require later generic numerical methods. Record the
   exact node list and results in REVIEW.md; do not edit, skip or xfail tests.
3. `scripts/build_docs.sh`: execute examples in separate x32/x64 processes and
   inspect the rendered guide. Report any environment/permission limitations.
4. `git diff --check`.

## Boundaries and stopping point

Direct-method policy/defaults remain A4.4–A4.5; built-ins, grids and actual
PH/AFT wrappers remain later steps. D-013 remains an A6 diagnostic-contract
handoff. Do not change dependencies, the spec, independent acceptance tests
or protected scripts. Log any newly uncovered spec ambiguity as an open
DEVIATIONS.md entry and ask before choosing its behavior.

Aim for roughly 300 changed lines including code and tests. If the necessary
implementation grows past that, propose a smaller split before expanding it.
After approval, implement one concept on `A4.3`, write REVIEW.md, commit with
an imperative subject and a body explaining why, then stop for review.
This plan-only update remains uncommitted.

Planning evidence: read the relevant spec, A4.2 implementation/review,
dtype helper, docs and read-only A4 acceptance sources. No runtime, lint,
type, derivative or documentation checks were run for this planning update.

# A4.2 review fix-up (2026-10-02)

Status: approved by the maintainer on 2026-10-02 after reading
`../review/slices/A4.2.md`, with public-docstring improvements added. One extra commit on `A4.2`, then stop.
No A4.3 implementation or documentation-page rewrite.

## Changes

- `src/survjax/distributions/base.py`, F-1: reject dataclass fields whose
  metadata does not identify an available field role (`param` or `static`),
  with a field-named TypeError at registration. Record the unspecified plain
  field policy in D-014; the maintainer approved this implementation policy.
- N-1: validate `component_ndim` in `param()`: a non-negative Python int,
  excluding bool. Use TypeError for wrong kinds and ValueError for negatives.
- N-2: require an initializer owned by the class before wrapping it. Reject
  inherited initializers with a TypeError explaining that the subclass must
  be decorated with `@dataclass` itself. This prevents double normalization
  and incorrect warning attribution; retain the documented decorator recipe.
- N-3: catch static hashability TypeError and re-raise naming the field and
  stating that static fields must be hashable.
- Rewrite the public field helpers, base, batch-shape property, extension hooks
  and registration docstrings for users: purpose, arguments, shapes, errors and
  one short declaration example. Keep spec/development bookkeeping out of them.
- `tests/unit/test_family_registration.py`: add separate tests for plain
  default/default-factory field rejection; invalid component counts; inherited
  initializer rejection and a re-decorated subclass's x32 warning location;
  field-named static hashability errors; insufficient component rank and
  non-frozen registration guards. Keep tests independent of tests/spec.
- `DEVIATIONS.md`: add open D-014. Do not change existing statuses or spec.
- `REVIEW.md`: overwrite with reading order, requirement coverage, exact
  check outcomes and remaining scope; correct the earlier claim that no
  registration ambiguity existed.

Public names stay unchanged. These are declaration/error checks for the
extension interface (spec 3.3, 4.1–4.2, 5); no parameter-value checks or
numerical-method work. Aim below 300 changed lines including tests.

## Verification

Run `scripts/check.sh --through A3`, the same 60 focused A4 acceptance nodes
listed in the prior REVIEW.md in both dtype processes, `scripts/build_docs.sh`
in both modes, and `git diff --check`. Use
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Record dtype-specific skips and
any failures. Commit this fix-up only after checks pass, then stop.

Planning checks: read the external review and current implementation;
working tree was clean before this plan/deviation update. Runtime checks
were not rerun for this plan-only task.

# A4.2 — Registration and flat family construction (2026-10-02)

Status: approved by the maintainer ("go for it and build it") and implemented
on the existing `A4.2` branch; see REVIEW.md for verification. This is the active step plan; the broader A4 sequence below remains
background. Based on specification revision 19 and completed A4.1 at
`15cfc6f` (current HEAD also contains the D-013 handoff note).

## Outcome and scope

A custom frozen dataclass can be explicitly registered, constructed by keyword
with normalized and broadcast parameter arrays, and passed through JAX as an
immutable PyTree. This step covers spec 3.3, 4.1–4.2, construction shapes in
section 5, and engineering/documentation requirements in 10.2–10.5.

Public additions: `SurvivalDistribution`, `register_family`,
`param(component_ndim=0)` and `static(...)`, exported from both `survjax` and
`survjax.distributions`. The base exposes typed `batch_shape` and the typed
`log_survival(t)` / `reference_time()` extension hooks. Hooks without family
implementations fail explicitly; no numerical defaults are supplied yet.
No base `shape` attribute: a family's own `shape` parameter remains usable.

Nested `distribution()` fields and batch indexing stay in A4.3. Direct-method
policy/defaults stay in A4.4–A4.5; built-ins and grids stay in A5. D-013 stays
open for A6. The guide will state these intermediate limitations explicitly.
No dependency, spec, protected-script or independent-test changes.

## Files and implementation

- Create `src/survjax/distributions/base.py` and its package `__init__.py`.
  Use dataclass field metadata to distinguish parameters from static fields,
  ordinary functions and explicit `jax.tree_util` registration. Require the
  documented frozen-dataclass subclass recipe; no automatic registration,
  metaclasses or dynamic attribute lookup. Keep the module under about 300 lines.
- Wrap the dataclass-generated constructor to reject positional arguments,
  preserving keyword argument errors, required parameters and static defaults.
  Support `kw_only=True`, including a defaulted static field before parameters.
  Static configuration must be hashable. Do not add parameter-value validation.
- Reuse `_dtypes.normalize_parameters` for all own parameters, including
  components. Pass the stack level needed for warnings to point to the user's
  constructor call. Constructors accept no `dtype` keyword.
- Derive each parameter's batch axes by excluding its declared trailing
  component axes. Broadcast those batch shapes, then broadcast each array to
  the common batch shape plus its original component shape. Translate
  incompatible batch shapes into `TypeError`. Never broadcast component axes
  against batch axes or inspect parameter values on the host.
- Flatten arrays as dynamic children, with static configuration in auxiliary
  data. Unflatten using allocation and frozen-field assignment, bypassing the
  constructor, normalization and broadcasting; supplied leaves retain identity,
  including JAX placeholders. Compute `batch_shape` from current leaves rather
  than store cached shape metadata, so tree mapping and vmap update it naturally.
- Update `src/survjax/__init__.py` and `tests/unit/test_import.py` for the four
  new names; preserve existing exports.
- Add `tests/unit/test_family_registration.py` with small independent custom
  families. Do not import acceptance-suite recipes or helpers.
- Create `docs/distributions.md` with an executable construction/PyTree example,
  decorator order, keyword-only recipe, static defaults, component axes, dtype
  behavior and absent `shape` alias. Add it to `docs/index.md` and update that
  page's current-feature description. Use supported construction operations;
  label numerical defaults, nested fields and indexing as planned.

If registration edge cases require choices the specification does not settle,
record a new open deviation and ask before implementing that behavior. If code,
unit tests and docs exceed roughly 300 changed lines, propose a further split
before expanding the step; approval here does not authorize a larger step.

## Tests and evidence

- Spec 4.1–4.2: immutability, explicit registration only, required/keyword-only
  arguments, static defaults/overrides and the static-first kw-only recipe.
- Spec 3.3: public-constructor dtype integration, components sharing dtype,
  rejected nonreal inputs and None, unavailable-float64 warning count and caller
  location, and rejection of `dtype=`. Reuse A4.1's detailed promotion tests.
- Spec 4.1–4.2/5: scalar and multi-axis broadcasting, component preservation,
  incompatible batch shapes, stored negative/NaN/infinite natural parameters,
  and a family with its own `shape` field alongside a family without one.
- Spec 4.2/7.2: round trips, static tree-structure equality/difference, sentinel
  identity on unflatten, tree-map slicing, eager/jit construction, and vmap
  construction/per-entry `batch_shape`. Use exact shape/value assertions where
  applicable; no derivative or direct-method policy claims in this step.

Run with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests, in separate dtype processes where applicable.
2. Focused A4 acceptance nodes in separate processes with
   `JAX_ENABLE_X64=1` and `0`, `SPEC_THROUGH=A4`, and
   `uv run --locked --offline pytest -q`. Select registration tests except
   the all-extension-names, method-purity and three nested-distribution tests;
   select parameter tests except `test_nested_distribution_keeps_its_own_dtype`,
   `test_construction_with_traced_parameters_inside_jit`,
   `test_construction_inside_vmap` and `test_gradient_through_construction`
   (these three use the later generic `log_prob`);
   select `test_a4_indexing.py::test_tree_map_slicing_updates_batch_shape` and
   `::test_batch_shape_inside_vmap_is_the_per_entry_shape`. Inspect dependencies
   before execution and record the exact node selection in REVIEW.md. Never
   skip/xfail or edit unfinished acceptance tests, or claim the full A4 gate.
3. `scripts/build_docs.sh`: execute examples in x32/x64 and inspect the rendered
   distribution guide. Report any permission or environment limitation honestly.
4. `git diff --check`.

Overwrite `REVIEW.md` with reading order, construction/PyTree invariants,
requirement-to-test mapping, exact commands/results/skips and open questions.
Commit this one implementation step on `A4.2` with an imperative subject and
why in the body, then stop. The plan-only update remains uncommitted.

Planning checks: read the spec, existing implementation/docs, A4.1 review and
read-only A4 acceptance sources. Working tree was clean on `A4.2`. No runtime,
lint, type or documentation checks were run for this planning update.
`git diff --check` passed with no output.

# A4.1 review fix-up (2026-10-02)

Approved by the maintainer's request to implement `../review/slices/A4.1.md`.
One fix-up commit on `A4.1`, then stop; no A4.2 implementation.

- F-1: reject explicit None own parameter leaves with a field-named TypeError
  in `_dtypes.py`; add the case to `test_parameter_dtype.py` (spec 3.3, 4.2).
- N-1: include a float32 leaf in the x32 warning test, so a single float64
  leaf must still warn (spec 3.3).
- N-2: pass the warning noun explicitly from `_data_validation.py` and
  `normalize_parameters` rather than derive it from input-kind strings.
- Public API unchanged. Preserve observation errors and warning attribution.
- Run `scripts/check.sh --through A3`, `scripts/build_docs.sh` in both dtype
  processes, and `git diff --check`; overwrite REVIEW.md with results.
- Remove the obsolete D-001 template as requested in review bookkeeping.
  Preserve the existing unstaged DEVIATIONS.md planning edit; no deviation
  statuses are changed by this fix-up.

# Step A4: distribution base and extension interface (2026-10-02)

Status: A4.1 approved by the maintainer on 2026-10-02; later parts await approval. Based on specification revision
19; re-evaluated against the synced A4 acceptance suite at source commit
`f62c81ff5b2317ecf692e546a2d3beeafd5c683f`. A4.1 implementation is on the maintainer-created `A4.1` branch. Earlier plans below
are history; `REVIEW.md` continues to describe the completed A3 work.

## Scope and decisions needed

Implement the Slice A distribution foundation: frozen registered families,
parameter/static/nested fields, construction and reconstruction, batch indexing,
and the seven pointwise direct methods. Cover sections 3.3, 4.1, 4.2, 4.5, 5,
7.2–7.3, 10.2–10.5 and the relevant parts of 12.1–12.2.

New top-level exports: `SurvivalDistribution`, `register_family`, `param`,
`static`, `distribution`. Existing exports remain. Methods added by completion:
`batch_shape`, `__getitem__`, `reference_time`,
`log_survival`, `survival`, `log_prob`, `log_cdf`, `cumulative_hazard`,
`log_hazard`, `hazard`. `log_survival` and `reference_time` are the required
family primitives. There is no base `shape` attribute and no constructor
`dtype` keyword. Signatures, public fields and extension interfaces are typed.

The nine A4 acceptance files are now present: 203 tests collect in each dtype
mode. Read-only `API_ASSUMPTIONS.md` Q-019–Q-025 and `TESTMAP.md` confirm the
scope below. In particular, Q-024 explicitly leaves `parameter_issues()` and
its unspecified return type for A6. Remove that API and generic nested-issue
aggregation from A4; D-013 remains open as an A6 handoff, not an A4 blocker.
Do not infer domain validity for arbitrary custom parameters. A4 preserves
NaNs produced by family formulas and generic defaults; its acceptance examples
exercise a negative rate through `log_prob` and NaN parameters through direct
methods. Passing them is not evidence for general parameter diagnostics.

Built-in Exponential/Weibull families, log-parameter constructors and survival
grids remain A5. Likelihoods/result containers remain A6. Interval methods,
wrappers, summaries, sampling, `HazardFamily` and `check_family` remain their
later slices. Use small analytical custom families in tests and documentation
to exercise A4 without introducing built-ins early.

## Review-sized implementation sequence

A4 needs multiple commits to respect the roughly 300-line review target,
including tests. Each numbered part below is a separate step: confirm its
scope in this plan, obtain approval, implement with tests, overwrite `REVIEW.md`,
commit on the maintainer's branch, then stop. Approval to begin A4.1 does not
authorize automatically continuing through the remaining parts. If a part
exceeds the target, propose a further split before expanding it.

### A4.1 — Shared floating-dtype normalization

- Files: new `src/survjax/_dtypes.py`; update `_data_validation.py`; add
  `tests/unit/test_parameter_dtype.py`. Reuse the existing dtype logic by a
  narrow extraction rather than coupling distributions to observation rules.
  Preserve observation behavior, error meaning and warning attribution.
- Normalize each family's own parameters to one dtype, including components;
  nested distributions retain their dtype. Preserve weak promotion, the
  configured default for integer/scalar-only inputs, and the float32 minimum.
  Reject bool, complex, string and object inputs. Warn once per user call site
  for float64 arrays under x32; do not warn for Python scalars/lists. Never
  inspect traced values on the host or change global JAX configuration.
- Tests: dtype combinations, float16 promotion, weak inputs, rejected input
  kinds, x32 warning count/location and traced normalization. Existing data
  dtype tests protect the extraction. Public API unchanged (spec 3.3).
- Docs: explain the shared policy in `docs/survival-data.md`; mark distribution
  examples as planned until registration exists.

### A4.2 — Registration and flat family construction

- Files: new `src/survjax/distributions/base.py` and package `__init__.py`;
  update top-level `__init__.py` and `tests/unit/test_import.py`; add
  `tests/unit/test_family_registration.py`.
- Add the base, `param(component_ndim=0)`, `static(...)` and explicit
  `register_family`. Use dataclass field metadata and small ordinary functions;
  no metaclasses, dynamic attribute lookup or automatic subclass registration.
- Enforce keyword-only construction, required parameter fields and optional
  static defaults. Normalize parameter dtypes; broadcast leading batch axes
  while retaining trailing component axes. Incompatible batch shapes raise
  `TypeError`. Construction does not validate parameter values.
- Register dynamic arrays as children and hashable static configuration as
  auxiliary data. Reconstruct using allocation/field assignment, bypassing
  construction and normalization, including for JAX placeholders.
  Derive `batch_shape` from current leaves, never cached shape metadata.
- Tests: immutability; required/keyword-only arguments; static defaults;
  component and batch shapes; dtype integration; eager/jit construction;
  flatten/unflatten and tree mapping; no implicit registration; absent base
  `shape`. Explicitly test the `kw_only=True` recipe with a defaulted static
  field before required parameters, unchanged object identity for unflattened
  sentinel leaves, and storage of negative/NaN/infinite parameter values.
  Test malformed static structure without checking parameter values.
- Docs: add `docs/distributions.md` to `docs/index.md`, showing the decorator
  order, field roles and construction; label methods not yet implemented.
  Spec: 3.3, 4.1–4.2, 5, 10.2–10.5.

### A4.3 — Nested distributions and batch indexing

- Files: extend `distributions/base.py`; add a focused private
  `distributions/_registration.py` if needed to keep responsibilities and
  modules small; export `distribution()`; update import tests; add
  `tests/unit/test_family_batching.py`.
- Include nested batch shapes in outer broadcasting. Broadcast nested leaves
  recursively, preserving their component axes, own dtypes and static fields.
  The no-argument `distribution()` field is a required nested distribution.
- Implement batch-only indexing, applying indices to leading axes while
  retaining component axes. Rebuild without public constructors. Support
  scalar/slice/array/tuple indexing within ordinary JAX indexing constraints.
- Tests: nested scalar/vector/multi-axis broadcasting; incompatible shapes;
  nested component axes and mixed dtypes; indexing, tree-map slicing and
  vmap agreement; per-entry `batch_shape` inside vmap. Confirm changed dynamic
  values reuse compilation with equal shapes/dtypes/structure/configuration,
  and changed static configuration may specialize separately.
- Docs: expand `docs/distributions.md` with nested fields and batching;
  distinguish paired `(N,)` evaluation from explicit `(D,1)` broadcasting.
  Spec: 3.3, 4.1–4.2, 5, 7.2, 12.1.

### A4.4 — Direct-method policy and survival/CDF defaults

- Files: `distributions/base.py`,
  registration integration, new private `distributions/_direct.py` for the
  shared value-policy machinery; `tests/unit/test_family_direct.py`; create
  `tests/tolerances.py` for numerical tolerances with the mandated scales.
- Implement `survival = exp(log_survival)`,
  `log_cdf = log(-expm1(log_survival))`, and
  `cumulative_hazard = -log_survival`.
  Keep the log-CDF calculation stable without probability floors.
- Apply section 4.5 at explicit registration to both inherited defaults and
  family overrides. Use one understandable policy wrapper, preserving raw
  family implementations for internal composition and avoiding recursion.
  Static time/batch mismatch raises `TypeError`; numerical time values follow
  the total-function policy. Do not confuse this with observation validation.
- For negative/NaN times, substitute a safe stopped-gradient reference before
  invoking family code; select policy outputs afterwards. Pass zero and positive
  infinity to the family. Preserve NaNs from family calculations without
  inventing generic domain checks for unknown custom parameters. Retain
  `reference_time()` as the required family hook; our choice to use it for
  safe negative/NaN-time substitution needs our own tests because the A4
  acceptance suite does not test that hook directly.
- Tests: analytical exponential-survival recipe; scalar/paired/draw broadcasting;
  all time-policy rows for these four methods, defaults and overrides; NaN
  parameter propagation;
  family code never receives negative/NaN times; finite parameter gradients
  through inactive branches and reference-input isolation. Test log-CDF tails
  against analytical/mpmath values in representable ranges.
- Docs: runnable primitive-family example in `docs/distributions.md`, direct
  method boundaries, dtype behavior and precision limits of the CDF default.
  Spec: 4.2, 4.5, 5, 7.2, 12.1.

### A4.5 — Autodiff density/hazard and A4 completion

- Files: `distributions/base.py`, `_direct.py`; new
  `tests/unit/test_family_autodiff.py`; extend `tests/tolerances.py` with
  justified derivative scales; finish `docs/distributions.md` and update
  `docs/playground.ipynb` with a short custom-family example.
- Compute the elementwise time derivative of log survival with forward-mode
  autodiff after broadcasting, holding parameters fixed. An all-ones time
  tangent yields the pointwise derivative; do not differentiate a batch sum
  or materialize a full Jacobian. Use it for generic log hazard and log density;
  do not recover hazard by subtracting two large log values. Keep the density
  calculation independent of the public `log_hazard` result: its zero-survival
  NaN policy must not contaminate a density that should be `-inf`.
- Supply `hazard = exp(log_hazard)`. Apply the same policy to density/hazard
  overrides. Detect zero survival from `log_survival == -inf`, not from
  `exp(log_survival) == 0`: finite log survival such as `-1000` still gives a
  valid generic hazard. Generic log hazard is NaN at true zero survival,
  including finite support endpoints; analytic overrides retain their values,
  and generic `hazard` uses the `log_hazard` override. Enforce
  `log_prob(+inf)=-inf` for valid
  parameters. Substitute safe inputs before logarithms and other potentially
  invalid inactive operations; use no clipping, epsilon or probability floor.
- Tests: hand-derived density/hazard for exponential and a smooth nonlinear
  cumulative-hazard recipe; bounded-support and infinite-time generic/override
  distinctions; zero hazard; eager/jit/vmap agreement; nested transformed
  construction. Cover zero-time density values for power-family shapes below,
  equal to and above one; negative-rate density NaNs and isolation of invalid
  batch entries; large cumulative hazards and tiny positive survival.
  Use `check_grads(order=2, modes=("fwd", "rev"))` on smooth parameter
  interiors in float64, with an explicit float32 skip for noisy second-order
  finite differences. Check analytical first derivatives and Hessians in both
  modes, including jit/vmap composition. Test finite gradients for the base's
  negative-time substitutions and use boundary-safe custom recipes for any
  additional boundary-gradient checks; do not promise that the base repairs
  arbitrary custom derivatives at zero or infinity. Built-in Weibull's `S(0)`
  shape-gradient requirement remains A5. Expected values come from algebra or mpmath,
  never from the implementation being tested. State the extra derivative order
  required of a primitive when density comes from autodiff.
- Finish the seven-method policy matrix and run the synced independent A4
  suite. Examples explain the extension contract and generic numerical limits;
  keep built-in grids, likelihood debug-NaN batches, R fixtures, wheel tests and
  performance baselines assigned to their later Slice A steps.
  Spec: 4.2, 4.5, 7.2–7.3, 10.5, 12.1–12.2.

## Acceptance coverage and checks at each stop

All paths below are under read-only `tests/spec/`. Keep implementer tests
independent: do not import `spec_families.py`, helpers or tolerances from there.

| Acceptance file | Planned coverage |
| --- | --- |
| `test_a4_registration.py` (25) | A4.2 registration, constructors and PyTrees; A4.3 nested fields; method-purity checks complete with A4.5 |
| `test_a4_parameters.py` (42) | A4.1 dtype helper, A4.2 public integration, A4.3 nested dtypes; tests using default density complete with A4.5 |
| `test_a4_indexing.py` (17) | A4.3 indexing and current-leaf batch shapes; density comparisons complete with A4.5 |
| `test_a4_nested.py` (14) | A4.3 broadcasting, components, dtypes and structure; density/vmap comparisons complete with A4.5 |
| `test_a4_generic_defaults.py` (23) | A4.4 survival/CDF/cumulative hazard; A4.5 hazard defaults |
| `test_a4_policy_enforcement.py` (20) | A4.4 primitive and derived policy; A4.5 density/hazard, overrides and inactive gradients |
| `test_a4_log_prob_default.py` (31) | A4.5 autodiff, shape/dtype, dispatch, gradients and Hessians |
| `test_a4_value_policy.py` (19) | A4.5 zero/negative/infinite/NaN times, invalid density and elementwise isolation |
| `test_a4_hazard_at_zero_survival.py` (12) | A4.5 true zero versus underflow, bounded support and analytic override dispatch |

These files cross substep boundaries. Use relevant explicit node selections
for intermediate feedback when their dependencies exist, in both dtype
processes, recording the selection; do not claim an entire file passes early,
change its step markers, or skip/xfail unfinished tests. The complete A4 gate
is required after A4.5.

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` for uv-based commands.
For A4.1–A4.4 run `scripts/check.sh --through A3` (all implementer tests
including the new ones, plus prior acceptance tests) and
`scripts/build_docs.sh`, both using separate x32/x64 processes. At A4.5 run
`scripts/check.sh --through A4` and the docs build. Run `git diff --check` at
every stop. The gate already includes Ruff formatting/lint and mypy.

Record exact commands, test counts, skips and failures. In particular, report
dtype-specific skips (float64 normalization cases and float64-only finite
differences) separately from failures.
Inspect the rendered distribution guide after its runnable examples are added.
No dependency additions, R execution or spec edits are planned.

Each `REVIEW.md` gives reading order, construction/PyTree/numerical invariants,
tests mapped to spec requirements, actual check output and open questions.
Each commit has an imperative subject and a body explaining why. Stop after
each commit; never commit to `main`. This planning change is left uncommitted.

Planning re-evaluation checks (2026-10-02): with the cache directory above,
`JAX_ENABLE_X64=1 SPEC_THROUGH=A4 uv run --locked --offline pytest --collect-only -qq tests/spec/test_a4*.py`
and the same command with `JAX_ENABLE_X64=0` both exited 0 and collected 203
A4 tests across nine files. Collection does not execute tests or establish
implementation correctness. No implementation tests, type/lint checks or docs
build were run for this plan-only update. `git diff --check` passed.

# A3 review fix-up: constructor consistency and metadata evidence (2026-10-01)

Status: approved and implemented on 2026-10-01; A3 gate passed. Source: `A3-review-handoff.md`,
reviewing `791517a`. One extra commit on the existing `A3` branch, targeting
under 300 changed lines including tests. Follow the handoff's explicit
plan/approval/implementation/review/stop sequence.

## Files and changes

- `src/survjax/_data_metadata.py`: replace recursive tracer detection
  with `jax.tree.leaves`; retain the separate module and existing kind
  precedence. Add the spec 3.3 explanation that `(0, +inf]` is right censored.
  Implement item 8: compute the six presence reductions with `jnp` on the
  device, stack the scalar results, and call `jax.device_get` once on that
  six-element array before converting to Python bools. Avoid transferring
  observation arrays to NumPy. Keep `may_contain` inside the compile-time
  evaluation block on the concrete path only; never pass it tracers.
- `src/survjax/data.py`: extend the bounds constructor's compile-time
  block through all issue calculations, making its six leaves concrete at
  trace time when supplied inputs are concrete. Add the requested JAX
  out-of-range clamping sentence to `__getitem__`'s docstring.
- `tests/unit/test_data_metadata.py`: add explicit scalar `(0, +inf]`
  precedence, NaN truncation and negative-entry flag/issue cases; check all
  six leaves inside jit for both concrete constructor forms; assert both
  forms retain the host-only `ValueError` (not `InvalidDataError`) when
  `raise_if_invalid()` is invoked inside jit; cover conservative metadata
  from vmap construction for both forms; extend concrete-inside-jit tests
  to compare every leaf with eager construction.
- `DEVIATIONS.md`: add D-012, step A3, spec 3.2, status open, documenting
  overlapping kind predicates at `(0, +inf]`, the existing right precedence
  following 3.3, and the proposed clarification for a future spec revision.
- `REVIEW.md`: overwrite with reading order, invariants, requirement/test
  mapping, actual check results and the A6 handoff below.

Public API: signatures and exports stay as implemented; `__getitem__`
documentation gains an explicit description of its existing behavior.
Spec coverage: 3.2, 3.3, 6.2, 12.1. No numerical likelihood implementation.

## Precise review wording and A6 handoff

Presence flags are not validity checks. Preserve the spec 3.2 invariant that
metadata includes normalized rows with issue bits. NaN bounds and negative
entry can leave truncation flags false despite invalid data; this does not
mean all invalid bounds necessarily leave those flags false. Invalidity is
determined from the stored issue bits.

Concrete construction requires one host synchronization to obtain static
Python flags (spec 3.2), transferring only six booleans. Slices inherit the
parent flags and perform no metadata synchronization. Existing metadata tests
and the new precedence/invalid-truncation tests cover item 8's unchanged
behavior; add no dedicated test or timing measurement. Large-N construction
timing belongs with the separate spec 12.3 CPU baseline.

For concrete supplied inputs, both constructors' six stored leaves are
evaluated at compile time inside jit. Subsequent `data.issues` decoding inside
jit is still traced, and its host checkpoint must fail explicitly there.
For traced supplied inputs, operations involving those inputs remain traced
and all six metadata flags are true; do not promise that constant-only
intermediate operations become tracers.

Record for A6:

- Reuse the normalized kind definitions from `_data_metadata.py` so row
  dispatch and static flags agree. `may_contain` returns static Python flags
  after a host transfer, not jit-safe row masks. A6 must plan any shared
  pure-JAX row classification needed for likelihood use; add no such helper
  in this fix-up.
- Issue-bearing rows must produce NaN and `invalid_data` from `_issue_bits`
  regardless of metadata pruning. Test NaN truncation with false flags.
- Resolve how Slice A detects unimplemented modes on traced data: all-true
  metadata alone cannot establish that a row uses an unsupported mode.
  The A6 plan must address row-value detection and explicit failure under
  transformations before implementing likelihoods.

## Verification and finish

Run `scripts/check.sh --through A3` in the existing two dtype processes,
with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`, and `git diff --check`.
Report actual results and any skips. Keep spec/tests/spec read-only and
exclude the existing notebook edit and maintainer-authored handoff from
staging. Commit as an extra A3 fix-up, without amend/rebase, then stop.

# Step A3: indexing and static observation metadata (2026-10-01)

Status: approved by maintainer and implemented on 2026-10-01; A3 gate passed.
Branch: existing maintainer-created `A3`. One implementation step and commit,
targeting about 300 changed lines including tests. If the implementation needs
a larger change, propose a split before expanding it. Earlier plans below are
history. The existing local edit to `docs/playground.ipynb` belongs to the
maintainer and is excluded from this step.

## Starting point and scope

A2 already supplies both pure constructors, the five normalized fields, the
sixth issue-bitfield leaf, diagnostics, and reconstruction without normalization.
`SurvivalData` currently has no `__getitem__`; its six auxiliary flags always
default to true. A3 completes indexing and value-derived concrete metadata.

Public API touched: add typed `SurvivalData.__getitem__(idx) -> SurvivalData`
and document its observation-axis indexing behavior. Both existing constructors
gain the specified metadata behavior without changing signatures or exports.
`data.issues` continues to decode masks and reduce `ok` over the resulting
object. Metadata remains private. No likelihood evaluation or branch pruning
is introduced; those belong to later steps/slices.

Spec coverage: 3.2 (indexing, six aligned leaves, auxiliary flags and
reconstruction), 6.2 (input tracing classification and indexed diagnostics),
12.1 (metadata, stored issues and transformation verification), and the data
portion of the Slice A gate in 12.2. Preserve the dtype policy in 3.3.

## Implementation and files

1. `src/survjax/data.py`
   - Apply `idx` to every array leaf, including the issue bits, and rebuild
     with the parent's auxiliary flags through the existing PyTree machinery.
     Do not reconstruct through a public constructor or reclassify a subset.
   - Support ordinary JAX array indexing: scalar/negative integer, slices,
     integer-array minibatches, concrete boolean masks, and multidimensional
     tuples. Under transformations, retain JAX's indexing constraints (for
     example, dynamic boolean masks do not have a fixed output shape).
   - Classify the supplied observation inputs at constructor entry, before
     normalization turns concrete inputs into tracers. Inspect every argument
     listed in section 6.2, including nested lists/tuples; `dtype` is not an
     observation input. Any tracer forces all six flags true.
   - For entirely concrete inputs, normalize and calculate flags in a
     compile-time evaluation context so concrete construction inside `jit`
     behaves like eager construction. Inputs already traced stay traced;
     unrelated traced function arguments do not affect this choice.
   - Pass the calculated flags explicitly when constructing either form.
     Keep `tree_unflatten` free of classification and validation so it also
     accepts JAX's reconstruction placeholders.
2. `src/survjax/_data_metadata.py` (new, small private module)
   - Hold the input-tracer check and concrete normalized-field classification.
     This responsibility is separate from row validation; `data.py` already
     exceeds 300 lines after the approved repr work, so do not add a general
     refactor or a second normalization implementation.
   - Reduce presence of exact/right/left/interval kinds and, independently,
     `left_trunc > 0` and `right_trunc < +inf` across all normalized rows,
     including invalid rows. Derive kinds from normalized endpoints, not the
     original keyword choices. Store only six Python bools in auxiliary data.
   - Convert only concrete values to host values; never convert a tracer.
3. `tests/unit/test_data_indexing.py` (new)
   - Cover indexing, preserved raw issue evidence, reconstruction, and
     transformed indexing as described below.
4. `tests/unit/test_data_metadata.py` (new)
   - Cover constructor classification, tree structures and compilation reuse
     as described below. Use existing dtype fixtures and exact assertions.
5. `REVIEW.md`
   - Overwrite after implementation with reading order, invariants, tests
     mapped to spec requirements, exact check results and remaining questions.
     Update this plan with the final scope if needed. Record any new spec
     question in `DEVIATIONS.md` and ask before choosing its behavior.

## Tests to add

- **Aligned indexing (3.2):** scalar, negative, strided/empty slice,
  repeated integer indices, concrete boolean selection and multidimensional
  indexing preserve each field's values, shape and dtype. Compare with
  explicitly indexed parent arrays, the identity required by the spec.
  Check agreement with `jax.tree.map` slicing and parent immutability.
- **Stored diagnostics (3.2, 6.2):** select rows whose normalized fields hide
  a raw-input fault (such as an invalid event indicator). Assert hand-derived
  issue masks survive selection and tree reconstruction. A valid subset of
  an invalid parent has scalar `ok=True`; an invalid subset has `ok=False`.
  Empty subsets retain parent flags and have no invalid rows.
- **Transformations (12.1):** eager, `jit` and `vmap` indexing agree for
  supported fixed-shape indices; decoded masks and scalar validity work
  inside `jit`. No numerical derivative checks are needed for this step.
- **Concrete flags (3.2):** both constructors yield the same flags for the
  same normalized data; separate kind and truncation changes alter the tree
  structure. Explicit default truncation matches omitted truncation. Include
  an interval flag with an infinite upper bound and rows with issue bits.
  Equal flags with different observation counts still give equal treedefs.
- **Input classification (3.2, 6.2):** concrete NumPy/Python inputs and
  captured concrete JAX arrays inside `jit` retain eager flags, including
  when another function argument is traced. An observation tracer, including
  one nested in a list/tuple or created by `jnp.asarray` inside `jit`, forces
  all flags true. Cover both constructor forms; compare leaves and issue
  masks separately because eager/traced metadata can intentionally differ.
- **Inheritance and compile counts (3.2, 12.1):** same shapes/dtypes/flags
  with different values trace once; different flags trace separately;
  equal-shape subsets of a mixed parent retain its flags and reuse a
  compilation. Traced construction and concrete data containing all kinds
  and both truncations share a tree structure and compilation.

## Verification and handoff

After approval and implementation, run `scripts/check.sh --through A3`:
the protected-path guard, Ruff formatting/lint, mypy, implementer tests and
independent acceptance tests through A3, with tests in separate float64 and
float32 processes. Run `git diff --check` as well. Record actual output and
any failure; do not claim A3 complete until the gate passes.

D-010 remains formally open, but the current acceptance test now checks
`batched[i].issues.ok`, matching the scalar-reduction contract. Its original
test discrepancy appears addressed; confirm through the A3 gate and leave
formal closure to the maintainer. Do not alter the read-only spec tests.

Finish with one commit on `A3` (proposed subject: `Add observation indexing
and static metadata`; body explains aligned issue preservation and compilation
reuse), the updated review, and a stop. Implementation/check evidence is in
`REVIEW.md`. The bounds constructor evaluates the normalized fields needed for
metadata at compile time; its issue-bit calculations remain pure JAX operations.
The implementation and tests total about 300 substantive changed lines;
normalization-block indentation accounts for additional diff lines.

# DOC-4 review fix-ups (2026-10-01)

Maintainer review of `1a15813`: print right censoring as its set
`(time, inf]` instead of `time+` (all censored rows now print as sets), and
make `docs/playground.ipynb` more informative. Extra commit on
`docs-startup`. Playground: short explanations between cells, the guide's
cohort (every censoring kind, truncation), stored fields, bad data with
`data.issues` and the host checkpoint (printed, so "Run all" completes),
`jax.jit`, and suggestions to try. No asserts, no saved outputs. Keep the
maintainer's `data` cell. Update repr tests and guide prose accordingly.

# Step DOC-4: readable reprs for SurvivalData and DataIssues (2026-10-01)

Requested by the maintainer after DOC-3 ("make a nice repr for SurvivalData
and DataIssues as well"). The spec does not cover repr. This replaces DOC-3's
"no summary repr" choice and adds no public API. One step, one commit on
`docs-startup`. Spec sections touched: 3.2 (private fields stay private), 6.2
(issue names), 6.3 (containers remain pytrees, unchanged).

## Behaviour

Both reprs read concrete values on the host. If any leaf is a JAX tracer, they
fall back to a short form with no value conversion (CLAUDE.md: never convert
tracers to NumPy), e.g. `SurvivalData(shape=(6,), dtype=float32, traced)`.

`SurvivalData`, eager: one line per observation in R-like notation, plus the
eligibility window when it is not the default `(0, inf]`. Invalid rows are
printed as stored, with a footer pointing at `.issues`:

```text
SurvivalData: 6 observations, float64
  0  4
  1  12+
  2  7
  3  9+
  4  (6, 9]   window (2, inf]
  5  (0, 3]   window (2, inf]
```

- Notation: exact `t`, right censored `t+` (as in R's print), every other
  censored row as its stored set `(lower, upper]`, so left censoring prints
  `(0, 3]` (maintainer's choice). Numbers use `:g`.
- More than 20 rows: first 10, `...`, last 5. N-d data: header shows the
  shape and rows show their index tuple.
- Footer only when non-empty: `issues: 4 invalid rows, 1 uninformative
  (see .issues)`.

`DataIssues`, eager: the overall verdict, then only the rules that fire,
with the row count and first index (the same wording as
`raise_if_invalid`):

```text
DataIssues: 4 of 5 rows invalid
  invalid_time            1 row, first index (0,)
  missing_time2           1 row, first index (1,)
  invalid_interval        1 row, first index (2,)
  censored_before_entry   1 row, first index (3,)
  uninformative           1 row, first index (4,)  (not an error)
```

A valid result prints `DataIssues: all 6 rows valid`.

## Files

Approved with changes: each repr lives next to its class, not in a separate
module, accepting files somewhat over 300 lines.

- `data.py`: `@dataclass(..., repr=False)` plus `__repr__` and two small
  formatting functions. The DOC-3 `field(repr=False)` declarations become
  redundant and are removed.
- `diagnostics.py`: the same for `DataIssues`. `raise_if_invalid` and the
  repr share one per-rule count helper; the exception message is unchanged.
- `tests/unit/test_data_repr.py`: replace the DOC-3 tests with hand-written
  expected strings for: each observation kind; truncation window; invalid
  footer; long-data elision; 2-d index tuples; traced fallback for both
  classes (inside `jax.jit`); valid and invalid `DataIssues`. Keep the
  six-leaves check.
- `docs/survival-data.md`: no code changes needed (cells already end with
  `data`); reread the rendered page and update prose that describes the
  printed repr. `REVIEW.md` as usual.

Checks: `scripts/check.sh --through A2`, `scripts/build_docs.sh`, `git diff
--check`. Expected diff about 250 lines.

## Decisions

- Left censoring prints as the stored set `(0, 3]`.
- No precision information beyond the dtype in the header.

# Step DOC-3: tutorial-style SurvivalData docs and a clean repr (2026-10-01)

Approved by the maintainer after reviewing DOC-2 (`726e561`): the guide reads
like a unit test, not a tutorial. One step, one commit, on a
maintainer-created branch (the worktree is a detached HEAD at `docs-startup`).
Spec sections: 3.1-3.3, 6.2, 10.5.

## Repr

- `src/survjax/data.py`: declare `_issue_bits` and `_may_contain` with
  `dataclasses.field(repr=False)`, so the repr shows only the five public
  arrays (spec 3.2 keeps the others private). No summary repr. `DataIssues`
  is unchanged. Pytree flattening and `__init__` are unchanged.
- `tests/unit/test_data_repr.py`: the eager repr names the five public
  fields and omits the private ones; `repr` of data built inside `jax.jit`
  does not raise and omits them; `jax.tree_util.tree_leaves` still gives six
  leaves.

## Docs: rewrite `docs/survival-data.md` as a tutorial

Keep the facts of DOC-2; change the form.

- Open with a short note: what works now (construction, diagnostics) and what
  doesn't yet (likelihoods). No closing scope section.
- Use one running synthetic story: a small cohort (about six patients,
  follow-up in months) built up section by section. Each section answers a
  reader task: "time and status, as in R"; "the event happened between two
  visits"; "known only to have occurred before the first visit"; "patients
  entered late (truncation)"; "finding bad rows"; "inside `jax.jit`".
- Show results instead of asserting them: end cells with the displayed value
  (`data`, `data.upper`, `data.issues.invalid_rows`). Use asserts only where a
  silent check helps the reader, if at all. The build failing on execution
  errors is the executable check (spec 10.5).
- Show real errors: call `raise_if_invalid()` and the misuse examples in cells
  tagged `raises-exception`, so the rendered page shows the message. No
  `try/except/else: raise AssertionError` scaffolding.
- Explain why before how: a sentence or two on what each censoring kind means
  for the patient before the call that encodes it. Introduce the normalized
  fields when the first output shows them, not as an opening list of rules.
- Present the R comparison as a table (`Surv(...)` call next to the survjax
  call, plus a notes column). Re-check each claim against the official `Surv`
  help page. In particular, the "(start, end]" wording there may refer to
  counting-process data, not interval censoring; drop the claim unless the
  page supports it.
- Dtype behavior: one short section with a single example, not a test of both
  modes.

`docs/playground.ipynb`: remove the empty trailing cell and reset the
kernelspec and `language_info` metadata to the DOC-1 minimal form. Keep the
maintainer's `data` cell, which now shows the clean repr. Have it follow the
same cohort as the guide, without asserts. `docs/index.md`: update the guide's
one-line description if needed.

## Checks and review

`scripts/build_docs.sh` (both dtype modes), `scripts/check.sh --through A2`,
`git diff --check`. Read the rendered x64 guide end to end and report in
`REVIEW.md` whether every output shown matches its surrounding prose. The
guide rewrite replaces most of a 176-line page, so the diff may exceed about
300 lines. If it does, say so in `REVIEW.md` rather than splitting the step.

# Documentation startup on `docs-startup` (2026-09-30)

The maintainer requested two separate tasks following specification section
10.5. Each is one reviewable step and one commit. Stop after the first step's
`REVIEW.md` and await instruction before starting the second. This plan is for
approval before either implementation step; the earlier plans below are history.

## Step DOC-1: local documentation process

Spec sections: 10.2, 10.3, 10.5, and the documentation item in 12.2.

Files to create or change:

- `pyproject.toml`, `uv.lock`: add a locked optional documentation group with
  Sphinx, MyST-NB, JupyterLab and a Python kernel. Keep core dependencies
  unchanged.
- `docs/conf.py`, `docs/index.md`: minimal Sphinx site for Markdown and
  notebooks, navigation, and a clear statement of the currently implemented
  subset. Include the existing `docs/simulation-guide.md` only if it can be
  visibly labelled as illustrative and unverified without build warnings.
- `docs/playground.ipynb`: a small synthetic, editable notebook using the
  current public `SurvivalData` API; no saved outputs.
- `scripts/build_docs.sh`: one documented, non-interactive command that builds
  HTML from a clean checkout, executes runnable content afresh in separate
  float32 and float64 processes, and fails on warnings or execution errors.
- `.gitignore`: exclude generated site, notebook execution artifacts and
  caches. `REVIEW.md`: report exact commands, outcomes and rendered-page
  inspection. Update this plan if implementation reveals a necessary split.

Public API: none. Tests: the build command itself is the end-to-end check;
verify it from a clean working tree and inspect the rendered index and notebook.
Run the existing A2 gate and `git diff --check` if dependency or tooling edits
affect those checks. No network, R, real data or demo dependencies at doc build
time. Do not publish or host the site.

## Step DOC-2: A1–A2 documentation

Spec sections: 3.1–3.3, 6.2, 10.2, 10.5, and 12.2. Cover implemented behavior
only; label A3 indexing/static-metadata guarantees and later likelihood modes
as planned. Revise this step plan, if needed, before writing its code and seek
approval under the working agreement.

The maintainer's follow-up authorizes DOC-2 and also asks for the commands to
open the rendered HTML. Add those commands to `docs/index.md`. A locally edited
playground currently contains exploratory cells and saved outputs: preserve its
cell code, clear saved outputs for the committed source, and extend it with the
new guide's working API. This is still the one documentation step below.

Files to create or change:

- `docs/survival-data.md`: executable synthetic examples for event/status,
  explicit right/left/interval flags, mixed data, `time2`, bounds, and
  truncation. Show normalized fields and the exactness rule for bounds.
- In the same page, explain issue masks, `invalid_rows`, `ok`, valid but
  uninformative rows, structural `TypeError`/dtype `ValueError`, and explicit
  host `InvalidDataError`. Demonstrate ordinary construction and diagnostic
  access inside `jax.jit`, with the host checkpoint after compiled execution.
- In the same page, compare the flag and bounds conventions to R's
  `survival::Surv` forms, including event coding, interval endpoints and
  counting-process `time2`. Check comparison wording against official R
  documentation. Claim no numerical agreement without R-generated fixtures.
- `docs/index.md` and `docs/playground.ipynb`: link the guide and make the
  notebook a short interactive path through the same working API.
- `REVIEW.md`: reading order, invariants, exact example/build outcomes in both
  dtypes, source of R-convention claims, known gaps and rendered-page review.

Public API: none. Tests: executable examples through `scripts/build_docs.sh`
in both dtype modes, plus the existing A2 gate and `git diff --check`. Examples
must check their stated outputs rather than merely run. No new unit tests that
repeat documentation examples. Keep this step near 300 changed lines; split it
before implementation if the reviewed plan requires more.

# D-006 diagnostic compatibility decision (2026-09-30)

The maintainer approved retiring `TracedInputError`, keeping `ValueError` for a
traced `DataIssues.raise_if_invalid()` call, and exposing only the
`data.issues` property (no callable `data.issues()` alias). On the
maintainer-created `spec-deviations` branch, update the normative contract in
`spec/SPECIFICATION.md` sections 6.2, 10.2 and 13, its review/decision history,
and D-006 in `DEVIATIONS.md`. Leave D-010 and the other deviation statuses
untouched. This explicit maintainer request authorizes the spec edit.

Public API: clarify that only `InvalidDataError` is exported for data
diagnostics; no code or behavior changes. No tests are added for this
documentation step. Review the diff for internal consistency and run
`git diff --check`. Overwrite `REVIEW.md` with the reading order, invariants,
evidence and uncertainties. Commit this one step on `spec-deviations` and stop.

# A2 constructor API migration — specification revision 9

## Pre-merge bookkeeping step (2026-09-30)

The maintainer requests a clarification of the non-exact equal-endpoint case
and cleanup of deviation IDs/statuses. Change only the `from_bounds` docstring
in `src/survjax/data.py`, the branch D-006 references in
`src/survjax/_data_validation.py` and
`tests/unit/test_data_construction.py`, and `DEVIATIONS.md`. Rename the branch
D-006 to D-011 and mark D-011, D-007, D-008 and D-009 as superseded by
revision 9, leaving final closure to the maintainer. Update `REVIEW.md` and
commit this small step on `a2-constructors`.

Public API: unchanged. Spec: sections 3.1, 6.2 and 13. Tests: the existing
independent A2 equal-endpoint acceptance cases and the full A2 gate; no new
test is needed for a docstring and bookkeeping change. Do not edit the
read-only specification or independent tests.

Branch: `a2-constructors`. A2-R1 exception alignment was committed as
`c6ef7e3`. The maintainer's 2026-09-30 request authorizes the remaining
constructor API migration: remove issue-returning constructor methods, accept
eager and traced observation inputs in the two ordinary forms, and expose
diagnostics through `data.issues`.

## Current step

Spec sections: 3.1–3.3, 6.2–6.3, 10.2 and 12.1–12.2.

Files and public API:

- `src/survjax/data.py`: retain exactly `SurvivalData(...)` and
  `SurvivalData.from_bounds(...)` as construction methods; remove automatic
  host validation and the `with_issues` method; add a typed `issues` property
  decoding the stored issue bitfield. No `from_bounds_with_issues` method.
- `src/survjax/_data_validation.py`: remove obsolete host-only constructor
  validation and references to the removed method, preserving normalization,
  dtype policy, argument-misuse checks and issue attribution.
- Own tests in `tests/unit/`: migrate callers to ordinary construction and
  `data.issues`; check eager, `jit` and `vmap` behavior, invalid-value masks,
  explicit host checkpoints, traced inputs, tree reconstruction and the absent
  methods. Keep tests with code in the same commit.
- `REVIEW.md`: map checks to requirements and report exact gate results.

The revision 9 specification and its acceptance criteria already require this
API; `spec/GOALS.md`, `spec/SPECIFICATION.md` and `tests/spec/` are read-only.
There are no current examples or docs modules to migrate. Historical deviation
entries retain their original wording and status.

## Verification

Run `scripts/check.sh --through A2` to cover Ruff, mypy, own tests and the
independent A2 tests in both dtype modes. Use focused A3 acceptance tests for
the removed-method and tracing checks, without claiming A3 complete. Run
`git diff --check`. If an independent test conflicts with the specification,
record the issue in `DEVIATIONS.md` and leave the test unchanged.

## Later handoff

A3 owns indexing and slicing, value-derived concrete static metadata and
compile-count tests (spec 3.2). The current all-true metadata is only a
conservative interim value. A6 owns likelihood issue propagation and explicit
failure for unimplemented observation modes. Neither is part of this step.

The traced host-checkpoint exception subtype and legacy diagnostic compatibility
remain open in spec section 13 and `DEVIATIONS.md`. This step preserves the
existing traced-checkpoint failure and does not introduce a callable alias.

## Performance amendment on `spec-speed` (2026-09-30)

The maintainer requests adapting the earlier performance draft to the current
specification at `9a403a3`, including revision 10 diagnostic compatibility.
This authorizes edits to the normally read-only specification and goals.

- Update `spec/GOALS.md` with speed, memory efficiency and acceleration.
- Update `spec/SPECIFICATION.md` sections 1, 7.2, 9.2, 10.3, 12 and 14:
  retain pure ordinary construction, `data.issues`, stored-issue propagation,
  conservative traced metadata and simulation fit-skipping semantics. Measure
  prepared-data evaluation and traced construction separately. Keep GPU checks
  and scalable metric algorithms optional, with a small CPU baseline in Slice A.
- Leave the final amendment revision number for acceptance: the current header
  says revision 9 while the history includes revision 10. No API changes,
  implementation edits, new dependencies or changes to independent tests.
- Overwrite `REVIEW.md` with reasoning, reading order, preserved contracts,
  documentation checks and unmeasured performance requirements.
- Checks: whitespace, Markdown/section references, allowed diff scope and exact
  preservation of unrelated statistical/API sections and example code. Runtime
  tests and benchmarks are not run for this documentation-only step.
- Commit this one documentation step on `spec-speed`; do not merge or publish.

## Illustrative simulation guide on `spec-speed` (2026-09-30)

The maintainer approved moving the simulation workflow out of the normative
contract and removing recipe-specific acceptance requirements. This authorizes
the following documentation step, including edits to the specification.

- Move the detailed recipe into `docs/simulation-guide.md`, explicitly
  illustrative and unverified; retain `lax.map` as one example choice.
- Update specification sections 6.2, 11.4, 12.1, 12.3 and 14: retain library
  guarantees, check compiled composition directly, and leave study workflow,
  external solver and loop selection to callers and measurements.
- Public API, goals, implementation, dependencies and tests remain unchanged.
  Add no runtime tests; check whitespace, local links, moved example code and
  preservation of unrelated contracts. Record results in `REVIEW.md`.
- Commit this single documentation step on `spec-speed` and stop.
