# Deviations and spec questions

## D-025 Plain comparisons versus subnormal acceptance expectations
- Step: A6.1 plain-comparison follow-up planning (2026-10-19).
- Spec: sections 4.7 and 12.1; independent
  test_a6_constraints.py::test_unit_interval_includes_zero_and_one.
- Spec says: nonnegative includes finite x>=0, unit_interval includes
  0<=x<=1, and constraint tests cover eager/jit/vmap agreement. The independent
  test requires nextafter(0,-inf) to fail unit_interval in every mode.
- Did / propose: the maintainer requests removal of jnp.frexp and has chosen
  JAX parameter-dtype/backend behavior. Propose plain comparisons, retaining
  inclusive nonnegative pending clarification of "greater than 0". A CPU
  probe in both dtypes shows finite x>=0 accepts the smallest negative
  subnormal eagerly/vmap but rejects it under jit. Removing the workaround
  therefore conflicts with the independent expectation and the stronger
  transformation guarantee at subnormal boundaries. Report the acceptance
  failure and preserve the independent test; ask the maintainer to approve
  this explicit backend-semantic deviation with the implementation plan.
- Revision-30 alignment note (2026-10-20): section 4.7 now explicitly allows
  backend-dependent classification of subnormal values/bounds, including at
  zero, and requires ordinary JAX comparisons with no added handling. The
  nonnegative domain remains finite x>=0. The normative question is settled;
  implementation and independent-test synchronization are pending. Formal
  closure remains the maintainer's.
- Implementation follow-up (2026-10-20): removed frexp and the own strict
  subnormal-sign guarantee under revision 30. The updated tester module at
  f527f75 uses normal neighbours around zero and passes alongside the focused
  own tests in both dtypes (98 passed per process). No independent test was
  changed here. Formal closure remains the maintainer's.
- Status: open

## D-024 Precision and subnormal semantics of constraint bounds
- Step: A6.1 review check (2026-10-19), reviewer Q1.
- Spec: section 4.7, unchanged in revisions 28–29.
- Spec says: bound factories take finite Python numbers and compare x>a,
  a<x<b or a<=x<=b, without specifying the precision of the Python bound or
  the required handling of nonzero subnormal bounds on a flushing backend.
- Did / propose: preserve the A6.1 implementation pending maintainer decision.
  Zero-bound comparisons use frexp for signed subnormals; nonzero bounds use
  JAX's parameter-dtype comparisons. The reviewer reports float32 cases
  greater_than(-1e-40)(0) and greater_than(0.1)(float32(0.1)) returning False,
  and notes NumPyro agrees. Ask whether the contract adopts parameter-dtype
  bound conversion and backend comparison behavior, or requires mathematical
  comparison against the Python bound with subnormal guarantees. Do not
  silently broaden the zero-bound workaround or change the spec.
- Maintainer decision (2026-10-19, fix-up approval): follow JAX's parameter-
  dtype/backend comparison behavior. Preserve the current comparisons and
  existing zero-bound frexp handling. No spec edit is requested; formal
  closure of this entry remains with the maintainer.
- Revision-30 alignment note (2026-10-20): section 4.7 and decision 0042 now
  record the maintainer's chosen parameter-dtype/backend behavior and specify
  no extra handling even at zero bounds. This supersedes the earlier note
  preserving zero-bound frexp handling. Removing it awaits the maintainer's
  test sync and instruction; formal closure remains the maintainer's.
- Implementation follow-up (2026-10-20): all constraints now use ordinary
  JAX comparisons; bound factories explicitly cast bounds to the parameter
  dtype. The zero-bound special cases have been removed. The focused
  revision-30 constraint checks pass in both modes. Formal closure remains
  the maintainer's.
- Status: open

## D-023 Explicit unsupported likelihood failures under tracing
- Step: A6 planning, revision 28 (2026-10-19)
- Spec: sections 3.2, 6.1–6.3, 7.1, 12.2; independent Q-037.
- Spec says: unsupported observation modes fail explicitly; traced data has
  all-true may-contain metadata, while raw/traced supported exact/right data
  must evaluate. The four LikelihoodIssues masks have no unsupported-mode
  field. Valid left/interval/truncated data is not invalid_data, and absence
  of an implementation is not numerical_failure or invalid_conditioning.
- Did / propose: no evaluator implemented. Ask the maintainer to specify the
  failure and diagnostics policy before A6.4. A possible policy is NaN only on
  unsupported rows in eager and transformed calls, retaining supported rows,
  with the existing four masks reserved for their stated meanings; this means
  NaN without an explanatory issue on unsupported valid data and needs an
  explicit decision. Independent tests permit eager NotImplementedError or
  row NaN, require traced row NaN, and deliberately do not settle the masks.
  Do not reject data solely because traced metadata says it may be unsupported.
- Revision-29 review note (2026-10-19): LikelihoodIssues and the _with_info
  entry point are removed. The four-mask premise above is historical; the
  remaining question is whether unsupported valid rows raise or give NaN,
  and how callers distinguish them from bad-data/bad-parameter rows and the
  unexplained NaN remainder of section 7.1's triage recipe. This still needs
  a maintainer decision before A6.4; no evaluator has been implemented.
- Status: open

## D-022 Log-CDF upper tail rounds to zero before S underflows
- Step: A5.4 review, 2026-10-14 (found after D-021)
- Spec: sections 4.2, 4.3, 7.1 and 12.1 ("stable tails"), revision 24.
- Spec says: section 4.2 defines the default `log_cdf` as
  `log(-expm1(log S))`, "accurate until |log S| underflows". Section 7.1
  says precision loss is never relabelled as mathematical zero. Section 4.3's
  HazardFamily recipe addresses only the lower tail.
- Observed: `log(-expm1(-H))` loses relative precision when H is moderately
  large. Once `exp(-H)` is below eps, `-expm1(-H)` rounds to exactly 1, so
  `log_cdf` returns 0 (F=1) although the true value `log F ~ -exp(-H)` is
  representable. Measured against mpmath (60 digits) with
  `Weibull(shape=1, scale=1)`, i.e. H=t:
  float32: H~20.1 gives 0 (true -1.89e-9), H~33.1 gives 0 (true -4.15e-15);
  float64: H~20.1 gives relative error 2.8e-8, H~33.1 gives 1.0e-2
  (-4.108e-15 vs -4.151e-15), H~54.6 gives 0 (true -1.94e-24).
  The absolute error stays below eps, so the Slice A exact and right-censored
  likelihoods (which do not use `log_cdf`) are unaffected. Later
  `log_interval_prob` with `lower=0` (section 4.2) and any code that takes
  `log1mexp` of `log_cdf` would inherit the loss.
- Affected: the D-021 overrides in `exponential.py` and `weibull.py` (their
  standard branch), and the generic default in `_survival_methods.py`
  (`log_cdf`), which every registered family without an override uses.
- Propose: one private `log1mexp`-style helper in an underscore module
  (Maechler 2012): for `x = log S <= 0`, use `log(-expm1(x))` when
  `x > -log 2` and `log1p(-exp(x))` otherwise, with safe operands substituted
  in both branches before evaluation (spec 7.2). Use it in both built-in
  overrides (which then reduce to the D-021 lower-tail switch plus this
  helper) and in the generic default. This changes the base, so do it as its
  own small step with mpmath tail tests in both dtype modes, before A5.5 or
  as the first step after it; not as an A5.4 fix-up.
- Implementation evidence (2026-10-15, A5.5 follow-up): the approved shared
  helper now serves the generic default and both built-ins, retaining D-021.
  The new upper-tail regression failed on all three paths in both dtypes
  before the fix; all 27 new cases and the full through-A5 gate now pass in
  both modes. See REVIEW.md for exact checks. Closure remains with the maintainer.
- Spec alignment (2026-10-18): revision 25 (`1e9011c`) makes the shared
  two-branch helper and upper-tail relative-precision checks normative in
  sections 4.1–4.3, 7.2 and 12.1. No new tests were run during alignment;
  formal closure remains with the maintainer.
- Status: open

## D-021 Built-in log-CDF lower-tail switch
- Step: A5.4 review fix-up, 2026-10-14
- Spec: sections 4.2, 4.3, 7.1 and 7.2, revision 24.
- Spec says: section 4.2 permits the default `log(-expm1(log S))` until
  `log S` underflows; section 4.3 prescribes switching to `log H` for a
  `HazardFamily` when H is tiny. Section 7.1 says precision loss must not be
  relabelled as zero mass.
- Did / propose: give Exponential and Weibull the same dtype-epsilon switch,
  computing `log H` directly in the lower tail and substituting a safe operand
  before evaluating the standard `expm1` branch. This extends the HazardFamily
  numerical approach to built-ins before that base class exists, so their
  lower-tail values remain representable even when forming H underflows.
- Spec alignment (2026-10-18): revision 25 (`1e9011c`), section 4.1, now
  explicitly requires the built-ins' lower-tail calculation from `log H`;
  the switch point remains an implementation choice preserving relative
  precision. Formal closure remains with the maintainer.
- Status: open

## D-020 Mixed-time Weibull density acceptance expectation omits log(shape)
- Step: A5.4 implementation, 2026-10-13
- Spec: sections 2, 4.1 and 4.5, revision 23 (numerical rules unchanged
  since revision 22); independent test cites revision 21.
- Spec says: S(t)=exp(-(t/b)**k), hence
  f(t)=(k/b)*(t/b)**(k-1)*exp(-(t/b)**k). At k=b=2 and t=1,
  log f=log(2)-log(2)+log(1/2)-1/4=-0.9431471805599453.
- Observed: `tests/spec/test_a5_boundaries.py`'s
  `test_weibull_policy_applies_elementwise_in_one_call` expects
  `2*log(0.5)-0.25=-1.6362943611198906`, omitting log(k).
  The discrepancy is exactly log(2), far beyond either dtype tolerance.
  The other closed-form acceptance tests include the log(k) term.
- Did / propose: keep the specified density formula. Do not edit, skip or
  xfail the independent test. Ask the maintainer to have the independent
  author correct this expectation and its hand-derivation docstring.
  Keep this entry open until the maintainer resolves it.
- Evidence: the complete five-module non-grid A5 selection exits 1 in each
  dtype mode, with this sole failure. x64: 321 passed, 1 failed; x32:
  311 passed, 10 skipped, 1 failed. All 18 own Weibull cases pass in each
  mode, including full density Jacobians/Hessians at k=2, b=2, t=1.
- Revision-24 follow-up (2026-10-14): the synchronized boundary test now
  includes `log(shape)` in its expected Weibull density. The full A5 run no
  longer reports D-020; the non-grid A5 acceptance modules pass in both dtype
  modes. The maintainer retains closure; this entry remains open.
- Status: open

## D-019 Natural parameter leaves versus overflow-safe log constructors
- Step: A5 planning
- Spec: section 4.1, revision 21; dtype policy in 3.3
- Spec says: natural parameters are stored as leaves under constructor names;
  `Exponential.from_log` and `Weibull.from_log` compute in log space internally,
  avoid float32 exponentiation overflow and return the same class.
- Conflict: float32 `exp(100)` cannot store a finite natural scale, and an
  underflowing natural rate loses its original finite log rate. Keeping only
  natural leaves cannot preserve these inputs for later log-space calculations.
- Did / propose: no implementation. Ask the maintainer to clarify retained
  log-parameter leaves for `from_log`, natural-field access, PyTree structure
  and normalization; alternatively narrow the guarantee to representable
  natural parameters. Do not choose a representation or edit the spec here.
- Planning decision (2026-10-08): the maintainer selected retained log leaves
  and a spec clarification before implementation. Details of field access,
  PyTree structure and normalization remain to be specified.
- Follow-up evidence (2026-10-11): specification revision 22, section 4.1,
  supersedes that direction. `from_log` exponentiates inputs and stores natural
  leaves; it keeps no log-space representation and makes no overflow-avoidance
  claim. Independent Q-028 and `test_a5_from_log.py` at tester commit `7f1d6df`
  follow the new contract. The revised A5 plan implements that contract; no
  representation decision remains pending. Formal closure stays with the maintainer.
- Status: open

## D-016 NaN parameter gradients of finite values at `t=+inf`
- Step: A4.4 review; decision deferred to A5
- Spec: section 4.5 (boundary rows and "Boundary values must also avoid NaN
  parameter gradients where the value is finite"), revision 19
- Spec says: at `t=+inf` the family supplies the value; finite boundary
  values have no NaN parameter gradients.
- Observed: for a correct family (`log S = -rate*t`), the generic defaults
  give `survival(+inf) = 0` and `log_cdf(+inf) = 0` with NaN `d/d rate`,
  because `exp(-inf) * (-inf)` is `0 * inf`. One `+inf` entry makes the
  gradient of a sum NaN. `log_survival` and `cumulative_hazard` at `+inf`
  are infinite, so the rule does not apply to them.
- Why not fixed now: masking the output cannot remove a NaN that arises
  inside the family's derivative. Applying `stop_gradient` at `+inf`
  would be wrong for improper (cure) families, where `S(+inf)` depends on
  the parameters. A generic fix needs a second family evaluation on a safe
  input for the gradient path. The A5 built-in overrides (`exp(-r*t)`,
  Weibull) face the same `0 * inf`.
- Did / propose: no change in A4.4. Decide in A5 with Exponential and
  Weibull as concrete cases. Options: a two-pass guard in the base,
  analytic built-in overrides plus a documented generic limitation, or a
  spec clarification limiting the rule to `t=0`. Likelihoods at A6 use
  `log S` at finite times and are unaffected.
- A5 planning decision (2026-10-08): include the generic base fix preserving
  parameter-dependent finite limits, alongside guarded built-in formulas.
  No implementation has been performed as part of planning.
- Follow-up evidence (2026-10-11): revision 22 section 4.5 explicitly requires
  the actual finite boundary derivatives at zero and infinity for built-ins,
  excluding derivatives across discontinuous parameter limits. Custom-family
  guarantees depend on the author's primitive, with a documented extension
  recipe. The revised A5 plan retains the selected generic improvement but
  does not promise repair of arbitrary custom primitives. Tester Q-029 and
  the new infinity-gradient cases in `test_a5_boundaries.py` cover this reading.
- Review follow-up (2026-10-12, A5.2 F-1): A5.2 fixes generic survival/log-CDF,
  but shared-parameter calls on [1,+inf] still give NaN finite-entry reverse
  derivatives for log density, log survival, cumulative hazard and quadratic
  generic log hazard. Nonfinite boundary derivatives are outside the guarantee;
  this finding concerns contamination of the finite interior entry. PLAN.md
  proposes A5.2b to isolate those entries, preserve cure limits and analytic
  overrides, and cover the generic hazard path as well. No numerical follow-up
  implementation has been performed; approval is pending.
- A5.2b implementation (2026-10-13): the maintainer approved the follow-up.
  The policy wrapper now substitutes before density evaluation at all positive
  infinities, and before primitive/generic evaluation at -inf survival limits.
  Original probe status determines density NaN precedence; finite cure limits
  and analytic overrides remain active. New mixed-entry tests cover forward
  and reverse derivatives and finite-component Hessians. Full documentation
  is deferred to the end of A5 by maintainer instruction. No guarantee is
  added for derivatives of infinite/NaN outputs or unsafe custom overrides.
- Spec alignment (2026-10-18): revisions 26–27 supersede the primitive-owned
  survival limits described above. Sections 4.2/4.5 now give the base
  responsibility for survival/CDF/cumulative-hazard boundaries for every
  registered family, using finite positive substitutes and the differentiable
  `log_survival_at_infinity()` hook. Implementation is a separate future
  step; this note does not claim conformance or resolve the entry.
- A5.6 implementation (2026-10-18): the base now supplies zero/infinity
  survival limits using finite positive substitutes and the differentiable
  infinity hook. The full through-A5 gate passes in both dtype modes,
  including the updated independent boundary suite. See REVIEW.md for exact
  results and derivative coverage; maintainer closure remains pending.
- Status: open (maintainer decision 2026-10-03: defer to A5)

## D-017 Dtype policy for direct-method times
- Step: A4.4 review
- Spec: sections 3.3 and 4.5, revision 19
- Spec says: 3.3 sets the dtype policy for observation fields and
  parameters (reject boolean/complex; warn once when float64 arrays are
  converted under x32). It does not cover the `t` argument of direct
  methods.
- Did / propose (maintainer decision 2026-10-03): direct-method times
  reject boolean and complex dtypes with TypeError and otherwise follow JAX
  conversion and promotion, without the float64-under-x32 warning. Reasons:
  NumPy grids are float64 by default, so the warning would fire on nearly
  every evaluation; output precision is already set by the parameters;
  and `t` is a transient argument, not stored data. Proposed spec sentence
  for 4.5.
- Status: resolved by maintainer, 2026-10-08; specification revisions 20–21,
  section 4.5 now states the direct-time dtype policy.

## D-018 Precedence of NaN family values over the negative-time policy
- Step: A4.4 review
- Spec: sections 4.1, 4.5 and 7.1, revision 19; D-013
- Spec says: 4.5 lists both "`t<0` -> policy value" and "invalid parameters
  -> NaN" without precedence; 4.1 says invalid parameters surface through
  NaN results.
- Did / propose (maintainer decision 2026-10-03): for `t<0` and NaN `t`, if
  the family's value at the substituted input is NaN, the result is NaN;
  otherwise the policy value applies. This detects NaN-producing parameters
  only; no validity domain is inferred (D-013). Valid parameters keep zero
  policy gradients. The same rule applies to `log_prob` at A4.5. Proposed
  spec wording for 4.5: "For `t<0` and NaN `t`, if the family's value at the
  substituted input is NaN, the result is NaN (this is how invalid
  parameters surface; no validity domain is inferred); otherwise the policy
  value applies." The synced A4 acceptance tests now cover this policy.
- Status: resolved by maintainer, 2026-10-08; specification revisions 20–21,
  section 4.5 now states NaN precedence and the unspecified substituted input.

D-017/D-018 follow-up evidence (2026-10-08): specification revisions 20–21
now state these policies in section 4.5. New acceptance files cover time dtypes
and NaN precedence. Both entries are closed above; their original gap and
decision descriptions are retained as history.

## D-015 Error policy for malformed nested distribution fields
- Step: A4.3 review fixes
- Spec: sections 4.2 and 6.2, revision 19
- Spec says: `distribution()` holds a nested distribution; family registration
  is explicit. Static misuse follows TypeError conventions, but the nested-field
  paragraph does not explicitly assign an error class for malformed values.
- Did / propose: the maintainer requested implementation of review F-1 in
  `../review/slices/A4.3.md`: require an instance of an explicitly registered
  family and raise a field-named TypeError otherwise. Implement this before
  reading nested batch shapes, including rejecting unregistered subclasses
  of registered parents. No parameter-value validation or spec change.
- Status: open

## D-014 Unmarked family fields and inherited constructors
- Step: A4.2 review fix-up
- Spec: sections 4.1–4.2, revision 19; dtype warning attribution in 3.3
- Spec says: frozen dataclass families use parameter/static/nested field helpers
  and preserve dynamic parameters through PyTree transformations. It does not
  define plain dataclass fields or registration of subclasses that inherit an
  already-registered initializer without being re-decorated as a dataclass.
- Did / propose: A4.2 silently omitted unmarked fields from flatten/unflatten;
  the review demonstrates changed defaults or missing attributes after jit.
  It also allowed wrapping an inherited registered initializer twice, which
  misattributes x32 warnings. Propose field-named TypeError for unmarked fields
  at registration, and rejection of classes without their own initializer.
  The maintainer approved both rejections on 2026-10-02; this fix-up
  implements them. The specification is unchanged; this entry remains open.
- Status: open

## D-013 Parameter diagnostic return contract
- Step: A4 planning; deferred to A6 after acceptance-suite sync
- Spec: sections 4.1, 4.2 and 4.5, revision 19
- Spec says: invalid parameters surface through `parameter_issues()` and NaN
  results; its generic default reports no known violation. The specification
  does not define the return type, shape, mask polarity or treatment of nested
  distribution diagnostics.
- Did / propose: the synced `tests/spec/API_ASSUMPTIONS.md` Q-024 and
  `TESTMAP.md` explicitly defer this unspecified interface to A6. Remove
  `parameter_issues()` and nested diagnostic aggregation from the A4 plan;
  this is no longer an A4 blocker. Preserve numerical NaN propagation without
  assuming a generic custom-parameter validity domain. Before A6, ask the
  maintainer to settle the return contract and ownership of nested diagnostics.
  The earlier boolean-invalid-mask proposal is an option only, not an accepted
  API. No spec change is made here, and this entry remains open.
- Revision-28 planning note (2026-10-19): section 4.7 now defines the
  constraint/condition-based ParameterIssues contract and section 7.1 defines
  likelihood attribution (decision 0042). A6.1–A6.5 in PLAN.md implement it;
  this removes the planning blocker. Formal resolution remains the maintainer's.
- Status: open

## D-012 Right precedence for the unbounded zero-lower observation
- Step: A3 review fix-up
- Spec: sections 3.2 and 3.3
- Spec says: section 3.2 describes derived kinds using `upper==inf` for right
  censoring and `lower==0` for left censoring. Both predicates match a
  non-exact `(0, +inf]` row. Section 3.3 calls this right censoring at entry
  and explicitly treats `from_bounds(0, inf)` as uninformative when `L=0`.
- Did / propose: retain the existing right-before-left classification:
  right means non-exact with `upper=+inf`; left excludes right rows and has
  `lower=0` (so valid left rows have `upper<+inf`). Document and test this
  precedence, as approved in `A3-review-handoff.md`. Propose a later spec
  revision make the precedence explicit in section 3.2. Metadata still
  includes rows with issue bits; this classification is not a validity check.
- Status: open

## D-010 A3 vmap acceptance test indexes scalar `data.issues.ok`
- Step: A2 constructor API migration; observed in an A3 acceptance test
- Spec: sections 3.2, 6.2 and 12.1
- Spec says: `data.issues.ok` is a scalar bool array reducing all invalid rows
  in that `SurvivalData` object. `jax.vmap` construction returns batched data
  leaves; decoding `batched.issues` therefore reduces across the batched shape.
- Did / propose: keep the scalar reduction required by the spec. The read-only
  `tests/spec/test_a3_tracing.py::test_bounds_vmap_agrees_with_eager_for_valid_and_invalid_rows`
  indexes `batched.issues.ok[i]`, which raises `IndexError` for the scalar. To
  compare per-batch validity, the independent test author could vmap a function
  returning `data.issues` or compare `~jnp.any(invalid_rows, axis=1)`.
- Status: open

## D-006 Legacy diagnostic compatibility after pure construction
- Step: specification revision 8 (`spec-constructor`)
- Spec: sections 6.2, 10.2 and 13; revision 7 sections 6.2 and 13
- Spec says: revision 7 uses `TracedInputError` for ordinary traced
  construction and spells the diagnostic decoder `data.issues()`. Revision 8
  accepts traced construction and requires the `data.issues` property plus an
  explicit host-only `raise_if_invalid()` checkpoint.
- Decision: retire `TracedInputError` and do not export it. A call to
  `DataIssues.raise_if_invalid()` on traced diagnostics raises a plain
  `ValueError` with a host-only message, not `InvalidDataError`. Expose only
  the `data.issues` property, without a callable `data.issues()` alias. The
  maintainer approved these choices on 2026-09-30; specification revision 10
  records them in sections 6.2, 10.2 and 13.
- Status: resolved by maintainer, 2026-09-30; specification revision 10

Entries are added by the implementer and resolved by the maintainer.

## D-001 Inactive `time2` entries in mixed observations
- Step: A1
- Spec: section 3.1
- Spec says: `time2` is an interval's upper endpoint, but does not state what
  supplied `time2` arrays must contain on non-interval rows.
- Did / propose: Ignore `time2` values on non-interval rows during
  normalization, including NaN; validate `time2` only where the interval flag
  is true. This allows one `time2` array to accompany mixed observation kinds
  without treating unused entries as endpoints. The maintainer approved this
  handling in conversation on 2026-09-25; the entry remains formally open.
- Status: open

## D-002 JAX weak Python inputs and float64 warnings
- Step: A1 review fix-up
- Spec: section 3.3
- Spec says: section 3.3, updated by commit `3041ebd`, requires JAX promotion
  including weak typing, default floating dtype for integer-only or
  scalar-only inputs, and a caller-site warning for float64 array input when
  64-bit mode is disabled.
- Did / propose: Follow the maintainer's 2026-09-25 decision recorded in the
  independent A1 review: Python `int`/`float` scalars are weak and never warn;
  Python lists/tuples take the dtype JAX gives `jnp.asarray` and never warn;
  NumPy/pandas values and strongly typed JAX arrays with float64 dtype warn
  once per call site in 32-bit mode; explicit `dtype=` controls the result
  without a conversion warning. Weakly typed JAX arrays and tracers retain
  their weak contribution. Integer-only or scalar-only input uses the default
  floating dtype. The clarifying spec edit has landed as `3041ebd`; this
  entry awaits maintainer resolution.
- Status: open

## D-003 Right censoring at entry
- Step: A2
- Spec: sections 2, 3.3, and 6.2
- Conflict: revision 4 rejected right censoring at `c <= L` in section 2
  while section 3.3 accepted `c = L = 0` as uninformative.
- Decision: Right censoring at `c = L` is valid and uninformative for every
  valid entry time, including `L = 0`. Its conditional numerator equals
  `Z = P(L < T <= U)`, so the log contribution is exactly zero. Right
  censoring at `c < L` remains invalid as a recorded censoring time before
  entry. Apply the rule to any normalized censored row with upper bound `+inf`,
  regardless of constructor. R's counting-process `Surv(start, stop, event)`
  rejects equality as an input-format rule; `survreg` does not support
  start-stop responses, while `flexsurvreg` receives them through `Surv`.
  Those package behaviors do not change the conditional likelihood identity.
- Status: resolved by maintainer, 2026-09-28; specification revision 5

## D-004 Attribution of `invalid_time2`
- Step: A2
- Spec: sections 3.1, 3.3, and 6.2
- Gap: revision 4 named `missing_time2`, `invalid_time2`, and
  `invalid_interval` without assigning every upper-endpoint case.
- Decision: `missing_time2` covers omitted or NaN active interval `time2`
  and NaN `upper` on a non-exact `from_bounds` row. NaN `upper` on an exact
  bounds row is allowed and normalizes to `lower`. `invalid_time2` covers
  negative or `-inf` active upper endpoints and an exact bounds row whose
  supplied upper differs from `lower` and is not NaN. `invalid_interval`
  covers an explicit interval or non-exact bounds row with `lower >= upper`
  after both endpoints pass presence and domain checks. `+inf` is valid for
  interval `time2` and non-exact bounds upper; a one-sided `time` uses
  `invalid_time` under section 3.3. Endpoint checks precede ordering to
  avoid labelling one malformed upper as a reversed interval. Independent
  faults may set multiple masks. Nonnumeric fields and incompatible shapes
  cannot form an object and raise.
- Status: resolved by maintainer, 2026-09-28; specification revision 5

## D-005 Attribution of invalid first endpoints
- Step: A2
- Spec: sections 3.3 and 6.2
- Gap: revision 5 assigned `invalid_time` to a one-sided `time` argument
  but did not say which mask reports an invalid interval-flag `time` (the
  lower endpoint) or an invalid `from_bounds` lower.
- Decision: `invalid_time` reports a section 3.3 violation in the flag
  constructor's active `time` on every row, including interval rows, and
  in `from_bounds` `lower`. Exact times must be finite and positive;
  censored lower endpoints must be finite and nonnegative. Check endpoint
  presence and domains before interval ordering, so a bad lower does not
  also produce `invalid_interval`. A NaN `time` or `from_bounds` lower uses
  `invalid_time`: both are required first endpoints, and there is no separate
  missing-time mask. This differs from the optional active upper endpoint,
  whose omitted or NaN value uses `missing_time2`. Nonnumeric fields and
  incompatible shapes still cannot form an object and raise.
- Status: resolved by maintainer, 2026-09-28; specification revision 5

## D-011 `event` combined with flags in `with_issues`
- Step: A2 independent-review fix-up
- Spec: sections 3.1, 6.1, and 6.2
- Spec says: `event` together with any non-None censoring flag is rejected;
  static misuse such as bad argument combinations raises; `with_issues`
  reports invalid numeric rows without value-dependent exceptions. It does
  not explicitly assign this argument-presence conflict to either category.
- Did / propose: A2 returned `conflicting_indicators` on every row from
  `with_issues`, while the eager constructor raised. The maintainer chose
  `TypeError` in both paths in conversation on 2026-09-29, reserving the bit
  for row-wise conflicts between true flags. Implement this in the A2 review
  fix-up; the entry remains formally open for maintainer resolution.
- Status: open; superseded by specification revision 9, which requires
  `TypeError` when `event` is combined with any non-None censoring flag.
  Awaiting maintainer closure.

## D-007 Concrete observation inputs captured inside `jit`
- Step: A2 independent-review fix-up
- Spec: sections 3.2 and 6.2
- Spec says: ordinary construction with traced observation values raises at
  trace time; static metadata is computed eagerly from concrete values. It
  does not say whether concrete inputs captured by a jitted closure must be
  accepted.
- Did / propose: A2 checks the computed issue bits and therefore rejects
  construction inside `jit` even when all observation inputs are concrete.
  The reviewer recommends checking inputs for tracers and evaluating fully
  concrete construction under `jax.ensure_compile_time_eval()`. Await
  maintainer decision before changing this behavior.
- Status: open; superseded by specification revision 9, which requires both
  ordinary constructors to accept traced and concrete inputs inside `jit`.
  Awaiting maintainer closure.

## D-008 Pure bounds constructor name and signature
- Step: A2 independent-review fix-up
- Spec: sections 3.1 and 6.2
- Spec says: `SurvivalData.with_issues(...)` is the pure constructor but only
  gives the flag-form signature. Revision 5 assigns diagnostic masks to bounds
  inputs without naming a pure bounds entry point.
- Did / propose: The reviewer records a maintainer decision that a jit-safe
  bounds counterpart is required, but its public name and signature still
  need to be placed in the normative spec. A possible form is
  `SurvivalData.from_bounds_with_issues(...)`. The maintainer will update the
  spec before adding a public method (conversation, 2026-09-29).
- Status: open; superseded by specification revision 9, which makes
  `from_bounds` the pure bounds constructor and removes issue-returning
  constructor methods. Awaiting maintainer closure.

## D-009 Unsupported outside-interval keyword in `from_bounds`
- Step: A2 independent-review fix-up
- Spec: section 3.1
- Spec says: `outside_interval_censored` is recognized only to raise an
  explicit unsupported-feature error, while the `from_bounds` signature does
  not include that keyword. Its applicability to bounds construction is not
  stated.
- Did / propose: `from_bounds(..., outside_interval_censored=...)` currently
  raises Python's `TypeError`. The maintainer will update the spec before a
  change to this behavior (conversation, 2026-09-29).
- Status: open; superseded by specification revision 9, whose constructor
  signatures reject the unsupported keyword with `TypeError`. Awaiting
  maintainer closure.
