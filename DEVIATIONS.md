# Deviations and spec questions

Entries are added by the implementer and resolved by the maintainer. Format:

```
## D-001 <short title>
- Step: A1
- Spec: section 3.3
- Spec says: ...
- Did / propose: ...
- Status: open
```

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

## D-006 `event` combined with flags in `with_issues`
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
- Status: open

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
- Status: open

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
- Status: open

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
- Status: open
