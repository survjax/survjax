# A5.3 — Exponential

Implemented after maintainer approval of PLAN.md, on `a5.3-exponential` from
main `7a2fef6`. Planning and spec-marker alignment to revision 23 (`a212ddd`)
are separate commit `9337969`. Revision 23 changes documentation scheduling;
revision-22 numerical acceptance tests remain applicable and untouched.

## Reading order and invariants

1. `src/survjax/distributions/exponential.py`: registered frozen family, only
   natural `rate` as its leaf, keyword-only constructor and `from_log`.
   Log inputs normalize before exp, so dtype misuse cannot be hidden by it.
   `from_log` stores natural exp-rate values; overflow stores infinity.
2. Analytic log survival, cumulative hazard, density and hazard methods.
   Hazard uses rate directly and log hazard uses log(rate), retaining their
   analytic infinity limits without large-H subtraction. Survival and log CDF
   inherit existing stable defaults; registration retains all time-policy rows.
3. `_cumulative_hazard`: substitute zero for infinite times before multiplying
   live rates. Infinity sign/NaN limits follow r*t, including invalid rates;
   no parameter-domain validation or clipping. The original primitive probe
   preserves NaN-rate precedence in density policy. Finite-component derivatives
   remain isolated in shared-rate and per-entry mixed infinity batches.
4. `reference_time`: 1/r within the normal representable reciprocal range;
   otherwise one time unit. Substitute the denominator before dividing.
   This changes reference inputs only; never active parameters/probabilities.
5. `tests/unit/test_exponential.py`, exports and import smoke test; PLAN.md's
   implementation accounting and A5 docs backlog.

Public docstrings give formulas, inverse-time units, shapes, domains,
rate=1/mean, log-rate/log-mean mappings, a short conversion example and misuse
errors. No docs pages, generic wrappers, registration machinery, dependencies,
independent tests or spec were edited. No global precision changes or host
conversion of tracers. Code/test/docstring changes total **299 lines**:
103 family lines, 183 own-test lines and 13 changed export/import-test lines.
Planning and review bookkeeping are separate from that implementation count.

## Requirements and checks

- Spec 3.3/4.1/4.2: natural leaves, registration, keyword construction,
  indexing/tree slicing and tracing checked by independent construction cases.
  Own tests add pre-exp kind rejection, single float64 conversion warnings,
  integer/default and float32 log-input dtypes, overflow storage, and extreme
  reference shape/dtype/positivity under jit.
- Spec 4.1/4.5/5: seven-method algebra with multiaxis jit/vmap calls, finite
  invalid-rate formulas and NaN isolation in own tests; full policy matrices,
  NaN-rate/time rows and pointwise pairing in independent tests.
- Spec 4.2/12.1: tiny H uses zero absolute tolerance; log CDF lower tail and
  large-H log density/log hazard use algebra. Independent cases also cover
  survival underflow and analytic hazard precision. Tolerances are unchanged.
- Spec 7.2/7.3: own exact gradient/Hessian and boundary derivative checks,
  shared/per-entry mixed-infinity forward/reverse Jacobians under jit; independent
  order-two forward/reverse finite differences and from_log chain-rule cases.

Final command:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
exited 0 with `check: all gates passed through A4`:

- Guard: ok (`a5.3-exponential`); Ruff: 42 files formatted, all checks passed;
  mypy: no issues in 10 source files.
- Own float64: **349 passed, 7 skipped** (19.82s).
- Own float32: **355 passed, 1 skipped** (19.81s).
- Source doctests: **1 passed** each (0.67s/0.58s).
- Independent through-A4 float64: **679 passed, 25 skipped, 370 deselected**
  (18.95s); float32: **693 passed, 11 skipped, 370 deselected** (15.79s).

A5 selection, in separate `JAX_ENABLE_X64=1` and `0` processes:

```bash
SPEC_THROUGH=A5 .venv/bin/python -B -m pytest \
  tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py \
  tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py \
  tests/spec/test_a5_derivatives.py \
  -k '(exponential and not weibull) or nan_rate or boundary_values_agree_under_vmap_over_time' \
  -q -p no:cacheprovider
```

Both exit 0: float64 **95 passed, 227 deselected** (4.33s);
float32 **91 passed, 4 skipped, 227 deselected** (2.78s). The four skips are
second-order finite differences requiring float64. This selection excludes
mixed-family construction, Weibull identities and grids; it is not full A5
acceptance. Own warning test adds one float64 skip since conversion is x32-only.

An initial gate attempt failed collection because the new test imported
`tests.impl_tolerances`; corrected to the repo's `impl_tolerances` import.
An exploratory duplicate finite-difference check used value tolerances and
failed by about 5e-8. It was omitted in favor of the unchanged independent
order-two checks at the same inputs, which pass; no tolerance was relaxed.
`git diff --check` passed; the code diff accounting was checked explicitly.

## Deferred work and stop

Narrative documentation remains in PLAN.md's A5 Docs to-do and separate A5.d1+
steps after the final A5 code merge. No docs build/render, R fixtures,
performance measurement or new wheel verification was run for this step.
No mutation experiments were run. Existing deviations remain for maintainer
closure. Commit A5.3 and stop before Weibull, grids, merging or publishing.
