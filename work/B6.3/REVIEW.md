# B6.3 — Flexsurv comparison coverage documentation

Independent review: round 1/3

## Read first

1. PLAN.md and spec 10.4–10.5 (rev 62), with section 2 for conditioning.
2. docs/technical-notes/verification.md: opening scope, R comparison table,
   numerical limits and exclusions; docs/likelihood.md supplies the entry link.
3. tests/fixtures/slice_b/README.md and scripts/fixtures/generate_slice_b.R.
4. Existing tests/reference/test_slice_b_r.py and
   tests/spec/test_observation_censoring_flexsurv_fixture.py.

One docs topic, 98 changed documentation lines. No source, tests, fixtures,
dependencies, public API or protected paths changed. This covers B3.1/B3.2's
docs to-do and the last documentation item deferred by B6.1/B6.2.

## Claims and invariants

The table represents ten cells: four untruncated, two left-truncated, three
right-truncated and exact-only two-sided truncation. Endpoint restrictions
match spec 10.4 and the generator. All six exclusion categories are listed:
left/interval with left truncation; right censoring with finite U; sets beyond
U; censoring at/before entry; observations outside the window; exact zero.
The prose distinguishes differing R conventions from library support, and
links the intersection/conditioning guide. Right censoring at entry is valid
and uninformative; before entry is invalid.

The source generator supplies seed 20261110, five rows per cell and two fixed
vectors per family (Weibull, LogNormal, LogLogistic), with no covariates.
The committed fixture contains 300 per-row fixedpars logliki values. The tests
check these values, numerical-range limits, endpoints at U and independent
mpmath agreement. The float64-only numerical R comparison is explicit;
float32 integrity checks and separate mathematical/derivative evidence are
distinguished. No R derivative, fitted-estimate, deep-tail or release-wide
certification claim is made. The old unsupported-mode introduction is removed.

## Validation

Focused command run in separate processes with JAX_ENABLE_X64=1 and 0:

```sh
SPEC_THROUGH=A12 \
SPEC_FEATURES=observation.censoring,observation.truncation,observation.traced_metadata,family.lognormal,family.loglogistic \
uv run --group test pytest -n 6 -q \
  tests/reference/test_slice_b_r.py \
  tests/spec/test_observation_censoring_flexsurv_fixture.py
```

- Float64: 80 passed (4.65s). Float32: 38 passed / 42 skipped (1.11s).
  No slow-marker filter. The skips are the float64-only comparisons.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Ruff
  format: 281 files already formatted; lint clean; mypy clean on 25 files.
  Implementation float64: 1212 passed / 8 skipped (15.85s); float32:
  1214 passed / 6 skipped (15.97s). Source doctests: 4 passed per mode.
  Spec float64: 4654 passed / 46 skipped (37.82s); float32:
  4597 passed / 93 skipped (39.48s), two existing NumPy warnings per mode
  at test_a11_ph_direct_methods.py:78.
- scripts/build_docs.sh run directly: exit 0, "Verified both dtypes".
  All existing runnable pages executed; no new code cells in this prose step.
  Existing Jupyter transport notices appeared; no Sphinx warning failure.
  Build identity: base 2a8dec1, correctly labelled dirty with this step,
  completed 2026-11-12T17:46:00Z, state success.
- Generated HTML inspected with HTMLParser in both modes: table, provenance,
  exclusions, corrected opening and reciprocal anchors/links present. Downloaded
  Slice B README is byte-identical to its source. No browser layout check.
- git diff --check clean. No R regeneration, benchmark run or non-fast gate.
  No new standalone tests were needed for this documentation-only change.

## Commits and remaining work

Step starting / previous-step final commit:
2a8dec18198d66322fc5be489a00064d2e6d2f01 (B6.2).
Original worktree base: cc0e30f.

Final Slice B evidence audit and maintainer integration remain. This step
makes no completion claim and does not update a shared roadmap. Benchmark
artifacts remain in the experiments project. No spec question identified.
