# B4.3 — Argument-form metadata for traced construction

Independent review: round 1/3

## Read first

1. PLAN.md and spec rev 62 sections 3.2, 7.2 and 12.3 (decision 0060).
2. src/survjax/data.py: only the two metadata selection sites change.
3. tests/unit/test_data_metadata.py and the independently synced
   tests/spec/test_observation_traced_metadata.py (51 acceptance tests).
4. scripts/spec-features.txt and the experiment archive record linked below.
5. The benchmark records below and DEVIATIONS.md's historical test discrepancy.

## Invariants and evidence

Only None versus supplied determines metadata when any observation input is
traced, including nested inputs or concrete arguments beside a tracer.
Flag-form construction retains exact; event/right-only forms retain right;
supplying left or interval keeps all four kinds. Bounds form retains all
censored kinds and retains exact only when supplied. Each truncation flag is
independently true only when that bound is supplied. time2 adds no kind.
No traced or captured observation values are inspected for specialization.

Concrete metadata still comes from normalized rows, including invalid ones.
Normalization, stored issues, current-field validation, dtype policy, slicing,
PyTree reconstruction and likelihood arithmetic are unchanged. No new host
transfer, callback or synchronization is introduced in the compiled library
path: metadata selection uses Python identity checks on argument presence.
Benchmark synchronization is explicit caller-side timing, not library behavior.

The updated unit metadata suite covers vmap and nested-tracer forms, supplied
false indicators, supplied default windows, omitted exact, unchanged leaves,
invalid-indicator preservation and compilation reuse across changed values
(spec 3.2, 6.2, 7.2). Existing construction/tracing tests retain validation
coverage. Both pruning suites retain hook omission, invalid-row isolation and
parameter derivative checks. The 51 independent tests cover every argument
form, slices, jit/vmap, compilation reuse and raw likelihood value/derivative
agreement (spec 3.2, 6.2, 7.2, 12.1). All ran without a slow-test filter.

D-B4.3-001 recorded a protected test that changed shape while expecting one
compilation. The maintainer supplied the independent author's correction via
3d9704a / merge b4b6700 and instructed work to resume. The corrected test now
passes. No protected test was changed by the implementation; the historical
entry's formal status remains maintainer-owned.

## Validation

Focused command (also run with JAX_ENABLE_X64=0):

```bash
JAX_ENABLE_X64=1 SPEC_THROUGH=A12 SPEC_FEATURES=observation.censoring,observation.truncation,observation.traced_metadata,family.lognormal,family.loglogistic uv run --group test pytest -n 6 -q --durations=15 tests/unit/test_data_metadata.py tests/unit/test_data_tracing.py tests/unit/test_data_construction.py tests/unit/test_likelihood_pruning.py tests/unit/test_likelihood_untruncated_pruning.py tests/spec/test_observation_traced_metadata.py
```

- Before the independent test correction: 124 passed, 1 failed in each dtype
  (19.63s float64, 19.60s float32), solely D-B4.3-001.
- After correction: 125 passed float64 (21.46s), 125 passed float32 (22.03s),
  both exit 0. All 51 acceptance tests are included. New unit tests were below
  the top-15 call-duration cutoffs of 0.48s / 0.45s; no slow-list refresh needed.
- First fast gate stopped at mypy: fixed-versus-variable-length tuple inference
  at data.py:147. Added a local tuple[bool, ...] annotation, with no runtime
  change; focused tests and benchmark runs predate only this annotation.
- Final scripts/check.sh --fast --through A12: exit 0, all gates passed.
  Formatting: 275 files already formatted; Ruff: all checks passed; mypy:
  no issues in 25 source files. Implementation tests: float64 1212 passed /
  8 skipped (14.94s), float32 1214 passed / 6 skipped (15.02s). Source doctests:
  4 passed each. Acceptance tests: float64 4654 passed / 46 skipped (40.82s),
  float32 4597 passed / 93 skipped (37.11s). Each acceptance run emitted the
  two existing NumPy warnings at test_a11_ph_direct_methods.py:78 (log zero
  and invalid multiply in its expected-value expression).
- git diff --check: clean.
- No full non-fast cumulative gate, docs build or R fixture regeneration run.

## Benchmark evidence (owned by the experiments repository)

At the maintainer's request, B4.3.CLEANUP moved the benchmark and measured
results out of the library's current tree. The existing independent
likelihood-pruning experiment owns the [historical B4.3 record](../../../experiments/projects/likelihood-pruning/runs/20261112-library-b43-import.md)
and [checksum manifest](../../../experiments/projects/likelihood-pruning/runs/20261112-library-b43-import.json),
committed as aca9b2300a6216bf47dd24eb0d9c8eb61ffbb7c3 in that repository.

The archive preserves the exact benchmark, tolerance source, both dtype outputs
and this handoff's original timing tables/methodology from library commit
200c1f7f4b64d147d5d40cbcba44f97e44d3c1f5. Checksums and retrieval instructions
are in the experiment record. Both original runs exited 0 with agreement and
compilation-reuse checks passing. No benchmark rerun was performed for relocation.
The active experiment remains independently pinned to 82c1826; these later
B4.3 measurements are historical evidence, not part of that active run.

## Commits and remaining work

Step starting commit: aaba08f6b42c00433792a5984d53faa85a78d7fd.
Resumed on maintainer merge b4b6700c24626e53463dec65430587be378bf5ae.
Previous branch step final: 82c1826435c065bfcd4b13e05a77be26f35e0a89
(B5.FAMILIES); accepted pruning dependency: f568b9b (B4.2).
Original worktree base: cc0e30f.

Both requested spec-since commands ran. Rev 62 acknowledgement was already
committed by 71cadbf (.spec-seen = a5c1dea); --mark changed no content, so no
redundant acknowledgement commit was made.

Likelihood documentation (including argument-form versus value-derived
metadata), final Slice B evidence audit and maintainer integration remain.
Reviewer-workflow investigation stays deferred. After review, attempt the
workspace diff-display hook using the saved step starting SHA and the final
reviewed SHA, as required by the updated workflow.
