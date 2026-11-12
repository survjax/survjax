# B6.2 — Observation metadata documentation

Independent review: round 1/3

## Read first

1. PLAN.md and spec 3.2, 6.2 and 7.2 (rev 62).
2. docs/survival-data.md, "Metadata, pruning and compilation reuse".
3. docs/technical-notes/verification.md, "Reproduce the CPU baseline".
4. Existing src/survjax/data.py and tests/unit/test_data_metadata.py;
   tests/spec/test_observation_traced_metadata.py supplies acceptance coverage.

One docs topic across two pages; source, tests, API, dependencies, tolerances
and protected paths are unchanged. Covers the B4.1–B4.3 docs to-do items,
including the historical benchmark distinction deferred by B6.1.

## Invariants and evidence

Concrete flags derive from all normalized rows, including invalid rows.
Any traced observation input selects argument-form rules for every observation
argument: None versus supplied only, never values. The table lists both forms;
the text covers independent truncation flags, supplied False/0/inf, nested
tracers and concrete inputs inside transformations. Invalid rows still carry
issues, regardless of which flags are false. Slices inherit flags. Different
values reuse compilation only with the other static/cache conditions fixed.

Executable docs assert equal leaves despite different tree structures when
default bounds are supplied under tracing, and equal structures for prepared
and traced exact/right data. A second example changes event values from all
exact to all right to an invalid indicator, asserting one trace and the
correct invalid_indicator mask. These demonstrate spec 3.2, 6.2 and 7.2 in
both dtypes without exposing private flag positions as a public interface.
Existing metadata and pruning suites supply the broader acceptance evidence.

Historical provenance was checked against benchmarks/slice_a_cpu.json and
git show 51319c8:src/survjax/data.py. The artifact records library revision
0a200085; 51319c8 contains the harness and all-true traced implementation.
The current scripts/benchmark_slice_a.py still asserts all-true traced flags
in _row. Its reproduction commands are now explicitly scoped to that historical
checkout; the current-branch incompatibility is disclosed. No benchmark was
run, repaired, copied or added; current pruning work belongs to experiments.

## Validation

- scripts/check.sh --fast --through A12: exit 0, all gates passed. Format:
  279 files already formatted; Ruff clean; mypy clean on 25 source files.
  Implementation float64: 1212 passed / 8 skipped (15.23s); float32:
  1214 passed / 6 skipped (15.36s). Source doctests: 4 passed in each mode.
  Spec float64: 4654 passed / 46 skipped (37.99s); float32:
  4597 passed / 93 skipped (37.56s), two existing NumPy warnings per mode
  at test_a11_ph_direct_methods.py:78.
- scripts/build_docs.sh run directly twice. The first executed both dtype
  builds but exited 1 because I corrected prose while it was running;
  its checkout-change guard correctly rejected that build. The second ran on
  settled files, exited 0 and reported "Verified both dtypes". All notebook
  assertions passed. Existing Jupyter transport notices appeared.
- Final build identity: base 30c378e, dirty with this step's files, completed
  2026-11-12T16:19:20Z, state success. Inspected generated HTML for both changed
  pages in both modes using HTMLParser: final prose, constructor table, anchor,
  cross-link and "Traces: 1" output present. No browser layout check claimed.
- git diff --check clean. No full non-fast gate, R regeneration, new standalone
  test or benchmark rerun. No unresolved spec question identified.

## Commits and remaining work

Step starting / previous-step final commit:
30c378e376667ebfe4b1619b5af43e1cde39b61d (B6.1).
Original worktree base: cc0e30f.

Remaining: document the ten comparable flexsurv cells and exclusions from
B3.1/B3.2, then final Slice B evidence audit and maintainer integration.
This step does not claim Slice B completion or update a shared roadmap.
