Independent review: round 1/3

# MNIST.1 — Verified acquisition and fixed cohorts

Read first:
1. PLAN.md's MNIST.1 section; spec 11.2 and both use-case 03 notes.
2. examples/survival_mnist/data.py, then its README.md.
3. tests/unit/test_mnist_data.py.

The maintainer requested starting use-case 3. This first step owns only data
acquisition and splits, with no new dependency or library API. Inherited
unfinished SIM/A9 plans are preserved below the new section. New source,
tests and README total 296 lines. Outcomes/calibration, network, fitting,
metrics, results and docs remain subsequent independently reviewed steps.

Invariants:
- Four compressed files are the raw IDX resources listed by torchvision:
  https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py
  (source checked 2026-10-30). First real downloads matched its published
  MD5s before SHA-256 values were measured and recorded in data.py.
- Both hashes gate fresh files; SHA-256 gates every cache read. Corrupt
  cache files raise without silently redownloading. Verified bytes are
  published atomically to avoid partial entries. No dataset is committed.
- Strict magic, dimensions, count, exact payload length and digit range.
  Official 60k training rows precede 10k official test rows. Pixel arithmetic
  is explicitly NumPy float32 even in an x64 JAX process. No fitted transform.
- Validation uses fold_in(fold_in(key(20261025),3),1), first 10k permutation
  indices; training uses the remainder. Official test rows stay ordered.
  Smoke is a prefix of each cohort. Global indices are retained for the
  future per-image outcome keys; no outcomes are generated in this step.
- Labels are for future synthetic outcomes and oracle evaluation, never an
  additional network input. Attribution is in the example README; page
  attribution is a recorded docs to-do.

Tests and actual evidence:
- JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_mnist_data.py:
  11 passed in 11.30s; JAX_ENABLE_X64=0 same command: 11 passed in 0.65s.
  Synthetic IDX tests cover payload values, invalid magic/count/length/labels,
  download MD5/SHA rejection, checked warm cache, corrupted-cache failure,
  float32 NCHW preprocessing, official global order, exact key derivation,
  disjoint full coverage and smoke prefixes. Network calls are mocked.
- Separate python3 urllib first-fetch probe downloaded all four resources
  to /private/tmp/survjax-mnist-verification; all published MD5s matched.
  Sizes: 9912422 / 28881 / 1648877 / 4542 compressed bytes in resource order.
  Digests in data.py are measured SHA-256 results, not invented fixtures.
- uv run python warm-cache probe called load_mnist twice on that directory:
  both returned (70000,1,28,28) float32, (70000,) uint8, range [0,1].
  Training-source digit counts were [5923,6742,5958,6131,5842,5421,5918,
  6265,5851,5949]; full splits 50000/10000/10000. No network needed on reuse.
- uv run ruff check .: All checks passed!
- uv run mypy src: Success: no issues found in 20 source files.
- scripts/check.sh --through A10: guard/static checks passed; own suites
  1076 passed/7 skipped (83.88s), 1082 passed/1 skipped (82.65s); doctests
  4 passed per mode (1.51s/1.53s). Float64 acceptance reached 2327 passed,
  28 skipped, 5 setup errors (99.91s): missing required SURVJAX_WHEEL.
  Float32 acceptance was not reached in this failed invocation.
- uv build --wheel --out-dir /private/tmp/survjax-mnist-wheel: succeeded.
  Re-running acceptance in both modes with that SURVJAX_WHEEL artifact,
  SPEC_THROUGH=A10, uv run --group test pytest tests/spec -q -n 6:
  2332 passed/28 skipped (62.72s) in float64; 2322 passed/38 skipped
  (59.19s) in float32. Both exit 0.

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, CPU device 0.
No docs build (no docs changed), training, oracle computation or full demo.
Fresh wheel built to satisfy the independent installation acceptance tests;
passed earlier own/static/doctest checks are not repeated. No claim that the full use case or release is complete.
