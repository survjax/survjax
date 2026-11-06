Independent review: round 1/3

# KWPATCH.1 — Installed-wheel keyword-only PH calls

Read PLAN.md KWPATCH.1, spec 4.4 constructor policy, then the two-line
change in tests/install/test_wheel.py.

Invariant: registered wrapper constructors require keyword arguments.
Both eager and jitted installed-wheel probes must pass baseline= explicitly.
Runtime code, expected analytic RMST values, isolation and dependency checks
are unchanged. No uncertainty or deviation identified.

Verification:
- Full tests/install/test_wheel.py with pytest -n 6: float64 4 passed
  in 47.78s; float32 4 passed in 47.96s. These exercise installed exports,
  wheel isolation/typing and eager/JIT PH RMST (spec 4.4, 10.3).
- Fast gate through A12: formatting 188 files, lint and mypy (22 files) clean;
  implementation float64 1066 passed / 8 skipped, float32 1071 / 3;
  source doctests 4 passed per mode.
- Independent acceptance float64: 3396 passed / 33 skipped, 2 NumPy reference
  warnings at zero (log and multiply). Float32: 3389 passed / 40 skipped, same 2 warnings.
  Gate exited 0: all gates passed through A12 (fast: slow tests skipped).
- git diff --check: passed. Full Tier 2 and docs build not run for this test-only
  correction; the touched file ran in full in both modes.

Start, previous step and worktree base: d9832baaa19559b7d49eb0e52279695c27efde34.
No remaining patch substeps after review.
