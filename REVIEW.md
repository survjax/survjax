Independent review: round 2/3

# RMST.1 — Staged base surface and Exponential RMST

Branch `release-rmst`, clean starting snapshot
`8757c0fda78e34e6e818249adedd2a144a33c33d`; spec revision 37.
Read PLAN.md's RMST.1 section first, then spec 8.7, 4.5, 7.2–7.3 and
12.1, then distributions/base.py, exponential.py, tests/unit/test_rmst.py.
The unfinished A9/A10 plan remains verbatim below the new section.

## Invariants and requirement evidence

- Base `rmst` fails for unsupported families; mean/median/quantile fail on
  all current families. `test_staged_summary_surface` exercises custom,
  Exponential and Weibull methods eagerly/at trace time and absent future
  methods. Weibull RMST is deliberately the next staged implementation.
- Exponential uses -expm1(-r*tau)/r without floors or epsilons. Parameter
  issues override every cutoff, including zero. Both rate and cutoff are
  substituted before arithmetic on invalid broadcast entries. Constants
  have zero tangents; no real active operand is stop-gradiented.
- Zero stays active: exact value 0, rate derivative 0, cutoff derivative 1.
  `test_rmst_zero_value_and_derivatives` checks both AD modes under debug_nans.
- The misuse tests cover nonreal dtypes and incompatible shapes at call and
  trace time (4.5/5). Conversion follows ordinary JAX promotion; no global
  dtype changes or stored cutoff array.
- The mpmath value test integrates survival independently at 60 digits after
  u=r*t substitution, across r*tau=0.01 through 1e5, six rates and three
  scales. Eager/jit/jit-vmap values and explicit (m,S,arms) cutoff axes use
  existing zero-absolute tail targets. No new tolerances.
- The invalid-value and isolation tests cover zero/negative/NaN/±inf rates,
  negative/NaN/±inf cutoffs, shared and per-entry parameters, forward/reverse
  first derivatives and Hessians of valid entries beside NaNs (7.2/8.7).
- Independent mpmath differentiation of the survival integral checks the
  gradient/full Hessian in (rate,tau), third/fourth rate derivatives in both
  modes, and d/dtau=S(tau). check_grads(order=2, fwd/rev) uses the existing
  O(1) finite-difference derivative scales. This tests through order four;
  no finite test establishes every derivative order. The expression uses
  ordinary differentiable JAX primitives without a custom derivative rule.

## Checks run

Environment: Python 3.12.9; JAX 0.11.2; NumPy 2.5.3; mpmath 1.4.1;
pytest 9.1.1; CpuDevice(id=0). Existing lockfile, no dependency changes.

- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_rmst.py
  tests/unit/test_exponential.py tests/unit/test_family_registration.py
  tests/unit/test_family_direct.py -q`: 140 passed, 3 existing skips, 16.28s.
  Same command with `=0`: 143 passed, 16.83s.
- Final focused `JAX_ENABLE_X64=1 uv run --group test pytest
  tests/unit/test_rmst.py -q`: 13 passed, 14.13s. With `=0`: 13 passed, 14.22s.
- `uv run ruff format --check .`: 112 files already formatted;
  `uv run ruff check .`: All checks passed;
  `uv run mypy src`: Success: no issues found in 20 source files.
  An initial targeted lint found E731/B023 in test lambdas; corrected before
  these final runs. `git diff --check`: clean.
- Fresh-wheel `scripts/check_wheel.sh --through A10`: own tests 1061 passed,
  7 skipped (131.58s) in float64; 1067 passed, 1 skipped (161.50s) in
  float32. Source doctests: 4 passed per mode (2.32s/2.29s).
  Independent through-A10: 2332 passed, 28 skipped (206.57s) / 2322 passed,
  38 skipped (177.05s). Existing skips only. Exit 0:
  `check: all gates passed through A10`. Wheel SHA-256:
  `57bfa188bd6c894adf12948ee47e2be0c4e97fca21b1baf91e26b3d14e754872`,
  artifact `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.O9jk9A/survjax-0.0.1.dev0-py3-none-any.whl`.
  The wheel build preceded the final base-docstring wording correction and
  test-only tightening; runtime implementation was identical. Final focused
  and static checks above used the final source/tests.

## Limits and ownership

Initial commit: 300 source/test changed lines (plan/review excluded). The
F-1 follow-up adds 18 test lines in a separate commit. Only
base.py, exponential.py and RMST-specific tests change behavior/evidence.
base.py was already 502 lines; no unnamed helper/module or refactor added.
No docs, exports, wrappers, dependencies or protected paths touched.
No implementation deviation added. Reviewer question S-1 remains for the
maintainer (below); no behavior change made for it. Staged RMST is incomplete.
After the separately reviewed Weibull core steps merge, hand base.py,
exponential.py and weibull.py to release-ph for private built-in closure
hooks and wrapper RMST integration. PH owns wrappers.py/exports already;
it must wait for that core handoff. A9/A10 docs remain separately owned.

## Round 1 response and round 2 request

Round 1 on 944582f: `fix-first`. F-1 fixed by the new
`test_rmst_cutoff_derivatives_isolate_invalid_entries`: differentiate a
masked sum in the cutoff vector [1,-1,NaN,+inf,-inf,0], in forward and
reverse modes under jit, against [exp(-2),0,0,0,0,1]. This tests the safe
cutoff operand directly, rather than only differentiating the selected rate.
Runtime implementation unchanged; this is an 18-line test-only fix-up.

- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_rmst.py -q`:
  14 passed, 12.85s. With `=0`: 14 passed, 12.55s.
- Mutant proof in separate dtype processes: `JAX_ENABLE_X64=1` / `=0`
  `uv run --group test python -`, compiling an in-memory copy of exponential.py
  with only `safe_cutoff = cutoff`, rebinding the public Exponential to that
  copy, then running pytest on the new test node. Both fail as intended:
  NaN-location mismatch at invalid cutoffs; 1 failed, 0.22s / 0.21s.
  No source file or reviewer-owned mutation worktree changed.
- Ruff format: 112 files already formatted; Ruff lint: All checks passed;
  mypy: no issues in 20 source files. The earlier fresh-wheel full gate is
  evidence for the initial implementation; it was not repeated for this
  test-only follow-up. Final scope/whitespace checked before commit.

N-1 deferred to planned RMST.3 full-range derivative evidence: honor 12.1's
absolute-scale tau derivative rule, with no relative-tail accuracy promise.
N-2 deferred: no unnamed module/helper abstraction introduced in this step;
validation reuse and file sizes must be considered in RMST.2 before code.
S-1 is the reviewer's spec question: does 8.4's NaN-tangent requirement apply
to invalid staged RMST entries, or does 7.2 safe substitution govern their
own zero tangents? The skill reserves spec questions for the maintainer;
no spec/deviation/code change made for S-1 and no resolution claimed.

The reviewer independently ran check.sh through A10: statics, own tests and
doctests passed in both modes; x64 independent tests had 2327 passed,
28 skipped and 5 setup errors because SURVJAX_WHEEL was unset. Its permission
mode refused check_wheel.sh, so it did not confirm wheel acceptance or start
x32 independent tests. Our fresh-wheel gate above passed all of those tests.
The reviewer's `/tmp/survjax-mut-RMST.1` worktree remains after its cleanup
was refused. Leave reviewer artifacts to the maintainer; do not edit them.
Request round 2 from the clean fix-up commit, then follow the skill verdict.
