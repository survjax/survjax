# A9.3 — Tiny synthetic example

Read `examples/slice_a.py`, then `tests/unit/test_slice_a_example.py`, then
the active A9.3 entry in PLAN.md. Approval: "review passed; on to next bit".
Branch `A9`, starting commit `51978e8`. This step adds 70 example lines and
83 test lines (153 total). No library source, public exports, dependencies,
shared tolerances, spec, independent tests, protected scripts or docs pages
changed. Spec alignment check: `spec unchanged since be26e50`.

The example constructs three explicit synthetic exact/right observations,
performs the data's explicit host-validation checkpoint, and evaluates both
built-ins through the public likelihood method. It compiles the summed
Weibull objective with its joint shape/scale gradient and Hessian, then
returns a time-first survival grid for two scales. `run_example()` returns
named JAX arrays; `main()` prints those same results under a main guard.
No global JAX config, fitting algorithm, plotting or external data is used.

Invariants and spec-to-test mapping:

- `test_example_contributions_match_exact_and_right_formulas` (6.1, 7.1,
  12.2): exact times 1 and 3, right censoring at 2, exponential rate 0.5
  and Weibull shape/scale 2. Checks the actual example's returned inputs,
  both contribution vectors, and every floating output's process-selected
  dtype. Expected log contributions come from log f/log S, with the formulas
  stated beside the assertions.
- `test_example_compiled_objective_has_analytical_derivatives` (7.3,
  12.2): scalar objective, two-component gradient and complete 2-by-2
  Hessian, including the mixed derivative. Expectations come from explicit
  differentiation of the exact/right Weibull formula, not library results.
  The objective equals log(3/4)-7/2; its scale derivative is 3/2 and second
  scale derivative -17/4. The shape and mixed entries use the stated powers
  and logs at t/scale=[1/2,1,3/2].
- `test_example_grid_is_time_first_and_starts_at_zero` (5, 12.2): checks
  grid times [0,1,2,3], scales [1,2], the (4,2) output and the exact S(0)=1
  row. Every value is compared to the analytical exp(-(t/scale)^2) formula.

The test fixture loads the script by path and calls its run_example(),
so CLI and regression tests share the same calculations. All tolerances
come from tests/impl_tolerances.py unchanged. No new derivative guarantee is
introduced; the example uses finite valid smooth interior parameters.
Existing library tests supply the broader transformation/derivative matrix.

Checks completed:

- `JAX_ENABLE_X64=<1|0> UV_CACHE_DIR=/private/tmp/survjax-uv-cache
  uv run --locked --offline python examples/slice_a.py`: exit 0 in both
  modes. Printed Weibull total -3.7876820724517812 (float64) and
  -3.787682056427002 (float32); gradient approximately [-0.0266918,1.5],
  Hessian approximately [[-0.990018,1.48901],[1.48901,-4.25]], and a (4,2)
  survival grid with first row [1,1]. Exponential/Weibull per-row outputs,
  inputs and grid coordinates are also printed.
- `JAX_ENABLE_X64=<1|0> UV_CACHE_DIR=/private/tmp/survjax-uv-cache
  uv run --locked --offline pytest tests/unit/test_slice_a_example.py -q`:
  3 passed in 2.17s per mode, no skips or failures.
- Ruff: 78 files already formatted / All checks passed! Mypy:
  Success: no issues found in 17 source files. `git diff --check`: exit 0.

CLI logs: `/private/tmp/survjax-a93-example-x1.log`,
`/private/tmp/survjax-a93-example-x0.log`. Focused logs:
`/private/tmp/survjax-a93-focused-x1.log`,
`/private/tmp/survjax-a93-focused-x0.log`.
The initial wheel-runner cumulative attempt exited 1 during its float64 own
suite: 793 passed, 7 skipped, 2 installation setup errors in 171.97s. Both
errors came from an existing JAX cache archive missing its dist-info/RECORD;
the unchanged installation tests could not copy it into a fresh environment.
A focused `pytest tests/install -q -x` diagnostic reproduced that setup error
(1 passed, 1 error in 0.34s), recorded in
`/private/tmp/survjax-a93-install-diagnostic.log`. Initial gate log:
`/private/tmp/survjax-a93-gate.log`.

Environment repair used official uv cache/install commands only. An initial
cache clean waited for the running gate, then removed the JAX cache entry;
a first refill also found a missing JAXlib RECORD. After the gate and clean
finished, the other five runtime cache entries were cleared and all six
runtime wheels reinstalled into `/private/tmp/survjax-a93-cache-repair` from
an export of the existing lock. A one-time network download restored the
pinned artifacts; package versions, source, tests and lockfile did not change.
The partially installed repair-only environment emitted missing-RECORD
uninstall warnings, so it is not used as verification evidence. Fresh
installation-test environments remain the verification target. Cache-clean
process inspection required sandbox escalation and confirmed both commands
had already exited; no process was killed. This setup repair does not make
regular tests require network.

Offline preflight with the repaired cache:
`JAX_ENABLE_X64=1 SURVJAX_WHEEL=<initial runner artifact>
UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run --locked --offline pytest
tests/install -q`: 3 passed in 16.68s, no errors or skips. Log:
`/private/tmp/survjax-a93-install-repaired.log`. Final rerun:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check_wheel.sh --through A7`
exited 0: `check: all gates passed through A7`. Guard: ok (A9). Ruff:
78 files already formatted / All checks passed! Mypy: no issues found in
17 source files. Tested Python 3.12.9, JAX 0.11.2, NumPy 2.5.3;
`jax.devices()` reports CpuDevice(id=0).

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 795 passed, 7 skipped, 192.90s | 801 passed, 1 skipped, 188.68s |
| Source doctests | 4 passed, 0.99s | 4 passed, 0.86s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 101.28s | 2118 passed, 24 skipped, 218 deselected, 93.42s |

Full log: [/private/tmp/survjax-a93-gate-final.log](/private/tmp/survjax-a93-gate-final.log).
The three new example cases pass in each dtype, with no new skips or
exclusions. All tests in the final run remained offline after cache setup.
The 218 deselections belong to later gate steps. The new runner artifact is
retained at `/private/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.TnEXsf/survjax-0.0.1.dev0-py3-none-any.whl`.
Only PLAN.md/REVIEW.md result records changed after this gate; final Ruff and
whitespace checks follow before commit.

No new deviation or contract ambiguity. No R, docs build or benchmark is
needed for this example step. A8 fixtures remain absent, so the cumulative
A9/Slice A completion gate is still pending A8. A9.4 benchmarks and A9.d1
narrative docs have not begun; the approved docs to-do remains in PLAN.md.
Commit A9.3 and stop for review before the next step.
