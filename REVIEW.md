Independent review: round 2/3

# SIM.7b.2 — Runtime provenance and metadata.json

Start/previous final: f4cfbfcc498c727ca95258fc1638b88309992d22.
Merge base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
User authorizes next provenance step; D-034 shared timings already approved.
Read PLAN.md SIM.7b.2, the new functions in existing _results.py, run.py's two
host calls, then check_metadata.py and actual CLI metadata checks.
No new module/dependency/library API, numerical fitter/generator change,
protected path, docs or spec edit. Preserve plan history.

Snapshot identity before simulation/CSV writes; runtime facts must not be
invented. Thread environment null means unset, not a measured worker count.
Actual Python process argv is sys.orig_argv, not a claimed uv shell wrapper.
Final JSON adds exact scenario/censoring and grouped synchronized timing;
allow_nan=False. Metadata writing happens only after both CSVs succeed.
## Reading order / scope

1. PLAN.md SIM.7b.2, use-case common/details metadata requirements and D-034's
   already-approved grouped timing scope.
2. _results.py: _cpu_model, capture_metadata, write_metadata and invalidation
   of prior metadata immediately before CSV replacement.
3. run.py: snapshot before calibration/fitting, final metadata after CSVs.
4. check_metadata.py and check_cli.py real smoke/ordering assertions;
   check_results.py stale-marker invalidation assertion.

No numerical batching, fitter, acceptance, generator, core/API/dependency,
protected path, docs or spec change. Existing helper only. Implementation /
check diff is near the 300-line target (about 346 including field/fixture
layout); helper 301 lines, runner 299, separate metadata checks 147 lines.
Metadata was split from CSV export before code. Preserve plan history.

## Invariants / requirement mapping

- capture_metadata is called after config/x64/output-directory validation and
  before calibration or fitting. Mocked grid wrapper asserts the directory
  exists and neither calibration nor grouped runner has been called yet.
  Source Git HEAD/dirty-at-start, lock SHA-256 and spec-seen are captured once,
  before new CSV files can affect worktree state. Dirty-at-start includes all
  pre-existing worktree changes, not a guarantee about code changes alone.
- Invocation is the actual sys.orig_argv list and cwd with run_config as the
  entry point. It includes Python flags/executable and does not claim to know
  a parent shell's uv wrapper. Direct imported calls still record their
  process context along with the requested configuration/output directory.
- Runtime records actual Python and package versions (survjax/JAX/jaxlib/
  NumPy/Optimistix/Matplotlib), OS/machine, CPU model/logical count, actual
  backend/devices/x64 and PRNG implementation. Darwin CPU model is sysctl;
  Linux-style proc text/fallback has a host fixture check. No cross-platform
  runtime matrix is advertised; only the local macOS CPU environment ran.
- Curated XLA/OMP/BLAS/TF/JAX thread variables are copied literally, with JSON
  null for unset. The effective worker count is explicitly 'not measured';
  logical CPU count is not reported as a measured worker-thread count.
- Root key 20261025 and fold_in(index)->split(R)->split(U,C) are recorded,
  with fixed DGP truth, equal arm allocation and independent Uniform censoring.
  Exact parameter order, solver/starter/acceptance/no-event/inference policies
  and CSV conventions describe the existing reviewed computation only.
- Final JSON contains once-per-target calibrated limits and every group's
  indices/n/R/C, target/limit/realized censoring, and timing per model. The
  group times are shared C*R-fit measurements and throughput, per approved
  D-034; synchronized first call may include compilation, warm is median(3).
  All loops are host metadata/schema serialization with comments, not fitting.
- write_metadata preserves the pre-run snapshot rather than mutating it;
  JSON allow_nan=False rejects undefined metadata values. Payload is rendered
  before writing. Old metadata is invalidated only when CSV replacement starts;
  fitting/early validation failures preserve an existing complete result set,
  while partial CSV/JSON failures cannot leave old provenance certifying new
  files. Metadata is written only after write_results returns successfully.
- No supplemental per-scenario fit/timing loop, cache clearing, new policy,
  floors or numerical tolerance changes. No full R=1000 study run yet.

## Meaningful checks

- CPU probe fixture: parse supplied Linux-style model name, then return None
  when the probe and platform.processor are unavailable. This is fixture
  coverage, not actual Linux runtime evidence.
- Two host batches with different n/R/index ranges, reversed model timing
  magnitudes and distinct censoring fractions round-trip exactly through JSON.
  Check every scenario entry/model timing and preserve original source fields.
  Inject NaN into a scenario's metadata and require strict JSON rejection.
- Actual CLI smoke remains one subprocess and produces the two CSVs plus
  metadata.json. Compare metadata c/realized censoring/timings with every
  corresponding CSV row; require correct n/R/C, config, root seed, CPU backend,
  x64, exact actual Python argv, Git HEAD, lock hash, spec-seen, DGP and BFGS
  tolerances. Compare package versions and CPU brand to independent host
  queries. Child OMP_NUM_THREADS='2' is recorded and deliberately unset
  MKL_NUM_THREADS becomes null. Existing raw-flag/count and primitive-key
  checks remain green; no second smoke invocation just for metadata.
- Existing-file --out/config/x64 guards still reject before calibration;
  snapshot wrapper checks pre-work ordering. The raw CSV fixture begins with
  a stale metadata marker and verifies write_results removes it.

## Actual checks / environment investigation

Initial restricted optional run: 3 failed/5 passed in 2.47s, all from macOS
sysctl CPU query denied by the sandbox, before calibration/fitting. Retried
with host access as required; no CPU value was fabricated, probe skipped,
error relabelled or numerical tolerance changed. Focused metadata/export/CLI
retry: 8 passed in 8.15s, smoke 5.22s. A Ruff import-layout issue was fixed.

Final full explicit optional command (host access required for sysctl):
`uv run --locked --group examples env JAX_ENABLE_X64=1 pytest -q -s examples/weibull_simulation/check_metadata.py examples/weibull_simulation/check_results.py examples/weibull_simulation/check_cli.py examples/weibull_simulation/check_performance.py examples/weibull_simulation/check_scenario.py examples/weibull_simulation/check_fit.py`
51 passed in 55.51s; direct-script smoke 6.62s (concurrent full gate load).
`JAX_ENABLE_X64=1/0 uv run pytest -q tests/unit/test_simulation_generator.py`:
17 passed in 3.87s / 17 passed in 3.82s, no skips.
Ruff format: 147 files already formatted; lint: All checks passed!
`uv run mypy src`: no issues found in 22 source files.
`uv lock --check`: Resolved 171 packages, exit 0. Whitespace check clean.
`scripts/check_wheel.sh --through A10`: exit 0, final
`check: all gates passed through A10`. Own x64: 1282 passed/8 skipped in
128.95s; x32: 1287 passed/3 skipped in 120.42s. Doctests: 4 per mode in
1.03s/.93s. Independent x64: 2333 passed/27 skipped in 69.37s; x32:
2324 passed/36 skipped in 62.25s. Log /private/tmp/survjax-sim7b2-gate.log.
Fresh wheel built; installed wheel/extension checks included. Dedicated demo
checks are explicit and separate from this ordinary gate.

Actual host probe: Apple M4 Max, arm64, logical CPUs=16, CPU backend cpu:0;
Python3.12.9, JAX/jaxlib0.11.2, NumPy2.5.3, Optimistix0.1.0,
Matplotlib3.11.2, survjax0.0.1.dev0. Versions unchanged, no lockfile edits.

## Remaining scope

Full six-scenario evidence/results and figures, then executable docs. Current
metadata proves run completion/provenance facts, not release coverage or
figure acceptance. No merge/rebase/push/publication. Stop after this review.

## Round 1 response

F-1: share fixed DGP/BFGS/gradient constants in existing study.py, consumed
by generation, calibration, fitting, failure masks and metadata. Values and
all function signatures stay unchanged; no library API/configuration layer.
Add an actual-fit spy comparing options passed to Optimistix with metadata,
plus independent use-case assertions. This must catch a call-site override
as well as a changed setting; metadata must not merely assert its own literals.
N-2: CPU model query failures now return explicit None on every platform;
Git/package inspection remains required. No fallback CPU model is invented.
N-1: helper split is deferred; no new module is introduced in the fix-up.
Final full optional suite: 54 passed in 55.79s, smoke 7.24s, using the same
host-access command as the baseline. Additional solver-type and half-threshold
assertions: metadata file 5 passed in 2.87s. The real jitted-fit spy observes
BFGS type/rtol/atol/max_steps/throw and compares both metadata and independent
use-case literals; a direct call at half the metadata gradient threshold is
accepted. New denied-CPU cases exercise actual macOS nonzero-process and
Linux read-permission errors. Generator suites: 17 passed in 3.50s per mode.
Ruff format/lint, mypy (22 source files), lock (171 packages) and whitespace
checks clean. No core/packaging changes; the unchanged core gate was not
repeated for same-value demo constant substitution. Prior round's raw gate
limits remain separate from the successful fresh-wheel wrapper.
Reviewer raw gate: ordinary suites/doctests passed, x64 spec 2328 passed /
5 missing-SURVJAX_WHEEL setup errors; x32 spec not run. Its /tmp mutation
worktree cleanup remains reviewer/maintainer-owned.
