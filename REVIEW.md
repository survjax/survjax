# SIM.integration — Maintainer-started main merge

Independent review is explicitly waived for this task: "i've merged in main
which moved a bit; handle merge conflicts. for this task don't invoke the
reviewer". No reviewer is invoked.

## Read first and commit trace

Read the newest PLAN section, then docs/index.md, scripts/build_docs.sh,
.gitignore and DEVIATIONS.md's ID reconciliation. Check the four simulation
check-file diffs: only deviation IDs in docstrings change.

Simulation parent / previous final: 61681c7c2f13d004eea3d248c4dac96a8c5cce89.
Incoming main: 23e8c3ef033e9c6c6e93c3fde21356626923c684.
Actual merge base: 963c7fc0e9b7d246272204eeb536462f29539694.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d;
previous integration base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Work stays on release-simulation. Complete the already initiated merge only;
no new merge/rebase, remote action or publication.

## Resolutions and invariants

Six unmerged paths are resolved. Keep main's new docs structure with the
accepted Weibull study under Use cases (simulation, MNIST, Bayesian, following
DOCS-MIGRATION's intended final order); the page stays non-orphaned and N-3
continues to describe data-file hashes precisely. Keep main's build script
byte-for-byte, preserving project/environment/output/cache isolation and both
locked optional groups. Preserve both the raw-replicate ignore and .jax-cache.

Retain every deviation entry body/decision/status. Main already uses D-033
for CI and D-034/D-035 for MNIST, so simulation failure accounting, shared
batch timing and compact evidence become D-042/D-043/D-044. Record the mapping
and update the active simulation checks' docstrings only. Executable ASTs
before/after these ID edits are identical. Historical plans/reviews retain
original IDs and are interpreted by that mapping.

Both complete parent PLAN.md files are preserved verbatim below the current
integration section. No unfinished history is dropped or archived as done.
Previous completed reviews remain in their parent Git commits; this file is
the current integration report. Cleanly merged core, independent spec tests,
lock/dependency changes (including main's removal of ArviZ) and other demos
are retained as supplied by main; they are not manually rewritten.

## Checks

uv sync --locked --group docs --group examples succeeded against incoming
main's lockfile: 166 resolved packages; removed ArviZ and its no-longer-needed
packages, retained the approved examples tools. No lockfile regeneration.
- bash -n scripts/build_docs.sh: passed.
- JAX_ENABLE_X64=1 and =0 uv run --locked --group test pytest -q
  tests/unit/test_docs_build_script.py: 7 passed in each mode (2.70 / 2.66 s).
- JAX_ENABLE_X64=1 uv run --locked --group examples --group test pytest -q
  examples/weibull_simulation/check_*.py: 72 passed in 60.42 s outside the
  sandbox. Initial sandbox run was 71 passed / 1 failed because the real CLI
  metadata check's sysctl query was denied; no code/tolerance change. An
  initial directory-only pytest command collected no check_ files (exit 5),
  so it was corrected to the explicit check_*.py selection before counting.
- scripts/build_docs.sh, run directly with approved escalation: exit 0,
  both modes build succeeded, 21 source pages including all three use cases.
  Simulation page execution 11.46 s / 8.63 s (x32/x64), live float64 fit
  accepted with HR .5794. No Sphinx warnings or notebook errors. The first
  kernel printed an IPython cache-parent fallback warning; stock TCP kernel
  banners also remain. Neither is hidden or relabelled as a Sphinx warning.
- One-off BeautifulSoup audit: both modes contain the Use cases group in
  simulation/MNIST/Bayesian order, all three targets exist, the simulation
  active-sidebar marker is present, local links/fragments resolve, five
  tables have 4/6/12/12/6 data rows, three SVGs/four downloads match original
  bytes, N-3 and live smoke/fit output are present, zero error/stderr blocks.
- Integrity audit: both parent plans are verbatim substrings; all parent
  deviation bodies/statuses are preserved with 44 unique IDs; no conflict
  markers or unmerged index entries. Incoming script, lock/config/protected
  files, src/survjax and tests/spec match main exactly. Only active-check
  docstrings changed, with identical executable ASTs.
- Resolution-specific git diff --cached --check against MERGE_HEAD passed
  on all 10 resolution paths. The broad cached diff against the simulation
  parent fails on inherited main whitespace: MNIST fig1/2/3 SVGs have
  1419/727/600 trailing-space lines, and one PH spec test has a blank EOF.
  These are unchanged from incoming main; protected tests/assets not edited.

## Cumulative gate and inherited acceptance failures

scripts/check.sh --fast --through A12: guard, Ruff format/lint (184 files)
and mypy (22 source files) passed. Regular tests: x64 1059 passed / 8 skipped,
x32 1064 passed / 3 skipped. Source doctests: 4 passed in each mode. The
script stopped at x64 spec tests: 2 failed / 3395 passed / 32 skipped.
Both failures are incoming PH acceptance tests against unchanged incoming core:
- test_a11_ph_construction.py::test_positional_baseline_is_rejected: PH
  accepts positional baseline; existing D-028 records the earlier explicit
  maintainer decision allowing that form. No new interpretation is made.
- test_a11_ph_parameter_issues.py::test_custom_baseline_condition_appears_with_the_prefix:
  actual condition [False, True, True], expected [False, True, False].

The first direct float32 spec follow-up omitted SURVJAX_WHEEL and therefore
reported 4 wheel setup errors alongside 3 PH failures (3384 passed, 38 skipped).
It was replaced with SURVJAX_X64_MODES=0 scripts/check.sh --fast --through A12
so the standard gate supplies its built wheel. That canonical float32 run
passed guard/Ruff/mypy, 1064 regular tests (3 skipped), and 4 doctests, then
reported 3 failed / 3388 passed / 38 skipped in 131.33 s, with no wheel setup
errors. The additional PH failure is log_cdf_grid reference comparison in a
float32 tail: JAX returns -0 where the float64 reference is -3.1677e-43 / -1.0069e-170.
Core and acceptance files are byte-identical to incoming main; no test is
skipped/xfail-ed, tolerance loosened, spec altered, or core fix attempted here.

## Remaining limitations

Browser visual inspection remains the prior external access limitation and
is not claimed as part of conflict resolution. Prose/formatting/dependency
nits stay assigned to the maintainer's separate work. No change to simulation
numerical semantics, key scheme, shared generator contract or compact results.


All six conflicts are resolved and staged. The conflict-resolution task has
no remaining substeps; the inherited PH acceptance failures above still need
separate maintainer/PH/spec-test reconciliation. The cumulative gate is NOT
reported as passing. Independent review was not invoked, as requested.
