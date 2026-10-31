Independent review: round 1/3

# MNIST.7 — CLI, precision processes and CSV records

Read PLAN.md MNIST.7; spec/use-cases/03 main and details (Precision, Splits,
References, Files and commands); run.py; check_mnist_run.py. No core changes.

Scope split: CLI plus full reproducibility metadata would exceed the step
budget, so this step owns executable orchestration and metrics/history CSV;
metadata writer, figures and committed full results follow separately.
No docs page edit, dependency, protected file or spec change. No model.py growth.

Invariants:
- Standard-library-only parent launches CPU subprocesses sequentially; x64=1
  before fitting/calibration imports, x64=0 before neural imports. No in-process
  global precision toggle; subprocess failure aborts the pipeline.
- Calibration uses exactly 60k source digits. Generate globally keyed outcomes
  before slicing. Intercept fit sees selected train rows only. D-030's approved
  fitted-bias/shape initialization passes float64 logs into train's explicit cast.
- Train sees train/validation only. Selected CNN predicts on test in128-row
  batches with the tail included, preserving index order. Only outcome/oracle
  code sees digit labels; test likelihood/C cannot affect checkpoint selection.
- eqx.filter_vmap keeps CNN parameters in the pytree before filter_jit wrapping.
  The original inference-only jax.vmap wrapper emitted an Equinox warning;
  this final wrapper removes it without changing the model or training.
- Evaluate all three models through MNIST.6's public-likelihood/C path. Persist
  scalar aggregate metrics and per-epoch histories only. Temporary precision
  bridge JSON is fresh for each invocation and cleaned after children finish.
- Smoke/full flags select the fixed existing splits/epoch counts, never tune.
  This step does not claim full-run acceptance or a complete results directory.

Tests:
- Parent child-call checks pin CPU/x64 order, shared preparation path/config,
  requested output and failure propagation (failed prepare never trains).
- Controlled preparation checks pin 60k calibration rows and training-only fit.
- Controlled run checks pin training/validation observations and image alignment,
  supplied initialization dtype, selected predictions, 129-row tail/order,
  oracle label indexing, metrics model order and exact history content.
- In x64 mode the training child explicitly refuses before loading anything;
  the float32 process covers its pipeline. Existing optional suites cover the
  actual fit, train, evaluation math, keys and input-failure rules.

Actual smoke evidence: verified warm cache, fixed config, no tuning.
Initial run completed with CNN mean LL -1.1935250759, C .6301584244, k1.1871363;
intercept mean LL -1.2371411324, C .5; oracle mean LL -1.1055958271, C .6948196888.
CNN k is ~14% below1.38 and does not meet the 5% target on this one-epoch run.
This is an observation to carry into the prescribed10-epoch full run, not a
reason to choose a seed/checkpoint on test. No tuning undertaken. Ran the same
fixed smoke again after changing the inference wrapper; final result below.
Temporary output paths: /private/tmp/survjax-mnist-run7 and
/private/tmp/survjax-mnist-run7-final; no MNIST rows/results committed.

Checks/results appended below. No R generation, docs build or full run claimed.
Previous step final: 45850eb54d66d0a86d023c3aa658b8c5cfb72824.
Original worktree base: ee72c837549d14cb26e9983bf6d63ac54563f189.

Final smoke (warning-free filter_vmap wrapper): c=6.9890429382458406;
intercept logs [0.16221817172201644,1.4258202233235766]; selected epoch1.
CNN mean LL -1.1935251951, C .6301579475, k1.1871362925;
intercept mean LL -1.2371411324, C .5; oracle mean LL -1.1055958271,
C .6948196888. Training5.6983s; preparation11.1488s under concurrent checks.
History: trainingNLL1.2032518387, validationNLL1.2119510174. C call times:
CNN .3944s, intercept .1741s, oracle .1103s, each on2000 test rows.
The wrapper correction changes predictions only within float32 precision.

Optional seven-file suites: x64 77 passed/1 skipped (49.28s), float32
55 passed/23 skipped (37.55s). A focused float32 runner rerun after the
wrapper correction is recorded below. Ruff format/check clean; initial lint
found unused import/dict constructors, corrected before the cumulative gate.

Completed checks:
- Final-wrapper focused runner checks, float32: 4 passed (2.32s).
- scripts/check_wheel.sh --through A10: all gates passed. Ruff format 136 files
  already formatted; Ruff all checks passed; mypy no issues in21 source files.
  Own suites: x64 1119 passed/7 skipped (113.61s); float32 1125 passed/1
  skipped (83.23s). Doctests4 passed each. Acceptance: x64 2333 passed/27
  skipped (65.86s), float32 2324 passed/36 skipped (60.14s).
- Final Ruff and git diff --check clean.

During implementation, maintainer-staged CLAUDE.md (new remaining-substep
reporting rule) and .spec-seen updates appeared. They are outside this step
and remain untouched. Requested maintainer direction on a separate commit;
independent-review must wait for a clean working tree per its precondition.
Commit attempt revealed an active maintainer-started merge of
84d2943e788440db60504afb49a13360d43ad95e (new claude instruction), with no
unresolved conflicts. Git refused the scoped commit: "cannot do a partial
commit during a merge". No commit created. All runner files remain staged.
AGENTS forbids merging without maintainer direction; independent review has
not run. Await merge completion or explicit authorization to conclude it.
