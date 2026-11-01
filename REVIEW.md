Independent review: round 1/3

# MNIST.8 — Reproducibility metadata

Read PLAN.md MNIST.8, use-case03 details Files/commands, Precision, Splits/keys,
Outcome generator, Training and References/metrics; run.py metadata dictionary
and prediction timing; check_mnist_run.py controlled pipeline assertions.

Invariants:
- Metadata only: no optimizer, model, outcome, split or selection changes.
  No core API, module, dependency, protected file or spec edit. D-030 stands.
- metadata.json retains c and fitted log parameters from the temporary float64
  preparation bridge and oracle constants computed/stored in host float64.
  Outcomes/training/evaluation remain float32 in the separate CPU child.
- Cohort sizes and realized censoring use actual selected indices/events, not
  advertised defaults or mean times. Test labels remain oracle-only after
  checkpoint selection. Metadata contains no image/label/prediction rows.
- Hashes come from the loader's SHA256-verified resource table. Runtime versions
  come from installed package metadata, CPU environment from platform/os/JAX.
  Settings and key descriptions document the fixed existing implementation.
- Preparation excludes MNIST loading; training includes initialization/compile/
  validation. Prediction blocks on the full concatenated result before its end
  timestamp. C timings are copied from each evaluate result, which synchronizes.
  All timing boundaries and possible first-call compilation are labelled.
- Strict JSON refuses nonfinite values; selected epoch matches the returned
  validation checkpoint. CSVs remain the existing aggregate metric/history form.

Test evidence:
- Extend the existing controlled float32 runner path with alternating event
  flags: train128/validation2 each half censored; test129 has64 censored/65 exact.
  Fractions are hand-counted, including the tail, independently of writer code.
- Controlled timer readings10/14/20/23 require training4s and prediction3s;
  preparation1s comes from the independent bridge fixture, C .1s per model
  from the existing metric collaborator. Checks pin timing provenance/labels.
- Controlled version/CPU queries pin runtime record sources. Explicit expected
  settings, root/derived keys, fit/oracle parameters, selected epoch, precision,
  cohort sizes and verified hash fields pin all reproducibility requirements.
- Existing pipeline checks still pin fit/training/oracle shape arguments,
  generation inputs, aligned cohorts, ordered/tail predictions and CSV content.
  x64 runner test checks the child refuses rather than silently promotes.

Actual smoke: fixed configuration, verified warm cache, temporary output only
at /private/tmp/survjax-mnist-metadata8. JSON/CSV agreement check passed for
all three model cohort sizes/C times, history length/selected epoch and realized
censoring fractions {train:.5022, validation:.499, test:.5045}.
Metrics reproduce MNIST.7: CNN meanLL -1.1935251951, C .6301579475,
k1.1871362925; intercept meanLL -1.2371411324, C .5; oracle meanLL
-1.1055958271, C .6948196888. The one-epoch shape remains ~14% low; no tuning.
This is not full-run acceptance. C timing here is n=2000, not10k.
Preparation4.9828s, training3.1311s, prediction .1123s under concurrent checks;
C times CNN .2487s/intercept .0910s/oracle .0802s. No performance target claimed.
Environment recorded: Python3.12.9, JAX/JAXlib .11.2, NumPy2.5.3,
Equinox .13.8, Optax .2.8, Optimistix .1.0, CPU arm16 logical cores/cpu:0.

Checks: optional seven-file suites x64 77 passed/1 skipped (28.23s), float32
55 passed/23 skipped (20.58s). Focused float32 runner4 passed (1.43s).
Ruff lint passed. Cumulative fresh-wheel gate results appended below.
No docs build, R fixtures or full run. Remaining sub-steps: figure writer,
fixed full run/committed results, executable page. No new deviations/questions.
Previous step final: d8af61a53bb2c7f4627d1de3df057fb5197234e2.
Original worktree base: ee72c837549d14cb26e9983bf6d63ac54563f189.

Cumulative check: scripts/check_wheel.sh --through A10 exit0, all gates passed.
Ruff format136 files already formatted; Ruff all checks passed; mypy no issues
in21 source files. Own suites x64 1119 passed/7 skipped (73.35s), float32
1125 passed/1 skipped (70.92s). Doctests4 passed each mode. Acceptance suites
x64 2333 passed/27 skipped (64.86s), float32 2324 passed/36 skipped (59.33s).
Final format and git diff --check clean.
