# MNIST.5 — Float64 intercept-only reference fit (2026-10-31)

Start clean on release-minst at c035ed1; maintainer approved MNIST.4 and
requests the next bit. MNIST.4 round2 returned merge; F-1/N-1/N-2 fixed.
D-028 placement acceptance remains recorded, with formal status maintainer-owned.

One concept: intercept-only MLE; keep metrics separate so model.py stays
under about 300 lines and precision/process boundaries remain reviewable.
Spec 11.2, 7.2–7.3, 10.3; 03 details References and metrics, Precision;
release-agent MNIST handoff (float64 fit, float32 neural/evaluation process).
No library API, module, dependency, protected or spec change.

Files:
- examples/survival_mnist/model.py: private mean Weibull NLL and
  fit_intercept(time,event) host entry returning float64 log-parameter array
  (log k, log b). Require x64 mode explicitly; never toggle global precision.
  Supplied training rows only, matching nonempty vectors, public SurvivalData
  validation and at least one event. Start k=1, b=sum(observed time)/events,
  with event count cast to float64 before division. Fixed approved Optimistix
  BFGS minimizes public survjax likelihood in log parameters, rtol/atol=1e-10,
  max_steps256, with a bounded roundoff-aware Armijo comparison near stationarity.
  Explicit failure on solver failure/nonfinite results or
  gradients; no floors, clipping, penalty likelihoods or fallback fits.
- tests/unit/check_mnist_reference.py: explicit examples-group offline checks.
  Float64 MLE against an independently derived mpmath profile-score root,
  all-observation/censoring influence, shape/invalid-row/no-event failures,
  unsuccessful and nonfinite optimizer-result paths. Explicit float32-process
  refusal, with unavailable-float64 cases skipped there for a stated reason.
  Existing smooth likelihood derivatives already tested; fit is host-only,
  no optimizer-gradient/JIT API promise. Recheck fitted log parameters cast
  explicitly for float32 evaluation in a separate process.
- README.md: exact fit invocation and process boundary; fitting checks.
- REVIEW.md: overwrite with reading order, invariants, evidence and limits.

Target <=300 changed source/test/README lines, model.py <300. No CNN training
change, metric helper, test-set evaluation, CLI, full-run result, figure or docs
page. Docs to-do: float64 training-only reference fit and explicit evaluation
cast. Later steps still own oracle/C metrics, 10k C timing, calibration/metadata
writer, full runs and executable page. No additional fit tuning on test data.

Checks: fitting checks in separate x64/float32 processes, existing explicit
neural/preparation checks, Ruff/mypy. Actual warm-cache full 50k training-only
fit in x64 with recorded parameters, stationarity and timing; save temporary
fit output, then a separate float32 process proves evaluation conversion.
Fresh-wheel scripts/check_wheel.sh --through A10. Ordinary tests stay offline
and demo-free. Commit, independent-review skill MNIST.5 (max3 rounds with
fix-ups), then stop after verdict; do not begin metrics until told.

Numerical implementation note: first probe used a mistaken Solution.y field;
installed Optimistix documents Solution.value, now corrected. Default reverse-
mode BFGS converged with log-shape errors 5.87e-9/9.63e-9, outside the existing
relative tolerance (no tests/tolerances loosened). Tightening solver tolerances
1e-12→1e-14/1e-15 did not change those results. Installed BFGS supports
options.autodiff_mode: forward AD for two parameters avoids a large reverse
tape and independently improves score residuals from ~1e-8 to <=1.3e-10 in
these probes. Use that supported option, retaining rtol/atol1e-12, defaults
otherwise and all original assertions. Add a right-censor-at-zero analytical
case (zero cumulative hazard term), without changing existing expected values.

Full training-only diagnostic: forward BFGS at rtol/atol1e-12 reaches the
same log parameters [0.1719941343787224,1.4325235160420127] and stationarity
~2.1e-14 as 1e-10, but refuses to terminate and hits256 steps. The 1e-10
criterion reports success in9 steps. Use fixed solver rtol/atol1e-10 for
termination; all analytical parameter-validation tolerances/assertions remain
unchanged. Solver failures remain failures (never accept a max-step status).
This is solver accuracy/roundoff diagnosis on training data, not test-metric
or seed/checkpoint tuning. Record initial failed probes honestly in REVIEW.md.

Implementation/evidence: source/test/README238 lines; model.py247 lines.
Forward-AD BFGS with fixed1e-10 termination passes all unchanged 50-digit
profile-MLE assertions and second-order objective checks. Actual full50k
training-only fit returns log parameters [.1719941343787224,1.4325235160420127]
(k1.1876708667659677,b4.189257522824442), NLL1.234515803406909, gradients
<=2.1e-14; CPU1.51943s including compile. Separate float32 process converts
stored logs and evaluates finite float32 likelihood with no precision toggle.
Combined checks:50 passed/1 skip x64 (18.21s),35 passed/16 precision skips
x32 (13.99s). Fresh-wheel gate exit0: own1087/1093 passed (7/1 skips),
doctests4 per mode, acceptance2333/2324 passed (27/36 skips), Ruff/mypy pass.
REVIEW.md records initial API/numerical failures, diagnosis, exact commands
and limits; no test assertions/tolerances loosened, no test-set model tuning.
Commit, invoke independent MNIST.5 round1, then stop before metric helpers.

Round1 F-1:1e-10 still stalls on the real5k smoke cohort and valid exact-only
and seeded n10 samples. The full50k probe missed this coverage gap. Near an
optimum, mean-NLL changes are quadratic in parameter errors, so float64
objective resolution limits loss-based line searches around sqrt(eps64).
Investigate fixed1e-8 BFGS termination (rtol/atol), retaining all prior accuracy
assertions and strict solver-status failure. Extend the same independent
profile-score oracle to exact-only [1,4,8] and fixed-seed small mixed/exact
cohorts; do not drop or loosen the original cases. Verify both real5k/50k
training-only fits and separate32bit conversion. N-1: update the Files bullet
in place to the final setting. Update REVIEW.md with all outcomes. Focused
checks both processes/static checks; no repeated core/wheel gate for this
example-only fix. Commit then exact MNIST.5 round2, max3 rounds total.

Fixed1e-8 resolves most new controls but seeded exact n10 still stalls, despite
accepted-iterate gradient max6.14e-10. Public Armijo slope1e-4 and RMS/two-norm
probes do not fix it. Source diagnosis: Cauchy termination is checked only on
accepted steps; Armijo can keep rejecting near roundoff at an already stationary
iterate. Implement one private concrete BFGS specialization with an additional
accepted-gradient max-norm<=1e-9 termination condition, guarded against the dummy
initial gradient. Preserve stock BFGS steps/search and result status; never
reclassify failed/max-step solves. This is the required BFGS fitter's stopping
rule, not a new optimizer framework, public API, module or fallback algorithm.
Keep all MP assertions unchanged and verify the broader controls plus5k/50k.

Gradient-only termination is insufficient for robust high precision: smoke's
accepted score max1.06e-8 and predicted correction [3.24e-9,-1.64e-8] still
need a final step. Replace that experimental specialization with a concrete
roundoff-aware Armijo search attached to stock BFGS. Keep its normal1e-10
Cauchy termination. Only when accepted gradient max<=sqrt(eps64), descent is
nonpositive/tiny and both objective values finite, permit a full trial whose
objective difference is within4*eps64*(1+abs(f)+abs(f_trial)). This changes only
the optimizer comparison, never likelihood/data/parameters. Reset step state
consistently on acceptance. Strict failure/max-step guards remain. Add direct
search checks for roundoff, real worsening, nonfinite trial and large gradient,
alongside unchanged MP regressions. No gradient-only early exit or fallback.

Final F-1 resolution: retained normal BFGS1e-10 termination; attached a private
roundoff-aware Armijo search. Only finite full-step trials, nonpositive descent
with predicted change below4*eps64*(1+abs(f)+abs(f_trial)), and accepted gradient
max<=sqrt(eps64) can use the allowance. Real increases, nonfinite trials, large
gradients and already shortened steps remain rejected; coherent search state
is tested. No likelihood value/data/parameter change or failed-status promotion.
All8 independent MP cases pass at their original assertions. Both real fits
now succeed: smoke5k logparams [.16221816847747333,1.4258202396725628], gradients
<=1.8e-15,1.42978s; full50k original logparams unchanged, gradients<=2.1e-14,
1.32151s. Both convert/evaluate float32 in a separate process. Combined focused
checks:59 passed/1 skip x64 (19.90s),35 passed/25 skips x32 (13.93s); Ruff/mypy
pass. All source files remain <300 lines (model292, checks223). Record exact
comparisons/limits in REVIEW.md, commit and request MNIST.5 round2, then stop.

Round2 on7a8fd27: F-1 fixed, including both real cohorts and60 reviewer stress
cohorts with no solver failures. F-2: add explicit rejection cases for large
predicted descent with worsening trial and for positive predicted direction;
assert accepted step resets to step_init. N-2: document sqrt(eps64)-scale
accuracy limits (reviewer found one stock-BFGS stopping point1.14x default
parameter tolerance); no further tuning/test relaxation. N-3: explain why
gradient/allowance guards exist in code comments. S-1 is logged as open D-029:
maintainer confirmation of the private modified Armijo under prescribed BFGS.
Do not alter solver/spec in response; finish authorized F-2 test fix and round3,
then stop for that decision before metrics. Targeted suites/static checks only;
no numerical logic or core changes. This is the last allowed review round.

F-2 validation: final suites61 passed/1 skip x64 (20.76s),35 passed/27 skips
x32 (14.01s); Ruff/mypy clean. Only tests, explanatory comments and records
changed. Commit and invoke final MNIST.5 round3, then stop for D-029 decision.

---

# MNIST.4 — Deterministic epoch training and checkpoint selection (2026-10-31)

Start clean on release-minst at 3d99b3d. Maintainer reviewed/agreed and asks
for next step, accepting D-028 preparation placement and the reviewed code.
Record that decision in D-028 without marking it resolved (maintainer-owned).
No spec edit is requested. Explicit optional checks follow the existing handoff.

One concept: host epoch orchestration with compiled float32 minibatch updates.
Spec 11.2, 7.2–7.3, 10.3; 03 details Precision, Splits and keys, Training.
No core API, dependency, protected, docs, generator or placement changes.

Files:
- examples/survival_mnist/model.py: train(train images/data, validation
  images/data, config=smoke/full), typed result with selected parameters,
  epoch train/validation NLL columns and 1-based selected epoch. Scalar
  log_k=0; CNN bias uses training observed mean only; init fold_in(uc,2).
  Adam(1e-3), batch128, epochs1/10, drop training remainder. Shuffle with
  fold_in(fold_in(uc,3), zero-based epoch). eqx.filter_jit updates and
  filter_value_and_grad; explicit finite loss/gradient host checkpoints on
  the first actual minibatch at initialization and after update1. Float32
  cohort inputs required; invalid data rejected at host entry. Full validation
  includes its remainder and is accumulated in float32, in bounded batches.
  Retain earliest strictly best finite validation result; epoch histories
  record mean minibatch training NLL and full validation NLL. No test inputs.
- tests/unit/check_mnist_training.py: explicit examples-group offline checks:
  independent constant-exponential validation mean including tail, instrumented
  10-epoch schedule/key/batch/initialization/checkpoint/history tests, initial
  and post-update nonfinite failures and float32/shape/config entry checks.
  Existing neural checks cover actual Adam/gradients; run actual one-epoch
  smoke training on verified warm-cache MNIST as separate integration evidence.
- README.md: training call and explicit optional check command.
- DEVIATIONS.md: dated D-028 acceptance note, status not changed.
- REVIEW.md: overwrite with reading order, invariants, evidence and limits.

Aim <=300 source/test/README changed lines. No CLI/metadata writer,
intercept/oracle metrics, full training results, figures or docs page yet.
Docs to-do: smoke training/history/selection and explicit runtime diagnostics.
Later run.py still owns calibration comparison/writer from MNIST.2 N-2.

Checks: optional suites in separate x64/float32 processes (neural work always
float32), Ruff/mypy; warm-cache 5k/1k one-epoch smoke with recorded timings,
finite gradients/history/selected model (no test-set model selection). Fresh
wheel scripts/check_wheel.sh --through A10. Ordinary tests stay offline and
need no demo packages. Commit and exact independent-review for MNIST.4,
max 3 rounds with fix-ups; stop after this step's verdict.

Implementation/evidence: 318 source/test/README changed lines; model.py 184
lines and optional training checks 186. Final combined suites before row-
marker strengthening: 34 passed/1 precision skip x32 (19.66s), 35 passed x64
(20.39s). Final strengthened training tests: 8 passed/1 skip x32 (6.48s),
9 passed x64 (8.06s). Actual fixed warm-cache smoke: 5k/1k, one epoch,
39 minibatches, 8 rows dropped, selected epoch1; required gradient checks
passed, training/validation NLL 1.244823575/1.246947408, CPU training incl.
compile 4.66429s. No test cohort used and no full-run acceptance claim.
Fresh-wheel gate exited 0: own tests 1087/1093 passed (7/1 skips), doctests
4 per mode, acceptance 2333/2324 passed (27/36 skips). Only optional-test
row markers and review-record formatting changed after that gate; targeted
checks pass. Exact commands/evidence/limits in REVIEW.md. Record D-028
acceptance without changing its formal status. Commit, independent MNIST.4
round 1, then stop after verdict.

Round 1 on 3f1f36c returned fix-first for F-1: symmetric event flags made
observed and event-only time means equal (129). Change fixture events to
rows 1..100 (event mean 50.5), retaining hand-derived observed mean129.
N-1: share one fixed Adam transformation for init/update, with all state
still local per run. N-2: comment the zero-based shuffle/one-based selected
convention. D-028 acceptance is recognized; no outstanding question there.
Run all optional/preparation checks in both modes, static checks, and repeat
the warm-cache smoke to compare metrics against the original. No duplicate
core/wheel gate for this example-only small refactor/test strengthening.
Commit fix-up and invoke independent MNIST.4 round 2, then stop/report.

Fix-up validation: combined suites 34 passed/1 precision skip x32 (14.91s),
35 passed x64 (15.07s); Ruff/mypy clean. Repeated actual smoke after shared
optimizer change is exactly identical in histories/selected epoch/shape
(2.37785s CPU wall time). Source/test/README total321 changed lines after
this fix-up. Commit and request MNIST.4 round2, then stop after verdict.

---

# MNIST.3 — Float32 CNN and likelihood loss (2026-10-30)

Start clean on release-minst at f232294. Maintainer approved MNIST.2 and
requested the next bit. MNIST.2 round 2 returned merge (F-1 fixed;
N-1 preconditions accepted, N-2 metadata writer remains for run.py).
Continue MNIST; preserve all current and inherited plan records below.

One concept: CNN/loss integration, before the epoch runner. Spec 11.2,
10.3, 7.2–7.3; 03 details Precision, Network, Training (one update only).
No library API, dependency, protected path, shared generator or docs edits.

Files:
- examples/survival_mnist/data.py: move the existing outcome constants and
  helpers here unchanged, keeping ordinary preparation checks free of demo
  dependencies. Existing model.py helper names remain re-exported.
- examples/survival_mnist/model.py: exact Equinox CNN, explicit float32
  Conv2d/Linear parameters, pool stride 2 for the specified geometry, scalar
  per-image log scale; final bias log(mean observed training time). CNN
  accepts an explicit initialization key (runner will use fold_in(uc,2)).
  Plain loss((cnn,log_k), images, data) uses vmap, Weibull.from_log and mean
  negative public likelihood. Runner will initialize scalar log_k=0.
- tests/unit/test_mnist_outcomes.py: import preparation from data.py.
- tests/unit/check_mnist_model.py: explicitly invoked optional-environment
  checks, not silently skipped or included in ordinary pytest discovery.
  Synthetic images/observations only: architecture/dtype/batch geometry,
  analytic constant-head exponential loss and derivatives, second-order
  forward/reverse checks on smooth head parameters, filtered-jit agreement,
  finite gradients before/after one Adam(1e-3) update with data[idx].
- README.md: preparation import update and exact optional-check commands.
- REVIEW.md: overwrite with reading order, invariants, evidence and limits.

Aim about 300 changed lines including mechanical move/tests. No training
loop, checkpointing, smoke/full run, reference fit or figure in this step.
Docs to-do: model/loss explanation and finite-gradient checks in live smoke.
Next training step must implement loud finite-gradient failures, prescribed
initialization/shuffle keys, epoch/batch budgets, validation-only selection
and history. MNIST.2 N-2 metadata writer remains assigned to later run.py.

Checks: focused ordinary preparation suite in separate dtype processes;
explicit JAX_ENABLE_X64={0,1} uv run --group examples pytest -q
 tests/unit/check_mnist_model.py (float32 neural arithmetic in both);
Ruff/mypy and fresh-wheel scripts/check_wheel.sh --through A10. No network
or real MNIST needed for this step. Record exact outputs/versions, commit,
invoke independent-review skill for MNIST.3 (max 3 rounds/fix-ups), then stop.

Implementation complete: CNN/loss and explicit optional checks, with unchanged
outcome helpers moved to data.py. Raw source/test/README diff 318 changed
lines including the mechanical move (new model module 65 lines). Initial
checks incorrectly included MaxPool Python metadata/-inf initializer; fixed
checks to select inexact arrays, with no runtime or tolerance change. Neural
checks pass 3 per mode (15.68s/12.09s); ordinary preparation 22 per mode
(6.68s/7.03s). Isolated offline test environment confirms preparation works
without Equinox/Optax. Fresh-wheel through-A10 gate exited 0: own suites
1087/1093 passed (7/1 skips), doctests 4 per mode, acceptance 2333/2324
passed (27/36 skips); Ruff/mypy clean. Exact commands and limits in REVIEW.md.
Commit, invoke independent MNIST.3 round 1, then stop after verdict.

Round 1 returned fix-first for F-1: a missing dense ReLU escaped checks. Add
an independent NumPy forward reference; remove unused re-exports (N-1) and
rename the preparation test alias (N-2). N-3: next runner must construct data
from float32 outcomes. S-1 invocation already follows the handoff's explicit
optional-check instructions; future standing-gate policy is maintainer-owned.
Record the file-role question as open D-028 without further layout/spec changes.
Complete the authorized test fix and review, then stop for maintainer's layout
decision before training. Focused suites in both modes and static checks;
no repeated full gate for these test/import/record-only changes. Commit a
fix-up and invoke MNIST.3 round 2, then stop/report all open questions.

---

# MNIST.2 — Fixed synthetic outcomes and censoring calibration (2026-10-30)

Start clean on release-minst at e63220b; maintainer approved MNIST.1 and
requested the next step. MNIST.1 round 2 returned merge (F-1/N-1 fixed;
optional N-2/N-3 remained). Continue the same MNIST sub-slice; retain its
sections and inherited unfinished SIM/A9 records below.

One concept: outcome preparation, before network/training/reference fitting.
Spec 11.2, 10.3; 03 details Precision, Splits and keys, Outcome generator;
00-common shared generator. No library API, dependency or protected edits.

Files:
- examples/survival_mnist/model.py: demo-only float64 oracle log-scale table,
  host calibration from exactly the 60k training-source labels with the
  unchanged shared calibrator; float32 outcomes via the shared generator,
  one fold_in(fold_in(fold_in(key(20261025),3),0),global_index) per image.
  Explicit indices permit checking subset/reordering stability; the eventual
  runner generates all 70k outcomes before splitting. Expose existing shared
  GeneratedData result, avoiding a new container or core sampling API.
- examples/survival_mnist/calibration.json: measured c, float64 oracle table,
  aggregate training digit counts, input hashes, versions, key scheme and
  realized full/smoke censoring fractions from one fixed run. No image rows.
- examples/survival_mnist/data.py and tests/unit/test_mnist_data.py: resolve
  prior optional N-2/N-3 with stable host integer dtype and the reviewed
  JAX 0.11.2 validation-prefix regression anchor.
- tests/unit/test_mnist_outcomes.py: offline synthetic digit labels; analytical
  oracle formula, independent high-precision quadrature of calibrated mixture,
  training-source-only calibration inputs, direct per-index U/C key algebra,
  eager/jit/vmap agreement and reordered/subset outcome identity; float32
  fields in both modes, valid SurvivalData and finite oracle likelihood.
- README.md: minimal preparation recipe and recorded calibration provenance;
  overwrite REVIEW.md with reading order, invariants, evidence and limits.

Target about 300 changed source/test/artifact lines; no MNIST data, network,
CNN, optimizer, full result claim, docs or shared generator/lock edits. Docs
to-do: generating rule, calibration c, split censoring and float32 exception.
Checks: focused tests both processes, Ruff/mypy; real warm-cache 70k outcome
probe (float32 draws, NumPy float64 calibration/oracle); independent mpmath
mixture integral; fresh-wheel scripts/check_wheel.sh --through A10. Keep
ordinary tests offline with synthetic labels only. Record exact evidence,
commit, invoke independent-review skill for MNIST.2 (max three rounds with
fix-ups), then stop after the verdict.

Implementation/evidence: float64 c=6.9890429382458406, independently checked
with 40-digit mixture quadrature (.5000000000001933). All 70k outcomes were
generated once in float32; public data validation passed; full censoring
fractions train/validation/test .50036/.5079/.4995 recorded with metadata.
Offline focused suites 22 passed per mode (5.54s/5.00s). Fresh-wheel gate
exited 0: own tests 1087/1093 passed (7/1 skips), doctests 4 per mode,
acceptance 2332/2322 passed (28/38 skips), Ruff/mypy clean. Initial README
blank-line formatting failure was corrected before the successful gate.
Source/test/artifact diff about 276 lines. Exact commands/timings/limits in
REVIEW.md. Commit then independent MNIST.2 round 1; stop after verdict.

Round 1 returned fix-first for F-1: dtype assertions lacked a strongly
typed float64 c input. Pass np.float64(7.0) in the existing per-key/dtype
test, retain all dtype assertions, run focused checks in both modes and
static checks, commit a test-only fix-up and request round 2. No runtime
change; prior fresh-wheel gate remains applicable. Optional N-1 preconditions
remain documented. N-2 to-do for the eventual run.py step: commit the metadata
writer and compare recomputed c/fractions with calibration.json. Reviewer
independently reproduced all stored values and cross-mode identical outcomes.

---

# MNIST.1 — Verified data acquisition and deterministic splits (2026-10-30)

Start clean on maintainer-created release-minst at ee72c83. Shared generator,
examples dependencies and A10 are integrated. Preserve unfinished inherited
SIM/A9 records below; do not archive another workstream as completed.

One concept: data loading, before outcomes/calibration or neural training.
Spec 10.3, 11.2, 12.2; use-cases/03 main and details Data acquisition,
Precision, Splits and keys. No library API, dependencies or protected paths.

Files:
- examples/survival_mnist/data.py: published compressed IDX filenames/MD5s,
  first-fetch SHA-256 values measured after MD5 verification, checked cache,
  strict IDX parsing, float32 NCHW pixels, fixed full/smoke global indices.
  Default cache SURVJAX_EXAMPLES_DATA or ~/.cache/survjax-examples/mnist.
- examples/survival_mnist/README.md: attribution, checksum source, usage,
  cache policy and honest pending implementation stages.
- tests/unit/test_mnist_data.py: tiny synthetic IDX bytes and mocked network;
  malformed format/counts/labels, corrupted cache/download, cache reuse,
  pixel preprocessing, deterministic disjoint split/smoke prefix checks.
- REVIEW.md: overwrite with reading order, invariants, evidence and limits.

Target <=300 source/test/README changed lines; split further if needed.
No outcomes, model, full results or docs in this step. Docs to-do: attribution,
cache/network requirements and live smoke page when remaining steps are ready.
Checks: offline focused tests in both dtype processes, Ruff, mypy, existing
through-A10 gate; separate real four-file acquisition/cache-reuse probe in
/private/tmp (never commit data). Commit then exact independent-review skill
for MNIST.1, at most three rounds, fix-up commits as needed; stop and report.

Implementation complete: 296 source/test/README lines; real four-file MD5
verification and SHA-256 recording, two real cache loads, offline 11-test
suite passing in both modes. Ruff/mypy pass. Direct cumulative gate hit five
missing-SURVJAX_WHEEL setup errors; built a fresh wheel and reran acceptance
in both modes successfully (2332/2322 passed, 28/38 skips). Prior own suites
1076/1082 passed (7/1 skips), doctests 4 per mode. Exact commands, times and
limits are in REVIEW.md. Commit and independent round 1 follow; stop after
the verdict before outcomes/calibration.

Round 1 returned fix-first for F-1 (default cache path test gap). Extend the
existing synthetic load check to exercise unset SURVJAX_EXAMPLES_DATA and
assert the expanded home path without any writes or HOME changes. Clarify
N-1's fresh-download failure message; leave optional N-2/N-3 unchanged.
Run focused tests per dtype and static checks, commit the fix-up, then request
round 2. No repeat full gate: no parsing, split or packaging change.

---

# SIM.3 — Complete the approved examples diagnostics environment (2026-10-30)

Start clean on release-simulation at
`ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d`; SIM.2 round 1 returned merge.
Preserve the unfinished SIM/A10 plan history below. The maintainer approved
ArviZ 0.23.0 in the existing dependency proposal and now requests the next step.
One concept: finish the examples dependency group for Bayesian tail ESS.

Files and scope:
- `pyproject.toml` / `uv.lock`: add exact `arviz==0.23.0` to examples;
  resolve without upgrading existing packages or changing runtime dependencies.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark the six-package setup
  complete, record the actual installed diagnostics probe and its command.
- Overwrite `REVIEW.md`: reading order, dependency isolation, actual evidence
  and limits; preserve the short temporary probe source for reproducibility.

Spec 10.3; shared use-case 00 dependency-group rule and Bayesian details
Sampling/Diagnostics: NumPyro split R-hat/ESS, ArviZ tail ESS. No source API,
shared generator, ordinary tests, docs/navigation/config or protected scripts.
Docs to-do: the future Bayesian page should use this group for tail ESS.
Target below 300 changed implementation/config/lock lines, records excluded.

Checks: `uv lock --check`, `uv sync --locked --group examples`, compare old/new
lock name/version mappings. Explicit optional-environment probe uses seeded
4x1000 iid Normal draws (known equal-chain law) and deliberately shifted
chains to verify the chain/draw axes, NumPyro summary and ArviZ bulk/tail ESS
and R-hat; this is compatibility evidence, not posterior convergence evidence.
The pinned ArviZ import writes a daily stamp under Path.home()/arviz_data, so
run that probe with required sandbox escalation rather than modifying HOME.
No new ordinary tests for the dependency-only change. Run the existing focused
generator suite in both dtype processes, Ruff, mypy, and unchanged fresh-wheel
through-A10 gate. Record versions/results, write REVIEW.md, commit, invoke the
exact independent-review skill for SIM.3, follow at most 3 rounds and stop.
PH fitting still waits for maintainer integration of reviewed prerequisites.


Implementation: added the approved ArviZ pin and six lock packages. All 165
prior package versions unchanged; runtime dependencies remain jax/numpy.
Actual installed-group CPU probe passes named chain/draw axes, iid gates and
shifted-chain detection: NumPyro R-hat 1.0000270083 / ESS 3976.58; ArviZ
R-hat 1.0001031616 / bulk ESS 3979.33 / tail ESS 3931.29. ArviZ's expected
import warning was emitted; sandbox escalation allowed its home-cache stamp.
Existing generator suites: 17 passed per mode (3.60s/3.18s), Ruff/mypy/lock
checks pass. Fresh-wheel through-A10 gate exited 0: own suites 1065/1071
passed (7/1 skipped), doctests 4 per mode, independent 2333/2324 passed
(27/36 skipped). ArviZ's pandas dependency enabled the existing x32 pandas
conversion test. Wheel hash is identical to SIM.2 and runtime METADATA still
requires only jax/numpy. REVIEW.md records exact commands, outputs and limits.
Commit and exact independent review follow; stop after the verdict.

---

# SIM.2 — Address shared foundation questions and enable examples (2026-10-29)

Start on `release-simulation` at `3556d12c1abca553f77607c1e1739df2bd1911f0`.
The only initial dirty file is study.py with three maintainer question comments.
SIM.1 round 2 returned merge. Preserve all unfinished history below.
Maintainer approves the proposed dependencies in this turn; no renewed
approval is needed for those exact pins. Keep PH fitting deferred.

Files and scope:
- `pyproject.toml` / `uv.lock`: examples group with exact tested Optimistix
  0.1.0, Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0.
  Keep existing runtime/dev/docs versions unchanged. The five-package lock
  preview is 160 changed lines; full six-package setup is 295 lock lines
  alone. Approved ArviZ 0.23.0/tail ESS is a separate later step so this step
  including generator checks remains about 300 lines. No protected scripts.
- `examples/weibull_simulation/study.py`: answer/consume the maintainer notes,
  define quadrature rules before their closure, and explain calibration as
  once-per-target host setup outside the compiled replicate loop. PH input
  preference is asked asynchronously (baseline/log-HR versus PH rate m).
  Preserve the current call contract; any PH convenience will be optional,
  demo-only, and checked against the existing scale formulation.
- `tests/unit/test_simulation_generator.py`: if adding PH input, analytical
  PH/scale and calibration identities, row broadcasts and transformed values/
  derivatives. Reuse existing tolerances. Existing numerical checks also
  exercise the quadrature block reorder.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark approval, staged pins,
  remaining approved ArviZ stage and exact optional-environment invocation.
- Overwrite `REVIEW.md` with reading order, input-contract updates, evidence
  and limits, preserving SIM.1's outcome in this plan and Git history.

Spec 4.4/4.6/7.3/10.3; use-cases/00-common and 01-details DGP/calibration.
No library API, shared docs/config or other workstream files. Docs to-do:
future simulation page should show PH inputs and setup versus compiled work.
Run generator checks in separate x64/float32 processes, approved-environment
compatibility probe, Ruff, mypy, lock consistency and fresh-wheel through-A10
gate. Inspect size before committing; split if needed. Commit, invoke exact
independent-review skill for SIM.2 (max 3 rounds), then stop/report.


Implementation: optional log_hazard_ratio=0.0 added to both generator and
calibration, with unchanged no-keyword calls and U/C key split. Rules now
precede the fraction closure; calibration explicitly remains host-only setup.
No response to the optional PH-form preference arrived during environment
work; used the stated baseline/log-HR convention from 01-details and spec 4.4.
Approved five-package group resolves/installs; all 155 existing locked
versions unchanged, 10 additions. Optional compiled BFGS/NN-update/NUTS/SVG
probe passes in the actual group environment. ArviZ remains approved for its
separate lockfile step. Total implementation/test/config/proposal diff is
303 changed lines, including 160 lockfile lines.

Focused final suites: 17 passed per dtype (10.17s/10.56s), Ruff/mypy/lock
checks pass. A joint three-parameter check_grads initially failed a float32
second-order projection near zero; no tolerance was changed. Retained the
original k/b finite-difference checks and verified all PH gradient/Hessian
entries against inverse-CDF algebra, including cross-partials. The full
fresh-wheel through-A10 gate exited 0: own suites 1064/1071 passed (7/1
skips), doctests 4 per mode, independent 2333/2323 passed (27/37 skips).
NumPyro enables one acceptance check previously skipped. REVIEW.md records
exact commands, times, initial numerical failure and final focused evidence.
Commit and exact independent review follow, then stop/report.

---

# SIM.1 — Shared Weibull generator and censoring calibration (2026-10-29)

Work on `release-simulation` from full base
`8757c0fda78e34e6e818249adedd2a144a33c33d` (clean launch snapshot).
The matching reviewer checkout already exists. Preserve the unfinished A10
history below; its docs and A9/PH/RMST work belong to other implementers.

One concept: reusable demo-only inverse-CDF generation and host-side numerical
calibration, without PH, fitting, core changes or optional dependencies.
Spec revision 37: 3.3, 4.4 (generating PH-equivalent scales only), 7.3,
10.3, 11.4, 12.1–12.3; use-cases/00-common.md and both 01 files.

Files and contract to implement:
- `examples/weibull_simulation/study.py`: `generate_weibull(key, shape, scale,
  censor_limit, *, sample_shape=())`, returning a named tuple with `time`,
  `event`, `event_time`, `censor_time`. Parameters broadcast to row/batch axes;
  sample_shape prefixes those axes. Exactly split the supplied key into U/C
  keys; callers derive scenario/replicate keys with the specified fold/split
  scheme. JAX generation supports jit/vmap and selected process precision.
- Same module: `calibrate_censoring(shape, scale, target) -> float` uses NumPy
  float64 quadrature and bracketed bisection of the mean integrated survival.
  Concrete positive finite broadcast parameters only; calibration is host-only,
  not an RMST implementation or differentiable library API. Check quadrature
  convergence; fail explicitly rather than return an unverified calibration.
- `tests/unit/test_simulation_generator.py`: analytical exponential calibration,
  independent mpmath Weibull quadrature, heterogeneous broadcasting, key mapping,
  transformation agreement and fixed-seed large-sample survival/HR/censoring
  recovery with justified statistical bounds. Centralize any new test bounds
  in `tests/impl_tolerances.py`. Ordinary checks remain offline.
- `examples/weibull_simulation/DEPENDENCIES.md` and a temporary isolated environment
  probe (record the exact invocation in REVIEW.md): concrete exact-version proposal for the later examples group, test
  Optimistix/Matplotlib/Equinox/Optax/NumPyro against locked Python/JAX/NumPy.
  No pyproject/lock edits or added diagnostics/loader packages in this step.
- `REVIEW.md`: reading order, invariants, evidence and limitations; freeze the
  reviewed consumer import contract and calibrated simulation c values.

Target at most about 300 implementation/test/proposal lines (plan/review
records separate); split fitting, scenario runner, results and docs into later
steps. Docs to-do: executable simulation page and recorded full results after
reviewed PH integration; propose fresh docs command with examples selected.

Checks: focused tests in separate JAX_ENABLE_X64=1/0 processes, Ruff formatting
and lint, mypy src, unchanged `scripts/check.sh --through A10` cumulative gate.
Record exact versions, commands and outputs. Commit the step, invoke the exact
independent-review skill command for SIM.1, follow up to three rounds, then stop.
Dependency installation into the project still awaits maintainer approval;
only an isolated proposal probe is authorized here. Never merge/push/publish.


Implementation: 300 added source/test/proposal lines, excluding plan/review
records; no library API, dependencies, lockfile or protected-file edits.
Reviewed consumer contract is recorded in REVIEW.md pending round 1.
Both simulation targets independently match 40-digit mpmath quadrature;
focused final suites: 12 passed in each dtype, including second derivatives.
Ruff formatting/lint and mypy pass. Calibrated c values: 47.793149053990135
and 19.17253543619806 years; pilot empirical censoring .249355/.550485.
Dependency versions proposed and tested in isolation: Optimistix 0.1.0,
Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0 against the
existing locked JAX/jaxlib 0.11.2 and NumPy 2.5.3; group/lock approval pending.
Cumulative fresh-wheel through-A10 gate: exit 0, all gates passed.
Own suites 1059/1066 passed (7/1 existing skips); independent 2332/2322
passed (28/38 existing skips); source doctests 4 per mode. REVIEW.md records
exact timings, commands and the focused derivative-test evidence. Commit and
independent round 1 now follow; stop after the review verdict.

Round 1 on 1e3f5fc returned fix-first for F-1: add explicit quadrature-failure
coverage. Add the k=20/b=5/target=.5 RuntimeError regression, and clarify N-1
underflow and N-2 time units in docstrings; N-3 import-helper suggestion stays
conditional. Fix-up is 7 net source/test lines (307 total), no behavior or
contract change. Focused suites: 13 passed per mode (4.32s/4.05s), Ruff and
mypy clean. Commit separately and request round 2 after round 1 completes.

---

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; all incoming A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.d1 — Using and verifying the installed Slice A subset (2026-10-29)

Authorized by "the current work is approved; if more work needed, get to it".
Work on the existing a9b branch from clean bd32ff9. The maintainer completed
the main merge, manually reran `scripts/check_wheel.sh --through A9` without
errors, and cleaned the reviewer-owned throwaway worktree. Record those as
maintainer-reported checks/cleanup, not as runs performed by this agent.
A9.4 is accepted; optional review nits N-1/N-2 do not block this docs step.

One topic: explain the installed subset's verification and limitations.
Spec: 6.1–6.2, 7.2–7.3, 10.3, 10.5, 12.1–12.3. Public API unchanged.
At most 300 changed docs lines. No dependency, source, test, fixture,
benchmark, protected-path or spec edits. PLAN.md keeps only current A9.

Files to create/change:

- Create docs/slice-a-verification.md as executable MyST. Give a compact
  synthetic exact/right example with construction and diagnostic access inside
  jit, value/parameter gradient/Hessian checked against hand algebra, eager
  and prepared/raw agreement, and vmap over rates. Document the smooth
  interior parameter-only guarantee and explicit host checkpoints.
- Explain the current subset and unsupported likelihood modes; link existing
  construction, likelihood, numerical, distribution/grid and concordance
  guides. Describe the fresh-wheel runner, installed out-of-tree custom-family
  checks, pinned synthetic R evidence and the exact CPU reproduction commands.
  Explain first-call versus warmed costs, prepared/traced flags, returned
  diagnostics, logical bytes versus unmeasured peak memory, and optional
  benchmark nits without silently changing accepted measurements.
- Add the new page to docs/index.md's prose and toctree. Audit the remaining
  executable pages and playground. Correct only stale capability statements
  within this verification topic; leave unrelated teaching work for separate
  steps. Document the locked JupyterLab command.
- The rendered audit found docs/survival-data.md's opening note still claims
  likelihoods, distributions and evaluation are unimplemented. Correct that
  one scope note and link current likelihood/concordance guides. This narrow
  factual correction belongs to the installed-subset verification topic;
  it adds no new tutorial or runtime behavior. Total docs remain below 300
  changed lines. Record the discovery before making the edit.
- Overwrite REVIEW.md with reading order, invariants, evidence mapping and
  remaining gate status. Record every Slice A gate item with concrete tests
  or artifacts and honest pass/pending status; this is not an A10/release
  completion claim. The agreed README/use-case example replacement is still
  separate from this small compiled-verification recipe.

Validation ownership: implementer owns fresh `scripts/build_docs.sh` in both
modes, Ruff and rendered-page/browser inspection. The maintainer's single-
runner preference means the independent reviewer should inspect source and
recorded outputs instead of repeating the same build/gate. No full wheel-gate
repeat for docs-only edits: the maintainer just reran it successfully on the
merged baseline, and no runtime/packaging change follows. Inspect links,
outputs, formulas, navigation and tables across the site in both builds;
inspect the new page carefully and record any existing warning/limitation.
Verify committed notebooks still have no outputs/counts. No new unit tests:
the executable page's assertions check its example, and existing suites
already cover the numerical and packaging behavior.

After successful build/source/visual checks, record exact commands/results,
commit this docs step on a9b with "Independent review: round 1/3" in REVIEW.md,
invoke the exact independent-review skill command for step A9.d1 in
../review/code-a9b, address blocker/should-fix findings in separate fix-ups
within three rounds, then stop and report. Do not start another step.

Docs/closure to-do: agreed replacement README/use-case teaching material and
any gaps from this rendered-site/evidence audit remain explicit; don't claim
Slice A or A10 complete until their separate gate items are accepted.

## A9.d1 evidence and audit conclusion

Implemented the verification page, index navigation and one stale data-guide
scope note; no code or notebook changes. Both final docs builds pass, nine
executed documents and 137 code cells per mode. The three new example cells
check exact analytical values, gradients/Hessians, input forms and jit/vmap.
All 1,048 local file/fragment links per rendered site pass. The committed
playground remains output-free. Ruff passes for the repository and the
extracted three-cell notebook; JupyterLab CLI reports 4.6.4. Native Brave
inspection covers all 11 content pages in float64, all pages' accessible
content in float32 and detailed views of the new page, corrected note,
playground and representative grids/formulas/tables. REVIEW.md records final
execution times, initial sandbox/interruption errors, checks and limits.

The final evidence map is included in this step's REVIEW.md rather than
requiring a separate metadata-only audit step. The new three-observation
verification recipe provides tiny synthetic smoke evidence; the larger
agreed README/use-case teaching replacement remains separate work and is
not claimed complete. The maintainer's post-merge fresh-wheel pass supplies
current regression evidence; no runtime/packaging changes follow it.
A9's verification evidence is now ready for independent review and maintainer
acceptance. No claim that A10, Slice A as a whole or the release is complete.
Stop after this docs step's review; no additional implementation starts.

## A9.d1 round-1 follow-up

Round 1 on 2c8cd60 returned fix-first for should-fix F-1: cite the explicit
unsupported-mode NaN/preserved-neighbor requirement and executed triage
recipe in the gate map, and map each narrative item in spec 12.2's paragraph
to its docs page. Existing code/tests already cover this behavior; no new
implementation or test is required. Also fix trivial N-1 (R paragraph line
wrap) and N-2 (clarify that tests/doctests, rather than static checks, run
in separate dtype processes). Code cells remain byte-identical.

The reviewer reran the build because its role requires it, despite the
initial ownership request. Its build passed in both modes (new page 3.12s /
3.07s); Ruff passed. No wheel gate or mutations were rerun. For this prose/
metadata fix-up, the reviewer alone owns round-2 builds/lint. The implementer
checks only the diff and unchanged cells, records F-1/N-1/N-2, commits a
separate fix-up and requests round 2. No duplicate build/full-gate run by
implementer; maximum three rounds, then stop.

---

# A9.4/main merge resolution (2026-10-29)

Authorized by "merged main into the branch; fix conflicts". The maintainer's
merge is in progress on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflict; resolve these metadata files. No public
API, code, docs, dependency, spec or test behavior changes are needed.

- Preserve A9.4's complete plan and evidence below. Independent review round
  1 on 51319c8 returned merge, with optional N-1/N-2; those nits remain open.
- Preserve all main's A10 plan sections verbatim in plans/A10.md, including
  A10.d2, A10.d1 and A10.4. Its older A10 history remains the unchanged suffix
  of that archive. PLAN.md keeps only the current A9 sub-slice. A10's incoming
  review record is preserved in REVIEW.md alongside the A9.4 record. These
  are historical records; this merge starts no new implementation step.
  This corrects the first resolution, which incorrectly kept incoming A10
  sections in PLAN.md; the maintainer explicitly checked the current-slice
  format before completing the merge.
- Preserve every original stage-0 index entry except the intentional
  plans/A10.md archive extension and plans/README.md index description.
  Incoming docs, concordance fixtures/tests, release-agent plans and
  .workmux.yaml remain untouched. The last is an incoming protected-path
  change, not an agent edit.
- Verify both parents' records are retained, no conflict markers/unmerged
  entries remain, whitespace is clean and every other staged object is
  unchanged from the captured merge index, apart from the two archive/index
  changes above. No runtime/build reruns: neither
  conflicted file affects execution, and no manual executable edit is needed.
  Preserve incoming A10.d2's single-runner validation ownership.
- Stage the two resolutions and archive/index corrections; leave the merge commit to the
  maintainer under the working agreement. Do not create a merge commit,
  start the next implementation step or rerun independent review for this
  metadata-only resolution.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.
