Independent review: round 1/3

# MNIST.4 — Epoch training and validation checkpoint selection

Read first:
1. PLAN.md MNIST.4; 03 details Training, Precision and Splits and keys.
2. examples/survival_mnist/model.py TrainingResult/train and private helpers.
3. tests/unit/check_mnist_training.py; README training/check invocation.
4. D-028's dated maintainer acceptance note, then evidence below.

Authorization/scope:
- Maintainer explicitly answered the MNIST.3/D-028 question on 2026-10-31:
  "ok reviewed and agreed; on to next step". This accepts preparation in
  data.py and authorizes training; the user decision is recorded in D-028.
  Its formal status remains unchanged because AGENTS.md forbids implementers
  marking entries resolved. There is no remaining placement decision to ask
  again and no requested spec edit. Dedicated optional checks continue under
  the existing release-agent handoff's explicit examples-group policy.
- One concept: host epoch orchestration and compiled updates. No new module,
  core API/dependency/protected/docs/generator change. 318 source/test/README
  changed lines (125 source + 186 test + 7 README), plus decision/plan/review
  records. model.py is 184 lines; the optional check file is 186.
- No CLI, fitted references, test metrics, figures or full-run results yet.
  MNIST.2 N-2 recomputation/metadata writer remains assigned to later run.py.

Invariants:
- Inputs are selected float32 training/validation cohorts; train has no test
  set or digit-label argument. Exact/right data come from generated float32
  outcomes. Shape/dtype checks and explicit data diagnostic host checkpoints
  precede initialization. No silent float64 loss promotion (MNIST.3 N-3).
- Init uses fold_in(fold_in(key(20261025),3),2), log_k is scalar float32 zero,
  CNN bias uses the mean observed training lower/time only. Validation data
  never determine initialization or optimizer updates.
- Adam(1e-3), fixed 128-row minibatches; smoke/full mean 1/10 epochs. Per-epoch
  shuffle key is fold_in(fold_in(uc,3), zero-based epoch). Histories/selected
  epoch are presented in 1-based order. Training remainder is dropped.
- eqx.filter_jit wraps actual optimizer updates and value/grad evaluations.
  Initial and post-update1 finite loss/gradient checks inspect all inexact
  arrays and synchronize explicitly on the host. MaxPool's Python -inf
  metadata is not a trainable parameter. Failures raise FloatingPointError;
  no clipping/floors or silent replacement. All epoch NLL must be finite.
- Validation visits every observation, including its last partial batch.
  Float32 batch means are multiplied by float32 observation counts, summed
  and divided by a float32 total count. This bounds CNN memory while retaining
  the full-cohort mean. No integer-count division or float64 evaluation.
- Histories retain mean online minibatch NLL per training epoch and mean
  full-validation NLL after that epoch. Strict improvement only selects a new
  immutable parameter tuple, so an exact tie keeps the earlier epoch.

Tests → requirements:
- test_validation_includes_tail_with_observation_weights: analytical k=1,
  b=2, all-right cohort of 129 rows; 128 times=2 and one time=1002 gives
  mean NLL=(128*1+501)/129=629/129. Catches dropped/unweighted validation tail.
- test_ten_epoch_schedule_shuffling_history_and_earliest_best: controlled
  optimizer/validation values isolate orchestration. Real CNN init captures
  mean(1..257)=129 and prescribed key, separate validation times=999. Exactly
  two 128-row batches per epoch match prescribed permutations, one row is
  dropped; pixel markers prove aligned image/data indexing and validation
  uses validation images; returned training history means (2e+1,2e+2). Validation sequence
  3,2,2,4,... retains the exact immutable epoch2 checkpoint, not latest.
- test_nonfinite_gradients_fail_at_explicit_checkpoints: finite loss with
  deliberately nonfinite gradient, at init and after step1. Asserts failure
  phase and zero/one optimizer updates. Catches checking only loss.
- test_compiled_adam_first_step_matches_exponential_head_algebra: actual
  compiled _train_step on constant b=2,k=1 with t=[1,2,3], event=[1,0,1].
  Initial NLL=1+2log2/3; gradients are g_bias=-1/3 and g_logk=(log3/2-2)/3.
  First bias-corrected Adam increment is -.001*g/(abs(g)+1e-8), using the
  installed Optax default eps (verified signature, version .2.8). Conv1
  weights remain unchanged because zero head weights imply zero front-layer
  gradients. This is optimizer algebra, not a likelihood epsilon/floor.
- test_training_rejects_wrong_precision_shapes_and_config: explicit misuse
  failures; test_float64_training_or_validation_data_are_rejected: both
  observation cohorts must remain float32 under x64, one explicit skip when
  float64 storage is unavailable. Covers MNIST.3 N-3.
- test_nonfinite_epoch_nll_never_returns_a_checkpoint: nonfinite training or
  validation epoch aborts rather than returning an unselected initial model.
- Existing model tests retain independent NumPy forward, analytical loss,
  second-order fwd/rev head checks, eager/jit agreement and finite actual
  full-tree gradients before/after Adam. Existing data tests remain offline.

Actual evidence:
- Earlier focused runs passed during development; strengthened schedule to
  two batches per epoch and added float64/epoch-failure cases before final
  validation. No numerical failure, implementation fix or tolerance change.
- Final JAX_ENABLE_X64=0 uv run --group examples pytest -q
  tests/unit/check_mnist_training.py tests/unit/check_mnist_model.py
  tests/unit/test_mnist_outcomes.py tests/unit/test_mnist_data.py:
  34 passed, 1 skipped in 19.66s (float64 observation storage unavailable).
- Same command with JAX_ENABLE_X64=1: 35 passed in 20.39s.
- Ruff: 125 files already formatted, All checks passed! Mypy: Success:
  no issues found in 20 source files. Mypy covers core src, not examples.
- Warm-cache actual MNIST smoke on CPU, JAX_ENABLE_X64=0, using stored c,
  all-70k once-generated outcomes and fixed smoke splits. train receives
  exactly 5k train/1k validation; no test input or metric was consulted.
  One epoch processed 39 minibatches, dropped 8 rows, selected epoch1.
  Both required gradient checks passed. Training NLL=1.2448235750198364;
  validation NLL=1.2469474077224731; learned k=1.0319867134094238. All selected
  parameter arrays are float32. Training including compile: 4.664289875s.
  Not a full ten-epoch acceptance result or evidence of the full shape gate.
  Probe output: /private/tmp/survjax-mnist-smoke-training.json (not committed).
- scripts/check_wheel.sh --through A10: exit 0, all gates passed through A10.
  Guard/Ruff/mypy passed. Own suites 1087 passed/7 skipped (104.74s) x64,
  1093 passed/1 skipped (84.98s) x32. Source doctests: 4 passed per mode
  (1.29s/1.18s). Acceptance: 2333 passed/27 skipped (91.74s) x64,
  2324 passed/36 skipped (76.25s) x32. Approved examples packages enable
  existing optional NumPyro/pandas acceptance cases.
- After the gate, strengthened only the optional schedule fixture with pixel
  row markers to pin image/data alignment. Focused training checks: 8 passed,
  1 skip in 6.48s (x32), 9 passed in 8.06s (x64). No runtime changes or full
  gate repeat. Ruff then found formatting in this new reproduction code
  block; formatted the block and repeat static checks before committing.
  Fresh wheel /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/
  survjax-wheel.eswnan/survjax-0.0.1.dev0-py3-none-any.whl.

Smoke reproduction (repository root, approved examples environment):

```python
import json, time
from pathlib import Path
import survjax
from examples.survival_mnist.data import load_mnist, generate_outcomes, split_indices
from examples.survival_mnist.model import train

images, labels = load_mnist(Path("/private/tmp/survjax-mnist-verification"))
c = json.loads(Path("examples/survival_mnist/calibration.json").read_text())[
    "censor_limit"
]
outcomes = generate_outcomes(labels, c)
data = survjax.SurvivalData(outcomes.time, outcomes.event)
rows = split_indices("smoke")
start = time.perf_counter()
result = train(
    images[rows["train"]],
    data[rows["train"]],
    images[rows["validation"]],
    data[rows["validation"]],
)
print(
    result.training_nll,
    result.validation_nll,
    result.selected_epoch,
    time.perf_counter() - start,
)
```

Run with JAX_ENABLE_X64=0 uv run --group examples python. The previously
MD5/SHA-verified cache was warm; no network or data write in this step.
Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, Equinox .13.8, Optax .2.8, CPU.
No docs build (no docs changed), test model evaluation, reference fit, full
run, CLI or figure. Commit then independent MNIST.4 review, max 3 rounds;
stop/report after this step's verdict. D-028 acceptance is recorded above.
