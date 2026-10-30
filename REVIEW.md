Independent review: round 2/3

# MNIST.3 — Float32 CNN and public likelihood loss

Read first:
1. PLAN.md MNIST.3; use-case 03 details Network/Precision and spec 11.2.
2. examples/survival_mnist/model.py (CNN and loss, 65 lines).
3. tests/unit/check_mnist_model.py (explicit optional checks).
4. data.py's moved preparation helpers, ordinary test import and README.

Scope/invariants:
- MNIST.2 was approved after its round-2 merge verdict. This step adds the
  CNN and loss, before epoch training, checkpoint selection or result metrics.
  No library API, dependency, core, protected, shared-generator or docs edit.
  318 source/test/README changed lines including the mechanical helper move.
- Preparation helpers/constants moved unchanged from model.py to data.py
  (222 lines total). Ordinary offline tests import data.py, which needs only
  core JAX/NumPy. model.py re-exports existing helper names but now requires
  Equinox; README preparation uses data.py directly. No calibration artifact
  or generation/key behavior is changed.
- Exactly Conv2d(1,16,3) → ReLU → pool2/stride2 → Conv2d(16,32,3) →
  ReLU → pool2/stride2 → flatten800 → Linear800→64 → ReLU → Linear64→1.
  Equinox defaults pool stride to 1, so stride=2 is explicit to satisfy the
  specified 26→13 and 11→5 dimensions. Output is scalar log scale per image.
- All layer arrays and final bias are explicitly float32 even in x64 mode.
  Bias is log(mean observed training time), a positive finite scalar supplied
  by the caller; no epsilon/floor. Caller supplies the prescribed init key.
  CNN receives only pixels, never digit labels.
- Loss accepts (CNN, scalar log_k), uses jax.vmap over images, public
  Weibull.from_log and negative mean log_likelihood(data). Test initializes
  log_k to float32 zero. Epoch runner will own that initialization and the
  actual training-row mean/key; neither is hidden in the loss.
- Optional tests live in check_mnist_model.py, deliberately outside default
  test_*.py discovery. Explicit --group examples commands fail if optional
  deps are missing; there is no importorskip or silent skipping. Ordinary
  core tests remain runnable without Equinox/Optax.
- Filtered gradients contain array leaves only. Equinox's pooling module
  contains Python metadata including its -inf reduction initializer, which
  must not be treated as a trainable parameter or a numerical failure.

Tests → requirements:
- test_cnn_geometry_bias_precision_and_batching: specified channel/kernel/
  dense shapes and pool geometry; float32 arrays from a strongly typed
  float64 mean; bias log(2); single-image/batched outputs agree (03 Network,
  Precision). Synthetic observation mean is (1+2+3)/3=2.
- test_loss_matches_exponential_algebra_and_second_derivatives: zero final
  weight gives constant b=2, log_k=0 means k=1. Exact/right times [1,2,3],
  events [1,0,1] give NLL=mean(t/2+event*log2)=1+2log2/3. Analytic d/dbias
  is mean(event-t/2)=-1/3 and second derivative mean(t/2)=1. Independent
  check_grads(order=2,fwd/rev) perturbs head bias and log shape on smooth
  interior cases using existing float32 derivative tolerances. This checks
  likelihood composition, without claiming smoothness at ReLU/maxpool ties.
- test_filtered_gradients_are_finite_before_and_after_adam_step: nonconstant
  random CNN, filtered eager/compiled full-tree values and gradients agree;
  sliced images/data[idx], scalar log_k and CNN arrays participate in one
  Adam(1e-3) update. Parameter arrays, losses and all gradient arrays remain
  finite before and after, and loss stays float32 (03 Training readiness).
  These are synthetic model checks, not the full training acceptance gate.
- Existing 22 ordinary preparation tests verify the mechanical move; their
  independent mpmath/key/precision checks are unchanged. No real data/network.

Actual commands/results:
- uv run --group examples python inspected installed signatures: Equinox
  0.13.8, Optax 0.2.8; Conv2d/Linear accept explicit dtype, MaxPool2d stride
  defaults to 1. No dependency/lock changes.
- Initial optional run: 1 passed, 2 failed (12.22s). The checks incorrectly
  treated pool metadata as arrays (AttributeError for float.dtype) and
  treated its legitimate -inf initializer as a nonfinite parameter. Changed
  the checks to filter inexact arrays; no production-code change or tolerance
  relaxation was needed. Analytic second-derivative check passed initially.
- JAX_ENABLE_X64=0 uv run --group examples pytest -q
  tests/unit/check_mnist_model.py: 3 passed in 15.68s.
- Same explicit command with JAX_ENABLE_X64=1: 3 passed in 12.09s.
- JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_mnist_data.py
  tests/unit/test_mnist_outcomes.py: 22 passed in 6.68s.
- Same ordinary command with JAX_ENABLE_X64=0: 22 passed in 7.03s.
- Ruff: 124 files already formatted, All checks passed! Mypy: Success:
  no issues found in 20 source files.
- JAX_ENABLE_X64=0 uv run --offline --isolated --group test python: asserted
  find_spec('equinox') and find_spec('optax') are None; imported data helpers
  and generated two rows. Output: Preparation imports and generates without
  Equinox/Optax: float32 (2,). No downloads or data cache required.
- scripts/check_wheel.sh --through A10: exit 0; all gates passed through A10.
  Guard/Ruff/mypy pass. Own suites: 1087 passed/7 skipped (88.92s) x64,
  1093 passed/1 skipped (70.60s) x32. Doctests: 4 passed per mode
  (1.06s/.92s). Acceptance: 2333 passed/27 skipped (67.10s) x64,
  2324 passed/36 skipped (61.87s) x32. Approved examples packages present
  enable the existing optional NumPyro/pandas acceptance checks.
  Fresh artifact /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/
  survjax-wheel.lBjqN3/survjax-0.0.1.dev0-py3-none-any.whl.

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, CPU. Optional neural
checks use the already approved examples group. No docs build or real-MNIST
training run: no docs or epoch runner in this step. Mypy covers core src,
not examples. No claim that the full demo or neural acceptance is complete.

Next-step obligations: epoch runner must initialize log_k=0 and CNN from
training mean with fold_in(uc,2), use prescribed shuffle keys/batch/epoch
budgets, loudly reject nonfinite gradients at initialization and after step
1, record epoch NLL, and retain only the earliest best validation checkpoint.
MNIST.2 N-2 metadata writer/recomputation comparison remains for run.py.
Independent review follows commit, then stop after this step's verdict.

## Round 1 follow-up

Round 1 on 6c6db1a returned fix-first for F-1: missing dense ReLU escaped
all tests. Added a NumPy reference pass using only layer weights/biases:
valid 3x3 cross-correlation, explicit ReLU, cropped stride-2 block max,
flatten/dense/ReLU/head. Expected output never calls Equinox layers or CNN.
No production forward/loss or tolerance changes. N-1 removed unused helper
re-exports; N-2 renamed the misleading data-as-model alias to mnist. N-3 is
recorded for the runner: construct SurvivalData from float32 outcome times.

S-1 remains for maintainer review, with no spec/gate changes:
- Optional checks follow plans/release-agents/mnist.md ("dedicated optional-
  environment checks") and plans/release-agents/README.md lines 77 onward ("Give demo checks an
  explicit optional-environment invocation without changing the standard
  runner or silently skipping failures"). This supplies the task's existing
  invocation policy; a future standing gate policy remains maintainer-owned.
- File-role placement is recorded as open D-028 in DEVIATIONS.md. I should
  have surfaced that layout choice before moving helpers. Proposed acceptance
  keeps preparation independent of demo packages; no further placement or
  spec change is made while awaiting the maintainer. Complete the already
  authorized F-1 fix/review, then stop for that decision before training.

Reviewer verified the helper move byte-identical, reproduced the analytic
loss/derivatives, and ran 25 focused tests per mode. Its direct cumulative
gate reproduced the known missing-SURVJAX_WHEEL setup errors; the implementer
fresh-wheel gate passed as recorded above. No full gate repeat for a test
addition, alias/re-export cleanup and records; focused suites and static
checks follow. Reviewer worktree cleanup remains maintainer-owned.

Fix-up evidence: uv run ruff format (2 reformatted/1 unchanged), Ruff lint
(All checks passed!), mypy (20 source files) pass. Combined explicit optional
and preparation suites: 26 passed in 14.37s (JAX_ENABLE_X64=0), 26 passed
in 13.52s (JAX_ENABLE_X64=1), using uv run --group examples pytest -q
tests/unit/check_mnist_model.py tests/unit/test_mnist_outcomes.py
tests/unit/test_mnist_data.py. No tolerance or expected-value relaxation.
