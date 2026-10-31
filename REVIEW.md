Independent review: round 3/3

# MNIST.5 — Float64 intercept-only reference fit

Read first:
1. PLAN.md MNIST.5; 03 details References and metrics/Precision; spec 11.2.
2. examples/survival_mnist/model.py _intercept_nll/fit_intercept.
3. tests/unit/check_mnist_reference.py; README process-boundary recipe.
4. Numerical investigation and actual training-only evidence below.

Scope/invariants:
- Maintainer approved MNIST.4 (round2 merge, F-1/N-1/N-2 fixed) and requests
  the next bit. This is the fitting part of references/metrics; metric helpers
  are a separate step so the model file remains below about300 lines.
- No core/public library API, dependency, module, protected, spec or docs edit.
  Only existing examples-group Optimistix .1.0 is used. D-028 acceptance is
  unchanged; no pending placement question or renewed approval needed.
- fit_intercept is a host function, not a differentiable/JIT estimator API.
  It requires an already enabled x64 process and never changes global mode.
  Returns a host NumPy float64 array [log_shape,log_scale], not float32 values
  cast back to double. Caller supplies training-only rows; no labels, validation
  or test input/selection enters the fit. Full runner later selects 50k rows.
- Matching nonempty vectors are required; the public SurvivalData constructor
  with dtype=float64 and issues.raise_if_invalid validates types/values.
  At least one event is required; all-censored data have no finite identifiable MLE.
  Censor-at-zero is admitted as an uninformative row and is tested.
- Optimize negative mean public Weibull.log_likelihood with from_log. Initial
  shape=1; scale=sum(all observed times)/event_count, the exponential MLE.
  Event count is summed explicitly as float64 before division (spec3.3).
- Fixed Optimistix BFGS recurrence/descent and normal termination, with a
  private roundoff-aware Armijo comparison (described under F-1 below).
  Forward autodiff for two parameters, rtol/atol=1e-10,
  max_steps256, throw=False. Unsupported/unsuccessful results raise, rather
  than returning an initial guess or relabeling a max-step result success.
  Successful solutions must have finite parameters, objective and gradients.
  No penalties, clipping, floors, post-fit Newton/root fallback or tolerance
  relaxation in numerical tests. Optimizer settings are described below.
- Returned float64 log parameters must be stored and explicitly cast to
  float32 in the separate neural/evaluation process. No metric helper or
  held-out test result is claimed in this step. Existing CNN/training code
  and calibration artifact are unchanged.

Tests → requirements:
- test_intercept_fit_matches_independent_profile_score_root: times
  [1,c,4,8], events [1,0,1,1], c=0/2/16. Independent 50-digit mpmath profile
  equation: d=3, event log-time sum=log32; b^k=sum(t^k)/d and
  d/k+sum_event(log t)-d*sum(t^k log t)/sum(t^k)=0. Expected log k/log b
  never calls the fitter. For c=0, its power and derivative vanish. Existing
  default float64 tolerances verify parameters; analytic stationarity gives
  zero gradients, using existing derivative tolerance. The two positive
  censoring times pin the effect of censored rows (not event-only fitting).
- test_reference_objective_values_and_parameter_transformations: k=1,b=2,
  t=[1,0,3], events=[1,0,1] gives mean NLL=(2+2log2)/3, and b=4 gives
  (1+2log4)/3. Eager/jit/vmap agree and check_grads(order2,fwd/rev) checks
  log-parameter derivatives on this smooth case with inactive right-censor0.
  Existing derivative tolerances/finite-difference step are unchanged.
- test_fit_refuses_float32_process_without_changing_precision: explicit
  RuntimeError, mode unchanged; x64 process skips this inapplicable guard.
  Other cases explicitly skip when float64 fitting is unavailable.
- Shape/invalid-observation/bool-time/all-censored checks preserve public
  construction rules and reject unusable inputs before optimization.
- Failed-result injection verifies explicit failure; independent injections
  of nonfinite parameters/loss/gradient verify every success guard. These
  tests never suppress optimizer failure or invent a fallback fit.

Actual numerical investigation (no tests/tolerances loosened):
- Installed Optimistix signatures/source inspected locally. The first probe
  incorrectly read Solution.y; installed Solution uses .value. Corrected
  that API use before numerical testing. No external dependency edits.
- Initial default reverse-mode BFGS, rtol/atol1e-12: 2 failed,11 passed,1
  skipped (2.30s) in x64; x32 refusal passed with13 unavailable-mode skips.
  Log-shape errors 5.869e-9/9.632e-9 exceeded the existing relative thresholds
  near log_shape=.4785/-.0939. Validation assertions were retained exactly.
- Tightening solver tolerances to1e-14/1e-15 left both results unchanged,
  terminating after39 steps with gradient residuals around1e-8. Installed
  source confirms stopping uses changes in parameters/objective, not the
  gradient. Use_inverse True/False gave identical results in these probes.
- Supported options.autodiff_mode='fwd' is suitable for two parameters and
  avoids a large reverse tape; with1e-12 it reduced score residuals to
  <=1.3e-10/6.7e-12 and passed all original parameter assertions. No new
  custom differentiation rule was introduced at that point.
- Actual 50k training-only forward fit at1e-12 hit256 steps; the guard raised
  RuntimeError. Diagnostic solves at1e-8/1e-10 return success in9 steps at
  exactly the same parameters as the max-step1e-12 result, with gradient
  components <=2.1e-14. Fixed termination tolerances1e-10 resolve the spurious
  nontermination. This setting is distinct from unchanged test tolerances,
  and was verified on analytical controls and training data only.
- Updated focused x64 checks: 15 passed,1 skip in4.91s; x32:1 passed,15
  explicit precision skips in.36s. Final combined-suite outputs follow below.

Actual full-training reference probe:
- JAX_ENABLE_X64=1 uv run --group examples python; verified warm-cache data
  loaded from /private/tmp/survjax-mnist-verification. Generate all70k once
  with the fixed float32 generator and stored c; select only
  split_indices('full')['train'] (50k); fit_intercept(time,event) casts those
  same observed outcomes to float64, never regenerates them in another dtype.
- Successful public fit: log parameters [0.1719941343787224,
  1.4325235160420127]; k=1.1876708667659677, b=4.189257522824442.
  Training NLL=1.234515803406909; post-fit gradients
  [-2.0734663297322477e-14,-4.3021843960795546e-15]. CPU fit including compile:
  1.519425750s. Actual output saved outside Git to
  /private/tmp/survjax-mnist-intercept-fit.json; no image/label rows committed.
- Independent JAX_ENABLE_X64=0 uv run --group examples python process reads
  those logs as NumPy float64, then explicitly casts jnp.float32. Public
  Weibull.from_log and SurvivalData([1,2],event=[1,0], float32) yield finite
  float32 likelihood; parameter shape/scale, theta and output all float32;
  asserted x64 remains disabled. No test-set evaluation or model selection.
- No claim that intercept shape matches1.38: it deliberately ignores the
  generating digit risk structure. Full neural acceptance/results are later.

Final commands/results:
- Explicit combined reference/training/model/preparation suites in separate
  processes with --group examples, pytest -q check_mnist_reference.py,
  check_mnist_training.py, check_mnist_model.py, test_mnist_outcomes.py,
  test_mnist_data.py (all under tests/unit): x64 50 passed/1 skipped in18.21s;
  x32 35 passed/16 skipped in13.99s. Skips explicitly identify unavailable
  float64 checks or the inapplicable float32 refusal when x64 is enabled.
- Ruff:126 files already formatted, All checks passed! Mypy: Success, no
  issues in20 source files. Final git diff --check is clean.
- scripts/check_wheel.sh --through A10: exit0, all gates passed. Own suites:
  1087 passed/7 skipped (57.01s) x64, 1093 passed/1 skipped (58.30s) x32.
  Doctests:4 passed per mode (.85s/.87s). Acceptance:2333 passed/27 skipped
  (63.03s) x64, 2324 passed/36 skipped (59.47s) x32. Optional NumPyro/pandas
  acceptance cases are enabled by the already approved examples environment.
  Fresh wheel: /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/
  survjax-wheel.EXiQnX/survjax-0.0.1.dev0-py3-none-any.whl.
- Initial commit: source/test/README diff238 lines; model.py247, reference checks165.
  Final after required review fix: diff343 lines, model.py292, checks223.
- No docs build, full neural run, metric/C timing, reference-result artifact
  committed, CLI or figure. Those remain later reviewed steps.

Reproduce training-only reference (root, separate x64 examples environment):

```python
import json
from pathlib import Path
from examples.survival_mnist.data import load_mnist, generate_outcomes, split_indices
from examples.survival_mnist.model import fit_intercept

_, labels = load_mnist(Path("/private/tmp/survjax-mnist-verification"))
c = json.loads(Path("examples/survival_mnist/calibration.json").read_text())[
    "censor_limit"
]
outcomes = generate_outcomes(labels, c)
idx = split_indices("full")["train"]
logs = fit_intercept(outcomes.time[idx], outcomes.event[idx])
print(logs)
```

Python3.12.9, JAX0.11.2, NumPy2.5.3, Optimistix0.1.0, CPU. Optional checks
need examples; ordinary discovery remains demo-free. Mypy covers src only.
Commit, invoke exact independent-review for MNIST.5, max3 rounds/fix-ups,
then stop after verdict before adding metrics.

## Round1 follow-up (F-1 and N-1)

Reviewer reproduced the full50k fit but found1e-10 fails on the actual5k
smoke cohort and some valid small samples; exact-only [1,4,8] stalls even
with4096 steps at a ~5e-9 score/parameter residual. This is a coverage gap in
my original probes. It confirms that tightening a loss/iterate-change
stopping rule below its near-optimum float64 resolution is not robust.
Investigate fixed1e-8 termination near sqrt(eps64), using the same BFGS and
public likelihood. Solver failures are still errors; no accepted max-step
iterate or fallback. All previous MP assertions/tolerances retained; extend
that oracle to exact-only and seeded mixed/exact10-row samples. Both actual
smoke/full training-only cohorts must return successful finite fits. N-1
updates the main Files bullet instead of leaving a conflicting old setting.
No new dependency, precision toggle, likelihood change, clipping or floor.

### Final resolution of F-1

Changing the Cauchy tolerance to1e-8 alone still stalled the retained seeded
exact n10 case (accepted score max6.14e-10). Public Armijo slope1e-4 and RMS/
two-norm probes did not resolve it. An experimental additional gradient-only
termination condition1e-9 passed small cases but still stalled the actual
smoke fit, whose score max was1.06e-8. It is not in the final code.

The final fix retains standard BFGS recurrence/descent and1e-10 Cauchy
termination. _ReferenceBFGS only attaches _RoundoffArmijo, a concrete subclass
of the existing search. The usual Armijo comparison is supplemented only when:
- both scalar objective values are finite;
- this is a full trial (never an already shortened step), not initialization;
- accepted-gradient max <=sqrt(eps64), i.e. already near stationarity;
- the predicted descent is nonpositive and its magnitude is below the comparison
  allowance4*eps64*(1+abs(f)+abs(f_trial));
- the trial objective does not worsen by more than that same allowance.

This permits a final BFGS correction that its scalar objective cannot resolve,
rather than returning the stalled iterate. Both returned step and search state
are reset coherently on acceptance. The objective, likelihood, observations and
parameters are never clipped/floored/altered. No Newton/root fallback, gradient-
only exit or promotion of failed/max-step results. The two concrete private
classes implement the prescribed BFGS fit's numerical comparison, not a new
optimizer API/framework; they use Optimistix0.1.0's existing extension protocol.
All extension arguments/returns are annotated (opaque third-party info/state
use Any); no shape-typing dependency is introduced.

Tests: same MP profile oracle generalized to d=event_count and
sum_event(log t), with its original parameter assertions/tolerances unchanged.
Old0/2/16 censor cases remain; add exact-only [1,4,8] and independent NumPy
seed0/20261025 n10 mixed/exact samples, quantized to the same float32 input
precision as generated outcomes. All8 MLE cases pass. Direct search controls
pin bounded acceptance: one-ulp unresolved descent accepted,1e-6 genuine
worsening rejected, nonfinite trial rejected, large gradient rejected, shortened
step rejected; next state and returned step agree.

Actual repeat, JAX_ENABLE_X64=1, same warm-cache/preparation recipe as above,
loop over split_indices(config)['train'] for smoke/full (no held-out inputs):
- smoke5000: logs [.16221816847747333,1.4258202396725628], k1.1761168049456001,
  b4.161269681783087, training NLL1.2264619928696128, gradients
  [-1.795408940186372e-15,-8.084890471682817e-16],1.429775084s CPU incl.compile.
- full50000: original logs/k/b/NLL unchanged, gradients still <=2.1e-14;
  1.321508459s CPU incl.compile.
Temporary output: /private/tmp/survjax-mnist-reference-fits-r2.json.
A separate JAX_ENABLE_X64=0 process reads both records and explicitly casts
logs to float32; public likelihoods on [1,0,3]/[1,0,1] are finite float32,
with x64 still disabled. No test-set metrics, tuning or accepted failed fit.

Final validation: all five explicit MNIST files, same --group examples
command as above:59 passed/1 skipped x64 in19.90s;35 passed/25 explicit precision
skips x32 in13.93s. Ruff126 formatted/All checks passed; mypy20 source files pass.
No repeat core/wheel gate after this example-only fix; the earlier fresh-wheel
pass remains recorded. Round1 reviewer's direct gate had the known missing-
SURVJAX_WHEEL setup errors; no core, packaging or protected path changed here.

## Round2 follow-up: F-2, N-2/N-3 and S-1

Reviewer confirms F-1 fixed: both real5k/50k fits succeed, all10 original
probes and60 additional random cohorts complete without solver failure.
F-2 test-only gap: dropping the predicted-change magnitude guard, direction
guard or accepted-step reset escaped. Added a near-stationary trial with
predicted descent~1e-12 but one-ulp observed worsening (must reject), an
ascent direction with tiny positive predicted change (must reject), and an
explicit accepted-step == step_init assertion. No solver logic or tolerance
change; all original MP/guard assertions retained.

N-3 adds the reason for gradient/allowance bounds: near an optimum, objective
change is quadratic in parameter error, so float64 value comparisons cannot
resolve descent at roughly sqrt(eps) scale. The comment states the purpose
rather than only saying that the likelihood is unchanged.

N-2 accuracy limitation: objective-based BFGS stopping does not guarantee
uniform1e-8 relative accuracy of log parameters on every cohort. For well-
conditioned problems, practical parameter accuracy is limited around the
sqrt(eps64) scale, with conditioning also relevant. Reviewer found one random
n5 cohort1.14x the default parameter tolerance (~1.6e-8 log-shape error);
stock BFGS returns the identical point, and the roundoff branch never fires
there. This is reviewer-reported evidence, not a new probe I ran. All committed
independent controls still pass unchanged. No further tuning, weaker assertion
or universal accuracy claim; reference evaluation is explicitly float32.

S-1 is logged as open D-029 for the maintainer: whether the private bounded
Armijo comparison is acceptable under "Optimistix BFGS". No implementation
change is made to resolve that question. Finish the already authorized F-2
fix and final (third) review, then stop for the maintainer decision before
adding metrics. D-028 acceptance remains unaffected and requires no renewed
approval. No core/wheel gate repeat for a test/comment/record-only fix; focused
checks/static checks follow. Reviewer worktree pruning remains maintainer-owned.

Round3 fix validation: Ruff126 files formatted, lint All checks passed!, mypy
20 source files pass. Combined five-file examples-group command: x64 61 passed,
1 skipped in20.76s; x32 35 passed,27 precision skips in14.01s. No numerical
logic, solver tolerance, expected value or test-accuracy threshold changed
since round2. Commit and final (third) review, then stop for D-029 decision.
