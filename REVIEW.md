# Preserved reviews from the maintainer-started main integration

These are separate historical reviews from the two merge parents, not fresh
checks of the combined tree. Preserve both before the next PH step replaces
this working review record. No new implementation or integration is approved
by this conflict resolution.

---

# Release-ph review record (68bd461)

# PH.1 — Construction, diagnostics and direct PH composition

Independent review: round 3/3 — completed; VERDICT: maintainer (D-028 only)

Branch: `release-ph`. Actual full base SHA:
`8757c0fda78e34e6e818249adedd2a144a33c33d` (requested snapshot 8757c0f).
Unfinished A9/A10 PLAN history is preserved. Reviewer checkout already exists
at `../review/code-release-ph`; no branch creation or integration performed.

## Read first, in order

1. `plans/release-agents/README.md`, `plans/release-agents/ph.md`, PH.1 at the
   top of PLAN.md, and spec 4.1–4.2, 4.4–4.7, 5–7, 8.7, 10.2, 12.1–12.2.
2. `src/survjax/distributions/wrappers.py` (163 lines): the complete new code.
3. `tests/unit/test_proportional_hazards.py` (301 lines): analytical evidence.
4. `tests/unit/test_proportional_hazards_guards.py` (97 lines): warning and
   generic-tail guard regressions.
5. Existing registration/policy in `distributions/base.py` and
   `_survival_methods.py`, nested diagnostics in `_parameter_issues.py`, and
   the shared `likelihood.py` evaluator (all unchanged).

## Scope and invariants

- This is a defining-module implementation, not public-export integration.
  Both __init__.py files and all exact export/install/reference assertions
  remain unchanged. A later export step needs the maintainer's docs exception.
- The wrapper is a registered frozen dataclass with a nested `baseline` and
  real-constrained `log_hazard_ratio`. The baseline may be positional or keyword, by the maintainer decision on
  D-028; eta remains keyword-only. Construction stays pure and value-unchecked.
  Registration owns dtype normalization, nested broadcasting,
  PyTree reconstruction and indexing. Baseline leaves retain their dtypes.
- H=exp(eta)*H0, log h=log h0+eta and log f=log h+log S. The wrapper never
  constructs or delegates to a PH-equivalent built-in distribution. The seven
  grid methods and the exact/right likelihood evaluator are inherited.
- Scaling uses sign and log magnitude to avoid premature exp(eta) overflow.
  Zero/infinite baseline log survival is substituted before arithmetic, and
  an overflowing final magnitude is restored after a safe exponential.
  This prevents inactive overflow tangents from poisoning boundary gradients.
  Where baseline log survival underflows to zero, use the baseline's existing
  log-CDF lower tail as log H0 (equal to rounding precision in this regime),
  then scale in log space. This recovers the value and its eta/baseline
  derivatives where the baseline retains that tail information. A genuinely
  zero log-limit keeps the exp(eta)-scaled baseline tangent through a safe
  linear branch when the factor is representable.
  No floors, clipping, epsilon or new derivative rules are used.
- The baseline infinity hook is scaled, so an improper baseline can retain
  its cure limit. Reference times retain the promoted baseline/predictor dtype;
  no narrowing cast or fallback remains.
- RMST ownership is respected: no edits to base.py, exponential.py, weibull.py
  or tolerances. No exports, docs, dependencies, protected paths, AFT, generic
  summaries or sampling. Initial source/test total was 292 added lines;
  round-1 fix-up changes 191 source/test lines (171 additions, 20 removals),
  below its 300-line budget. Round-2 fix-up adds 120/removes 2 source/test
  lines (122 total), below its 150-line budget; three files now total 561 lines.
- The PH-only initializer adapter calls the saved registered initializer with
  keywords, preserving its checks/normalization. Eta is first normalized at
  the adapter's caller-aware stack depth, so float64 conversion warnings still
  deduplicate at each user line. Registration receives the converted array and
  emits no duplicate. No warning filter or registry is changed by library code.
  Registration's shared rule
  stays untouched while PH accepts exactly one optional positional argument.

## Requirement-to-test evidence

- `test_broadcast_pytree_indexing_and_dtypes`: spec 3.3/4.1–4.2/4.4,
  (2,1) baseline × (2,2) predictor, preserved float32 nested dtype with
  mode-default predictor dtype, frozen leaves, tree round trip and slicing.
- `test_traced_diagnostics_include_predictor_and_prefixed_baseline`: spec
  4.7, pure traced construction and eager/jit/vmap masks, real predictor
  versus positive baseline violations, dotted keys and indexed diagnostics.
- `test_static_misuse_raises`: spec 4.1–4.2 / D-028 decision, keyword-only eta, registered
  nested type, nonreal predictor and incompatible construction shapes.
- `test_direct_and_grid_exponential_formulas` (seven cases): spec 4.4–4.5/5,
  all direct methods via shared grids and vmap of pointwise methods, values
  independently derived from hazards [1,4], at t=0, 1 and +inf; eager/jit/vmap.
- `test_likelihood_and_second_order_interior_derivatives`: spec 6–7/12.1,
  independently derived exact/right contributions, prepared/raw forms,
  eager/jit, debug_nans on a valid mixed batch and forward/reverse order-2
  finite-difference checks in the baseline rate and eta. Existing central
  derivative scales apply to this O(1) averaged likelihood, unchanged.
- `test_extreme_predictors_do_not_form_overflow_times_zero`: spec 7.1–7.2,
  representable H despite overflowing exp(eta), compared to the analytical
  log H in host float64 with zero absolute tolerance; zero/infinity survival
  and exact zero reverse-mode boundary Jacobians for eta of both signs.

## Initial-commit checks (965cbeb; retained history)

Environment: own worktree .venv; `uv sync --locked --offline --dev --group test`
exited 0 (155 packages resolved, 34 checked; no metadata changes).
Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, pytest 9.1.1, CPU device 0.

- Initial focused command below: 11 passed / 1 failed in each dtype
  (18.48s each). The extreme predictor test exposed NaN boundary reverse-mode
  derivatives from an inactive overflowing exponential. Fixed by substituting
  its input and restoring the overflow value separately; test unchanged.
- Final focused commands:
  `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_proportional_hazards.py -q`
  → 12 passed in 10.05s;
  same command with `JAX_ENABLE_X64=0` → 12 passed in 9.99s. No skips.
- `uv run ruff format --check .` → 113 files already formatted;
  `uv run ruff check .` → All checks passed!;
  `uv run mypy src` → Success: no issues found in 21 source files.
- `git diff --check` → exit 0, no output.
- `scripts/check_wheel.sh --through A10` → exit 0,
  `check: all gates passed through A10`. Unchanged guard passes on release-ph.
  Ruff/mypy repeat the clean outputs above. Fresh-wheel install expectations
  and all own suites: 1060 passed, 7 skipped (126.40s) float64; 1066 passed,
  1 skipped (174.25s) float32. Source doctests: 4 passed per mode (1.75s/1.61s).
  Independent through-A10: 2332 passed, 28 skipped (205.76s) float64;
  2322 passed, 38 skipped (179.15s) float32. Existing skips only.
  Wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.XcrWfu/survjax-0.0.1.dev0-py3-none-any.whl`.
  SHA-256: `7f28b3791c7c6130dee9d7f2cf4bf11dfe8347d655fba0e862575cea4aec928f`.
- Final staged whitespace and four-path scope audit are clean. The first
  sandboxed git add could not create the shared worktree index.lock; rerunning
  the same staging command with the approved escalation succeeded.

## Limits and follow-up

PH.1 is not the complete first-release PH gate. Full exponential/Weibull
identity values and parameter derivatives, draw/observation/time broadcasting,
additional custom and saturated baselines, invalid-row first/second derivative
isolation, nonzero boundary derivatives and the extended tail matrix belong
to PH.2. These are not claimed verified by this initial test selection.
RMST/closure waits for the reviewed RMST integration. The default log-CDF
inherits the generic lower-tail range (spec 4.2); no new log-H extension hook
or built-in dispatch is introduced here. Public export and executable topic
integration remain separate. No R generation, demos or docs build were run.
Round 1 identified a constructor question (S-1 / D-028). The maintainer has
now chosen a positional baseline. This follow-up implements the decision and
records it in D-028 without editing the read-only spec or marking it resolved.


## Independent round 1 outcome — historical record before the decision

Reviewed implementation commit: `965cbeb0b61efafc0b6a981fa5fedeec5044d3dd`.
Report: `../review/slices/PH.1.md`, round 1. Final `VERDICT: maintainer`.
One review invocation, no retry. No fix-up implementation or second step begun.

- S-1 / D-028 (open): spec 4.2 keyword-only constructors conflict with the
  positional baseline in 4.4/11.3 examples. The implementation followed 4.2;
  the reviewer requires a maintainer decision before the next fix-up.
- F-1 (open, should-fix): a baseline log survival rounded to zero at positive
  time is returned unscaled, losing the exp(eta) factor in its baseline
  tangent. Review probes: float32 r=1e-30, t=1e-20, eta=40 gives d/dr=-1e-20
  instead of about -2.35e-3; float64 r=t=1e-200, eta=300 also fails. This
  refutes a general derivative-correctness claim; focused tests missed it.
- F-2 (open, should-fix): improper-baseline infinity scaling and NaN eta hook
  behavior need tests in the same step. Deferring these implemented behaviors
  to PH.2 was insufficient. Mutants removing either behavior survived.
- N-1 / N-2 (open nits): promoted reference dtype versus predictor downcast;
  an overflow-threshold regression near log(max). Record for the fix-up scope.

Reviewer checks: own suites 1060 passed/7 skipped and 1066 passed/1 skipped;
4 source doctests per dtype; Ruff/mypy clean. Its plain check.sh gate exits 1
because SURVJAX_WHEEL was not supplied: float64 independent suite 2327 passed,
28 skipped, 5 wheel-fixture setup errors, then stops before float32 acceptance.
The reviewer could not invoke check_wheel.sh under its permissions. This does
not supersede the successful fresh-wheel gate above; it is an independent
verification limit. Mutation checks caught 2/5 mutants in both modes.

The reviewer reports a leftover prunable temporary-worktree registration in
shared Git metadata after refused cleanup. I have not touched reviewer-owned
paths or pruned other worktrees. No merge, push, publication or spec edit.
This record and D-028 are a follow-up bookkeeping commit only; the reviewed
implementation remains unchanged and is not approved for merge.


## Round-1 follow-up for independent round 2

Maintainer reply: “allow for positional baseline in PH wrapper”. Resume PH.1
from b0b0f8b, retain release-ph ownership boundaries and public-export deferral.
I am using the independent-review skill again after the fix-up commit; this
is round 2/3 for PH.1, not a reset of the review count.

- S-1 / D-028: implement the maintainer's constructor decision. Positional and
  keyword baseline forms share registration; eta stays keyword-only. New
  `test_positional_baseline_matches_keyword_under_tracing` compares both forms
  eagerly/jit/vmap; `test_static_misuse_raises` checks extra positional eta,
  duplicate baseline, missing eta and existing malformed types/shapes. The
  obsolete positional-baseline rejection was changed for this explicit decision.
- F-1: recover a zero baseline log survival from an accurate baseline log-CDF
  lower tail, rather than returning an underflowed zero. The new test
  `test_lower_tail_recovers_value_and_first_derivatives` checks the reviewer's
  r/t/eta cases in both dtypes against hand-derived -exp(eta)*r*t and
  (-exp(eta)*t, -exp(eta)*r*t), with zero absolute tolerance; eager/jit values,
  jit forward/reverse first derivatives. No expectation uses production code.
- F-2: `CureBaseline` supplies log S0(t)=log(p)*(1-exp(-t)). The new infinity
  test checks S(inf)=p**exp(eta) and its p/eta derivatives, for p=.3 and p=1,
  including the true zero log-limit tangent. A separate test checks NaN eta
  in the direct hook and all seven infinity direct methods. Both reported
  infinity-hook mutants are now directly targeted.
- N-1: retain reference dtype by ordinary promotion; the new x64-only test
  keeps the baseline reference 1e60 with a float32 eta. Its float32-mode skip
  states why mixed float64 inputs cannot be tested there.
- N-2: the extreme-predictor test now includes log(max)+.25 in each dtype;
  exact zero boundary Jacobians pin the overflow guard's near-threshold band.

Checks of the follow-up (source/tests unchanged after these runs):

- `uv run ruff format --check .` → 113 files already formatted;
  `uv run ruff check .` → All checks passed!;
  `uv run mypy src` → Success: no issues found in 21 source files.
  Initial mypy check rejected the initializer's dataclass-derived Array-only
  annotation/signature. The saved initializer now has a Callable[..., None]
  annotation; the method assignment explicitly ignores the generated-signature
  mismatch, as registration already does for its own initializer replacement.
- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_proportional_hazards.py -q`
  → 18 passed in 9.98s.
  Same command with `JAX_ENABLE_X64=0` → 17 passed, 1 skipped in 10.12s.
- Fresh-wheel `scripts/check_wheel.sh --through A10` → exit 0,
  `check: all gates passed through A10`; guard, Ruff and mypy pass.
  Own suites: 1066 passed, 7 skipped (70.01s) float64; 1071 passed, 2 skipped
  (73.11s) float32. Source doctests: 4 passed in each mode (1.35s/1.19s).
  Independent through-A10: 2332 passed, 28 skipped (95.92s) float64;
  2322 passed, 38 skipped (84.82s) float32. The one additional float32 skip
  versus round 1 is the new x64-only mixed-reference-dtype test.
  Wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.YhuvWQ/survjax-0.0.1.dev0-py3-none-any.whl`.
  SHA-256: `b9735cf8f54a912fb3720ce0ca5611eed64f3e7f4a2e3be75547c9c2faf2709a`.
- `git diff --check` → exit 0, no output. Five-path scope audit only:
  wrappers.py, PH tests, PLAN.md, REVIEW.md and DEVIATIONS.md. RMST-owned
  files, exports, tolerances and all protected paths are unchanged.

No new spec question, dependency, protected-path edit or shared ownership
change is needed for this follow-up. Full PH.2 evidence remains separate.


## Independent round 2 outcome and final fix-up

Reviewed implementation: `13b8b58ec22c41990d7f63f2a3261131a9ee6687`.
Report: `../review/slices/PH.1.md`, round 2; `VERDICT: fix-first`.
The reviewer independently confirms S-1 settled by the maintainer decision,
F-1/F-2 fixed and N-1/N-2 fixed. Its old surviving mutants are now caught.
D-028 stays formally open; the reviewer recommends a maintainer spec update
when closing it. No spec edit is made here.

- F-3 addressed: the PH adapter normalizes eta with the existing helper at
  the correct warning stack depth, then passes that converted array to the
  registered initializer. No catch/re-emit context, warning filter mutation,
  shared registration edit or caller-location cache is introduced.
  `test_float64_warnings_deduplicate_two_user_call_sites` checks positional
  and keyword forms at distinct caller lines, repeats both twice, and requires
  exactly two RuntimeWarnings at the two user locations. This guards warning
  attribution and deduplication, beyond merely counting emitted warnings.
  The x64 process skips it explicitly because no dtype conversion occurs.
- F-4 addressed: `GenericExponential` implements only log survival/reference
  time, so its generic log-CDF also underflows. The new guard test checks the
  hand-derived rate tangent -exp(eta)*t in both derivative modes and dtypes,
  with zero absolute tolerance. It detects removing isfinite(log_lower_tail).
  The added cure=1/eta=1000 test checks S(inf)=1 and its exact zero eta
  derivative, detecting removal of the linear-branch overflow bound. It does
  not assert an unrepresentable derivative with respect to cure probability.
- N-3: add a short case map above the masks. Retain lower-tail recovery because
  F-1's value and eta derivative require it. The reviewer measures 225 traced
  operations for PH versus 37 for its Exponential baseline; no speed claim or
  optimization is made by this correctness step.

Reviewer round-2 checks: plain check.sh still exits 1 because it has no
SURVJAX_WHEEL. Own tests: 1066 passed/7 skipped (86.10s) and 1071 passed/
2 skipped (166.34s); doctests 4 per mode; float64 acceptance 2327 passed,
28 skipped, 5 wheel fixture errors, then no float32 acceptance run. Ruff/mypy
clean. Reviewer mutant M8 and M10 survived and motivated F-4. It reports a
refused cleanup leaving mutated code in a temporary reviewer worktree. I have
not edited, reused or removed that worktree or reviewer-owned metadata.

Final fix-up checks and intermediate results:

- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_proportional_hazards.py tests/unit/test_proportional_hazards_guards.py -q`
  → 20 passed, 1 skipped in 21.41s.
  Same command with `JAX_ENABLE_X64=0` → 20 passed, 1 skipped in 21.58s.
  Skips are warning-conversion (x64) / mixed-reference-dtype (float32).
- Ruff format: 114 files already formatted; Ruff lint: All checks passed!;
  mypy: Success: no issues found in 21 source files.
- Intermediate fresh-wheel `scripts/check_wheel.sh --through A10` started
  before the additional mixed-dtype promotion regression was found. It exited
  0: own suites 1068 passed/8 skipped (163.39s), 1074 passed/2 skipped
  (159.31s); doctests 4 per mode (1.28s/1.15s); acceptance 2332 passed/
  28 skipped (88.20s), 2322 passed/38 skipped (59.48s).
  This is not the final-source gate: its wheel and own suites precede the
  promoted-eta fix below. It is superseded by a complete fresh-wheel rerun.
  Intermediate wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.RX4FgV/survjax-0.0.1.dev0-py3-none-any.whl`,
  SHA-256 `5c73c76a4f40e9de2fd97660d26b93501b56214ce1ebea22b1e6dc554b622cd3`.

Next independent review is round 3/3 for PH.1; no reset of the round count.
Stop after its verdict, even if further findings remain. No next step begun.


### Additional F-4 mixed-dtype correction before round 3

While checking the linear bound, a local probe found the exponentiation still
used eta's own dtype. With a float64 cure baseline at p=1 and float32 eta=90,
it returned a baseline derivative 0 instead of the representable float64
exp(90)=1.2204032943178408e39. The helper now promotes eta with baseline log
survival before every scaling branch. No parameter leaf dtype is changed.
`LogCureBaseline` and `test_zero_limit_linear_factor_uses_promoted_dtype`
check the independently derived derivative in forward/reverse mode, with
zero absolute tolerance. The test skips explicitly when x64 is disabled.

- Final focused command (both PH files) as above: 21 passed, 1 skipped in
  15.08s float64; 20 passed, 2 skipped in 15.23s float32. The extra float32
  skip is this mixed-dtype regression. The earlier 20/20 counts precede it.
- Ruff formatting/lint and mypy repeated clean after this source correction.
- Final-source fresh-wheel `scripts/check_wheel.sh --through A10` → exit 0,
  `check: all gates passed through A10` (completed 2026-10-30). Guard clean;
  Ruff: 114 files already formatted / All checks passed!; mypy: no issues in
  21 source files. Own suites: 1069 passed, 8 skipped (56.19s) float64;
  1074 passed, 3 skipped (67.69s) float32. Doctests: 4 passed per mode
  (1.24s/1.18s). Independent through-A10: 2332 passed, 28 skipped (89.18s)
  float64; 2322 passed, 38 skipped (72.71s) float32.
  Final wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.KmxcYM/survjax-0.0.1.dev0-py3-none-any.whl`.
  SHA-256: `e466d40246ec6fda2b2384c587c2083f350971a47218b372ac04130e2c852131`.
- Final whitespace/scope audit clean. Only PLAN.md, REVIEW.md, wrappers.py,
  the main PH test module and the new PH guard test module changed. No source
  changes followed the final focused and fresh-wheel verification.


## Independent round 3 outcome — code findings cleared; stop

Reviewed implementation: `a4ca299efb3e95388dbe209a08457533bd054f80`.
Report: `../review/slices/PH.1.md`, round 3 dated 2026-10-30.
Final `VERDICT: maintainer`, solely for D-028's formal spec disposition.
Three automated rounds used; no retry and no fourth invocation.

The reviewer states: no open code finding; every blocker/should-fix fixed;
the code is ready to merge. F-1/F-2/N-1/N-2 remain fixed, F-3/F-4 are fixed,
and N-3 is addressed. All five round-3 mutants are caught in their applicable
dtype modes: missing isfinite recovery guard, missing linear overflow bound,
missing eta promotion, wrong warning stack depth and missing adapter
normalization. Focused unmutated results: 21 passed/1 skipped float64 and
20 passed/2 skipped float32. Reviewer also probed dataclasses.replace and
pickle successfully; these are its probes, not new committed test coverage.

The constructor question S-1 is settled by the maintainer's explicit decision.
D-028 stays open because only the maintainer closes deviations, and protected
spec changes were excluded from this workstream. The remaining concrete spec
text could be added to section 4.2:

> Exception: ProportionalHazards accepts baseline positionally or by keyword;
> log_hazard_ratio remains keyword-only.

No constructor choice needs repeating, and no code fix is waiting on this text.
I have not edited spec/, changed .spec-seen, marked D-028 resolved, merged,
pushed or published. Formal spec synchronization remains with the maintainer.

Reviewer gate rerun: clean Ruff/mypy; own tests 1069 passed/8 skipped (64.22s)
float64 and 1074 passed/3 skipped (62.89s) float32; doctests 4 per mode;
float64 acceptance 2327 passed/28 skipped/5 wheel fixture errors (unsupplied
SURVJAX_WHEEL), then no float32 acceptance stage. The reviewer did not have
permission for check_wheel.sh. Its earlier gate was discarded because the
reviewer accidentally ran mutations in its review checkout concurrently; it
restored the file, verified a clean status, and reran. Our final fresh-wheel
through-A10 pass above is independent of those reviewer setup/procedure limits.

The reviewer still reports refused cleanup of temporary worktree metadata for
/tmp/survjax-mut-PH.1 and /tmp/survjax-mut-PH.1r3. No reviewer paths or shared
worktree metadata have been edited or pruned by me. Record this outcome in a
bookkeeping-only commit; implementation and tests remain exactly a4ca299.
Stop after PH.1, with PH.2, exports and RMST integration still separate.


---

# Incoming main review record (14155ed)

Independent review: round 2/3

# A9.d1 — Using and verifying the installed Slice A subset

Authorized after A9.4 approval and the maintainer's completed main merge.
Branch a9b, clean base bd32ff9. One verification topic: the new page and
index navigation, plus a stale capability note discovered during the audit.
No source, tests, notebook, benchmark, fixture, dependency or protected-path
changes. Earlier A9.4/integration reviews remain in Git at bd32ff9:REVIEW.md.

## Reading order

1. docs/slice-a-verification.md: supported subset, three exact analytical
   smoke cells, installed-wheel runner, CPU reproduction and evidence limits.
2. docs/index.md's prose/toctree entry and docs/survival-data.md's corrected
   opening note. The old note incorrectly said distributions/likelihoods/
   evaluation were unimplemented; the replacement links current guides.
3. PLAN.md A9.d1: authorization, validation ownership and audit conclusion.
4. The gate map below, then existing tests/benchmark/fixture files only as
   evidence references. These executable inputs are unchanged by this step.

## Validation ownership

The maintainer requested one runner per step, not duplicate implementer and
reviewer runs. The implementer owned initial execution/build/lint and browser
inspection. The reviewer nevertheless reran build/lint in round 1 because
its role requires it. **For this round-2 prose/metadata fix-up, the reviewer
alone owns build/lint; the implementer does not repeat those checks.**
Initial build and audit results are retained below, with round-1 results and
fix-up details added separately. The maintainer reported rerunning
`scripts/check_wheel.sh --through A9` without errors after completing the
main merge, and manually cleaning the A9.4 throwaway worktree. Those are
maintainer-provided outcomes, not commands rerun or counts inferred here.
There are no later runtime/packaging changes requiring another wheel gate.

## Invariants and specification mapping

- Spec 6.1/6.2/7.2/7.3: ordinary construction and both diagnostics are inside
  jit; log likelihood is summed by the caller. For two events and total
  follow-up 3, l(r)=2 log(r)-3r, l'(r)=2/r-3, l''(r)=-2/r**2. At r=1 the
  eager/compiled values are exactly -3/-1/-2. Prepared/raw contributions
  are exactly [-0.5,-1,-1.5]. Doubling follow-up gives mapped values [-3,-6],
  gradients [-1,-4], Hessians [-2,-2]. Expected values come from this hand
  algebra, not another evaluation of the library. Exact binary cases need
  no new tolerance. Explicit host checkpoints are kept outside jit.
- Spec 7.3: parameter derivatives are promised only through second order
  for finite valid smooth interior cases; arbitrary unsafe custom code,
  time derivatives and invalid/infinite/support-boundary cases excluded.
- Spec 10.3/12.1: explain why check_wheel supplies SURVJAX_WHEEL and plain
  check.sh alone does not. Installed core/custom-family typing checks,
  offline cache requirements and separate source-archive release check are
  accurately distinguished. No publishing or dependency change.
- Spec 10.4/12.1: R claims are only committed synthetic fixed-parameter
  Exponential/bc-like Weibull references, with the pinned versions. No real
  patient-data, fitted-model or unsupported-mode agreement claim.
- Spec 12.3: benchmark descriptions match the accepted harness and recorded
  artifact. Keep prepared/traced flags, returned diagnostics, warmups,
  synchronization, compilation reuse, logical bytes and unmeasured peak
  memory explicit. Explain optional A9.4 N-1/N-2 caveats without changing
  accepted timing data or pretending the benchmark's pre-harness revision
  contains its script. No universal speedup or accelerator claim.
- Spec 10.5/12.2: add executable verification/navigation, audit all existing
  content pages and editable playground, retain the represented/evaluated
  observation distinction and failure behavior. The new data-guide note
  corrects an objectively stale capability statement within this topic.

## Checks actually run

Final command: `scripts/build_docs.sh`, permitted outside the sandbox.
Exited **0**; both builds report **build succeeded**, with no Sphinx
warnings, execution failures or process-inspection errors. Existing
IPKernelApp TCP-transport warnings remain in the kernel logs; these are not
Sphinx warnings. No configuration/timeout/selection changes were made.
Final log: /private/tmp/survjax-a9d1-docs-final4.log.

| Executed document | float32 seconds | float64 seconds |
| --- | --- | --- |
| builtin-families | 2.81 | 2.56 |
| concordance | 3.05 | 3.11 |
| distributions | 4.77 | 4.75 |
| grids | 3.49 | 3.50 |
| likelihood | 3.79 | 3.74 |
| numerical-behaviour | 3.34 | 3.30 |
| playground | 4.44 | 4.49 |
| slice-a-verification | 3.07 | 3.09 |
| survival-data | 2.76 | 2.76 |

All nine runnable documents execute afresh: **137 code cells per mode**,
including three new assertion cells and 16 playground cells. The final
executed notebooks were inspected for execution counts and unexpected
error outputs; intentional raises-exception cells remain instructional
examples. The committed playground has no outputs/counts and is unchanged.
The HTML/parser audit checked **14 generated HTML files, 1,048 local
file/fragment links per mode**, with no missing targets. External links were
not fetched. The simulation guide remains deliberately excluded because
it describes future likelihood workflows; it is not counted as executed.

`uv run --group docs ruff format --check .`: **115 files already formatted**.
`uv run --group docs ruff check .`: **All checks passed!** The new MyST code
cells were also extracted to a temporary notebook and checked with the same
project configuration: **All checks passed!**, **1 file already formatted**.
`uv run --locked --group docs jupyter lab --version`: **4.6.4**. The documented
launch command is `uv run --locked --group docs jupyter lab docs/playground.ipynb`.
No interactive JupyterLab server/browser session was launched; notebook
execution in both build kernels is the reproducible playground check.
`git diff --check` passes. Final scope/staging audit follows before commit.
No mypy, pytest, R generator, benchmark, wheel, sdist or accelerator rerun
in this docs-only step; see the maintainer's current wheel result above.

Initial attempts/corrections: sandboxed docs build logged sysctl process-
inspection permission errors during kernel shutdown; interrupted that run
to rerun with permission (exit 2, DeadKernelError/cleanup AssertionError).
Subsequent builds passed. A visual pass prompted wrapping the benchmark
commands; the site audit found the stale data-guide note; a code-cell
format check prompted collapsing one mapped call. Rebuilt after each
source change; the table reports the final source only. An auxiliary .py
extraction incorrectly treated notebook display expressions as B018;
using actual code-cell notebook semantics resolved that tooling artifact.
A temporary notebook generator also added final newlines Ruff removes;
strip those from the temporary cells. Final extracted-cell checks pass.
No assertion, tolerance, behavior, selection or required check was weakened.

## Rendered-page audit

Native Brave through CUA inspected local html-x64 pages: index, survival-data,
builtin-families, likelihood, numerical-behaviour, concordance, grids,
distributions, api, playground and the new verification page. All 11 pages'
accessible content was also inspected in html-x32; screenshots covered the
new page, corrected data note and playground in both modes. Detailed views
covered the new derivative equation, exact outputs, compiled/map examples,
wrapped benchmark commands, prepared/traced table, peak-memory caveat,
index/toctree navigation, concordance pair table, API signatures, custom-
family formula table and grid survival/hazard figure. No desktop layout
issue observed after wrapping the new long shell commands. Existing long
code blocks use horizontal scrolling. This is desktop inspection at the
current browser width, not a mobile/responsive or external-link audit.

Observed values agree with the hand calculations in both modes: new page
-3/-1/-2, then mapped [-3,-6]/[-1,-4]/[-2,-2]; existing likelihood fit shows
6/103 and Weibull about (1.2904,16.006); tails distinguish zero survival from
finite log contribution near -10000; concordance C=0.7 and the five pairs;
playground prints the correct dtype mode and retains supported/unsupported/
zero-pair examples. The data note now links the available likelihood and
concordance guides while keeping interval/left/truncated likelihoods future.
No broad rewrite or new notebook topic was needed.

## Slice A completion-gate evidence map (spec 12.2)

Numerical/install rows below are **covered by existing tests and the
maintainer-reported post-merge through-A9 wheel pass**, not newly executed
by the implementer in this docs step. Docs rows have fresh results above.
The gate is restricted to Slice A's supported modes and methods; Slice B
interval/truncation mathematics and later-slice requirements are excluded.

| Gate item | Concrete evidence | Status/owner |
| --- | --- | --- |
| Construction, both forms, dtype and traced arguments | tests/unit/test_data_construction.py, test_data_validation.py, test_data_dtype.py, test_data_tracing.py | Gate pass reported by maintainer |
| Data diagnostics, indexing, size and metadata | test_data_issues.py, test_data_indexing.py, test_data_size.py, test_data_metadata.py (including test_static_flags_control_compilation_reuse) | Gate pass reported by maintainer |
| Parameter diagnostics and stored raw issues | test_constraints.py, test_parameter_issues.py, test_parameter_issue_container.py, test_likelihood_parameters_depth.py; test_likelihood_core.py::test_stored_input_issues_survive_indexing_and_reconstruction; test_observation_inputs.py | Gate pass reported by maintainer |
| Likelihood evaluation, both public input forms and shapes | test_likelihood_core.py, test_likelihood_entry_points.py; direct/grid shapes in test_family_batching.py and test_survival_grid.py | Gate pass reported by maintainer |
| Unsupported-mode NaN, preserved supported rows and triage support check | tests/unit/test_likelihood_core.py::test_unsupported_modes_preserve_supported_values_and_derivatives; tests/spec/test_a6_unsupported_modes.py; executed docs/likelihood.md#telling-the-causes-apart (`supported_mode`, `unsupported_nan`, `other`) | Existing test gate pass reported by maintainer; triage cells executed in both fresh dtype builds |
| First/second derivatives and compiled composition | test_likelihood_transformations.py, test_likelihood_composition.py, test_likelihood_custom_derivatives.py; test_survival_grid.py::test_grid_analytic_rate_gradient_hessian_and_order_two | Gate pass reported by maintainer; exact new docs assertions pass both modes |
| Stable tails and independent math | test_likelihood_tails.py, test_log_cdf_tails.py (mpmath values and differentiated identities), test_likelihood_boundaries.py | Gate pass reported by maintainer |
| Boundaries and inactive-input safety | test_family_boundaries.py, test_family_infinity.py, test_family_mixed_infinity.py; test_survival_grid.py::test_weibull_shape_one_zero_grid_and_nan_isolation | Gate pass reported by maintainer |
| Committed synthetic bc-like per-row R references | tests/fixtures/bc_like/; tests/reference/test_weibull_bc_like_r.py::test_likelihood_rows_match_r and test_time_first_survival_grid_matches_r; tests/reference/test_exponential_r.py | Gate pass reported by maintainer; no R regeneration |
| Eager jax_debug_nans mixed batches | tests/unit/test_likelihood_debug_nans.py::test_exponential_mixed_values_and_gradients_under_debug_nans and test_weibull_mixed_values_and_gradients_under_debug_nans | Gate pass reported by maintainer |
| Installed wheel, exports, type marker and external family | tests/install/test_wheel.py (three tests), tests/install/custom_family.py; independent A9 wheel acceptance tests | Fresh-wheel gate pass reported by maintainer |
| Tiny synthetic example | Three-observation exact checked recipe in docs/slice-a-verification.md; existing synthetic likelihood fit in docs/likelihood.md | Fresh docs smoke passes both modes; larger README/use-case teaching replacement remains separate |
| Recorded CPU baseline | scripts/benchmark_slice_a.py, benchmarks/slice_a_cpu.json; A9.4 review round 1 merge on 51319c8; 48 cases with analytical checks/reuse | Accepted by maintainer; no rerun, optional N-1/N-2 remain |
| Executable docs and editable playground | All nine runnable documents/137 cells per mode, clean source notebook, CLI 4.6.4, build command above | Fresh two-mode builds pass; interactive server not opened |
| Rendered pages and accurate subset/failure descriptions | 11-page audit above, both builds, corrected data note and current diagnostic/tail/unsupported guides | Implementer desktop audit complete; no mobile/external-link claim |

### Required narrative coverage (spec 12.2 paragraph after the gate table)

| Required explanation | Executed/rendered page coverage |
| --- | --- |
| Both observation constructors; represented v0 kinds versus supported likelihood modes | docs/survival-data.md: Time and status, The same cohort as endpoints, interval/left/truncation sections and corrected opening note; docs/likelihood.md opening support note |
| data.issues and explicit host validation | docs/survival-data.md: Finding bad rows and Inside jax.jit; docs/slice-a-verification.md: Construction and derivatives inside JIT |
| Dtype behavior | docs/survival-data.md: Floating-point precision; docs/distributions.md: Details; both modes executed separately |
| Direct methods, limits and grids | docs/builtin-families.md: What happens at time zero and at infinity / Precision in both tails; docs/distributions.md: method table and boundary sections; docs/grids.md |
| Transformations and parameter derivative limits | docs/slice-a-verification.md: checked eager/jit/vmap recipe; docs/grids.md: Grids inside jit and vmap; docs/playground.ipynb |
| Explicit unsupported-mode failures and NaN-triage support check | docs/likelihood.md: When a contribution is NaN / Telling the causes apart, including `supported_mode`; docs/slice-a-verification.md opening subset explanation |
| Working local build and editable playground | scripts/build_docs.sh results above; docs/index.md Local commands and docs/playground.ipynb launch instructions; outputs cleared in source |

All pages in this narrative map ran/rendered in the initial two-mode audit.
The fix-up changes only plain text in the verification page and this map;
no code cell, supported-mode predicate, expected value or test is changed.

## Round 1 findings and round-2 fix-up

Round 1 on 2c8cd60 returned **fix-first** with no blockers/spec questions:

- **F-1, should-fix:** add the explicit unsupported-mode row above, naming
  the own mixed-row preservation test, independent A6 unsupported suite and
  executed triage recipe. Add the paragraph's narrative coverage map so
  constructors, issues/host validation, dtype, direct methods, transformations
  and unsupported failures each point to the actual page/section.
- **N-1:** rewrap the synthetic R comparison paragraph; unchanged claim.
- **N-2:** say lint/type checks run once and tests/doctests run separately in
  each dtype, matching scripts/check.sh. No command or behavior changes.

Reviewer-owned round-1 checks: initial sandbox build failed at uv cache
access (exit 2); permitted rerun exited 0, both dtype builds succeeded.
New page execution: 3.12s float32 / 3.07s float64. Ruff format/check passed
(115 files). No wheel gate, mutation, benchmark or interactive session ran;
no reviewer browser inspection. Its checkout is detached at 2c8cd60; no
throwaway mutant worktree reported. It reran build/lint despite initial
ownership, explaining that its role requires them.

Implementer fix-up checks: source/diff/whitespace and byte-identical code
cells only. No automated build/lint/pytest rerun; round 2's sole runner is
the reviewer. The initial counts/visual evidence above are not new runs
on this follow-up. Commit these F-1/N-1/N-2 changes separately and request
round 2; don't claim review acceptance before the next verdict.

## Closure and remaining work

This supplies A9's verification page and completion-evidence audit, pending
this independent review and maintainer acceptance. The tiny recipe is a
verification smoke example; it does not restore the withdrawn standalone
A9.3 demo or complete the agreed README/use-case teaching replacement.
That teaching material and the concordance-specific held-out example are
still separate work. A10, the full Slice A milestone and first release are
not claimed complete here; their remaining steps/gates belong to their
respective plans. The workspace maintainer owns any ROADMAP.md update.

The maintainer reports the previous throwaway worktree cleaned up. Do not
create or clean reviewer-owned worktrees from the implementer. Stop after
this step's review; do not begin another step. No merge or publication.
