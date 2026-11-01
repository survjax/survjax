# MNIST.16 — Showcase walkthrough and README

Independent review: round 1/3 (MNIST.16)
Independent code review: MNIST.15 round 2/3.
MNIST.14 round 2: merge; F-1/F-2/N-1/N-2 addressed.

## Read first

1. INSTRUCTION.md sections 6–7 and Process; PLAN.md MNIST.16.
2. examples/survival_mnist/README.md: one command, outputs, results, attribution.
3. docs/use-cases/survival-mnist.md: payoff and three-line loss, generating
   curves, model/loss, live smoke, full results/three figures, takeaways.
4. Built docs/_build/html-x32/use-cases/survival-mnist.html and html-x64
   equivalent. Check both tables, the selected-epoch figure, links and code.

This is one authorized docs topic. The new prose is 224 lines; the README is
27. The larger deletion count comes from replacing the 178-line process-heavy
README and the old notebook. No implementation/dependency change in this step.
Maintainer authorized all steps without intermediate stops. Starting/previous
step commit 7db112e; original worktree base ee72c83; whole rewrite starts at
c1a0915. Spec update lives separately in design commit 00e7225, revision 41.

## Invariants and evidence

The page uses committed full results, not a rerun or new checkpoint. Its smoke
run is executed live during each build on 5k/1k/2k cohorts in float32; parent
docs kernels still follow the normal build modes. Full cohorts are 50k/10k/10k.
Tables round to three decimals; original CSVs preserve values. Fig3 is the
single epoch summary with selected epoch 7. Labels never train the network.
No narrative acceptance guards, calibration prose or minibatch trace remains.

The main lesson is explicit: network output enters from_log directly,
minibatching is data[idx], and held-out likelihood and C use survjax. The
reader sees a runnable smoke command and its live result, with notebook
subprocess plumbing hidden. MNIST attribution is retained in both documents.

Checks completed:
- scripts/build_docs.sh, invoked directly, succeeded in both parent dtype
  modes. First sandbox build succeeded but had IPython/Matplotlib permission
  warnings and kernel-shutdown PermissionErrors. Rerun with approved build
  escalation succeeded without those errors; x32 MNIST notebook took 15.14s.
- HTML audit: all 66 local links/assets/fragments in each MNIST page resolve;
  cohort sizes and rounded metrics are correct; copied fig1/2/3 are byte-equal
  to committed results. No stale precision/calibration/progress prose, stderr
  or hidden smoke plumbing appears in rendered text.
- Native Brave visual inspection: intro/loss, smoke/full tables, three result
  figures (hard reload required to clear old SVG cache), timing and takeaways.
  Browser-host connector was unavailable, so used native browser UI.
- Visual inspection identified long command lines; wrap each command across
  two shell lines (still one command). Final direct two-mode rebuild passed; both modes report build succeeded.
- Ruff: All checks passed; 139 files already formatted. mypy: no issues in
  21 source files. Diff whitespace clean. Pre-commit checks passed on example
  source, docs, README, hook config and three generated SVGs (TOML skipped).

No new tests for prose. Existing code acceptance and live notebook execution
supply the relevant evidence. Final rebuild and rendered-command recheck are
recorded below before completion; independent review is still required.

## MNIST.15 code review handoff

The code step is 36c290e plus 4fadb51 (stage deleted progress tests) and 7db112e
(full-fit regression). Its detailed handoff is `git show 7db112e:REVIEW.md`.
Read INSTRUCTION.md sections 3–5, model.py, run.py, check_mnist_* and results.
No src/, shared generator, dependencies or library API changed.

Approved solver change: BFGS rtol=atol=3e-4, max_steps=256, default throw=True.
Old 1e-5 stalled after 6 accepted steps even with 2048 iterations. Approved
3e-4 terminates in 6 at exactly the same float32 parameters; independent
profile-score bisection gives mean NLL gap 2.624e-12. No test tolerances changed.
The real full-cohort regression compares with that independent profile optimum
and catches restoring the old solver tolerance.

Full experiment: selected epoch 7, CNN k=1.368050814 (0.87% from 1.38),
LL=-1.126855135 and C=.689008653, above intercept -1.237543344/.5. Oracle
LL=-1.110933185, C=.697534919. Realized censoring .4998/.5074/.4994.
No seed/checkpoint/test tuning. Full command exits 0. Training 19.267s.
Final optional suite: 28 passed in 19.71s (float32, all then-existing tests);
subsequent full-fit regression addition: reference suite 2 passed in 4.58s.
No runtime change followed. Test mapping:
- data/outcomes: hashes, parsing, cohort sizes/keys and deterministic float32
  inverse-CDF algebra;
- model/training: tiny analytical loss and actual real-smoke gradients at
  initialization and after _train_step;
- reference: successful full-cohort fit and independent profile parameters;
- metrics: hand likelihood/risk cases, final prediction chunk and warm timing;
- run: real smoke CLI and full-run acceptance;
- figures: digit selection, analytical curves, grouping and selected epoch.

MNIST.14 review2 gate: own tests 1106/1112 passed (7/1 skipped), doctests
4 passed per mode. Spec float64: 2328 passed, 27 skipped, 5 setup errors due
to missing SURVJAX_WHEEL; gate stopped before float32 spec tests. Not a clean
full wheel-gate claim. No packaged code changed; implementer did not rerun it.
The reviewer accepted the data step despite that environmental gate issue.

## Remaining

Only independent MNIST.15/MNIST.16 reviews remain.
No further implementation sub-step, publishing, merge or release action planned.
Legacy spec wording notes S-1 in the earlier reviews remain maintainer-owned;
the explicit current brief governs fixed censoring and reduced metadata.

Final docs verification: direct escalated scripts/build_docs.sh exited 0 for
both modes after command wrapping; x64 MNIST notebook executed in 8.51s.
HTML audit again passes 66 local links/assets/fragments per mode, with clean
rendered output and committed figures. Native Brave confirms the x64 smoke
command fits, three-decimal live/full tables render, and no setup output is
visible. The x32 page was also visually checked. Ruff/mypy pass unchanged.

## MNIST.15 round-1 findings and fixes

F-1: restored a small train() behavioural test with actual Weibull validation
likelihood on hand-chosen censored exponentials and stubbed updates/predict.
It checks all 257-row shuffle/batch boundaries, means, [3,2,2] validation,
earlier tied epoch2 and identity of the returned checkpoint. No runtime change.
F-2: dated supersession notes added to D-029/D-031/D-032. All statuses remain
open for the maintainer; no new deviation or spec edits. N-2: execution-evidence
pointer added under the MNIST.15 heading. N-1 remains an accepted size nit.
S-1/S-2: explicit current brief governs metadata and test-only gradient checks.
Reviewer reproduced history.csv and all numerical metrics exactly, passed
29 tests, and confirmed the full-cohort regression catches old BFGS tolerance.
Round-1 full gate again stopped on missing SURVJAX_WHEEL, as already documented.

Round-2 validation: JAX_ENABLE_X64=0 uv run --group examples pytest -q
 tests/unit/check_mnist_training.py: 2 passed in 7.57s. Ruff check passes;
139 files already formatted; diff whitespace clean. Full experiment and docs
builds were not repeated because neither runtime nor docs changed.
