# AFT.3 — Accelerated failure time tutorial

Independent review: round 3/3

Start and previous step: 607a11b. Original worktree base: cc0e30f.
Read PLAN.md AFT.3, spec 4.4/5/10.5, docs/proportional-hazards.md for style,
then docs/accelerated-failure-time.md and the minimal index/API links.
User explicitly requests tutorial-like exposition with academic flavor,
rather than unit-test property examples. One bounded AFT documentation topic:
221-line executable guide, plus navigation/status and API cross-links.
No library, public API, docstring, test, dependency or protected-path edits.

Content and mathematical invariants:
- Start with T=exp(eta)*T0, time ratios and log-time shifts. Positive AFT
  eta means longer times; PH's positive eta means higher hazard.
- Synthetic Weibull(k=1.5,b0=12 months), time ratio 1.5 gives scale 18.
  Survival at 18 equals baseline at 12; analytical medians 9.4/14.1 are
  marked in a survival plot. No calls to unsupported summary methods.
- Derive survival by transforming the event T>t, then the density Jacobian
  and hazard. AFT is generally not PH; common-shape Weibull has HR=exp(-k*eta).
- Explain caller-owned X@beta, exact/right likelihood under independent
  noninformative censoring, and the Jacobian term on event contributions.
  Example evaluates an objective/gradient, not an estimator or fit.
  Free baseline scale plus free predictor intercept is redundant.
- Hypothetical draws/groups broadcast (3,1) with (3,2); grid shape (181,3,2)
  has time first. Unsupported AFT summaries and numerical rescaling failures
  are stated without implying PH's derivative exemption applies to AFT.
- Code cells present quantities for interpretation, no asserts or test loops.
  PH guide was read as a style model and remains unchanged.

Validation:
- Direct scripts/build_docs.sh: exit 0, both strict sites built successfully,
  both dtype processes verified; success identity in docs/_build/build-identity.txt
  (completed 2026-11-09T20:51:28Z). All tutorial cells executed in both modes.
- Parsed both generated AFT HTML pages: six section headings, example outputs,
  math nodes, plot resource and local guide links present. Index and API link
  to the tutorial in both builds. Displayed medians 9.398637/14.097956 (x32),
  9.398637225/14.097955838 (x64); objective 7.972455, derivative -0.62113;
  grid shape (181,3,2). Outputs agree with the prose's displayed precision.
- Visually inspected the generated survival PNG: labelled axes/legend, both
  curves, median markers and 0.5 guide fit cleanly. Both modes produced the
  same plot resource hash. Browser access host was unavailable; full browser
  page-layout/MathJax rendering was not verified. Native browser selection
  yielded no tutorial preview; no interactive browser check is claimed.
- scripts/check.sh --fast --through A10: all gates passed. Formatting 196
  files already formatted; Ruff clean; mypy 22 source files clean.
  Implementation x64 1154 passed/8 skipped (23.42s), x32 1159 passed/3 skipped
  (21.26s); source doctests 4 passed per mode. Spec x64 2130 passed/27 skipped
  (13.48s), x32 2136 passed/21 skipped (11.49s).
- git diff --check passed. Tier 2 full gate not run (maintainer/CI tier).

Review focus: mathematical interpretation, pedagogical progression and
current API boundaries. No known content issue; browser page layout remains
unverified as noted. After AFT.3 no planned AFT sub-steps remain; generic RMST,
other families and summary infrastructure belong to other worktrees.


AFT.3 follow-up from bd93232: maintainer asks for both groups in one batch,
log_time_ratio=log([1.0,1.5]), batch_shape (2,). Round 1 returned merge, no
findings. Update tutorial table, medians and plot to consume that batch;
retain the shared scalar baseline in the regression section. All statistical
identities and scope remain unchanged. Follow-up validation pending.
Previous completed step: 607a11b; original worktree base: cc0e30f.

Follow-up validation:
- Initial direct sandboxed scripts/build_docs.sh exited 2: Jupyter process
  inspection raised PermissionError from sysctl/psutil; execution of the
  unrelated Weibull simulation page failed. No source workaround introduced.
- Direct escalated scripts/build_docs.sh then passed: both strict sites built,
  all cells executed in x32 and x64; success identity recorded. No docs edits
  after that successful run. No numerical/test/API changes in this follow-up.
- Parsed both generated AFT pages: opening output batch (2,), three-column
  month/group-1/group-2 table; median vector about [9.4,14.1]. Later regression
  remains valid with objective 7.972455. Shared baseline variable remains scalar.
- Visually inspected the new PNG: labelled group 1/time ratio 1 and group 2/
  time ratio 1.5, correct survival curves and median markers. Both modes use
  the same new plot hash. No full browser-layout inspection claimed.
- git diff --check passed. Numerical gates not repeated for prose/example-only
  changes; both-dtype executable docs builds are the required docs-step check.


AFT.3 follow-up from 93924cc: round 2 returned merge, with line-wrapping
nit N-1. Maintainer asks what the density Jacobian means and whether it is
needed. Replace the unexplained term with the inverse-map derivative and
conservation of probability over short intervals. It is needed for densities
and exact-event likelihoods, but survival/censoring transform without it.
Rewrap existing long lines (N-1); no mathematical/API change. Both-dtype
build validation pending. This uses the third and final review round.


Density-explanation validation: direct scripts/build_docs.sh outside the
Jupyter-restricted sandbox passed both strict dtype sites (exit 0, success
identity). All tutorial cells executed unchanged. Both generated AFT HTML
pages contain the inverse-map derivative, the f(t)dt=f0(u)du equation, the
1.5-wide-interval example, automatic density adjustment and censoring
explanation. Browser MathJax layout not inspected; unchanged plot retained.
git diff --check passed. No code/test/API changes; no numerical gate repeated
for this prose-only follow-up. The third review round is the final allowed
round; remaining AFT implementation sub-steps: none.


Round 3 (218defa): VERDICT merge. Density adjustment and its role in exact/
right likelihood were checked by hand; N-1 remained a harmless wrapping nit.
Maintainer follow-up from 218defa: the opening hazard derivation previously
also stated the density without explaining its derivation. Split the display:
cumulative hazard differentiates to h; then explicitly substitute AFT h and S
into f=hS, using h0*S0=f0. Retain interval-width intuition afterwards and
remove the now-redundant rhetorical question. Finish rewrapping N-1.
No numerical/API/test change. All three AFT.3 review rounds are used; no
fourth review is requested or run for this maintainer-directed clarification.
Validation pending. Original worktree base cc0e30f; previous step 607a11b.

Follow-up validation: direct escalated scripts/build_docs.sh passed both
strict dtype sites with success identity (exit 0). All tutorial cells executed.
Both generated HTML pages contain the separate hazard equation, the f=hS
explanation and aligned substitution through h0*S0 to f0, followed by interval
width intuition. No code or test changes; no numerical gate repeated.
git diff --check passed. N-1's remaining hazard-ratio line is now rewrapped.
This clarification is not independently re-reviewed: AFT.3 rounds 1–3 are
already used; the last saved verdict was merge for 218defa. No sub-steps remain.
