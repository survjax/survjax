# AFT.3 — Accelerated failure time tutorial

Independent review: round 1/3

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
