Independent review: round 2/3

# BAYES.d1 — Compact results and executable documentation

Read current PLAN.md revision, results/README.md and delivery.json, compact
posterior.npz inventory, page section order/takeaways, then original integration
in docs/index.md/scripts/build_docs.sh. Parent/previous step b45f511; worktree
base b1291a8. Original round1 docs commit086bfc4 is superseded, not retained as
an ancestor: the maintainer explicitly forbids the 19MB storage and agrees to
compact data. No merge/rebase/force-push/tag/global GC. Only the latest local
unmerged docs commit is replaced; no published tracking/tag ref contains it.

Round1 findings:
- F-001: compact storage explicitly chosen. One posterior.npz (94,330 bytes)
  holds exact saved trial/time/event/arm, three (4,1000) log-parameter arrays,
  divergence flags and grids. No derived curves, prior arrays, predictive
  arrays or split archives. Entire results delivery about822KB with five PNGs.
- Existing posterior_draws regenerates derived arrays in /tmp; README supplies
  a runnable recipe that also copies metadata, producing a renderer-ready dir.
  The existing renderer regenerates prior/predictive trials and both PNG/SVG
  outputs. No new numerical/helper module, dependency or library API.
- F-002: full diagnostics/truth table is now after route B in the figures/table
  section. Takeaways are the final section after reproduction instructions.
- N-002: route A uses literalinclude of posterior_draws, no hand-copy drift.
- S-001: maintainer agrees to the recommended compact-data/PNG-preview option.
  D-034 records PNG previews vs illustrated fig*.svg; SVG is reproducible with
  existing renderer. Status open, no spec edit or self-resolution. Priors D-033
  remain unchanged/open and are described as experimental, not recommended.

Checks of revision:
- Executed the README materialization recipe with JAX_ENABLE_X64=1 using
  /private/tmp/bayesd1-compact-20261105 as the new destination. All source
  fields match original full arrays exactly; all regenerated fields match at
  atol1e-12, rtol0, pickle disabled. No posterior sampling rerun or values edited.
- Dedicated check_figures against that materialized dir: exit0, predictive
  generation/key/domain checks and all KM/quantile/five-figure/overwrite checks
  pass. This validates consumption by the existing plotting pipeline.
- Compact data 94,330 bytes; total results 822,073 bytes at verification.
  Manifest has no archive parts and hashes only the compact delivery files.
- Ruff check and step-owned Markdown format/whitespace pass. The unrelated
  untracked bayesian-rmst-v2.md remains user work and is excluded/untouched.
- scripts/build_docs.sh directly, approved escalation: exit0, both modes.
  Approved page executed in19.79s/11.87s (x32/x64), with the unrelated orphan
  v2 draft also discovered and executed successfully; it remains untouched.
- Final HTMLParser checks: route A < route B < figures/table < diagnostics
  table; takeaways last, five figures present and resolving, table six rows.
- Staged pre-commit hooks all applicable pass, including added-large-files;
  no skips/exclusions/config changes. Final compact manifest hashes pass, no
  archive parts, entire delivery under1MB. History check follows replacement.
- Source doc/notes now272+48 lines; required compact regeneration recipe and
  reviewer-requested takeaways extend the original ~300-line documentation
  unit. This revision adds no numerical functionality.

Original approved integration and invariants remain:
- 2026-11-05 maintainer approves index link/toctree and adding existing
  --group examples to build_docs.sh. No dependency or lockfile changes.
- Parent docs mode remains x32/x64; live factor and four-chain100+100 smoke
  use float64 children; recorded four-chain1000+1000 tables/figures are labelled.
- Full original audit/metadata/timings and PNGs unchanged. Manifest records
  generation commits and SHA256. All data synthetic, not reference expectations.
- Previous docs build passed both modes. Fresh isolated offline locked group
  probe resolves Sphinx9.1.0, myst-nb1.4.0, NumPyro0.22.0, ArviZ0.23.0.
- Browser host unavailable in this environment; structural HTML checks do not
  claim visual layout QA. Figures were visually inspected in BAYES.4/.5.
- No new numerical fit, core gate, API, code module, tests/spec or spec changes.

Remaining: independent review round2, formal maintainer D-033/D-034 disposition
and release integration/evidence audit. No next feature started.
