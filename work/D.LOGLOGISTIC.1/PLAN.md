# D.LOGLOGISTIC.1 — built-in log-logistic family

Base/worktree starting commit: 3abefc5683a3aee8169eb2deffe57602897a3af3.
Preceding implementation step: A12 acceptance, ff07cf9 (merged by base 3abefc5).
Dependencies: existing family registration, parameter diagnostics, direct policy
and grid adapters; no dependency on the parallel LogNormal implementation.
Cumulative gate target: A12 (the existing gate; this family is separately
maintainer-authorized Slice D work, not completion of Slice D).
Preceding records: [A12 handoff](../../REVIEW.md);
[record isolation](../meta-parallel-records/PLAN.md).

## Planned change

- Create src/survjax/distributions/loglogistic.py with registered frozen
  LogLogistic(shape=..., scale=...), natural-parameter from_log, reference_time,
  analytic log survival, log CDF, log hazard and log density, with API docstrings.
- Export LogLogistic from both package __init__.py files. These are the only
  expected overlaps with the parallel LogNormal branch; retain both exports
  during integration. Do not change shared numerical machinery.
- Create tests/unit/test_log_logistic.py: hand-derived interior values and
  boundaries, mpmath tails, natural/log construction and diagnostics, pytree
  slicing/broadcasting, all pointwise/grid methods under eager/jit/vmap,
  first/second parameter derivatives and valid mixed likelihood debug-NaN run.
- Update exact export inventories in unit/import and installed-wheel tests;
  record the pending docs topic explicitly in the API-reference test.
- Use existing tolerances; add no dependencies. About 300 implementation/test
  lines including exports; no unrelated Slice D APIs or summaries.

## Contract and validation

Accepted specification revision 47: sections 3.3, 4.1, 4.2, 4.5–4.7, 5,
7.1–7.3 and 12.1. Keep both softplus log tails and analytic hazard in log
space; substitute safe operands before boundary operations.
Run touched tests in full with six workers in both dtype processes (Tier 0),
then scripts/check.sh --fast --through A12 (Tier 1). If any new test exceeds
one second, regenerate and commit tests/slow_tests.txt. Commit the step and
handoff before independent review; stop after its verdict/fix rounds.

## Docs to-do

Add LogLogistic parameterization and usage to the distribution reference in a
later docs step, including flexsurv llogis correspondence and from_log example.

## Maintainer authorization

The maintainer approved a size exception for the complete formatted draft
(343 changed implementation/test lines, excluding generated timing metadata).
Keep D.LOGLOGISTIC.1 as one step; no D.LOGLOGISTIC.2 split is planned.
