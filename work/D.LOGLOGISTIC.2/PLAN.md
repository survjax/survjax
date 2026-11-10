# D.LOGLOGISTIC.2 — document the built-in log-logistic family

Feature: LogLogistic parameterization, executable guidance and API reference.
Base and preceding integration step: efd1b5a4e37a9822c580eeb36fdaaa898275c833
([merge handoff](../D.FAMILIES.MERGE.1/REVIEW.md)).
Dependencies: [code plan](../D.LOGLOGISTIC.1/PLAN.md) and
[code handoff](../D.LOGLOGISTIC.1/REVIEW.md), implementation commit 2b70e8c,
final code-step commit c31f733. Original worktree base: 3abefc5683a3aee8169eb2deffe57602897a3af3.
Cumulative gate target: A12. The maintainer requested the next step after family
integration. This is the deferred docs step, not a split of D.LOGLOGISTIC.1.

## Scope

- docs/builtin-families.md: include all four implemented families in the intro;
  add LogLogistic survival/hazard formulas, domains, units, flexsurv llogis
  correspondence and log-time logistic conversion, with executable natural/log
  constructors, time-first grids, boundary and stable-tail examples.
- docs/api.md: add LogLogistic autodoc and its implemented overrides/from_log.
- tests/unit/test_api_reference.py: remove the completed temporary LogLogistic
  docs exemption; restore coverage of every public export without exceptions.
- src/survjax/distributions/loglogistic.py: correct a Returns description
  whose colon is misparsed by Napoleon as a nonexistent type reference.
  Docstrings only; public API, numerical code, dependencies and protected paths
  are unchanged. Preserve LogNormal docs and all other step records. One docs topic,
  under 300 changed implementation/documentation lines; no new test abstraction.

Contract: current accepted spec revision 52, sections 3.3, 4.1, 4.5–4.7, 5 and
10.5. Revision 52 quadrature changes are outside this family-docs scope.

## Validation and handoff

Run the existing API-reference test in both dtype modes with six workers.
Run scripts/build_docs.sh directly, executing the documentation in both dtypes
and checking strict Sphinx warnings. Inspect the generated family section and
API entry. Run scripts/check.sh --fast --through A12. Commit this step and its
handoff, then independent review and stop. No new tests are required for this
reversible docs change; examples assert hand-derived properties.

## Docs to-do covered

Covers all D.LOGLOGISTIC.1 docs-to-do items: API, natural/log constructor usage,
parameter domains/units and flexsurv mapping. No further family-only docs
substeps are planned. Other Slice D APIs are not documented as implemented.
