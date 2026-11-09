# D.LOGNORMAL.2 — document the log-normal family

Feature: LogNormal API and executable user guidance.
Base and preceding-step commit: c3d097ca0bb8f581a795de1e84717298720aa363
([D.LOGNORMAL.1 plan](../D.LOGNORMAL.1/PLAN.md),
[handoff](../D.LOGNORMAL.1/REVIEW.md)).
Original worktree base: 3abefc5683a3aee8169eb2deffe57602897a3af3.
Dependency: D.LOGNORMAL.1. Cumulative gate target: A12.
The maintainer requested the next bit after the implementation handoff; this
step covers that handoff's deferred docs to-do, without expanding Slice D.

## Files and scope

- `docs/api.md`: add the LogNormal autodoc entry and describe the inherited
  direct/grid methods without limiting the reference to two families.
- `docs/builtin-families.md`: update the obsolete planned-family statement
  and add a LogNormal section with defining formula, parameter domains and
  R/NumPyro mappings, natural-parameter storage in `from_log`, executable
  constructor/grid examples, boundary behavior and guidance for log tails.
- Public API and docstrings: unchanged. No code, new tests, dependencies,
  protected paths, other step records or unrelated Slice D features.
- One documentation topic, under 300 changed implementation/documentation lines.

Contract: specification revision 51, sections 3.3, 4.1, 4.5, 4.6, 5 and 10.5.
Examples use only the implemented family and existing JAX facilities.

## Validation and completion

Run `tests/unit/test_api_reference.py` in both dtype modes (six workers),
`scripts/build_docs.sh` directly for both executable documentation builds,
and `scripts/check.sh --fast --through A12`. No new unit tests are needed
for prose and an autodoc entry; executable examples assert their identities.
Write the step handoff, commit, then use independent-review (up to three
rounds) and stop/report. Record shared-page integration needs for LogLogistic.

## Docs to-do covered

Covers D.LOGNORMAL.1's missing API entry and parameterization/tail guidance.
No additional docs substeps are planned for this family-only assignment.
