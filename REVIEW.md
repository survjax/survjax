# Spec alignment — revisions 29–30

Requested 2026-10-20 on `A6.1`, after implementation commit `142b778`.
Read the full normative range from `0683e35` (revision 28) to
`fd150db3e82ad744277caecd1c68b2a524ad24df` (revision 30), and current GOALS.
This step records reading; it does not claim implementation conformance.

## Read first

1. PLAN.md's new revision-30 alignment section and the updated pending
   plain-comparison plan directly below it.
2. DEVIATIONS.md D-024/D-025 revision-30 notes; statuses stay open for the
   maintainer to close. The earlier questions/decisions remain as history.
3. .spec-seen, now pointing at the current spec commit.

## Contract changes and invariants

- Revision 29 adds the base dist.log_likelihood method, identical to the
  top-level function in values and derivatives. It removes LikelihoodIssues
  and log_likelihood_with_info; NaN causes come from data.issues and
  parameter_issues(), with the remainder representing conditioning/numerical
  failure. Update the later likelihood plans before their approval.
- Revision 30 requires ordinary JAX comparisons in the parameter dtype after
  bound conversion. It permits backend-dependent subnormal classification
  and adds no special handling, including at zero. nonnegative remains
  inclusive (finite x>=0); the earlier clarification is no longer needed.
- Do not change implementation or own/independent tests until the maintainer
  syncs the acceptance tests and gives the next implementation instruction.
  The current source still contains frexp and the own strict subnormal test;
  marking the spec as read is not proof of revision-30 conformance.
- No spec sources, protected tests, docs, dependencies or runtime APIs changed.

## Checks

- ../scripts/spec-since.sh dev: listed be8010c and fd150db; normative diffstat
  was 91 insertions, 44 deletions in spec/SPECIFICATION.md.
- Read the complete diff and current GOALS. design/spec has no uncommitted
  changes. No tests, lint, mypy, runtime gate or docs build run in this
  metadata-only step.
- ../scripts/spec-since.sh dev --mark: "dev/.spec-seen -> fd150db; commit it
  in dev". Repeated read-only command: "dev: spec unchanged since fd150db".
- git diff --check: exit 0, no output.

Stop after the alignment commit. The maintainer performs the spec-test sync;
no implementation work has begun or been resumed by this request.
