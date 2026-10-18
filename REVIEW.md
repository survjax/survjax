# Spec alignment — revisions 25–27 (2026-10-18)

Read first: the new opening section of `PLAN.md`, the alignment notes in
D-022/D-021/D-016 in `DEVIATIONS.md`, then `.spec-seen`.
The reviewed design diff is `b917ebe..b92334a -- spec`.

## Recorded contract changes

- Revision 25: stable two-tail log CDF, two-branch private `log1mexp`,
  explicit tail/derivative/transform checks and zero absolute tolerance for
  tiny log values and derivatives (4.1–4.3, 7.2, 12.1).
- Revision 26: base-supplied survival/CDF/cumulative-hazard boundaries;
  differentiable `log_survival_at_infinity()` with a proper-family default;
  density/hazard exceptions and NaN precedence (2, 4.2–4.5, 7.1, 8.2, 12).
- Revision 27: finite positive substituted times for valid parameters;
  Slice A includes honoring improper-family infinity-hook overrides
  (4.5, 12.2). Implementation and docs follow separately after plan approval.

## Invariants and evidence

The marker records the latest reviewed spec commit, not implementation
conformance. Spec sources, independent tests and deviation statuses are
unchanged. No runtime code, docs pages or tests were changed or added.
The revised acceptance requirements are recorded in PLAN.md; no test-to-spec
verification is claimed by this bookkeeping step.

- `../scripts/spec-since.sh dev`: exit 0; listed `1e9011c`, `0008c82`,
  `b92334a`; one spec file, 155 insertions and 55 deletions.
- Read `git -C ../design diff b917ebe b92334a -- spec/SPECIFICATION.md`.
- `git -C ../design diff --exit-code HEAD -- spec/SPECIFICATION.md`:
  exit 0, no output. The script warning comes from the untracked generated
  `spec/SPECIFICATION.pdf`; it is outside this alignment.
- `rg -n 'log_survival_at_infinity' src tests/unit docs`: exit 1, no
  matches. The new hook has not been implemented in those files.
- `../scripts/spec-since.sh dev --mark`: exit 0; set `.spec-seen` to
  `b92334a1a4b584a7970039e321a6b10fe7c176ff`, revision 27; repeated the
  generated-PDF warning.
- `git diff --check`: exit 0, no output.
- `../scripts/spec-since.sh dev` after marking: exit 0,
  `dev: spec unchanged since b92334a` (same generated-PDF warning).

No runtime tests, dtype gates, lint/type checks or docs builds were run:
this step changes review records and the spec marker only.
Commit remains pending: the checkout is `main`, where AGENTS.md forbids
commits; the maintainer must provide the step branch. No branch was created
or switched. Deviation closure remains with the maintainer.
