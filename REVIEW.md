# D-006 diagnostic compatibility review

## Read first

1. `spec/SPECIFICATION.md` sections 6.2, 10.2 and 13: the normative decision.
2. `spec/SPECIFICATION.md` sections 14 and 15: revision and decision history.
3. `DEVIATIONS.md` D-006: maintainer resolution.
4. `PLAN.md`: scope of this documentation step.

## Invariants

Ordinary traced `SurvivalData` construction remains valid. A traced
`DataIssues.raise_if_invalid()` call raises a plain `ValueError` with a
host-only message; concrete invalid diagnostics raise `InvalidDataError`.
`TracedInputError` is retired and not exported. Diagnostics are accessed through
the `data.issues` property, without a callable alias. This step changes no
runtime code or other deviation status.

## Evidence and uncertainty

The existing `tests/unit/test_data_issues.py::test_raise_if_invalid_is_host_only`
covers the traced-call error, and the A2 gate reported by the maintainer passed
in both dtype modes at commit `2e1ac04`. No tests were run for this
documentation-only step. `git diff --check` passed with no output. No remaining
D-006 question is known; D-010 and the other open entries are outside this step.
