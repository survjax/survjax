# Review: Slice A delivery steps for the blind spec tester

Read in this order:

1. `roles/TESTER.md`, especially "Slice A delivery steps" and the corrected
   `test_a<N>_<topic>.py` instruction.
2. `CLAUDE.md`, "Slice A: suggested step order", to compare the A0–A9 schedule.
3. `PLAN.md`, first section, for the narrow scope of this metadata step.

## Invariants

- Spec 12.2 defines Slice A's deliverable and completion gate. The numbered
  A0–A9 steps are a maintainer delivery schedule, not new normative requirements.
- The blind tester derives assertions from `spec/GOALS.md` and
  `spec/SPECIFICATION.md` only. The schedule controls file names and step markers.
- A1 may use valid constructor paths to check representation and dtype; A2
  covers complete construction validation and `DataIssues`; A3 covers indexing
  and static metadata. A test is marked at the first step with all required
  public API available.
- `roles/REVIEWER.md` needs no change: it already reads the step plan and checks
  the spec gate. `tests/spec/` is read-only on this metadata branch.

## Evidence and checks

- No public API or executable test changed. The role table was compared with
  `CLAUDE.md` A0–A9 and the A1 boundary in the prior A1 plan.
- `git diff --check`: passed with no output.
- `scripts/check.sh --through A1`: guard passed; Ruff format said "13 files
  already formatted"; Ruff lint said "All checks passed!"; mypy said
  "Success: no issues found in 2 source files". Implementer tests: float64
  22 passed, 3 skipped; float32 25 passed. Independent spec tests: float64
  58 passed, 13 skipped, 313 deselected; float32 66 passed, 5 skipped,
  313 deselected. Final line: "check: all gates passed through A1".

## Follow-up for the independent tester

`tests/spec/API_ASSUMPTIONS.md` Q-001 currently assigns `DataIssues` to A3 and
several A1 tests use constructor paths. The tester should compare existing
markers and `TESTMAP.md` against the supplied schedule, retag or split tests as
needed, and update Q-001 on a `spectests-*` branch. These protected files were
not changed here. No uncertainty remains in the role text itself.
