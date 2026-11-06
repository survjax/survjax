# Independent Codex review outcome

Independent review: round 1/3, fresh gpt-6-astra session, reasoning low,
explicitly requested by the maintainer. Reviewed commit 91a4503 over range
23e8c3e..91a4503. Verdict: merge; no actionable findings and no fix-ups.
Full independent report: ASTRA-REVIEW.md (unaltered by implementation author).
Reviewer independently reran the fast gate successfully through A12 in both
modes and checked docs/example AST equivalence. Its limits and exact results
are in the report. Full Tier 2 remains the maintainer/CI gate.

This completed independent review supersedes the earlier waiver for the
alignment delta. The verification record below describes the state before
this review. Starting/previous step: 91a4503; worktree starting point: 3cb4cd7.

---

# ALIGN.3 — Final alignment verification

Independent review remains waived by the maintainer while reviewer tokens
are unavailable. Starting commit: 3b00d10; previous completed substep:
97d473f; original worktree starting point: 3cb4cd7.

Read first: PLAN.md ALIGN.3, incoming corrections in merge 3b00d10 and the
.spec-seen alignment. D-043 preserves all declared condition violations;
D-044 uses independent reference survival to limit upper-tail accuracy
checks to the active dtype's normal range, while preserving full-grid
shape/direct-grid agreement. Protected tests were supplied by their author,
not edited here. .spec-seen matches main and design HEAD 213cfb0 exactly.

No library changes in this step. Verify the full fast gate through A12;
record all outputs. Before the merge commit, the two affected spec files
passed x64 102 tests / x32 99 tests with 3 skips, plus 2 NumPy reference
warnings per mode. Prior implementation, touched-file, Bayesian and docs
build evidence remains in e8e9a51:REVIEW.md and a4c8b42:REVIEW.md.

Validation complete: scripts/check.sh --fast --through A12 exits 0. Tier 2, release
publication and a new independent review are outside this verification step.
This work closes the spec/acceptance alignment only, not all release demos.

## Final gate evidence

Ran `scripts/check.sh --fast --through A12` after the maintainer committed
merge 3b00d10. Output ends:
`check: all gates passed through A12 (fast: slow tests skipped)`.

- Protected-path guard: ok (spectests-rmst).
- Ruff format: 173 files already formatted; Ruff lint: All checks passed!
- Mypy: Success: no issues found in 22 source files.
- Implementation tests, x64: 1059 passed, 8 skipped in 14.36s.
- Implementation tests, x32: 1064 passed, 3 skipped in 15.46s.
- Source doctests, x64: 4 passed in 0.70s; x32: 4 passed in 0.45s.
- Spec acceptance through A12, x64: 3397 passed, 32 skipped, 2 warnings
  in 53.45s.
- Spec acceptance through A12, x32: 3391 passed, 38 skipped, 2 warnings
  in 37.76s.
- The warnings come from the independent NumPy hazard reference at t=0
  (log(0) and its zero product), not from a failed assertion.
- Gate builds one wheel for the installation and spec-wheel tests. Both
  dtype test runs use that wheel through the normal runner setup.
- Full log: /tmp/align3-fast-gate.log. `git diff --check` passes.

No code, tests, spec, docs pages or dependency changes in this verification
step. The final gate used the merged author corrections without skips,
xfails or tolerance edits by the implementer. The fast gate intentionally
omits existing slow-listed cases; full touched files were checked in ALIGN.1
and the two author-corrected files were checked in full in both modes during
conflict resolution. Full Tier 2 is not run here, per the tier convention.
Docs builds last passed both modes in ALIGN.d1; no executable doc/library
changes since then justify another build.

All planned alignment work is complete. Broader release demos, Tier 2 and
publication are not claimed complete. Independent review remains waived
as requested; no new reviewer invocation. Formal deviation statuses remain
for the maintainer, with merged correction evidence recorded in DEVIATIONS.md.
