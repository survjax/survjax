Independent review: round 1/3

# RMST.integration — Main merge conflicts and remaining ownership

Read PLAN.md RMST.integration first; compare the two merge parents
(ddd7f1f and 4f17065), then tests/impl_tolerances.py and the preserved plans.
Read the release-agents/rmst.md and ph.md handoffs for remaining ownership.

The maintainer started this merge of main into release-rmst and explicitly
requested conflict handling. This commit completes that existing merge only;
no rebase, branch-to-main merge, push or publication is performed.

Resolution invariants:
- Only PLAN.md, REVIEW.md and tests/impl_tolerances.py conflicted.
- Tolerance additions are disjoint: keep both blocks verbatim, with unchanged
  constants/comments. RMST parameter/cutoff factors and simulation MCSE checks
  both remain available. Numerical functions and tests are unchanged.
- Both parents' PLAN.md contents are preserved verbatim as labelled records
  below the new current integration plan. Nothing unfinished was discarded
  or silently marked complete. Existing main archives remain unchanged.
- REVIEW.md is the new step record, following the overwrite convention.
  Prior full RMST review is in ddd7f1f:REVIEW.md; prior incoming README review
  is in 4f17065:REVIEW.md. RMST.3's final independent verdict was merge after
  3 rounds; all implementation/spec questions were answered by the maintainer.
- Except README.md's Ruff-only blank-line correction, every other inherited
  staged file equals main; source under src/ and the
  three RMST unit files equal the RMST parent. Imported CLAUDE.md and
  .workmux.yaml remain exactly main's versions, without local protected edits.
- .spec-seen is already 1389af4 (spec revision 39) on both parents; design main
  contains it. No spec edit or mark is necessary for a non-spec settings commit.

Remaining scope: the RMST core and mathematical appendix are complete. PH has
not landed in this main snapshot; it owns wrappers/exports and the built-in
closure edits after RMST core integration. The RMST worktree retains the
separate executable RMST docs topic and combined PH/RMST evidence once that
prerequisite lands. No PH implementation is authorized by this merge step.

Completed preservation checks: both plan records are exact complete parent
contents; both tolerance additions retained verbatim; no unmerged entries or
conflict markers; source and RMST tests byte-identical to ddd7f1f; inherited
files byte-identical to 4f17065 except the recorded README spacing. README's
Python fence AST equals main's, proving no example behavior change.

Completed commands: uv run ruff format --check . (125 files already formatted),
uv run ruff check . (all checks passed), uv run mypy src (no issues in 21
source files), uv lock --check (171 packages resolved, no lock change),
git diff --check / git diff --cached --check clean.
Focused tests (the three RMST unit files plus test_simulation_generator.py):
JAX_ENABLE_X64=1 uv run --group test pytest ... -q: 49 passed in 77.97s;
separate JAX_ENABLE_X64=0 process: 49 passed in 77.87s. The scope checks
both independently added tolerance blocks through their unchanged consumers.
scripts/check_wheel.sh --through A10 exited 0: all gates passed. Own suites:
1097 passed / 7 skipped x64; 1103 passed / 1 skipped x32. Source doctests:
four passed per mode. Independent acceptance: 2332 passed / 28 skipped x64;
2322 passed / 38 skipped x32. The wheel was built from the resolved tree and
provided to the unchanged gate, including install checks. Existing imported docs are unchanged;
this is not a docs authoring step. No new tests are needed for concatenating
unchanged tolerance constants and resolving plan/review records.

Initial check: Ruff format found two missing blank lines in the incoming
README's Python fence. Apply the formatter's whitespace-only correction;
no example semantics or normal docs page change. Rerun static checks.

For review of resolution edits, use the merge's two parents or its remerge
diff; the large incoming main delta was already authored on other branches.
No unresolved mathematical or ownership choice is introduced here.
