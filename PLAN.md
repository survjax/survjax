# Revision-46 alignment — inspection and proposed repair steps

ALIGN.0 complete, 2026-11-06, authorized by the maintainer's "on to next
part". Starting commit: 94a4474; previous docs step: 3cb4cd7; worktree starting
point: 3cb4cd7. Scope: move the stale .venv to a reversible temporary backup,
recreate from uv.lock, verify launcher paths, versions and static checks.
Repository files touched: PLAN.md and REVIEW.md (progress/evidence only).
No public API, dependency, code, test or protected-path changes. No new tests
for this environment-only repair. Spec 10.1/10.3. Stop after ALIGN.0;
ALIGN.1 and later steps remain prospective.

Completed evidence: rebuilt with `uv sync --locked --group test`; all 34
package versions match the retained environment exactly. pytest, mypy and
pre-commit shebangs now name this worktree's .venv/bin/python. Backup retained
at /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-align0-u4rcxltr/.venv.
Normal `uv run pytest --version`, `uv run mypy src`, Ruff format/check pass;
tests/unit/test_import.py passes all 3 tests in each dtype mode through the
normal pytest launcher. pyproject.toml and uv.lock have no diff. REVIEW.md
records exact commands and limits; full numerical gate was not repeated for
this environment-only repair. ALIGN.1 is next, not started.

## Starting points and inspected changes

- Inspection base: 23e8c3e (`merge spectests until a12`), branch spectests-rmst.
- Previous implementation/docs step: 3cb4cd7 (DOCS.6); intervening spec-seen
  update: 807edaf. Worktree's earliest reflog entry is also 3cb4cd7.
- Archive-only commit: 5b6f798. The completed DOCS plan was moved verbatim to
  plans/docs-through-docs6.md, with its index updated.
- .spec-seen already equals design HEAD c27ca14b89d50f72743c026bd1b5e238ca71fa07.
  Compared prior recorded design 00e7225 to this revision 46: revisions 42–46
  clarify wrapper construction/hazard, RMST derivative tolerances, shape
  Hessians, absent summary APIs and cutoff conversion/broadcast errors.
- Acceptance source: tests/spec/SOURCE_COMMIT records
  663d53df41f76e12985a816659ca52f60facb96c. Read the new A11/A12 tests, their
  TESTMAP/API_ASSUMPTIONS and affected implementation. Spec/tests unchanged.
- Section 12.2's first-release milestone explicitly brings PH and staged
  RMST forward; keep AFT, generic summaries and sampling deferred.

## Evidence and classification

Ran the new tests in full, with no slow-test filter, in separate processes:

```sh
JAX_ENABLE_X64=1 SPEC_THROUGH=A12 uv run --group test python -m pytest -n 6 tests/spec/test_a11* tests/spec/test_a12* -q --tb=short
JAX_ENABLE_X64=0 SPEC_THROUGH=A12 uv run --group test python -m pytest -n 6 tests/spec/test_a11* tests/spec/test_a12* -q --tb=short
```

Outputs: x64 **2 failed, 1246 passed, 5 skipped, 2 warnings** (116.37s);
x32 **3 failed, 1232 passed, 18 skipped, 2 warnings** (113.95s).
Warnings originate in the spec test's NumPy hazard reference at zero.
Logs: /tmp/rmsttest-a11-a12-x64.log and /tmp/rmsttest-a11-a12-x32.log.

1. Confirmed implementation gap, both modes: PH accepts a positional baseline;
   test_positional_baseline_is_rejected fails. Remove the adapter implementing
   the older D-028 decision and migrate callers to the current explicit rule.
2. Suspected acceptance-test discrepancy, both modes: nested rate_above_cap
   expects False for 1 > -1, despite the declared predicate returning True.
   D-043 contains the evidence; maintainer/spec-test author must resolve it.
3. Suspected acceptance-test discrepancy, x32 only: log_cdf_grid is compared
   to nonzero float64 references below float32's normal range, one below its
   representable range. Grid/direct agreement passes. D-044 records it.
4. No A12 failure. New value, derivative, Hessian, tail, invalid-entry isolation,
   broadcasting and summary-surface tests passed where not explicitly skipped.
   No numerical RMST rewrite is justified by this inspection.
5. Existing PH log_hazard override carries baseline limits and passes the new
   boundary checks. Cutoff dtype/shape checks and absent summary APIs pass.
6. D-042: spec prose/Bayesian snippet retain positional construction despite
   the explicit keyword-only rule. Request maintainer cleanup; no spec edits.

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, pytest 9.1.1.
The moved .venv contains launchers pointing at dev-spectests-rmst, its old
location. `uv sync --group test` succeeds but does not rewrite them. The first
`uv run ... pytest` attempts failed to spawn pytest; Python module invocation
works. An extra x32 construction-only run confirmed 1 failed, 25 passed,
1 skipped (the positional rejection).

Static checks: `uv run python -m ruff format --check .` reports 172 files
already formatted; Ruff check reports All checks passed; mypy src reports
Success: no issues found in 22 source files. `scripts/check.sh --fast --through
A12` passes its guard and Ruff checks, then exits 2 because its mypy launcher
cannot start. It never reaches tests. Full old-suite regression, wheel checks,
docs builds and Tier 2 were not run; this is not a passing release gate.

## Proposed sequence (each implementation/docs step ends with review and stop)

### ALIGN.0 — Restore the worktree environment

Recreate the local .venv at its current path from the unchanged uv.lock,
retaining a reversible backup until module and launcher checks pass. No
dependency/version changes, repository scripts or public API changes. Confirm
pytest/mypy launchers resolve here. This is an environment repair, not a
library step. Use normal runner commands for subsequent verification.

### ALIGN.1 — Enforce keyword-only PH construction and migrate executable users

Spec 3.3, 4.2, 4.4 (revision 42), 11.3; public API touched:
ProportionalHazards.__init__ rejects positional baseline, retains named fields.

Files: src/survjax/distributions/wrappers.py; eight unit-test files containing
27 positional calls (including intentional error cases);
examples/bayesian_rmst/{model.py,run.py,bayesian-rmst.ipynb}. Five positional
calls occur in the example Python files. Use AST/cell-aware inspection to
identify notebook callers. Retain intentional invalid-call tests where useful.
Remove the adapter/imports, correct the constructor docstring, migrate valid
calls, replace the obsolete test requiring positional/keyword equivalence with
explicit rejection coverage. Preserve keyword caller-site warning attribution,
jit/vmap construction, nesting, dtype promotion and broadcasting regressions.

Read the Bayesian use-case main/details docs before editing its example;
changes are constructor spelling only, with no regeneration of posterior data.
Run all touched unit files fully in both modes; the A11 construction suite;
Bayesian factor/route checks appropriate to the existing runner. Then the fast
gate through A12, recording D-043/D-044 failures if still unresolved. If these
block the review's passing-gate requirement, stop for resolution, not a waiver
invented by the implementer. Write REVIEW.md, commit and run independent-review
as required by its skill. Target under 300 changed implementation/test lines;
propose a split before exceeding that. Do not edit docs/ in this code step.

### ALIGN.2 — Resolve acceptance discrepancies with their owners

Maintainer/spec-test-author action: D-043/D-044; spec snippet cleanup D-042.
No protected test changes, skips, xfails, tolerance relaxation or diagnostic
suppression by the implementer. Rerun the precise failures after independently
supplied corrections. If a maintainer decision changes the actual contract,
re-plan the corresponding implementation before coding; keep it separate from
constructor migration. No numerical patch is proposed on current evidence.

### ALIGN.d1 — Update executable PH documentation

Spec 4.4, 8.7, 10.5, 11.3. Docs to-do from ALIGN.1: migrate PH examples and
remove the promise of positional baselines. Pages found:
docs/proportional-hazards.md, docs/rmst.md, docs/technical-notes/rmst.md and
docs/use-cases/bayesian-rmst.md; scan remaining executable sources too.
No public API changes. Keep correct shape-derivative limitations and the
deferred API surface. Current interior shape-Hessian demonstrations raise;
do not claim every supported differentiation order must raise.

Use one page/topic per docs step, at most 300 changed lines; split this into
d1a–d1d as needed. Run scripts/build_docs.sh directly, both dtype builds, with
rendered output inspection. Separate commits, REVIEW.md and independent review
for each step; no review waiver carries over from historical DOCS work.

### ALIGN.3 — Close verification evidence

After owner resolutions and repairs, run scripts/check.sh --fast --through A12
with working launchers. Repeat the full new A11/A12 suites in both modes when
changes justify it. Record exact outputs, unresolved deviations and links to
the relevant acceptance tests. No redundant RMST tests or formula changes
merely to manufacture a code step. Tier 2 remains the maintainer's merge gate;
demos/simulation retain their separate end-to-end gates and existing deviations.
Passing A11/A12 alone does not close those release obligations.

ALIGN.1 and later steps remain unimplemented. ALIGN.0 is a local environment
repair with planning/evidence records only, not a code or docs implementation
step requiring independent review.
