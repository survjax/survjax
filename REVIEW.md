# SIM.merge-alignment — Maintainer-started merge

Read the newest PLAN section, the PH call-site edits, then the deviation-ID
mapping. Preserve the incoming independent report ASTRA-REVIEW.md unchanged.
No new reviewer or full suite is requested for this merge-resolution task.

Start/previous final: ac15812; incoming main: d1825f0; original worktree base:
8757c0f. Main removes the positional PH adapter. Simulation helper, plotting,
docs and notebook calls use baseline= explicitly, with no numerical change.
The notebook's local changes were execution counts/outputs only; preserve
those in the working file while staging only source changes against HEAD.
No protected acceptance files are edited by this resolution.

Both full parent plans are retained verbatim. Main deviation IDs 42-44 remain;
simulation IDs become 45-47. Entry bodies/statuses are preserved. Full parent
review records remain below so neither branch's verification is lost.

Validation complete:
- JAX_ENABLE_X64=1 uv run --group examples pytest -q
  examples/weibull_simulation/check_demo.py: 2 passed in 3.40 s.
- Fresh notebook smoke via nbclient: passed; execution copy saved at
  /tmp/weibull-merge-smoke.ipynb. No local notebook output was overwritten.
- scripts/build_docs.sh: exit 0, both x32/x64 builds succeeded.
- uv run ruff check .: All checks passed; ruff format --check .: 179 files
  already formatted. git diff --cached --check passes.
- No unmerged paths or conflict markers in the three resolved files.
- All shared deviation bodies match incoming main; all IDs are unique.
  Simulation entries retain their complete bodies/statuses with new IDs.
- git diff MERGE_HEAD -- tests/spec is empty. Incoming protected tests and
  ASTRA-REVIEW.md are retained without author modifications.
- Notebook source calls are keyword-only; local execution counts and outputs
  match the pre-resolution copy. The staged notebook has no saved outputs.
No full suites or new independent review ran. No merge substeps remain.

---

# Previous simulation review (ac15812)

# SIM.notebook-cleanup — Lean demo support

Read the newest PLAN section, run.py, and the notebook/docs changes. The
maintainer asked for an interactive notebook and removal of integration-test
infrastructure. The existing no-full-suite/no-reviewer waiver remains in force.
Start/previous final: 8349e7f; original worktree base: 8757c0f.

The cleanup replaces the raw diagnostic exporter and audit reducer with one
runner that writes the tables, histograms and figures used by the tutorial.
Six check modules and the stale dependency-probe document are removed; two
focused statistical checks remain. The generator/calibration functions remain
because MNIST and Bayesian examples import them. The fitter's numerical
operations and acceptance rules are unchanged. Failure reasons and rejected
fits remain in the study outputs. Existing committed full-study evidence is
retained as a historical artifact; new runs do not create audit manifests.

Support Python shrinks from 3,018 lines to 792 (including the two checks).
No dependency or public API changes. The notebook will run a fresh study and
show fresh plots; it never overwrites committed results. The maintainer's new
sentence about a future sample method is preserved.

Validation so far:
- Two focused checks pass (3.93 s): analytic exponential MLE/SE and rejection
  when an arm has no events. Initial test collection needed the example/test
  paths added; no tolerance change was made.
- The simplified full runner completed 12,000 fits. All 12 summary rows match
  saved statistical summaries and failure counts at the standard float64
  tolerances after aligning by scenario/model. Timing columns are excluded.
  Initial comparison failed only because rows were in a different order.
- Ruff import fixes applied to the new notebook and checks.

Remaining authorized delivery: execute the full notebook in a fresh kernel,
execute smoke via both docs builds, finish README/download links and record
final validation. No independent reviewer or full suites will run.

## Notebook delivery and final validation

Support cleanup is committed as b757e88. The 23-cell notebook follows the
accepted prose, preserves the maintainer's sample-method addition, computes
its own single-trial curves and study figures, and exposes CONFIG=full/smoke.
The README and docs link to it. Notebook formatting configuration matches the
other two use cases; checkpoint directories are ignored throughout the repo.

Both full and smoke notebook variants executed successfully in fresh Python 3
kernels via nbclient (300/180-second per-cell limits). Each produced three
figure outputs and an HTML results table with no error outputs. Source has
no execution counts or outputs. Executed review copies are in
/tmp/weibull-notebook-executed.ipynb and /tmp/weibull-notebook-smoke.ipynb.
A final prose-only fix removed the docs self-download paragraph from the
notebook; no code changed after execution.

scripts/build_docs.sh exited 0: both dtype builds passed with the simplified
smoke runner and notebook download. After the prose-only fix, both downloadable
notebook assets were refreshed and verified byte-for-byte against source.
Ruff check, Ruff format check (6 files) and git diff --check pass. No full test
suite or independent reviewer ran. No browser visual inspection is claimed.
All requested work is complete; no substeps remain.

---

# Incoming alignment review (d1825f0)

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
