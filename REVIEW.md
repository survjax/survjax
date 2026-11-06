# Merge resolution - Main wheel-test patch

Only PLAN.md and REVIEW.md required conflict resolution. Both parent review
records are preserved verbatim below; their evidence applies to those parent
states. Incoming historical plans already match the existing plan archive.
All other initially staged merge entries are retained unchanged.
No new implementation, numerical tests, docs build or independent review is
part of this record-only resolution. The maintainer owns the merge commit.

## Incoming main review (verbatim)

Independent review: round 1/3

# KWPATCH.1 — Installed-wheel keyword-only PH calls

Read PLAN.md KWPATCH.1, spec 4.4 constructor policy, then the two-line
change in tests/install/test_wheel.py.

Invariant: registered wrapper constructors require keyword arguments.
Both eager and jitted installed-wheel probes must pass baseline= explicitly.
Runtime code, expected analytic RMST values, isolation and dependency checks
are unchanged. No uncertainty or deviation identified.

Verification:
- Full tests/install/test_wheel.py with pytest -n 6: float64 4 passed
  in 47.78s; float32 4 passed in 47.96s. These exercise installed exports,
  wheel isolation/typing and eager/JIT PH RMST (spec 4.4, 10.3).
- Fast gate through A12: formatting 188 files, lint and mypy (22 files) clean;
  implementation float64 1066 passed / 8 skipped, float32 1071 / 3;
  source doctests 4 passed per mode.
- Independent acceptance float64: 3396 passed / 33 skipped, 2 NumPy reference
  warnings at zero (log and multiply). Float32: 3389 passed / 40 skipped, same 2 warnings.
  Gate exited 0: all gates passed through A12 (fast: slow tests skipped).
- git diff --check: passed. Full Tier 2 and docs build not run for this test-only
  correction; the touched file ran in full in both modes.

Start, previous step and worktree base: d9832baaa19559b7d49eb0e52279695c27efde34.
No remaining patch substeps after review.

---

## Docs-workflow parent review (verbatim)

# DOCS-WORKFLOW.1 — Merge resolution (uncommitted)

HEAD d89f2ba, incoming main d9832ba, original worktree base e6b3f9e.
Maintainer owns the merge commit. Round 3/3 remains pending that commit.
Read PLAN.md merge section, scripts/build_docs.sh, both docs build test suites,
DOCS-BUILD.md and the verification page's build section.

The parent records below are historical, preserved verbatim; their review
counts and completion statements refer to those parent trees.
Resolution: retain main's --output-dir/script-selected environment/cache/kernel
isolation and this branch's whole-pair identity/traps/fingerprint checking.
Identity lives in output_root; Jupyter runtime uses .cache/jupyter-runtime.
Existing success is invalidated before incoming uv dependency/path setup.
Main's index stays byte-identical; build/identity guidance is in its technical
verification page. Parent plans/reviews are retained verbatim. Contributor
inspection instructions are linked in root CONTRIBUTING.md and DOCS-BUILD.md;
the agent-only CONTRIBUTING section is excluded from its Sphinx include.
Round-2 N-2 is covered by incoming tests' dtype-to-output assertions. No
CLAUDE/role edits; discoverability now includes the contributor/build handoffs.

Validation on 2026-11-06:
- `JAX_ENABLE_X64=1 uv run --locked --group test pytest -q tests/tooling/test_build_docs.py tests/unit/test_docs_build_script.py --durations=3`:
  13 passed in 5.97s, slowest call .62s. Same with JAX_ENABLE_X64=0:
  13 passed in 5.76s, slowest .53s. An earlier 12-test interim run passed
  before the setup-failure regression was added. The added test verifies that
  dependency/path setup failure invalidates prior custom-root success.
  Both existing suites' assertions remain; fixture changes accommodate real
  Git identity, incoming uv path resolution and the .cache directory layout.
- `uv run ruff format --check .`: 192 files already formatted;
  `uv run ruff check .`: All checks passed; `uv run mypy src`: no issues
  in 22 source files. `bash -n scripts/build_docs.sh`: exit 0.
- `git diff --check`: exit 0. Full `git diff --cached --check`: exit 2 on
  incoming simulation CSV CRLFs and a new EOF blank line in incoming
  tests/spec/test_a11_ph_direct_methods.py. These objects are unchanged from
  the initial merge index. Resolution-file scoped cached whitespace check
  passes; no protected/unrelated cleanup performed.
- First direct `scripts/build_docs.sh` attempt: exit 1 on a Sphinx unresolved
  link from CONTRIBUTING.md's public include to CONTRIBUTING_DOCS.md. Added
  an include boundary; second default-permission attempt exited 2 with an
  invalid MyST end-before value (# treated as a comment) plus actual psutil
  cleanup PermissionError: [Errno 1] Operation not permitted (originated from
  sysctl() malloc 1/3). Both attempts left failed identity records, no success.
  Corrected boundary to an HTML-comment marker, reran direct
  `scripts/build_docs.sh` with approved escalation: exit 0, both fresh strict
  builds succeeded, 25 source pages per mode. No cleanup failure in the
  escalated invocation; known IPKernelApp TCP warnings remain. Identity:
  this checkout, d89f2ba with dirty merged source, started 20:00:23Z,
  completed 20:02:49Z, dtypes x32/x64, final state success. Warm MNIST cache;
  cold download and custom-root real notebook execution not tested (custom
  root/environment behavior covered by focused subprocess tests).
- Structural HTML checks (stdlib HTMLParser): both verification pages contain
  DIR/build-identity.txt, last-state and JUPYTER_RUNTIME_DIR text; both public
  contributor pages omit agent section/link. Index exists in both builds.
  These do not demonstrate screenshot/layout correctness.
- Visual attempt: cua.getState returned no apps/browsers, with errors
  "Native apps: Error: Sky Computer Use native pipe startup failed" and
  "Browsers: Error: The browser access host is unavailable". Documented native
  fallback cua.getApp('com.google.Chrome') returned "Sky Computer Use native
  pipe startup failed". No screenshots/layout inspection possible this turn.
- No unrelated numerical tests/gate, shellcheck or independent round 3 run:
  the maintainer owns the pending merge commit and review needs committed,
  clean input. No agent commit, merge, rebase or global configuration changes.

Remaining: maintainer completes merge d89f2ba + d9832ba; request independent
round 3/3 on that commit, and visual inspection when native/browser tools work.
Post-build edits below are plan/review evidence and staging only, not rendered
source changes. Final checks verify no unresolved entries/markers and preserve
all unrelated stage-0 objects from the initial merge index.

## This branch's parent review

Independent review: round 2/3

## Round-1 fixes for round 2

Round 1 on 11b19c3 eventually completed with VERDICT: maintainer. The review
is ../review/slices/DOCS-WORKFLOW.1.md. The prior usage-limit section below
is historical. Maintainer approved F-1/F-2/N-1 fixes and explicitly deferred
merging main into this branch until later. B-1 remains open for that merge;
this round requests review of the fixes on the existing base, not approval
of integration with main. No merge/rebase/protected edits or runner changes.

Read PLAN.md's follow-up, CONTRIBUTING_DOCS.md, docs/index.md Local commands,
then tests/tooling/test_build_docs.py. Original invariants/evidence below still
apply. New evidence maps directly to findings:

- F-1: agent computer-use, native Chrome fallback, screenshot/review guidance
  and narrowly scoped escalation now live in root CONTRIBUTING_DOCS.md,
  outside the Sphinx source tree. Public docs retain build/cache/identity,
  ordinary browser viewing, offline MathJax and JupyterLab information.
- F-2: test_further_edit_to_already_dirty_file_invalidates_success dirties
  .gitignore before a successful baseline build, changes it again in x64,
  verifies identical porcelain status before/after, and asserts failure/no
  success. This isolates the fingerprint check from status detection.
- N-1: stub uv logs its actual dtype invocations in ignored bin/dtype-calls;
  successful-build test asserts exactly ["0", "1"].

Validation on 2026-11-06 (this worktree, no escalation for these checks):

- `JAX_ENABLE_X64=1 uv run --locked --group test pytest -q tests/tooling/test_build_docs.py --durations=5`:
  5 passed in 4.76s, slowest call 0.51s.
- Same command with JAX_ENABLE_X64=0: 5 passed in 2.62s, slowest call 0.52s.
- `uv run --locked --group test python -` with temporary-checkout mutation
  harness: M2 replaces only fingerprint comparison with false; the F-2 test
  fails as expected. M4 replaces dtype loop with x64 only; the success test
  fails as expected. Harness exits 0 with both mutants caught. No mutation
  of source runner, tests or reviewer worktree, no persistent test assets.
- `uv run ruff format --check .`: 166 already formatted;
  `uv run ruff check .`: All checks passed; `uv run mypy src`: no issues
  in 22 source files; `bash -n scripts/build_docs.sh` and `git diff --check`:
  exit 0. No shellcheck or unrelated numerical rerun.
- Direct `scripts/build_docs.sh`: exit 0, both strict fresh dtype builds
  succeeded without escalation. Identity: this checkout, 11b19c3 dirty with
  planned changes, 14:36:51Z–14:39:08Z, x32/x64, final state success.
  Known IPKernelApp TCP warnings only; no Sphinx error or cleanup failure.
  Warm default MNIST cache; no cold-cache claim.
- Browser discovery again reports "Browsers: Error: The browser access host
  is unavailable". Native Chrome worked. Opened one new tab using absolute
  this-checkout local URLs, refreshed AX state after concurrent user activity,
  inspected AX content and screenshots for x32 index.html#local-commands and
  the x64 section's lower paragraphs. Readable wrapping/navigation, no overlap
  or horizontal clipping at 1920x1003; agent instructions absent, public math
  limitation retained. Closed only our tab. No full-site/math rendering audit.

Current follow-up starts at 11b19c3; previous completed work/worktree base is
still e6b3f9e. Remaining: maintainer integration with main (B-1) and review of
that resulting tree; later feature candidates below are unstarted. Stop after
this round's verdict; this fix-up does not claim B-1 resolved.

---

# DOCS-WORKFLOW.1 — Local build provenance and inspection

Read PLAN.md, scripts/build_docs.sh, tests/tooling/test_build_docs.py, then
Local commands in docs/index.md. Spec 10.1/10.5: locked optional tools, fresh
execution in separate dtype processes, warnings/errors fatal, rendered review.
No public API, dependency, numerical source, protected path or spec changes.

Starting worktree and previous completed work: e6b3f9e. Separate archive-only
commit: 8517177; incoming PLAN.md preserved verbatim in plans/pre-docs-workflow.md.

Invariants: the last state in the whole-run identity is authoritative. Starting
an attempt truncates any old success; only two successful Sphinx invocations
and unchanged tracked sources/commit/status append success. TERM/INT produce
failure; uncatchable death leaves building. Existing HTML may remain but is
uncertified. No concurrent builds/edits; untracked content and post-build edits
are explicitly outside detection. Dirty status is provenance, not a snapshot.
Build-local supported environment variables replace runtime/cache locations;
HOME and the shared MNIST default/explicit override are preserved. Installed
jupyter_core.paths, IPython.paths and matplotlib.__init__ confirm these variables.

Checks performed in this checkout on 2026-11-06:

- Direct `scripts/build_docs.sh`: exit 0, no escalation, both Sphinx builds
  succeeded with -E -n -W --keep-going and MyST force execution unchanged.
  Identity commit 85171775b6596757302168a3f3b0c2b3bedb6c81, dirty with planned
  changes; started 11:32:13Z, completed 11:34:48Z; dtypes x32/x64. Fifteen source
  pages per mode; runnable notebooks executed afresh. Kernel TCP warnings
  were emitted by IPKernelApp; no Sphinx build warning/error or cleanup failure.
  This used the warm default MNIST cache; cold acquisition was not tested.
- `uv run --locked --group test pytest -q tests/tooling/test_build_docs.py`:
  4 passed in 2.59s. Final rerun after formatting: 4 passed in 1.88s; slowest call 0.39s.
  Tests cover paths containing spaces, actual mkdir/export environment seen
  by the child process, preservation of explicit MNIST path, dirty provenance,
  absolute output paths, second-dtype failure/TERM and tracked edits removing
  old certification. Stub uv avoids numerical reruns; it does not test Sphinx.
- `bash -n scripts/build_docs.sh`: exit 0. Shellcheck not run.
- Initial `uv run ruff format --check .` caught one test-fixture quote format;
  `uv run ruff format tests/tooling/test_build_docs.py` fixed it (1 reformatted).
  Final `uv run ruff format --check .`: 165 already formatted;
  `uv run ruff check .`: All checks passed; `uv run mypy src`: no issues in
  22 source files. `git diff --check`: exit 0.
- No unrelated numerical gate/test rerun, per the task's explicit instruction.
  Git staging/commit needed escalation after sandbox index.lock denial;
  documentation execution itself did not need escalation.

Visual evaluation: cua.getState returned no browsers and exactly
"Browsers: Error: The browser access host is unavailable". Native
cua.getApp('com.google.Chrome') worked. Opened only one new tab, navigated to
absolute file URLs in this checkout, read AX content and captured/examined
screenshots of x32 index.html#local-commands (build/cache/identity paragraphs)
and x64 same section scrolled to fallback/escalation/JupyterLab paragraphs.
Text, inline code, margins and navigation are readable without overlap or
horizontal clipping at 1920x1003. Closed only our new tab and observed the
original tab remaining. This is a changed-topic visual check, not a full-site
math/table audit; no offline MathJax rendering claim. Both index sources are
identical across dtypes. Existing examples/results remain unchanged.

Uncertainties/limits: no cold MNIST download, no process-restriction failure
reproduced, no guarantee of native browser/escalation on other hosts, no
concurrency support. The post-build test formatting/this review are metadata
changes after the recorded build; rendered docs and runner remain identical.

Remaining candidates (not started): single-dtype authoring preview, offline
MathJax, docs CI/artifacts, browser-host repair. No further required sub-step
in this first workflow topic once independent review is accepted. Stop then.

## Independent-review invocation outcome

Implementation commit df167c62599b7beb9b6acab5cdafad6bfec96064 was clean before
invocation. Ran the exact skill command in ../review/code-docs-workflow;
exit 1: "You've hit your 5-hour limit · resets 3pm (Europe/Amsterdam)".
The expected ../review/slices/DOCS-WORKFLOW.1.md does not exist. The skill's
single permitted no-round retry returned the same exit/error; a second file
read again reported no such file. No reviewer verdict or findings exist.
Stop here under the skill; do not claim this step reviewed or accepted.
No reviewer configuration or global permission/settings changes were made.
The implementation is validated and committed; independent review remains
required when the maintainer resumes it after reviewer capacity is available.


## Incoming main parent review (verbatim)

# Merge resolution — Preserve both branch records

Historical records from both parents are retained verbatim below. Their
completion statements describe their respective branch states.

## Incoming release-prepare record

Independent review: round 1/3

# RELEASE.7 — Public repository cleanup

## Read first

1. PLAN.md RELEASE.7 and the maintainer-authorized cleanup scope.
2. Public guide and operator-handoff diffs; publication remains pending.
3. Archive renames, DOCS-BUILD link, removed simulation draft/config exclusion.
4. Bayesian notebook warning-only removal and fixture README status corrections.

## Invariants

No numerical/API/dependency/version/spec/protected-path changes. Archived
instructions remain verbatim historical records. Notebook sources and useful
results remain unchanged. Public-ready prose is not evidence of changed remote
visibility, hosting or package publication. Spec and active local orchestration
portability are explicitly deferred; author emails/history remain intact.

## Verification

- scripts/check.sh --fast --through A10: exit 0; all gates passed.
  Ruff: 181 files formatted, lint passed. mypy: 22 source files clean.
  Implementation float64: 1066 passed / 8 skipped; float32: 1071 / 3.
  Source doctests: 4 passed per mode. Independent spec float64: 2149 passed /
  27 skipped; float32: 2156 / 20. Slow cases deselected by the fast gate.
- Archive byte comparisons against HEAD passed for both moved files.
- Parsed notebook comparison passed: only cell 1 warning output removed;
  all cell sources, metadata and other outputs unchanged.
- Searches found no private-visibility or old-owner narration in active public
  guides, and no active reference to the removed simulation draft.
- git diff --check: passed. No new tests needed for prose/output cleanup.
- scripts/build_docs.sh directly: exit 0; both float32 and float64 builds
  succeeded with forced notebook execution. No hosting/public-access claim.
- python3 /tmp/survjax-public-cleanup-audit.py: exit 0; each dtype site has
  27 HTML pages and 2,069 valid local links/assets/fragments. Updated pages
  contain no private-visibility wording, simulation page is absent, Bayesian
  notebook download byte-matches the cleaned source. No browser-layout claim.

## Commit trace and remaining work

Previous step/start: dfa24b5. Original worktree base: 3cb4cd7.
Remaining: portable agent/spec setup, hosted checks and release acceptance and
publication. No independent completion claim for those workstreams.


---

## Existing main record

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
