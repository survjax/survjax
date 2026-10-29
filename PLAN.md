# A10.3 — Public concordance transformation and permutation evidence (2026-10-29)

Status: **F-2 fix-up verified; maintainer-authorized round 4 pending**, following
"implement and review (automatic via claude, use the skill) the next part of
A10" and the subsequent "approved". Work on the maintainer-created `a10b`
branch from clean `a5cfea7`.
The reviewed A10.2 API and main/A9 integration are present. The next outline
item is A10.3, before the separate A10.4 R-fixture step below.

Completed implementation: two new own-test modules, 231 lines, seven tests.
Focused concordance suites: 72 passed per dtype (21.99s float64 / 22.07s
float32), no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited
0, ending `check: all gates passed through A9`: own suites 1043 passed,
7 skipped / 1049 passed, 1 skipped; source doctests 4 passed per mode;
independent through-A9 suites 2211 passed, 28 skipped / 2202 passed,
37 skipped. Five non-fixture independent A10 modules: 113 passed (22.55s)
float64 / 112 passed, 1 skipped (30.07s) float32. Ruff: 92 files formatted,
All checks passed; mypy: no issues in 20 source files. Existing skips only.
Exact commands, timings, wheel identity, initial cache/lint failures and
limits are recorded in REVIEW.md. No production code changed; no full A10
claim. Final staged audit, commit and automatic Claude review follow.

Round 1 on fc7f82c returned fix-first for should-fix F-1 (no blockers/nits).
Add prepared-data static-metadata compilation evidence: equal flags reuse
one trace despite changed data/scores; changing flags produces another.
Retain raw tracing coverage with an accurate description. Final source/test
total: 260 lines, eight tests. Focused own suites now 73 passed per mode
(12.16s float64 / 12.22s float32), no skips; Ruff formatting/lint clean.
The earlier full gate/A10 selection are initial-commit evidence, not reruns
after this test-only follow-up. REVIEW.md records the distinction and the
reviewer's inability to execute checks under its headless permissions.
Commit the fix-up and request round 2 using the same skill; then stop.

The maintainer requested round 3 after correcting reviewer permissions.
Round 3 on bd1bd84 could execute checks/mutants and returned fix-first for
F-2: the third prepared-data call changed weak_type as well as metadata,
so a constant-metadata mutant survived. Use an explicit all-True event
array on that call, matching the other constructor paths. Final focused
suites: 73 passed per mode (13.31s float64 / 13.33s float32), Ruff clean.
The corrected test catches the constant-metadata mutant in both modes in
isolated temporary source copies. Source/test total: 263 lines, eight tests.
REVIEW.md records the reviewer's gate results (including missing-fixture and
unsupplied-wheel failures), mutant results, deferred N-1 and reviewer-owned
temporary worktree cleanup. Commit the F-2 fix-up and stop at three rounds;
no merge approval or full A10 claim, and no automatic A10.4 start.

The subsequent "do another round of review" explicitly authorizes one
additional independent review beyond the three-round limit. Review the
unchanged F-2 fix-up from 491f7cc; record round 4 and follow its verdict.
This request does not authorize starting A10.4 or changing protected paths.

One concept: complete own public-API evidence for transformations and
observation permutations. Aim for at most 300 changed source/test lines.
Public API and implementation behavior: unchanged. Spec sections: 6.1–6.3,
7.2 (compilation reuse), 9.1–9.2, 12.1 (transformations and metric shapes).

Files to create or change:

- Create `tests/unit/test_concordance_transformations.py`. Use distinct,
  hand-enumerated tied-time/tied-score cohorts, with counts and fractions
  derived in comments. Check `(B,1,N)` data against `(D,N)` scores, using
  different B and D lengths; compare eager, jit, nested vmap and jit-of-vmap
  to those expectations for values, all eight count words and all masks.
  Cover prepared inputs and raw construction inside the compiled function.
- In that module, test transformed invalid and unsupported cohorts beside
  valid and evaluated zero-pair cohorts. Check cohort isolation, preserved
  overlapping masks, zero failed counts and result shapes against explicit
  expectations. Add a trace counter showing equal-shape/dtype changes to
  scores, times and event values reuse compilation. No score derivatives.
- Create `tests/property/test_concordance.py`. Reproducible Hypothesis cases
  (`derandomize=True`) use small positive integer times, boolean events and
  bounded integer scores so ties and exact arithmetic are frequent. Verify
  joint row-permutation invariance of the complete result, count partition
  (`comparable = concordant + discordant + tied_score`), and score negation
  swapping concordant/discordant while preserving ties, masks and C's
  complement when defined. Cover no-pair NaNs explicitly. Identity-based
  expectations must not use production helpers or independent test helpers.
- Update this plan with actual results and overwrite `REVIEW.md` with the
  reading order, invariants, spec/test mapping, checks and limits. Retain
  historical plan records below. No changes to spec, `.spec-seen`, docs,
  protected tests/scripts, fixtures, dependencies or likelihood code.

Verification after approval:

- Run all own concordance modules, including both new modules, in separate
  `JAX_ENABLE_X64=1` and `=0` processes. Use existing central tolerances for
  non-exact floating identities; compare count words and masks exactly.
- Run Ruff format/check and mypy, then the unchanged
  `scripts/check_wheel.sh --through A9` for the combined-tree regression and
  installed exports. Run the five non-fixture independent A10 modules in
  both dtypes with `SPEC_THROUGH=A10`, without changing their selection logic
  or contents. The missing concordance R fixture precludes a full A10 claim.
- Inspect whitespace and the final diff; commit this step on `a10b` with
  `REVIEW.md` recording `Independent review: round 1/3`. From a clean tree,
  invoke the exact independent-review skill command in
  `../review/code-a10b`, step `A10.3`, and read `../review/slices/A10.3.md`.
  Follow its verdict, with at most three rounds and separate fix-up commits
  for blocker/should-fix findings; stop after this step and report.

Docs to-do: existing concordance topic/playground and held-out synthetic
example remain later steps. A10.4 owns pinned-R generation/reference tests.
Planning checks: read branch/status/history, the A10 outline, current public
metric and own/independent shape/property coverage, shared test fixtures and
spec sections above. No implementation, tests or review run during planning.

---

# A10 integration with merged main/A9 (2026-10-29)

Authorized by "main had moved and i've merge; check merge conflicts and
bring up to date; A9 has been merged now". The maintainer-initiated merge
is in progress: A10-cindex HEAD f71140a, incoming main ff9786f.
Five files conflict. This is merge resolution/integration of reviewed work,
not a new concordance feature or spec decision. Snapshot the initial index
and preserve every incoming non-conflict change except the two explicitly
listed integration edits below. Leave the merge commit to the maintainer.

Files to resolve: PLAN.md, REVIEW.md, src/survjax/__init__.py,
src/survjax/likelihood.py and tests/unit/test_import.py.
Necessary integration edits: src/survjax/distributions/base.py and
tests/install/test_wheel.py. No edits to spec, .spec-seen, independent
tests, protected scripts, incoming dependencies, fixtures or docs.

- Preserve main's method-only likelihood API and the two public A10 exports.
  Remove the obsolete standalone likelihood wrapper. Keep input-form
  normalization in _observation_inputs.py and import that same helper from
  the distribution method and metrics; do not duplicate the old resolver.
- Keep all incoming numerical likelihood behavior, A8 references, A9
  transformation/wheel tests, fixture handoff and spec alignment. Keep the
  withdrawn standalone A9 example absent. Extend the installed-wheel test's
  exact expected export set by the two reviewed C-index functions.
- Retain both branches' planning history below this current integration
  record. Overwrite REVIEW.md with current reading order, invariants,
  parent records, validation results and remaining A10 work.
- Sync the environment from the incoming lockfile without changing project
  metadata. Run focused integration checks, then scripts/check_wheel.sh
  --through A9, including a freshly built wheel, full own suites and unchanged
  independent through-A9 suites in both dtypes. Run the five non-fixture A10
  acceptance modules in both dtypes. A10 R fixtures remain a separate step;
  do not claim a full through-A10 gate or regenerate R data.
- Verify conflict markers/unmerged entries are gone, all seven intended
  resolution paths are staged, every other index entry matches the snapshot,
  and protected incoming files equal main. Record results and stop with the
  merge resolved and staged; no new merge, rebase or merge commit.


Validation completed 2026-10-29. All five conflicts are resolved/staged;
main's method-only API regression, the two C-index exports and the one
shared resolver are preserved. The incoming likelihood evaluator AST is
unchanged apart from moving its identical input resolver; only the method's
import changes. All other 173 original stage-0 index entries are unchanged.
Environment synced with uv sync --locked --offline --dev; no metadata edit.
Focused shared-input/likelihood/concordance/API suites: 120 passed per mode
(12.49s float64 / 11.93s float32). Fresh-wheel full gate:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check_wheel.sh --through A9`
exited 0, ending `check: all gates passed through A9`.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests, including installed wheel | 1036 passed, 7 skipped, 56.62s | 1042 passed, 1 skipped, 59.24s |
| Source doctests | 4 passed, 1.09s | 4 passed, 1.10s |
| Independent through A9 | 2211 passed, 28 skipped, 60.17s | 2202 passed, 37 skipped, 53.97s |
| Five non-fixture independent A10 modules | 113 passed, 28.26s | 112 passed, 1 skipped, 21.81s |

Ruff: 89 files already formatted / All checks passed! Mypy: no issues in
20 source files. No source/test changes followed these runs. Complete
commands, wheel identity, selection and remaining limits are in REVIEW.md.
Net-to-main and resolution-only whitespace checks are clean. The full staged
check reports only two pre-existing incoming-main EOF blank lines in
read-only tests/spec/API_ASSUMPTIONS.md and test_a6_likelihood_shapes.py;
confirmed identical in main and left unchanged. No new whitespace issue.
The incoming .spec-seen remains be26e50. Subsequent design changes only add
use-case companion links; concordance/likelihood requirements are unchanged.
Final metadata restaging/index audit follows; leave the merge commit to
the maintainer under the repository agreement. No full A10 gate claim.

Spec: 6.1, 7.1-7.3, 9.1-9.2, 10.2-10.3, 12.1. Public API is the current
method-only likelihood plus the already reviewed concordance functions.
No new tests beyond updating existing exact export expectations are needed
for this merge; use both branches' existing transformation/acceptance suites.
Docs to-do: concordance topic/playground/held-out example, as recorded below.
Earlier missing A8/A9 integration evidence is historical; update current
results here after the combined-tree checks have actually run.

---

# A10.2b2 review follow-up — N-5 and N-8 (2026-10-28)

Review follow-up to the approved public integration step, requested by
"check the nits in ../slices/A10.md" and the standing "check the review and
address" instruction. Read the actual report at ../review/slices/A10.md,
round 4 for 8109470: merge verdict, no blockers or should-fix findings.
Work on A10-cindex from clean 8109470; one test-only fix-up commit, then stop.
No new public API, numerical behavior or spec decision.


Completed 2026-10-29: N-8 adds three eager/jit unsupported-mode cases; N-5
adds distinct ClosedJaxpr descent after the Jaxpr case. Only 38 test lines
were added; production code is unchanged. Focused five-file selection:
67 passed in 6.56s (float64), 67 passed in 5.83s (float32), no skips.
Ruff: 78 files already formatted / All checks passed. Mypy: no issues in
20 source files. Both reported unsupported-mode mutants are caught in both
dtypes: E1 fails right-trunc only (1 failed, 2 passed); E2 fails interval
and left-censored (2 failed, 1 passed). Each mutation run deselects 23 tests.
REVIEW.md records exact selections, timings, scope and the untested distinct
ClosedJaxpr representation. Final whitespace/staged/scope checks follow.
Ready for the one test-only fix-up commit and review stop.

Files: tests/unit/test_concordance_entry_points.py,
tests/unit/test_concordance_counts.py, PLAN.md and REVIEW.md.
Spec: 9.1 (unsupported modes), 9.2 (whole cohorts and workspace), 12.1.

- N-8: add three parametrized from_bounds cases: finite right truncation,
  interval observation and left censoring. Each unsupported cohort has a
  valid neighbor with hand-counted C=3/8 and counts (1,2,1,4). Check eager/jit
  agreement, unsupported-only masks, zero failed counts, and no evaluated
  zero-pair flag. Inputs must be valid observations; unsupported is distinct
  from invalid. Preserve the existing left-truncation coverage.
- N-5: descend through distinct ClosedJaxpr wrappers in the recursive own
  workspace guard, after the Jaxpr case. Locked JAX 0.11.2 aliases the two
  classes; handle that without recursing on Jaxpr.jaxpr repeatedly. Do not
  claim other JAX versions were executed.
- Reconcile the review's focused-count difference: its four files total 63;
  the previously reported 64 also included test_api_reference.py. Keep the
  original evidence intact and record the selection explicitly in REVIEW.md.
- Run the five focused own modules in separate dtype processes, Ruff, mypy
  and whitespace/protected-scope checks. In temporary source copies only,
  remove each unsupported-mode term in turn and require the new cases to
  catch both mutants in both modes. Do not mutate the working source.
- This is test-only: production runtime is unchanged from the passing public
  integration gate. No full regression/independent A10 rerun is necessary;
  record that limit honestly. No fixture, docs, dependency, protected test,
  .spec-seen, sync or likelihood changes. No merge or next subslice.

---

# A10.2b2 — Public concordance entry points (2026-10-26)

Status: **implemented and verified; ready for review**, authorized by
"review approved ... get on with next subslice". Work in dev-a10 on A10-cindex from clean `715a7e3`; no merges.
The maintainer also explicitly allows the two API-reference entries needed by
the export-coverage test, as a narrow exception to the code-step docs rule.
Tutorials and broader docs remain separate. The shared spec is now revision
37; its concordance contract is unchanged. Likelihood alignment/spec sync
remain owned elsewhere; do not change .spec-seen or that API here.


Final evidence (completed 2026-10-27): both public entry points and their
exports are implemented. Own focused suites pass 64 tests per dtype
(6.87s float64 / 6.27s float32), no skips. The five selected independent A10
modules pass 113 tests in float64 (45.13s), and 112 with 1 existing skip in
float32 (47.97s). This is focused A10 evidence, not a full A10 gate.

The initial through-A7 attempt failed only the old exact export smoke list
(1 failed, 848 passed, 7 skipped); adding the two intended names fixed it.
REVIEW.md records that failure and correction. The final unmodified
through-A7 gate exited 0, ending `check: all gates passed through A7`:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 849 passed, 7 skipped, 171.12s | 855 passed, 1 skipped, 172.98s |
| Source doctests | 4 passed, 1.03s | 4 passed, 0.97s |
| Independent through A7 | 2123 passed, 28 skipped, 223 deselected, 121.84s | 2127 passed, 24 skipped, 223 deselected, 116.32s |

Ruff: 78 files already formatted / All checks passed. Mypy: no issues in
20 source files. No new skips or exclusions. Optional complete output:
`/private/tmp/survjax-a102b2-final-gate.log`; this table is the durable summary.
REVIEW.md was finalized before the final gate and is unchanged afterward.
Source/test changes total 291 lines, plus seven approved API-reference lines;
the metrics module is 220 lines. Final static/whitespace/scope checks follow
this plan-only result edit. No merge, spec/sync marker, independent-test,
likelihood API or dependency changes; no full A10 completion claim.
Stop after this step's commit for review.

One concept: connect checked observations and caller scores to the existing
counter/result layer, with public exports. Target at most 300 source/test
changed lines; split if that budget is exceeded.

- Update metrics.py with concordance_index and concordance_index_with_info
  using the shared _resolve_data helper and one private cohort evaluator.
  Accept real JAX-supported score arrays without casting their precision to
  the data dtype. Reject static type/shape misuse; invalid runtime values give
  NaN plus masks. Recheck current data fields and retain construction issues.
- Require matching non-scalar last observation axes; broadcast only leading
  axes. Reduce invalid/support evidence per cohort and explicitly broadcast
  every mask to result shape (review N-7). Replace entire failed cohorts with
  safe finite arrays/boolean masks before counting; never drop observations.
  Flatten leading batches for vmap and restore result shapes. Empty cohorts
  remain evaluated zero-pair cases. Preserve data-float ratio dtype.
- Export both functions in __init__.py. Add only their autodoc entries and
  minimal section wrapper in docs/api.md, as approved. Rename result field
  docstring headings to Attributes (review N-6). Do not change other docs.
- Add tests/unit/test_concordance_entry_points.py: hand-counted raw/prepared
  values, all five shape-table rows with every info leaf checked, failed-
  cohort isolation/mask overlap, edited-field validation, empty/scalar/mismatch
  cases, preserved score ordering/precision and jit/vmap agreement.
- Update tests/unit/test_import.py's exact export set with the two intended
  additions; retain every existing name pending separate likelihood alignment.
- Update D-027's implementation note, this plan, and overwrite REVIEW.md with
  reading order, invariants, spec mapping, actual check outputs and limits.

Spec: 3.2–3.3, 6.1–6.3, 7.2, 9.1–9.2, 10.2, 12.1; accepted D-027. No new
helpers/dependencies outside this concrete evaluator, no score gradients.
Run new/existing concordance own suites plus the five non-fixture independent
A10 modules in separate dtype processes; static checks and the unmodified
through-A7 regression gate. A10 R fixtures/A8/A9 work remain unfinished;
selecting the five modules is focused evidence, not a complete through-A10 gate.
Verify only approved API-reference edits, protected paths and sync marker,
record results, commit this one step on A10-cindex, then stop for review.
Docs to-do: executable concordance tutorial, playground and held-out example;
no full docs-build claim for this code step's two reference entries.

---

# A10.2b1 — Concordance result containers and ratio (2026-10-25)

Status: **implemented and verified; ready for review**, authorized by
"review approved ... get on with next subslice". Work only in dev-a10 on A10-cindex, starting clean at
`10b8877`. Do not merge, rebase or touch the separately owned spec alignment,
test sync, .spec-seen, likelihood API or protected paths. The checkout retains
revision-34 API alignment while accepted spec revision 35 is visible through
the shared read-only link. This step is independent of the likelihood change.


Implementation evidence: a 100-line result module and 172-line test module
(272 source/test lines). Focused result/count suites pass, 39 per dtype
(2.71s x64 / 2.15s x32), including all 14 new result cases. No failures/skips.
The unmodified through-A7 gate exited 0, ending
`check: all gates passed through A7`:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 826 passed, 7 skipped, 145.51s | 832 passed, 1 skipped, 145.95s |
| Source doctests | 4 passed, 0.87s | 4 passed, 0.81s |
| Independent through A7 | 2123 passed, 28 skipped, 223 deselected, 96.59s | 2127 passed, 24 skipped, 223 deselected, 90.68s |

Ruff: 77 files already formatted / All checks passed. Mypy: no issues in
20 source files. No new skips or exclusions. Optional raw output:
`/private/tmp/survjax-a102b1-final-gate.log`; this table is the durable summary.
REVIEW.md was finalized before the gate and is unchanged afterward. Final
format/lint/mypy and working/staged whitespace/protected-path checks follow
this plan-only result edit. Both package/module imports resolve to dev-a10.
No spec, .spec-seen, independent tests, likelihood API, docs, dependencies,
lockfile, top-level exports or other branches changed. No merges or full
through-A10 completion claim. Stop after this step's commit for review.

One concept: turn already-counted pairs into transformable diagnostics and a
floating C-index. The full public integration exceeds one review-sized step;
implement this result layer first, with source/tests under 280 changed lines.

- Create `src/survjax/metrics.py`: frozen registered `ConcordanceCount` (hi/lo
  uint32 array fields) and `ConcordanceInfo` (the four accepted count fields,
  invalid, unsupported and no_comparable_pairs arrays). Result types stay in
  their module (spec 6.3/10.2); no top-level export or metric entry point yet.
- Add private `_concordance_result(counts, invalid, unsupported, dtype)` in
  that module. Inputs are aligned count words and per-result masks. Preserve
  exact uint32 words; zero all counts on failed cohorts, retaining both masks.
  Cast words into the data float dtype before reconstructing totals and
  dividing. Award half credit to tied scores. Substitute a safe denominator
  for zero-pair cohorts before division; return NaN with no_comparable_pairs
  only on evaluated zero-pair cohorts. No floors, epsilons or score gradients.
- Create `tests/unit/test_concordance_results.py`: hand-counted fractions,
  high-word totals, undefined/failed/overlapping-mask cases, per-result array
  shapes, preserved data dtype, frozen fields and pytree reconstruction;
  eager/jit/vmap agreement in both dtype processes. Expected counts/fractions
  come from explicit integer arithmetic, not code under test. Use existing
  central tolerances; do not add numeric tolerance constants.
- Update this plan, the implementation note in D-027 (retain formal open
  status), and overwrite REVIEW.md with reading order, invariants, requirement
  mapping, exact commands/results and limits. No docs or dependency changes.

Spec: 3.3 (floating division), 6.3 (result pytrees), 7.2 (safe operands),
9.1–9.2 (counts/ratio), 10.2 (result placement), 12.1; accepted D-027 contract.
Validate focused result/count suites in separate float64/float32 processes,
Ruff format/lint, mypy, and the unmodified through-A7 regression gate. Verify
imports use dev-a10. No through-A10 completion claim or independent-test edits.
Record actual results, inspect staged/protected diffs, commit and stop.

Docs to-do: retain the later concordance topic/API reference, adding named
hi/lo counts and result-mask interpretation once public integration is reviewed.
Next step: public observation/score validation, cohort-safe inputs, leading-axis
broadcasting and entry points, after coordinating pending alignment and the
code-step/API-reference documentation constraint. No automatic next step.

---

# A10 review follow-up — N-1 through N-4 (2026-10-25)

Status: **implemented and verified; ready for review**, authorized by
"check the review and address". Read all of `../review/slices/A10.md` for
`04bc1a7`: merge verdict, no blockers or should-fix findings. One follow-up
commit on `A10-cindex`, then stop; starting tree was clean at `04bc1a7`.

Files: `src/survjax/_concordance.py`, its own test module, PLAN.md,
DEVIATIONS.md and REVIEW.md. Public API: none. Source/test budget: under 60
changed lines. Spec: 9.1–9.2 workspace/preconditions and 12.1 verification.
No numerical, dependency or docs changes; the shared resolver is unchanged.

- N-1: document the boolean (N,) exact-mask precondition.
- N-2: inspect the traced JAX program recursively, including loop bodies, for
  an N-by-N intermediate at N=7. Verify sensitivity in a temporary source
  copy using an equivalent N-by-N comparison inside the loop; require the
  shape test to fail and the hand-counted value test to pass in both dtypes.
- N-3: require finite safe substitutes for entire invalid/unsupported cohorts
  before public counting. Retain cohort length/masks and zero failed counts;
  never drop rows. Record this in the later public integration outline.
- N-4: remove the redundant high-word assertion/import, make stale merge text
  historical, and commit complete check commands/results in REVIEW.md.
  Temporary logs supplement that durable evidence; no logging infrastructure.
- Record the maintainer's accepted D-027 decision, retaining its formal open
  status for maintainer disposition. No public result type implemented here.
- S-1: revision 35 alignment, spec-test sync and `.spec-seen` belong to another
  agent/branch. Do not edit, mark, merge or sync them here. S-2's early A10
  authorization is recorded; no delivery/role edits. Reconcile dependencies
  after those changes land, before starting public integration.

Verification: focused counter/shared-input/likelihood entry-point suites pass,
93 tests per dtype (5.90s x64 / 5.40s x32), no skips. The N-by-N mutant is
rejected while its hand-counted value test passes: 1 expected failure / 1 pass
in 0.21s per dtype. REVIEW.md records commands and sensitivity details.
Final Ruff: 75 files already formatted / All checks passed. Mypy: no issues
in 19 source files. Working/staged whitespace checks pass. An AST comparison
confirms production code is unchanged except docstrings; protected paths,
`.spec-seen`, docs and exports are unchanged. No full numerical gate rerun for
this docstring/test follow-up; the reviewer's passing through-A7 gate is
historical evidence only. No through-A10 completion claim. Docs to-do: retain
the separate public concordance documentation step. Stop after this commit.

---

# A10.2a — Share observation input normalization (2026-10-25)

Status: **implemented and verified; ready for review**, authorized by
"ok no issues here on to the next bit, which is approved by me". A10.1 is accepted. Split the next integration
step: first relocate its existing data-input resolver, then add public metrics
once D-027 is settled. This preparatory part introduces no public contract.

At planning start, `A10-cindex` was at `67bac4d` with the maintainer's merge
in progress and 21 resolved paths staged. This step preserved that index and
kept its own work unstaged until the maintainer completed the merge. Incoming
protected spec tests and the sync script were not edited.

The maintainer completed the merge as `34a7657` during this step. Verified
no merge remains and its index is empty; all new work stayed unstaged. The
shared resolver is independent of D-027, which the maintainer wants to discuss
before public integration; no proposed public choice has been adopted.


Implementation evidence: the resolver AST is identical to the original function
in the merged HEAD. The helper is 21 lines; new shared-input tests are 78 lines.
Together with its import/relocation, 116 source/test lines changed. Focused
shared-input/public-likelihood tests: 68 passed per dtype (4.35s / 3.79s),
without skips. The full through-A7 regression gate exited 0:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 811 passed, 7 skipped, 153.02s | 817 passed, 1 skipped, 169.74s |
| Source doctests | 4 passed, 1.13s | 4 passed, 1.05s |
| Independent through A7 | 2123 passed, 28 skipped, 223 deselected, 115.04s | 2127 passed, 24 skipped, 223 deselected, 107.53s |

Ruff: 75 files already formatted / All checks passed. Mypy: no issues in
19 source files. Full output: `/private/tmp/survjax-a102a-final-gate.log`;
last line `check: all gates passed through A7`. No new skips, test exclusions,
public API or numerical changes. The larger deselection count reflects incoming
later-step independent tests. REVIEW.md was finalized before the gate and is
unchanged afterward. Final static/diff checks follow this plan-only result edit.
D-027 records discussion only. The maintainer's merged docs/protected files
remain unchanged. No R, docs build, installation or performance check was run.

Files and public API:

- Create `src/survjax/_observation_inputs.py` and move `_resolve_data` there
  verbatim from `likelihood.py`; import it back at its existing call site.
  Preserve all checks, error messages and pure SurvivalData normalization.
  No public API, dtype, numerical, dependency or docs change.
- Create `tests/unit/test_observation_inputs.py` for shared-normalizer behavior
  needed by both callers: raw invalid-indicator evidence through eager/jit/vmap,
  missing/mixed input rejection, and prepared rich observations preserving
  every pytree leaf (including issue bits). Reuse the existing public likelihood
  suite for the unchanged values/first/second derivatives and input contract.
- Update this plan and REVIEW.md with this step's actual checks and limitations.
  Preserve the staged merge versions of those files until merge completion.
  Target under 150 changed source/test lines; one concept and one commit.
- Record the newly synced count-reader assumption and the maintainer's request
  for discussion in D-027, retaining its open status; adopt no public contract.

Spec: 6.1–6.2 (single shared checked normalizer), 3.2–3.3 (prepared leaves and
raw issues), 7.1/12.1 (likelihood regression evidence). No score contract or
ConcordanceInfo field/representation choice is made here. D-027 remains open;
new independent Q-042–Q-045 explicitly label their public choices assumptions.

After editing, run focused new/likelihood entry-point tests in separate x64/x32
processes, Ruff format/lint and mypy. Run the unmodified through-A7 gate for
regression evidence; A8 fixtures/A9 completion and the public A10 API remain
unimplemented, so do not claim a through-A10 pass. Record failures honestly.
Check whitespace, source diff and preservation of the incoming staged index.
Commit only this step after the maintainer finishes the merge, then stop.
If the merge is still pending, leave a checked, reviewable diff and report
that the step cannot yet be committed under the no-merge rule.

Docs to-do: the separately approved concordance docs topic/API reference after
public integration. No documentation is needed for moving a private resolver.

---

# Merge-conflict resolution (2026-10-24)

The maintainer requests resolving the in-progress merge on `A10-cindex`.
Resolve only `PLAN.md` and `REVIEW.md`, preserving the A10.1 records and the
incoming A7 docs and spec-test sync records. Keep prior reviews as historical
evidence; their statements describe their original checkouts. No public API,
source, spec, independent test or requirement changes; no new tests needed.
Check conflict markers, whitespace, preservation of both sides and the Git
unmerged-file list. Stage the two resolutions without creating a merge commit.

---

# A10 — Harrell's concordance (2026-10-23)

Status: **A10.1 implemented and verified; ready for review**, authorized by
"Approved go for it". The maintainer requested checking the updated roadmap
and starting A10. The existing branch is `A10-cindex`, starting at `9a034f4`.
The request authorizes bringing concordance forward now, ahead of the draft's proposed after-A9 sequence; it does not
authorize spec edits or publication. Statistical requirements remain accepted
spec revision 34, sections 3.3, 6.1–6.2, 9.1–9.2, 10.2 and 12.1.

Implementation evidence: a 61-line private counting module and 169-line test
module (230 source/test lines). Exact comparisons, event-first equal-time
pairs and two-word totals; no public API change. Focused checks pass, 24
per dtype (1.37s x64 / 1.34s x32), without skips. Final through-A7 regression
gate exited 0 and ended `check: all gates passed through A7`:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 801 passed, 7 skipped, 141.70s | 807 passed, 1 skipped, 143.20s |
| Source doctests | 4 passed, 0.82s | 4 passed, 0.81s |
| Independent through A7 | 2123 passed, 28 skipped, 44 deselected, 94.06s | 2127 passed, 24 skipped, 44 deselected, 90.73s |

Ruff: 73 files already formatted / All checks passed. Mypy: no issues in
18 source files. Full output: `/private/tmp/survjax-a101-final-gate.log`.
REVIEW.md records the focused commands, initial cache/lint failures and limits;
its metadata was finalized during the gate, with final static checks afterward.
No new skips or exclusions. No A10 independent suite, R fixture, docs build,
installation or performance evidence claimed. D-027 remains open. Stop after
this step's commit; A10.2 has not started.

Read `../design/ROADMAP.md` and its linked
`../design/proposals/concordance-at-end-of-slice-a.md`. Both describe a draft
delivery amendment: accepted section 12.2 still assigns concordance to E.
Do not modify the design repo or treat its proposed completion gate as adopted.
Do not claim A8, A9, the A7 docs or Slice A completion through this work.

Starting working-tree changes belong to the unfinished A7 docs step:
`PLAN.md`, `docs/index.md`, `docs/likelihood.md` and untracked
`docs/numerical-behaviour.md`. Preserve them. Prepend this plan without
rewriting its existing content; stage only A10 changes when committing.
No docs edits in A10 code steps. No dependencies, fitting helpers, KM, IPCW,
Brier, IBS, AUC, approximate metric API or score-gradient guarantee.

## A10.1 — Private comparable-pair counting (first approval and review stop)

One concept: exact counts for one valid, supported cohort, without public
entry points. Target approximately 230–280 changed source/test lines, plus
the plan and review. Split and replan if implementation exceeds 300 lines.

Files to create:

- `src/survjax/_concordance.py`: a private typed counting function for
  one-dimensional finite scores, finite times and exact/right status.
  Return concordant, discordant and tied-score totals as private `(hi, lo)`
  tuples of `uint32` scalar arrays. Each total denotes `hi * 2**32 + lo`.
  Keep this representation private pending D-027. Comparable totals are
  accumulated during the same loop. No public API or export changes.
- `tests/unit/test_concordance_counts.py`: hand-enumerated pair counts and
  eager/jit/vmap checks; private accumulator carry checks using synthetic
  word values. Test expectations come from Python integer arithmetic and
  explicit pair lists, never from the implementation being tested.

Update this plan's status and overwrite `REVIEW.md` only at the implementation
review stop, recording reading order, invariants, requirement-to-test mapping,
actual check output and remaining limitations.

Implementation invariants (spec 9.1–9.2):

- A JAX loop compares one row with the cohort and counts only partners with
  larger indices. Work is O(N squared), pair workspace O(N); never form an
  N-by-N array. Later leading-axis batching calls this core with `vmap`.
- For unequal times the earlier row must be an event. At equal times only
  event/censor pairs are comparable, with the event earlier. Compare scores
  exactly; higher score on the earlier event wins. No near-tie tolerance.
- Sum each loop's boolean counts as `int32`, then add the nonnegative count
  into two unsigned words with explicit carry. No int64 requirement and no
  total-count cap imposed to avoid int32 overflow. The cohort length must fit
  the per-row int32 counting contract; no quadratic large-cohort test needed.
- This core accepts already validated cohorts. Invalid scores, unsupported
  modes, stored data issues and floating ratios belong to the integration
  step; no public function returns partial or placeholder results meanwhile.

Own tests accompanying this step:

1. A small cohort with explicitly listed concordant, discordant and score-tied
   pairs; all counts sum to the hand-counted comparable total.
2. Separate equal-time event/event, event/censor and censor/censor cases,
   plus censoring earlier than an event. Include reversed array order to
   ensure array position never substitutes for event-time order.
3. Perfect, reversed and all-tied scores; singleton, empty and all-censored
   inputs produce zero counts in the private core. Public acceptance of empty
   inputs remains a D-027 question; do not infer it from core behavior.
4. Row permutations preserve all totals; eager, jit and cohort-wise vmap
   agree. Counts remain exact in separate x64/x32 processes.
5. Carry across `2**32`, repeated carries and high-word totals, checked
   against arbitrary-precision Python integers. Exercise totals beyond signed
   int32 capacity directly without allocating an enormous cohort.

Verification after approval:

- Run `tests/unit/test_concordance_counts.py` in separate
  `JAX_ENABLE_X64=1` and `=0` processes via `uv run --group test pytest`.
- Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
  for the established implementation baseline. The existing A8 acceptance
  tests require unfinished fixture work; do not label this an A10 full gate.
  No independent A10 suite is currently synced; request maintainer coordination
  before declaring A10 complete. Never edit `tests/spec/` or the gate script.
- Run `git diff --check`; record pre-existing failures separately without
  repairing unrelated docs. Check staged diff excludes the A7 docs changes.
- Commit only this approved step on `A10-cindex`, with imperative subject and
  a body explaining why, then stop for review. No automatic A10.2 start.

Docs to-do: after public integration and review, a separate concordance topic
with score direction, pair/tie conventions, support, batching, masks,
undefined results, complexity and held-out synthetic evaluation. Update the
API reference and playground in the separately approved docs integration.

## Later A10 steps — outline only; each needs its own approved plan

- A10.2b: after the separately owned spec alignment and test sync land, refine
  the public integration plan around the maintainer's accepted D-027 decision.
  Introduce `metrics.py`, frozen registered count and `ConcordanceInfo` types,
  and both public entry points. `_resolve_data` is now shared by A10.2a; use
  that same helper. Validate last-axis lengths before broadcasting leading
  axes; reduce invalid/unsupported evidence per cohort. Before calling
  `_pair_counts`, substitute finite safe scores/times and a boolean exact mask
  for each entire invalid/unsupported cohort, preserving observation count and
  failure masks. Never drop rows or count a partially valid cohort. Force all
  returned count words to zero for those failed cohorts, and set
  `no_comparable_pairs` only for evaluated cohorts. Honor stored raw-input
  issues; compute the ratio in the data float dtype,
  casting words before floating arithmetic and using a safe denominator for
  undefined cohorts. Add input-form, masks, dtype and shape tests with code;
  split integration if its source/test budget exceeds 300 lines. Resolve the
  current export-reference test's docs requirement before adding top-level
  exports: code steps cannot silently edit docs or weaken that test.
- A10.3: finish public transformation/broadcast and permutation evidence in
  both dtypes; independent acceptance tests arrive only from the maintainer.
  No score derivatives are promised. Maintain one concept per small commit.
- A10.4: synthetic R generator, fixture README and reference tests, following
  section 10.4's pinned R/package requirements. Use `reverse=TRUE`,
  `timewt="n"`, `timefix=FALSE`. If the pinned environment is unavailable,
  hand off the generator and stop; never invent reference output.
- A10 docs and final audit: executable topic/playground, held-out synthetic
  example, installed wheel/source-archive exports and evidence map. Coordinate
  with unfinished A8/A9 work; a concordance step alone cannot close Slice A.

Planning checks actually run: branch/status/history inspection; accepted
sections above, roadmap and draft amendment read; existing input resolver,
dtype policy, data/diagnostic containers, export-reference test and gate read.
`../scripts/spec-since.sh dev` prints
`dev: spec unchanged since d243a96`. No implementation, tests, build, R
environment check, spec edit, spec-seen update or commit during planning.

---


# A9 scope for merge — remove the standalone example (2026-10-29)

Approved directly by the maintainer: merge the A9 work except the slice_a
example, which is superseded by use-case 1 and a mini version in the README.
This narrow removal updates PLAN.md/REVIEW.md and deletes exactly:

- `examples/slice_a.py`;
- `tests/unit/test_slice_a_example.py` (its three dedicated regression cases).

Public API touched: none. No tests are added for deleting the withdrawn demo.
Spec context: 7.3, 10.3, 12.2; this removal makes no new completion claim and
does not change the read-only specification. Check for surviving references,
collect the remaining A9 tests, and run available static/whitespace checks.

Preserve the A9.1 derivative/composition suite and A9.2 wheel/extension checks,
and all other files already staged in the maintainer-initiated main merge.
Incoming main c8adba5 supplies A8 fixtures, reference tests, docs and runner
changes. The generated bc-like candidate and both protected copies were
verified byte-identical during conflict resolution; its SHA-256 is
`4edbdee917ae335da11911b39ec60442e21998687d27c62494f54e7eb1d0e450`.
No incoming code, fixtures, protected tests/scripts, dependencies or spec
marker changes are part of this removal. The merge commit remains with the
maintainer; stage this cleanup with the existing resolved merge.

Use-case 1 and the README mini example will carry the replacement teaching
material. Their design and implementation are outside this removal request.
A9.4 CPU measurements and A9.d1 docs remain pending. Combined-tree A8/A9
validation has not been established by these cleanup changes.

Removal checks: no surviving code/docs references; 18 retained A9 tests
collected in 0.64s. Ruff reports 82 files already formatted / All checks
passed! Mypy reports no issues in 17 source files. Whitespace checks pass.
All 171 other index entries are unchanged. The staged net diff from incoming
main contains only PLAN.md/REVIEW.md and the five retained A9.1/A9.2 files;
the withdrawn example and its tests are absent. No full numerical gate ran.

---

# A9 — Transformations, installation and Slice A evidence (2026-10-26)

Status: **A9.1–A9.2 review passed; A9.3 withdrawn by the maintainer**.
Planning started
on the maintainer-created `A9` branch at `030314b`, with a clean working tree.
The scope note above records current status. A9.1 and A9.2 remain accepted.
The standalone A9.3 example is removed; use-case 1 and a mini README example
will supersede it. The remaining step descriptions retain their prior scope
and results. A9.4 and A9.d1 still need approval before implementation.
Each step ends in one commit, an overwritten REVIEW.md and a stop. Target at
most 300 changed code/test/docstring lines per step; split and replan before
exceeding that budget. Do not create or switch branches without instruction.

## Contract, scope and existing evidence

- Spec revision 37; `.spec-seen` records `be26e50`. Planning command
  `../scripts/spec-since.sh dev` exited 0: `spec unchanged since be26e50`.
  No spec, marker, dependency or public-library-API change is planned.
- Sections 7.2–7.3, 10.2–10.3, 12.1–12.3 govern this work. Section 10.5
  requires a separate closing docs step and rendered-page review.
- Own tests already cover built-in second-order finite differences for both
  likelihood input forms, analytical derivatives, parameter-issue isolation
  (including nested families), boundary gradients and grid transformations.
  Reuse that evidence; do not duplicate the earlier full matrices in A9.
- Read-only A9 acceptance tests cover analytical derivatives, custom-family
  autodiff defaults and reference substitution, compiled composition and
  compilation reuse, package metadata and wheel installation. They have been
  inspected, not executed in this planning turn.
- A9.1–A9.2 supply derivative/composition tests and the wheel runner.
  The standalone A9.3 example has been withdrawn; use-case 1 and a mini README
  example will cover that work later. The CPU baseline remains pending.
  A9 wheel tests require an absolute SURVJAX_WHEEL path, supplied by
  scripts/check_wheel.sh.
  Keep protected scripts/check.sh and all tests/spec content unchanged.
- Incoming main now supplies scripts/fixtures, tests/fixtures and
  tests/reference, including the bc-like synthetic data and the protected
  acceptance fixture. The old missing-fixture blocker is superseded by this
  integration. No R regeneration is needed; verify the combined gates before
  claiming their results.
- A9 excludes later likelihood modes, wrappers, RMST, metrics/concordance,
  sampling, inference libraries, release demos, publication and accelerator
  claims. Revision 37's first-release items have separate gates.

## A9.1 — Remaining derivative and compiled-composition evidence

One concept: verify the current public likelihood composes with JAX. Public
API touched: none. Files to create:
`tests/unit/test_likelihood_composition.py` and
`tests/unit/test_likelihood_custom_derivatives.py`. Reuse
`tests/impl_tolerances.py` without loosening its values; edit comments there
only if a new finite-difference case needs an explicit scale justification.
Spec: 4.2, 6.2, 7.2–7.3, 12.1 and compilation reuse in 12.3.

- For Exponential and Weibull, construct ordinary data from traced arrays,
  access and return data/parameter diagnostics, and evaluate the summed
  likelihood with first and second parameter derivatives inside `jit`.
  Cover flag and equivalent bounds construction; compare to prepared-data
  evaluation and explicit analytical derivatives on a small mixed batch.
  Expected numbers come from differentiated exact/right formulas, not the
  prepared implementation alone. Keep parameters finite and interior.
- Check forward/reverse derivatives and Hessians, `vmap` over independent
  parameter vectors or datasets, and `jit(vmap(...))`. Include diagnostic
  control flow (`lax.cond`) with an invalid replicate and verify valid
  replicates retain their analytical values and derivatives. No fit/solver
  implementation is necessary for this library-composition check.
- Count traces for changed parameters and observation values with fixed
  shape/dtype/weak typing/tree structure, and for equal-sized minibatches
  sliced from one parent. Synchronize calls and assert actual numerical
  outputs as well as reuse. Preserve inherited static metadata.
- Add one small registered custom family with only `log_survival` and
  `reference_time`, exercising generic density differentiation through the
  likelihood. Verify analytical gradient/Hessian and second-order
  `check_grads` in both modes on a smooth unit-scale interior case.
- Add a reference-substitution regression whose positive finite reference
  has a singular parameter derivative, so a leak into an inactive density
  branch would poison exact/right mixed-batch gradients and Hessians.
  Expected derivatives come from the active exponential formula.
- Existing violation-isolation suites remain the evidence for invalid
  parameters. Do not extend derivative promises to times, unsupported modes,
  invalid/infinite objectives or support-change boundaries. If a new test
  reveals a numerical defect, stop and plan a focused fix rather than
  silently expanding this verification step.

Checks after approval: focused new modules in separate `JAX_ENABLE_X64=1`
and `=0` processes; independent `test_a9_likelihood_derivatives.py` and
`test_a9_compiled_composition.py` with `SPEC_THROUGH=A9` in each mode;
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`;
`git diff --check`. The A7 gate is the available cumulative regression gate,
not an A9 completion claim. Record exact commands, counts, skips and failures.
Write REVIEW.md with reading order, invariants and requirement/test mapping,
update this plan's evidence, commit on `A9` and stop.

### A9.1 implementation evidence

Added two own test modules totaling 299 lines, 15 cases per dtype. Library
source, shared tolerances and protected files are unchanged. Final focused
runs: 15 passed in 22.61s (float64), 15 passed in 21.95s (float32).
Independent A9 derivative/composition modules: 40 passed in 37.28s (float64);
27 passed, 13 existing float64-only skips in 18.37s (float32).
CPU environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3. Ruff passes
(74 files already formatted / All checks passed!), mypy reports no issues in
17 source files. The initial focused failures came from negating NumPy
boolean indicators in the new expectations; cast them to float. No library
fix or tolerance change was needed. REVIEW.md records the requirement mapping
and commands. The cumulative command
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
exited 0: `check: all gates passed through A7`.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 789 passed, 7 skipped, 186.51s | 795 passed, 1 skipped, 173.74s |
| Source doctests | 4 passed, 1.01s | 4 passed, 0.95s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 106.77s | 2118 passed, 24 skipped, 218 deselected, 107.94s |

Full log: `/private/tmp/survjax-a91-gate.log`. No new skips or exclusions.
Only this result record and REVIEW.md changed after the gate; final Ruff and
whitespace checks follow. No R, wheel or docs build ran; no later A9 step has
begun. Commit this approved step on `A9` and stop for review.

## A9.2 — Installed wheel and out-of-tree extension

One concept: verify the distributed artifact independently of the source
checkout. Public API touched: none. Create `scripts/check_wheel.sh`,
`tests/install/test_wheel.py` and `tests/install/custom_family.py`.
Spec: 4.2, 10.1–10.3, 12.1–12.2. Target under 300 changed code/test lines.

- Provide a small runner that builds a wheel into a fresh temporary directory
  using uv and the locked backend, sets `SURVJAX_WHEEL` to its absolute path
  and delegates to the unchanged maintainer gate (default through A9).
  Support an explicit lower gate while A8 is absent; never suppress failures.
- Own install tests accept that artifact or build one once into pytest's
  temporary directory when run via the standard own-test runner. No test
  requires network access: build/install use offline cached locked tooling.
  A missing setup artifact/cache is an explicit failure, not a skip or an
  implicit download. No extra dependency or build hook is needed.
- Create a fresh environment outside the source tree with only the locked
  core runtime closure and the wheel, installed with uv from its cache.
  Export runtime requirements from the existing lock without dev/docs/demo
  groups; do not resolve different versions or use editable source installs.
- From an empty external working directory, remove source-path injection,
  assert the imported package lives in that environment, check exact Slice A
  exports, metadata, Apache license and `py.typed`, and evaluate a likelihood
  against hand-derived values. Confirm demo packages are absent and unused.
- Copy the small fully annotated custom family into that external directory;
  import, construct and evaluate it against analytical values and derivatives.
  Run the existing mypy against it using the installed environment for package
  discovery. This checks installed typing as well as runtime extension use.
- Use fixture/session lifetime to avoid rebuilding for every assertion. Keep
  temporary environments and archives outside tracked files. An sdist check
  is a pre-release requirement, not part of this Slice A wheel step.

Checks: own install module in both dtype processes, independent A9 package
and wheel modules with `SURVJAX_WHEEL` set in both processes, out-of-tree mypy,
and `scripts/check_wheel.sh --through A7` plus `git diff --check`. The same
wheel must pass all applicable checks; report Python/JAX/NumPy/backend
versions actually used. Keep the normal `scripts/run_tests.sh` working in
both dtypes and excluding `tests/spec`. Update PLAN.md/REVIEW.md, commit and
stop. A cumulative through-A9 run remains pending A8.

### A9.2 implementation evidence (2026-10-27)

Created the wheel runner, three own install tests and the annotated external
family (281 script/test/example lines). The runtime export uses pylock.toml
to preserve locked artifact identities and hashes and avoid needing cached
registry metadata. Installs use a fresh external environment, offline binary
sync, then the wheel with no dependency resolution. Strict external mypy and
parameter derivatives pass; no library or dependency change was needed.
Initial focused own runs after correcting the export format: 3 passed in
17.16s (float64), 3 passed in 21.98s (float32). Final metadata checks are
included in the cumulative gate now running.

The runner-built wheel is used by both cumulative dtype processes and the
independent A9 package/wheel modules: 16 passed in 2.74s in each mode.
Its WHEEL metadata records `Generator: uv 0.11.19`, `Root-Is-Purelib: true`,
`Tag: py3-none-any`. Python 3.12.9, JAX 0.11.2, NumPy 2.5.3; CPU only.
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check_wheel.sh --through A7`
exited 0: `check: all gates passed through A7`. Guard passed; Ruff reports
76 files already formatted / All checks passed! Mypy reports no issues in
17 source files. The external example passed strict mypy in both processes.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 792 passed, 7 skipped, 211.16s | 798 passed, 1 skipped, 192.31s |
| Source doctests | 4 passed, 0.89s | 4 passed, 0.82s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 99.80s | 2118 passed, 24 skipped, 218 deselected, 95.99s |

Full log: `/private/tmp/survjax-a92-gate.log`. The three new own install
cases run in each dtype, with no new skips or exclusions. The retained wheel
has SHA-256 `700418efd9bf12297ed2e0851d5c133767dc7abf575d146ed85b3d4da6f1df3f`;
REVIEW.md records its absolute path and focused/spec logs. Bash syntax and
whitespace checks passed. Initial requirements.txt setup attempts failed
because offline registry metadata was unavailable; pylock export fixed them.
An auxiliary verbose build with no task cache variable hit the sandbox's
restricted default cache; successful builds use the explicit permitted cache.
No spec deviation or dependency change was needed. No R, sdist, docs build
or later step ran. Only result records changed after the gate; final static
and whitespace checks follow. Commit A9.2 on `A9` and stop for review.

## A9.3 — Withdrawn standalone example (2026-10-29)

The maintainer requested removal of `examples/slice_a.py` and its dedicated
`tests/unit/test_slice_a_example.py`. They are excluded from the A9 merge.
The previous fixed-parameter demo and its three regression cases are removed;
A9.1 derivative coverage and A9.2 wheel/extension checks remain unchanged.
Use-case 1 and a mini README example will supersede this standalone demo.
No replacement example, fitting implementation or README edit belongs to
this removal step. Their definition and validation remain future work.

## A9.4 — Reproducible CPU baseline

One concept: measure current behavior, without optimizing it. Public API
touched: none. Create `scripts/benchmark_slice_a.py` and
`benchmarks/slice_a_cpu.json`. Spec: 7.2, 12.1–12.3. Target at most 300
changed script/test lines; the machine-readable measurements are evidence.

- Fixed-seed synthetic exact/right data, CPU only, separate dtype processes.
  Measure Exponential and Weibull, N=1,000/10,000/100,000, shared and
  per-observation parameters, with a compiled summed objective and parameter
  gradient. Compare prepared data against ordinary traced construction plus
  diagnostics and evaluation. Return diagnostics so they cannot be eliminated;
  record the prepared flags and conservative all-true traced metadata.
- Separate host generation, device placement and prepared construction costs,
  compilation/first-call costs, and warmed repeated execution. Synchronize all
  measured output leaves. State warmup/repetition counts and report a timing
  distribution, not a single asynchronous dispatch duration.
- Check values and gradients before recording timings against independent
  exact/right analytical formulas, using the existing dtype tolerances.
  Verify compilation reuse for changed dynamic values and same-shaped parent
  minibatches; reuse the A9.1 tests as regression evidence.
- Record commands, seed, shapes, parameter layout, versions, hardware/OS,
  actual CPU device, repetition count and logical input/output byte counts.
  Report peak memory with its method if measured, otherwise explicitly mark
  it unmeasured. Do not present byte counts as peak memory or claim speedups.
- Commit compact reproducible results for both modes; inspect scaling and
  preparation/construction overhead and record limitations in REVIEW.md.
  No speed threshold, benchmark dependency, optimizer run or timing-sensitive
  regular test. Numerical failures require investigation, not relaxed checks.

Checks: execute the full benchmark matrix in both dtype processes, inspect
all rows and correctness checks, run Ruff/mypy and the available cumulative
regression gate, then `git diff --check`. Update PLAN.md/REVIEW.md, commit and
stop. This provides CPU evidence only.

## A9 Docs to-do — A9.d1, separate approval after code merges

One topic: using and verifying the installed Slice A subset. Public API
touched: none. Create `docs/slice-a-verification.md` and add its navigation
entry in `docs/index.md`, at most 300 changed lines. Spec: 10.3, 10.5,
12.1–12.3. Describe compiled derivatives, the wheel runner, benchmark
reproduction and the evidence's limitations. Tie R-comparison claims to the
integrated A8 evidence and actual executed checks. Fitting/example content
will be supplied by use-case 1 and a mini README example, rather than the
withdrawn standalone A9.3 demo; link that material once it exists.

Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh` with
fresh execution in both modes; inspect rendered results, links and layout.
Audit existing executable pages and `docs/playground.ipynb` for the Slice A
gate; record the working JupyterLab command and any remaining docs topics as
separate steps rather than expanding this page's scope. Update REVIEW.md,
commit on the maintainer-selected branch and stop.

## Final evidence audit and planning checks

The last A9 review must list every Slice A gate item with its concrete test,
artifact or measurement and an honest pass/pending/fail status. Reuse earlier
review evidence where applicable, but run the final cumulative
`scripts/check_wheel.sh --through A9` against the combined tree, now that
A8's fixtures are present. This example-removal turn does not run or claim
that gate. Do not skip acceptance tests, manufacture fixtures or claim
Slice A complete. Use-case 1 and the mini README example, CPU measurements
and docs still need their own evidence and approved steps.

Planning checks actually run: branch/status/history and protected-path diff
inspection; read spec/goals, current plan/review/deviations, runner/build
configuration, relevant own tests and independent A9/A8 contracts; spec
alignment command above. No tests, builds, installs, benchmarks or R commands
ran during planning. No contract ambiguity requiring a new deviation found.
At the planning stop only PLAN.md had changed; `git diff --check` passed.
That stop ended when the maintainer approved A9.1; implementation evidence
for the approved step is recorded above.

---

# Incoming A8 history from main

Retained verbatim from main's plan additions. Its branch names, approvals,
check results and pending-handoff wording describe that work at the time;
the current integration note above records the fixture's arrival here.

# A8 integration audit and follow-up plan (2026-10-29)

The maintainer initiated merging main (be1a5b3) into a8-1-r-generator and
requested conflict resolution and an A8 relevance audit. Resolve only PLAN.md
and REVIEW.md; retain all incoming code, independent tests, scripts and marker
unchanged. Prior plans below are historical; this entry is current.

## Findings

- Incoming .spec-seen advances from d243a966 (revision 34) to be26e50
  (revision 37). Spec 6.1, 7.1, 10.2 and 12.1 now require method-only
  likelihood (revision 35). main implements the removal and migrates its
  tests/docs. A8's own reference tests were not in that migration.
- tests/spec/test_a8_bc_fixtures.py now uses dist.log_likelihood in prepared,
  raw, jit/design-matrix and total comparisons. The redundant separate method
  test is removed. Its fixture schema, parameters and numerical targets stay
  unchanged. tests/spec/ remains read-only.
- Spec 3.2 size/truth-value interface is implemented and independently tested
  by incoming main; A8 constructs data explicitly and needs no size migration.
- Revision 37 PH/RMST and release-demo changes concern later work, not A8's
  fixed-vector exponential/Weibull fixtures. No fixture regeneration follows.
- main adds pytest-xdist test tooling and defaults gates to six workers;
  retain those incoming changes. They do not alter numerical tolerances.
- Protected tests/spec/fixtures/bc_synthetic_weibull.json remains absent;
  durable maintainer handoff and sync still precede the through-A8 gate.

## A8 compatibility code step — approved

Approved by "ok do it" after the integration audit. Merge completed as
f4e0eae before this separate compatibility step.
Completed: both focused dtype suites pass (111 cases each); all 48 method
row/grid corruption controls are detected per dtype. The incoming six-worker
through-A7 gate exits 0; REVIEW.md records exact counts and evidence limits.
Commit this compatibility step and stop for review.

Files: tests/reference/test_exponential_r.py and
tests/reference/test_weibull_bc_like_r.py, plus PLAN.md/REVIEW.md.
Public API: none added; align calls with spec 6.1/7.1/10.2/12.1.
Remove redundant function-entry tests/case axes, retain method raw/prepared
comparisons, and migrate the exponential supplemental total to the method.
Keep every vector/source, ordered-row expectation, dtype/shape assertion,
eager/jit/vmap path and central tolerance unchanged. Do not restore an alias,
edit protected tests or regenerate payloads. Target under 300 changed lines.

Run both reference modules in each dtype and repeat focused row-order/grid
sensitivity controls through the method. Then run scripts/check.sh --through A7
with the incoming runner, final Ruff/mypy/diff checks, record exact outcomes,
commit one compatibility step and stop. The earlier audit failure from the
removed API is corrected by this step; full through-A8 remains pending.

## Subsequent docs correction — approved

Approved by "ok do the docs correction" after d5c0a3c.
Completed: both fresh dtype docs builds exit 0; HTML checks confirm method-only
wording, all 48 local links/anchors and byte-identical recipe downloads per
mode. REVIEW.md records evidence and verified fixture-handoff details.
Commit this docs correction and stop for review.

Change only the obsolete function/method claim in docs/builtin-families.md to
method raw/prepared coverage. Audit A8 fixture READMEs for the same claim;
no such explicit two-entry-point claim was found during this audit. Build
both dtype docs, check links/rendered wording, update review, commit and stop.
No broader docs/size-interface or later-slice work in this A8 correction.

## Closure still required

Maintainer supplies the committed acceptance-schema JSON to the authoritative
tester repo and syncs it. Verify byte identity/checksum, then run the unchanged
independent A8 module in both dtypes and full through-A8 gate. Neither fixture
copying into protected paths nor a full A8 pass is implied by this merge.

---

# Remaining Slice A8 work — plan for approval (2026-10-23)

Status: main through A7 is integrated at incoming commit 3389ee0. The
maintainer accepted A8.4a.1 and approved A8.4a.2. The bc group/grid/export
checks are accepted after review. A8.4b exponential reference comparisons
are implemented, validated through A7 and ready for review. Protected tester-repo
fixture delivery remains pending. A8.5 is accepted. A8.d1 is implemented and ready for review.
Historical plans below retain their original status wording.

Work on the existing maintainer-created `a8-1-r-generator` branch, in this
checkout only. Keep one concept and roughly 300 changed code/test lines per
step, one commit, REVIEW.md and a review stop. Replan before exceeding that
budget; preserve the requested readable R structure and spacing. The existing
343-line bc generator already exceeds the file-size guideline: if the export
addition makes it unwieldy, split that concrete export responsibility into a
small R file in the same step, without creating a general serialization layer.

## Contract and boundaries

Specification revision 34, sections 3.3, 4.6–4.7, 5, 6.1–6.3, 10.4–10.5,
11.1 and 12.1–12.2. No public API, library algorithm, dependency or tolerance
changes are planned. A7 owns tail/boundary/debug-nans evidence; A9 owns the
broader derivative, packaging and performance completion work. Imported A9/A10
acceptance tests do not expand A8's scope; no concordance work belongs here.
Spec and independent tests stay read-only. Unexpected numerical failures go
back to the responsible implementation; record genuine contract gaps in
DEVIATIONS.md and ask rather than changing tests or loosening tolerances.

The existing fixtures remain the reviewed source of synthetic evidence:
- exponential: 16 rows, three fixed rates, two R paths;
- bc_like: 686 rows, Good/Medium/Poor sizes 229/229/228, seed 8201, three
  fixed vectors, observed-time predictions and 141-by-3 survival grids.

Never load patient data or copy real-cohort likelihood totals. Keep all R
exports at 17 significant digits. The chosen validation policy remains:
R checks exact exported text; Python separately verifies numeric round trips
against original R hexadecimal doubles. Python never generates expected values.
The two R package paths are not fully independent numerical oracles.

## A8.3a — Export the independent acceptance schema

Review fix-up (2026-10-25), authorized by "check the review, address the nits":
read ../review/slices/A8.3a.md. N-1 checks the assembled export against the
original R observations and per-source references. N-2 moves grid identity
into that output validation and distinguishes preservation from the upstream
independent orientation check. N-3 remains an A8.3b handoff: restore a fully
passing regeneration recipe, including comparison of the new committed JSON.
N-4 retains the shared-oracle limitation; exact source-field checks protect
mapping labels without claiming independent numerical evidence.
Files: bc_spec_export.R, bc_like README, PLAN.md and REVIEW.md. No payload,
manifest, generator main-script, library, dependency or protected-test edits.
Re-run pinned R two-run/hex validation, E2/E4 mutations in check-only/export,
final Ruff/diff checks and the through-A7 gate. All pass; the gate exits 0
and REVIEW.md records exact counts and mutation results. One fix-up commit,
then stop.

Execution (2026-10-25): the approved export mapping is isolated in
scripts/fixtures/bc_spec_export.R, sourced by the existing generator. Its
checks also run under check-only; full generation writes the additional JSON
in the chosen output directory. The old JSON header now records the exporter
checksum. README documents the interim provenance difference until A8.3b.
Scratch two-run, exact R-double and mutation checks pass. The final through-A7
gate exits 0; counts and limits are recorded in REVIEW.md. No payload/manifests,
independent tests, public API, dependencies or tolerances are changed.
Code/README diff stays below 300 changed lines; do not start A8.3b.

Files: `scripts/fixtures/generate_bc_like.R`, optionally one small R export
file in `scripts/fixtures/` if needed for readability, and
`tests/fixtures/bc_like/README.md`. No committed payload changes in this step.

Extend the R generator to write an additional
`bc_synthetic_weibull.json` inside its chosen output directory, using the
already evaluated R objects. Retain data.csv and expected.json and their
reviewed numerical contents. Follow `tests/spec/fixtures/README.md` exactly:
- embed time/event/group arrays; metadata reports actual seed 8201, generator,
  pinned R/packages (normalize survival's displayed 3.7.0 to 3.7-0 as needed);
- map unit_shape to fixed_a, alternative to fixed_b, fitted to r_optimum;
- map shape/scale_Good/beta_Medium/beta_Poor to the defined parameter names;
- map each source's log_likelihood and survival to loglik and survival;
- export stored grid times and flexsurv curves, time first, Good/Medium/Poor.

Do not reparse decimal CSV/JSON inside R to build the additional export:
that would reintroduce the pinned R parser's one-ULP issue. Do not write under
`tests/spec/`. Default output stays tests/fixtures/bc_like; document that
no-argument generation overwrites generated files, including the new JSON.
Keep check-only free of writes and reject unknown flags before writing.

Checks with code: exact field/case mappings, row and group order, parameter
values, both sources, finite lengths/shapes, grid labels/orientation, exported
text and precision. Exercise swapped cases/columns and shortened decimals in
scratch copies and require detection. Confirm check-only does not write.
Use pinned R 4.4.2, flexsurv 2.3.2, survival 3.7-0 only; if unavailable, deliver
the generator and stop for the maintainer's run. Record all actual checks.
Spec: 10.4, 11.1, 12.1. Docs to-do: export path and evidence limits in A8.d1.

## A8.3b — Commit regenerated outputs and hand off the spec fixture

Execution (2026-10-25): authorized by "all good; next bit of A8" after the
A8.3a review fix-up ef4a038. Regenerate using that unchanged generator/exporter,
validate two-run byte identity, preservation of reviewed values and exact
R-double read-back, then import only the refreshed expected.json and additional
JSON. Update the three-entry manifest and passing README recipe; CSV remains
byte-identical. Validate the candidate against an untouched temporary copy of
the A8 acceptance module in both dtypes, without writing in the real tests/spec.
Finish with REVIEW.md, one data-only commit and a maintainer handoff; do not
start A8.4. No generator, library, dependency, tolerance or protected-path edits.

Files: bc_like data.csv/expected.json, the additional acceptance-schema JSON,
SHA256SUMS and README under `tests/fixtures/bc_like/`. Change only generated
provenance where needed in existing outputs; CSV and all prior numeric values
must remain identical. No independent test or spec-file edits.

Run the approved generator twice in the pinned R environment into temporary
directories. Compare both runs, compare existing observations/reference numbers
with committed A8.3, and independently check exact Python read-back against R
hexadecimal values. Check every additional-schema field against the reviewed
R output, including all three cases/sources and every grid cell. Update the
manifest to cover all three payloads, then verify all entries. Keep regeneration
commands comparing fresh outputs both to each other and to committed files.
A8.3a review N-3 is closed only when the full documented recipe exits 0,
including cmp of the additional JSON against its committed candidate; remove
the interim provenance-drift notice then.
Completed local validation: fresh two-run generation and exact R-double
read-back pass; all existing observations and numbers are preserved. The
unchanged A8 acceptance module passes 44 tests per dtype in a temporary copy
with the candidate installed. The full README recipe now exits 0, with all
six cmp checks and all three manifest entries passing; N-3 is closed.
The real tests/spec fixture remains absent until the maintainer handoff.

This generated-data step is separate from the generator code; no new large
CSV diff is expected. Ask about a size exception only if a new large diff occurs.

Protected fixture handoff: supply the verified additional JSON and checksum
to the maintainer for inclusion at
`tests/spec/fixtures/bc_synthetic_weibull.json` in the authoritative tester
repo, followed by the usual spectests-* sync and maintainer merge. The sync
replaces tests/spec wholesale, so a local-only copy there is not durable.
No implementer write to that path, no test-loader adaptation and no guard bypass.
After handoff, verify the synced file is byte-identical to the generated
candidate and run the focused A8 acceptance module in both dtypes. If delivery
is pending, report it explicitly; do not claim a through-A8 pass.
Spec: 10.4, 11.1. Docs to-do: durable handoff/regeneration provenance.

## A8.4a — Shared fixture loader and integrity checks

A8.4a.2 execution (2026-10-26), authorized by "good on A8.4a.1; get to the
next bit": finish bc group/block counts, time-first grid validation and exact
acceptance-export consistency in tests/conftest.py. Add the bc corruption
suite as tests/unit/test_bc_reference_fixtures.py so the existing 204-line
common test module stays focused and below the file-size guideline. Validate
rehashed bad groups/metadata, grid shape/order/finiteness, acceptance metadata,
case/parameter/source/row/grid mismatches, and stored float64 precision. All
hashes still pass before parsing. No payload, generator, library, tolerance,
dependency, docs or protected-path changes. Keep the step below 300 code/test
lines, run focused tests in both modes and the through-A7 gate, commit and stop.
Final evidence: 58 focused tests pass per dtype; all 15 new guard-removal
mutations are rejected per dtype; the full through-A7 gate exits 0. REVIEW.md
records exact counts and limitations.
A8.4b distribution comparisons are not authorized by this step.

A8.4a.1 review fix-up (2026-10-25), explicitly authorized by "check the
review and fix": read ../review/slices/A8.4a.1.md. SF-1 adds rehashed bad
CSV/JSON tests that kill L6–L11 (nonfinite arrays, event counts, CSV MD5,
positive times/parameters, and log_survival validation). N-1 gives each damage
its exact expected rejection message. N-2's KeyError behavior is retained as
accepted test infrastructure; no loader behavior change is needed. Files:
unit test module and PLAN.md/REVIEW.md only. Demonstrate each formerly
surviving mutation is rejected in both dtypes, run the focused suite and full
through-A7 gate, then commit one fix-up and stop. Validation is complete:
28 focused tests pass per dtype, L6–L11 are rejected per dtype, and the full
gate exits 0. REVIEW.md records counts and mutation evidence. That fix-up
did not start A8.4a.2.

Execution (2026-10-25), approved by "go on with next bit; plan approved":
implement in tests/conftest.py and tests/unit/test_reference_fixtures.py only,
plus PLAN.md/REVIEW.md. Expose r_fixture(name, root=...) as a pytest callback;
return the checked header/cases/grid plus host data arrays. Float arrays remain
NumPy float64 regardless of JAX mode. Verify all known manifest payloads before
any parsing; validate common schema/provenance and the bc acceptance mapping.
Tests use temporary copies for damage/rehash cases; no R, dependencies, payload,
public API, numerical tolerance or protected-test edits. Keep code/tests under
300 changed lines. The full loader/negative tests plus bc consistency exceed
that budget, so use the permitted split: A8.4a.1 implements all-payload hash
verification, common provenance/rows/arrays and corruption tests; A8.4a.2 adds
bc group/grid and acceptance-schema mapping checks with their negative tests.
Stop for review after each commit; neither part alone completes A8.4a.
Focused tests now pass 28 per dtype after the review fix-up; the full
through-A7 gate exits 0. REVIEW.md
records exact counts and the precision-control mutation. Commit and stop.
The A8.4a.1 history above is retained; A8.4a.2 is now implemented as recorded
in its execution entry. Distribution comparisons still wait for A8.4b/A8.5.

Files: `tests/conftest.py`, `tests/unit/test_reference_fixtures.py`.
No public API changes. All implementation reference tests use one loader,
exposed as a pytest fixture; permit a temporary fixture root for loader tests.
Use stdlib CSV, JSON and hashlib, existing NumPy, and plain return values.

Verify manifest entries and raw bytes before parsing any payload. Require the
known payloads for each set (two exponential files, three bc_like files);
reject missing/malformed/duplicate entries and hash mismatches. Check schema,
pinned provenance, row IDs, finite positive times, event coding, group order,
case labels and result dimensions. For bc_like, require the extra export to
match the original row values, vectors, per-source arrays and grid. Preserve
stored numeric precision on the host; tests cast inputs to the active JAX
dtype while expected R values remain float64. Missing data fails, never skips.

Tests: both committed sets load; missing/malformed manifests fail; independent
CSV and JSON edits in temporary copies fail before parsing (include the extra
JSON); a correctly rehashed malformed schema fails validation; changed row
alignment is detected. No general fixture framework or new tolerance values.
Spec: 3.3, 10.4, 12.1. Keep loader plus tests within the step budget; split
additional schema assertions if necessary rather than compressing them.

## A8.4b — Exponential reference comparisons

Execution (2026-10-26), approved by "review passed; implement next subslice":
create tests/reference/test_exponential_r.py using r_fixture only. Explicitly
cast CSV inputs and rates to the active JAX dtype while preserving R outputs
as NumPy float64. For all three rates and both sources compare every row for
function/method and raw/prepared likelihood, direct survival/log survival,
and eager/jit/row-vmap execution; check valid diagnostics and scalar R totals
separately. Update only the exponential fixture README's obsolete evidence
and loader wording, plus PLAN.md/REVIEW.md. No payload, tolerance, library,
dependency, docs/ or protected-path edits. Stay below 300 changed code/test
lines; run focused tests per dtype and the through-A7 gate, commit and stop.
Final evidence: 45 focused tests pass per dtype; a row-permutation control
fails all 24 likelihood row cases while scalar totals still pass; the full
through-A7 gate exits 0. REVIEW.md records exact counts and limits.
A8.5 remains separate; protected acceptance-fixture handoff is still pending.

File: `tests/reference/test_exponential_r.py`. No public API changes.
For all rates and both R sources, compare every row's likelihood, survival
and log survival using the shared loader and central tolerances. Cover both
survjax.log_likelihood and dist.log_likelihood, raw/prepared inputs, clear data
and parameter diagnostics, eager/jit/vmap agreement. A summed likelihood is
secondary evidence and never replaces per-row assertions. Keep tests small
and name the R origin explicitly. Spec: 4.6–4.7, 6.1–6.3, 12.1.
Docs to-do: describe which exponential quantities are verified, in A8.d1.

## A8.5 — Weibull reference comparisons and the A8 code gate

Execution (2026-10-27), authorized by "implement A8.5, plan is approved":
implement the approved reference tests on a8-1-r-generator. Protected fixture
delivery remains pending; run the through-A7 gate and report that limitation.
No loader, library, payload, docs/, dependency or tolerance changes.
Completed: 102 focused tests pass per dtype; all 36 row/grid sensitivity
controls fail as intended per dtype. The through-A7 gate exits 0; exact counts
and evidence limits are recorded in REVIEW.md. Commit and stop for review.

File: `tests/reference/test_weibull_bc_like_r.py`. No public API changes.
For all vectors and both sources compare per-row log likelihood, survival,
log survival and the (141, 3) grid. Derive row scales from recorded group
coefficients, exercise common shape/batched scale, natural/from_log forms,
raw/prepared likelihood, and focused eager/jit/vmap comparisons to stored R
values. Use the stored grid.time values; rebuilding R seq can change one ULP.
Use the active dtype and central tolerances without adjustments.

After the protected fixture handoff is integrated, run the full through-A8
gate, including unchanged independent acceptance tests. Record coverage and
remaining limits in REVIEW.md; an A8 pass is not completion of all Slice A.
Spec: 3.3, 4.6–4.7, 5, 6.1–6.3, 11.1, 12.1–12.2.
Docs to-do: explain synthetic scope, grids and package-path limitations.

## A8.d1 review fix-up — approved (2026-10-29)

Read ../review/slices/A8.d1.md for 726896d. Both should-fix findings are
confirmed. Approved by "yes do it"; implement only S-1 and S-2.
Completed: both fresh dtype docs builds exit 0. HTML checks confirm the two
paragraph fixes, cross-page anchor and all 85 local links/anchors per mode.
REVIEW.md records exact results and the remaining visual-inspection limit.

- S-1: docs/builtin-families.md: replace the final workflow-status paragraph
  with one sentence identifying the acceptance-schema copy and saying ordinary
  regeneration never writes to tests/spec/. Keep pending status in records.
- S-2: docs/survival-data.md: replace the blanket no-R-agreement statement
  with the verified exact/right-without-truncation scope and link to
  builtin-families.md#comparisons-with-r. The table still compares syntax.
- PLAN.md and REVIEW.md: record scope, evidence and remaining inspection limits.
  Leave optional N-1/N-2 edits out of this focused should-fix commit.

Public API/tests: none changed or added. Spec: 10.4–10.5, 11.1, 12.1.
Build fresh docs in both dtype processes with scripts/build_docs.sh; check
both changed paragraphs and cross-page anchors in HTML, inspect layout where
browser access permits, then run Ruff and git diff --check. No R regeneration
or full code gate for two prose edits. Commit one fix-up and stop for review.

## A8.d1 — R reference evidence and regeneration (separate docs step)

Execution (2026-10-28), authorized by "reviewer is ok; on next subslice,
plan is approved": A8.5 is accepted. Update builtin-families.md with the
committed reference evidence and regeneration workflow; correct the two stale
likelihood-support statements found while auditing distributions.md. api.md
needs no change. No new executable examples, source or fixture changes.
Build both dtype renders, inspect the new section and links, record results,
commit this documentation topic and stop.
Completed: both fresh dtype docs builds pass; HTML checks pass for all 48
local links/anchors, two new tables and byte-identical recipe downloads per
mode. REVIEW.md records exact results and the incomplete browser visual
inspection caused by concurrent Safari use. No through-A8 completion claim.

After reviewed A8 code is accepted, edit `docs/builtin-families.md` for this one
topic. Audit `docs/distributions.md` and `docs/api.md`; change only obsolete
verification claims, splitting another topic if needed. Explain synthetic
fixtures, fixed parameters, pinned versions, checksum checks, safe regeneration,
acceptance-schema handoff and dtype tolerances. Explain shared R package routes
and that survreg grid predictions are direct psurvreg evaluations, checked
against flexsurv; singleton likelihood checks cover observations, not grid cells.
Use committed evidence only. Build and inspect both dtype documentation renders
with scripts/build_docs.sh. At most 300 changed lines, REVIEW.md, commit, stop.
Spec: 10.4–10.5, 11.1. No fitting demo or later-slice claims.

## Validation and approval stops

For code steps: focused checks and mutation checks appropriate to the change,
Ruff formatting/lint, mypy src, own suites in separate x64/x32 processes and
the maintainer gate through A7 while A8 delivery is incomplete; through A8
once its fixture is available. Use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.
Normal tests must require no R, network or external data. Each REVIEW.md names
actual commands/results, reading order, invariants, spec-to-test mapping and
limits. Do not start the next step without approval.

A8.3a is accepted. A8.3b is locally validated and ready for its commit and
review, with protected fixture delivery pending. Stop here; A8.4a and later
steps retain their separate approval and review stops.

---

# Historical A8 preparation and integration plans

These are retained for provenance, superseded by the active plan above.

# A8 R fixtures and reference tests

Status: main through `2c030c5`, including the independent A7/A8 tests and
revision-34 spec marker, is integrated into this checkout (2026-10-23).
A8.1/A8.2 are accepted; A8.3 fixtures were generated and validated on
2026-10-21, with review nits addressed in `15b0a16`. The generated-data size
exception was explicitly approved. A8 remains on `a8-1-r-generator`;
A8.4/A8.5 are planned, not implemented. REVIEW.md records this integration;
prior generation and nit-fix evidence remains in its Git history.

The maintainer created this isolated checkout at
`/Users/wamster3/dev/survjax/dev-a8`, branch `a8-1-r-generator`, from main
at `9623594`. Implement the authorized step only, write REVIEW.md, commit
and stop for review; later steps keep their separate approval stops. Paths
below are relative to this checkout. Historical plans below the A8 section
are background only.

A6 is merged into main at `d4d4752` and is included by this integration. Stay
in this A8 checkout; do not change the main checkout's branch or files. The
incoming `.spec-seen` is retained unchanged. When checking spec alignment,
the workspace helper's target is `dev-a8`, not `dev`
(`../scripts/spec-since.sh dev-a8`).

A8 supplies committed, synthetic R references for exponential and Weibull
exact/right-censored likelihoods without truncation. The required larger
fixture is a 686-row synthetic `bc`-like dataset, evaluated at the three
parameter vectors in specification 11.1. Current contract: specification revision 34,
sections 4.6, 6.3, 10.3–10.5, 11.1 and 12.1–12.2.

## Coordination with A6 and A7

Acceptance-test integration (2026-10-23), authorized by the maintainer's
request to include the new main commits: merge `5d105ab` and `2c030c5`,
retaining the incoming spec marker and all independent tests unchanged.
Update only this plan and REVIEW.md; run the gate through A6, and probe the
A8 fixture test separately without claiming A7/A8 completion.

The imported `tests/spec/fixtures/README.md` defines a different fixture
schema/path: `tests/spec/fixtures/bc_synthetic_weibull.json`, with vectors
`fixed_a`, `fixed_b`, `r_optimum`, embedded data, per-source `loglik` and
`survival`, and flexsurv grid curves. The existing A8.3 files remain valid R
references but do not supply that file. Before claiming A8 completion, plan
a separate generator/export adaptation to this schema, with byte/numeric
validation against the existing R outputs. No values may be synthesized by
Python or copied by hand. The destination is under read-only `tests/spec/`;
its delivery must be coordinated with the maintainer under the protected-path
rules rather than silently writing there. Do not edit the independent loader
or assertions to accept our current schema. This integration does not
authorize that generator change or the A8.4/A8.5 implementation.

Integration (2026-10-23), authorized by the maintainer's request to resolve
the in-progress merge: retain the A8 plan as active and the incoming A6 plan
as historical context; overwrite REVIEW.md with integration evidence. Only
PLAN.md and REVIEW.md conflict. Keep all cleanly merged files unchanged,
including the incoming spec marker and independent tests. Run the gate through
A6 and verify both fixture manifests before completing the merge. No A8.4
implementation is authorized by this conflict-resolution step. A8.4's A6 API
prerequisite is now available; A7 and later A8 steps remain separate work.

- The maintainer has created this separate branch and worktree for A8
  preparation. Do not switch the A6 checkout, edit its PLAN.md or REVIEW.md,
  update its spec marker, or change its diagnostic decisions.
- Generator/schema preparation can proceed alongside A6 using the existing
  A5 distributions and normative parameter mappings. This is the parallel
  preparation proposed here; normal integration still follows A6 then A7.
- Likelihood reference tests require merged A6, including the final public
  `survjax.log_likelihood` and `dist.log_likelihood` contract. Diagnostics come
  from `data.issues` and `dist.parameter_issues()`; there is no likelihood
  `_with_info` variant (revision 29, unchanged by revision 30). A8 calls that API;
  numerical defects go back to the responsible implementation step.
- A7 owns mpmath, delicate tails, boundary derivatives and debug-nans evidence.
  A8 owns agreement with committed R values. Coordinate any shared edit to
  `tests/conftest.py` or `tests/impl_tolerances.py` during integration.
- The maintainer integrates approved preparation into the normal A8 branches
  after A6/A7. No A8 completion claim before the through-A8 gate passes.
  No new agent is needed for this planning request.

## Fixture contract

Each fixture directory contains `data.csv`, `expected.json` and a short README;
its generator lives in `scripts/fixtures/`. Keep the schema small and explicit:

- CSV: stable row ID, time in years, event (1 exact, 0 right), and group where
  applicable. Synthetic observations only, with no truncation or patient rows.
- JSON header: schema version, seed and RNG settings, R/package versions,
  generating script, platform, units, event coding and parameter convention.
  Record a checksum of the CSV and generator for provenance.
- JSON values: explicitly named parameter cases, row-aligned log likelihoods,
  log survival and survival values; a time-first survival grid where relevant.
  Keep the flexsurv and survreg references separately identifiable, including
  the package evaluation paths used. They are not fully independent oracles.
- Export sufficient digits to preserve R doubles. Avoid JSON NaN/Infinity by
  choosing finite reference cases; numerical extremes remain A7's remit.
- Regeneration uses explicit seed, row/group order and fixed parameters,
  without optimizer-dependent expected values. Normal tests only read files.
  Use base R for the fixed-schema export; no new dependency is proposed.

The bc-like dataset uses 686 simulated observations in Good/Medium/Poor groups
of 229/229/228, generated from the recorded fitted Weibull model in 11.1.
Simulate independent censoring with a fixed, documented rule that produces a
similar censoring pattern; report realized counts rather than claiming exact
agreement with the real cohort. Never load `flexsurv::bc` to construct it.

Evaluate the recorded fitted vector and the fixed vectors
`(k, b_Good, beta_Medium, beta_Poor) = (1, 10, -0.5, -1)` and
`(1.5, 12, -0.6, -1.2)`. Store explicit natural parameters and AFT equivalents:
`b_i = b_Good * exp(beta_group)`, `mu_i = log(b_i)`, `sigma = 1/k`.
For the fitted vector use the constants recorded in 11.1, without refitting.
Its synthetic-data totals must be newly generated, never copied from the
real-data audit totals. Use the specified 0–7 year grid in steps of 0.05,
shape `(141, 3)`, and include per-observation survival at observed times.

## A8.1 Small exponential generator

Review corrections (2026-10-20), authorized by the request to address
`../review/slices/A8.1.md`: SF-1 adds exact CSV text read-back, per-number JSON
format checks, case-label/rate checks and written-JSON verification. N-1 rejects
unknown flags; N-3 uses one generating rate. Keep N-2/N-4 as reviewed. Files:
the generator, its README, PLAN.md and REVIEW.md; no public API changes.
The maintainer chose R-only exact text checks after the pinned R decimal
reader lost one ULP on correct 17-digit output. Verify exact numeric round
trips separately with Python against the original R hexadecimal values.
Exercise exports and M6/M7/M8 mutations in temporary directories, check the
CLI typo fails before writing, and run the through-A5 gate (spec 4.6, 10.4,
12.1). S-1 updates only the future A8 plan; the inherited `.spec-seen` still
describes the implementation baseline pending A6 integration. One fix-up
commit, then stop; do not start A8.2 or commit A8.3 fixture payloads.

Review follow-up: the maintainer requested clearer R structure and spacing.
Expand calls/checks, separate the script's phases and make the output schema
readable as R lists. Preserve parameters, checks and export contents; rerun
check-only validation and record the result in REVIEW.md. One fix-up commit.

Files: `scripts/fixtures/generate_exponential.R` and
`tests/fixtures/exponential/README.md`. No library/public API changes.

Create a compact seeded exact/right dataset and several fixed rates. Evaluate
per-row likelihood and survival using the pinned R reference packages, with
the survreg log-mean to rate conversion explicit. Establish the fixed-parameter
package evaluation paths from their installed documentation while coding;
do not substitute a fitted optimum or label handwritten formulas as package
outputs. If no suitable path exists, report the gap before finalizing schema.

Generator checks accompany the code: exact versions, positive finite times,
both event kinds, valid row alignment, expected output shapes, finite interior
values, and agreement of per-row sums with the package total at fixed
parameters. Compare the two reference paths before emitting outputs. Record
commands/results in REVIEW.md and stop. Target at most 300 changed lines.

Implemented with 16 seeded rows and rates 0.1, 0.35 and 1.2. The generator's
`--check-only` mode exercises both pinned package paths and assertions without
writing fixtures. Default or explicit-destination CSV/JSON export is handed
to the maintainer for A8.3, including the two-run comparison. Docs to-do:
explain the eventual comparison evidence in A8.d1 after fixture tests pass.

## A8.2 Synthetic bc-like generator

Review fix-up (2026-10-20), authorized by the request to address
`../review/slices/A8.2.md`: SF-1 checks every grid column against a separate
prediction for its exported group label; reproduce B4 in check-only/export
modes. N-1 uses character group labels for named-vector indexing. N-2 expands
calls/checks and separates phases; keep the fix-up small without compressing
it to preserve the initial step's cumulative line count. N-3 documents using
stored grid times in A8.5; retain the N-4 grid-path limitation for A8.d1.
Files: generator, fixture README, PLAN.md and REVIEW.md. No public API or
fixture payload changes. Check unchanged export values, numeric read-back,
mutants and the through-A5 gate (spec 4.6, 10.4, 11.1, 12.1). One commit,
then stop; no merge or A8.3 generation/acceptance.

Implementation details: seed 8201, groups in Good/Medium/Poor blocks and
independent Uniform(0.5, 7.5) year censoring. Use fixed flexsurv Weibull AFT
coefficients, and survreg log-scale locations supplied through offsets with
fixed sigma and zero iterations. Check singleton likelihoods, row alignment,
all parameter conversions, case labels and both observed/grid predictions.
Preserve the A8.1 R-only export text checks; verify exact numeric read-back
with Python in scratch validation. Run mutation checks and through-A5 gate.
No library/public API, dependency, spec-marker, narrative-doc or A8.3 payload
changes. Docs to-do: describe this fixture evidence in A8.d1 after integration.

Files: `scripts/fixtures/generate_bc_like.R` and
`tests/fixtures/bc_like/README.md`. No library/public API changes.

Implement the larger fixture contract above, including both R reference paths,
all three parameter vectors and the survival grid. Reuse the simple file
format, without creating a general fixture framework. Include generator
checks for group sizes, event coding, parameter conversion, row alignment,
grid orientation, reference agreement and per-row/total agreement.
Target at most 300 changed lines; stop with REVIEW.md before the next step.

## A8.3 Generate and review the committed R outputs

Review nits (2026-10-22), authorized in this conversation: N-1 adds committed
payload comparisons and a no-argument overwrite warning to both READMEs.
N-2 adds a two-entry SHA256SUMS manifest per fixture, a documented verification
command and an A8.4 loader requirement. Files: both fixture READMEs/manifests,
PLAN.md and REVIEW.md; no payload, generator or public API changes. Validate
all hashes, committed-file comparisons and independent CSV/JSON corruption
checks in temporary copies; run git diff --check. One fix-up commit, then stop.

Current execution: the maintainer requested this step in this conversation.
Run the reviewed, unchanged generators in the pinned R environment twice per
fixture into temporary destinations. Validate byte identity, schema, row/group
counts, finite values, case labels, checksums and exact R-to-Python numeric
read-back. Import verified candidates; update both fixture READMEs and REVIEW.md.
No public API, generator, Python/test, dependency, spec-marker or narrative-doc
changes. Run the through-A5 gate; leave A6/A7 integration and A8.4/A8.5 untouched.
The generated-data diff exceeds 300 lines; prepare the concrete payloads for
the explicit exception below before committing. One data-only step, then stop.

Files: `data.csv` and `expected.json` in both fixture directories; update their
READMEs with exact regeneration commands and provenance.

The workflow assigns R execution to the maintainer. Hand over the generators
with `Rscript --vanilla scripts/fixtures/generate_exponential.R` and
`Rscript --vanilla scripts/fixtures/generate_bc_like.R`. Require successful
package loading and generator checks in the pinned environment. Generate
twice into separate temporary destinations and compare committed payloads;
exclude volatile timestamps from deterministic payloads. Inspect counts,
checksums, finite values and parameter labels before importing.

This data-only step needs an explicit exception to the approximate 300-line
review target: the required 686-row CSV alone exceeds it. Keep generated
outputs out of the code diffs and review them through provenance, regeneration
and structural checks. If that exception is declined, agree a partitioned
fixture layout before generation. Do not silently exceed the code-step budget.

The reviewed generators ran successfully for A8.3 outside the sandbox with
R 4.4.2, flexsurv 2.3.2 and survival 3.7.0 (release 3.7-0). Both fixture sets
were generated twice and byte-compared; exact numeric read-back was checked
against original R hexadecimal values. Four payloads are 706 generated lines
and 341,548 bytes. The data-only size exception was approved in the final commit request;
source code and handwritten verification values were not substituted.

## A8.4 Fixture loader and exponential reference tests

Requires A6's completed API and genuine committed outputs.
Files: extend `tests/conftest.py`; consume the fixture SHA256SUMS manifests; add
`tests/reference/test_exponential_r.py`. No new public exports.

Add one shared loader exposed through a pytest fixture. It reads CSV/JSON with
the standard library, checks schema, provenance and row alignment, and returns
host arrays in their stored precision. Tests convert inputs to the active JAX
dtype; expected values remain the R doubles. Missing or malformed fixtures
fail explicitly rather than skip. All reference tests use this loader.

Before parsing, use the standard library's hashlib to verify both `data.csv`
and `expected.json` against each fixture's SHA256SUMS manifest. Require both
entries; missing or malformed manifests and mismatched payloads fail explicitly.
Add loader tests that independently alter CSV and JSON in temporary copies
without updating the manifest, and require rejection in each case.

Compare per-row exponential likelihoods against each R source, using both
`survjax.log_likelihood` and `dist.log_likelihood`, and requiring clear
`data.issues` and `dist.parameter_issues()` diagnostics for valid inputs.
Compare direct survival/log survival and include a summed-likelihood
sanity check without letting totals replace per-row assertions. Use existing
central tolerances. Loader checks and reference tests accompany its code;
keep the step within 300 changed lines and stop with REVIEW.md.

## A8.5 Weibull reference tests

Files: `tests/reference/test_weibull_bc_like_r.py`; shared loader changes only
if the reviewed schema requires them. No public API or tolerance changes.

For all three cases, construct row-specific scales from group indicators and
compare every per-row log contribution, observed-time survival and the
`(141, 3)` grid with both R sources. Check scalar common shape plus batched
scale broadcasting, natural/from_log parameter equivalence, raw/prepared-data
likelihood equivalence, and eager/jit/vmap agreement against the same stored
references. Keep transformation coverage focused on these reference cases;
the broader derivative suite belongs to A7/A9. Both dtypes use existing
`tests/impl_tolerances.py` targets; investigate failures without loosening them.
Keep the step within 300 changed lines and stop with REVIEW.md.

Use the fixture's stored `grid.time` values as inputs, rather than rebuilding
the R `seq` grid with JAX arange/linspace; decimal-looking times can differ by
one ULP when rebuilt. Describe the grid's direct psurvreg route in A8.d1:
locations are supplied explicitly and cross-checked with flexsurv, without
singleton-fit checks for grid cells.

## Validation and documentation

For each code step run appropriate focused checks, Ruff, mypy, own tests in
separate `JAX_ENABLE_X64=1` and `0` processes, and `git diff --check`.
Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Before integration, run the
gate only through the last completed step on that checkout. After A7 and all
A8 steps are integrated, run `scripts/check.sh --through A8`. Normal Python
tests must succeed without R, network access or local real datasets.

A8.d1 follows reviewed code: update `docs/builtin-families.md` with the scope
of the R comparison evidence and a regeneration pointer. Audit
`docs/distributions.md` and `docs/api.md` for obsolete verification claims;
edit only where needed. Build and inspect both dtype renders through
`scripts/build_docs.sh`, documenting remaining limits. One topic, at most
300 changed lines, its own REVIEW.md and stop.

Every approved implementation step gets its own commit on a maintainer-created
branch, REVIEW.md with invariants and spec/test mapping, and a stop. No merges,
tags, protected test edits, spec edits, new Python dependencies or later-slice
fixtures. Planning used repository inspection and R metadata checks only;
pytest, lint, typing, docs builds and runtime gates were not run.

---

---

# Historical main plans through A7

Imported verbatim; old status/next-step instructions are historical only.

---

# Incoming main plans (preserved integration history)

# SurvivalData size interface — revision 37 (2026-10-26)

Status: **implemented and verified; ready for review**. Approval: "ok i switched
branches you do it". The maintainer-created branch is `len_data`; the only
pre-existing change was this plan. Planning started on clean `main`.
This entry is the active plan; older entries below are historical.
One code step, one commit, then a review stop. Target under 300 changed
source/test lines.

## Contract and current evidence

- Spec section 3.2 (revision 37) adds `data.shape`, `len(data)` and an explicit
  prohibition on truth testing. Sections 3.2 and 12.1 cover the shape,
  indexing, PyTree and transformation checks for this step.
- `shape` is a tuple of Python ints obtained from stored leaves, static under
  jit and per-example under vmap. It must not enter PyTree auxiliary data.
- `len(data)` counts the leading axis, including zero for empty data. Scalar
  data raise TypeError. Multidimensional length is not the total element count.
- `bool(data)` raises TypeError for every shape, including scalar and empty
  data; implementing this is required to prevent Python falling back to len.
  The contract adds neither `size` nor `ndim`.
- At planning time `src/survjax/data.py` had indexing and PyTree reconstruction but none
  of these three interfaces. Existing indexing and metadata tests provide
  regression coverage for the six leaves and inherited may-contain flags.
- `.spec-seen` already records `be26e50cc35db1c2affc3dafcb4bf4cf4baf1ebf`;
  `../scripts/spec-since.sh dev` reports `spec unchanged since be26e50`.
  No spec or marker change is needed. No contract ambiguity was found.

## Implementation after approval

Public API touched: `SurvivalData.shape`, `SurvivalData.__len__` and
`SurvivalData.__bool__`. No new package exports.

- `src/survjax/data.py`: add a read-only `shape` property annotated
  `tuple[int, ...]`, returning `self.lower.shape`; add `__len__ -> int` with
  an explicit scalar TypeError and leading-axis result; add `__bool__ -> bool`
  that always raises TypeError with a clear truth-value message. Document
  shapes, return values and errors in their docstrings and mention the size
  interface in the class summary. Keep construction, leaves, auxiliary data
  and reconstruction unchanged. No new helper or dependency is needed.
- Create `tests/unit/test_data_size.py` with small cases and spec 3.2
  docstrings. Expected shapes and lengths come directly from explicit input
  dimensions, independently of the new methods:
  - Both constructors: scalar `()`, vector `(3,)`, empty `(0,)` and matrix
    `(2, 3)`; verify shape elements and eager lengths are Python ints, scalar
    len raises TypeError, and matrix len is 2 rather than 6. Include `(2, 0)`
    to distinguish a nonempty leading axis from zero total elements.
  - Integer indexing, slices and tree-map slicing: shape and len follow the
    selected leaves; integer selection of a vector produces scalar data.
  - Eager/jit queries agree; inspect shape types inside tracing because jit
    converts returned Python integers to scalar arrays. Under vmap over a
    matrix, the body sees `(3,)` and len 3; over a vector, it sees `()` and
    scalar len raises TypeError. Include jit-of-vmap coverage.
  - Truth testing raises TypeError for scalar, empty, vector and matrix data,
    eagerly and while tracing under jit/vmap. This includes `if data:`.
  - Tree flatten/unflatten preserves the six leaves and auxiliary flags;
    differently shaped slices retain the parent's tree structure. Assert
    `size` and `ndim` are absent. No derivative checks apply to this static
    shape interface; no tolerances are needed.
- Update this PLAN.md with actual evidence and overwrite REVIEW.md with
  reading order, invariants, spec-to-test mapping, exact checks and doubts.
  Keep `tests/spec/`, the spec, docs and protected process files unchanged.

## Verification and completion

1. Run the new size tests plus existing data indexing, representation,
   metadata and tracing tests in separate float64/float32 processes:
   `JAX_ENABLE_X64=1` and `=0`, using `uv run pytest`.
2. Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
   on the final review snapshot. This runs Ruff, mypy, own tests, source
   doctests and independent acceptance tests through the implemented gate in
   both dtypes. Do not edit, skip or xfail independent tests if they fail.
3. Run `git diff --check`; report exact results, failures and skips. Tests
   and static checks have not been run during planning. No R checks are
   needed for a shape-only interface.
4. Commit code, tests, PLAN.md and REVIEW.md together on the maintainer-created
   branch with an imperative subject and rationale. Stop for review; do not
   start another step or claim Slice A completion.


### Implementation evidence

Implemented the three interfaces and 22 new own test cases. Source/test diff:
34 added source lines and 139 test lines, below the 300-line budget.
Focused own tests passed in both processes: 65 passed in 5.31s (float64),
65 passed in 4.30s (float32), no skips or failures.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
exited 0: `check: all gates passed through A7`. Ruff: 72 files already
formatted / All checks passed! Mypy: no issues found in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 774 passed, 7 skipped, 157.24s | 780 passed, 1 skipped, 150.46s |
| Source doctests | 4 passed, 0.95s | 4 passed, 1.03s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 103.10s | 2118 passed, 24 skipped, 218 deselected, 102.64s |

Full log: `/private/tmp/survjax-len-data-gate.log`. No new skips or exclusions;
independent tests were not modified. No R or docs build was run. REVIEW.md
records invariants and the test mapping. Final Ruff/diff checks follow this
record update; commit on `len_data` and stop for review.

### Docs to-do

A later approved docs step can explain shape, leading-axis len, scalar errors
and explicit truth-value rejection in the observation-data guide. Public
API docstrings ship with this code step; do not edit docs/ here.

---

# Revision 35 — Remove the public likelihood function (2026-10-25)

Status: **LL.1 merged; LL.2 implemented and verified; ready for review** ("ok merged; now up to
next bit"). Reuse the maintainer-created `remove-ll` branch at `fab100f`,
which is also the merged main tip. The starting tree is clean. No further
plan approval is needed for the LL.2 scope below.
This entry supersedes earlier plans that require two public likelihood entry
points; their implementation/review records below remain historical evidence.
Two small steps, each with its own commit, REVIEW.md and approval stop.
Approval of the first step does not start the second.

## Contract and current evidence

- Spec revision 35, decision 0048: `dist.log_likelihood(data=None, *,
  time=None, event=None)` is the only public likelihood entry point. Remove
  `survjax.log_likelihood`, retaining the method's signature, numerical
  behavior, derivatives, diagnostics and prohibition on family overrides.
  Sections covered: 4.2, 6.1, 7.1–7.3, 10.2, 12.1 and 12.2.
- `.spec-seen` already records `9696894`, the revision-35 spec commit;
  `../scripts/spec-since.sh dev` reports `spec unchanged since 9696894`.
  No marker update or design-repo change is needed.
- Updated independent tests arrived in `744fb55`, with source commit
  `8041b326a194e89f56aade79b112e7f3c5bbc601`. They now require absence of
  the top-level name and agreement of the method's two input forms in values,
  gradients and Hessians. Later-step tests also migrated to the method;
  their presence does not authorize implementing those steps.
- Current code still re-exports the function. The base method delegates to
  that function, and its docstring describes equivalence with it. Implementer
  tests still exercise the function/method matrix. The likelihood tutorial,
  API reference and simulation guide also retain function calls.
- Planning checks were read-only inspection of source, tests, docs, spec diff,
  branch/history/status and spec alignment. No tests or docs builds have run
  for this change yet. No unresolved contract question was found.

## LL.1 — Public API and implementer tests (first approval stop)

Public API change: remove the top-level function; retain the existing method
and both of its input forms. Target less than 300 changed source/test lines.

Files and implementation:

- `src/survjax/__init__.py`: remove the function import and `__all__` entry.
- `src/survjax/likelihood.py`: delete the public `log_likelihood` wrapper.
  Keep `_resolve_data` and the shared `_log_likelihood` evaluator here, as
  section 10.2 specifies. Their numerical bodies remain unchanged.
- `src/survjax/distributions/base.py`: delegate directly to those existing
  private helpers, retaining the deferred import to avoid a cycle. Make the
  method docstring self-contained: input forms, shapes, common dtype,
  unsummed output, NaN/zero/overflow behavior, diagnostics and static misuse.
  Do not add a compatibility alias or another public function.
- `tests/unit/test_import.py`: update the exact export set and explicitly
  assert absence of `survjax.log_likelihood`, including after importing the
  evaluator module and invoking the method.
- `tests/unit/test_likelihood_entry_points.py`: remove function/method
  parametrization and function-only invalid-distribution tests. Retain all
  method input-validation and transformation coverage. Add direct comparison
  of raw/prepared values, first parameter derivatives and Hessians for the
  two built-ins, eagerly and under jit; retain independent analytical anchors.
- `tests/unit/test_likelihood_transformations.py`: remove the obsolete
  entry-point axis while keeping both forms, both built-ins and second-order
  forward/reverse finite-difference checks, eager and compiled.
- `tests/unit/test_likelihood_tails.py`,
  `tests/unit/test_likelihood_boundaries.py` and
  `tests/unit/test_likelihood_debug_nans.py`: migrate function calls to the
  method and remove only duplicate function/method cases. Preserve tail
  expectations, tolerances, zero-censoring regression, debug mode and
  eager/jit/vmap coverage. Existing tests importing `_log_likelihood` in
  `test_likelihood_core.py` and `test_likelihood_parameters.py` need no change.
- Update this plan's status and overwrite `REVIEW.md` with reading order,
  invariants, spec-to-test mapping, exact results and remaining limitations.

Verification after approval:

1. Run the changed own modules plus likelihood core/parameter modules in
   separate `JAX_ENABLE_X64=1` and `=0` processes using `uv run pytest`.
2. Run the updated independent A6 entry-point/method modules in both dtypes
   with `SPEC_THROUGH=A7`. Never edit, skip or xfail independent tests.
3. Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
   on the final review snapshot. This includes Ruff, mypy, all own tests,
   source doctests and independent tests through the implemented A7 gate.
   A8–A10 acceptance tests are deferred to their own implementation work.
4. Search tracked source/tests/docs/notebook content for obsolete public
   calls; at this step remaining calls must be confined to the LL.2 docs
   to-do or explicit absence assertions. Run `git diff --check`.

Report actual counts, removed duplicate cases, skips and failures; never infer
numerical preservation from a rename alone. If a numerical change is needed
or the source/test diff exceeds the budget, split/replan before expanding.
Commit on `remove-ll` with an imperative subject and rationale, then stop.

Implementation evidence: source/test diff is 177 changed lines; all five
private likelihood helper ASTs match the starting commit. Focused own suites:
205 passed per dtype (114.52s float64 / 114.19s float32). Updated independent
A6 entry-point/method suites: 35 passed per dtype (7.01s / 7.03s). No skips
or failures. Removed 42 function-only/duplicate cases and added 17 cases
(16 input-form comparisons and one API-absence test), a net decrease of
25 cases per dtype. Ruff reports 71 files already formatted / All checks
passed; mypy reports no issues in 17 source files.

The final `scripts/check.sh --through A7` ran after saving REVIEW.md and
exited 0: `check: all gates passed through A7`. Ruff: 71 files already
formatted / All checks passed; mypy: no issues in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 752 passed, 7 skipped, 160.92s | 758 passed, 1 skipped, 158.62s |
| Source doctests | 4 passed, 1.09s | 4 passed, 1.03s |
| Independent through A7 | 2082 passed, 28 skipped, 218 deselected, 112.08s | 2086 passed, 24 skipped, 218 deselected, 109.65s |

Complete output: `/private/tmp/survjax-ll1-final-gate.log`. REVIEW.md remains
unchanged after the gate; only this plan's result record is updated afterward.
No new skips/exclusions. Final Ruff formatting/lint and diff checks follow
this result update, then commit LL.1 and stop. LL.2 has not started.

### Docs to-do

The narrative docs intentionally wait for the separate LL.2 docs step below.
The API reference and executable tutorial still reference the removed function
after LL.1, so the full docs build is expected to be broken until LL.2 lands.
LL.1 is not completion of the whole migration or of Slice A.

## LL.2 — Likelihood API documentation (separate approval stop)

Status: **implemented and verified; ready for review**. Followed the existing
docs plan below; write REVIEW.md, commit on `remove-ll` and stop. The spec is still
revision 35; `.spec-seen` already matches. No source/test changes are needed.

One topic: migrate documentation to the distribution method, below 300 changed
lines. Files: `docs/likelihood.md`, `docs/api.md`,
`docs/simulation-guide.md`, PLAN.md and REVIEW.md. Public API touched: none.
Spec: 6.1, 7.1, 10.2, 10.5 and 12.1.

- Rewrite “Two entry points, two input forms” around one method accepting
  `SurvivalData` or raw `time`/`event`. Compare those two forms in the
  executable example and remove function-choice advice.
- Remove the obsolete `autofunction` directive from the API reference;
  retain the method documentation and adjust nearby wording if needed.
- Change the simulation objective to construct a distribution and call its
  method. Audit the tracked notebook and other current guides for stale
  function calls; the distribution guide's local objective function is not
  the removed package API and need not be renamed.
- Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`
  with fresh execution, warnings fatal, in separate float32/float64 processes.
  Inspect rendered method documentation, the two-form example's output,
  links and layout in both modes. The simulation sketch uses optional
  optimistix outside the core; do not add a dependency or claim solver
  execution from a static code block. Record any visual inspection limit.
- Repeat the obsolete-call audit, Ruff formatting/lint and `git diff --check`.
  Update PLAN.md and overwrite REVIEW.md with exact checks and reading order;
  commit this docs topic on `remove-ll`, then stop for review.

LL.2 evidence: 36 changed lines across the three docs pages. The docs build
exits 0 with `build succeeded` in float32 and float64; likelihood-page execution
takes 4.01s / 3.35s. Log: `/private/tmp/survjax-ll2-docs-build.log`.
The initial sandbox run could not open Jupyter kernel sockets; the permitted
rerun passed. Existing kernel TCP transport notices appear in the log; there
are no Sphinx documentation warnings. Chrome screenshot inspection passes for
the tutorial's two-form section and API method reference in both dtype builds.
Both forms show -23.124392 / -23.12439364 respectively. The HTML audit checks
273 local page/anchor links per build and the method's reference anchor.
The simulation guide is excluded from Sphinx; a temporary check extracts only
its revised objective and verifies value/gradient/Hessian = 6/4/6 at log-rate
zero on synthetic data, in both dtypes. Its optional solver was not run.
Tracked docs/notebook audit finds no obsolete package-function references.
REVIEW.md records commands, evidence and scope. Final Ruff formatting/lint
passes after the review edit: 71 files already formatted / All checks passed.
`git diff --check` exits 0. The full library gate and R were not run for this
docs-only step. Generated HTML and temporary checks remain untracked.

---

---


# A7.d1 follow-up — Zero-censoring diagnostics (2026-10-23)

Status: **implemented and verified; ready for review**, approved directly
by the maintainer's requested correction.
On isolated A7 at `f7236c4`, add the valid/uninformative diagnostic result
to the zero-censoring section of docs/numerical-behaviour.md, with one
executable output. Spec 2, 3.3 and 6.2: invalid_rows=False, ok=True,
uninformative=True; the last flag is non-fatal. No source/API/test changes.
Rebuild in both dtypes, inspect the new HTML output locally, update REVIEW.md,
run final Ruff/diff checks, commit on isolated A7, then stop for review.
The existing browser visual-inspection limitation remains.

Evidence: diagnostics and host checkpoint verified in separate x64/x32
processes; both docs builds pass (page execution 2.90s / 2.82s). HTML checks
pass with the new valid/uninformative display among 8 outputs per mode.
Final Ruff formatting/lint and diff checks pass after the review edit.

---

# A7.d1 — Numerical behaviour tutorial (2026-10-23)

Status: **implemented; build and HTML checks pass, ready for review**.
Authorized by "ok merged, implement the next bit; plan is approved".
A7.1/A7.2 and their review fixes are merged at `9a034f4`;
the existing `A7` branch is reused in `/private/tmp/survjax-a7-docs`.
The shared checkout switched to A10-cindex during the work; only this step's
docs/plan changes were transferred, and the shared checkout was left clean
on A10-cindex. Final verification and the commit use this isolated A7 tree.
Spec revision 34 is unchanged. This implements the approved A7 docs topic below, one docs commit
and review stop; it does not start A8 or claim Slice A completion.

- Create executable `docs/numerical-behaviour.md`: mathematics, small synthetic
  exact/right examples, then interpretation. Cover finite log tails when
  probabilities round to zero/one, log-contribution overflow, true zero versus
  invalid/unsupported NaNs, zero censoring and parameter derivative limits.
  Show why a finite gradient does not certify a finite loss and how to check
  that at a host checkpoint, without introducing an optimizer or fit helper.
- Link the page from `docs/index.md`. Correct only the obsolete paragraph in
  `docs/likelihood.md` that identifies every -inf as true zero likelihood;
  link to the new explanation. Retain its construction-only triage wording.
  Target at most 300 changed narrative/code-cell lines across these pages.
- Use current public APIs, existing dependencies and synthetic data only.
  No source, docstring, test, spec, dependency or protected-path edits.
  No assert-cell matrix or repeated fitting tutorial. Broader custom-family
  authoring and stable log-CDF details remain in their existing guides.
- Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`
  (fresh execution, warnings fatal, separate float32/float64 builds). Inspect
  rendered navigation, mathematics, output and layout in both modes. Check
  examples against hand algebra and the merged A7 evidence, including overflow
  -inf with clear diagnostics and NaN loss with finite gradient. Make no R
  agreement claim. Record exact build/check results and remaining limits in
  REVIEW.md; format final Markdown before final Ruff/diff checks and commit.
  A full code/test gate is not planned for this docs-only change.

Evidence: final isolated-A7 docs build exits 0 with build succeeded in both
dtype modes; new-page execution 2.95s float32 / 2.77s float64. Local HTML/output
inspection passes (7 outputs, 6 headings, 2 display equations, 5 table rows,
30 local links/anchors per mode). Narrative/code-cell diff: 222 lines.
REVIEW.md records exact commands, branch isolation and environment retries.
Browser visual inspection was blocked by the file-URL policy and remains a
review limitation; no A7/Slice A completion claim. Final formatting/lint and
diff checks pass after the review edit. Commit only from isolated A7, then stop.

---

# A7.2 review fixes — F1 and F2 (2026-10-23)

Status: **implemented and verified; ready for review**, authorized by
"immediately implement the required fixes". Read the updated
`../review/slices/A7.md` for `8f903d7`.
One small follow-up commit on the existing `A7` branch, then review and stop.

- F2: in `tests/unit/test_likelihood_debug_nans.py`, retain ordinary gradient
  comparisons, add exact assertions for the reference's structural zeros,
  and use the unchanged relative-only tail tolerances on the explicitly
  identified tiny-right column in each per-row built-in case. No new tolerance
  numbers, source changes or test-case matrix are needed. Spec 12.1.
- Verify sensitivity in an isolated temporary source copy: stop gradients on
  survival only where `0<t<1e-8`, as in the reviewer mutation. Both per-row
  cases must fail in each dtype, especially float32. Run the ordinary focused
  boundary/debug suites in both dtypes, with no new exclusions.
- F1: overwrite `REVIEW.md` for this follow-up, including the correction to
  the previous final-snapshot gate claim. Format its Python snippet (or use
  plain command text) with Ruff. Finish editing/formatting this review before
  running `scripts/check.sh --through A7`; save the complete final gate output
  in a temporary log referenced by the review. Record the final counts here
  and in the final response without changing REVIEW.md after that gate.
- Files: that test module, PLAN.md and REVIEW.md only; public API, numeric
  tolerances, dependencies, protected paths and docs remain unchanged. Target
  under 50 changed test lines. Run the full through-A7 gate with the existing
  tmp uv cache, then final Ruff formatting and `git diff --check` after any
  plan-only result update. Record actual commands and results. Commit on A7,
  then stop. The separate numerical docs step remains pending.

Final evidence: focused own/independent tests pass, 53 per dtype (10.73s /
10.81s). The F2 mutation is rejected by both per-row tests in each dtype:
2 expected failures in 1.35s per process, including relative-only float32
failures for the exponential and both Weibull tiny-gradient components.
No source or numeric tolerance changes were needed.

The complete through-A7 gate ran after the final REVIEW.md edit, exited 0
and ended `check: all gates passed through A7`. Ruff: 70 files already
formatted / All checks passed; mypy: no issues in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 777 passed, 7 skipped, 141.39s | 783 passed, 1 skipped, 149.10s |
| Source doctests | 4 passed, 0.99s | 4 passed, 0.90s |
| Independent through A7 | 2123 passed, 28 skipped, 44 deselected, 96.82s | 2127 passed, 24 skipped, 44 deselected, 91.00s |

Full output: `/private/tmp/survjax-a72-fixes-final-gate.log`, linked from
REVIEW.md. REVIEW.md was left unchanged after the gate. Only this plan's
result record was updated afterward; final Ruff formatting and diff checks
also passed. No new skips/exclusions; stop after this follow-up commit.

---

# A7 — Likelihood numerical verification (2026-10-23)

Status: **A7.1 merged; A7.2 implemented and verified, ready for review**,
authorized by "yeah do it".
The maintainer requested the next subslice and review follow-up on 2026-10-23.
The existing maintainer-created `A7` branch is reused; it and `main` were both
at merged A7.1 commit `3930115` when this planning step began.
The maintainer created `A7` and synced the independent tests.
Starting tree was clean at `2c030c5`;
spec revision 34 is marked read at `d243a96`. No implementation changes have
been made during planning. This entry supersedes the old A7 outline below;
older entries remain history, including their now-stale deviation statuses.

## Scope and evidence already present

A7 verifies exact/right-censored likelihoods without truncation and the
existing direct methods (spec 12.2). The synced A7 acceptance suite already
passes in both dtypes, including revision 34's overflowing contributions.
No source or public API change is currently needed. Add focused implementer
tests with independent mathematical expectations; do not copy the independent
suite or import its helpers, families or tolerances.

- `test_likelihood_core.py` and `test_likelihood_entry_points.py` already
  cover ordinary hand-derived contributions, shapes, input forms, invalid
  data, true zero density/mass and unsupported modes.
- `test_likelihood_transformations.py` already checks second-order parameter
  derivatives by finite differences through both public entry points/forms.
  The parameter suites cover declared-violation isolation. Retain this
  evidence; do not repeat its full matrices in A7.
- `test_log_cdf_tails.py`, `test_exponential.py`, `test_weibull.py`,
  `test_family_boundaries.py`, `test_family_infinity.py` and
  `test_family_mixed_infinity.py` already cover direct-method tails, mpmath
  derivatives, branch splits, zero/infinity boundaries and proper/improper
  limits. Rerun them through the gate and cite them in the A7 review.
- New work concentrates on public likelihood values across scales, tiny and
  underflowing probabilities, overflowing log contributions, and eager
  mixed-batch safety including parameter gradients.

Narrow intervals, interval cancellation, truncation, zero/tiny normalization
and the bounded-support numerator/denominator distinction are Slice B.
R fixtures are A8; installation, the final derivative/coverage audit, example
and remaining Slice A gate evidence are later work. No dependencies, numerical
helpers, benchmark framework or new public containers are introduced here.

## A7.1 — Likelihood tails and overflow (first approval and review stop)

Files: create `tests/unit/test_likelihood_tails.py`; update
`tests/impl_tolerances.py` only to explain applicability of the existing tail
tolerances; overwrite `REVIEW.md` with this step's evidence. Public API touched:
none; exercise `survjax.log_likelihood` and `dist.log_likelihood` only.
Target 220–270 test/comment lines, below 300 changed code/test lines. If the
step needs a source fix or exceeds that budget, report and replan before it.

Tests and expected-value origins:

1. Compare Exponential exact/right contributions with the defining formulas
   `log(r)-r*t` and `-r*t`, and Weibull with
   `log(k)-log(b)+(k-1)*log(t/b)-(t/b)**k` and `-(t/b)**k`.
   Use a small set of named, explicit scale/shape cases, including shapes
   below, at and above one. Compute references locally with mpmath at 100
   digits, from inputs rounded to the process dtype before conversion to
   mpmath. Keep this helper local to the test module; never obtain an expected
   value from a distribution method or a likelihood call.
2. Include tiny normal cumulative hazards (for example `1e-20` and `1e-30`)
   where survival rounds to one, and upper tails where density/survival
   underflows but the log contribution is finite. Include a representable
   cumulative hazard around `1e32` and early exact times with decreasing and
   increasing Weibull hazard. Assert the appropriate sign/finiteness as well
   as relative accuracy, so neither zero nor a floor can pass.
3. Check first parameter derivatives at one tiny-hazard and one finite
   upper-tail point, using hand derivatives: Exponential `event/r-t`;
   Weibull shape `event/k+(event-H)*log(t/b)` and scale
   `k*(H-event)/b`. Use forward and reverse mode on these valid smooth
   cases; retain the existing interior Hessian/finite-difference suite.
4. Separate probability underflow from log-contribution overflow: with
   Weibull `k=50, b=1, t=1e10`, `H=1e500`, exact/right values must be
   `-inf` with clear data/parameter diagnostics. A finite neighbour keeps its
   hand-derived value. Check that neighbour's forward derivative for shared
   parameters and reverse derivative for per-row parameters. Do not assert
   derivatives of the infinite row or shared-parameter reverse derivatives
   next to it (spec 7.3, revision 34).
5. Exercise representative mixed exact/right vectors eagerly, under `jit`
   and row-wise `vmap`, with both public entry points represented. Reuse A6's
   exhaustive entry-point/input-form equivalence evidence instead of
   multiplying every tail case by the entire call-form matrix.

Use `TAIL_ATOL=0` and existing `TAIL_RTOL` (1e-8/1e-4) for normal nonzero
tail values and derivatives. Explain in the tolerance file that the selected
likelihood cases have bounded log/power error amplification within those
targets. Use ordinary `ATOL`/`RTOL` for O(1) anchors and expected derivative
zeros; use exact comparisons for infinities and structural zeros. No numeric
tolerance changes are planned; report failures rather than loosening them.

Spec covered: 2, 4.1, 5, 7.1–7.3, 12.1 and the Slice A boundary in 12.2.

After approval: run the new own module and independent
`test_a7_likelihood_mpmath.py` / `test_a7_likelihood_tails.py` in separate
`JAX_ENABLE_X64=1` and `=0` processes (`SPEC_THROUGH=A7` for spec tests),
then `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
and `git diff --check`. Record exact commands, counts, skips and failures in
`REVIEW.md`, with reading order, invariants, spec-to-test mapping and limits.
Commit this step on `A7`, then stop; approval of A7.1 does not start A7.2.

Implementation: 36 new cases in a 231-line module, plus three tolerance-comment
lines. Focused own/independent tests pass in both dtypes (91 per process).
No source or numeric tolerance changes were needed. The through-A7 gate passes:
own suites 758/764 passed (7/1 skipped), independent suites 2123/2127 passed
(28/24 skipped, 44 deselected each), and source doctests 4 passed per dtype.
The A7.1 `REVIEW.md` records exact commands/results. A7.2 implementation has
not started. The reviewer approved A7.1 with no blocker or should-fix; S1 is
incorporated into the A7.2 plan below.

## A7.2 — Boundary and inactive-branch safety (current plan)

Status: **implemented and verified; ready for review** ("yeah do it"). Read all of
`../review/slices/A7.md`, including S1 and its mutation evidence. Spec revision
34 remains unchanged. This is one test-only step, followed by one commit and
review stop on `A7`. No source/public API, numeric tolerance, dependency,
independent-test or narrative-doc changes are planned.

### Files and test budget

Create
`tests/unit/test_likelihood_boundaries.py` and
`tests/unit/test_likelihood_debug_nans.py`, targeting approximately 240–280
lines combined, below 300 changed code/test lines. Overwrite `REVIEW.md` with
reading order, invariants, requirement mapping, mutation results and exact
check outputs; update this plan's status. If a source change is needed or the
tests exceed the budget, report and replan before expanding the step.

### Boundary values and derivatives

- At the smallest positive normal time, check exact/right values and finite
  parameter gradients against hand algebra; avoid a subnormal ratio by
  evaluating `log(t)-log(b)` in the independent reference. Include Weibull
  exact rows at shapes below/at/above one. For right rows require relative
  precision only where H stays normal (Exponential rate 1, Weibull shape 0.5).
  Use ordinary tolerances for the finite exact values/derivatives, relative-only
  tail tolerances for normal tiny right contributions. No new tolerance scales
  or backend-specific subnormal assertions.
- At right censoring `c=0`, require exactly zero contribution and parameter
  derivatives, including Weibull with infinite density at zero. Confirm a
  neighbouring event retains its analytical derivative. Distinguish this from
  an invalid exact time at zero, already covered by A6.
- Expected values and gradients come from the same defining algebra as A7.1,
  evaluated independently on dtype-rounded inputs with NumPy/mpmath as needed.
  Keep reference work local and host-side; do not import another test module.
  Do not repeat A7.1's scale/tail matrix or A6's call-form/Hessian matrix.

### Mixed batches under eager NaN debugging

- Run public likelihood values and first parameter derivatives eagerly inside
  `jax.debug_nans(True)` on valid batches mixing interior events, early events,
  finite log tails, zero censoring and tiny positive censoring. Cover both
  built-ins with shared/per-row parameters, a `from_log` example, flag/bounds
  data and both public entry points without a full Cartesian product.
- Include small local custom-family probes: the square-root-hazard family
  specified below, and a cure mixture using
  `S(t)=p+(1-p)*exp(-r*t)`. For the latter, right censoring uses `log S(c)`
  including the cure mass; exact events have `log((1-p)*r)-r*t`. Keep exact
  rows where the generic density's hazard is representable, and check a far
  right-censoring row tending to `log(p)`. The square-root probe replaces the
  earlier constant-hazard probe, keeping the step small while adding S1's
  missing sensitivity. These are local test families, not public additions.
- Check mathematical values alongside finite gradients; debug mode alone is
  not correctness evidence. Keep invalid/unsupported NaNs and infinite-row
  derivative probes outside debug mode. Reuse A7.1 tail references and existing
  direct-method boundary coverage by citation, without importing test modules.

### Reviewer S1 — enforce inactive density-time substitution

The reviewer showed that the existing own and independent suites survive
replacing the density-time substitution with the actual censoring time.
Built-in boundary handling and the old constant-hazard/cure probes therefore
do not demonstrate this safeguard. Test the observable value/gradient failure
using a frozen, registered `RootHazard` with one positive rate parameter,
`log_survival(t)=-rate*sqrt(t)`, and `reference_time()=ones_like(rate)`.
Leave its density to the generic autodiff default; do not guard its primitive
at zero, which would hide the regression under test.

On `[right at 0, exact at 1]`, the first contribution and its rate derivative
are exactly zero; the event contributes `log(rate/2)-rate`. At rate 2 the
summed value is -2 and the rate derivative is -0.5. Assert these mathematical
anchors without NaN debugging and under `jax.debug_nans(True)`, with both
public entry points represented. Include eager/jit/vmap agreement on this
finite, valid regression case without differentiating observation times.

After the tests pass on the real checkout, copy `src/` to an isolated
temporary directory under `/private/tmp` and change only the copied
`density_time = jnp.where(~bad & exact_rows, time, reference)` to
`density_time = time`. In separate x64/x32 processes, verify `survjax.__file__`
points into that copy, then run only the new S1 cases against it (ordinary
and debug modes). Require the mutant to fail: a wrong/nonfinite gradient
outside debug mode, and a NaN debug exception in debug mode. Record actual
failure names/reasons; the reviewer results are motivation, not our evidence.
Keep the repository source and independent tests untouched throughout.
No persistent mutation framework, Git worktree or broad mutant suite is needed.

### Verification, docs to-do and stop

Spec covered: 2, 3.3, 4.2, 4.5, 7.1–7.3, 12.1–12.2. Run the two new own
modules and independent `test_a7_likelihood_boundaries.py` /
`test_a7_debug_nans.py` in separate x64/x32 processes, with `SPEC_THROUGH=A7`
for the independent tests. Verify S1's mutation is caught in both modes as
above. Then run
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
and `git diff --check`. Use the unchanged project tolerances. Record failures
rather than loosening them; log any spec question as a deviation and ask.
One commit on `A7`, then stop; no A7 docs implementation in this step.

Docs to-do: retain the numerical-behaviour topic below. Add the custom-family
inactive-input guarantee and its boundary limits to the later page's evidence
map if useful; keep low-level mutation details in `REVIEW.md`.

Planning checks for this refinement: clean tree and merged commit verified;
full reviewer report, relevant own/independent tests, existing substitution
and spec derivative limits read; `../scripts/spec-since.sh dev` reports no
change since `d243a96`. No new tests, mutation check, full gate or docs build
were run during planning. S1's empirical results above remain reviewer evidence
until implementation verifies the new test's sensitivity.

Implementation evidence (A7.2): 19 new cases in two modules, 299 test lines
combined. The focused own/independent boundary/debug run passes in both dtypes
(53 per process). S1 is verified against an isolated temporary source copy:
each dtype reports four expected failures, two NaN gradient mismatches and
two NaN debug exceptions. No repository source or tolerance change was needed.
The full through-A7 gate passes: own 777/783 passed (7/1 skipped), source
doctests 4 passed per dtype, independent 2123/2127 passed (28/24 skipped,
44 deselected each). `REVIEW.md` records exact commands and mutation evidence.
The numerical docs step has not started.

## A7 Docs to-do and separate docs step

After the code/test steps merge, plan one numerical-behaviour topic under
spec 10.5: a short executable `docs/numerical-behaviour.md` and navigation in
`docs/index.md`, at most 300 changed lines. Explain log-space tail behavior,
finite log values versus underflowed probabilities, true zero versus overflow
(`-inf`) versus invalidity (`NaN`), and the derivative limits at infinite
likelihoods and upstream of parameter leaves. Use mathematics followed by a
small synthetic example and interpretation, following the maintainer's
tutorial style. Show checking a finite loss during fitting without adding an
optimizer. Keep the existing likelihood guide's construction-only triage
qualification; D-026 is resolved by revision 33.

Refine that page plan after seeing the final evidence. Build with
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`, executing
both dtype modes with warnings fatal; inspect rendered pages, record exact
results and limitations in `REVIEW.md`, commit, then stop. A7 completion still
does not claim Slice A completion or R-reference agreement.

## Planning checks actually run

- `../scripts/spec-since.sh dev`: unchanged since `d243a96` (revision 34).
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A7 .venv/bin/python -m pytest -q
  tests/spec/test_a7_likelihood_mpmath.py tests/spec/test_a7_likelihood_tails.py
  tests/spec/test_a7_likelihood_boundaries.py tests/spec/test_a7_debug_nans.py`:
  **89 passed in 8.48s**. Same command with `JAX_ENABLE_X64=0`:
  **89 passed in 8.52s**.
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A7 .venv/bin/python -m pytest -q
  tests/spec/test_a3_static_metadata.py tests/spec/test_a4_registration.py
  tests/spec/test_a4_nested.py`: **77 passed in 2.83s**. Same command with
  `JAX_ENABLE_X64=0`: **76 passed, 1 skipped in 2.82s**. These include the
  synced revision-33 requirements checked before planning A7.
- Full through-A7 gate, own suites, Ruff, mypy and docs build have not been
  run during planning. The targeted baseline is not a full-gate result.

---

# A6 docs rewrite — tutorial likelihood guide (2026-10-22)

Status: **implemented, ready for review**. Requested directly by the maintainer
("redo them; make them tutorial like. start with math followed by code").

- Rewrite `docs/likelihood.md` as a tutorial on one synthetic 10-patient
  cohort: derive the censored-data likelihood, show per-patient contributions,
  closed-form exponential MLE with a batched profile plot, Newton fits in log
  parameters with `jax.grad`/`jax.hessian` (exponential, then Weibull), a
  likelihood-ratio comparison, and NaN diagnostics. No assert cells.
- Update the guide's one-line description in `docs/index.md`.
- Docs only; no source, test or spec changes. D-026 stays open: the guide no
  longer presents a general NaN-triage recipe.

---

# A6 docs — Likelihood usage (2026-10-22)

Status: **implemented, ready for review**. The maintainer authorized this plan with
"get to it" and selected `docs-A6`; the branch is verified.

## One topic and files

- Create `docs/likelihood.md` as an executable MyST guide, following the
  existing notebook front matter. Cover exact/right contributions without
  truncation: log density for events, log survival for censoring, both input
  forms and both entry points, explicit caller reduction, trailing-axis
  broadcasting, and a small jitted objective with parameter derivatives.
- Explain data and parameter diagnostics, explicit host raising, NaN for
  unsupported modes and invalid rows, and genuine zero likelihood as -inf.
  Use newly constructed data in the Slice A triage example. D-026 remains
  open: do not generalize that recipe to edited/rebuilt normalized fields
  or settle the disputed remainder wording in this docs step.
- Update `docs/index.md` to describe the available likelihood subset and link
  the guide in the toctree; retain the incomplete Slice A qualification.
- Update `docs/api.md` with a guide link and generated entries for the
  inherited likelihood method and parameter diagnostics needed by the guide.
  Reuse existing docstrings; no public API or source changes are planned.
- Overwrite `REVIEW.md` with reading order, invariants, executable checks,
  actual build results and remaining limitations. One docs commit, then stop.

## Examples, checks and scope

Keep the documentation diff at or below approximately 300 changed lines;
this is one likelihood topic, not a full A6 extension-author reference.
Use synthetic data only and existing dependencies. Leave the simulation guide,
playground and broader constraint-authoring guide for separate docs steps.

Executable cells will check hand-derived exponential exact/right values,
function/method and raw/prepared equality, batch output shape, analytical
log-rate gradient, and row-wise diagnostic/unsupported-mode behavior.
Use existing project tolerances if numerical assertions need tolerances.
No new standalone tests or changes to tests/spec are needed for prose/examples.

Spec coverage: 3.3 (dtype), 4.7 (parameter diagnostics), 5 (broadcasting),
6.1–6.3 (inputs/diagnostics), 7.1–7.3 (likelihood and derivative limits),
10.5 (executable docs), 12.2 (Slice A support boundary).

After approval, run `scripts/build_docs.sh` with
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache` (both dtype builds, warnings fatal),
inspect the generated guide and API entries in both outputs, and run
`git diff --check`. Record exact outputs; no code gate or Slice A completion
claim. Planning checks only: git status/branches and read-only inspection of
existing docs, build configuration, likelihood source, spec and D-026.

Evidence: both executable dtype builds passed; generated HTML navigation and
API anchors were checked. REVIEW.md records exact checks and the sandbox retry.
The new guide is 193 lines; guide/navigation/API changes total 215 changed lines.

---

# A6.4 replanning — Exact/right likelihood (2026-10-21)

## Current step — A6.5 remaining derivative verification (2026-10-22)

Status: **implemented, ready for review**, authorized by "ok get to it" after the
remaining-gap explanation. Start at `dca2ee8` on the existing maintainer-
selected A6 branch, at the same commit as main. One test-only commit with
REVIEW.md, then stop; docs follow in their own reviewed steps.

- Create `tests/unit/test_likelihood_transformations.py` with two explicit
  tests: Exponential rate and joint Weibull shape/scale. On a small valid
  exact/right batch at positive finite times, run check_grads(order=2,
  modes=("fwd", "rev")) through the public function and method, raw and
  prepared forms, eager and jit. Differentiate only model parameters.
  Expected derivatives come from central finite differences; choose
  representative O(1) interior parameters, not boundary/tail cases.
- Use the mean contribution so the scalar objective is O(1). In
  `tests/impl_tolerances.py`, retain the existing derivative tolerances and
  add documented finite-difference step sizes: h=1e-4 for x64 and 1e-3 for
  x32. For these smooth O(1) cases, centered truncation is O(h**2); rounding
  is O(eps/h). These scales fit the existing 1e-7/1e-3 derivative targets.
  Never loosen the tolerances to pass a test; report a failure if they fail.
- Spec 7.3/12.1: second-order finite-difference checks close the identified
  gap. Existing suites already cover analytical values, Jacobians/Hessians,
  invalid-parameter isolation, jit/vmap, compile reuse, host-transfer guards
  and mixed valid debug_nans evaluation. Keep that evidence rather than
  duplicating its matrices. No src/, public API, docs/, spec-test, dependency
  or numerical implementation changes. Target under 150 test/comment lines.
- Run focused new tests in separate JAX_ENABLE_X64=1/0 processes, then
  scripts/check.sh --through A6 and git diff --check. Use the existing tmp
  uv cache; record actual outputs and any limits in REVIEW.md.
- Docs to-do: parameter diagnostics first, then likelihood usage. Maintainer
  writing instruction: tutorial prose, introduce the mathematics, follow it
  with code applying that idea, then interpret the result. Use one coherent
  synthetic example and meaningful displayed output. Examples teach use;
  avoid unit-test-style assert sequences or verification matrices. Keep
  API details in the reference. D-026 remains open for the triage wording.

Evidence: all 16 new checks pass on their first run in each dtype. The full
through-A6 gate passes (own 722/728 passed, existing 7/1 skips; independent
2023/2027 passed, existing 28/24 skips). REVIEW.md records exact commands and
results. This test-only step adds 87 test/comment lines. A6 implementation
and its planned verification are ready for review; tutorial docs are next.

## A6.4c review follow-up — Distribution argument type (2026-10-22)

Status: **implemented, ready for review**, authorized by "look at the review
and implement". Read all of `../review/slices/A6.4c.md`, reviewing `b1edf82` on A6.
One small follow-up commit on the same branch, then REVIEW.md and a stop.

- S1: add an isinstance guard at the public function's entry in
  `src/survjax/likelihood.py`, raising TypeError with a useful message for a
  non-SurvivalDistribution instance. Preserve signatures, resolver and numerics.
- In `tests/unit/test_likelihood_entry_points.py`, add one regression with
  named cases for None, an unrelated object and a distribution class rather
  than an instance, eager/jit. First demonstrate failure without the guard.
  Spec 6.1 and the public docstring's existing TypeError contract.
- N1: correct the keyword-only test's citation from 7.1 to 6.1 in that file.
  N2 explicitly requires no A6.4c action: defer the base.py module split.
- Run the own entry-point and independent entry-point/method suites in both
  dtype modes, then `scripts/check.sh --through A6` and `git diff --check`.
  Overwrite REVIEW.md with actual checks and the reading order. Public API
  shape is unchanged; no docs/, spec/, independent-test or dependency changes.
- Docs to-do: unchanged; D-026 remains open. No docs build is needed for this
  static guard and test-docstring citation correction.

Evidence: the new regression fails all six cases in each dtype before the
guard. With the guard, 113 focused cases pass per dtype. The full through-A6
gate passes (own 706/712 passed; independent 2023/2027 passed, with existing
skips). REVIEW.md records exact outputs. Source/test diff: 19 changed lines.

## Current step — A6.4c public likelihood integration (2026-10-22)

Status: **implemented, ready for review**, authorized by "do it".
Planning starts on clean `A6` at
`6a35b7a`, after the A6.4b2 review fixes and N3 rename. Spec revision 32 is
unchanged since `.spec-seen` (`cf4fef9`); the public integration outline below
still applies. This entry refines its files, test budget and validation order.
One implementation commit with REVIEW.md on the existing A6 branch, then stop.

### Files and public API

- `src/survjax/likelihood.py`: add typed
  `log_likelihood(dist, data=None, *, time=None, event=None) -> jax.Array`.
  One private resolver accepts SurvivalData unchanged, or constructs it from
  both raw keywords using the existing pure checked constructor. Reject mixed
  or missing forms and positional raw arrays with TypeError; point positional
  array users to `time=`/`event=`. Delegate evaluation to `_log_likelihood`.
  Keep all existing numerical preparation and row checks intact.
- `src/survjax/distributions/base.py`: add the same typed base method without
  the dist argument, forwarding directly to the public function. Use a local
  import for the function and TYPE_CHECKING for the data annotation if needed
  to avoid a cycle. No registration/declaration or direct-method changes.
- `src/survjax/__init__.py` and `tests/unit/test_import.py`: add exactly the
  `log_likelihood` export. No result wrapper, reduction or new diagnostic API.
- `docs/api.md`: add one generated function entry under the standing minimal
  API-reference authorization. Public function/method docstrings describe
  inputs, trailing-axis broadcasting (no time-grid axis), unsummed output,
  static TypeErrors and invalid/unsupported-row NaNs. No narrative docs here.

### Own tests and specification coverage

Create `tests/unit/test_likelihood_entry_points.py`, aiming for 130–180 lines;
total source/test/docstring changes should stay around 250–300 lines. Replan
before exceeding the budget; do not copy the existing core/masking matrices.

- Check raw/prepared and function/method values against hand-derived
  Exponential exact/right contributions. Confirm a custom family inherits the
  method using existing test coverage in the independent suite (4.2/6.1/7.1).
- Small named cases check mixed/missing forms, positional arrays, extra
  constructor keywords and incompatible raw shapes, through both entry
  points. Include a traced misuse case; signature checks remain static (6.1).
- Invalid raw indicator values must retain their constructor issue evidence
  and yield NaN only on the affected rows, eager/jit/vmap, through both forms.
  Reuse the core suites for normalized-field reconstruction cases (6.2/7.1).
- Compare values, Jacobians and Hessians in both entry points/forms with
  analytical Exponential anchors, using named transform cases and existing
  tolerances. Exercise jit and vmap without duplicating the full isolation
  matrix; independent method/isolation suites provide broader coverage
  (3.3/5/7.2–7.3/12.1).

### Verification, limits and delivery

Use UV_CACHE_DIR=/private/tmp/survjax-uv-cache and separate x64/x32 processes.
First run the new own tests and the independent entry-point, method,
gradient-isolation and unsupported-mode modules with SPEC_THROUGH=A6.
Then run `scripts/check.sh --through A6`: it covers every other synced A6
module plus Ruff, mypy, the full own suite and source doctests in both modes.
Run `scripts/build_docs.sh`, inspect the generated function entry in both
dtype builds, and finish with `git diff --check`. Record exact results and
any optional skips in REVIEW.md; independent tests and gate scripts stay
read-only. Planning used read-only inspection; implementation evidence is
recorded below and in REVIEW.md.

D-026 remains an open recipe-wording question; this step preserves the
specified stored-issue and current-field behavior without editing the spec.
Docs to-do: likelihood usage, diagnostic triage, method reference and examples
remain separate docs work. A6.5 composition checks and A7–A9/remaining Slice A
completion evidence remain later steps. Approval covers this integration
only; stop after its implementation commit and REVIEW.md.

Evidence: 52 own API cases and 544 focused independent cases pass in each
dtype. The complete through-A6 gate passes (own 700/706 passed; independent
2023/2027 passed), as do both executable docs builds. REVIEW.md records exact
counts, skips and the sandbox-only docs retry. The source/test/docstring/API
entry diff is 234 lines. No A6.5 or Slice A completion is claimed.

## A6.4b2 nit N3 — Name the current-field check (2026-10-22)

Status: **implemented, ready for review**, explicitly requested by the maintainer.
Rename the private helper to `current_field_issues` in `_data_validation.py`
and its import/call in `likelihood.py`; update the review notes. No logic,
public API or spec changes. Spec 3.3/6.2/7.1 coverage remains unchanged.
Run Ruff, mypy and the existing likelihood-core tests in both dtype modes;
no new test is needed for this rename. Commit on A6, then stop.
Docs to-do: unchanged; D-026 remains open.

## A6.4b2 review follow-up — Readability (2026-10-22)

Status: **implemented, ready for review**, authorized by
"check the review in ../review/slices and update",
on A6 at `401ce87`. Read all of `../review/slices/A6.4b2.md`. One follow-up
commit, then stop; no A6.4c or constructor rewrite.

- R1: give likelihood evaluation named private stages for unusable data,
  mode support and safe evaluation, leaving the entry point as orchestration.
- R2: move normalized-field checks into `current_field_issues` in
  `_data_validation.py`. Explain reconstruction bypassing constructors,
  normalized exact upper=lower, and unavailable raw indicator evidence;
  retain the existing rules and shared row_issues packing.
- R3/R4/R6: build one bad-row mask, use its complement for substitutions,
  name the broadcast exact mask separately, and read data.issues.invalid_rows
  rather than the private bitfield.
- R5: cite dtype promotion; explain the reference, data-time and output
  casts, including references/overrides with fixed precision. Add a fixed-float32 custom family
  test that demonstrates why the output cast remains necessary.
- SQ1: record the recipe-wording gap as D-026, with a proposed clarification;
  no spec or diagnostic behavior change without the maintainer's decision.
- Run core and core-matrix focused tests in both dtypes, then the full
  through-A5 gate and git diff checks. Overwrite REVIEW.md with actual
  results and outstanding SQ1. No public API, docs/ or spec-test changes.
- Docs to-do: clarify the recipe's numerical remainder after edited fields,
  pending SQ1 approval; existing likelihood usage docs remain a later step.

Evidence: 62 focused core cases pass in both dtypes; the fixed-precision
regression also passes after pinning its infinity limit. An in-memory
output-cast deletion fails that regression in x64 as expected. The full
through-A5 gate passes in both dtypes (own suites 648/654 passed with
existing 7/1 skips). REVIEW.md records exact results. R1–R6 are implemented;
D-026/SQ1 remains open pending the separate spec-clarification decision.
The source/test diff is 172 lines; files remain below 300 lines.

## Current step — A6.4b2 implementation

Status: **implemented, ready for review**, approved by "now do the next bit
of slice A6". Starting at
`523532a`; returned from main to the existing authorized A6 branch at the
same commit. Implement the private core and tests below, then commit and
stop. Reuse the existing parameter-isolation matrix by running it against
both the helper and the core; no copied family/test scaffolding. A6.4c is
still separate. No new public API or docs/ edits in this step.

## A6.4b1 review fix-up — Two nested levels and precise test diagnostics

Status: **implemented, ready for review**; authorized by the explicit review request, based on
`2e7ae42` on `A6`. One fix-up commit, then stop. No A6.4b2 work.

- Add `tests/unit/test_likelihood_parameters_depth.py` with a custom wrapper
  nested twice around a Weibull. A NaN deepest scale sits between valid
  scales; both wrapper levels have shared valid parameters. Compare valid-row
  values and first/second derivatives with the hand-derived isolated formula,
  eager and jit, naming transforms via pytest ids. Spec 7.1/7.3.
- N1: retain the explicit broadcast in the helper. Correct REVIEW.md:
  implicit where broadcasting makes deleting that line behaviorally equivalent;
  structural tests catch skipped masking of rebuilt leaves, not that deletion.
- N2: add transform names to the existing loop's assertions, including the
  jit-vmap check. Keep the existing case builder and coverage intact.
- Run focused tests in both dtypes, Ruff/mypy and the through-A5 gate.
  Record actual results in REVIEW.md; no spec-test/public-API/docs changes.
- Docs to-do: unchanged from A6.4b1; no narrative documentation in this fix-up.

Evidence: all 14 depth cases pass in each dtype. An in-memory mutation that
skips the deepest masking is caught by four reverse-mode cases per dtype.
The full through-A5 gate passes (own suites 586/592 passed, existing 7/1
skips). REVIEW.md records exact results and the corrected N1 claim. Library
source is unchanged; stop after this review fix-up commit.

Status: **A6.4b1 implemented; ready for review**.
The maintainer's "ok now do the actual next step" authorizes A6.4b1 on `A6`
at `9099310`. Finish the helper and its tests, commit and stop; A6.4b2/c
remain proposed. A6.4a rework is complete.
Initial alignment on maintainer-selected branch `A6` at `21c2632` read the complete normative
diff from `be7b253` to `cf4fef9`, current sections 7.1–7.3/12.1, decision 0045,
and the A6.4a review and designer request. Ran spec-since.sh dev --mark;
.spec-seen remains `cf4fef9`. After the maintainer's sync/merge, planning is
now based on `f07c560`, with independent snapshot `08d97a6` (revision 32).
Read all seven synced-file changes; collection is not conformance evidence.

Review B1 demonstrates that A6.4a's family-local guards isolate only zeros;
NaN, infinity and negative violations still contaminate reverse derivatives.
Revision 32 settles the mechanism: the likelihood broadcasts every parameter
leaf to the current distribution batch shape, then stops its gradients only
at entries in parameter_issues().invalid_entries, before any family code.
Nested leaves and component axes are included. Parameter values stay unchanged;
final invalid-row contributions are NaN. Valid rows retain values, gradients
and Hessians in forward and reverse modes. Invalid-row gradients remain
unspecified, and undeclared invalidity has no isolation guarantee.

This expressly authorizes the invalid-entry exception to the older
stop_gradient wording in CLAUDE.md; valid parameters and active valid
calculations still must not be stopped. No edit to that protected file or
new design decision is needed here. Direct-method policy remains section 4.5.
The superseded zero-only guards are now removed. Rework and core integration
are separate reviewable steps below; the initial alignment pass changed no
code. A6.4a rework and the separately approved A6.4b1 are complete.

The expanded A6.4a–c plan below replaces the obsolete revision-28 likelihood
outline. Revision 29 requires one array-returning function plus the base
method; no LikelihoodIssues, likelihood _with_info variant, extra result tuple
or new issue container. The A6.5 outline is updated to match this split.

D-023's policy was approved after the maintainer requested an explanation:
NaN only on unsupported rows, eagerly and under transformations, preserving
supported rows and guaranteed derivatives. Such rows acquire no data/parameter
issue. The section 7.1 recipe now checks mode support after the diagnostics,
before attributing a remaining supported NaN to normalization/numerical failure.
Spec commit `be7b253` and decision 0044 (`9cc2ab2`) record the agreement.
D-023's formal status stays with the maintainer; no implementation is claimed.

The work is split up front to keep each implementation/test/docstring diff
around 300 lines or less. Each part needs separate approval, a maintainer-
selected branch, one commit, an overwritten REVIEW.md and a stop. Stop for
review of A6.4b1; A6.4b2/c remain proposed. D-023 and review B1's
design decisions are settled by revisions 31 and 32 respectively. Do not
rewrite the published A6.4a commit or treat its checks as revision-32 evidence.

Standing maintainer authorization (2026-10-20): "exception approved in
general". Code steps may add the minimal generated API-reference entries
needed alongside new exports, keeping the existing export-coverage check
passing. This is a general exception to the working agreement's code-step
docs/ restriction; do not request it again for each symbol or step. Narrative
guides, worked examples and broader reference changes remain separate docs
steps. A6.4c's one function autodoc entry is covered by this approval.

## Synced revision-31/32 acceptance coverage (2026-10-21)

The sync removes the pending acceptance-test prerequisite. It does not
change the chosen mechanism, step order or implementation approval status.
tests/spec/ remains read-only; no acceptance tests are copied into our suite.

- `test_a6_likelihood_gradient_isolation.py`: 41 violation/family cases;
  finite valid-row values and derivatives equal evaluation alone as functions
  of the same flat parameter vector, including shared leaves and zero cross
  derivatives. Checks jacfwd/jacrev and all four second-derivative combinations
  (fwd/rev over fwd/rev), plus selected jit/method checks and hand-derived
  Exponential, shared-shape Weibull and nested shared-log-ratio anchors.
  The matrix covers five violation classes, component leaves, shared leaves,
  unguarded custom families and base/outer named conditions.
- `test_a6_unsupported_modes.py`: seven modes across flag/bounds construction,
  truncation and censoring. Requires NaN without a value-dependent exception
  in both entry points; supported neighbors retain values, gradients and
  Hessians with per-row/shared parameters, including jit/traced construction
  and vmap value checks. Diagnostics stay clear; triage uses unsupported_nan
  rather than the numerical remainder. Q-037's exception alternative is gone.
- `helpers.py`, `test_a6_likelihood_parameter_issues.py`, TESTMAP.md and
  API_ASSUMPTIONS.md now reflect revisions 31/32. No API change is needed.
- Retain own helper/evaluator checks for current shapes after reconstruction,
  smaller leaves, input immutability, unconstrained sibling masking, draws
  and per-leaf vmap; the new isolation module does not exercise these paths.
  Own valid-interior transformation and analytical checks remain required.
- A6.4a rework and A6.4b1/b2 still gate through A5 while the public entry
  points are absent. A6.4c must explicitly run both modules above in each
  dtype, then the complete through-A6 gate. No tests are edited or skipped
  to make an intermediate step pass.

## Historical revision-31 planning evidence, superseded by revision 32

Read-only probes actually run with UV_CACHE_DIR=/private/tmp/survjax-uv-cache,
JAX_ENABLE_X64=1 and 0, `uv run --locked python`; all exited 0. No likelihood
code was saved. For exact/exact/right rows at times [1,1,3], a candidate
composition of direct methods followed by parameter-issue NaN selection gave:

- Exponential rates [0.5,0,2]: valid-row forward Jacobian is [[1,0,0],[0,0,-3]],
  but reverse mode contains NaN in the invalid rate's column of both valid
  rows, instead of zero.
- Weibull shape [2,0,2], scale=2: valid-row reverse derivatives in the shape
  column for the invalid row are NaN. The same happens for scale [2,0,2]
  with shape=2. Valid diagonal scale derivatives are -0.75 and 2.25.

These are observed prerequisite gaps in zero-cotangent arithmetic, not spec
ambiguities or evidence that an evaluator exists. No pytest, runtime gate,
static checks or docs build was run for planning. No branch or commit created.
`git diff --check` exited 0 with no output after the planning edits. Only
PLAN.md and DEVIATIONS.md changed; the read-only design diff from .spec-seen
to design HEAD has no changes to SPECIFICATION.md or GOALS.md.

Approved-policy alignment: specification revision 31 is committed separately
in design (`be7b253`); supporting decision/index links are in `9cc2ab2`.
Ran ../scripts/spec-since.sh dev --mark, then its read-only check: "dev: spec
unchanged since be7b253". Executed the revised recipe in both dtype processes
against supplied synthetic likelihood outputs; data/parameter issues,
unsupported modes and the numerical remainder separate with (2,6) masks.
This checks only the recipe, not an unimplemented evaluator. No runtime gate
or docs build was run for the spec/metadata change; git diff --check is clean.

# A6.3 review fix-up — Mask batch shapes (historical)

Status: **implemented; ready for review**. Read
`../review/slices/A6.3.md` against clean `A6` at `b0b44a5`. Both should-fix
findings S1/S2 are valid under spec 4.7's requirement that every mask has
batch_shape, including when reconstruction changes current leaf shapes.
One small follow-up commit on the same maintainer-selected `A6` branch,
followed by an overwritten REVIEW.md and a stop. No A6.4 work.

- S1: in `src/survjax/distributions/_parameter_issues.py`, broadcast each own
  constrained field mask to the current batch_shape after component reduction
  and before storing it, matching the existing nested path. Preserve values,
  polarity, ordering and aggregate calculation. No API/signature/export change.
- In `tests/unit/test_parameter_issues.py`, add a two-positive-field regression
  whose reconstructed leaves have batch shapes (1,) and (3,). Check eager/jit
  masks have exactly batch_shape and hand-enumerated values. Add a per-leaf
  vmap in_axes regression (one parameter mapped, the other shared), asserting
  the full stacked shapes and values so union broadcasting cannot hide a
  malformed individual mask.
- S2: strengthen the existing smaller-nested-batch regression with explicit
  `.shape == (3,)` assertions for both the nested field and condition masks.
  Keep its value assertions. Demonstrate the regressions reject removal of
  own, nested-field and nested-condition broadcasting, each separately, using
  in-memory mutants without changing the checkout or independent tests.
- N1: leave nonboolean custom constraint results outside this fix-up. Such
  results violate the existing bool-array protocol; the review explicitly
  leaves additional validation to the maintainer. Do not silently cast them
  or introduce a new exception policy. N2: retain the private-module split;
  no unrelated base.py refactor. No deviation entry or status change needed.
- Files: the private module and own test file above, PLAN.md and REVIEW.md.
  No docs/, spec/, tests/spec/, dependency, constructor or direct-method edit.
  Existing A6 docs to-do remains sufficient for this shape-contract repair.
- Validation after approval: UV_CACHE_DIR=/private/tmp/survjax-uv-cache;
  separate JAX_ENABLE_X64=1/0 focused own diagnostics/declaration suites plus
  complete independent A6 constraints/parameter-issues modules with
  SPEC_THROUGH=A6; mutation checks in both modes; then
  `scripts/check.sh --through A5` and `git diff --check`. Record exact results
  and any failures in REVIEW.md. No docs build or complete A6 gate is claimed.

Read-only probes actually run: JAX_ENABLE_X64=1 and 0, `uv run --locked python`.
Both exited 0. Eager and jit report batch_shape (3,), own field shapes
rate=(1,), cap=(3,), invalid_entries=(3,). In both processes,
`np.testing.assert_array_equal(jnp.array(True), [True]*3)` passes, confirming
the missing shape assertion. No pytest suite or gate was rerun for this plan.

Implementation evidence: own field masks now broadcast after reduction; the
new reconstructed-own and per-leaf vmap cases and the strengthened nested
case assert exact shapes. Focused tests pass in both modes (176 passed,
1 optional skip); all three broadcast-removal mutants are rejected by shape
assertions in both modes. The full through-A5 gate passes. Exact commands,
counts and scope limits are in REVIEW.md. Stop after this fix-up commit.

# A6.3 planning — Base-owned parameter diagnostics (historical)

Status: **implemented; ready for review**. A6.2 is merged; planning
starts from clean `main` at `a715d3e`. The local spec marker remains
`fd150db3e82ad744277caecd1c68b2a524ad24df` (revision 30), and the synced
acceptance snapshot remains `f527f75325db5cac62c87c5bd28c7213cce51d8f`.
The expanded A6.3 step below supersedes its revision-28 outline. A6.1/A6.2
branch, approval and review statuses below are historical. A6.4/A6.5 still
need likelihood-plan alignment before approval; D-023 does not block A6.3.

Read current GOALS, spec 4.1–4.2/4.5/4.7/6.3/7.2/10.2/10.5/12.1–12.2,
the merged container and family registration/batch-shape code, existing own
declaration tests, and the synced parameter-issues acceptance module. This
initial planning pass changed only PLAN.md. The maintainer approved A6.3 and
selected branch `A6`, currently at `a715d3e`; use that branch for this step.
Implementation ends in one commit, REVIEW.md and a stop. No later step is
authorized.

Planning verification: `git diff --check` exited 0 with no output; only
PLAN.md changed. The read-only design diff from .spec-seen to design HEAD
contains no changes to SPECIFICATION.md or GOALS.md. No new spec ambiguity
was identified; existing deviation statuses are unchanged.

# A6.2 planning — ParameterIssues container and host checkpoint (historical)

Status: **implemented; ready for review** (2026-10-20). Planning starts
from clean `main` at `7274dd0`, after A6.1 merged. The maintainer approved the
plan including the minimal API-reference entry, and explicitly authorized
creating branch `A6.2`; that branch is now selected. The expanded A6.2
step below is the current proposal and supersedes its revision-28 outline.
Earlier A6.1 status and test-sync notes below are historical: the local
`tests/spec/SOURCE_COMMIT` now records `f527f75325db5cac62c87c5bd28c7213cce51d8f`,
and `.spec-seen` matches revision 30 at
`fd150db3e82ad744277caecd1c68b2a524ad24df`.

Read current GOALS and specification sections 4.7, 6.2–6.3, 10.2, 10.5 and
12.1–12.2, existing DataIssues/checkpoint code, public-export checks, and the
synced A6 parameter-diagnostic acceptance tests. The initial planning pass
changed only PLAN.md; approval covers A6.2 only. It does not authorize or
implement A6.3–A6.5. Their original
likelihood plans still need revision-29 alignment before approval.

Approval of A6.2 includes the single API-reference exception
specified below: the new top-level exception needs an autodoc entry to satisfy
the existing export-coverage test. The working agreement otherwise defers
docs/ edits to docs steps; the earlier A6.1 exception was specific to A6.1.

# Spec alignment — revision 30 (2026-10-20; historical record)

Status: read and marked in `b26a461`; the maintainer resumed implementation on
2026-10-20. **A6.1 plain-comparison follow-up implemented, ready for review.**
The alignment commit made no source/test edits. The separately approved
implementation follow-up is recorded below; later A6 steps remain unapproved.

Reviewed the full normative diff from marker `0683e35` (revision 28) through
`fd150db3e82ad744277caecd1c68b2a524ad24df` (revision 30), and current GOALS.

- Revision 29 (`be8010c`), sections 4.1–4.2, 6.1–6.3, 7.1, 11.4 and 12:
  add the base-owned dist.log_likelihood(data=None, *, time=None, event=None),
  identical to the top-level function in values and derivatives. Remove
  log_likelihood_with_info and LikelihoodIssues; explain NaN rows using
  data.issues and parameter_issues(), broadcast to the contribution shape.
  Preserve invalid-row NaNs, genuine zero likelihood -inf and gradient
  isolation. Later A6 likelihood plans below remain historical until replanned.
- Revision 30 (`fd150db`), section 4.7 and decision 0042: use ordinary JAX
  comparisons in the parameter dtype, converting bounds to that dtype.
  Backend-flushed subnormal values/bounds and values within rounding of a bound
  may be classified either way. Add no extra handling, including at zero.
  The table still specifies finite x>=0 for nonnegative; zero remains valid.
  This settles the outstanding boundary clarification and the normative
  questions behind D-024/D-025; formal entry closure remains the maintainer's.
- After the independent tests are synced and the maintainer resumes work,
  update the plain-comparison follow-up below to the synchronized acceptance
  suite: remove frexp and the own strict subnormal-sign guarantee, retain
  signed-zero inclusion, ordinary domains, exact simplex threshold and bound
  TypeError coverage. No implementation change is made now.
- Run ../scripts/spec-since.sh dev --mark and confirm no unread spec changes.
  The marker records reading, not conformance. Commit alignment bookkeeping
  on A6.1 so scripts/sync_spec_tests.sh can start from a clean dev checkout.
  No runtime checks are needed or claimed for this metadata-only step.

---

# A6 — Revision 28 parameter diagnostics and exact/right likelihood

Status: **A6.1 plain-comparison follow-up implemented, ready for review** (2026-10-19). This section supersedes
older forward plans for A6; the A5 records below remain historical.
Planning starts on clean `main` at `9623594`. The maintainer approved A6.1
and created branch `A6.1`. Only that code/test step is authorized; later steps
still need approval. The maintainer creates/selects each step branch; do not
commit on main. Every implementation step needs its own
approval, tests, commit, overwritten REVIEW.md and stop.

## A6.1 follow-up — Plain comparisons without frexp (approved 2026-10-20)

Requested after `142b778`: remove jnp.frexp for transparent comparisons.
The maintainer explicitly resumed this work after the revision-30 spec and
independent-test updates. Same `A6.1` branch, one small follow-up commit,
overwrite REVIEW.md and stop. Later A6 steps remain unapproved.

Revision-30 tests are at tester HEAD `f527f75`, with matching .spec-seen.
This branch still records tests/spec/SOURCE_COMMIT `1d87b72`; the maintainer
has a spectests-A6r1 branch, but its snapshot has not arrived on A6.1. Run the
updated constraint module directly from ../spectest without merging/syncing
or changing protected paths.

- Revision 30 confirms inclusive `nonnegative` as finite x>=0. Signed zeros,
  zero weights and unit_interval(0) remain valid. No clarification is pending.
- `src/survjax/constraints.py`: replace all four frexp expressions with plain
  JAX comparisons: finite x>=0 for nonnegative, finite x>bound for
  _GreaterThan, and direct lower/upper comparisons in _Interval. No zero-bound
  special case or replacement decomposition/bit-level workaround. Keep the
  exact simplex tolerance and finite-bound error normalization unchanged.
- `tests/unit/test_constraints.py`: replace the own mathematical-subnormal-sign
  assertion with explicit signed-zero boundary inclusion/exclusion cases.
  Keep the existing ordinary domain, nonfinite, transformation, simplex exact
  threshold and huge-bound coverage. Do not assert eager/jit identity for
  subnormal values when the chosen backend does not provide it.
- Public API: same signatures/exports. Values at subnormal boundaries inherit
  JAX/backend comparisons, per the maintainer's preference. No docs, spec,
  dependency or later-step changes. D-025 records the pre-revision-30 acceptance
  conflict; the revised tester module now avoids those subnormal assertions.
  Entry statuses remain the maintainer's responsibility.
- Validation: focused own constraints/declarations/import tests and the
  independent A6 constraint module in both dtype processes, then the full
  through-A5 gate, Ruff/mypy and git diff --check. Run against the newly synced
  revision-30 tester module and report all results; do not change/skip/xfail it.
  No docs build needed if docstrings and narrative docs remain unchanged.

Read-only probe (JAX_ENABLE_X64=1/0, uv run --locked python): on
[-nextafter(0,1), -0, 0, nextafter(0,1)], finite x>=0 gives
[True,True,True,True] eagerly/vmap and [False,True,True,True] under jit,
for both dtypes in this CPU environment. This is observed backend behavior,
not a verified cross-backend guarantee. The external independent
`test_unit_interval_includes_zero_and_one` expects False for the first input.
No pytest or gate was rerun during this planning check.

## A6.1 review fix-up — S1 and small nits (approved)

Reviewed `../review/slices/A6.1.md` against `98be832` on `A6.1`.
The maintainer approved this fix-up and chose JAX parameter-dtype/backend
comparison behavior for Q1. One small fix-up commit on this same branch,
then overwrite REVIEW.md and stop.

- S1: add one focused simplex tolerance regression to
  `tests/unit/test_constraints.py`, exercised in both separate dtype processes.
  With n=10, all entries zero except x[0]=1+k*eps, the sum is exact. Check
  k=30 and 40 accepted, k=41 and 50 rejected, eagerly/jit/vmap. This covers
  the review's 3n/5n cases and pins the inclusive 4n endpoint and its next
  representable neighbour. Expected masks come directly from spec 4.7's
  abs(sum-1)<=4*n*eps; no assertion tolerance or new tolerance constant.
  Demonstrate that the review's M2 tenfold-tolerance mutant fails this test
  in both modes, using an in-memory mutant rather than editing the checkout.
- N1: in `src/survjax/constraints.py`, translate overflow from the finite-bound
  check to TypeError with the existing bounds message. Add a small regression
  for 10**400 through all three public bound factories, including both
  interval endpoints. No other domain/value-policy change is proposed.
- N2: overwrite REVIEW.md after implementing the fix-up; state precisely that
  the callable dataclass instances are frozen, while module functions have
  mutable Python attributes. Do not change function representations.
- N3: no registration refactor in this fix-up. A6.3 still puts the new
  diagnostic composition in a separate private module as planned.
- Q1: record the maintainer's decision in D-024: follow JAX parameter-dtype
  and backend comparison behavior. Preserve the current implementation; no
  new mathematical Python-bound comparison or spec edit. Formal deviation
  closure stays with the maintainer.
- Public API touched: no signatures or exports; N1 changes the exception for
  an overflowing bound check. Files: constraints.py, test_constraints.py,
  PLAN.md and REVIEW.md; deviation bookkeeping in DEVIATIONS.md. No docs,
  independent-test, dependency or later-slice changes.
- Validation: focused own constraints/declarations/import tests plus the
  independent A6 constraint module directly from ../spectest in both modes;
  M2 rejection in both modes; scripts/check.sh --through A5; git diff --check.
  Record exact outputs, including mutation failures, in REVIEW.md. No docs
  rebuild is needed for this test/exception fix with unchanged docstrings.

Read-only review probes actually run with UV_CACHE_DIR=/private/tmp/survjax-uv-cache
and JAX_ENABLE_X64=1/0, using uv run --locked python: both modes report current
simplex masks [True, True, False, False] for k=[30,40,41,50], versus
[True, True, True, True] for an in-memory M2; 10**400 raises OverflowError.
No pytest suite or runtime gate was rerun while checking this review.
The maintainer's previous "ok make the minimal entry" explicitly authorized
A6.1's docs/api.md exception; the review's scope question needs no new approval.

Revision-29 follow-up: read `../scripts/spec-since.sh dev` and the complete
0683e35..be8010c normative diff. A6.1's sections 4.2/4.7 are unchanged.
Do not implement the stale revision-28 likelihood references below (A6.2 is
now separately replanned against revision 30):
remove LikelihoodIssues/log_likelihood_with_info and add base-owned
`dist.log_likelihood`, equal to the function in values/derivatives, in a
separate planning/alignment pass before the next step is approved. D-023 has
an open revision-29 note. Leave .spec-seen at revision 28 for this review
check; re-mark it as part of that explicit alignment pass.

## Contract and independent tests reviewed

- Spec revision 28, design commit `0683e358a312a8a1f8264af8f60eb61b0dba7a69`.
  Read the normative diff from the previous marker `b92334a`, current GOALS,
  and the relevant current sections 4–7, 10.2 and 12. The intervening
  `9d7483c` adds PDF helpers, with no runtime contract change.
- Revision 28 adds declared constraints, base-owned `parameter_issues()`,
  `parameter_conditions()`, `ParameterIssues`, `InvalidParameterError`,
  built-in declarations and likelihood invalid-parameter attribution. Direct
  methods retain their existing value policy; constructors do not check values
  or transform parameters. Slice A explicitly includes this contract.
- Reviewed the new independent A6 modules and Q-024/Q-034–Q-038 in
  `../spectest/tests/spec`, commits `b9bce65` and `e22001c` (tester HEAD).
  They cover constraints, parameter diagnostics, likelihood parameter issues,
  entry points, values, shapes, stored data issues and unsupported modes.
- Dev's `tests/spec/SOURCE_COMMIT` is still `1d87b72` (revision 27).
  The new tests have **not** been synced here. Alignment of `.spec-seen` is
  requested now; syncing requires the maintainer-created `spectests-*` branch
  and `scripts/sync_spec_tests.sh ../spectest`, separately from code work.
  Do not edit/copy tests into that protected directory while implementing.
- D-013's return-contract question is answered by revision 28. Leave its formal
  status to the maintainer. D-023 below records the remaining unsupported-mode
  contract gap identified by tester Q-037; approval of the early parameter
  steps need not wait for that likelihood decision.

## Step A6.1 — Constraint vocabulary and declarations

Spec: 4.1, 4.2, 4.7, 10.2, 12.1. Target about 300 changed lines including tests.

- Create `src/survjax/constraints.py`: typed pure callable constraints `real`,
  `positive`, `nonnegative`, `unit_interval`, `ordered`, `simplex`, and bound
  factories `greater_than`, `interval`, `closed_interval`. Use plain functions
  and small frozen callable dataclasses as needed, with `event_dim` metadata.
  Check finite Python bounds and ordering with TypeError for misuse. Preserve
  the specified NaN/infinity domains; simplex uses exactly `4*n*eps` of the
  parameter dtype, permits zeros, and reduces only the final axis.
- Extend `param(component_ndim=0, constraint=None)` in
  `src/survjax/distributions/base.py`: store constraints in field metadata;
  validate callable/event-axis compatibility at registration, without running
  constraints at construction. Add declaration-ordered read-only
  `Family.parameter_constraints`, excluding nested fields. Preserve constraint
  object identity and the existing PyTree leaves and construction behavior.
- Declare positive rate/shape/scale in `exponential.py` and `weibull.py`.
  Export the constraints module from `src/survjax/__init__.py`.
- Update the explicit export set in `tests/unit/test_import.py`.
- Add `tests/unit/test_constraints.py` and
  `tests/unit/test_parameter_constraints.py`: boundary/domain masks,
  malformed bounds/declarations, metadata order/immutability/identity,
  component compatibility, no added leaves or constructor checks, built-ins,
  and eager/jit/vmap. Check simplex softmax lengths through 1000 and rejection
  of a length-10 sum of 1.001. Expected masks follow the section 4.7 domains.
  Keep focused cases compact; split before exceeding the step budget.
- Maintainer approved the minimal constraints autodoc entry in docs/api.md
  during A6.1 so the existing public-export documentation check stays valid.
- Docs to-do: vocabulary and declaration recipe after code review.

## Step A6.2 — ParameterIssues container and host checkpoint

Status: **implemented, ready for review**. One container/checkpoint concept and one
implementation commit, targeting at most about 300 changed lines including
tests and review notes. Keep diagnostics.py below about 300 lines; if the
implementation or step budget grows beyond this, stop and propose a split.

Spec coverage: 4.7 (container fields, mask polarity/order and checkpoint),
6.2 (checkpoint behavior mirrored from DataIssues), 6.3 (PyTree result),
10.2 (module/export), 10.5 (public docstrings and minimal reference), and the
applicable container/checkpoint requirements of 12.1–12.2. No likelihood or
family aggregation is required for this step; D-023 does not block it.

### Files and public API

- `src/survjax/diagnostics.py`: add a small typed frozen `ParameterIssues`
  dataclass with four explicit fields: `fields: dict[str, jax.Array]`,
  `conditions: dict[str, jax.Array]`, `invalid_entries: jax.Array`, and
  `ok: jax.Array`. Document True-as-violation, batch-shaped masks and scalar
  ok. Like DataIssues, it stores supplied arrays; A6.3's producer will compute
  the union and scalar reduction. No new validation/normalization framework,
  derived properties, custom repr, or deep-immutability promise for the dicts.
- Register the container as a PyTree with explicit flatten/unflatten methods.
  Store the two ordered key tuples as static auxiliary data; expose their
  mask values, invalid_entries and ok as array leaves. Reconstruct dicts in
  that order, including dotted names, empty mappings and nonalphabetical
  keys. Do not return raw dict children to JAX's sorted-dict flattening.
  Unflatten only reconstructs: no array operations, host conversions or
  recomputation of ok, including when vmap stacks per-example scalar ok.
- Add `InvalidParameterError(ValueError)` and
  `ParameterIssues.raise_if_invalid(self) -> None`. Reject tracing before
  any NumPy conversion by checking all array leaves, even if ok is concrete;
  raise a plain ValueError containing "host-only" for both valid and invalid
  traced diagnostics. No callbacks. For concrete masks, reuse `_rule_counts`
  without changing DataIssues behavior, returning None when no violation
  fires and otherwise naming all violated fields and conditions, their counts
  and first batch-index tuples, with dotted names intact. Qualify the two
  categories so matching field/condition names remain distinguishable.
- `src/survjax/__init__.py`: re-export InvalidParameterError beside
  InvalidDataError and add it to __all__. ParameterIssues remains accessible
  as `survjax.diagnostics.ParameterIssues`, following DataIssues; no top-level
  container alias. Public docstrings cover attributes, return and exceptions.
- `tests/unit/test_import.py`: extend the explicit export assertion and
  check that the exception is the diagnostics class and subclasses ValueError.
- `tests/unit/test_parameter_issue_container.py`: create the focused tests
  below. No changes to tests/spec/, spec/, dependencies or check scripts.
- `docs/api.md`, **only if the requested exception is approved**: add one
  `autoexception:: survjax.InvalidParameterError` block. No narrative guide or
  family-method documentation in A6.2. Keep the existing export-coverage test.
- PLAN.md: record approval/completion; REVIEW.md: overwrite after the approved
  implementation with reading order, invariants, test/spec mapping, exact
  check output and uncertainties. Leave the A6.1 review intact while planning.

### Tests and evidence

Construct containers directly from hand-enumerated boolean masks, with explicit
expected invalid_entries/ok. This verifies storage and transformations;
calculation of those aggregates from family declarations belongs to A6.3.
Use exact boolean/structural assertions, without numerical tolerances.

- Spec 4.7/6.3: scalar, multidimensional and empty-mapping cases retain their
  masks, shapes and scalar ok through tree flatten/unflatten. Assert frozen
  field assignment and that leaves are arrays, while keys remain metadata.
- Spec 4.7/6.3: deliberately unsorted own/dotted field and condition keys
  retain insertion order and values after tree round-trip, jit and vmap.
  Jitted identity must preserve even a vmap result's vector ok; vmap of scalar
  examples returns the expected vector of per-example ok values.
- Spec 4.7/7.2: changing only same-shaped mask values reuses a jitted function
  (trace-count assertion), with fixed keys and structure and updated results.
- Spec 4.7/6.2: all-valid and empty mappings return normally; field-only,
  condition-only and overlapping violations raise InvalidParameterError.
  Assert all names/categories, counts and first indices: scalar `()`, vector
  `(i,)`, and multidimensional `(i, j)`. Expected counts/indices are enumerated
  in the tests, not computed by `_rule_counts` or the container under test.
- Spec 4.7/6.2/6.3: a concrete container returned by jit can be checked on the
  host. Calling the checkpoint inside jit/vmap with valid or invalid traced
  masks raises exactly plain ValueError with "host-only". Include a case
  with concrete ok and a traced mask to catch incomplete tracer guards.

### Validation and stop

After approval, use UV_CACHE_DIR=/private/tmp/survjax-uv-cache. Run focused
container/import/DataIssues regression tests in separate JAX_ENABLE_X64=1
and 0 processes. Run the independent
`tests/spec/test_a6_parameter_issues.py::test_invalid_parameter_error_is_a_public_value_error`
with SPEC_THROUGH=A6 in both modes; it is the independent A6.2-only check.
The other tests in that module call the A6.3 producer and are not yet an A6.2
gate. Do not edit, skip or xfail them.

Run `scripts/check.sh --through A5`: it includes Ruff format/lint, mypy src,
the entire own suite in both modes, source doctests and independent A0–A5
tests. If the minimal API-reference exception is approved, also run
`scripts/build_docs.sh` clean. Finish with `git diff --check`. Do not claim
the complete A6 gate, numerical derivatives, R fixtures, install checks or
benchmarks. No gradient tests are needed for a boolean-only result container.

Write REVIEW.md with exact commands, exit codes, pass/skip counts and any
failures, commit this step only on the maintainer's A6.2 branch with an
imperative subject and a body explaining why, then stop for review. Do not
start A6.3.

Planning verification actually run: `git diff --check` exited 0 with no
output. Only PLAN.md changed. No runtime tests, static checks or docs build
were run for this planning-only edit; no branch or commit was created.

Docs to-do (A6 docs step): document ParameterIssues fields and polarity,
ordered/dotted names, explicit host checkpoint and traced rejection, and
clarify that no reported issue is not a validity guarantee. Add the container
reference and examples using the A6.3 producer after that producer exists.

Implementation evidence (2026-10-20): 17 container cases added; focused own
regressions plus the independent exception check pass (32 tests per dtype).
`scripts/check.sh --through A5` passes both modes; `scripts/build_docs.sh`
passes after a sandbox port restriction required an escalated rerun. Exact
commands, counts, invariants and remaining scope are in REVIEW.md. No A6.3
producer or likelihood work started. Stop after the A6.2 implementation commit.

## Step A6.3 — Base-owned diagnostic composition

Status: **implemented, ready for review**. One concept: compose the declared
parameter diagnostics into the A6.2 result. Target about 250–300 changed
implementation/test/docstring lines: roughly 75 for the private module,
25 for the base hooks and 150–180 for focused own tests. If this cannot fit,
propose a split before coding beyond the approved scope. No registration
refactor or general diagnostic framework.

Spec: 4.1–4.2 (pure methods and extension hooks), 4.5 (direct values unchanged),
4.7 (composition, order, shapes and hook errors), 6.3 (PyTree results), 7.2
(compiled execution and reuse), and relevant 12.1–12.2 acceptance checks.

### Files and public API

- `src/survjax/distributions/base.py`: expose typed base methods
  `parameter_conditions() -> Mapping[str, jax.Array]` and
  `parameter_issues() -> ParameterIssues`, following the existing method
  attachment pattern where useful. The default condition hook returns a fresh
  empty mapping. Families override that hook; parameter_issues is base-owned.
  Include public docstrings describing violation polarity, exact batch shapes,
  ordering, return type and TypeError for malformed hook results. No new
  top-level exports, constructor arguments or distribution PyTree leaves.
- Create `src/survjax/distributions/_parameter_issues.py` for aggregation;
  use TYPE_CHECKING imports to avoid a base/private-module import cycle.
  The 449-line base module gets only the small hook integration, without
  moving existing registration, dtype, broadcasting or value-policy code.
- For each constrained own parameter in declaration order, evaluate its
  constraint, invert the valid-domain mask, and reduce exactly the remaining
  trailing component axes (`component_ndim - event_dim`) with any. Preserve
  all batch axes, including scalar batches. Use declaration metadata and
  current leaf shapes; never infer an undeclared parameter's validity domain.
- Recursively obtain each nested distribution's issues in field declaration
  order, prefix all field/condition names with `field_name + "."`, and
  broadcast masks to the current outer batch_shape. Retain deeper dotted
  prefixes. Fields contain own entries before nested entries; conditions
  likewise contain own hook entries before nested entries. Preserve false
  masks and unsorted insertion order so values cannot change the result tree.
- Validate the own condition hook using Mapping, key types/content, array
  dtype and exact shape only. Reject non-mappings, non-string/empty/dotted
  names, nonboolean values and any shape other than batch_shape with TypeError,
  also while tracing. Accept other Mapping implementations and preserve their
  iteration order. Do not coerce invalid values or broadcast condition masks.
  Fixed keys are the hook author's contract; add no key registry or cache.
- Start invalid_entries as a boolean all-False array of batch_shape, OR every
  field/condition mask into it, then compute scalar ok as negated any. Empty,
  undeclared and nested-only families use the same path. Return ParameterIssues
  directly; its reconstruction and explicit host checkpoint stay unchanged.
- Create `tests/unit/test_parameter_issues.py` for the evidence below.
  PLAN.md records approval/completion and docs to-do; overwrite REVIEW.md only
  after the implementation. No changes to diagnostics.py, docs/, spec/,
  tests/spec/, dependencies, public exports or gate scripts are planned.

### Tests and invariants

Use small frozen custom families and explicit expected boolean masks derived
from their inequalities. Do not copy the independent fixture families or
repeat A6.1's entire constraint-domain suite. Exact mask assertions need no
new tolerance; use existing tolerances for numerical identity comparisons.

- Spec 4.7: own fields with no constraint, scalar constraints on component
  arrays, and event_dim smaller than/equal to component_ndim. Hand-enumerate
  failures in different batch/component positions to distinguish any from all
  and catch accidental batch reduction. Verify union, boolean dtype and
  scalar ok, including no declarations, conditions only and scalar batches.
- Spec 4.7: nested and nested-only families, own-first unsorted key order in
  both mappings, dotted prefixes, nested conditions and outer broadcasting.
  Include reconstruction with a smaller nested batch shape to exercise
  diagnostic broadcasting independently of constructor broadcasting.
- Spec 4.7/6.3: compare eager, jit, vmap and traced-construction diagnostics
  against explicit masks; check per-example ok under vmap and the recomputed
  reduction after indexing. Verify keys/tree structure stay fixed between
  valid/invalid parameter values, with a one-trace compilation-reuse check.
  Exercise compiled diagnostics under a device-to-host transfer guard after
  input placement; host checkpoints remain outside that guard.
- Spec 4.7: malformed conditions raise TypeError eagerly and while tracing;
  include key, dtype and exact-shape errors and an accepted non-dict Mapping.
  The independent module supplies the complete eight-case error matrix.
- Spec 4.1/4.5: from_log overflow/underflow is diagnosed on the stored natural
  parameters. Constrained/undeclared twins retain matching direct values,
  including a finite formula at a declared violation; a hook that fails if
  called demonstrates constructors/direct methods never invoke diagnostics.
  Include an explicit host checkpoint on diagnostics from the new producer;
  the full host-error matrix is already covered by A6.2 and acceptance tests.

### Validation and stop

After approval use UV_CACHE_DIR=/private/tmp/survjax-uv-cache. In separate
JAX_ENABLE_X64=1 and 0 processes run focused own parameter-issue/container/
declaration tests, then with SPEC_THROUGH=A6 run the complete synced
`tests/spec/test_a6_constraints.py` and
`tests/spec/test_a6_parameter_issues.py`. Do not edit, skip or xfail acceptance
tests. NumPyro interoperability may retain its existing explicit skip when
that optional package is absent; do not install a new dependency.

Run `scripts/check.sh --through A5` for the protected-path guard, Ruff
format/lint, mypy src, the full own suite in both modes, source doctests and
independent A0–A5 checks. Finish with `git diff --check`. Boolean diagnostic
masks need no gradient suite. No docs build is planned because docs/ and
executable examples are unchanged. No complete through-A6 gate is claimed:
likelihood work and its independent tests remain A6.4/A6.5.

Write REVIEW.md with reading order, these invariants, test/spec mapping,
exact commands/counts/outputs and remaining concerns. Commit only on the
maintainer-selected `A6` branch with an imperative subject and a body explaining why,
then stop. No A6.4 work, unsupported-mode decision or deviation closure.

Docs to-do (A6 docs step): document both new methods and the hook contract,
including violations-as-True versus constraints' validity-as-True; component
reductions; own/nested ordering and dotted prefixes; fixed keys; no-issue
limitations; and custom-condition, nested-family and compiled-host-checkpoint
examples using the merged producer. No API reference exception is needed:
the export set stays unchanged.

Implementation evidence (2026-10-20): producer and hook integrated on `A6`;
299 production/test lines changed. The focused own/independent suite passes
in both dtypes (173 passed, 1 optional skip per mode). The through-A5 gate
passes; exact counts and commands are in REVIEW.md. No spec/acceptance-test
edits, new dependency or A6.4 work. Stop after this step's commit.

## Step A6.4a rework — Withdraw the superseded family-local guards

Status: **implemented, ready for review** (2026-10-21).
Replace the approach in `21c2632` with a follow-up commit on `A6`, preserving
history. Spec 4.5/7.1–7.3/12.1, revision 32; public API unchanged.

- Restore only `src/survjax/distributions/exponential.py` and `weibull.py`
  to their pre-A6.4a implementations at `3b0c3bf`. Retain all A5 guards for
  valid-parameter time boundaries. No parameter-domain repair or replacement
  direct-method policy is introduced.
- Remove `tests/unit/test_likelihood_operands.py`, whose test-only evaluator
  and zero-only isolation checks served the rejected mechanism. Replacement
  isolation tests belong with the actual parameter preparation in A6.4b1
  and its evaluator integration in A6.4b2. Existing direct-method, boundary
  and valid-interior derivative suites remain and must pass in both dtypes.
- Verify the two source files exactly match `3b0c3bf`, then run focused
  Exponential/Weibull and A6 constraint/parameter-issue suites, followed by
  scripts/check.sh --through A5. This rollback makes no isolation claim.
- Target about 200 changed source/test lines, mainly deletions. No new
  tests of the discarded behavior, helper module or export in this part.
- Docs to-do: no direct-method documentation change; explain the central
  likelihood mechanism with its implementation in later parts.

Implementation evidence: both distribution files exactly match `3b0c3bf`;
the obsolete operand-test module is removed. Focused own/A6 diagnostic
suites pass (176/177 passed, 2/1 existing skips), and the full through-A5
gate passes in both dtypes, including A5 boundary/derivative regressions.
194 source/test lines changed, mainly deletions. REVIEW.md records exact
commands and counts. No new helper, likelihood API or A6.4b1 work in this
step; stop after committing for review.

## Step A6.4b1 — Prepare parameter leaves for likelihood evaluation

Status: **implemented, ready for review** (2026-10-21).
One concept: section 7.1's parameter-gradient barrier. Spec 4.1/4.7/5/7.1–7.3/
12.1, revision 32. Target at most 300 implementation/test/docstring lines.

- Create private `src/survjax/_likelihood_parameters.py`: a typed helper
  accepts a distribution and its declared invalid-entry mask, reconstructs
  a temporary distribution without invoking constructors, and preserves
  types/static configuration. No public API, family-specific dispatch,
  custom derivative rule or registry change.
- Use field metadata to broadcast own and recursively nested parameter
  leaves to the outer distribution's current batch_shape plus each leaf's
  component_shape, including leaves rebuilt smaller than their siblings.
  Expand the outer union mask across component axes. Apply
  `where(mask, stop_gradient(leaf), leaf)` only after broadcasting; do not
  substitute numerical values, alter diagnostics or stop valid gradients.
- Add `tests/unit/test_likelihood_parameters.py`: assert unchanged leaf
  values (including NaN/inf), exact shapes/static fields and input immutability;
  compare selected valid contributions and their first/second derivatives
  with those rows evaluated alone, plus hand-derived Exponential checks.
  Exercise zero, negative, NaN and both infinities in built-ins, an unguarded
  custom family and a nested custom family, plus a named-condition violation.
- Cover component axes, an unconstrained sibling leaf in an invalid entry,
  outer-only condition masks reaching nested leaves, reconstructed smaller
  leaves, a shared leaf, per-leaf vmap and a draws batch shaped (S,1).
  Use eager/jit/vmap, forward/reverse Jacobians and all four compositions
  of forward/reverse differentiation to second order, matching revision-32
  acceptance coverage. Include a shared valid leaf beside a per-row violation.
  Make no assertion about invalid-row gradients. These helper-level checks
  are not yet evidence for the complete likelihood contract.
- Run focused tests and the through-A5 gate in both dtypes. Keep tolerances
  in tests/impl_tolerances.py; no tolerance loosening. If the full matrix
  exceeds the budget, replan before adding more code or tests.
- Docs to-do: declared constraints/conditions suffice for likelihood
  isolation; undeclared invalidity and direct-method calls are outside it.

Original implementation evidence (`2e7ae42`): the helper and 38 tests add 300 lines
(41 source, 259 tests). All 35 violation/family cases preserve valid-row
values and derivatives through second order, including jit/vmap and shared
leaves. Structural tests cover smaller rebuilt nested leaves, two component
axes, per-leaf vmap and draws columns. The full through-A5 gate passes in
both dtypes (own suites 572/578 passed, existing 7/1 skips). Initial test-hook
shape errors and their correction are recorded honestly in REVIEW.md.
No public likelihood or core integration yet; stop after this commit.

## Step A6.4b2 — Private exact/right likelihood core

Status: **implemented, ready for review**. D-023's policy is
approved in revision 31; parameter isolation follows revision 32.
One concept: evaluate prepared SurvivalData. Spec 3.2–3.3/4.2/4.5/5/6.2/7.1–7.3/
12.1–12.2. Target at most 300 implementation/test/docstring lines; no public
entry point until A6.4c, so no partially functional public export.

- Create `src/survjax/likelihood.py` with a typed private prepared-data core
  and only the private validation support it needs. Reuse row_issues from
  `_data_validation.py` for normalized endpoint/window checks, combining them
  with stored raw-input bits. Do not rebuild through a constructor, erase
  evidence, mutate the input, or recompute concrete metadata during evaluation.
  Newly invalid values after tree reconstruction must also produce NaN.
- Compute the trailing-axis broadcast of data shape and current batch_shape;
  incompatible shapes raise TypeError. Promote time arrays and temporary
  distribution leaves to the common data/parameter float dtype, including
  nested leaves; preserve original objects, component axes and static fields.
  Diagnostics and mode masks must explicitly have the contribution shape.
- Obtain parameter_issues().invalid_entries and apply A6.4b1's helper before
  calling reference_time, log_prob, log_survival or any other family code.
  Broadcast the same declared mask to the contribution shape for final NaN
  selection. Do not recompute it from repaired values or infer extra domains.
- Support only exact rows and non-exact rows with upper=+inf, all with
  left_trunc=0 and right_trunc=+inf. Determine this from current array fields;
  all-true traced metadata must not reject supported inputs. Derived-kind
  precedence treats (0,+inf] as right censoring, not unsupported left censoring.
- Dispatch through dist.log_prob and dist.log_survival, including custom
  overrides and generic defaults. Before inactive density evaluation use the
  stopped reference_time; before inactive set probability use the untruncated
  window's zero lower endpoint through the base boundary method. Substitute
  inactive/invalid/unsupported observation operands before operations. Valid
  parameters remain differentiable; the only stopped parameter entries are
  those already declared invalid under revision 32. No family-type dispatch.
- Exact rows contribute log density; right-censored rows log survival. At
  censoring time zero the base's log S(0) yields structural zero with zero
  parameter gradient. Preserve true zero density/survival as -inf. Do not
  compute truncation normalization, interval probabilities or Slice B pruning.
- Force final NaN on data-issue rows, parameter_issues().invalid_entries and
  unsupported rows per the approved D-023 policy. Do not invent parameter
  validity domains or assign unsupported observations a data/parameter issue.
  There is no numerical_failure mask or result container; unexplained family
  NaNs remain NaN. Invalid or infinite rows have no derivative guarantee.
- Add `tests/unit/test_likelihood_core.py`: hand-derived Exponential/Weibull
  values, exact/right mixtures, custom dispatch, scalar/paired/draw-column
  broadcasting with explicit shape assertions, dtype promotion, stored bits
  and changed normalized values after indexing/reconstruction, -inf zero mass,
  and zero-censoring value/gradient. Include supported neighbors of invalid
  or unsupported rows and the approved eager/jit unsupported-mode behavior.
- Verify eager/jit/vmap and valid-row values/Jacobians/Hessians against
  isolated valid rows through the actual core, reusing A6.4b1's family cases
  and violation matrix. Cover exact and right-censored contributions and
  all four second-derivative compositions. For unsupported rows separately,
  verify supported neighbors' gradients and Hessians with per-row and shared
  parameters; parameter-issue masking must not be used to flag unsupported
  modes. Run a valid mixed batch eagerly with
  jax_debug_nans. The public independent modules are not an A6.4b2 gate:
  the public API arrives
  in A6.4c. Do not edit or skip tests to accommodate this intermediate step.
- Docs to-do: supported subset, return shape/dtype, stored-issue propagation,
  zero-mass and zero-censoring behavior, D-023 policy and triage limitation.

Implementation evidence: a 69-line private evaluator, 219 lines of focused
core tests and 9 changed lines reusing the existing matrix (297 source/test
lines total). The 35 core isolation cases and 26 new core cases pass in
both dtypes through the full through-A5 gate; own suites report 647/653
passed with existing 7/1 skips. Mixed valid likelihoods pass eager
jax_debug_nans. REVIEW.md records exact counts and the corrected NumPy
expected-value expression. No public function/method/export or A6.4c work;
stop after this commit for review.

## Step A6.4c — Public function, method and shared input convention

Status: **implemented, ready for review**; see the current A6.4c entry at the top.
Follows reviewed A6.4b2. One integration
concept, targeting 150–250 changed implementation/test/docstring lines.
Spec 4.2/5/6.1–6.3/7.1–7.3/10.2/10.5/12.1–12.2.

- Add `log_likelihood(dist, data=None, *, time=None, event=None) -> jax.Array`
  to likelihood.py. One private resolver accepts prepared SurvivalData or
  constructs it with the pure checked raw time/event constructor, with both
  keywords required. Mixed/missing forms, positional raw arrays, constructor
  keywords and incompatible shapes raise TypeError, eagerly and under tracing.
  Preserve constructor dtype/warning behavior and raw invalid-indicator bits.
- Add the matching typed base method to distributions/base.py, delegating
  directly to the function. Use a local/TYPE_CHECKING import as needed to
  avoid a cycle. No alternative evaluator or copied constructor signature.
- Re-export exactly log_likelihood in src/survjax/__init__.py and extend
  tests/unit/test_import.py. No log_likelihood_with_info, LikelihoodIssues,
  reduction, weights, opt-out validation or convenience aliases.
- Add `tests/unit/test_likelihood_entry_points.py`: raw/prepared equality;
  malformed argument combinations and invalid raw values; traced construction;
  function/method values, gradients and Hessians against hand-derived smooth
  exponential expectations. Pin unchanged stored issue evidence through both
  forms. Public docstrings cover args, shapes, returns, errors and Slice A's
  unsupported-mode policy, with longer examples deferred to docs.
- Use the standing docs/api.md exception recorded at the top: one
  `autofunction:: survjax.log_likelihood` block, keeping the export-coverage
  test unchanged. No narrative guide or method-reference expansion here.
- The complete through-A6 gate runs every synced A6 likelihood module,
  including method, shapes, values, invalid-data, parameter-issue and
  entry-point tests in both dtypes.
  Explicitly run `SPEC_THROUGH=A6 uv run --locked --group test pytest
  tests/spec/test_a6_likelihood_gradient_isolation.py
  tests/spec/test_a6_unsupported_modes.py -q --tb=short` in separate
  JAX_ENABLE_X64=1/0 processes with the cache setting below. These public-API
  suites are available now; do not defer isolation to A6.5 or A9.
  Then run scripts/check.sh --through A6 and scripts/build_docs.sh.
  Record actual skips and failures without modifying the independent tests.
  A passing A6 gate is not completion of Slice A or the later A6.5 verification.
- Docs to-do: public usage, raw/prepared equivalence and errors, unsummed
  contributions, function/method interchangeability, diagnostics/triage and
  the D-023 limitation. Full API-method reference and examples follow merged
  code in the A6 docs steps.

### Validation and delivery for each A6.4 part

Use UV_CACHE_DIR=/private/tmp/survjax-uv-cache and separate JAX_ENABLE_X64=1/0
processes. Run focused own tests and applicable existing independent modules,
then scripts/check.sh --through A5 for A6.4a rework/b1/b2 and through A6 for A6.4c.
Those gates include Ruff format/lint, mypy src, the entire own suite, source
doctests and independent acceptance checks in both modes. Finish with
git diff --check. Keep tests/spec/ and the gate scripts unchanged.

Each approved part updates PLAN.md, overwrites REVIEW.md with reading order,
invariants, test/spec mapping, exact commands/counts/outputs and uncertainties,
then makes one imperative-subject commit (body explains why) on the
maintainer-selected step branch. Stop after every part; do not start the next
part without instruction. Replan before exceeding the code/test budget.

## Step A6.5 — Likelihood composition and derivative regressions

Status: **implemented, ready for review**, narrowed to the remaining finite-difference
checks in the current entry at the top. The reviewed A6.4 implementation and
passing suites already cover the other items from the original outline.
Do not repeat their matrices. Later A7–A9 and the remaining Slice A completion
evidence remain separate. Docs to-do: transformation/diagnostic composition
examples in tutorial style, mathematics followed by applied code.

## Validation and delivery for every approved code step

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run focused own and available
independent modules in separate JAX_ENABLE_X64=1 and 0 processes, Ruff format
and lint, mypy src, scripts/run_tests.sh and git diff --check. Run
`scripts/check.sh --through A5` while A6 is partial; after A6.3 run the focused
independent constraint/diagnostic modules with SPEC_THROUGH=A6. At completion
run `scripts/check.sh --through A6` without changing/skipping future tests.
Record exact counts, skips, failures and commands in REVIEW.md for each step.
Do not add NumPyro; its optional independent interoperability check may skip
when it is absent. No new runtime/test dependencies or later-slice features.

Narrative docs follow reviewed code in separate topic-sized docs steps
(at most 300 changed lines each): parameter declarations/conditions and host
checks first, likelihood usage/subset second. Edit docs/api.md and the relevant
existing guide or new focused guide; review need for playground updates.
Run scripts/build_docs.sh and inspect both dtype renders for each docs step.
Public API docstrings accompany code rather than waiting for docs.

Planning checks: read-only git/spec/test/source inspection only; no pytest,
lint, typing, runtime gate or docs build run. The marker records that the spec
was read, not implementation conformance. Mark using the existing hyphenated
`../scripts/spec-since.sh dev --mark` (the requested underscore path does not
exist). Leave this planning change uncommitted on main for maintainer review.

---

# A5.6 — Base boundary policy, then A5.d2 docs refresh

Status: code and review fix-up are merged by the maintainer. Step 2 is
implemented on the maintainer-created `docs-A5.6` branch on 2026-10-19,
based on `282dcca`, and ready for review. The maintainer explicitly requested
direct execution without a further planning/approval stop and asked to keep
the existing writing style. Only `docs/distributions.md` and `docs/api.md`
changed (188 documentation lines); the tutorial structure is retained.
Both dtype docs builds pass; exact checks and output inspection are in
REVIEW.md. No source/docstring, test, spec or dependency changes were needed.
Each step ends with its own commit, REVIEW.md and a stop.

Contract: specification revision 27, `.spec-seen` at `b92334a`. The newly synced
independent suite records author commit `1d87b726a15072f6e7223a60fe1afefa5bb999be`
(dev sync `645479e`, merged in `9146251`). The only later design/spec change,
`9d7483c`, adds PDF conversion helpers; it changes no normative requirement.
Leave the marker and spec sources unchanged during implementation.

## A5.6 review fix-up — F-1/F-2 and nits (2026-10-19)

Authorized by the maintainer's request to address `../review/slices/A5.6.md`,
including its explicit F-2 decision. Same step and `A5.6` branch; one fix-up
commit, then stop. No new public API, spec edit or narrative-doc work.

- F-1: add an own `log_cdf(+inf)` regression for Cure at fraction=1e-12,
  expecting analytical `log1p(-fraction)` with the existing zero-atol tail
  tolerances, in eager/jit/vmap and both dtype modes. Demonstrate that reviewer
  mutant M4 (naive log(1-exp)) fails this test in both modes, then restore.
- F-2: restrict hook-NaN precedence in `_survival_methods.py` to the four
  base-boundary methods and log density. Add a NaN-hook family with an
  analytic log-hazard override: log hazard/hazard keep log(rate)/rate at
  infinity; log survival/log density remain NaN. Run both modes and
  eager/jit/vmap. Keep every method's own NaN result intact.
- N-1: simplify Cure primitives in both own infinity modules and remove
  NaNLimit's dead primitive infinity branch; their hooks declare the limits.
- N-2: clarify the hook docstring's role and explicitly state that its default
  does not raise for parameter values; do not invent a raised exception.
- Files: these two own infinity test modules, `_survival_methods.py`,
  `base.py` (docstring only), PLAN.md and REVIEW.md. Cover spec 4.2/4.5/7.2/12.1.
  Correct the superseded every-method hook-NaN assertion below and in REVIEW.md.
  No independent-test or deviation-status changes.
- Checks: focused own infinity modules in both dtype processes, M4 regression
  evidence, `scripts/check.sh --through A5`, and `git diff --check`. Record
  exact outputs in REVIEW.md and stop before the narrative-docs step.

## Step 1 — Code, tests and public API docstrings (A5.6)

Outcome: every registered family receives the section 4.5 base boundary values,
even when its primitive has NaN values/gradients at zero or infinity. Improper
families declare their limit through a differentiable hook. Preserve interior
formulas, existing log-CDF tail precision, dtype/broadcasting rules, inherited
overrides, time-first grids and finite-entry derivatives in mixed batches.

Public addition: `SurvivalDistribution.log_survival_at_infinity(self) ->
jax.Array`, inherited by registered families and the built-ins. Its default is
`-inf`, exactly `batch_shape`, in the parameter dtype; component axes are
excluded. Custom overrides return log cure fractions with active parameter
derivatives. No new top-level export, constructor option or stored field.
Covered spec: 2, 3.3, 4.1–4.2, 4.5–4.6, 5, 7.1–7.3, 12.1–12.2.

Files and approach:

- `src/survjax/distributions/base.py`: add the typed default hook with summary,
  arguments, return-shape/dtype and error documentation. Derive its output dtype
  from normalized parameters, including nested-only families; do not use the
  transient time dtype. Keep it an ordinary overridable method, separate from
  the seven policy-wrapped pointwise methods. Pass the direct method's explicit
  name to the existing wrapper so base-owned boundaries also cover overrides.
- `src/survjax/distributions/_survival_methods.py`: replace the raw-primitive
  infinity probe/zero-limit mechanism with explicit boundary masks. For log S,
  S, log F and H, substitute at zero and infinity before calling any raw family
  formula, including inherited overrides. At zero select 0, 1, -inf, 0. At
  infinity derive values from the hook using exp, the existing safe log1mexp,
  or negation. Base constants must retain zero parameter gradients; hook values
  must retain their genuine derivatives. Substitute safe operands before
  boundary transformations as needed, never mask only their final results.
- For log density, substitute before evaluating at infinity and select -inf.
  For generic log hazard and generic hazard based on it, substitute at infinity
  and select NaN even for a nonzero cure limit. Analytic log-hazard overrides
  still receive zero/infinity, with matching hazard behavior; retain existing
  explicit hazard dispatch. At zero the generic density/log-hazard defaults
  must differentiate the raw primitive, preserving the specified exception;
  they acquire no value or parameter-gradient guarantee there.
- Use one stopped finite positive reference per direct call for valid
  parameters. Propagate NaNs from the family's substituted-input evaluation
  and, on base-supplied infinity values, from the hook, as section 4.5 requires.
  The maintainer's 2026-10-19 F-2 decision excludes analytic hazard overrides
  from hook-NaN precedence. Eliminate probes of the primitive at actual infinity. Preserve finite-support generic hazard behavior
  at finite times, negative/NaN-time policy and inherited raw-method unwrapping.
  No floors, clipping, inferred parameter validity or stopped active values.
- Update the base class, `log_survival`, `reference_time` and affected generic
  method docstrings in those same files. Audit `exponential.py`, `weibull.py`
  and `_grid_methods.py` docstrings; correct only obsolete ownership/gradient
  claims. Built-in formulas remain unless a demonstrated conformance failure
  requires a scoped correction. No `docs/` edits in this step.

Own tests, kept alongside the code in the same commit:

- Add `tests/unit/test_family_boundaries.py` with a small naive log-logistic
  primitive (log S=-softplus(beta*(log t-log alpha))) and an interior-only
  primitive (log S=-r*t*(t/t)). Pin base values at zero/infinity, gradients at
  zero, proper survival/log-CDF gradients at infinity, and mixed finite-entry
  derivatives. Cover eager/jit/vmap, shared parameters and time-first grids;
  use valid mixed batches under `jax_debug_nans` for methods with defined
  non-NaN outputs. Do not demand debug-nans success for deliberately undefined
  generic hazard outputs or invalid inputs.
- Extend the two existing infinity modules rather than duplicate their cure
  coverage: give Cure the explicit log-fraction hook. Check log S=log p,
  S=p, log F=log(1-p), H=-log p and their first/second parameter derivatives;
  rate derivatives at infinity are zero. Reuse the mixed-batch Hessian tests.
  Add hook-default scalar/batch/component/nested shape and dtype cases, plus
  finite overridden formulas whose boundary values must come from the base.
- Migrate obsolete own assertions in `test_family_infinity.py`,
  `test_family_mixed_infinity.py` and `test_family_direct.py`: survival/log-CDF/
  cumulative-hazard overrides no longer supply their own boundary values;
  raw primitive NaN at infinity no longer determines density NaN status;
  NaN precedence comes from substituted values/the hook. Move NaNLimit's
  declared NaN limit into its hook. Preserve positive-time override dispatch
  and analytical density/hazard boundary checks. Replace the public
  log-survival time-derivative-at-zero assertion with coverage of the generic
  density's raw-primitive differentiation; direct-time boundary derivatives
  are not part of section 7.3's guarantee.
- Expected values come from the algebra above or analytical identities stated
  in each test. Reuse `tests/impl_tolerances.py`; zero absolute tolerance for
  tiny log/derivative comparisons. New justified tolerances belong only there.
  Keep one behavior per test and do not edit, skip or xfail independent tests.

Independent evidence to require:

| Requirement | Independent coverage |
| --- | --- |
| Hook shape/dtype, proper and improper boundaries, NaN precedence, unsafe primitives and boundary gradients | `test_a4_base_boundaries.py` |
| Negative/NaN rows, raw defaults, finite zero survival, overrides, time dtypes | A4 policy, defaults, hazard, NaN-precedence and time-dtype modules |
| Built-in limits and gradients, unchanged closed forms | `test_a5_boundaries.py`, `test_a5_closed_forms.py`, `test_a5_derivatives.py` |
| Generic and built-in two-tail precision and derivatives (revision 25) | `test_a4_log_cdf_upper_tail.py`, `test_a5_log_cdf_tails.py` |
| Time-first boundary values, shapes and transformations | `test_a5_survival_grids.py` |

Validation: use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run new/changed
own tests and the focused boundary acceptance module in separate x64/x32
processes, then `scripts/check.sh --through A5` once after focused checks pass.
That gate includes Ruff, mypy, all own tests, source doctests and the complete
through-A5 independent suite in both modes. Finish with `git diff --check`.
Do not build narrative docs until step 2; their old boundary recipes are
intentionally refreshed there. Record exact failures/skips and any scope issue.

Aim for about 300 changed code/test/docstring lines; reuse current coverage and
remove superseded policy logic. If the cohesive change exceeds the working
agreement's size cap, stop and propose a code split before expanding it,
keeping all code steps before narrative docs. No new dependency, helper module,
registration protocol, custom derivative rule or later-slice feature is planned.
Overwrite REVIEW.md with reading order, safety/dispatch/gradient invariants,
spec-to-test mapping and exact checks. Commit on the maintainer's step branch,
then stop for review. Leave deviation statuses to the maintainer.

Docs to-do for step 2: replace primitive-owned boundary guidance with the new
base contract and hook; keep custom density/hazard responsibilities explicit.

## Step 2 — Narrative docs after code review (A5.d2)

Status: completed on `docs-A5.6` (2026-10-19), following the maintainer's
direct implementation request. Built-in/grids guides and the playground were
audited and needed no changes. The following scope describes the completed
boundary-contract documentation update.

One topic: explaining and using the revised family boundary contract. Public API
and runtime behavior stay as approved in step 1. Cover spec 2, 4.2, 4.5, 5,
7.2–7.3, 10.5 and the boundary evidence in 12.1–12.2.

- `docs/distributions.md`: update the edges table and custom-family checklist;
  describe finite-positive primitive inputs and base-supplied limits. Simplify
  the log-logistic example so naive log survival now works for survival/CDF/H
  boundaries. Retain the separate guarded analytic density/hazard extension
  example and its NaN handling. Replace the cure primitive's infinity branch
  with `log_survival_at_infinity() = log(fraction)`. Execute boundary values,
  parameter-gradient checks and mixed valid/NaN entries; explain the generic
  density/hazard exception at zero and generic NaN hazard at infinity.
- `docs/api.md`: list/document the optional hook and distinguish required
  primitive/reference methods from this override. The generated method help
  comes from step 1's docstrings.
- Audit `docs/builtin-families.md`, `docs/grids.md` and
  `docs/playground.ipynb`; edit only statements/examples that become wrong.
  Their built-in results and grid shapes should stay the same. Notebook source
  inspection during planning found no boundary/hook/primitive recipe to
  migrate, so no notebook edit is planned without a concrete need.

Keep this one topic within 300 changed lines; propose a split before exceeding
that cap. No unrelated tutorial/theme work or new notebook feature. Source
edits are limited to docstrings found to be wrong; a numerical defect goes
back to a separately reviewed code fix.

Run `scripts/build_docs.sh`: it executes and builds all docs afresh, with
warnings as errors, in separate x32/x64 processes. Inspect rendered guide/API
sections and their outputs in both modes. Run `git diff --check`; rerun a
runtime gate only if a source change or discovered failure warrants it.
Overwrite REVIEW.md with reading order, executed examples/spec mapping, exact
build results and remaining uncertainty. Commit on the maintainer's docs
branch, then stop. This step does not claim Slice A's remaining gates complete.

### Tutorial nuance follow-up (2026-10-19)

The maintainer directly requests clarifying that hazard/density endpoint
limits are optional, preserving their concurrent edit. On `docs-A5.6`, add
this explanation at "The hazard and density need more" and make nearby
instructions/checklist entries conditional on actually needing endpoint
values. Preserve the maintainer's "In many medical settings..." sentence in
the piecewise-hazard introduction. No executable examples or library code
change. Run `scripts/build_docs.sh` in both modes, inspect generated text,
update REVIEW.md, commit this small docs follow-up, then stop.

## Planning checks actually run

No source, test, spec or narrative-doc changes were made during planning.
Ran the following in separate processes with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`,
`SPEC_THROUGH=A5` and `JAX_ENABLE_X64=1` / `0`:

`uv run --locked --group test pytest tests/spec/test_a4_base_boundaries.py -q --tb=short`

- x64: exit 1, **32 failed, 52 passed in 2.11s**.
- x32: exit 1, **29 failed, 55 passed in 2.19s**.

Failures include the absent default hook, raw boundary NaNs, boundary gradients,
and improper-family generic hazard behavior. These are pre-implementation
results, not regressions caused by this plan. The full gate, tail modules and
docs builds were not run during planning. Historical records follow below.

# Spec alignment — revisions 25–27 (2026-10-18)

The maintainer requested reading `../scripts/spec-since.sh dev`, recording the
changes, then running it with `--mark`. This bookkeeping step changes only
`PLAN.md`, `DEVIATIONS.md`, `REVIEW.md` and `.spec-seen`. No public API or
runtime changes, no new tests. It supersedes the historical revision-24
alignment statement below; marking records review of the contract, not proof
that the implementation satisfies revision 27.

Reviewed committed range: `b917ebe` (revision 24) to
`b92334a1a4b584a7970039e321a6b10fe7c176ff` (revision 27).

- **Revision 25, `1e9011c` — sections 4.1–4.3, 7.2, 12.1:** specify the
  private two-branch `log1mexp` helper, with safe operands in both branches.
  Generic `log_cdf` and built-in standard branches use it; Exponential and
  Weibull retain the lower-tail calculation from `log H`. Both tails must
  keep relative precision over the specified representable ranges. Acceptance
  checks include mpmath upper tails, non-unit Weibull parameters, generic and
  grid paths, first/second derivatives around the branch split and in tails,
  and mixed boundary/interior calls under `jax_debug_nans`. Tiny log values
  and derivatives use zero absolute tolerance. The spec records the decisions
  behind D-021/D-022; formal deviation closure stays with the maintainer.
- **Revision 26, `0008c82` — sections 2, 4.2–4.5, 7.1, 8.2, 12:** the base
  supplies log survival, survival, log CDF and cumulative hazard at zero and
  infinity. Add the optional `log_survival_at_infinity()` extension method:
  default `-inf`, exactly `batch_shape`, parameter dtype; improper-family
  overrides retain parameter derivatives. These methods' family formulas and
  overrides need only handle finite positive times. Density/hazard limits
  remain family-supplied; generic autodiff at zero has no value/gradient
  guarantee, and generic log hazard at infinity is NaN. Preserve NaN
  precedence from substituted evaluations and the infinity hook. Base boundary
  gradients now cover every registered family, including naive primitives
  with unsafe boundary derivatives. Tests must cover proper and improper
  custom families. Section 2 also clarifies that sets closed at infinity
  include mass at infinity; future evaluator/solver work uses the declared
  limit. HazardFamily, solver and `check_family` changes stay in later slices.
- **Revision 27, `b92334a` — sections 4.5, 12.2:** valid-parameter substituted
  inputs must be finite and positive. Slice A must honor infinity-hook
  overrides in the base, including improper-family values and gradients;
  supplying only a hard-coded proper-family default is insufficient.

Implementation follow-up requires a separate approved plan on a
maintainer-created step branch. `rg -n 'log_survival_at_infinity' src
tests/unit docs` found no matches (exit 1), so the new hook is not present in
those files. No conformance gate was run in this bookkeeping step.

Docs to-do: refresh the extension recipe, boundary table, API reference and
playground for base-supplied limits and the infinity hook in a separate docs
step after implementation.

The script warns about uncommitted spec changes because design has an
untracked `spec/SPECIFICATION.pdf`; the tracked specification matches HEAD.
Do not incorporate or commit that generated file. The dev checkout is on
`main`; AGENTS.md prohibits commits there and reserves branch creation for
the maintainer. Leave this bookkeeping ready for the step-branch commit.

# A5 — Exponential, Weibull and time-first grids

Status: A5.1 (`3f50962`) was approved by the reviewer and merged into main by
the maintainer. A5.2 (`a58c57c`) received a merge verdict on 2026-10-12.
The F-2/F-3 review nits are corrected; A5.2b (`71403c3`) and its review fix-up
(`63874e8`) are merged into main. A5.3 (`cc2f85d`) and its broadcasting fix-up
(`cd3f213`) are present on main. A5.4 and both log-CDF review fix-ups
(`c477d31`, `4f867cb`) are merged into main. The maintainer selected branch
`A5.5`; the approved implementation from `861ac71` passes the full through-A5
gate in both dtype modes on 2026-10-15 and awaits review.
The spec and `.spec-seen` agree at revision 24 (`b917ebe`). The synchronized
tests correct D-020's density expectation. The maintainer's D-022 upper-tail
note is committed in `861ac71`; its numerical fix is implemented below as a
separate step on the same `A5.5` branch and awaits review.
A5 docs follow the remaining code work.
The original planning started from a
clean `A5` checkout at `3f50962`, then the main tip. No branch was created
or switched by this planning request; implementation uses the maintainer's
selected step branch after approval. Based on
specification revision 22, design commit `9323cb5`, and the independent tester
commit `7f1d6df`. Implementation starts on the maintainer-selected branch `A5`
at `89cb9f4`, after the maintainer aligned the marker and synchronized tests.
Commit this approved planning/deviation record separately from the A5.1
implementation so each diff remains reviewable. The A4 plans below are history.

## What changed after the spec/test review

Read the complete revision-21-to-22 diff, the six new A5 test modules,
Q-027–Q-033 and their TESTMAP entries, plus the three new A4 NaN-precedence
cases. Revision 22 replaces the earlier log-leaf proposal: `from_log` simply
exponentiates inputs and stores the resulting natural parameters. The previous
requirement to retain extreme log inputs is removed. No log storage, private
representation flag or registration redesign is needed (D-019).

Other settled requirements:

- Built-ins are registered frozen subclasses of `SurvivalDistribution`, with
  natural `rate` or `shape`/`scale` as their only dynamic leaves (4.1–4.2).
- `reference_time()` returns an array of exactly `batch_shape`, in the parameter
  dtype, positive and finite for valid parameters; invalid cases are unspecified.
- `log_prob(+inf)` is NaN when the family's raw `log_survival(+inf)` is NaN;
  otherwise it is -inf, including when a density formula yields inf-inf.
  Testing the raw density for NaN alone cannot distinguish these cases (4.5).
- At zero and infinity, finite built-in values have the actual parameter
  derivatives of their limits. Derivatives across a discontinuity are excluded:
  do not require the Weibull shape derivative of log density at zero and k=1.
  The scale derivative there is required. Custom-family guarantees depend on
  the author's primitive; the extension recipe must explain this (D-016/4.5).
- Revision 24 expands the base surface to seven methods: `survival_grid(t)`,
  `log_survival_grid(t)`, `log_prob_grid(t)`, `log_cdf_grid(t)`,
  `cumulative_hazard_grid(t)`, `log_hazard_grid(t)` and `hazard_grid(t)`.
  Every registered family inherits them.
  The first positional argument is named `t` and also accepts `t=...`; it must
  be one-dimensional and follows the direct-time dtype rules (5/8.1).
- The zero-time grid example is self-contained: Weibull shape 1, scale 10.
  The section 11.1 fixed vector additionally gives group scales
  `10 * exp([0, -0.5, -1])`; it requires no R fixture.

These decisions supersede the old pending choices; no new spec ambiguity was
identified in this review. Record revision-22 evidence in DEVIATIONS.md and
leave formal deviation closure to the maintainer.

## Scope, staging and independent-test availability

Public additions: `survjax.Exponential(rate=...)`,
`survjax.Weibull(shape=..., scale=...)`, keyword-only `from_log` classmethods,
all seven direct methods on both families, and their seven grid base methods.
Covered spec: 3.3, 4.1–4.2, 4.5–4.6, 5, 7.2–7.3, 8.1 (grids only),
10.2–10.5 and applicable family/boundary requirements of 12.1–12.2.
No likelihoods, interval/truncation methods, parameter diagnostics (D-013),
summaries, sampling, wrappers, `check_family` or new dependencies. R fixtures,
installation checks and the CPU likelihood baseline remain later Slice A work.

There are five commit-sized steps below, separating the newly specified policy
correction from the generic gradient work. Aim for roughly 300 changed
source/test/docstring lines per code step, at most 300; split before expanding
if necessary. Each step requires its own plan approval, tests and public API
docstrings with the code, one commit and an
updated REVIEW.md, then a stop. This proposal does not authorize proceeding
through all five steps. Use the maintainer-selected branch; never commit on main.

From A5.2b onward, narrative docs follow specification revision 23 (decision
0038): code steps don't edit `docs/`; record their documentation needs under
"A5 docs" below. Docs steps start after the last A5 code step merges, and A6
starts only after the last A5 docs step merges. Earlier A5 docs are drafts;
the A5.1/A5.2 instructions and verification evidence below retain their history.

The maintainer aligned `.spec-seen` to `9323cb5` in `96fc32d` and synchronized
the independent suite from `7f1d6df` in `89cb9f4` before implementation. The
current through-A4 gate includes the three revision-22 regressions. The
revision-24 suite now collects 438 A5 cases, including 116 in the grid module.
Both independent suites and the alignment marker remain untouched by the
implementation agent. Never edit, skip or xfail acceptance tests to obtain a pass.

## A5.1 — Correct NaN precedence for density at infinity

- Change `src/survjax/distributions/_survival_methods.py`'s density policy to
  consult the raw primitive at +inf. Return NaN iff that primitive value is NaN;
  otherwise preserve the unconditional -inf density limit. Leave negative/NaN
  time handling, normal broadcasting, override dispatch and non-density methods
  unchanged. No exports, signature changes or generic validity-domain inference.
- Add `tests/unit/test_family_infinity.py`: constant-hazard and power custom
  families with NaN parameters; mixed NaN/valid entries where the valid generic
  density formula is NaN at infinity; an explicit density override whose result
  differs from the primitive's NaN status. Exercise eager/jit/vmap. Expected
  constants follow the revision-22 policy, not the implementation.
- Preserve the existing one-reference-call invariant. Keep any extra primitive
  evaluation restricted to the density rule; do not double-wrap methods.
- Update the infinity rule in `docs/distributions.md` and relevant base-method
  docstrings. Cover 4.2/4.5/7.2. Run the three new acceptance tests in
  `test_a4_nan_precedence.py`, then the complete through-A4 regression gate
  once the suite is synced. Stop with REVIEW.md and the step commit.

## A5.2 — Generic finite boundary gradients and extension recipe (D-016)

Outcome: generic survival and log-CDF give their correct finite values and
parameter gradients at +inf for proper primitives such as `log S=-r*t`, while
preserving finite, parameter-dependent limits of custom cure families. Public
API, exports, constructors, dtypes and broadcasting are unchanged. Covered spec:
4.2, 4.5, 5, 7.2–7.3, 10.5 and relevant transformations in 12.1. No built-ins,
grids, density/hazard changes, dependency additions or custom derivative rules.

### Numerical change — `src/survjax/distributions/_survival_methods.py`

- Integrate the guard in the existing policy wrapper, where its already-computed
  reference input can be reused. This refines the earlier defaults-local sketch:
  calling `reference_time()` again inside each default would break the existing
  one-reference-call invariant. Keep the change in this module; registration
  and `base.py` need no change.
- After resolving the existing raw-method marker, recognize only the generic
  `survival` and `log_cdf` functions using function identity, a static Python
  decision. Explicit and inherited family overrides retain their dispatch.
  No new marker protocol, cache, parameter specialization or wrapper stack.
- Retain the single stopped reference input in a named local. Evaluate raw
  `log_survival` on the current safe times to obtain the true limit. The guard
  is elementwise: original time is positive infinity AND raw log survival is
  negative infinity. Use this result only as a Boolean selection condition.
- Before the default's numerical operations, replace those guarded times with
  the stopped reference input. Evaluate the default on these safe times, then
  select the exact finite boundary value 0 for survival or log-CDF. The raw
  primitive has still been called at the actual infinity to establish its limit.
  Stop only the substituted reference, never real parameters or active results.
- Leave finite raw limits at +inf active: if S(+inf)=p, gradients in p must
  survive. NaN raw limits must remain NaN. The guard must not activate at finite
  times, even when linear survival underflows or log survival overflows to -inf;
  no precision loss is reclassified as zero mass. Preserve zero-time behavior,
  A5.1's density rule and negative/NaN-time policy.
- A second primitive evaluation is expected for these generic methods, including
  finite calls. Record that cost without claiming a measured speedup or adding
  an optimization layer. Custom authors remain responsible for primitive
  differentiability and safe behavior at their chosen reference inputs.

### Own tests — `tests/unit/test_family_infinity.py`

Extend the existing 111-line module, keeping it below roughly 300 lines.
Expected values/derivatives come from algebra written in each test; run both
dtype modes with `tests/impl_tolerances.py`'s existing appropriate tolerances.
Add a scale only with a separate numerical justification in that file.

- Existing Constant and Quadratic primitives: survival(+inf)=0 and
  log-CDF(+inf)=0; first parameter derivatives are exactly 0 in forward and
  reverse modes, eagerly and composed with jit/vmap.
- Scalar rate with times [1,+inf]: sum survival=e**(-r), derivative=-e**(-r)
  and second derivative=e**(-r). Sum log-CDF=log(1-e**(-r)), derivative
  1/expm1(r) and second derivative=-e**r/expm1(r)**2. Cover per-entry parameter
  batches as well, so an inactive infinity entry cannot poison active entries.
- A guarded cure primitive S(t)=p+(1-p)*e**(-r*t), with 0<p<1 and r>0:
  at infinity survival=p and log-CDF=log(1-p). Their p derivatives are 1 and
  -1/(1-p), and their r derivatives are 0. Preserve these under jit/vmap and
  check a finite-time case; the primitive itself guards its infinity operands.
- NaN parameter entries beside valid entries at +inf retain NaNs only in their
  positions; finite-time underflow and active lower-tail values retain current
  behavior. Do not assert derivative guarantees for invalid parameters.
- Explicit/inherited survival and log-CDF overrides remain honored. Extend
  reference-call coverage to these methods at infinity, checking one call.
  Existing finite/negative-time, zero-CDF, primitive time-derivative, density
  and second-order interior tests remain regressions, not targets for alteration.
- Demonstrate before/after evidence with the new infinity-gradient cases.
  After the fix, temporarily remove only the safe-time substitution: the
  constant-family gradient cases must detect NaN despite final output masking.
  Restore the source before final checks; never mutate independent tests.

### Documentation — `docs/distributions.md`

Preserve the maintainer's tutorial. Add a short executable guarded-boundary
extension recipe and parameter-gradient assertions, explain the proper-family
zero limit and the parameter-dependent cure limit, and state the author's
responsibility for safe primitive operands/NaN propagation. Update method
docstrings with the same scope. Amend PiecewiseHazard's `reference_time()` to
return exactly `batch_shape` in `rates.dtype`, and revise the checklist's
current "broadcastable" promise to the revision-22 exact-shape contract.
No new page, notebook or later-slice feature is needed.

The separately requested Gompertz notation correction is applied during this
planning turn: display `-(b/c)*(e**(c*t)-1)` mathematically, with `jnp.expm1`
still explained in prose and used by executable code. It changes no computation.
Planning verification: `UV_CACHE_DIR=/private/tmp/survjax-uv-cache
scripts/build_docs.sh` exited 0, reporting `build succeeded` for x32 and x64
and executing the existing examples afresh. `git diff --check` passed with no
output. No numerical implementation tests were run during this planning turn.

### Validation, commit and stop

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run the focused own suite in
separate x64/x32 processes, then `scripts/check.sh --through A4`,
`scripts/build_docs.sh` (both modes), and `git diff --check`. The independent
author does not guarantee arbitrary custom-family boundary gradients; the new
own tests supply that evidence. All 370 A5 cases remain outside this step's
gate, since built-ins/grids are not yet implemented. Do not edit or skip them.

Aim for roughly 300 changed source/test/doc lines; split before expanding if
necessary. Overwrite REVIEW.md with reading order, safe-input/dispatch invariants,
spec-to-test mapping, mutation evidence, exact results/skips and the unmeasured
extra-evaluation cost. Leave D-016 open for maintainer closure. Commit the
approved step on the maintainer-selected branch, then stop before A5.3.

## A5.2b — Isolate finite-entry derivatives from infinity entries (review F-1)

### Review fix-up (2026-10-13)

Authorized by the maintainer's request to implement the A5.2b review comments.
Same step and branch; no new public API or numerical policy.

- `tests/unit/test_family_mixed_infinity.py`: add a density override to the
  existing cure family whose formula is -r*t. Pin [1,+inf] values [-2,-inf]
  at r=2, p=0.25, and the finite-component rate derivative -1 in forward and
  reverse modes, including jit/vmap. Spec 4.5/7.2; expected numbers are algebra.
- `_survival_methods.py`: consolidate each generic method's guard flag and
  boundary value in one static conditional chain; retain the hazard check
  against its underlying log-hazard and rename that local `log_hazard_method`.
  No registration-table redesign or new configuration layer is needed.
- F-3 remains a structural argument: `isposinf(time)` explicitly excludes
  finite times. Do not assert unsupported NaN-gradient behavior to test it.
- Demonstrate that reviewer mutant M1 (density substitutes only zero_limit)
  fails the new test in both dtype modes; restore before the full through-A4
  gate. Run focused tests, Ruff/mypy through that gate and `git diff --check`.
  Overwrite REVIEW.md, commit one review fix-up, then stop. Full docs stay
  deferred to the end of A5; spec and independent tests remain read-only.

Maintainer update (2026-10-13): code steps include tests and public API
docstrings, with no `docs/` edits. Narrative docs and their builds are separate
A5.d1, A5.d2, … steps after the last A5 code step merges; see "A5 docs" below.

Status: implemented, reviewed and merged with its fix-up on 2026-10-13.
This is a separate numerical
step before A5.3, on the maintainer-selected branch. It addresses the review's
shared-parameter mixed-batch leak; it does not promise derivatives of infinite
or NaN outputs. Covered spec: 4.2, 4.5 and 7.2–7.3; the extension recipe in
10.5 is updated and verified at the end of A5.
No public API, exports, dependencies, built-ins, grids or spec-test changes.

Files and numerical approach:

- `src/survjax/distributions/_survival_methods.py`: retain the single stopped
  reference and probe the primitive at the original policy-safe times before
  any infinity substitution. Use probe results only as Boolean conditions.
  For density, substitute all positive-infinity times before running the raw
  density, including overrides. Select -inf or NaN using the original probe's
  NaN status; a substituted reference must not determine that status.
- For the policy-wrapped log-survival primitive and the generic cumulative
  hazard, extend the existing zero-limit guard (positive infinite time AND
  negative infinite raw log survival); substitute before evaluation and select
  -inf/+inf respectively. Finite cure limits stay active. Cumulative-hazard
  overrides retain their own dispatch.
- Guard generic log-hazard with the same zero-limit condition and select its
  existing NaN boundary value. Guard generic hazard only when its underlying
  log-hazard is also generic. Analytic log-hazard/hazard overrides, including
  inherited overrides, must preserve their infinity values and derivatives.
  This prevents the generic hazard default from reintroducing the same leak.
- `src/survjax/distributions/base.py`: if needed, pass an explicit private
  primitive flag from registration to the existing policy wrapper. The
  log-survival primitive has no generic function identity; do not infer its
  role from an author's function name or introduce a new marker protocol.
  Other generic defaults continue to use static raw-function identity.
- Preserve NaN precedence, negative/NaN and zero-time behavior, finite tails,
  and reference-call count. Stop only substituted inputs, never active values
  or parameters. Reuse the probe for density instead of probing at the newly
  substituted time. Record extra evaluation cost without optimizing it here.

Tests: create `tests/unit/test_family_mixed_infinity.py` to keep the existing
infinity module below 300 lines. Use small explicit constant/quadratic/cure
families and hand-derived expectations, with existing test tolerances.

- Shared scalar rate r=2 at times [1,+inf]: finite-entry derivatives are -1
  for log survival, +1 for cumulative hazard, 1/r-1 for log density, 1/r for
  log hazard and 1 (constant) or 2 (quadratic) for hazard. Check the finite
  output component with jacfwd/jacrev, eager/jit and jit/vmap compositions;
  do not differentiate a sum containing a nonfinite output.
- Check second derivatives of the finite output component against algebra:
  log density/log hazard -1/r**2, other listed methods zero. Independent
  parameter batches must keep finite-entry derivatives separate too.
- Preserve A5.1 density values for mixed NaN/valid parameters, including
  finite and NaN overrides. Preserve nonzero fraction derivatives of finite
  log-survival/cumulative-hazard cure limits and analytic hazard overrides.
- Pin one reference call and unchanged finite values. Demonstrate pre-fix
  failures in both dtype modes, then remove only the new input substitutions
  as a mutation: finite-component reverse derivatives must fail again.

Update function docstrings with mixed-entry isolation and its limits, including
that derivatives of nonfinite outputs and arbitrary unsafe custom overrides
are not guaranteed. Do not update full docs or run `scripts/build_docs.sh`
during A5.2b. Record this behavior under "A5 docs" for the docs steps to cover
and verify with the full documentation build in both dtype modes.
`DEVIATIONS.md`: extend D-016 evidence without resolving it.
Overwrite `REVIEW.md` with invariants, test/spec mapping and exact check results.
Run focused tests in both processes, `scripts/check.sh --through A4` and
`git diff --check`. Record the deferred documentation work in REVIEW.md.
Aim for about 300 source/test/docstring lines, split if needed, commit one
numerical step, then stop before A5.3.

## A5.3 — Exponential

### Reviewer broadcasting fix-up

Authorized by the maintainer's request to implement the reviewer suggestion.
On the existing `a5.3-exponential` branch, update `log_hazard` and `hazard`
in `src/survjax/distributions/exponential.py` to compute the broadcast shape
of `t` and `self.batch_shape` explicitly before broadcasting their values.
No public API or time-policy change; this removes the raw methods' dependency
on the registration wrapper pre-broadcasting time. Covers spec 4.1/4.5/5.

Add an own regression in `tests/unit/test_exponential.py` for raw and wrapped
hazard methods with scalar time and singleton-axis time arrays against batched
rates, eagerly and with jit/vmap. Expected values are rate and log(rate).
Demonstrate the new regression fails before the source change. Run the focused
suite in both dtype modes, the through-A4 gate, the existing Exponential-only
A5 acceptance selection, and `git diff --check`. Overwrite REVIEW.md, commit
one fix-up and stop. Full docs remain deferred to the end of A5.

Status: approved and implemented on 2026-10-13 from main at `7a2fef6`, on
`a5.3-exponential`; awaiting review. This is the code step after merged A5.2b,
before Weibull (A5.4) and grids (A5.5). Approved planning and spec-marker
alignment were committed separately in `9337969` before implementation.

Spec: 3.3, 4.1–4.2, 4.5–4.6, 5 (pointwise broadcasting only), 7.2–7.3,
10.2–10.5 and the applicable 12.1 checks. Revision 23 changes documentation
scheduling only; the synced revision-22 numerical acceptance tests remain
applicable. `.spec-seen` was aligned from `9323cb5` to `a212ddd` after reading
the diff, in the separate bookkeeping commit. Spec tests and their revision
labels remain unchanged.

### Files and public API

- Create `src/survjax/distributions/exponential.py`: registered frozen dataclass
  subclass of `SurvivalDistribution`, with `rate: jax.Array = param()` as its
  only leaf. Constructor is keyword-only `Exponential(rate=...)`; no default,
  dtype keyword, value validation, log-storage flag or new dependency.
- Add `Exponential` imports/exports in `src/survjax/distributions/__init__.py`
  and `src/survjax/__init__.py`; update the exact public export assertion in
  `tests/unit/test_import.py`.
- Add typed `from_log(*, log_rate)` returning the same class. Reuse
  `_dtypes.normalize_parameters` before exponentiation so boolean/complex/
  string/object inputs cannot become valid parameters through `exp`. Keep
  weak/default dtype promotion and the float64-to-float32 warning policy.
  Store only `exp(log_rate)`; overflow yields an infinite natural rate.
- Implement `log_survival=-r*t`, `cumulative_hazard=r*t`,
  `log_prob=log(r)-r*t`, `log_hazard=log(r)` and `hazard=r` analytically.
  Inherit `survival` and stable `log_cdf` from the existing base. Preserve
  pointwise output broadcasting even for time-independent hazard methods.
  Use the existing policy wrappers; no registration or generic-wrapper changes.
- `reference_time()` returns exactly `batch_shape` in `rate.dtype`, positive
  and finite for valid rates. Use `1/r` where representable; substitute safe
  operands before division and choose a finite positive reference if reciprocal
  overflow or underflow would make it unusable. Invalid-rate references have
  no value guarantee. This fallback never changes active rates or probabilities.

### Numerical invariants and docstrings

- Negative/NaN times use the base policy. At zero: log S=H=0, S=1,
  log F=-inf, log density/log hazard=log(r), hazard=r. At +inf: log S=-inf,
  H=+inf, S=0, log F=0, log density=-inf, log hazard=log(r), hazard=r.
- Substitute safe times before inactive infinity arithmetic in analytic
  overrides where needed. Retain NaN-rate propagation at every time, including
  negative/zero/infinite times; don't hide it behind boundary constants.
  Invalid finite rates are accepted and follow the formula; add no domain mask.
- At finite-valued boundaries, derivatives are the actual limit derivatives:
  S(0), H(0), S(+inf) and log F(+inf) have zero rate derivative;
  log density/log hazard at zero and log hazard at infinity have derivative
  1/r; hazard at infinity has derivative 1. Mixed finite/infinite batches
  must preserve finite-output derivatives for shared and per-entry rates.
- Class and method docstrings state purpose, arguments, returns/shapes and
  dtype/shape misuse errors. Include defining formulas, positive finite rate
  domain, inverse-time units, rate=1/mean, log-rate/log-mean conversions and a
  short conversion example. `from_log` states natural leaves and overflow
  limits. No narrative `docs/` edits or documentation build in this code step;
  tutorial/navigation work remains in "A5 docs" below.

### Own tests and diff budget

Create `tests/unit/test_exponential.py`, with algebra or independently evaluated
NumPy formulas as expectations and existing `tests/impl_tolerances.py` values.

- Seven-method values, paired/multiaxis broadcasting, all time-policy rows,
  NaN isolation, large-H analytic hazard and underflowed survival/log density.
  At H=2e-30 assert relative precision with zero atol; check the lower-tail
  log CDF against log(2e-30), without accepting a zero-H result.
- Only natural rate as a PyTree leaf; keyword requirements, no value checks,
  no `.shape` alias, batch indexing/tree slicing and construction inside jit/vmap.
  Test ordinary/from_log integer/scalar/array dtype behavior and rejected kinds,
  plus reference shape/dtype and finite positive extreme-rate references.
- Finite log-input values and gradients against exp-parameterized algebra:
  d log f/d log r = 1-r*t. Test from_log keyword-only misuse and natural-rate
  overflow without promising a specific invalid-rate method result.
- Eager/jit/vmap agreement; interior forward/reverse derivative 1/r-t and
  Hessian -1/r**2, finite boundary derivatives and mixed-batch finite-component
  Jacobians. Use x64 order-two `check_grads`; use algebra in both modes.

Budget: about 90–110 source/docstring lines, 150–170 own-test lines and under
15 export/import-test lines; keep code/tests/docstrings at most 300 changed
lines. Reuse existing generic registration/policy tests and the independent
suite rather than copying their full matrices. Split before exceeding the
limit; don't defer correctness tests just to fit the diff.

Implementation accounting: 103 family lines, 183 own-test lines, 13 changed
export/import-test lines: **299 changed lines**. Complete boundary matrices,
registration behavior and x64 order-two finite differences reuse the independent
suite. Own tests focus on additional extreme-reference, input-dtype and
mixed-infinity cases. No coverage is deferred beyond the stated step scope.

### Checks, evidence and stop

After implementation, run focused own tests in separate x64/x32 processes,
then `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
and `git diff --check`. Run the following read-only A5 selection in each mode
with `JAX_ENABLE_X64=1` and `0`, `SPEC_THROUGH=A5`:

```bash
.venv/bin/python -B -m pytest \
  tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py \
  tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py \
  tests/spec/test_a5_derivatives.py \
  -k '(exponential and not weibull) or nan_rate or boundary_values_agree_under_vmap_over_time' \
  -q -p no:cacheprovider
```

Planning evidence: this selection collected successfully with `--collect-only
-qq` in both modes (exit 0): construction 10, from_log 4, closed forms 13,
boundaries 54, derivatives 14 — **95 cases**. No implementation tests or full
gate ran during planning. The broader `-k exponential` selection also includes
seven Weibull identity cases, so it is not the A5.3 selection. Mixed-family
tests, Weibull identities and grid tests wait for their implementing steps;
report these exclusions and the existing x32 finite-difference skips explicitly.

After code validation, overwrite `REVIEW.md` with reading order, invariants,
spec/test mapping, exact commands/results and deferred docs. Commit this one
approved code step on the selected branch and stop before A5.4. Don't merge,
publish, edit the spec/independent tests or resolve existing deviations.

## A5.4 — Weibull

### Review fix-up — parameter notation and visible formulas

Status: implemented and verified; see REVIEW.md for the focused results.

Authorized by the maintainer's request to clarify the docstring, visually
separate the mathematics in both built-ins and address the inline WA question.
Same A5.4 step and branch; no new public API or numerical policy.

- `src/survjax/distributions/weibull.py`: define k=shape and b=scale in the
  first docstring sentence. Add whitespace and spec-linked comments that guide
  readers from safe operands to the interior formula, then boundary limits.
  Replace WA's reference-time question with its explanation: scale is already
  a positive finite time for valid Weibull parameters, whereas Exponential's
  inverse rate needs representability guards.
- `src/survjax/distributions/exponential.py`: visibly separate H=rate*t
  from input substitution and infinity restoration, naming its intermediate
  value; identify the log-density and constant-hazard formulas with comments.
  Clarify that the reciprocal guard concerns inactive reference inputs only.
- No tests added for comments/docstrings or an equivalent named intermediate.
  Run existing built-in own tests in both dtype modes, Ruff format/lint, mypy
  and git diff --check. Check the diff for unchanged operations and masks.
  Record exact results and the existing D-020 acceptance failure in REVIEW.md;
  do not rerun unchanged acceptance or edit its tests, spec or docs pages.
- Commit one review fix-up on A5.4, then stop before grids or docs steps.

Status: approved and implemented on 2026-10-13 from `cd3f213`, on the
maintainer-selected branch `A5.4`; awaiting review. Planning changed only
PLAN.md and created no branch or commit. Specification revision
23 and `.spec-seen` agree with design commit `a212ddd`; the synchronized
revision-22 numerical tests remain applicable. No spec ambiguity identified.

### Review fix-up — preserve tiny built-in CDF tails

Authorized by the maintainer's request to implement the review's proposed
tail-stable `log_cdf` branches for both built-ins. This is an A5.4 follow-up
on the same branch, with no public API or export changes. The current spec is
revision 24 (`b917ebe`); its update adds grid twins and does not change the
CDF-tail formulas. See D-021 for the numerical-policy deviation.

- `weibull.py`: add `_log_cumulative_hazard` from `shape * log(t / scale)`;
  use safe time/scale operands before logs and restore the zero/infinity limits.
  Override `log_cdf` to switch to `log_H` when `log_H < log(eps)`. Substitute
  before evaluating the `expm1` branch, including the infinite upper limit and
  values at which `exp(log_H)` would overflow.
- `exponential.py`: add the matching `log_H = log(rate) + log(t)` helper and
  `log_cdf` override, with the same lower-tail switch and safe branch operand.
- Add own tests for hand-derived lower-tail values (including a float32 case
  where forming H underflows), continuity around the dtype-specific switch,
  and finite shape/rate gradients under `jax.debug_nans` on both sides.
  Do not edit the independent revision-24 spec tests.
- Cover spec 4.2, 4.3, 4.5, 7.1 and 7.2. Run
  `scripts/check.sh --through A5`; report its current revision-24 A5.5 grid-test
  failures separately, and run the non-grid A5 acceptance modules in both dtype
  modes. Then overwrite REVIEW.md with mappings, exact results and this gate
  boundary. Commit this fix-up on A5.4 and stop.

### Review fix-up — assert switch gradients with shared tolerances

Authorized by the maintainer's test review. This A5.4 follow-up changes only
the two own `log_cdf` switch tests and REVIEW.md; no runtime code or public API.

- In both tests, compare values using `ATOL[x64_enabled]` and
  `RTOL[x64_enabled]` from `tests/impl_tolerances.py`. Keep `eps` only for
  constructing points on opposite sides of the algorithm's threshold.
- Assert analytic gradients, not just finiteness: Weibull's shape derivative
  is approximately `log(t)` on both sides; Exponential's rate derivative is
  approximately `1` at rate one. Add a short algebra comment explaining the
  O(H) difference on the standard branch. The existing RTOL covers that
  rounding-scale difference, so no tolerance constant is added.
- Run both targeted unit modules in x64 and x32, Ruff format/lint, mypy and
  `git diff --check`. Overwrite REVIEW.md with exact results and commit this
  test-only review fix-up on A5.4, then stop.

Outcome: keyword-only `survjax.Weibull(shape=..., scale=...)` and
`Weibull.from_log(log_shape=..., log_scale=...)`, with seven pointwise direct
methods, analytic hazard limits and safe finite boundary parameter derivatives.
Covered spec: 3.3, 4.1–4.2, 4.5–4.6, 5 (pointwise broadcasting), 7.2–7.3,
10.2–10.3, 10.5 (code docstrings) and applicable 12.1/12.2 family checks.
Grids remain A5.5. Likelihoods, summaries, sampling, parameter diagnostics,
R fixtures, benchmarks and installation checks remain later steps.

### Files and public API

- Create `src/survjax/distributions/weibull.py`: a registered frozen dataclass
  subclass of `SurvivalDistribution`, with `shape: jax.Array = param()` and
  `scale: jax.Array = param()` as its only dynamic leaves. Reuse registration
  for keyword requirements, common dtype, parameter broadcasting, indexing
  and pytree behavior. `.shape` is the natural parameter; `.batch_shape` is
  the array shape. No defaults, dtype keyword or parameter-value checks.
- Export `Weibull` from `src/survjax/distributions/__init__.py` and
  `src/survjax/__init__.py`; update `tests/unit/test_import.py`'s exact export
  expectation. Existing public names and signatures stay as they are.
- Type `from_log` with keyword-only `ArrayLike` inputs and `Self` return.
  Normalize both log inputs together with `_dtypes.normalize_parameters`
  before exponentiation, preserving input-kind rejection, weak promotion and
  stored-float64 warning behavior. Construct with natural `exp(log_shape)`
  and `exp(log_scale)` leaves. Overflow stores infinity; retain no log state.
- `reference_time()` returns scale directly, exactly `batch_shape` and the
  common parameter dtype. For valid parameters it is positive and finite;
  no guarantee for invalid parameters. The base stops substituted references.
- Override `log_survival`, `cumulative_hazard`, `log_prob` and `log_hazard`
  analytically; inherit `survival`, stable `log_cdf` and `hazard`. The inherited
  hazard honors the analytic log-hazard override, including at infinity.
  Keep existing base/registration/policy modules unchanged. Any need to change
  them is a scope expansion to explain before implementation.

### Numerical invariants

- At positive finite time let `u=log(t)-log(b)` and `H=exp(k*u)`.
  Use log S=-H, log h=log(k)-log(b)+(k-1)*u, and log f=log h-H.
  Computing the log ratio without first dividing avoids losing a representable
  H when t/b underflows, or overflowing t/b before a small shape is applied.
  Keep helpers private and local to this family; introduce no numerical layer.
- For zero/infinite time, substitute a safe positive finite input before logs,
  products and exponentiation in each helper. Select the actual boundary
  values afterward, using explicit shape regimes and xlogy-style treatment of
  the zero coefficient at k=1. A call to xlogy alone does not establish safe
  gradients: guard operands before operating. Stop only substituted inputs;
  all active shape/scale expressions, including -log(b), remain differentiable.
- With valid parameters, zero gives log S=H=0, S=1, log F=-inf.
  Log density/log hazard give +inf for k<1, -log(b) for k=1, -inf for k>1;
  hazard gives +inf, 1/b, 0 respectively. Positive infinity gives log S=-inf,
  H=+inf, S=0, log F=0, log density=-inf; log hazard gives -inf, -log(b),
  +inf, and hazard gives 0, 1/b, +inf in those same shape regimes.
- The base supplies negative/NaN-time policy and infinity density precedence.
  Raw log survival must preserve NaN shape/scale at zero and infinity so a
  boundary constant cannot conceal it. Analytic log hazard must also preserve
  NaNs. Accept invalid finite parameters without exceptions or a blanket
  validity mask; their values remain formula-driven as specified in 4.5.
- Mixed [zero, finite, +inf] entries must preserve finite-component first and
  second derivatives for shared and per-entry parameters. Never take a sum
  containing infinite outputs as a derivative test. Do not treat finite-time
  underflow/overflow as an exact time boundary or claim its gradients verified.
  No floors, epsilon, clipping, log-parameter storage or custom AD rules.

### Docstrings and docs to-do

Include summary, arguments, returns/shapes and static misuse errors for public
APIs. The class docstring gives S(t), positive finite domain, dimensionless k,
time units of b, broadcasting and all spec 4.6 mappings: R/flexsurv shape/scale,
NumPyro concentration/scale, survreg mu=log(b) and sigma=1/k, WeibullPH
m=b**(-k), log inputs and inverse scale b=1/r. Include the common-shape AFT/PH
relation gamma=-k*beta, natural versus logged flexsurv coefficients, and a
short conversion example. State from_log overflow behavior and derivative
limits at discontinuous boundaries. Narrative examples remain in A5 docs;
this code step edits no `docs/` and runs no documentation build.

### Own tests, independent coverage and budget

Create `tests/unit/test_weibull.py`. Prefer small named tests with stated
algebra/analytical expectations, using `tests/impl_tolerances.py`; no oracle
comes from running survjax. Reuse the independent suite's complete seven-method
value/boundary matrices, construction/indexing/dtype cases, shape-one identity,
lower/upper tails and x64 order-two forward/reverse `check_grads`.
Own tests add evidence beyond that suite:

- Full two-parameter log-density gradient and Hessian, including the mixed
  entry. For u=log(t/b) and H=(t/b)**k: gradient
  `(1/k+(1-H)*u, k*(H-1)/b)`; Hessian entries
  `-1/k**2-H*u**2`, `(H-1+k*H*u)/b`, and
  `k*(1-(k+1)*H)/b**2`. Check both differentiation modes and jit/vmap in
  both dtype processes. These are smooth positive-time tests, including k=1.
- from_log chain-rule gradients in both parameters: multiply the above
  natural gradients by k and b respectively. Check natural leaves, common
  dtype before exp, rejected input kinds and overflow without prescribing
  a particular invalid-parameter method value.
- Mixed-boundary finite-component Jacobians/Hessians with shared parameters
  and independent parameter batches, including analytic hazard. Pin zero
  k/b derivatives of S(0), log S(0), H(0), S(+inf), log F(+inf).
  At k=1 pin scale derivatives -1/b for finite log density/log hazard and
  -1/b**2 for hazard at zero/infinity; log density at infinity is excluded.
  Do not require shape derivatives at the discontinuous k=1 hazard limits.
- NaN shape/scale isolation beside valid entries at zero/finite/infinite time;
  raw analytic methods broadcast scalar and multiaxis times against batches
  without depending on wrapper pre-broadcasting (A5.3 review regression).
- Positive finite references across small/large valid scales and shapes;
  safe finite evaluations at t=b. Add independent power expectations for
  representable H despite t/b underflow/overflow (e.g. b=1e30, t=1e-30,
  k=0.25 gives H=1e-15, with the reciprocal case H=1e15). Compare tiny H
  with zero atol and existing rtol; compare underflowed probabilities in logs.

Budget: approximately 135–145 source/docstring lines, 140–145 own-test lines
and under 10 export/import-test lines, at most 300 changed code/test/docstring
lines. Tests ship with the code. If safe formulas and meaningful own coverage
cannot fit, propose a split and obtain approval before implementation; do not
omit correctness tests or compress code to force the budget.

Implementation accounting: 145 family/docstring lines, 150 own-test lines,
and 5 export/import-test lines: **300 changed code/test/docstring lines**.
The own mixed-boundary tests also cover the full interior two-parameter
Jacobian/Hessian in both differentiation orders and jit/vmap at k=0.7, 1, 2;
this avoids duplicating a standalone density-Hessian test. Scale derivatives
of the finite log-density/log-hazard limits are demonstrated by the independent
suite; own tests add hazard's -1/b**2 derivative at zero/infinity. No required
coverage was deferred. The sole non-grid acceptance failure is D-020, an
expected value missing log(shape); keep the correct formula and test untouched.

### Validation, commit and stop

After approval and branch selection, run the own Weibull module in separate
`JAX_ENABLE_X64=1` and `0` processes, then
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`.
Run all five non-grid A5 acceptance modules in each mode, with no `-k` filter:

```bash
JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest \
  tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py \
  tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py \
  tests/spec/test_a5_derivatives.py -q -p no:cacheprovider
```

Repeat with `JAX_ENABLE_X64=0`. This includes Exponential regressions,
Weibull/Exponential identities and mixed-family construction. Only the 48 grid
cases remain outside this A5 selection; full through-A5 acceptance waits for
A5.5. Report existing dtype skips exactly; never edit/skip/xfail spec tests.

Planning evidence: the above five-module command with `--collect-only -q`
exited 0 and reported **322 tests collected in 0.21s** in each dtype process.
Collection verifies selection only; no implementation tests or gate were run
in this planning turn. Planning `git diff --check` exited 0 with no output;
only PLAN.md is modified. Repeat the diff check after implementation.

After implementation, overwrite REVIEW.md with reading order, safe-input and
NaN/dispatch invariants, spec-to-test mapping, exact commands/results/skips
and deferred docs. Record any newly discovered ambiguity in DEVIATIONS.md and
ask; leave existing entries open. Commit one approved implementation step on
the maintainer-selected branch, never main, then stop before A5.5.

## A5.5 — Seven time-first grid methods

### Review fix-up — boundary Jacobians, tracing claims and docstrings

Authorized by the maintainer's review of `2d0c22d`; same A5.5 branch and API.
Implemented and verified: 42 own grid cases pass in each dtype mode; Ruff,
mypy and whitespace checks pass. REVIEW.md records the current fix-up evidence.

- `tests/unit/test_survival_grid.py`: differentiate the full mixed-boundary
  vector in both forward/reverse modes and assert analytical derivatives at
  the interior and every finite-valued boundary. Mark the nonfinite-valued
  rows outside spec 4.5's derivative guarantee explicitly in the expectation
  table. This makes the forward-mode part check actual boundary behavior.
- Keep dynamic tracing checks for JAX-compatible rank/bool/complex misuse.
  Capture string/object inputs as constants inside a zero-argument jitted call
  and match the grid conversion error, so JAX's argument rejection cannot
  satisfy the test before the grid method is traced.
- `_grid_methods.py`: replace the compressed Args/Returns/Raises lines with
  concise prose, retaining input/rank rules, output shapes and direct-method
  conversion/value policy. No executable implementation changes.
- Run the own grid module in both dtype processes, Ruff format/lint, mypy and
  `git diff --check`. No full gate rerun is needed for this test/docstring-only
  fix-up; retain the parent commit's gate evidence as historical. Overwrite
  REVIEW.md with actual results, commit this fix-up and stop for review.

Status: implemented and verified on 2026-10-15; awaiting review. Started from
`861ac71` on the
maintainer-selected `A5.5` branch. Planning used `4f867cb`, after A5.4 merged;
the maintainer has since committed D-022 in `861ac71`. The marker
`.spec-seen` records revision-24 content commit `b917ebe`. Design HEAD is now
`addc1cc` (archive snapshot); SPECIFICATION.md and GOALS.md have no diff from
the marker. The local acceptance tests are already synchronized. No branch,
implementation or spec/test synchronization was part of the planning request.

Outcome: all seven inherited grid methods use a single rank/reshape helper and
delegate to public registered direct methods. Own grid coverage has 42 cases in
each dtype mode, including analytical derivatives through `jit(vmap(...))`.
The full `scripts/check.sh --through A5` passes: own x64 415 passed/7 skipped,
x32 421 passed/1 skipped; independent x64 1116 passed/26 skipped, x32 1121
passed/21 skipped; source doctests 1 passed in each mode. Ruff and mypy pass.
Exact commands, reading order, invariants and the D-022 limitation are in
REVIEW.md. No spec, independent tests, docs pages or exports were changed.

### Public API and shape contract

Add the seven inherited methods to `SurvivalDistribution`:
`survival_grid`, `log_survival_grid`, `log_prob_grid`, `log_cdf_grid`,
`cumulative_hazard_grid`, `log_hazard_grid`, `hazard_grid`.
Each has signature `(self, t: ArrayLike) -> jax.Array`, accepts positional or
keyword `t`, and is available on built-ins and registered custom families.
No new top-level exports or family-specific grid implementations are needed.

- Convert a one-dimensional grid with JAX, then reshape it to
  `(K,) + (1,) * len(self.batch_shape)`. Evaluate the corresponding public
  direct method, returning `(K, *batch_shape)`. K=N still evaluates every time
  against every batch entry. Batch rank comes from current leaves, including
  inside `vmap`; component axes are excluded.
- Reject non-1-D grids with `TypeError`, including during tracing. Normalize
  array-conversion misuse to `TypeError`; let the registered direct method
  apply its existing dtype, promotion and value policy. This preserves integer
  inputs, silent float64-time conversion in x32, overrides and NaN precedence.
- Preserve input order and duplicates; support empty grids and empty batches
  through the same reshape/broadcast operations. Shapes are inspected statically;
  grid values never drive Python branching, sorting or host conversion.
- Grid wrappers delegate to the public direct methods, so their safe-input
  calculations and boundary derivatives remain active. No second registration
  wrapper, raw-method bypass, numerical formula or derivative rule is added.

### Files to create or change

- Create `src/survjax/distributions/_grid_methods.py`: seven explicit typed
  thin methods and one private time-conversion/rank/reshape helper. Use a
  type-checking-only base import to avoid a circular runtime import. Each
  public method docstring gives its quantity, 1-D `t`, time-first return shape,
  conversion/value policy and `TypeError` conditions.
- Change `src/survjax/distributions/base.py`: import the private grid module
  and bind its seven methods on the base. The existing registration list
  continues to wrap only direct methods. This file is already 339 lines;
  keeping the grid functions separate limits its addition to import/bindings
  and avoids an unrelated registration refactor.
- Create `tests/unit/test_survival_grid.py`: own analytical and structural
  checks described below. Prefer a small explicit expectation table over a
  generated method factory or a testing helper layer.
- Change `tests/impl_tolerances.py` only to document the rounding/truncation
  scale for applying the existing derivative tolerances to unit-scale smooth
  grid functions. Reuse its ATOL/RTOL for analytical values and derivatives;
  keep the existing finite-difference tolerance values unchanged.
- Update this plan's outcome and overwrite REVIEW.md after implementation.
  Spec files, `tests/spec/`, exports, dependencies and `docs/` stay untouched.

### Own tests and independent acceptance

Keep own tests focused; the read-only grid acceptance module already covers
all seven names, ordinary values, shapes, keyword arguments and policy rows.

- Hand-derived Exponential values for all seven grids with K=N and an unsorted
  repeated time vector: use `S=exp(-r*t)`, `H=r*t`, `h=r`, `log f=log(r)-r*t`
  and `log F=log(-expm1(-r*t))`. Assert full matrices and row order.
- Scalar batches, several batch axes, empty grids and empty batches: assert
  exact time-first shapes. A small custom family reduces its own parameter
  components and verifies they never become grid batch axes; a direct override
  verifies dispatch through the public registered method rather than a default.
- Rank and dtype misuse in eager/jit calls; integer inputs, explicit float32
  inputs in x64, and warning-free NumPy float64 grids in x32. Exercise the
  positional and keyword `t` forms.
- Eager/jit/vmap agreement for every grid, including `vmap` over distributions
  whose per-entry batch rank differs from the original. Count tracing for
  changed parameter/grid values with unchanged shape/dtype/PyTree structure.
- Assert analytical first and second rate derivatives for all seven
  Exponential grids on a positive finite grid. Use order-two `check_grads` in
  both forward/reverse modes on unit-scale interior cases with shared derivative
  tolerances. Expected derivatives come from the same stated algebra, never
  another library call. No time or discontinuous boundary derivative is promised.
- Weibull shape-one grid from zero with the section 11.1 scales
  `10 * exp([0, -0.5, -1])`: verify analytical survival, density and hazard,
  plus the zero shape derivative of `S(0)`. Mixed zero/interior/infinity
  entries retain the existing finite-entry derivative isolation; NaN times
  and parameters remain confined to the corresponding rows/columns.

Covered spec: 4.2, 4.5, 5, 7.2–7.3, 8.1, 10.2–10.3, 10.5 (docstrings),
12.1 and the grid part of Slice A in 12.2. The 116-case independent module is
`tests/spec/test_a5_survival_grids.py`; do not edit, skip or xfail any case.
The full A5 gate must now pass, including all seven grid twins.

### Validation, review and scope boundary

After approval, target roughly 300 changed source/test/docstring lines in one
code commit. Keep the private module and own test file below about 300 lines
each. If the tests exceed that step budget, propose a split before expanding.
Run focused grid tests in separate x64/x32 processes, then
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`
and `git diff --check`. The full gate includes own suites, Ruff, mypy, source
doctests and independent tests in both dtype modes. Report exact skips/results.
Write REVIEW.md with reading order, shape/dispatch invariants, spec-to-test
mapping, actual commands/results and remaining limitations. Commit on A5.5
and stop for review.

D-022 is the maintainer's existing upper-tail issue, committed in `861ac71`. This plan
preserves that file and keeps grid semantics equal to the current direct
methods, including their known log-CDF upper-tail limitation. Propose a
separate numerical step after A5.5 for the shared `log1mexp` fix, with its own
plan approval, mpmath and derivative checks. The grid gate alone does not
establish upper-tail accuracy or complete Slice A. Likelihoods, summaries,
sampling, fixtures, benchmarks and installation checks keep their later scope.

Docs to-do:

- Extend the later grids topic to all seven methods, paired-vector versus
  shared-grid examples (including K=N), posterior batches, order/duplicates,
  component axes and zero-time Weibull examples. Update API navigation and
  playground examples in the separate A5 docs steps.

Planning evidence (collection only; no tests or full gate executed):

- In each dtype process, `SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  tests/spec/test_a5_survival_grids.py --collect-only -q -p no:cacheprovider`
  reports **116 tests collected**.
- In each dtype process, the same collection against `tests/spec` with
  `-k test_a5` reports **438/1142 tests collected (704 deselected)**.
- Only PLAN.md is changed by this planning turn. The pre-existing
  DEVIATIONS.md change is preserved; REVIEW.md remains the last code review.
- Planning `git diff --check` passes with no output. No implementation,
  numerical test execution, docs build or commit was performed.

## A5.5 follow-up — D-022: preserve log-CDF upper tails

Status: approved on 2026-10-15, from clean branch `A5.5` at `ace2f4e`.
Implemented and verified on 2026-10-15; full through-A5 gate passes in both
dtype modes. REVIEW.md records the regression failure and final results.
The maintainer confirmed D-022 as the target and authorized using this branch.
This is one numerical step after the grid work, with its own implementation
commit and review stop. The maintainer approved implementation in conversation.
Specification revision 24 remains read-only. D-022 records the proposed change
to the section 4.2 default formula; leave its status open for the maintainer.

Outcome: generic, Exponential and Weibull `log_cdf` retain small negative
upper-tail values while representable, instead of rounding to zero when the
intermediate CDF rounds to one. `log_cdf_grid` inherits the correction through
its existing delegation. Public names, signatures, shapes and exports do not
change. Covered spec: 4.1–4.3 (numerical recipe only), 4.5, 5, 7.1–7.3 and
12.1. Do not implement HazardFamily, interval methods or later-slice features.

### Files and numerical invariants

- Create `src/survjax/distributions/_log_probability.py` with one typed private
  `log1mexp(x)` helper, taking floating log probabilities and preserving dtype
  and shape. For x <= 0 compute log(1-exp(x)): `log(-expm1(x))` for
  x > -log(2), otherwise `log1p(-exp(x))`. Form the split constant in the
  input dtype. At x=0 return -inf; at x=-inf return zero; NaN and positive
  inputs give NaN. Substitute safe operands in both inactive branches before
  exp/expm1/log/log1p, including the zero-mass case. Stop gradients only on
  substituted values. No floors, probability clipping or new epsilon.
- In `src/survjax/distributions/_survival_methods.py`, delegate the default
  `log_cdf` calculation to this helper, replacing its local zero-mass guard.
  Keep `_raw_log_survival`, the policy wrappers and infinity/cure guards
  unchanged. Update the public docstring to describe both tail branches and
  the remaining lower-tail limitation when the primitive rounds log S to zero.
- In `src/survjax/distributions/exponential.py` and `weibull.py`, replace only
  the standard log-CDF branch with the shared helper applied to -H. Retain
  D-021's direct log-H lower-tail switch and safe log-H substitution before
  exponentiation, including the overflow/upper-limit guard. This preserves
  the existing tiny-tail behavior and boundary parameter derivatives.
- Add `tests/unit/test_log_cdf_tails.py` for the focused public-API regressions
  below. Define one minimal registered exponential custom family locally to
  exercise the generic default independently of the built-in overrides.
- In `tests/impl_tolerances.py`, add named tail tolerances with zero absolute
  tolerance and relative targets 1e-8 (x64), 1e-4 (x32). Explain why absolute
  tolerance would accept an erroneous zero for these tiny values and why
  these targets cover rounding in log-H reconstruction and transcendental
  operations on the chosen normal-range cases. Use the same relative targets
  for tiny analytic derivatives; retain existing finite-difference derivative
  tolerances for `check_grads`. Report failures without loosening tolerances.
- Overwrite `REVIEW.md` after implementation with reading order, invariants,
  spec-to-test mapping, exact commands/results, skips and remaining limits.
  Update this plan's status and D-022's evidence without resolving the entry.

### Tests accompanying the code

Keep the new test module below about 250 lines. Each behavior states its spec
section and expected-value origin. Use the process-selected dtype throughout;
mpmath references use at least 100 decimal digits and the actual rounded input
values converted on the host, never values computed by library methods.

- **Upper-tail values (4.2/7.1/12.1):** compare the generic family and both
  built-ins with mpmath `log1p(-exp(-H))`. Include H=20.1, 33.1 and 54.6
  from D-022, an ordinary H, and deeper normal-range cases (e.g. H=70 in x32,
  H=600 in x64). For Weibull include a nonunit shape/scale case using
  H=(t/b)**k. Require negative, nonzero results and relative accuracy; compare
  eager, jit and vmap. These tests must fail against the original formula.
- **Branch switch (7.2–7.3):** use times on both sides of H=log(2), plus
  the split itself, against independent values and first/second parameter
  derivatives. Test smooth interior points with
  `check_grads(order=2, modes=("fwd", "rev"))`; compare analytical or mpmath
  derivatives at the split rather than finite differences through a branch.
- **Upper-tail derivatives (7.3):** compare forward/reverse parameter Jacobians
  and Hessians with analytical or mpmath derivatives at an upper-tail point,
  including both Weibull parameters. At H=rate*t the rate derivatives are
  t/expm1(H) and -t**2*exp(H)/expm1(H)**2. Use relative-only assertions so
  a zero derivative cannot pass. Compose derivative calls with jit and vmap.
- **Boundary and inactive operands (4.5/7.2):** exercise public log-CDF on a
  mixed vector with zero, ordinary interior, upper tail and +inf. Check full
  forward/reverse Jacobians at finite-valued rows and finite-component
  Hessians, including zero derivatives at +inf. Log CDF at zero is -inf;
  do not assert its derivative as a spec guarantee. Run valid mixed calls
  and derivative checks eagerly under `jax.debug_nans(True)` to detect
  unselected unsafe operations. Check NaN times/parameters separately outside
  that context. Preserve parameter-dependent finite cure limits through the
  existing infinity tests; the generic wrapper must still dispatch unchanged.
- **Grid integration (5/12.1):** one batched, time-first `log_cdf_grid` tail
  comparison against independent references for all three family paths;
  include K=N to distinguish grid shape from paired broadcasting. Existing
  grid tests retain responsibility for the other six adapters.
- Rerun the existing D-021 lower-tail/threshold regressions and generic
  zero-mass, boundary, cure-limit and NaN-policy tests. No edits to
  `tests/spec/`, no new dependency, and no R fixture generation.

### Validation, budget and stop

After plan approval, first run the new value regression against the old code
and record the expected failure. After the fix, run in separate processes:

```bash
JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5
git diff --check
```

The full gate covers Ruff format/lint, mypy, own tests, source doctests and
independent acceptance in both dtype modes. Report actual counts and outputs,
including any failures; the previous grid gate is not evidence for this fix.
Aim for about 300 changed lines including tests and review; keep source changes
small and split the step for approval if the diff exceeds that budget. Commit
the approved implementation on `A5.5` with an imperative subject and a body
explaining the lost upper-tail precision, then stop for review before A5 docs.

Docs to-do: explain both-tail log-CDF evaluation and D-021's retained lower-tail
switch in the later direct-method/custom-family docs topic. State the generic
primitive's representability limitation. Do not edit `docs/` in this code step.

Planning checks: only PLAN.md changes in this turn. `git diff --check` passes
with no output. No implementation, numerical tests, full gate, docs build or commit at this
planning stop. REVIEW.md remains the last completed grid review.

## A5.5 review follow-up — consolidate guarded log arithmetic

Status: approved on 2026-10-15 at `0369331`, branch `A5.5`. The maintainer
selected `_log_math.py`: log and log1mexp also apply outside probability
calculations and approved implementing this refactor in conversation.
Implemented and verified on 2026-10-15: 36 tail cases pass per dtype mode,
and the full through-A5 gate passes. REVIEW.md records exact commands/results.

### Files, responsibilities and API

- Rename `src/survjax/distributions/_log_probability.py` to `_log_math.py`.
  Name its existing guarded kernel `safe_log1mexp(x)`, explicitly documenting
  the convention log(1-exp(x)) for x<=0, dtype/shape preservation, boundaries
  and NaN behavior. Keep the currently verified guarded formulas in this
  organizational step. Reusing jax.nn with a corrected custom JVP changes the
  derivative implementation and should have its own numerical plan.
- Move `_safe_log` from `_survival_methods.py` here as `safe_log`, preserving
  its zero substitution and existing behavior. "Safe" means protecting
  inactive/boundary operands; it does not mean clipping or validating values.
- Add `log_cdf_from_log_hazard(log_h)` in `_log_math.py`, moving the identical
  dtype-epsilon switch, overflow guard and safe exponentiation out of the two
  built-in log-CDF overrides. Its standard branch calls `safe_log1mexp`.
- In `_survival_methods.py`, import `safe_log` and `safe_log1mexp`; keep the
  distribution-aware primitive/derivative helpers and policy wrappers here.
- In `exponential.py` and `weibull.py`, reduce each log-CDF body to a call to
  `log_cdf_from_log_hazard(self._log_cumulative_hazard(t))`. Each family keeps
  its distinct analytic log-H calculation. Future log-H families can reuse the
  shared function without adding a family base class in this step.
- Public API and exports are unchanged. No dependency or docs-page changes.
  Covered spec: 4.1–4.2, 4.5, 7.2–7.3, 10.2 and 12.1. The current spec is
  revision 25; `.spec-seen` still records revision 24. Preserve both during
  this planning/refactor step; spec alignment remains maintainer bookkeeping.

### Tests, checks and review stop

- Extend `tests/unit/test_log_cdf_tails.py`'s derivative checks to H=54.6
  and the existing deep-tail values (H=70 in x32, H=600 in x64), against
  independent mpmath references with the existing relative-only tolerances.
- Add the analytical mixed objective discussed in review as a regression:
  L(eta)=exp(eta)-eta-log F(c), at eta=log(2), with c=100 in x32 and c=1000
  in x64. Check finite value, gradient and Hessian eagerly, under jit and
  debug_nans. Its expected derivatives follow from H=c*exp(eta); evaluate
  the tiny correction with mpmath, rather than asserting an approximate zero.
  This is a synthetic mathematical composition, not a Slice B likelihood API
  or an optimizer implementation. The unmodified JAX rule gives a NaN Hessian
  in the local review probes; retain the useful finite curvature of this sum.
- Rerun existing generic density/hazard, D-021 lower-tail switch, boundary,
  cure-limit and grid coverage. No private helper is exported for testing.
- Run focused log-CDF/family tests in separate x64/x32 processes, then
  `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`
  and `git diff --check`. Report exact results; do not modify spec tests.
- Aim for about 300 changed lines. Update REVIEW.md with reading order,
  numerical invariants, spec mappings and actual checks; commit the approved
  refactor on A5.5 and stop for review. Keep deviations open for the maintainer.

Docs to-do: explain the guarded arithmetic and its limits in the later A5
numerical/custom-family topic. The helper names remain private implementation
details, not a user-facing API.

Planning changed only PLAN.md. Implementation changes the files above and
REVIEW.md; the spec, marker, docs pages and independent tests remain untouched.

## A5.5 review follow-up — WA log-math notes and cumulative-hazard naming

Status: approved in conversation on 2026-10-15 at `7208c81`, branch `A5.5`.
Implemented and verified on 2026-10-15: the three WA notes in `_log_math.py`
are replaced as approved. Targeted x64: 72 passed/1 skipped; x32: 73 passed;
static and whitespace checks pass. REVIEW.md records exact commands/results.

- In `src/survjax/distributions/_log_math.py`, replace the math-note request
  with F=H-H**2/2+O(H**3), hence log F=log H-H/2+O(H**2). This explains
  the existing tiny-H branch without changing its threshold or formula.
- Answer the gradient note beside the substitution: `zeros_like(log_h)` is
  a constant with zero tangent, so an explicit stop_gradient is redundant.
  Keep the real log-H branch differentiable and retain the existing guards.
- Rename `standard` to `log_cdf_from_h` and `value` to `log_cdf` in that
  function, resolving the request for more informative intermediate names.
- Rename the private function to `log_cdf_from_log_cumulative_hazard`, because
  it takes log H, not the instantaneous log hazard. Update imports/calls in
  `exponential.py` and `weibull.py`; keep their log-CDF overrides. The generic
  default already calls safe_log1mexp(log S). Computing log H from log S
  would lose the built-ins' lower-tail guarantee after H underflows to zero.
- Public API and numerical behavior are unchanged; cover spec 4.1–4.2, 7.2
  and 12.1. No docs pages, new abstractions, new tests or tolerance changes.
  Existing independent-reference lower-tail, switch and transformation tests
  cover these naming/docstring edits; retain them without modification.
- Run the log-CDF tail, Exponential and Weibull unit modules in both dtype
  processes, Ruff format/lint, mypy and git diff --check. Record exact results
  and the existing full-gate evidence in REVIEW.md; do not rerun or claim a new
  full gate for this naming/docstring-only fix-up unless a failure warrants it.
- Keep the diff below about 300 lines. Update REVIEW.md, commit the approved
  review fix-up on A5.5 and stop. Spec, marker, independent tests and deviation
  statuses stay unchanged. Docs to-do: use H for cumulative hazard and h for
  instantaneous hazard consistently in the later numerical-methods topic.

Planning evidence: float32 probes of the unchanged public built-ins give
Exponential(rate=1e-30), t=1e-30: log S=-0, generic log CDF=-inf, built-in
log CDF=-138.1551055908203; Weibull(shape=10, scale=1), t=1e-5: log S=-0,
generic log CDF=-inf, built-in log CDF=-115.1292495727539. Only PLAN.md is
changed by the planning turn; the maintainer's WA notes remained intact then.

## A5.5 review follow-up — explicit cumulative-hazard names and helper placement

Status: approved in conversation after `9a8bb2a` on branch `A5.5`.
Implemented and verified: focused x64 249 passed/1 skipped; x32 250 passed.
Ruff, mypy and whitespace checks pass; REVIEW.md records the exact results.
The naming audit found cumulative quantities abbreviated in the shared
conversion and three own test modules. Instantaneous hazard API names are
correct. Starting working tree was clean; this planning turn changes PLAN.md.

### Files and changes

- Move `log_cdf_from_log_cumulative_hazard` from `_log_math.py` into
  `_survival_methods.py`, alongside the generic log-CDF method. Leave only
  general guarded log operations in `_log_math.py`. Update its two imports in
  `exponential.py` and `weibull.py`; retain their analytic log-H overrides.
- In the moved helper, rename `log_h` to `log_cumulative_hazard`,
  `safe_log_h` to `safe_log_cumulative_hazard`, and `log_cdf_from_h` to
  `log_cdf_from_cumulative_hazard`. Keep the formulas, thresholds, guards,
  mathematical H notation and gradient behavior unchanged.
- In `tests/unit/test_log_cdf_tails.py`, use `cumulative_hazard` and
  `cumulative_hazards` for H quantities, including the parametrized derivative
  test argument and its mpmath references.
- In `tests/unit/test_exponential.py` and `test_weibull.py`, rename lower-tail
  switch locals to `log_cumulative_hazard`; rename the Weibull switch's
  `hazard` local to `cumulative_hazard`. In its mixed-boundary derivative
  derivation, replace `h`, `gh` and `hh` (which actually describe H) with
  `cumulative_hazard`, `cumulative_hazard_gradient` and
  `cumulative_hazard_hessian`. Use `log_hazard_gradient` and
  `log_hazard_hessian` for the instantaneous log-hazard derivative arrays so
  the two quantities remain distinguishable throughout the same formula.
- Public API/exports are unchanged. No changes to spec, independent tests,
  marker, docs pages, dependencies, deviations, numerical assertions or
  tolerances. Covered spec: 4.1–4.2, 4.5, 7.2–7.3, 10.2 and 12.1.

### Validation and stop

- Audit `src/` and own tests for remaining ambiguous cumulative-hazard names;
  inspect the meanings rather than renaming actual instantaneous hazards.
- Reuse the existing tests; add no tests for a pure relocation/rename. Run
  the three changed test modules in separate x64/x32 processes, plus generic
  direct/density/infinity and grid modules to check the relocated helper's
  imports and callers. Run Ruff format/lint, mypy and git diff --check.
- Keep the step below about 300 changed lines. Overwrite REVIEW.md with reading
  order, unchanged numerical invariants, spec/test mapping, exact results and
  historical full-gate evidence. No full-gate rerun is planned for this
  relocation/naming-only fix-up unless a failure warrants it. Commit on A5.5
  and stop for review; do not start docs work.

Docs to-do: distinguish cumulative hazard H, log cumulative hazard log H,
instantaneous hazard h and log hazard log h in the later numerical-methods
topic. Internal helper identifiers are not public documentation API.

### A5 docs

Docs to-do:

- Audit all `docs/` against merged A5 code: mark stale, missing or fine.
  A5.1 commit `3f50962` changed `docs/distributions.md` (12 insertions,
  1 deletion); treat it as a draft. Include the earlier tutorial rewrite
  `60bee93` and its distributions, SurvivalData, index and config changes.
- Custom families: guarded-boundary extension recipe, proper-family zero
  limits and parameter-dependent cure limits; primitive safe operands/NaN
  propagation; mixed-entry derivative isolation and its limits. Correct
  PiecewiseHazard's reference to exactly `batch_shape` in `rates.dtype`.
- Built-in families: Exponential units/domain, rate=1/mean, log-rate/log-mean
  conversions and from_log overflow limits; Weibull's complete 4.6 table and
  conversions for R/flexsurv, NumPyro, survreg mu=log(b), sigma=1/k,
  WeibullPH m=b**(-k), inverse scale and log inputs; common-shape PH gamma=-k*beta.
  A5.4 includes these mappings in its class docstring as required by 4.6;
  the docs step expands them into executable conversion examples and explains
  natural estimates versus logged flexsurv coefficients. Cover zero/infinity
  hazard limits and the exclusion of discontinuous shape derivatives at k=1.
- Direct methods and grids: executable synthetic examples, paired-vector
  pitfalls and posterior batch `shape=k[:, None]`, `scale=jnp.exp(beta @ X.T)`.
- Update `docs/index.md` navigation/current coverage and `docs/playground.ipynb`
  for the implemented APIs. Preserve the tutorial; claim no unverified R
  agreement or full-Slice-A completion. Inspect rendered pages and execute
  examples/notebooks in both dtype modes.
- Code steps append any additional documentation needs here.

Docs steps:

- A5.d1: audit `docs/` against merged A5 code (A5.1 drafts included); list
  stale, missing, fine; split the rest into steps.
- A5.d2+: one per page or topic, e.g. SurvivalData interface, built-in families,
  custom families, grids, navigation or playground, as determined by the audit.
  Each is at most 300 changed lines; `scripts/build_docs.sh` builds clean and
  examples/notebooks execute. Review against the spec and merged code, covering
  every Docs to-do. A5 is done after its last docs step merges, before A6 starts.

### A5.d1 — audit result and docs-step split (2026-10-15)

Audited on branch `docs-A5` at `055e8b8` (equal to main) against the merged
A5 code, spec revision 25 and the docs to-do above. The maintainer asked for
narrative, tutorial-style pages with links to the mathematics and, where it
helps, to R's `survival`/`flexsurv`, rather than assertion-style examples,
and for a review of the public docstrings in the same style.

Audit of `docs/`:

| Page / part | Verdict | Finding |
| --- | --- | --- |
| `index.md` | stale | Says built-in distributions are planned; nav lacks built-ins and grids. |
| `distributions.md` top note, "Many distributions" last paragraph | stale | Calls `Exponential`, `Weibull` and time-first grids planned. |
| `distributions.md` constant hazard, Gompertz, batches, piecewise, JAX values | fine | Executes; tutorial voice the maintainer wants. `PiecewiseHazard.reference_time` already has exactly `batch_shape` in `rates.dtype` (to-do item satisfied). |
| `distributions.md` `ProportionalHazards.reference_time` | fine | Corrected on checking: the nested baseline is broadcast at construction, so it already returns `(3,)`. |
| `distributions.md` "edges of the time axis" (A5.1 draft) | stale/missing | Correct values, but assertion blocks instead of narrative; no mixed-batch derivative isolation (A5.2b), no cumulative-hazard/log-hazard guards. |
| `distributions.md` "why the log scale" | missing | No H / log H / h / log h distinction; no upper-tail `log_cdf` behaviour (rev 25, D-022). |
| `distributions.md` checklist/details | fine | Checklist already says "exactly `batch_shape`". |
| built-in families page | missing | Exponential and Weibull: formulas, domains, units, `from_log`, boundaries, k=1 derivative exclusion, 4.6 conversions. |
| grids page | missing | Seven `_grid` twins, paired-vector pitfall, posterior batch example, zero-time k=1 grid. |
| `survival-data.md` | fine | A1–A3 only; no A5 content needed. |
| `playground.ipynb` | stale | SurvivalData only; to-do asks for the implemented distribution APIs. |
| `simulation-guide.md` | fine | Excluded from the build; describes later slices. |

Audit of public docstrings (rendered only through `help()`; no autodoc):

- Wrong: `SurvivalDistribution.reference_time` says "broadcastable to
  `batch_shape`"; spec 4.2 (rev 22) requires exactly `batch_shape` in the
  parameter dtype.
- Terse and test-like: `Exponential`, `Weibull` and their methods compress
  formulas, limits and spec citations into fragments ("pairing equal-length
  vectors", "The infinity rate derivative is 1", "Uses no tail subtraction").
  Several use one-line `Args: t: ...` forms that do not read as sections.
- Implementation detail in user docs: base `log_survival`, `survival`,
  `log_cdf`, `cumulative_hazard`, `log_prob`, `log_hazard` describe the policy
  wrapper's substitution mechanics. Users need the observable rule (values at
  0/+inf/negative/NaN; which derivatives are guaranteed), not the mechanism.
- Grid docstrings are correct but identical boilerplate; they lack the
  `(K, *batch_shape)` worked shape example and the reason to prefer them.
- Fine: `param`, `static`, `distribution` (doctest), `batch_shape`,
  `__getitem__`, `register_family`.

Proposed steps (one commit, REVIEW.md and stop after each; each at most 300
changed lines; `scripts/build_docs.sh` clean in both dtype modes):

- **A5.d2 — built-in families page** (`docs/builtin-families.md`, toctree
  entry). Story: time to progression in a synthetic cohort. Exponential:
  constant hazard, `rate=1/mean`, memorylessness, `from_log(log_rate=...)`
  and log-mean conversion, overflow giving an infinite rate. Weibull: role of
  k (falling, constant, rising hazard), b as the time where S=e^{-1}, k=1
  equals the exponential, values at t=0 and +inf for k<1, =1, >1, and why the
  shape derivative of log f(0) is excluded at k=1. Math links: Wikipedia
  exponential and Weibull distributions, survival/hazard function articles.
  Executed examples print values; expected numbers are derived in the prose.
- **A5.d3 — coming from R and NumPyro** (section of the same page, or
  `docs/parameterizations.md` if d2 is near budget). Spec 4.6 table as a
  narrative: `dweibull`/`pweibull`, `flexsurvreg(dist="weibull")` natural
  estimates vs `coef(fit)` logs, `WeibullPH` m=b^-k, `survreg` mu=log b,
  sigma=1/k with a worked conversion, common-shape AFT→PH gamma=-k*beta,
  NumPyro `concentration`. R code appears as non-executed `r` blocks labelled
  as input conventions; no numerical agreement claimed (10.5; fixtures are A8).
  Links to the CRAN reference pages of `survival::survreg` and `flexsurv`.
- **A5.d4 — survival curves and time-first grids** (`docs/grids.md`).
  Pointwise pairing vs `_grid` twins; the pitfall when K equals N; posterior
  draws `Weibull(shape=k[:, None], scale=jnp.exp(beta @ X.T))` with synthetic
  draws giving `(K, D, N)`; the section 11.1 fixed vector (k=1, scales
  `10*exp([0,-0.5,-1])`) on a grid starting at t=0 with finite gradients;
  grids under `jit`/`vmap`.
- **A5.d5 — custom-family page refresh** (`docs/distributions.md`). Remove
  stale "planned" text and link to the built-ins; fix
  `ProportionalHazards.reference_time` to broadcast to `batch_shape`; rewrite
  the time-edges section as narrative with printed outputs instead of asserts,
  covering mixed-batch derivative isolation and finite cure limits; add H,
  log H, h, log h and both `log_cdf` tails to the log-scale section.
- **A5.d6 — built-in docstrings** (`exponential.py`, `weibull.py`). Narrative
  summary with formula, domains and units, Google-style Args/Returns/Raises,
  4.6 mapping kept in the Weibull class docstring, a short doctest each
  (run by `check.sh`), limits stated as observable behaviour. No code change.
- **A5.d7 — base and grid docstrings** (`base.py`, `_survival_methods.py`,
  `_grid_methods.py`). Fix `reference_time`; replace mechanism with the
  observable value/derivative rules; give grid twins a worked shape example.
  No code change; split if over budget.
- **A5.d8 — navigation, playground, slice evaluation.** `index.md` coverage
  and toctree; add a distributions section to `playground.ipynb`
  (Exponential/Weibull, a grid, `from_log`, things to try); the 10.5
  end-of-sub-slice rendered-page evaluation in REVIEW.md.

Maintainer decision (2026-10-15): do d2–d8 as one docs step, without the
300-line limit; plots and `matplotlib` (docs group only) are allowed;
document what is implemented (spec rev 25), `.spec-seen` follows separately.
Status: implemented on `docs-A5` as one commit; see REVIEW.md.

Open questions (resolved by the decision above):

- Plots would help the hazard-shape story but need `matplotlib` in the docs
  group (a new dependency). Default: no plots; small printed tables instead.
- The spec is at revision 25 (`1e9011c`) while `.spec-seen` is `b917ebe`
  (rev 24). Docs will describe rev 25 behaviour (upper-tail `log_cdf`), which
  the merged code implements; marking `.spec-seen` is a separate action.

## Verification and acceptance mapping

Independent A5 modules: construction/reference arrays -> A5.3/A5.4;
from_log -> A5.3/A5.4; closed forms/tails -> A5.3/A5.4;
boundaries -> A5.1/A5.2/A5.3/A5.4; derivatives -> A5.3/A5.4;
time-first grids (all seven methods) -> A5.5. The three revision-22 A4 additions
are A5.1 regressions. Tests purposely do not define finite-invalid-parameter
values, overflowing-from_log method values, discontinuous shape gradients,
arbitrary custom boundary derivatives or a particular reference-time formula.

- Own value tolerances remain in `tests/impl_tolerances.py`. For tiny H use
  relative-only comparisons at the existing rtol, with zero atol; do not accept
  an absolute tolerance that makes zero pass. New derivative/tail scales need
  justification in that same file. Expected values have stated algebra/identity
  origins. Never loosen tolerances to make failures pass.
- Run interior order-two forward/reverse `check_grads` with justified tolerances;
  independent finite-difference cases are x64-only, while hand-derived first/
  second derivatives run in both modes. Own x32 finite differences need their
  own scale justification. Do not toggle JAX x64 inside a process.
- Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Each code step runs
  `scripts/check.sh --through A4` and
  `git diff --check`, plus read-only independent tests for completed behavior
  in separate x64/x32 processes. At A5.5 run the full
  `scripts/check.sh --through A5` against the synced revision-24 suite.
  Report focused selection and still-unimplemented cases explicitly; never
  present an unsynced local gate as revision-24 acceptance.
  Docs steps run `scripts/build_docs.sh` in both dtype modes and inspect the
  rendered pages and outputs against the merged A5 implementation.
- REVIEW.md records reading order, numerical/storage invariants, spec-to-test
  mapping, exact outputs/skips and open deviations; docs steps also record
  executable docs and rendered-page review. One commit,
  then stop. No merge, publishing, remote setup or next-step work without approval.

## Evidence gathered while updating this plan

Read-only execution used this checkout's `.venv/bin/python -B`, with
`-p no:cacheprovider`, against `../spectest` at `7f1d6df` (no sync or writes):

- x64: `JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  ../spectest/tests/spec --collect-only -q -k test_a5 -p no:cacheprovider`
  exited 0: **370/1074 tests collected (704 deselected)**, 0.44s.
- x32: `JAX_ENABLE_X64=0 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  ../spectest/tests/spec --collect-only -qq -k test_a5 -p no:cacheprovider`
  exited 0: boundaries 186, closed forms 58, construction 29, derivatives 39,
  from_log 10, grids 48 (**370 total**).
- Ran `JAX_ENABLE_X64=1 SPEC_THROUGH=A4 .venv/bin/python -B -m pytest
  ../spectest/tests/spec/test_a4_nan_precedence.py -q -k infinity
  -p no:cacheprovider`, then the identical command with `JAX_ENABLE_X64=0`.
  Both exited 1:
  **3 failed, 7 passed, 24 deselected** (x64 0.66s; x32 0.46s).
  The three new NaN-parameter cases get -inf rather than NaN from the current
  density policy. This is an implementation gap against the changed contract,
  not a suspected spec-test error; no new deviation is needed.
- A5 tests were collected/read, not executed. The full gate, docs build,
  numerical A5 implementation checks and R work were not run for this plan.
- `git diff --check` exited 0 with no output after the planning edits.

## Remaining sub-slices: docs planning

These entries reserve the final docs steps; code-step plans fill in the page
and topic details. A1–A4 and the older plans below are completed history;
the A5 transition does not reopen them. For each remaining sub-slice, docs
steps start after its last code step merges, cover one page or topic and at
most 300 changed lines each, and pass `scripts/build_docs.sh` with examples
and notebooks executing. The next sub-slice waits for the last docs merge.

### A6 docs

Writing direction (maintainer, 2026-10-22): tutorial style, with mathematics
followed by code that applies it and prose interpreting the output. Build a
coherent example; avoid writing the guide as a collection of unit tests.

Docs to-do:

- A6.3: parameter_issues/parameter_conditions reference and hook contract,
  component reductions, own/nested ordering and dotted keys, fixed structure,
  and custom-condition/nesting/compiled-checkpoint examples. Explain the
  opposite polarity of constraint results and diagnostic masks.
- A6.2: ParameterIssues fields, ordering and polarity, dotted names, host-only
  raising (including traced rejection), the container reference, and the
  distinction between no reported issue and guaranteed validity. Producer
  examples follow A6.3; A6.2's minimal exception autodoc entry is implemented.
- A6.4: exact/right untruncated likelihoods, function/method and raw/prepared
  forms, unsummed output and shapes/dtypes, issue propagation, zero mass,
  uninformative rows, and the approved unsupported-mode policy. Document the
  approved D-023 mode-support check in NaN triage. Add the method reference
  and executable examples after the public implementation merges. There is
  no likelihood issue container or _with_info variant.

Docs steps:

- A6.d1+: one page or topic per step against merged A6 code; cover all Docs to-do.

### A7 docs

Docs to-do:

- Verified tail/boundary behavior, numerical limits and supported diagnostics;
  fill in any documentation needs found by A7 code/test steps.

Docs steps:

- A7.d1+: one page or topic per step against merged A7 code; cover all Docs to-do.

### A8 docs

Docs to-do:

- Reference-fixture provenance and verified R comparison claims; fill in from
  committed fixtures and A8 check results.

Docs steps:

- A8.d1+: one page or topic per step against merged A8 code; cover all Docs to-do.

### A9 docs

Docs to-do:

- Tiny synthetic example, supported Slice A workflows and limitations, and
  verified installation/transformation behavior; refine during A9 code steps.

Docs steps:

- A9.d1+: one page or topic per step against merged A9 code; cover all Docs to-do
  and record the final Slice A rendered-documentation review.

# A4.5 reviewer fix-up — F-1 and N-1 through N-4

Status: implemented and verified; see REVIEW.md for the successful A4 gate and
D1 mutation evidence. Authorized by the maintainer's request to address
`../review/slices/A4.5.md`.
Continue on `A4.5` after `fe9aa8e`; implement the review checklist in one
fix-up commit, then stop. No public API, protected-file or A5 changes.

- `_survival_methods.py`: rename the guarded logarithm to `_safe_log(x)`;
  remove the default density's duplicate +inf guard and the unreachable
  integer cast, retaining either only if a test result changes; have hazard
  resolve the raw log-hazard method via its marker, honoring overrides.
- `tests/unit/test_family_density.py`: add the quadratic-survival zero-time
  hazard gradients (single time and mixed batch, hand-derived 0 and 2);
  make the override deliberately different from the primitive hazard; count
  reference-time calls for every direct method. Covers spec 4.2/4.5/7.2/7.3.
- Temporarily remove the safe logarithm operand and show that the new F-1
  tests fail with NaN in both dtype modes; restore the implementation before
  final validation. No tolerance or independent-test changes.
- Run the full A4 gate, whole-word stale-name search and git diff checks.
  Overwrite REVIEW.md with findings addressed, corrected evidence, mutation
  results, exact gate output and the unchanged A5/D-016 handoff.
- Commit the review checklist as one fix-up with an imperative subject, then stop.

# A4.5 review fixes — WA naming comments

Status: implemented and verified; see REVIEW.md. Authorized by the maintainer's
request to address the inline WA comments. Continue on `A4.5` after `e2c08af`;
one naming-only fix-up commit, then stop.

- `src/survjax/distributions/_survival_methods.py`: rename the JVP helper to
  `_log_survival_and_derivative`, matching its actual return values. Rename
  `_log_rate` to `_log_hazard`, name its argument and safe operands for hazard,
  and document that it takes the logarithm of a linear hazard value. Update
  both defaults' calls and remove the three addressed WA comments.
- Public API and numerical behavior: unchanged. No new tests; existing density,
  hazard, zero-hazard derivative and transformation cases cover spec 4.2/7.2.
- Run `scripts/check.sh --through A4` with the existing temporary uv cache,
  check for remaining WA/stale helper names and run `git diff --check`.
- Overwrite REVIEW.md with the fixes, invariants and exact results. Commit this
  review fix-up on the same branch under the working agreement, then stop.

# A4.5 — Autodiff density and hazard defaults

Status: implemented and verified on maintainer-selected branch `A4.5`,
starting at `9d1d08d`, against specification revision 21 (2026-10-08).
The maintainer approved implementation and keeping density/hazard together
despite the roughly 375-line source/test/doc diff. See REVIEW.md for exact
evidence. The maintainer authorized the completion commit on 2026-10-08;
commit this step, then stop for review.
This is the active plan; the A4.4 entries below are completed history.

## Outcome and scope

Complete the Slice A distribution base's pointwise methods by adding typed
`SurvivalDistribution.log_prob(t)`, `log_hazard(t)` and `hazard(t)` defaults,
with the section 4.5 policy enforced for defaults and family overrides.
No new top-level exports. Their output shape is the ordinary broadcast of
`t.shape` and `batch_shape`; equal-length vectors pair entries, and singleton
axes request all pairs. These are not shared time-first grid methods.

Covered spec: 3.3, 4.1–4.2, 4.5, 5, 6.2, 7.2–7.3, 8.1 and 12.1–12.2.
Built-in Exponential/Weibull families and survival grids remain A5; likelihoods
and `parameter_issues()` remain A6. The review's D-016 positive-infinity
survival/log-CDF gradient fix stays explicitly deferred to A5. No dependencies,
support API, custom derivative rules, parameter-domain validator or new cache.

## Files and numerical implementation

1. Extend `src/survjax/distributions/_survival_methods.py`; keep numerical
   additions here, as directed by the A4.4 review, rather than growing base.py.
   Add a small typed private helper returning raw log survival and its pointwise
   time derivative using forward-mode `jax.jvp` with an all-ones time tangent.
   Reuse `_raw_log_survival`; differentiate the family primitive on the times
   already substituted by the outer policy, not the policy constants.
   Do not build a full time Jacobian or flatten distribution batch/component
   axes. Cast integer times to the common promoted floating calculation dtype
   before JVP; preserve normal weak typing and time/parameter promotion.
2. `log_prob(t)` uses `log(-d log S/dt) + log S`; it is independent of any
   `log_hazard` override. `log_hazard(t)` uses the derivative directly, avoiding
   cancellation from `log_prob - log_survival` at large cumulative hazard.
   Its generic result is NaN where raw log survival is exactly `-inf`, including
   at finite support ends; finite log survival remains eligible even when
   exp(log survival) underflows to zero. `hazard(t)` is exp(log hazard) and
   honors the family's `log_hazard` override, including its analytic +inf limit.
3. Substitute safe operands before inactive logarithms/additions; use the
   double-where pattern, stopping only placeholders/substituted inputs.
   True zero hazard gives log hazard/log density `-inf` where the formulas
   apply. Invalid negative/NaN derivatives stay NaN; never clip, floor, add an
   epsilon or infer a custom family's parameter validity domain.
4. Extend the existing policy wrapper only as needed for density at +inf.
   Negative times give log density/log hazard `-inf`, hazard zero; NaN times
   give NaN. Preserve the revision-21 NaN precedence at negative/NaN times.
   Zero remains active and obtains boundary values from the primitive or
   override. Enforce `log_prob(+inf)=-inf` for defaults and overrides, including
   formulas that otherwise yield `+inf - inf`. Keep family evaluation at zero
   and +inf consistent with 4.5, and do not change existing survival/CDF policy.
5. `src/survjax/distributions/base.py`: bind the three defaults explicitly and
   add them to registration's policy list. Keep the raw-method marker behavior
   so decorated subclasses preserve overrides without stacking wrappers.
   No construction, PyTree, static metadata or public diagnostic changes.

## Own tests, with the implementation

Create `tests/unit/test_family_density.py` (aim below 200 lines), using small
custom frozen dataclass families and hand-derived expectations. Extend the
existing method-name cases in `test_family_direct.py` only where they give
useful dtype/policy coverage for the three added methods. Use shared value and
existing justified derivative tolerances from `tests/impl_tolerances.py`;
change that file only if a distinct numerical scale needs justification.

- Constant hazard: log density `log r-r*t`, log hazard `log r`, hazard `r`;
  scalar, paired vectors, explicit outer broadcasting, integer times and
  mixed time/parameter dtypes. Exercise jit/vmap, shape-mismatch TypeError,
  and repeat compiled calls with changed parameter values.
- Power survival `log S=-(t/b)**k`: hand-derived log density/hazard across
  decreasing, constant and increasing hazards, zero-time values for k below,
  equal to and above one, and large H with representable hazard. Pin density
  at +inf to -inf and distinguish generic hazards from analytic overrides.
- A bounded-support primitive: generic hazards become NaN at/beyond the
  support end; finite very negative log survival keeps a valid hazard despite
  linear survival underflow. Include a locally zero hazard without log(0)
  contaminating an otherwise active batch's finite derivatives.
- Explicit/inherited density and log-hazard overrides: verify dispatch,
  hazard's use of the log-hazard override, policy at negative/NaN/+inf times,
  no mutation, and no stacked wrappers after subclass registration.
- Unguarded primitive and parameter-dependent reference: negative/NaN time
  slots see safe inputs and retain zero parameter gradients with valid
  parameters. Mixed valid/NaN parameters keep NaNs isolated by entry.
- Smooth positive-time interior: hand-derived gradient and Hessian (constant
  density has `d/dr=1/r-t`, `d2/dr2=-1/r**2`); check first/second parameter
  derivatives forward/reverse, composed with jit/vmap. Run order-two
  `check_grads` with the established tolerances. Do not promise parameter
  derivatives at invalid inputs, support changes or infinite log densities.

## Documentation, verification and stopping point

- Update `docs/distributions.md` with an executable concrete density/hazard
  example, the complete negative-time policy, generic zero-survival behavior,
  override dispatch, and the representable-linear-hazard limitation of the
  autodiff defaults. Update `docs/index.md` to reflect the completed methods.
- Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run the full maintainer gate
  `scripts/check.sh --through A4`: Ruff, mypy, own tests and all independent A4
  tests in separate x64/x32 processes. The former focused subset is no longer
  sufficient; this step completes the remaining A4 acceptance surface.
- Run combined own/independent sessions with `SPEC_THROUGH=A4` in both modes
  to catch collection/import conflicts, source doctests on base.py and
  _survival_methods.py in both modes, executable docs in both modes, and
  `git diff --check`. No network, R fixture generation or wheel work here.
- Keep spec, acceptance tests, gate scripts and alignment markers read-only.
  On a genuine contract ambiguity or apparent acceptance-test error, record
  an open deviation and ask rather than silently choosing or editing tests.
- Aim for roughly 300 changed source/test/doc lines. If it grows materially
  beyond that, propose a split before expanding the implementation.
- Overwrite REVIEW.md with reading order, numerical/dispatch invariants,
  spec-to-test mapping, exact results/skips and D-016's A5 handoff. Update this
  status, then stop for review. Do not begin A5 or commit until instructed.

# A4.4 completion — specification revision 21 (2026-10-08)

Status: implemented and verified; the maintainer authorized the completion
commit on 2026-10-08. See REVIEW.md for current results and the A5 handoff.

Approved by the maintainer's request to finish A4.4 after spec/acceptance updates.
This continues the existing approved step on `A4.4` at `d0b5538`; no A4.5 work.
The earlier implementation was committed by the maintainer as `8326b45`.
Two new untracked files under tests/spec are read-only inputs; leave them alone.
The maintainer's 2026-10-08 commit request supersedes the earlier hold.

- Rename `distributions/_direct.py` to `_survival_methods.py` and update the
  base import/bindings/wrapping calls, as directed by review A4.4 item 3.
- In that module, reject nonreal/non-numeric times with field-independent
  TypeError using dtype only, including under jit/vmap. Preserve JAX conversion
  and promotion without float64-time warnings (spec 4.5 revision 21).
- Keep numerical NaN precedence; replace its D-013 comment with the now-settled
  spec 4.5 citation. Consolidate repeated raw-primitive lookups into the
  review-requested typed `_raw_log_survival` helper. No public API additions.
- Add own dtype rejection/transformation and no-warning/promotion tests in
  `tests/unit/test_family_direct.py`, plus the exact hand-derived time derivative
  at zero (`-rate = -2`). Keep existing tolerances unchanged.
- Add the review's D-016/D-017/D-018 entries to DEVIATIONS.md. D-016's fix is
  explicitly deferred to A5. D-017/D-018 now have normative revision-21 text;
  annotate that evidence but leave all statuses open for maintainer closure.
- Update docs/distributions.md for direct-time dtypes, NaN precedence and
  finite-invalid-parameter behavior. Update the active plan/status and overwrite
  REVIEW.md with current invariants and exact evidence. No protected edits.

Coverage: spec 3.3, 4.2, 4.5, 5, 6.2, 7.2–7.3, 12.1. Validation: through-A3
maintainer gate; prior 112 A4 nodes plus new acceptance nodes using only the
four A4.4 methods, in both dtype modes; combined own/spec through-A3 sessions;
docs build and source doctests in both modes; git diff --check. Do not run or
claim the full A4 gate until A4.5. Keep this fix-up under roughly 300 lines.
Commit this completion as authorized, then stop before A4.5.

# A4.4 — Direct-method policy and survival/CDF defaults

Historical plan: approved by the maintainer ("go for it and implement") and
implemented on `A4.4`, starting at `6dc4747` against specification revision 19.
The maintainer committed that implementation as `8326b45`. The revision-21
completion above is the active plan; this and earlier plans are history.

## Outcome and public API

Registered custom families gain `survival(t)`, `log_cdf(t)` and
`cumulative_hazard(t)` defaults. Registration enforces the section 4.5 time
policy for these methods and `log_survival(t)`, including family overrides.
The primitive family implementation sees only nonnegative, non-NaN times;
zero and positive infinity still reach it. Pointwise shapes follow ordinary
trailing-axis broadcasting, with TypeError for static time/batch mismatches.

No new top-level exports. Density and hazard defaults/policy stay A4.5;
built-ins and time-first grids stay A5. D-013 remains an A6 handoff: do not
invent a parameter diagnostic contract or custom parameter validity domain.

## Files and implementation

- Create `src/survjax/distributions/_direct.py`: typed ordinary functions for
  the three generic defaults and a single registration-time policy wrapper.
  Keep this numerical responsibility separate from construction/PyTrees.
  Use TYPE_CHECKING imports for the base type to avoid import cycles.
- `src/survjax/distributions/base.py`: explicitly bind the three default
  functions as base methods and integrate policy wrapping at registration.
  Preserve the concrete nested-family docstring and all A4.3 behavior.
  Avoid a large mechanical registration refactor just to add these methods.
- Defaults are `exp(log_survival(t))`, `-log_survival(t)`, and
  `log(-expm1(log_survival(t)))`. Guard zero CDF with safe inputs before the
  logarithm, then return -inf for true zero; no floors, epsilons or clipping.
  NaNs remain NaNs. Document the generic CDF lower-tail underflow limit.
- The policy wrapper first broadcasts times against `batch_shape`. For
  negative/NaN times, substitute the family's reference time before executing
  family code and stop gradients only through the substituted input. Apply
  negative-time constants (0, 1, -inf, 0 respectively) and NaN-time outputs
  afterwards. Preserve NaN family outputs; do not hide invalid parameters
  behind a negative-time constant or add host-side parameter inspection.
- Wrap both inherited defaults and overrides of the four in-scope methods.
  Retain raw implementations through a transparent wrapping convention, so
  registration of explicitly decorated subclasses does not stack policy
  wrappers or introduce recursion. Cover inherited overrides in own tests.
  This is registration-time setup, not a mutable evaluation cache.
- `tests/unit/test_family_direct.py`: independent custom primitive and override
  families, one behavior per test. Extend `tests/impl_tolerances.py` with the
  mandated float32/float64 value tolerances and any justified tail-specific
  tolerance; retain existing derivative tolerances.
- Add the required reference hook to the existing batching-test Wrapper,
  whose numerical method now goes through the policy.
- `docs/distributions.md` and `docs/index.md`: describe the now-available
  methods, boundary policy and shape convention. Add an executable primitive
  family example with actual output values and positive-time/negative-time
  evaluation; mark density/hazard defaults as later A4 scope.
- Overwrite REVIEW.md after implementation: reading order, raw/policy method
  dispatch invariants, spec-to-test mapping, exact checks/results and limits.

Spec coverage: 4.2, 4.5, 5, 7.1–7.3, 10.2–10.5 and 12.1.

## Tests and evidence

- Hand-derived constant-hazard values: `log S=-r*t`, `S=exp(-r*t)`,
  `H=r*t`, `log F=log(-expm1(-r*t))`. Check scalar, paired vectors and
  explicit `(D,1)` by `(N,)` evaluation; incompatible shapes raise TypeError.
- All four methods: negative infinity/negative/zero/interior/positive infinity/
  NaN time values, eagerly and under jit/vmap, including mixed batches.
  Include invalid parameter NaNs beside valid entries and verify isolation.
- Independently override each derived method and verify dispatch plus policy
  enforcement; register a decorated subclass inheriting an override and
  verify it is not wrapped twice. Methods do not mutate PyTree leaves.
- A deliberately unguarded positive-time primitive verifies safe substituted
  times and finite zero parameter gradients for negative policy values.
  Test parameter-dependent reference times without stopping active parameter
  calculations; test second-order forward/reverse gradients on valid smooth
  interior cases using existing justified derivative tolerances.
- Generic log-CDF tails: tiny representable mass compared in log space against
  high-precision mpmath; zero mass remains -inf. No expected value is derived
  from the implementation under test. Include zero-boundary inputs where
  inactive operations must not produce spurious gradient NaNs.

## Validation and stopping point

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` throughout:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests in separate dtype processes.
2. Repeat the existing 89 focused A4 nodes in x64/x32. Additionally select
   from `test_a4_generic_defaults.py` every test except the seven tests using
   `log_hazard`/`hazard`; from `test_a4_policy_enforcement.py`, select
   `test_unguarded_log_survival_is_zero_for_negative_time`,
   `test_unguarded_survival_is_one_for_negative_time`,
   `test_unguarded_cumulative_hazard_is_zero_for_negative_time`,
   `test_unguarded_log_cdf_is_minus_infinity_for_negative_time`,
   `test_unguarded_infinite_time_log_survival_comes_from_the_family`,
   `test_negative_time_log_survival_has_zero_parameter_gradients` and
   `test_negative_time_survival_has_zero_parameter_gradients`.
   Inspect dependencies and record exact selectors/results in REVIEW.md.
   Do not edit, skip or xfail protected acceptance tests. The full A4 gate
   remains required at A4.5.
3. Combined own/independent sessions with `SPEC_THROUGH=A3`, in both dtype
   modes, preserving the unique implementer tolerance-module name.
4. `scripts/build_docs.sh` (approved kernel-port escalation if needed), plus
   the existing concrete nested-field doctest in both modes.
5. `git diff --check`.

Aim for about 300 changed code/test lines. If implementation requires a
larger refactor or split, propose it before exceeding the approved step.
Do not change dependencies, the specification, protected scripts or tests/spec.
Log newly uncovered spec ambiguities as open deviations and ask before choosing.
The earlier no-commit instruction remains in effect: implement and verify after
approval, leave the result uncommitted, then stop for review.

Planning evidence: inspected branch/status, current source, relevant spec,
existing plans/deviations, docs and read-only generic-default/policy acceptance
sources. No runtime, lint, type or documentation checks ran in this planning
turn. `git diff --check` passed with no output.

# A4.3 review fixes — approved, uncommitted

The maintainer requested implementation of `../review/slices/A4.3.md`.
Preserve the prior concrete docstring change; leave all fixes uncommitted
until further review, overriding the review file's suggested commit.

- `src/survjax/distributions/base.py`: F-1, mark each explicitly registered
  class and reject other nested values with field-named TypeError. Do not
  inherit registration from a parent. F-2, use the short tree-map indexing
  form and a spec 4.1 shared-leading-batch-rank comment; delegate index errors
  to JAX. Wrap the long declaration error (N-3). No new public API.
- `tests/unit/test_family_batching.py`: test array/None/unregistered nested
  values and a subclass that inherits a registered parent's marker; update
  the derivative-tolerance import (N-1). Existing indexing tests cover F-2.
- Rename `tests/tolerances.py` to `tests/impl_tolerances.py` as decided by the
  maintainer (N-1). Tighten float32 derivative tolerances to 1e-3 and justify
  the rounding scale (N-2). Do not change the independent tolerance module.
- `docs/index.md`: wrap the flagged line. `docs/distributions.md`: document
  JAX's out-of-bounds integer clamping, as decided by the maintainer.
- Record the review-directed nested-field TypeError policy as open D-015 in
  DEVIATIONS.md; no approval question is needed because the maintainer
  explicitly requested these fixes. Overwrite REVIEW.md with current reading
  order, invariants and exact check results. Spec coverage: 3.3, 4.1–4.2, 5, 6.2, 7.3, 10.5, 12.1.

Run `scripts/check.sh --through A3`, the previous 89 focused A4 nodes in both
modes, `scripts/build_docs.sh`, combined `SPEC_THROUGH=A3` unit/spec pytest
sessions in both modes, the concrete docstring doctest in both modes, and
`git diff --check`. No new dependencies, spec or protected-file edits.

# A4.3 review follow-up — concrete nested-field example

Approved by the maintainer's request to do this edit and leave it uncommitted.
Change only the `distribution()` docstring in
`src/survjax/distributions/base.py`, plus this plan and the review note.
Show complete custom constant-hazard and scaled-hazard families, then construct
and index an actual nested instance with numerical outputs. Use the existing
A4.3 API; no built-in families, numerical defaults or new public API.
Covers spec 4.2, 5 and 10.5. Execute the docstring example in both dtype modes,
run focused Ruff and mypy checks, and record results in REVIEW.md.
No new test file or commit; stop for further review.

# A4.3 — Nested distributions and batch indexing (2026-10-02)

Status: approved by the maintainer ("ok implement it") and implemented on
`A4.3`; see REVIEW.md for evidence. This is the active step plan; earlier plans below are history. Work on the maintainer-created
`A4.3` branch, starting at `138db08` (A4.2 review fix-up), against specification
revision 19. The working tree was clean before this plan update.

## Outcome and public API

Registered frozen families can declare a required nested distribution with
`survjax.distribution()` and select batch entries with `dist[idx]`. Own
parameters and all nested distributions share the outer batch shape after
construction. Nested component axes, dtypes, family types and static
configuration survive broadcasting, indexing and JAX transformations.

Export `distribution()` from both `survjax` and `survjax.distributions`.
Add typed `SurvivalDistribution.__getitem__`; extend the existing
`batch_shape` property and `register_family` behavior. No other public names.
Spec coverage: 3.3 (independent nested dtypes), 4.1–4.2 (representation,
registration and indexing), 5 (batch shapes), 7.2–7.3 (compilation reuse and
parameter transformations), 10.2–10.5 and 12.1 (engineering/docs/tests).

## Files and implementation

- `src/survjax/distributions/base.py`: add the no-argument, no-default
  `distribution()` dataclass field helper. Recognize nested fields alongside
  `param()` and `static()` in declaration checks and document all three roles.
  Include nested objects as dynamic children, so their leaves and static
  auxiliary data participate in the outer PyTree. Reconstruction continues
  to bypass public constructors, normalization and value inspection.
- Extend construction to broadcast own parameter batch shapes together with
  nested `batch_shape` values. Normalize only own parameters, using the existing
  helper and preserving warning attribution. Recursively broadcast nested
  parameter leaves to the common batch shape plus their declared component
  shapes, retaining their original dtypes and static values. Rebuild nested
  objects without public constructors; leave the supplied baseline immutable.
  Incompatible batch shapes raise TypeError. Support wrappers whose dynamic
  fields are exclusively nested distributions.
- Derive `batch_shape` from current own and nested leaves, excluding component
  axes. Store no cached batch shape or extra static shape metadata.
- Implement batch-only indexing for integers, slices, integer arrays and
  tuples, including ellipsis and inserted singleton axes. Resolve indexing
  against the batch rank before appending untouched component-axis slices;
  ellipsis must never consume component axes. Recursively index nested
  distributions and reconstruct without constructors. Follow ordinary JAX
  indexing constraints, including traced integer indices under jit and the
  restrictions on dynamic-size indexing. Do not add custom bounds semantics.
- Keep `base.py` under about 300 lines. Only if necessary, move registration
  internals to `src/survjax/distributions/_registration.py`, preserving public
  imports and caller-site warning behavior; no general helper framework.
- `src/survjax/distributions/__init__.py`, `src/survjax/__init__.py` and
  `tests/unit/test_import.py`: add and check the new export.
- Create `tests/unit/test_family_batching.py` with independently defined custom
  families and small behavior-specific tests. Use existing tolerances;
  derivative-specific additions belong only in `tests/tolerances.py` with a
  scale justification.
- `docs/distributions.md`: add executable nested-field and batch-selection
  examples, explain component axes and independent nested dtypes, and show
  paired `(N,)` versus explicit `(D,1)` evaluation using a simple custom
  `log_survival` implementation at valid positive times.
  `docs/index.md`: update the current-feature description.
- Overwrite `REVIEW.md` after implementation with reading order, invariants,
  spec-to-test mapping, exact checks/results/skips and remaining questions.

## Tests and evidence

- Spec 4.2: helper takes no arguments; nested fields are required; round trips
  retain nested types, static configuration and supplied sentinel leaf identity.
- Spec 4.1–4.2/5: scalar baseline/vector wrapper, vector baseline/scalar wrapper,
  `(2,1)` with `(3,)`, nested-only wrapper and at least two nesting levels;
  incompatible batches fail. Verify stored leaf shapes and explicit values,
  nested components (including two component axes), and unchanged input objects.
- Spec 3.3: x64-only mixed float32/float64 nesting preserves each family's dtype
  through broadcasting and indexing; x32 covers available dtypes without
  changing global configuration or claiming float64 preservation.
- Spec 4.1–4.2: integer, slice, array and multi-axis tuple selection preserves
  components/static fields; ellipsis and singleton insertion operate on batch
  axes only. Compare indexed leaves to independently indexed input arrays.
  Tree-map slicing agrees with batch selection and updates outer/nested shapes.
- Spec 7.2–7.3: eager/jit/vmap construction and selection agree; jit supports
  a traced integer index; vmap sees per-entry nested batch shapes and full
  component vectors. Compile-count tests vary outer and nested dynamic values
  at fixed shapes/dtypes/structure and require reuse; changed nested static
  configuration specializes separately. Use a smooth custom log-survival
  formula with hand-derived derivatives and second-order forward/reverse
  `check_grads` to verify gradients through nested construction and selection.

Run with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests, separately in both dtype modes.
2. Focused read-only A4 acceptance nodes with `JAX_ENABLE_X64=1` and `0`,
   `SPEC_THROUGH=A4`, and `uv run --locked --offline pytest -q`:
   - `test_a4_registration.py`: all except `test_methods_do_not_mutate_parameters`.
   - `test_a4_parameters.py`: all except
     `test_construction_with_traced_parameters_inside_jit`,
     `test_construction_inside_vmap` and `test_gradient_through_construction`.
   - `test_a4_indexing.py`: all except
     `test_indexed_distribution_matches_indexed_values`,
     `test_vmap_over_distribution_matches_batched_call` and
     `test_vmap_over_distribution_and_time_pairs_entries`.
   - `test_a4_nested.py`: all except
     `test_vmap_over_outer_distribution_with_broadcast_base` and
     `test_broadcast_nested_values_in_log_prob`.
   These excluded nodes require later generic numerical methods. Record the
   exact node list and results in REVIEW.md; do not edit, skip or xfail tests.
3. `scripts/build_docs.sh`: execute examples in separate x32/x64 processes and
   inspect the rendered guide. Report any environment/permission limitations.
4. `git diff --check`.

## Boundaries and stopping point

Direct-method policy/defaults remain A4.4–A4.5; built-ins, grids and actual
PH/AFT wrappers remain later steps. D-013 remains an A6 diagnostic-contract
handoff. Do not change dependencies, the spec, independent acceptance tests
or protected scripts. Log any newly uncovered spec ambiguity as an open
DEVIATIONS.md entry and ask before choosing its behavior.

Aim for roughly 300 changed lines including code and tests. If the necessary
implementation grows past that, propose a smaller split before expanding it.
After approval, implement one concept on `A4.3`, write REVIEW.md, commit with
an imperative subject and a body explaining why, then stop for review.
This plan-only update remains uncommitted.

Planning evidence: read the relevant spec, A4.2 implementation/review,
dtype helper, docs and read-only A4 acceptance sources. No runtime, lint,
type, derivative or documentation checks were run for this planning update.

# A4.2 review fix-up (2026-10-02)

Status: approved by the maintainer on 2026-10-02 after reading
`../review/slices/A4.2.md`, with public-docstring improvements added. One extra commit on `A4.2`, then stop.
No A4.3 implementation or documentation-page rewrite.

## Changes

- `src/survjax/distributions/base.py`, F-1: reject dataclass fields whose
  metadata does not identify an available field role (`param` or `static`),
  with a field-named TypeError at registration. Record the unspecified plain
  field policy in D-014; the maintainer approved this implementation policy.
- N-1: validate `component_ndim` in `param()`: a non-negative Python int,
  excluding bool. Use TypeError for wrong kinds and ValueError for negatives.
- N-2: require an initializer owned by the class before wrapping it. Reject
  inherited initializers with a TypeError explaining that the subclass must
  be decorated with `@dataclass` itself. This prevents double normalization
  and incorrect warning attribution; retain the documented decorator recipe.
- N-3: catch static hashability TypeError and re-raise naming the field and
  stating that static fields must be hashable.
- Rewrite the public field helpers, base, batch-shape property, extension hooks
  and registration docstrings for users: purpose, arguments, shapes, errors and
  one short declaration example. Keep spec/development bookkeeping out of them.
- `tests/unit/test_family_registration.py`: add separate tests for plain
  default/default-factory field rejection; invalid component counts; inherited
  initializer rejection and a re-decorated subclass's x32 warning location;
  field-named static hashability errors; insufficient component rank and
  non-frozen registration guards. Keep tests independent of tests/spec.
- `DEVIATIONS.md`: add open D-014. Do not change existing statuses or spec.
- `REVIEW.md`: overwrite with reading order, requirement coverage, exact
  check outcomes and remaining scope; correct the earlier claim that no
  registration ambiguity existed.

Public names stay unchanged. These are declaration/error checks for the
extension interface (spec 3.3, 4.1–4.2, 5); no parameter-value checks or
numerical-method work. Aim below 300 changed lines including tests.

## Verification

Run `scripts/check.sh --through A3`, the same 60 focused A4 acceptance nodes
listed in the prior REVIEW.md in both dtype processes, `scripts/build_docs.sh`
in both modes, and `git diff --check`. Use
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Record dtype-specific skips and
any failures. Commit this fix-up only after checks pass, then stop.

Planning checks: read the external review and current implementation;
working tree was clean before this plan/deviation update. Runtime checks
were not rerun for this plan-only task.

# A4.2 — Registration and flat family construction (2026-10-02)

Status: approved by the maintainer ("go for it and build it") and implemented
on the existing `A4.2` branch; see REVIEW.md for verification. This is the active step plan; the broader A4 sequence below remains
background. Based on specification revision 19 and completed A4.1 at
`15cfc6f` (current HEAD also contains the D-013 handoff note).

## Outcome and scope

A custom frozen dataclass can be explicitly registered, constructed by keyword
with normalized and broadcast parameter arrays, and passed through JAX as an
immutable PyTree. This step covers spec 3.3, 4.1–4.2, construction shapes in
section 5, and engineering/documentation requirements in 10.2–10.5.

Public additions: `SurvivalDistribution`, `register_family`,
`param(component_ndim=0)` and `static(...)`, exported from both `survjax` and
`survjax.distributions`. The base exposes typed `batch_shape` and the typed
`log_survival(t)` / `reference_time()` extension hooks. Hooks without family
implementations fail explicitly; no numerical defaults are supplied yet.
No base `shape` attribute: a family's own `shape` parameter remains usable.

Nested `distribution()` fields and batch indexing stay in A4.3. Direct-method
policy/defaults stay in A4.4–A4.5; built-ins and grids stay in A5. D-013 stays
open for A6. The guide will state these intermediate limitations explicitly.
No dependency, spec, protected-script or independent-test changes.

## Files and implementation

- Create `src/survjax/distributions/base.py` and its package `__init__.py`.
  Use dataclass field metadata to distinguish parameters from static fields,
  ordinary functions and explicit `jax.tree_util` registration. Require the
  documented frozen-dataclass subclass recipe; no automatic registration,
  metaclasses or dynamic attribute lookup. Keep the module under about 300 lines.
- Wrap the dataclass-generated constructor to reject positional arguments,
  preserving keyword argument errors, required parameters and static defaults.
  Support `kw_only=True`, including a defaulted static field before parameters.
  Static configuration must be hashable. Do not add parameter-value validation.
- Reuse `_dtypes.normalize_parameters` for all own parameters, including
  components. Pass the stack level needed for warnings to point to the user's
  constructor call. Constructors accept no `dtype` keyword.
- Derive each parameter's batch axes by excluding its declared trailing
  component axes. Broadcast those batch shapes, then broadcast each array to
  the common batch shape plus its original component shape. Translate
  incompatible batch shapes into `TypeError`. Never broadcast component axes
  against batch axes or inspect parameter values on the host.
- Flatten arrays as dynamic children, with static configuration in auxiliary
  data. Unflatten using allocation and frozen-field assignment, bypassing the
  constructor, normalization and broadcasting; supplied leaves retain identity,
  including JAX placeholders. Compute `batch_shape` from current leaves rather
  than store cached shape metadata, so tree mapping and vmap update it naturally.
- Update `src/survjax/__init__.py` and `tests/unit/test_import.py` for the four
  new names; preserve existing exports.
- Add `tests/unit/test_family_registration.py` with small independent custom
  families. Do not import acceptance-suite recipes or helpers.
- Create `docs/distributions.md` with an executable construction/PyTree example,
  decorator order, keyword-only recipe, static defaults, component axes, dtype
  behavior and absent `shape` alias. Add it to `docs/index.md` and update that
  page's current-feature description. Use supported construction operations;
  label numerical defaults, nested fields and indexing as planned.

If registration edge cases require choices the specification does not settle,
record a new open deviation and ask before implementing that behavior. If code,
unit tests and docs exceed roughly 300 changed lines, propose a further split
before expanding the step; approval here does not authorize a larger step.

## Tests and evidence

- Spec 4.1–4.2: immutability, explicit registration only, required/keyword-only
  arguments, static defaults/overrides and the static-first kw-only recipe.
- Spec 3.3: public-constructor dtype integration, components sharing dtype,
  rejected nonreal inputs and None, unavailable-float64 warning count and caller
  location, and rejection of `dtype=`. Reuse A4.1's detailed promotion tests.
- Spec 4.1–4.2/5: scalar and multi-axis broadcasting, component preservation,
  incompatible batch shapes, stored negative/NaN/infinite natural parameters,
  and a family with its own `shape` field alongside a family without one.
- Spec 4.2/7.2: round trips, static tree-structure equality/difference, sentinel
  identity on unflatten, tree-map slicing, eager/jit construction, and vmap
  construction/per-entry `batch_shape`. Use exact shape/value assertions where
  applicable; no derivative or direct-method policy claims in this step.

Run with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests, in separate dtype processes where applicable.
2. Focused A4 acceptance nodes in separate processes with
   `JAX_ENABLE_X64=1` and `0`, `SPEC_THROUGH=A4`, and
   `uv run --locked --offline pytest -q`. Select registration tests except
   the all-extension-names, method-purity and three nested-distribution tests;
   select parameter tests except `test_nested_distribution_keeps_its_own_dtype`,
   `test_construction_with_traced_parameters_inside_jit`,
   `test_construction_inside_vmap` and `test_gradient_through_construction`
   (these three use the later generic `log_prob`);
   select `test_a4_indexing.py::test_tree_map_slicing_updates_batch_shape` and
   `::test_batch_shape_inside_vmap_is_the_per_entry_shape`. Inspect dependencies
   before execution and record the exact node selection in REVIEW.md. Never
   skip/xfail or edit unfinished acceptance tests, or claim the full A4 gate.
3. `scripts/build_docs.sh`: execute examples in x32/x64 and inspect the rendered
   distribution guide. Report any permission or environment limitation honestly.
4. `git diff --check`.

Overwrite `REVIEW.md` with reading order, construction/PyTree invariants,
requirement-to-test mapping, exact commands/results/skips and open questions.
Commit this one implementation step on `A4.2` with an imperative subject and
why in the body, then stop. The plan-only update remains uncommitted.

Planning checks: read the spec, existing implementation/docs, A4.1 review and
read-only A4 acceptance sources. Working tree was clean on `A4.2`. No runtime,
lint, type or documentation checks were run for this planning update.
`git diff --check` passed with no output.

# A4.1 review fix-up (2026-10-02)

Approved by the maintainer's request to implement `../review/slices/A4.1.md`.
One fix-up commit on `A4.1`, then stop; no A4.2 implementation.

- F-1: reject explicit None own parameter leaves with a field-named TypeError
  in `_dtypes.py`; add the case to `test_parameter_dtype.py` (spec 3.3, 4.2).
- N-1: include a float32 leaf in the x32 warning test, so a single float64
  leaf must still warn (spec 3.3).
- N-2: pass the warning noun explicitly from `_data_validation.py` and
  `normalize_parameters` rather than derive it from input-kind strings.
- Public API unchanged. Preserve observation errors and warning attribution.
- Run `scripts/check.sh --through A3`, `scripts/build_docs.sh` in both dtype
  processes, and `git diff --check`; overwrite REVIEW.md with results.
- Remove the obsolete D-001 template as requested in review bookkeeping.
  Preserve the existing unstaged DEVIATIONS.md planning edit; no deviation
  statuses are changed by this fix-up.

# Step A4: distribution base and extension interface (2026-10-02)

Status: A4.1 approved by the maintainer on 2026-10-02; later parts await approval. Based on specification revision
19; re-evaluated against the synced A4 acceptance suite at source commit
`f62c81ff5b2317ecf692e546a2d3beeafd5c683f`. A4.1 implementation is on the maintainer-created `A4.1` branch. Earlier plans below
are history; `REVIEW.md` continues to describe the completed A3 work.

## Scope and decisions needed

Implement the Slice A distribution foundation: frozen registered families,
parameter/static/nested fields, construction and reconstruction, batch indexing,
and the seven pointwise direct methods. Cover sections 3.3, 4.1, 4.2, 4.5, 5,
7.2–7.3, 10.2–10.5 and the relevant parts of 12.1–12.2.

New top-level exports: `SurvivalDistribution`, `register_family`, `param`,
`static`, `distribution`. Existing exports remain. Methods added by completion:
`batch_shape`, `__getitem__`, `reference_time`,
`log_survival`, `survival`, `log_prob`, `log_cdf`, `cumulative_hazard`,
`log_hazard`, `hazard`. `log_survival` and `reference_time` are the required
family primitives. There is no base `shape` attribute and no constructor
`dtype` keyword. Signatures, public fields and extension interfaces are typed.

The nine A4 acceptance files are now present: 203 tests collect in each dtype
mode. Read-only `API_ASSUMPTIONS.md` Q-019–Q-025 and `TESTMAP.md` confirm the
scope below. In particular, Q-024 explicitly leaves `parameter_issues()` and
its unspecified return type for A6. Remove that API and generic nested-issue
aggregation from A4; D-013 remains open as an A6 handoff, not an A4 blocker.
Do not infer domain validity for arbitrary custom parameters. A4 preserves
NaNs produced by family formulas and generic defaults; its acceptance examples
exercise a negative rate through `log_prob` and NaN parameters through direct
methods. Passing them is not evidence for general parameter diagnostics.

Built-in Exponential/Weibull families, log-parameter constructors and survival
grids remain A5. Likelihoods/result containers remain A6. Interval methods,
wrappers, summaries, sampling, `HazardFamily` and `check_family` remain their
later slices. Use small analytical custom families in tests and documentation
to exercise A4 without introducing built-ins early.

## Review-sized implementation sequence

A4 needs multiple commits to respect the roughly 300-line review target,
including tests. Each numbered part below is a separate step: confirm its
scope in this plan, obtain approval, implement with tests, overwrite `REVIEW.md`,
commit on the maintainer's branch, then stop. Approval to begin A4.1 does not
authorize automatically continuing through the remaining parts. If a part
exceeds the target, propose a further split before expanding it.

### A4.1 — Shared floating-dtype normalization

- Files: new `src/survjax/_dtypes.py`; update `_data_validation.py`; add
  `tests/unit/test_parameter_dtype.py`. Reuse the existing dtype logic by a
  narrow extraction rather than coupling distributions to observation rules.
  Preserve observation behavior, error meaning and warning attribution.
- Normalize each family's own parameters to one dtype, including components;
  nested distributions retain their dtype. Preserve weak promotion, the
  configured default for integer/scalar-only inputs, and the float32 minimum.
  Reject bool, complex, string and object inputs. Warn once per user call site
  for float64 arrays under x32; do not warn for Python scalars/lists. Never
  inspect traced values on the host or change global JAX configuration.
- Tests: dtype combinations, float16 promotion, weak inputs, rejected input
  kinds, x32 warning count/location and traced normalization. Existing data
  dtype tests protect the extraction. Public API unchanged (spec 3.3).
- Docs: explain the shared policy in `docs/survival-data.md`; mark distribution
  examples as planned until registration exists.

### A4.2 — Registration and flat family construction

- Files: new `src/survjax/distributions/base.py` and package `__init__.py`;
  update top-level `__init__.py` and `tests/unit/test_import.py`; add
  `tests/unit/test_family_registration.py`.
- Add the base, `param(component_ndim=0)`, `static(...)` and explicit
  `register_family`. Use dataclass field metadata and small ordinary functions;
  no metaclasses, dynamic attribute lookup or automatic subclass registration.
- Enforce keyword-only construction, required parameter fields and optional
  static defaults. Normalize parameter dtypes; broadcast leading batch axes
  while retaining trailing component axes. Incompatible batch shapes raise
  `TypeError`. Construction does not validate parameter values.
- Register dynamic arrays as children and hashable static configuration as
  auxiliary data. Reconstruct using allocation/field assignment, bypassing
  construction and normalization, including for JAX placeholders.
  Derive `batch_shape` from current leaves, never cached shape metadata.
- Tests: immutability; required/keyword-only arguments; static defaults;
  component and batch shapes; dtype integration; eager/jit construction;
  flatten/unflatten and tree mapping; no implicit registration; absent base
  `shape`. Explicitly test the `kw_only=True` recipe with a defaulted static
  field before required parameters, unchanged object identity for unflattened
  sentinel leaves, and storage of negative/NaN/infinite parameter values.
  Test malformed static structure without checking parameter values.
- Docs: add `docs/distributions.md` to `docs/index.md`, showing the decorator
  order, field roles and construction; label methods not yet implemented.
  Spec: 3.3, 4.1–4.2, 5, 10.2–10.5.

### A4.3 — Nested distributions and batch indexing

- Files: extend `distributions/base.py`; add a focused private
  `distributions/_registration.py` if needed to keep responsibilities and
  modules small; export `distribution()`; update import tests; add
  `tests/unit/test_family_batching.py`.
- Include nested batch shapes in outer broadcasting. Broadcast nested leaves
  recursively, preserving their component axes, own dtypes and static fields.
  The no-argument `distribution()` field is a required nested distribution.
- Implement batch-only indexing, applying indices to leading axes while
  retaining component axes. Rebuild without public constructors. Support
  scalar/slice/array/tuple indexing within ordinary JAX indexing constraints.
- Tests: nested scalar/vector/multi-axis broadcasting; incompatible shapes;
  nested component axes and mixed dtypes; indexing, tree-map slicing and
  vmap agreement; per-entry `batch_shape` inside vmap. Confirm changed dynamic
  values reuse compilation with equal shapes/dtypes/structure/configuration,
  and changed static configuration may specialize separately.
- Docs: expand `docs/distributions.md` with nested fields and batching;
  distinguish paired `(N,)` evaluation from explicit `(D,1)` broadcasting.
  Spec: 3.3, 4.1–4.2, 5, 7.2, 12.1.

### A4.4 — Direct-method policy and survival/CDF defaults

- Files: `distributions/base.py`,
  registration integration, new private `distributions/_direct.py` for the
  shared value-policy machinery; `tests/unit/test_family_direct.py`; create
  `tests/tolerances.py` for numerical tolerances with the mandated scales.
- Implement `survival = exp(log_survival)`,
  `log_cdf = log(-expm1(log_survival))`, and
  `cumulative_hazard = -log_survival`.
  Keep the log-CDF calculation stable without probability floors.
- Apply section 4.5 at explicit registration to both inherited defaults and
  family overrides. Use one understandable policy wrapper, preserving raw
  family implementations for internal composition and avoiding recursion.
  Static time/batch mismatch raises `TypeError`; numerical time values follow
  the total-function policy. Do not confuse this with observation validation.
- For negative/NaN times, substitute a safe stopped-gradient reference before
  invoking family code; select policy outputs afterwards. Pass zero and positive
  infinity to the family. Preserve NaNs from family calculations without
  inventing generic domain checks for unknown custom parameters. Retain
  `reference_time()` as the required family hook; our choice to use it for
  safe negative/NaN-time substitution needs our own tests because the A4
  acceptance suite does not test that hook directly.
- Tests: analytical exponential-survival recipe; scalar/paired/draw broadcasting;
  all time-policy rows for these four methods, defaults and overrides; NaN
  parameter propagation;
  family code never receives negative/NaN times; finite parameter gradients
  through inactive branches and reference-input isolation. Test log-CDF tails
  against analytical/mpmath values in representable ranges.
- Docs: runnable primitive-family example in `docs/distributions.md`, direct
  method boundaries, dtype behavior and precision limits of the CDF default.
  Spec: 4.2, 4.5, 5, 7.2, 12.1.

### A4.5 — Autodiff density/hazard and A4 completion

- Files: `distributions/base.py`, `_direct.py`; new
  `tests/unit/test_family_autodiff.py`; extend `tests/tolerances.py` with
  justified derivative scales; finish `docs/distributions.md` and update
  `docs/playground.ipynb` with a short custom-family example.
- Compute the elementwise time derivative of log survival with forward-mode
  autodiff after broadcasting, holding parameters fixed. An all-ones time
  tangent yields the pointwise derivative; do not differentiate a batch sum
  or materialize a full Jacobian. Use it for generic log hazard and log density;
  do not recover hazard by subtracting two large log values. Keep the density
  calculation independent of the public `log_hazard` result: its zero-survival
  NaN policy must not contaminate a density that should be `-inf`.
- Supply `hazard = exp(log_hazard)`. Apply the same policy to density/hazard
  overrides. Detect zero survival from `log_survival == -inf`, not from
  `exp(log_survival) == 0`: finite log survival such as `-1000` still gives a
  valid generic hazard. Generic log hazard is NaN at true zero survival,
  including finite support endpoints; analytic overrides retain their values,
  and generic `hazard` uses the `log_hazard` override. Enforce
  `log_prob(+inf)=-inf` for valid
  parameters. Substitute safe inputs before logarithms and other potentially
  invalid inactive operations; use no clipping, epsilon or probability floor.
- Tests: hand-derived density/hazard for exponential and a smooth nonlinear
  cumulative-hazard recipe; bounded-support and infinite-time generic/override
  distinctions; zero hazard; eager/jit/vmap agreement; nested transformed
  construction. Cover zero-time density values for power-family shapes below,
  equal to and above one; negative-rate density NaNs and isolation of invalid
  batch entries; large cumulative hazards and tiny positive survival.
  Use `check_grads(order=2, modes=("fwd", "rev"))` on smooth parameter
  interiors in float64, with an explicit float32 skip for noisy second-order
  finite differences. Check analytical first derivatives and Hessians in both
  modes, including jit/vmap composition. Test finite gradients for the base's
  negative-time substitutions and use boundary-safe custom recipes for any
  additional boundary-gradient checks; do not promise that the base repairs
  arbitrary custom derivatives at zero or infinity. Built-in Weibull's `S(0)`
  shape-gradient requirement remains A5. Expected values come from algebra or mpmath,
  never from the implementation being tested. State the extra derivative order
  required of a primitive when density comes from autodiff.
- Finish the seven-method policy matrix and run the synced independent A4
  suite. Examples explain the extension contract and generic numerical limits;
  keep built-in grids, likelihood debug-NaN batches, R fixtures, wheel tests and
  performance baselines assigned to their later Slice A steps.
  Spec: 4.2, 4.5, 7.2–7.3, 10.5, 12.1–12.2.

## Acceptance coverage and checks at each stop

All paths below are under read-only `tests/spec/`. Keep implementer tests
independent: do not import `spec_families.py`, helpers or tolerances from there.

| Acceptance file | Planned coverage |
| --- | --- |
| `test_a4_registration.py` (25) | A4.2 registration, constructors and PyTrees; A4.3 nested fields; method-purity checks complete with A4.5 |
| `test_a4_parameters.py` (42) | A4.1 dtype helper, A4.2 public integration, A4.3 nested dtypes; tests using default density complete with A4.5 |
| `test_a4_indexing.py` (17) | A4.3 indexing and current-leaf batch shapes; density comparisons complete with A4.5 |
| `test_a4_nested.py` (14) | A4.3 broadcasting, components, dtypes and structure; density/vmap comparisons complete with A4.5 |
| `test_a4_generic_defaults.py` (23) | A4.4 survival/CDF/cumulative hazard; A4.5 hazard defaults |
| `test_a4_policy_enforcement.py` (20) | A4.4 primitive and derived policy; A4.5 density/hazard, overrides and inactive gradients |
| `test_a4_log_prob_default.py` (31) | A4.5 autodiff, shape/dtype, dispatch, gradients and Hessians |
| `test_a4_value_policy.py` (19) | A4.5 zero/negative/infinite/NaN times, invalid density and elementwise isolation |
| `test_a4_hazard_at_zero_survival.py` (12) | A4.5 true zero versus underflow, bounded support and analytic override dispatch |

These files cross substep boundaries. Use relevant explicit node selections
for intermediate feedback when their dependencies exist, in both dtype
processes, recording the selection; do not claim an entire file passes early,
change its step markers, or skip/xfail unfinished tests. The complete A4 gate
is required after A4.5.

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` for uv-based commands.
For A4.1–A4.4 run `scripts/check.sh --through A3` (all implementer tests
including the new ones, plus prior acceptance tests) and
`scripts/build_docs.sh`, both using separate x32/x64 processes. At A4.5 run
`scripts/check.sh --through A4` and the docs build. Run `git diff --check` at
every stop. The gate already includes Ruff formatting/lint and mypy.

Record exact commands, test counts, skips and failures. In particular, report
dtype-specific skips (float64 normalization cases and float64-only finite
differences) separately from failures.
Inspect the rendered distribution guide after its runnable examples are added.
No dependency additions, R execution or spec edits are planned.

Each `REVIEW.md` gives reading order, construction/PyTree/numerical invariants,
tests mapped to spec requirements, actual check output and open questions.
Each commit has an imperative subject and a body explaining why. Stop after
each commit; never commit to `main`. This planning change is left uncommitted.

Planning re-evaluation checks (2026-10-02): with the cache directory above,
`JAX_ENABLE_X64=1 SPEC_THROUGH=A4 uv run --locked --offline pytest --collect-only -qq tests/spec/test_a4*.py`
and the same command with `JAX_ENABLE_X64=0` both exited 0 and collected 203
A4 tests across nine files. Collection does not execute tests or establish
implementation correctness. No implementation tests, type/lint checks or docs
build were run for this plan-only update. `git diff --check` passed.

# A3 review fix-up: constructor consistency and metadata evidence (2026-10-01)

Status: approved and implemented on 2026-10-01; A3 gate passed. Source: `A3-review-handoff.md`,
reviewing `791517a`. One extra commit on the existing `A3` branch, targeting
under 300 changed lines including tests. Follow the handoff's explicit
plan/approval/implementation/review/stop sequence.

## Files and changes

- `src/survjax/_data_metadata.py`: replace recursive tracer detection
  with `jax.tree.leaves`; retain the separate module and existing kind
  precedence. Add the spec 3.3 explanation that `(0, +inf]` is right censored.
  Implement item 8: compute the six presence reductions with `jnp` on the
  device, stack the scalar results, and call `jax.device_get` once on that
  six-element array before converting to Python bools. Avoid transferring
  observation arrays to NumPy. Keep `may_contain` inside the compile-time
  evaluation block on the concrete path only; never pass it tracers.
- `src/survjax/data.py`: extend the bounds constructor's compile-time
  block through all issue calculations, making its six leaves concrete at
  trace time when supplied inputs are concrete. Add the requested JAX
  out-of-range clamping sentence to `__getitem__`'s docstring.
- `tests/unit/test_data_metadata.py`: add explicit scalar `(0, +inf]`
  precedence, NaN truncation and negative-entry flag/issue cases; check all
  six leaves inside jit for both concrete constructor forms; assert both
  forms retain the host-only `ValueError` (not `InvalidDataError`) when
  `raise_if_invalid()` is invoked inside jit; cover conservative metadata
  from vmap construction for both forms; extend concrete-inside-jit tests
  to compare every leaf with eager construction.
- `DEVIATIONS.md`: add D-012, step A3, spec 3.2, status open, documenting
  overlapping kind predicates at `(0, +inf]`, the existing right precedence
  following 3.3, and the proposed clarification for a future spec revision.
- `REVIEW.md`: overwrite with reading order, invariants, requirement/test
  mapping, actual check results and the A6 handoff below.

Public API: signatures and exports stay as implemented; `__getitem__`
documentation gains an explicit description of its existing behavior.
Spec coverage: 3.2, 3.3, 6.2, 12.1. No numerical likelihood implementation.

## Precise review wording and A6 handoff

Presence flags are not validity checks. Preserve the spec 3.2 invariant that
metadata includes normalized rows with issue bits. NaN bounds and negative
entry can leave truncation flags false despite invalid data; this does not
mean all invalid bounds necessarily leave those flags false. Invalidity is
determined from the stored issue bits.

Concrete construction requires one host synchronization to obtain static
Python flags (spec 3.2), transferring only six booleans. Slices inherit the
parent flags and perform no metadata synchronization. Existing metadata tests
and the new precedence/invalid-truncation tests cover item 8's unchanged
behavior; add no dedicated test or timing measurement. Large-N construction
timing belongs with the separate spec 12.3 CPU baseline.

For concrete supplied inputs, both constructors' six stored leaves are
evaluated at compile time inside jit. Subsequent `data.issues` decoding inside
jit is still traced, and its host checkpoint must fail explicitly there.
For traced supplied inputs, operations involving those inputs remain traced
and all six metadata flags are true; do not promise that constant-only
intermediate operations become tracers.

Record for A6:

- Reuse the normalized kind definitions from `_data_metadata.py` so row
  dispatch and static flags agree. `may_contain` returns static Python flags
  after a host transfer, not jit-safe row masks. A6 must plan any shared
  pure-JAX row classification needed for likelihood use; add no such helper
  in this fix-up.
- Issue-bearing rows must produce NaN and `invalid_data` from `_issue_bits`
  regardless of metadata pruning. Test NaN truncation with false flags.
- Resolve how Slice A detects unimplemented modes on traced data: all-true
  metadata alone cannot establish that a row uses an unsupported mode.
  The A6 plan must address row-value detection and explicit failure under
  transformations before implementing likelihoods.

## Verification and finish

Run `scripts/check.sh --through A3` in the existing two dtype processes,
with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`, and `git diff --check`.
Report actual results and any skips. Keep spec/tests/spec read-only and
exclude the existing notebook edit and maintainer-authored handoff from
staging. Commit as an extra A3 fix-up, without amend/rebase, then stop.

# Step A3: indexing and static observation metadata (2026-10-01)

Status: approved by maintainer and implemented on 2026-10-01; A3 gate passed.
Branch: existing maintainer-created `A3`. One implementation step and commit,
targeting about 300 changed lines including tests. If the implementation needs
a larger change, propose a split before expanding it. Earlier plans below are
history. The existing local edit to `docs/playground.ipynb` belongs to the
maintainer and is excluded from this step.

## Starting point and scope

A2 already supplies both pure constructors, the five normalized fields, the
sixth issue-bitfield leaf, diagnostics, and reconstruction without normalization.
`SurvivalData` currently has no `__getitem__`; its six auxiliary flags always
default to true. A3 completes indexing and value-derived concrete metadata.

Public API touched: add typed `SurvivalData.__getitem__(idx) -> SurvivalData`
and document its observation-axis indexing behavior. Both existing constructors
gain the specified metadata behavior without changing signatures or exports.
`data.issues` continues to decode masks and reduce `ok` over the resulting
object. Metadata remains private. No likelihood evaluation or branch pruning
is introduced; those belong to later steps/slices.

Spec coverage: 3.2 (indexing, six aligned leaves, auxiliary flags and
reconstruction), 6.2 (input tracing classification and indexed diagnostics),
12.1 (metadata, stored issues and transformation verification), and the data
portion of the Slice A gate in 12.2. Preserve the dtype policy in 3.3.

## Implementation and files

1. `src/survjax/data.py`
   - Apply `idx` to every array leaf, including the issue bits, and rebuild
     with the parent's auxiliary flags through the existing PyTree machinery.
     Do not reconstruct through a public constructor or reclassify a subset.
   - Support ordinary JAX array indexing: scalar/negative integer, slices,
     integer-array minibatches, concrete boolean masks, and multidimensional
     tuples. Under transformations, retain JAX's indexing constraints (for
     example, dynamic boolean masks do not have a fixed output shape).
   - Classify the supplied observation inputs at constructor entry, before
     normalization turns concrete inputs into tracers. Inspect every argument
     listed in section 6.2, including nested lists/tuples; `dtype` is not an
     observation input. Any tracer forces all six flags true.
   - For entirely concrete inputs, normalize and calculate flags in a
     compile-time evaluation context so concrete construction inside `jit`
     behaves like eager construction. Inputs already traced stay traced;
     unrelated traced function arguments do not affect this choice.
   - Pass the calculated flags explicitly when constructing either form.
     Keep `tree_unflatten` free of classification and validation so it also
     accepts JAX's reconstruction placeholders.
2. `src/survjax/_data_metadata.py` (new, small private module)
   - Hold the input-tracer check and concrete normalized-field classification.
     This responsibility is separate from row validation; `data.py` already
     exceeds 300 lines after the approved repr work, so do not add a general
     refactor or a second normalization implementation.
   - Reduce presence of exact/right/left/interval kinds and, independently,
     `left_trunc > 0` and `right_trunc < +inf` across all normalized rows,
     including invalid rows. Derive kinds from normalized endpoints, not the
     original keyword choices. Store only six Python bools in auxiliary data.
   - Convert only concrete values to host values; never convert a tracer.
3. `tests/unit/test_data_indexing.py` (new)
   - Cover indexing, preserved raw issue evidence, reconstruction, and
     transformed indexing as described below.
4. `tests/unit/test_data_metadata.py` (new)
   - Cover constructor classification, tree structures and compilation reuse
     as described below. Use existing dtype fixtures and exact assertions.
5. `REVIEW.md`
   - Overwrite after implementation with reading order, invariants, tests
     mapped to spec requirements, exact check results and remaining questions.
     Update this plan with the final scope if needed. Record any new spec
     question in `DEVIATIONS.md` and ask before choosing its behavior.

## Tests to add

- **Aligned indexing (3.2):** scalar, negative, strided/empty slice,
  repeated integer indices, concrete boolean selection and multidimensional
  indexing preserve each field's values, shape and dtype. Compare with
  explicitly indexed parent arrays, the identity required by the spec.
  Check agreement with `jax.tree.map` slicing and parent immutability.
- **Stored diagnostics (3.2, 6.2):** select rows whose normalized fields hide
  a raw-input fault (such as an invalid event indicator). Assert hand-derived
  issue masks survive selection and tree reconstruction. A valid subset of
  an invalid parent has scalar `ok=True`; an invalid subset has `ok=False`.
  Empty subsets retain parent flags and have no invalid rows.
- **Transformations (12.1):** eager, `jit` and `vmap` indexing agree for
  supported fixed-shape indices; decoded masks and scalar validity work
  inside `jit`. No numerical derivative checks are needed for this step.
- **Concrete flags (3.2):** both constructors yield the same flags for the
  same normalized data; separate kind and truncation changes alter the tree
  structure. Explicit default truncation matches omitted truncation. Include
  an interval flag with an infinite upper bound and rows with issue bits.
  Equal flags with different observation counts still give equal treedefs.
- **Input classification (3.2, 6.2):** concrete NumPy/Python inputs and
  captured concrete JAX arrays inside `jit` retain eager flags, including
  when another function argument is traced. An observation tracer, including
  one nested in a list/tuple or created by `jnp.asarray` inside `jit`, forces
  all flags true. Cover both constructor forms; compare leaves and issue
  masks separately because eager/traced metadata can intentionally differ.
- **Inheritance and compile counts (3.2, 12.1):** same shapes/dtypes/flags
  with different values trace once; different flags trace separately;
  equal-shape subsets of a mixed parent retain its flags and reuse a
  compilation. Traced construction and concrete data containing all kinds
  and both truncations share a tree structure and compilation.

## Verification and handoff

After approval and implementation, run `scripts/check.sh --through A3`:
the protected-path guard, Ruff formatting/lint, mypy, implementer tests and
independent acceptance tests through A3, with tests in separate float64 and
float32 processes. Run `git diff --check` as well. Record actual output and
any failure; do not claim A3 complete until the gate passes.

D-010 remains formally open, but the current acceptance test now checks
`batched[i].issues.ok`, matching the scalar-reduction contract. Its original
test discrepancy appears addressed; confirm through the A3 gate and leave
formal closure to the maintainer. Do not alter the read-only spec tests.

Finish with one commit on `A3` (proposed subject: `Add observation indexing
and static metadata`; body explains aligned issue preservation and compilation
reuse), the updated review, and a stop. Implementation/check evidence is in
`REVIEW.md`. The bounds constructor evaluates the normalized fields needed for
metadata at compile time; its issue-bit calculations remain pure JAX operations.
The implementation and tests total about 300 substantive changed lines;
normalization-block indentation accounts for additional diff lines.

# DOC-4 review fix-ups (2026-10-01)

Maintainer review of `1a15813`: print right censoring as its set
`(time, inf]` instead of `time+` (all censored rows now print as sets), and
make `docs/playground.ipynb` more informative. Extra commit on
`docs-startup`. Playground: short explanations between cells, the guide's
cohort (every censoring kind, truncation), stored fields, bad data with
`data.issues` and the host checkpoint (printed, so "Run all" completes),
`jax.jit`, and suggestions to try. No asserts, no saved outputs. Keep the
maintainer's `data` cell. Update repr tests and guide prose accordingly.

# Step DOC-4: readable reprs for SurvivalData and DataIssues (2026-10-01)

Requested by the maintainer after DOC-3 ("make a nice repr for SurvivalData
and DataIssues as well"). The spec does not cover repr. This replaces DOC-3's
"no summary repr" choice and adds no public API. One step, one commit on
`docs-startup`. Spec sections touched: 3.2 (private fields stay private), 6.2
(issue names), 6.3 (containers remain pytrees, unchanged).

## Behaviour

Both reprs read concrete values on the host. If any leaf is a JAX tracer, they
fall back to a short form with no value conversion (CLAUDE.md: never convert
tracers to NumPy), e.g. `SurvivalData(shape=(6,), dtype=float32, traced)`.

`SurvivalData`, eager: one line per observation in R-like notation, plus the
eligibility window when it is not the default `(0, inf]`. Invalid rows are
printed as stored, with a footer pointing at `.issues`:

```text
SurvivalData: 6 observations, float64
  0  4
  1  12+
  2  7
  3  9+
  4  (6, 9]   window (2, inf]
  5  (0, 3]   window (2, inf]
```

- Notation: exact `t`, right censored `t+` (as in R's print), every other
  censored row as its stored set `(lower, upper]`, so left censoring prints
  `(0, 3]` (maintainer's choice). Numbers use `:g`.
- More than 20 rows: first 10, `...`, last 5. N-d data: header shows the
  shape and rows show their index tuple.
- Footer only when non-empty: `issues: 4 invalid rows, 1 uninformative
  (see .issues)`.

`DataIssues`, eager: the overall verdict, then only the rules that fire,
with the row count and first index (the same wording as
`raise_if_invalid`):

```text
DataIssues: 4 of 5 rows invalid
  invalid_time            1 row, first index (0,)
  missing_time2           1 row, first index (1,)
  invalid_interval        1 row, first index (2,)
  censored_before_entry   1 row, first index (3,)
  uninformative           1 row, first index (4,)  (not an error)
```

A valid result prints `DataIssues: all 6 rows valid`.

## Files

Approved with changes: each repr lives next to its class, not in a separate
module, accepting files somewhat over 300 lines.

- `data.py`: `@dataclass(..., repr=False)` plus `__repr__` and two small
  formatting functions. The DOC-3 `field(repr=False)` declarations become
  redundant and are removed.
- `diagnostics.py`: the same for `DataIssues`. `raise_if_invalid` and the
  repr share one per-rule count helper; the exception message is unchanged.
- `tests/unit/test_data_repr.py`: replace the DOC-3 tests with hand-written
  expected strings for: each observation kind; truncation window; invalid
  footer; long-data elision; 2-d index tuples; traced fallback for both
  classes (inside `jax.jit`); valid and invalid `DataIssues`. Keep the
  six-leaves check.
- `docs/survival-data.md`: no code changes needed (cells already end with
  `data`); reread the rendered page and update prose that describes the
  printed repr. `REVIEW.md` as usual.

Checks: `scripts/check.sh --through A2`, `scripts/build_docs.sh`, `git diff
--check`. Expected diff about 250 lines.

## Decisions

- Left censoring prints as the stored set `(0, 3]`.
- No precision information beyond the dtype in the header.

# Step DOC-3: tutorial-style SurvivalData docs and a clean repr (2026-10-01)

Approved by the maintainer after reviewing DOC-2 (`726e561`): the guide reads
like a unit test, not a tutorial. One step, one commit, on a
maintainer-created branch (the worktree is a detached HEAD at `docs-startup`).
Spec sections: 3.1-3.3, 6.2, 10.5.

## Repr

- `src/survjax/data.py`: declare `_issue_bits` and `_may_contain` with
  `dataclasses.field(repr=False)`, so the repr shows only the five public
  arrays (spec 3.2 keeps the others private). No summary repr. `DataIssues`
  is unchanged. Pytree flattening and `__init__` are unchanged.
- `tests/unit/test_data_repr.py`: the eager repr names the five public
  fields and omits the private ones; `repr` of data built inside `jax.jit`
  does not raise and omits them; `jax.tree_util.tree_leaves` still gives six
  leaves.

## Docs: rewrite `docs/survival-data.md` as a tutorial

Keep the facts of DOC-2; change the form.

- Open with a short note: what works now (construction, diagnostics) and what
  doesn't yet (likelihoods). No closing scope section.
- Use one running synthetic story: a small cohort (about six patients,
  follow-up in months) built up section by section. Each section answers a
  reader task: "time and status, as in R"; "the event happened between two
  visits"; "known only to have occurred before the first visit"; "patients
  entered late (truncation)"; "finding bad rows"; "inside `jax.jit`".
- Show results instead of asserting them: end cells with the displayed value
  (`data`, `data.upper`, `data.issues.invalid_rows`). Use asserts only where a
  silent check helps the reader, if at all. The build failing on execution
  errors is the executable check (spec 10.5).
- Show real errors: call `raise_if_invalid()` and the misuse examples in cells
  tagged `raises-exception`, so the rendered page shows the message. No
  `try/except/else: raise AssertionError` scaffolding.
- Explain why before how: a sentence or two on what each censoring kind means
  for the patient before the call that encodes it. Introduce the normalized
  fields when the first output shows them, not as an opening list of rules.
- Present the R comparison as a table (`Surv(...)` call next to the survjax
  call, plus a notes column). Re-check each claim against the official `Surv`
  help page. In particular, the "(start, end]" wording there may refer to
  counting-process data, not interval censoring; drop the claim unless the
  page supports it.
- Dtype behavior: one short section with a single example, not a test of both
  modes.

`docs/playground.ipynb`: remove the empty trailing cell and reset the
kernelspec and `language_info` metadata to the DOC-1 minimal form. Keep the
maintainer's `data` cell, which now shows the clean repr. Have it follow the
same cohort as the guide, without asserts. `docs/index.md`: update the guide's
one-line description if needed.

## Checks and review

`scripts/build_docs.sh` (both dtype modes), `scripts/check.sh --through A2`,
`git diff --check`. Read the rendered x64 guide end to end and report in
`REVIEW.md` whether every output shown matches its surrounding prose. The
guide rewrite replaces most of a 176-line page, so the diff may exceed about
300 lines. If it does, say so in `REVIEW.md` rather than splitting the step.

# Documentation startup on `docs-startup` (2026-09-30)

The maintainer requested two separate tasks following specification section
10.5. Each is one reviewable step and one commit. Stop after the first step's
`REVIEW.md` and await instruction before starting the second. This plan is for
approval before either implementation step; the earlier plans below are history.

## Step DOC-1: local documentation process

Spec sections: 10.2, 10.3, 10.5, and the documentation item in 12.2.

Files to create or change:

- `pyproject.toml`, `uv.lock`: add a locked optional documentation group with
  Sphinx, MyST-NB, JupyterLab and a Python kernel. Keep core dependencies
  unchanged.
- `docs/conf.py`, `docs/index.md`: minimal Sphinx site for Markdown and
  notebooks, navigation, and a clear statement of the currently implemented
  subset. Include the existing `docs/simulation-guide.md` only if it can be
  visibly labelled as illustrative and unverified without build warnings.
- `docs/playground.ipynb`: a small synthetic, editable notebook using the
  current public `SurvivalData` API; no saved outputs.
- `scripts/build_docs.sh`: one documented, non-interactive command that builds
  HTML from a clean checkout, executes runnable content afresh in separate
  float32 and float64 processes, and fails on warnings or execution errors.
- `.gitignore`: exclude generated site, notebook execution artifacts and
  caches. `REVIEW.md`: report exact commands, outcomes and rendered-page
  inspection. Update this plan if implementation reveals a necessary split.

Public API: none. Tests: the build command itself is the end-to-end check;
verify it from a clean working tree and inspect the rendered index and notebook.
Run the existing A2 gate and `git diff --check` if dependency or tooling edits
affect those checks. No network, R, real data or demo dependencies at doc build
time. Do not publish or host the site.

## Step DOC-2: A1–A2 documentation

Spec sections: 3.1–3.3, 6.2, 10.2, 10.5, and 12.2. Cover implemented behavior
only; label A3 indexing/static-metadata guarantees and later likelihood modes
as planned. Revise this step plan, if needed, before writing its code and seek
approval under the working agreement.

The maintainer's follow-up authorizes DOC-2 and also asks for the commands to
open the rendered HTML. Add those commands to `docs/index.md`. A locally edited
playground currently contains exploratory cells and saved outputs: preserve its
cell code, clear saved outputs for the committed source, and extend it with the
new guide's working API. This is still the one documentation step below.

Files to create or change:

- `docs/survival-data.md`: executable synthetic examples for event/status,
  explicit right/left/interval flags, mixed data, `time2`, bounds, and
  truncation. Show normalized fields and the exactness rule for bounds.
- In the same page, explain issue masks, `invalid_rows`, `ok`, valid but
  uninformative rows, structural `TypeError`/dtype `ValueError`, and explicit
  host `InvalidDataError`. Demonstrate ordinary construction and diagnostic
  access inside `jax.jit`, with the host checkpoint after compiled execution.
- In the same page, compare the flag and bounds conventions to R's
  `survival::Surv` forms, including event coding, interval endpoints and
  counting-process `time2`. Check comparison wording against official R
  documentation. Claim no numerical agreement without R-generated fixtures.
- `docs/index.md` and `docs/playground.ipynb`: link the guide and make the
  notebook a short interactive path through the same working API.
- `REVIEW.md`: reading order, invariants, exact example/build outcomes in both
  dtypes, source of R-convention claims, known gaps and rendered-page review.

Public API: none. Tests: executable examples through `scripts/build_docs.sh`
in both dtype modes, plus the existing A2 gate and `git diff --check`. Examples
must check their stated outputs rather than merely run. No new unit tests that
repeat documentation examples. Keep this step near 300 changed lines; split it
before implementation if the reviewed plan requires more.

# D-006 diagnostic compatibility decision (2026-09-30)

The maintainer approved retiring `TracedInputError`, keeping `ValueError` for a
traced `DataIssues.raise_if_invalid()` call, and exposing only the
`data.issues` property (no callable `data.issues()` alias). On the
maintainer-created `spec-deviations` branch, update the normative contract in
`spec/SPECIFICATION.md` sections 6.2, 10.2 and 13, its review/decision history,
and D-006 in `DEVIATIONS.md`. Leave D-010 and the other deviation statuses
untouched. This explicit maintainer request authorizes the spec edit.

Public API: clarify that only `InvalidDataError` is exported for data
diagnostics; no code or behavior changes. No tests are added for this
documentation step. Review the diff for internal consistency and run
`git diff --check`. Overwrite `REVIEW.md` with the reading order, invariants,
evidence and uncertainties. Commit this one step on `spec-deviations` and stop.

# A2 constructor API migration — specification revision 9

## Pre-merge bookkeeping step (2026-09-30)

The maintainer requests a clarification of the non-exact equal-endpoint case
and cleanup of deviation IDs/statuses. Change only the `from_bounds` docstring
in `src/survjax/data.py`, the branch D-006 references in
`src/survjax/_data_validation.py` and
`tests/unit/test_data_construction.py`, and `DEVIATIONS.md`. Rename the branch
D-006 to D-011 and mark D-011, D-007, D-008 and D-009 as superseded by
revision 9, leaving final closure to the maintainer. Update `REVIEW.md` and
commit this small step on `a2-constructors`.

Public API: unchanged. Spec: sections 3.1, 6.2 and 13. Tests: the existing
independent A2 equal-endpoint acceptance cases and the full A2 gate; no new
test is needed for a docstring and bookkeeping change. Do not edit the
read-only specification or independent tests.

Branch: `a2-constructors`. A2-R1 exception alignment was committed as
`c6ef7e3`. The maintainer's 2026-09-30 request authorizes the remaining
constructor API migration: remove issue-returning constructor methods, accept
eager and traced observation inputs in the two ordinary forms, and expose
diagnostics through `data.issues`.

## Current step

Spec sections: 3.1–3.3, 6.2–6.3, 10.2 and 12.1–12.2.

Files and public API:

- `src/survjax/data.py`: retain exactly `SurvivalData(...)` and
  `SurvivalData.from_bounds(...)` as construction methods; remove automatic
  host validation and the `with_issues` method; add a typed `issues` property
  decoding the stored issue bitfield. No `from_bounds_with_issues` method.
- `src/survjax/_data_validation.py`: remove obsolete host-only constructor
  validation and references to the removed method, preserving normalization,
  dtype policy, argument-misuse checks and issue attribution.
- Own tests in `tests/unit/`: migrate callers to ordinary construction and
  `data.issues`; check eager, `jit` and `vmap` behavior, invalid-value masks,
  explicit host checkpoints, traced inputs, tree reconstruction and the absent
  methods. Keep tests with code in the same commit.
- `REVIEW.md`: map checks to requirements and report exact gate results.

The revision 9 specification and its acceptance criteria already require this
API; `spec/GOALS.md`, `spec/SPECIFICATION.md` and `tests/spec/` are read-only.
There are no current examples or docs modules to migrate. Historical deviation
entries retain their original wording and status.

## Verification

Run `scripts/check.sh --through A2` to cover Ruff, mypy, own tests and the
independent A2 tests in both dtype modes. Use focused A3 acceptance tests for
the removed-method and tracing checks, without claiming A3 complete. Run
`git diff --check`. If an independent test conflicts with the specification,
record the issue in `DEVIATIONS.md` and leave the test unchanged.

## Later handoff

A3 owns indexing and slicing, value-derived concrete static metadata and
compile-count tests (spec 3.2). The current all-true metadata is only a
conservative interim value. A6 owns likelihood issue propagation and explicit
failure for unimplemented observation modes. Neither is part of this step.

The traced host-checkpoint exception subtype and legacy diagnostic compatibility
remain open in spec section 13 and `DEVIATIONS.md`. This step preserves the
existing traced-checkpoint failure and does not introduce a callable alias.

## Performance amendment on `spec-speed` (2026-09-30)

The maintainer requests adapting the earlier performance draft to the current
specification at `9a403a3`, including revision 10 diagnostic compatibility.
This authorizes edits to the normally read-only specification and goals.

- Update `spec/GOALS.md` with speed, memory efficiency and acceleration.
- Update `spec/SPECIFICATION.md` sections 1, 7.2, 9.2, 10.3, 12 and 14:
  retain pure ordinary construction, `data.issues`, stored-issue propagation,
  conservative traced metadata and simulation fit-skipping semantics. Measure
  prepared-data evaluation and traced construction separately. Keep GPU checks
  and scalable metric algorithms optional, with a small CPU baseline in Slice A.
- Leave the final amendment revision number for acceptance: the current header
  says revision 9 while the history includes revision 10. No API changes,
  implementation edits, new dependencies or changes to independent tests.
- Overwrite `REVIEW.md` with reasoning, reading order, preserved contracts,
  documentation checks and unmeasured performance requirements.
- Checks: whitespace, Markdown/section references, allowed diff scope and exact
  preservation of unrelated statistical/API sections and example code. Runtime
  tests and benchmarks are not run for this documentation-only step.
- Commit this one documentation step on `spec-speed`; do not merge or publish.

## Illustrative simulation guide on `spec-speed` (2026-09-30)

The maintainer approved moving the simulation workflow out of the normative
contract and removing recipe-specific acceptance requirements. This authorizes
the following documentation step, including edits to the specification.

- Move the detailed recipe into `docs/simulation-guide.md`, explicitly
  illustrative and unverified; retain `lax.map` as one example choice.
- Update specification sections 6.2, 11.4, 12.1, 12.3 and 14: retain library
  guarantees, check compiled composition directly, and leave study workflow,
  external solver and loop selection to callers and measurements.
- Public API, goals, implementation, dependencies and tests remain unchanged.
  Add no runtime tests; check whitespace, local links, moved example code and
  preservation of unrelated contracts. Record results in `REVIEW.md`.
- Commit this single documentation step on `spec-speed` and stop.

## Spec-test sync default (2026-10-23)

The maintainer requests defaulting the tester path to `../spectest`.
Update `scripts/sync_spec_tests.sh` and its usage text; retain explicit path
overrides and all existing sync guards. No public API or spec requirements
change. Verify Bash syntax and default/explicit-path syncs in temporary repos,
then record results in `REVIEW.md` and commit this step.
