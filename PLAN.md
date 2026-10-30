# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Other sub-slices are archived verbatim in `plans/` (index: `plans/README.md`).
Still open there: A9 (`plans/A9.md`, "A9 Docs to-do" and "Final evidence
audit and planning checks") and A8 (`plans/A8.md`, "Closure still required").

---

# RMST.appendix — Permanent mathematical appendix and conservative guarantees (2026-10-30)

Maintainer approved the code/report and requested conservative guarantees,
a permanent math-only appendix outside docs/, tied to code, then the next
implementation step. Start clean release-rmst at
1a007fceb52ef1303cd739f14b58ce9d67f1cce9. Preserve unfinished PLAN history.

One documentation/contract concept; no numerical behavior change:

- Clarify spec 8.7/12.1: Weibull/PH-Weibull zero-cutoff guarantees cover
  value and first derivatives only; higher guarantees require valid smooth
  interior inputs. A NaN result's own tangents have no staged guarantee;
  valid-entry derivative isolation remains mandatory (also settles S-1).
  Bump revision 37 to 38 and commit this alone in the design repo. Its
  tracked files are clean; leave unrelated untracked .claude/ alone. The
  launch authorizes branch creation: use release-rmst there too, avoiding
  main. Mark this dedicated worktree via spec-since.sh dev-release-rmst
  --mark; literal dev would mark a different checkout. Commit .spec-seen here.
- Move/refine reports/weibull-rmst-math.md to appendices/weibull-rmst.md:
  retain the mathematical derivations and bounds; remove commit/step/review
  context, API proposals, empirical evidence and approval questions. Add
  appendices/README.md as the list of mathematical appendices with code
  pointers. Keep an adjacent rebuilt PDF; retire the old report/PDF paths.
- Add stable appendix-section references beside the three code branches
  and zero boundary; correct Weibull.rmst's boundary docstring. Do not
  alter its numerical expression, exports, wrappers or public docs pages.
- Record the maintainer decision in D-028 while leaving its status field
  maintainer-owned. Overwrite REVIEW.md with the contract and appendix map.

Aim at about 300 changed text lines, using a rename/extraction rather than
copying the historical report. Pandoc/XeLaTeX compilation, all-page Poppler
visual inspection, Ruff/mypy, focused RMST tests in separate dtype processes
and scripts/build_docs.sh (approved escalation for kernel cleanup) validate
this step. Commit and invoke independent-review exactly as RMST.appendix;
fix factual findings, then continue to the explicitly authorized next step.
No merge, rebase, push, tags or publication.

Next implementation: RMST.3 full-range derivative verification. Write its
concrete bounded plan after this appendix step; measure value-scaled error
factors before choosing any new RMST derivative tolerances. Keep the native
shape-Hessian restriction and code ownership; PH closure still waits for
reviewed core integration and handoff. Split evidence if about 300 lines
would be exceeded. No generic quadrature tangent or summary options/API.

---

# RMST.2-report — Mathematical approval report (2026-10-30)

Requested before maintainer approval of _weibull_rmst; start clean at
b3340749469e1513938a0cd5140f45305ab0d4bb on release-rmst. Preserve all
unfinished PLAN history. This is a maintainer report, not the later user
RMST guide or RMST.3 implementation. No production/spec/export changes.

- reports/weibull-rmst-math.md: one PDF-ready mathematical report, about
  300 source lines maximum. Derive the gamma identity, degree-12 series,
  remainder bounds, the mean-tail Chernoff bound and threshold choices;
  explain log arithmetic, safe operands, promotion, derivative limitations,
  measured evidence versus proof, and the staged decision against rmst_grid.
- output/pdf/weibull-rmst-math.pdf: compile with Pandoc/XeLaTeX; keep the
  editable source and exact build command in the report. Render pages with
  Poppler and inspect every page, extract text/check compile warnings.
  Temporary render/LaTeX products go under tmp/pdfs, not into the commit.
- DEVIATIONS.md: record D-028, discovered while explaining the zero branch:
  at k=1,b=2,tau=0 the right second cutoff derivative is -1/2 but autodiff
  returns 0. Ask whether 8.7's higher nonshape derivative promise includes
  existing higher right derivatives at the zero boundary. Leave status open,
  with no implementation/spec decision. Finishing the requested explanatory
  artifact does not depend on that decision; stop/report it afterward.
- REVIEW.md: overwrite for this report step, round 1/3; reading order,
  mathematical assumptions, factual checks, build/render evidence and limits.

References: spec 8.7/12.1; implementation at b334074; existing RMST tests and
independent-review report; primary NIST DLMF 8.2/8.7 definitions. No new
library/build dependencies. Existing Pandoc, XeLaTeX and Poppler are present.
Run scripts/build_docs.sh as the documentation regression, static checks and
whitespace checks; numerical report probes are separate from production tests.
Report completed: 269 Markdown lines plus D-028, 5 compiled/visually checked
PDF pages. Bound constants and zero-boundary probe verified at 60 digits /
x64 respectively. Final XeLaTeX log has no layout/font warnings; docs builds
pass in both modes after approved escalation removed sandbox cleanup errors.
Production/spec/tests/docs sources and project metadata remain unchanged.
REVIEW.md records exact checks, PDF identity, warning history and limits.
Commit report/source/PDF and records; invoke independent-review exactly for
RMST.2-report; act on factual/layout findings only, route spec questions to
maintainer, then stop.

Review round 1 on 6448c46: maintainer; proofs/constants/PDF verified.
S-2 expands D-028 (k=2 third derivative; k<1 divergent second derivative).
The user's loop question arrived during that review: add the explicit JAX
staging/count/option discussion with official documentation and a local
Jaxpr check; correct factual nits N-1/N-2. Final report is 289 lines / 5 PDF
pages, recompiled and visually checked. Follow-up additions are not separately
reviewed because the verdict requires a maintainer decision, not another
round seeking approval. N-3 artifact repository placement remains a maintainer
integration choice. Commit this bounded report update; stop and ask D-028.
No production, spec, test or website source change. No RMST.3 work, merge, push or publication.

---

# RMST.2 — Weibull RMST values and safe tails (2026-10-29)

Authorized by “ok for now; on to the next bit”. Continue clean release-rmst
at 01b29dc9e0ea5830580770c0183865cb159fc376; original workstream base stays
8757c0fda78e34e6e818249adedd2a144a33c33d. Same unfinished RMST sub-slice;
keep RMST.1 and unfinished A9/A10 history below, with no archive commit.
RMST.1 round 2 returned merge. Carry S-1 without a behavior change.

One concept: staged Weibull RMST values, with safe lower/upper tails. Target
at most about 300 changed source/test lines; split evidence before growing.
Spec: 4.5, 5, 7.2–7.3, 8.7, 10.2, 12.1–12.2. No dependency, export,
wrapper, docs, protected-path or generic quadrature changes.

- Create src/survjax/_numerics.py, explicitly named by spec 10.2 (not an
  unnamed module). Only shared RMST cutoff dtype/shape validation and the
  private Weibull value expression belong here; no configuration/backend API.
  Reuse validation in Exponential without changing its formula or policies
  (review N-2). Keep invalid substitution in the public family methods.
- Weibull.rmst(tau): thin typed method, parameter-issue/invalid-cutoff NaNs,
  safe inputs before arithmetic. Keep weibull.py around 300 lines (275 now).
  For log x=k*(log tau-log b), use the integrated exponential series for
  x<0.1 (12 terms; next term <=0.1**13/13!); incomplete gamma in the middle;
  the mean for x>64+4/k, safely classified in log space. The last threshold
  is well beyond the gamma tail over the required k range. Evaluate every
  inactive branch on finite safe inputs. Zero returns the active cutoff,
  preserving d/dtau=1 and zero parameter derivatives. No custom JVP.
- New tests/unit/test_weibull_rmst.py: independent mpmath quadrature values
  across k=[0.2,50], tau/b=[0.01,1e5], scale changes, explicit power underflow
  and overflow; k=1 identity, scalar and cutoff-grid broadcasting, dtype/shape
  misuse, eager/jit/vmap, zero/invalid/parameter-issue policies, first
  derivative and supported Hessian smoke checks, shape-Hessian failure.
  Existing central tolerances only; no unmeasured tolerance additions.
  Final audit adds common-dtype promotion plus its mixed-float32/float64
  regression (9 lines): total 309 source/test lines, one numerical concept.
- Overwrite REVIEW.md with round 1/3, reading order, invariants, evidence map,
  actual commands/results and remaining RMST.3 derivative-range evidence.

Checks: focused RMST/Weibull/Exponential suites in separate x64 and float32
processes; Ruff format/check, mypy, unchanged fresh-wheel through-A10 gate.
Inspect diff/size/scope; commit; use independent-review exactly for RMST.2,
follow up to three rounds and stop. No further step or PH handoff yet.

Implementation: 309 source/test changed lines after the mixed-dtype fix.
Weibull 300 lines, private numerical module 70 lines; no new tolerances.
Final focused four-module suites: 57 passed/1 existing skip (x64), 58 passed
(x32). Ruff: 114 files formatted, all lint checks pass; mypy: 21 files clean.
Final fresh-wheel through-A10 gate: exit 0; own tests 1069/7 skipped and
1075/1 skipped, doctests 4 per mode, independent 2332/28 skipped and
2322/38 skipped. REVIEW.md records exact commands/timings, the initial
pre-correction gate, corrected artifact and mixed-dtype mutation proof.
Initial implementation commit b552201. Round 1 returned fix-first for F-1:
missing tail-gradient assertions. Add 10 test lines (no runtime change),
checking finite first derivatives in every direction and [1,0] cutoff
slopes at the two power tails under jit-vmap, jacfwd and jacrev. Focused
suites 21 passed per dtype; both unsafe-input mutants now fail in both
modes; statics clean. N-1's promised value-range/float32 limitation is in
Docs to-do below. Commit the fix-up, request round 2/3 and follow its verdict.

Docs to-do: RMST topic after core/PH integration, stating the Weibull shape
higher-derivative limit and the absolute-scale cutoff-derivative accuracy.
Accuracy is promised on k in [0.2,50], tau/b in [0.01,1e5]; outside that
range, float32 can return NaN for valid inputs when the gamma probability
underflows (review RMST.2 N-1).
RMST.3 still owns full-range measured derivative tolerances/evidence (N-1).
PH owns wrappers.py/exports and waits for the reviewed Weibull core merge
and explicit ownership handoff before private built-in closure edits.

---

# RMST.1 — Staged base surface and Exponential RMST (2026-10-29)

Authorized release workstream on `release-rmst`; actual clean base:
`8757c0fda78e34e6e818249adedd2a144a33c33d` (8757c0f).
Preserve unfinished A9/A10 history below; no archive of unfinished work.
One concept, at most about 300 source/test lines. No dependencies, exports,
protected paths, docs, wrappers or Weibull edits in this step.

- `src/survjax/distributions/base.py`: add spec-named `rmst(tau)`, `mean()`,
  `median()` and `quantile(p)` methods raising NotImplementedError. The RMST
  default covers unsupported families; Weibull remains staged until RMST.2.
- `src/survjax/distributions/exponential.py`: override only `rmst(tau)` with
  `-expm1(-rate*tau)/rate`. Validate dtype/shape statically, substitute invalid
  cutoff/parameter entries before arithmetic, return NaN for those entries.
  Keep tau=0 active so value=0, parameter derivatives=0, tau derivative=1.
- `tests/unit/test_rmst.py`: public surface/absence, unsupported custom family,
  misuse, zero and invalid values, scale changes, explicit cutoff axes,
  eager/jit/vmap, independent mpmath integration/derivatives, forward/reverse
  Hessians and higher derivatives, second-order check_grads and invalid-entry
  derivative isolation (shared and per-row parameters). Use existing central
  value/tail/derivative tolerances; add no tolerance without demonstrated need.
- `REVIEW.md`: reading order, invariants, spec-to-test map, exact checks/results,
  known staged limits and independent-review round count; overwrite now.

Spec read in full: 4.1–4.2, 4.4–4.5, 4.7, 5, 7.2–7.3, 8.1–8.7,
10.2, 12.1–12.2. Apply 8.7's explicit staged exception to 8.4.
Checks: focused RMST/distribution regression in separate x64/float32 processes;
Ruff format/check, mypy; unchanged fresh-wheel cumulative gate through A10.
Inspect diff and scope, commit, invoke exact independent-review skill for
RMST.1 in `../review/code-release-rmst`, act on at most three rounds, stop.

Docs to-do: separate executable RMST topic after core and PH integration;
state staged surface and Weibull shape-derivative limitation. A9/A10 docs
remain with their assigned implementers. Ownership handoff: after reviewed
Exponential AND Weibull core merges, release-ph may edit built-in private
closure hooks and integrate wrapper RMST in its own reviewed step. No core
handoff yet after RMST.1; wrappers.py and exports remain PH-owned.

Implementation/evidence: exactly 300 source/test changed lines; no new
central tolerances. Final focused RMST: 13 passed per dtype. Distribution
regression: 140 passed/3 existing skips (x64), 143 passed (x32). Ruff: 112
files formatted, all lint checks pass; mypy: no issues in 20 source files.
Fresh-wheel unchanged through-A10 gate: exit 0; own tests 1061/7 skipped
and 1067/1 skipped, doctests 4 per mode, independent tests 2332/28 skipped
and 2322/38 skipped. Exact commands, times, environment, artifact identity
and final-source limits are in REVIEW.md. Initial commit 944582f.

Round 1: fix-first for F-1, missing cutoff-derivative mutation coverage.
Add an 18-line test-only fix-up; no runtime changes. Both dtype suites now
14 passed, and the safe-cutoff-removal mutant fails the new test in both
modes. Final statics clean. N-1/full-range evidence remains RMST.3;
N-2/validation reuse remains RMST.2. S-1 invalid-entry tangent semantics
is reserved for the maintainer; no implementation change for it. REVIEW.md
records the reviewer gate's missing-wheel environment and artifact cleanup
limits. Commit this bounded follow-up and request round 2/3, then stop.

---

# A10.4 — Synthetic R concordance references (2026-10-29)

Authorized by “plan and implement A10.4”. Work on maintainer-created `A10.4`
from clean `e8dbbed`. One concept: committed synthetic Harrell-C reference
data and comparisons, with no public API or production behavior change.
Spec: 6.1, 9.1–9.2, 10.4, 12.1. Target about 300 changed lines including
generated payloads, excluding plan/review records; split if it grows beyond
that. Existing A10 planning remains below because this is the same sub-slice.

Files to create/change:

- `scripts/fixtures/generate_concordance.R`: require R 4.4.2, survival 3.7-0
  and flexsurv 2.3.2 before output; fixed seed/RNG, synthetic cohorts only.
  Base-R export (no dependency added), optional output directory/check-only.
  Evaluate `survival::concordance` with `reverse=TRUE`, `timewt="n"`,
  `timefix=FALSE`. Include 100 heavily tied rows covering all time-tie kinds,
  plus untied positive times with >50% censoring and perfect/reversed scores.
- `tests/fixtures/concordance/{data.csv,concordance_synthetic.json,SHA256SUMS,README.md}`:
  CSV observations/scores and the already documented independent-test JSON
  schema; record versions, seed, generator hash and exact R call. Regenerate
  twice into temporary directories and compare every payload byte. Explain
  R's count mapping and exact ties, with no convention difference assumed.
- `tests/conftest.py`: extend the existing shared manifest-checked loader
  for this set, preserving CSV/JSON row/score alignment and provenance.
- `tests/reference/test_concordance_r.py`: compare R values and all four
  count pairs/masks for prepared/raw, eager/jit/vmap calls in both dtypes;
  existing central tolerances only, exact counts. Missing fixtures fail.
- Overwrite `REVIEW.md` with reading order, invariants, requirement mapping,
  actual commands/results and limits; update this section with evidence.

The independent loader requires its copy under read-only `tests/spec/fixtures/`.
Produce the matching export in our fixture directory for maintainer delivery,
as with A8. Do not edit protected paths or change its loader/selection. A full
through-A10 gate remains unavailable until the maintainer supplies that copy.
No spec/deviation/dependency/docs changes are needed. Docs to-do: the existing
concordance topic/playground and held-out example remain separate steps.

Checks: verify pinned R/package versions and installed concordance help; run
generator check-only and two independent generations, compare with committed
payloads and check hashes. Run focused own references/concordance in separate
dtype processes, Ruff, mypy and `scripts/check_wheel.sh --through A9`.
Probe all independent A10 modules in both modes and report the missing-file
failures explicitly. Inspect final diff/whitespace/scope, write REVIEW.md
with “Independent review: round 1/3”, commit on `A10.4`, then invoke the exact
independent-review skill command from `../review/code-A10.4`. Follow at most
three rounds, with separate fixes if needed, then stop and report.

Planning evidence: clean branch/status, current A10 outline and earlier
review read; normative sections above, shared fixture loader, both existing
reference sets and independent fixture schema read. R reports 4.4.2,
survival 3.7.0, flexsurv 2.3.2. The official concordance manual confirms
orientation, weighting, timefix and five count fields; pinned installed help
will govern generation. No implementation or tests before this plan.

Implementation completed: exactly 300 changed implementation/test/fixture
lines, including 105 CSV lines; public API and production code unchanged.
Pinned R generated 104 synthetic observations in two cohorts, three score
draws each. Both independent regenerations and committed payloads are
byte-identical; SHA-256 checks pass. All six R outputs also match a separate
pair enumeration of the spec rules, including all three time-tie kinds.
Focused own concordance/reference suites: 77 passed in 27.85s (float64),
77 passed in 27.86s (float32), no skips. Ruff: 104 files formatted / All
checks passed; mypy: no issues in 20 source files. Independent A10 probe:
8 failed, 113 passed (57.50s) / 8 failed, 112 passed, 1 existing skip
(59.90s); all failures are the absent protected fixture, before assertions.
REVIEW.md records commands, the corrected working-directory error and limits.
The matching JSON is ready for maintainer delivery; no full A10 pass claimed.
Fresh-wheel `scripts/check_wheel.sh --through A9`: exit 0, all gates passed.
Own suites: 1048 passed, 7 skipped / 1054 passed, 1 skipped; source doctests:
4 passed per mode; independent through-A9: 2211 passed, 28 skipped / 2202
passed, 37 skipped. Existing skips only. REVIEW.md records timings and wheel
hash. Final scope/whitespace/static audit is clean; commit and round 1 follow.

---

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
