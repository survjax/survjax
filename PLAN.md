# SIM.merge-alignment — Complete the maintainer-started merge

Start/previous step: ac15812. Incoming main: d1825f0. Original worktree base:
8757c0f. Resolve the three record conflicts, preserving both histories and
all deviation decisions. Incoming D-042/D-043/D-044 remain; simulation entries
move to D-045/D-046/D-047 with a mapping. Update only simulation calls to the
incoming keyword-only PH constructor, including docs/notebook source. Preserve
local notebook execution outputs outside the commit. No spec or acceptance
test edits; inherited changes remain as merged. No new API/dependencies.
Validate with the two demo checks, notebook smoke execution and both docs
builds. No full suites or new independent review under the current waiver.
Complete only this already-started merge, then stop.

Completed: conflicts resolved; keyword-only demo calls migrated. Both docs
builds, fresh notebook smoke, two focused checks, repository Ruff and staged
whitespace checks pass. No unresolved paths; no remaining merge substeps.

---

# Preserved simulation plan (ac15812)

# SIM.notebook-cleanup — Interactive tutorial and lean demo

Start/previous final: 8349e7f; original worktree base: 8757c0f.
Preserve the maintainer's sample-method sentence. Add an executable notebook
following the docs, with editable smoke/full configuration and freshly computed
plots/results; add download/launch links and a short example README.

The requested major cleanup removes the separate raw-CSV exporter, evidence
reducer, six extensive check modules and stale dependency-probe document.
Replace them with a compact pandas-based runner, retain the statistical fitter
and the generator/calibration functions shared with MNIST/Bayes, and trim
plotting to what the tutorial shows. Keep recorded failure reasons, performance
measures, seeds, settings and timings required by use-case 01; no public API,
dependencies or numerical model changes. Existing saved results remain usable.

This authorized cleanup exceeds 300 changed lines because of deletions; split
source cleanup from notebook/docs delivery in commits. Add two focused checks
for the retained statistical behavior; execute a fresh smoke run, compare a
full run against saved summaries, restart/run the notebook, and build both docs
modes. No full suites or independent reviewer (existing waiver). No spec change
is needed: the implementation loses audit infrastructure, not study outputs.

Completed: cleanup b757e88; full runner reproduces saved summaries. Fresh
full/smoke notebook executions, both docs builds, two focused checks and Ruff
pass. Notebook/docs delivery completes this task; no substeps remain.

---

# SIM.docs-generation — Expose simulated observations

Start/previous final: e33d4b5; original worktree base: 8757c0f.
Edit only the Weibull docs page and editorial records. Show event and censor
sampling explicitly with JAX and construct SurvivalData from observed times
and indicators; use the requested 20-year censoring limit. Rebuild executable
docs, with test suites and independent review still waived. The maintainer authorized updating the design and regenerating the full study.
Commit the spec separately, then the fixed-limit runner/export/figure changes
with their focused checks, then regenerate and commit compact evidence and
updated prose. Replace target-censoring labels with actual limits throughout.
This separates source provenance from generated results. No full test suite
or independent review is requested; run only changed example checks and docs.
No public API or dependencies change.

Completed: design 213cfb0, source ac0fa01, regenerated compact evidence and
all figures. Focused checks and both final docs builds passed. No substeps
remain; full test suites and independent review were not run.

---

# SIM.docs-rewrite — Incorporate maintainer edits

Replace docs/use-cases/weibull-simulation.md with the edited example draft;
remove the draft after moving it. Preserve the maintainer's concise structure,
resolve WA comments, expose a self-contained teaching fitter, and fix asset
paths. No public API or dependencies change. Use-case 01 and its details govern
the model, inference and study results. Validate by executing the docs build;
no test suites or independent review, as requested. Start/previous step:
b7a1a65; original worktree base: 8757c0f. This is one authorized page replacement.

---

# SIM.integration — Resolve maintainer-started main merge

Start / previous simulation final: 61681c7c2f13d004eea3d248c4dac96a8c5cce89.
Incoming main: 23e8c3ef033e9c6c6e93c3fde21356626923c684.
Actual merge base: 963c7fc0e9b7d246272204eeb536462f29539694.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d;
previous integration base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Keep release-simulation. The maintainer initiated this merge and explicitly
requests conflict resolution; independent review is waived for this task.
Complete only that existing merge after checks; no new merge/rebase/push.

Resolve six unmerged paths:
- .gitignore: retain both the local raw-diagnostics exclusion and JAX cache.
- docs/index.md: retain main's executable introduction/grouped navigation;
  add Weibull simulation under Use cases, in the documented intended order
  simulation, MNIST, Bayes. Keep the accepted page non-orphaned and N-3.
- scripts/build_docs.sh: keep main's complete portal/worktree/output/cache
  support, including both locked dependency groups and python -m sphinx.
- DEVIATIONS.md: retain both sides' entry bodies/decisions/statuses; main's
  existing IDs remain, simulation D-033/D-034/D-035 become D-042/D-043/D-044.
  Record this mapping; update only active simulation check-docstring IDs.
  Historical plans/reviews remain verbatim and use that mapping.
- PLAN.md: preserve both parent plans verbatim below this new section.
- REVIEW.md: write current merge-resolution evidence; previous reviews remain
  in both parent commits. No reviewer invocation under the explicit waiver.

No new API/dependency/numerical behavior. Cleanly merged core, spec tests,
lockfile, docs restructuring and other demos remain as supplied by main.
This is integration of inherited changes, not a new oversized implementation.
Check script syntax and its 7 CLI tests in both dtype modes, full optional
simulation checks in x64 against integrated PH/RMST, direct both-mode docs
build, rendered three-use-case navigation/assets, and the new required fast
cumulative gate through A12. Verify no conflict markers/unmerged index entries,
unique deviation IDs, exact parent-plan preservation and clean lockfile.
Browser visual inspection remains the previously reported external limitation;
this conflict-resolution task does not claim a browser inspection pass.

Completed: all six conflicts resolved; both plans/deviation decisions retained;
72 simulation checks, 7 build-script checks per mode, both docs builds and
rendered navigation/assets passed. Fast cumulative checks passed static,
regular tests and doctests but expose inherited PH acceptance failures
(2 in x64, 3 in x32); core/spec files match incoming main and are not changed
for these failures. See REVIEW.md for commands, counts and environmental
retries. Independent review waived and not invoked. No conflict-resolution
substeps remain; PH acceptance reconciliation stays separate.

---

# Preserved incoming main plan (23e8c3e, verbatim)

# DOCS.6 — Integrate documentation after rebase onto main

Base: e6b3f9e; previous docs step: 3a1c399 (rebased). Preserve main's Bayes
page under Use cases and archive its planning records with the existing plans.
Update stale deferred-Bayes wording in DOCS-MIGRATION.md and verification.md;
keep simulation deferred. No API, dependency or numerical changes. Validate
with both docs builds, build-script unit tests and whitespace checks.

---

# DOCS.5 — Prepare repository docs for a future workspace portal

Starting/previous step: 80aee9e; original worktree base: 2093ca2.
Execute docs-worktree-instruction.md: keep all build/theme/source ownership in
this repository; no portal, discovery, shared-spec or workspace tooling changes.
Change scripts/build_docs.sh to accept --output-dir, resolve its own worktree,
select that .venv/source and isolate build caches under the output root. Document
arguments, output layout, exits, dependencies and portal responsibilities in
DOCS-BUILD.md. Extend DOCS-MIGRATION.md with a page URL map; link build instructions
from the existing verification note. Keep Furo extensible via its existing
Sphinx configuration/template overrides, without a premature selector.

No library API/dependency changes. Tests: shell syntax/argument handling, normal
and explicit-output builds, source/environment identity in notebook execution,
AST/cell unchanged checks where applicable, HTTP nested-prefix link/asset crawl,
rendered navigation/search asset inspection. Use existing Python stdlib for the
local test server/audit; no new permanent test framework. Add
tests/unit/test_docs_build_script.py using the existing pytest setup to pin
worktree/environment selection, spaces in output paths, argument errors and
stop-on-failure behavior with a stub uv executable. Existing approved
larger docs scope and review waiver apply. Preserve user's callout edit unstaged.
Record build/compatibility evidence in REVIEW.md and commit one completed step.
Completed: default/custom builds pass both modes; live kernel source/env checked;
nested-prefix HTTP audits pass; 7 CLI tests pass per mode; Ruff and whitespace
checks pass. No remaining substeps.

---

# DOCS.4 — Remove internal contract citations from documentation

Starting/previous step: 300c368; original worktree base: 2093ca2.
Maintainer asks to purge specification/decision-log references from docs,
docstrings and appendices while retaining the technical text. Scope: package
module/class/function docstrings, docs/conf.py docstring, simulation guide,
verification note, PH appendix Markdown/PDF; PLAN.md and REVIEW.md records.
Rendered audit also found source comments in a SurvivalData expected-error
traceback: print its TypeError/message instead, preserving the demonstrated
error while removing internal implementation details from the page.
Keep internal implementation comments and historical records unchanged.
Preserve the maintainer's existing callout-title edit without staging it.
No public API, dependency, algorithm, numerical formula or spec changes.

Use AST ranges to edit only docstrings; compare Python ASTs with docstrings
removed before/after to prove executable structure unchanged. Re-scan all
package docstrings and docs/appendix prose for citations, including PDF text.
Build the changed appendix with its documented Pandoc/XeLaTeX command; render
and inspect its pages. Run both dtype docs builds, Ruff, mypy and whitespace
checks. No numerical regression suite for documentation-only changes.
Independent review remains waived for this documentation cleanup.
Completed: 46 docstrings cleaned; docs/appendix sources and both PDF texts
clean; both dtype builds and rendered text/link audits pass. Executable ASTs
and internal comments unchanged. PDF pages visually checked. No remaining work.

---

# DOCS.3 — Link the circle of survival teaching article

Starting/previous step: c95fdca; original worktree base: 2093ca2.
Add the approved, explicitly attributed callout after the method table in
Working with distributions. Link the independent teaching article on the
maintainer's personal website; no copied figure or API changes. Files:
docs/working-with-distributions.md, PLAN.md, REVIEW.md. Spec 10.5.
Validate with scripts/build_docs.sh (both dtype modes), rendered callout/link
inspection and git diff --check. No new tests; executable cells unchanged.
Independent review remains waived for this docs restructuring conversation.
Completed: both dtype docs builds pass; rendered callout, attribution, URL
and placement verified. Whitespace clean; no remaining substeps.

---

# DOCS.2 — Address the maintainer's documentation review

Starting/previous step: cc5f106; original worktree base: 2093ca2.
Implement docreview.md within the ongoing docs restructure. Change index.md to
name SurvivalData, explain both containers and distributions are pytrees, state
minimal core dependencies, and place likelihood immediately after the common
distribution interface. Remove PH's general likelihood/derivative and diagnostics
sections; retain batching and the specific extreme-scaling limit as a note.
Update DOCS-MIGRATION.md and REVIEW.md. No API, dependency or spec changes.
Spec: 3.2, 4, 7, 10.3/10.5. No new tests: execute scripts/build_docs.sh directly
in both dtype modes, inspect affected rendered content/navigation, audit local
links and run git diff --check. Independent review remains waived for this docs
review task. One follow-up commit; no subsequent substeps planned.

Completed: both dtype builds pass (17 executable pages / 145 cells each),
rendered review assertions and 1,739 local links/assets per site pass;
whitespace clean. All docreview.md items addressed.

---

# DOCS.1 — Restructure the pre-release documentation

Maintainer approved the full proposed structure, exceeding 300 changed lines,
removing the playground, and completing the whole restructure without repeated
independent reviews. Simulation remains tentative and excluded; Bayes v2 is
reviewed background in its separate branch, not imported here.

Starting/previous step and original worktree base: 2093ca2 (MNIST.20 review).
Plan-only archive commit: bfa2a3a. Existing integrated plans are preserved
verbatim in plans/pre-docs-restructure.md; this does not close other workstreams.

Scope: docs/index.md and navigation, docs guides and technical notes, remove
playground.ipynb, update links, document the complete migration in
DOCS-MIGRATION.md, and record verification in REVIEW.md. No public API,
dependencies, spec, protected paths, or example implementation changes.
Spec coverage: 3–9 (existing documented capabilities), 10.5 (executable docs),
11.2 (unchanged MNIST use case). Keep limitations near the relevant usage.
The MNIST notebook satisfies interactive exploration after playground removal.

Implementation:
1. Group navigation and introduce Working with distributions; concise overview.
2. Separate basic custom-family construction from advanced boundary examples;
   use the built-in PH wrapper for the exponential-to-piecewise progression.
3. Consolidate numerical explanations; shorten built-ins, likelihoods and RMST;
   reorganize data guidance; preserve substantial advanced examples as notes.
4. Move interoperability and verification material to Technical notes; expose
   existing mathematical appendices through a small index. Remove repetitive
   demonstrations and stale future-work prose, recording their disposition.
5. Remove playground and all live links to it. Retain simulation-guide.md as an
   excluded tentative source, and reserve future use-case placement in the
   migration record rather than publish empty navigation entries.

Validation: scripts/build_docs.sh directly (both dtype processes, warnings as
errors, fresh executable cells); inspect rendered HTML, formulas/navigation,
local links and fragments; Ruff and git diff --check. No new unit tests or
full numerical gate: library and examples are unchanged. No independent review
for this approved restructuring task. Commit all completed work and report.

Completed: all five implementation items. Full docs build passes both dtype
modes (17 executable pages, 147 cells each); both rendered sites pass 1,744
local link/asset checks and exact source matching. Ruff and whitespace checks
pass. REVIEW.md records browser inspection and earlier failed attempts.
No restructuring substeps remain; Bayes/simulation merges stay separate.

---

# Preserved simulation parent plan (61681c7, verbatim)

# SIM.11 — Navigation integration and rendered inspection (2026-11-06)

Start / previous final e7660a728a9fd00c2612fb28c6055d7a8d21e4ac (SIM.10
round 2 merge). Original base 8757c0fda78e34e6e818249adedd2a144a33c33d;
integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e. Work only on
release-simulation; preserve all unfinished PLAN history.

The maintainer explicitly requests navigation integration and inspection,
plus N-3, and reserves prose/formatting/dependency changes for other work.
This authorizes the previously shared docs/index.md edit. Add the existing
use-cases/weibull-simulation page to its toctree; remove the page's orphan
flag. Resolve N-3 by stating that the checked hashes cover data files.
No new SVG hashing scheme, dependencies, public API, numerical/generator,
results, shared config, scripts or unrelated prose/formatting changes.

Spec 10.5 rendered-doc evaluation and use-cases/00-common deliverable shape /
01 live-smoke acceptance apply. Run scripts/build_docs.sh directly in both
modes, check home/sidebar navigation and page links/tables/math/figures and
live outputs. Inspect browser layout at desktop/narrow widths when computer
use is available. Current cua.getState returned no browser/app and "The
browser access host is unavailable"; requested restoration while proceeding
with independent work. Structural HTML checks are not a visual-inspection
substitute. Record any remaining access limitation honestly.

No new implementation-mirroring tests for these three small docs edits.
Run Ruff and diff checks; record exact build/inspection evidence in REVIEW.md,
commit, invoke independent-review round 1 for SIM.11, and stop/report.
N-1/N-2/N-4 remain deliberately out of scope under the maintainer's direction.

---

# SIM.10 round 2 — Approved docs build environment (2026-11-06)

Start / previous final ae94e658ca5dc9775831c072953b1f505cc14768.
Original base 8757c0fda78e34e6e818249adedd2a144a33c33d; integration base
d2469e0a7e719dbcb276277e6fd80a447626ad4e. Keep release-simulation and all
unfinished PLAN history. Round 1 returned maintainer for S-1; the maintainer
now explicitly approves the recommended --group examples build-script edit.
This approval overrides the handoff's prohibition for this precise change.

Change scripts/build_docs.sh's existing uv run command to select both docs
and examples from the unchanged lockfile. Update this page's reproduction
instructions to use the resulting one-command build. No dependency/API,
generator, numerical, core, shared navigation/config or other script change.
Spec 10.5 and use-cases/00-common deliverable shape / 01 acceptance require
live smoke execution with the optional demo dependencies available.

Meaningful check: sync only the docs group, verify optimistix is absent,
then run scripts/build_docs.sh directly (no wrapper/env assignment/redirect).
Confirm the command installs the examples packages and both notebook builds
pass, including live float64 smoke and single-fit output. Run bash syntax,
Ruff/mypy and diff checks. No new implementation-mirroring test for this
one-line command edit. Update REVIEW.md for round 2, commit, invoke the exact
independent-review skill and stop. Navigation and browser visual inspection
remain separate outstanding work; do not imply that this resolves them.

---

# SIM.10 — Executable Weibull simulation page (2026-11-06)

Start / previous final 04ec2d8436a75125adc89eaf5361f968c0bf476b (SIM.9b
round 2 merge). Original base 8757c0fda78e34e6e818249adedd2a144a33c33d;
integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e. release-simulation.
Maintainer says "ok do it" for the remaining executable docs page. Preserve
all unfinished PLAN history; the demo workstream is still in progress.

One docs topic, <=300 page lines: docs/use-cases/weibull-simulation.md as a
MyST notebook. Introduction/design, labelled committed illustration/true
curves, actual model/solver source included inline, a live smoke run
(n=100,R=20,two levels), one live smoke-replicate fit, then labelled committed
full results/figures and takeaways. Load performance/metadata from reviewed
compact files; display bias/coverage and precision/power with every supplied
MCSE, accepted/failed counts, calibrated/realized censoring and shared-batch
first/warm timings, CPU/versions/thread context. Explain strict failure gate,
accepted denominators, RMS model SE, full-source snapshot and D-034/D-035
storage/timing semantics. Verify input hashes before displaying full results.
All Python loops format host records; simulation keeps reviewed JAX batching.
Use temporary smoke output, never overwrite committed results during a build.
The study selects float64 in its own notebook kernel in both docs-build modes.

Use orphan:true for this standalone page pending maintainer navigation
integration; do not change docs/index.md, docs/conf.py or protected scripts.
Existing approved env command: uv sync --locked --group docs --group examples,
then run scripts/build_docs.sh directly as required (no wrappers/redirects/env
assignments). If fresh docs-only execution fails, report/propose the exact
maintainer change rather than edit the script. No new dependency/API/module,
core, generator, other-agent files, or spec changes. Reuse committed SVGs via
relative image paths and small-data download links; do not copy assets.

Checks: warning-clean full docs build executes both modes; confirm the new
page runs live smoke within 120s, assertions pass, figures/tables and downloads
resolve, and visually inspect rendered HTML in a browser. Ruff/mypy as
appropriate. Write REVIEW.md with exact commands, evidence/limitations,
commit and independent-review; stop after this docs step. Shared navigation
and build-tool integration remain maintainer-owned. No merge/push/publication.

---

# SIM.9b round 2 — Coverage tick/point association (2026-11-06)

Round-1 fix-first F-1 is test-only: tie Figure 3's visible tick labels to the
six scenario records, and assert each model's x positions at scenario index
+/- .08. Existing y checks remain. No SVG/runtime/numerical change. Focused
full-panel check and Ruff/mypy, fix-up commit, independent review round 2.

---

# SIM.9b — Deliver CLI figures and compact SVG assets (2026-11-06)

Previous final aec234d255c65c3ad3e18b442a02b4c48bcb55de (SIM.9a round 2
merge; F-1 fixed). Continue the explicitly approved figure delivery, then
stop/report. Original base 8757c0fda78e34e6e818249adedd2a144a33c33d;
integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e. Preserve history.

Small delivery step: run.py calls write_figures after compact evidence;
its docstring describes CSV/metadata/figures. Existing real direct-script
smoke checks also require the three SVG paths. _plots.py's arm colours become
gray/green (SIM.9a N-1), distinct from model blue/orange in the other figures.
No numerical, key, fitting, acceptance, dependency or API change.

Generate/commit results/fig1_survival.svg, fig2_sampling.svg,
fig3_coverage.svg from the reviewed compact bundle, without refitting.
Keep PNG previews external. Result README gains regeneration and figure links.
Add one host-only full-panel check in check_performance.py: all six histogram
panels and coverage values map to the correct scenario/model records. No
simulation in this test. Run the complete optional example checks in x64,
Ruff/mypy, SVG XML/hash/geometry checks and inspect full previews. The prior
manual SVG audit preserved all original geometry/attributes/text (169/769/217
nodes); repeated rendering checks deterministic bytes. Update REVIEW.md,
commit and independent review, then stop. Optional N-2 helper extraction is
left alone. Docs to-do: executable MyST page with live smoke and labelled
committed full evidence is the remaining separate step. No publish/merge/push.

---

# SIM.9a round 2 — Discriminating true-DGP display fixture (2026-11-06)

Round 1 fix-first F-1 is test-only. Use true k=3,b=2,HR=.25, distinct from
both fitted laws and with unequal shape/scale; expected survival is
exp(-(t/2)^3) and exp(-.25*(t/2)^3). Assert the histogram's true-HR line at
.25 (not log(.25)). No renderer or data change; original full previews remain
valid. Focused checks and Ruff/mypy, fix-up commit and independent round 2.
Optional arm-colour N-1 is reserved for SIM.9b before SVG delivery; retain the
explicit sequential renderer rather than refactor solely for optional N-2.

---

# SIM.9a — Three-figure renderer and analytical display checks (2026-11-06)

Previous final fb7fb97e09cd31e6ed3241a70a88abf205128ed4 (SIM.8c.2 round 1
merge). Both figure implementation and delivery were explicitly approved.
Preserve unfinished history. Original/integration bases unchanged.

About 295 implementation/test lines: approved _plots.py renders the three
specified figures from compact evidence, verifying every payload hash before
creating output. Public PH.survival_grid evaluates both arms over a time-first
grid; distinct baseline families require separate graph construction only.
Histogram panels use exact counts / accepted / bin widths; coverage bars
show +/- one MCSE, explicitly labelled. Empty accepted sets have zero-count
histograms and an unavailable fitted illustration label. No fitter/generator
calls. Python loops construct host plot objects, never dispatch scenario fits.
SVG export retains text/vector geometry, omits date/random IDs and removes
only XML/path whitespace; no numeric rounding or raster payload.

Append display hand checks to check_fit.py (stays under ~300 lines): true
Exp(1), fitted Weibull k=2/b=3 and Exp(.25), both fitted HR=.5; density integral
one over accepted fits; exact +/-one-MCSE bar bounds; empty-accepted case,
valid SVG XML and refusal of changed input hashes before creating figures.
Use existing x64 tolerances. No public library API/dependency/spec/protected
path/docs edits, and no change to the shared generator contract.
Also close SIM.8c.2 N-1 trivially: the metadata check hashes the producer
file at its recorded Git commit, so valid-but-wrong IDs or zero hashes fail.
It uses local history only; add one sentence to results/README.md.
Run optional focused checks and Ruff/mypy; render/inspect full PNG previews.
Write REVIEW.md, commit, independent review. Artifact/entry-point integration
is SIM.9b to keep this source/test diff at about 300 lines: run.py writes
figures, smoke checks assert them, then commit the three compact SVGs.
Docs to-do: live smoke and labelled committed full results/figures remain a
separate executable-docs step. Do not merge/push/publish.

---

# SIM.8c.2 — Commit compact full-run evidence and checks (2026-11-06)

Previous final: c08a14281e4aa42a72e3882e6f19445ada0eb401 (SIM.8c.1 round 2,
merge). Continue the explicitly approved compact-evidence and figure work.
Original base 8757c0fda78e34e6e818249adedd2a144a33c33d; integration base
 d2469e0a7e719dbcb276277e6fd80a447626ad4e. Preserve unfinished history below.

About 270 implementation/artifact/check lines: add results/failures.csv,
sampling_histograms.json, illustration.csv and evidence.json from the audited
raw a3792a7 run using clean reviewed reducer c08a142, plus results/README.md
with meaning/regeneration instructions. Keep original metadata/performance
unchanged; do not copy/stage raw CSV. No new module/dependency/API/spec/docs
or protected path. D-035's committed-output replacement remains in force.

Add independent compact-bundle count/coverage checks to check_results.py:
complete six-scenario/model grid, unique in-range failed IDs, separate reason
totals, accepted-ID complement, every accepted HR included in integer bins,
common accepted illustration ID and both n=1000 three-MCSE coverage gates.
Add hash/source checks to check_metadata.py: all five payloads bound to the
manifest, original provenance preserved, clean producer and documented seed.
Host file inspection only; no regeneration or simulation in these tests.
Run focused optional checks, Ruff/mypy and manual source-object/hash audit;
no repeat of unchanged core gate or full numerical fitting. REVIEW.md then
independent review. Next authorized step: three SVG figures; executable docs
remain separate. Generator and all numerical acceptance policies unchanged.

---

# SIM.8c.1 round 2 — Approved module layout and provenance checks (2026-11-06)

Maintainer "ok approved" accepts the recommended _evidence.py reducer and
_plots.py figure modules; check_*.py remains independent checks. This settles
F-1. Earlier pending module/quotas below are historical, preserved verbatim.
Current start 38a9e64848b9a32148f640d4d765a8cebdf65dc8; previous reviewed
final 61c5cc62a7f890175b40aa6a1333566800d2e809. Bases unchanged.

Move check_full.py to _evidence.py, update run.py and check imports, and record
approval in D-035 without changing its open status. No numerical behaviour
or generated artifacts change. F-2: add controlled clean/dirty provenance
checks in check_metadata.py, using the existing hand raw fixture; the fake
Git status differs for the reducer and PLAN.md so a wrong scope is caught,
and the true case catches a constant-False flag. Assert the exact source path.
Run focused optional export/CLI/metadata checks plus Ruff/mypy, update REVIEW
round 2 and invoke independent-review. Continue the previously approved
compact-data/figure steps after review clears. No public library API,
dependency, protected path, spec or generator contract changes.

---

SIM.8c.1 resumed review (2026-11-06): first written round at 4e6c475,
verdict maintainer. F-1 requests explicit runtime-module approval: recommend
_evidence.py for reduction and _plots.py for figures, reserving check_*.py
for independent checks. F-2 requires clean/dirty file-scoped provenance tests
once the name is settled. Maintainer question sent; code work pauses pending
the decision. Older quota notes below are historical, preserved verbatim.

Review availability note (SIM.8c.1, 2026-11-05): both exact Claude attempts
for 494abca returned the five-hour quota limit, reset 02:40 Europe/Amsterdam.
No written review. Skill's one retry is spent. Await maintainer waiver to
finish compact artifacts/figures with review deferred, or reviewer availability.
Preserve every unfinished section below. Last reviewed final is SIM.8b
61c5cc62a7f890175b40aa6a1333566800d2e809 (round 2 merge).

# SIM.8c.1 — Full-run provenance and reducer placement (2026-11-05)

Previous final: 61c5cc62a7f890175b40aa6a1333566800d2e809, SIM.8b round 2
merge (F-1/F-2 and N-1/N-2/N-3 fixed). Original/integration bases unchanged.
Continue the explicitly approved compact-evidence and next-figure work.

Budgeted artifact sub-step: original results/metadata.json (200 lines) and
performance.csv (13 lines), copied byte for byte from the audited clean
source a3792a7703ed539db777472d876bd00ac2d1568e. No raw CSV in Git.
Move the reducer from _plots.py to the already proposed check_full.py so
both evidence auditing and the coming figure module stay below ~300 lines.
Update existing imports/CLI references; generator/public library API unchanged.
N-5 provenance refinement: record reducer source path and file-dirty state
alongside commit/hash. N-4: extend the histogram-edge hand test to a maximum
near .45; this is a test-only addition to the reviewed display-edge policy.
Checks: all focused optional export/CLI/provenance checks, Ruff/mypy,
byte hashes, six-scenario/count/coverage/provenance checks; no refit or repeat
of the already passing core wheel gate for this host-only move/artifact step.
REVIEW.md and independent review, then the authorized derived-data sub-step
and three-figure step. This commit intentionally does not write an evidence
manifest until all of its compact payloads are available in the next split.

---

# SIM.8b — Approved compact evidence writer (2026-11-05)

Previous final / code start: a3792a7703ed539db777472d876bd00ac2d1568e.
Integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e; original base
8757c0fda78e34e6e818249adedd2a144a33c33d. Branch release-simulation.
Maintainer: "approved; do it and implement the next step" approves D-035's
22 KiB compact evidence contract and the following figure step. Preserve all
unfinished plan history below. Each small implementation/artifact split gets
its own commit and independent review; continuation to figures is authorized.

First code step (<=300 implementation/check lines):
- Approved _plots.py helper: host-only write_evidence(source,out), CLI
  --results DIR --out DIR. Read existing CSV/metadata; validate unique complete
  IDs and count/reason correspondence. Keep original performance/provenance,
  every rejected ID and separate reasons/max gradient/log HR, exact common
  .025 HR-bin counts including every accepted estimate, and first mutually
  accepted illustration replicate at n=300/55% (n=100/55% in smoke).
  Manifest binds source raw checksums, reducer source identity, output hashes,
  histogram convention and the accepted-ID complement convention. No refitting.
- run.py invokes reduction after complete raw export/metadata so the approved
  --config/--out entry point produces compact evidence too. Raw CSV stays local.
- .gitignore excludes only this demo's results/replicates.csv, including explicit
  staging checks before any artifact commit. No blanket result exclusion.
- check_results.py: hand-created raw records check exact counts, failures,
  edge-bin inclusion, selection, source/output hashes and incomplete-input
  refusal. Existing optional suite remains separate from normal core tests.
- D-035: record maintainer approval; status stays open. No spec/core/API/
  dependency/protected-script/docs changes. Existing generator contract unchanged.
- Run optional checks in x64, Ruff/mypy and cumulative wheel gate. REVIEW.md
  records full-run analytical audit and exact evidence/review commands.

Next authorized step SIM.9: extend _plots.py with three SVG figures from
compact inputs: public PH survival curves per arm, six HR sampling histograms,
and coverage +/- one MCSE labelled explicitly. Host loops only construct plot
panels/curves, never simulate or fit; vectorize prediction over time/arms.
Check plotted values/labels against hand cases and inspect rendered full figures.
Keep generated SVG small with text fonts and compact XML whitespace; no raw
fit diagnostics embedded. Code and generated evidence split to normal budgets.

Artifact splits: original metadata+performance (213 lines) separately from
failure/plot-input/manifest files (about 180 lines), then figures. All derive
from the already audited clean a3792a7 full run; no artificial rerun of timings.
Docs to-do: executable smoke, labelled full table/figures, D-035 storage and
regeneration instructions, shared timing scope and failure investigation.

---

# SIM.8a revised storage proposal — no bulk fit records in Git (2026-11-05)

Maintainer rejected committing the 4.6 MiB replicates.csv because it would
remain in Git history. This supersedes the generated-data exception request
below; preserve that unfinished planning history verbatim. No raw fit records
have been staged or committed. Source/previous final remains a3792a7 (full SHA
below), integration base d2469e0, original base 8757c0f. Remain release-simulation.

Concrete measured alternative, derived outside the repo from the audited run:
/private/tmp/survjax-sim-evidence-proposal, 22,243 bytes total before figures.
- performance.csv: 3,926 bytes, all 12 original summary rows.
- metadata.json: 5,729 bytes, original source/seed/environment/settings/timings.
- failures.csv: 9,257 bytes, all 127 rejected IDs; failed plus every separate
  reason column, max-absolute gradient, fitted log HR. Other replicate IDs
  in the complete six-scenario/R=1000 design are accepted by complement.
- sampling_histograms.json: 3,132 bytes, 63 common HR bins of width .025,
  exact integer counts for each scenario/model, spanning every accepted HR.
  All per-model totals equal the corresponding accepted counts; total 11,873.
  Preserve exact edges and binning convention so the figure is reproducible.
- illustration.csv: 199 bytes, complete fitted parameters for both models,
  scenario 3 / replicate 0, both accepted. Public predictions can draw curves.

This is a small, purpose-specific committed evidence set, not a compressed
version of all per-fit diagnostics. Full run.py outputs continue to retain
all raw diagnostics locally; ignore the raw output path in Git when approved.
Record source raw SHA-256 in a small evidence manifest so derivation can be
checked against the original audited output. Regeneration follows the locked
source/seed; cross-platform bitwise identity is not promised. No upload/LFS,
new storage dependency, core API, protected path, or solver-policy change.

D-035 records the proposed change to the use case's committed replicates
requirement. Await the maintainer's storage-contract decision before changing
export code or committing artifacts. The small evidence allows ordinary
review budgets: provenance/summary and derived diagnostics/plot inputs can be
separate <=300-line artifact steps; audit and plotting code remain separate
small reviewed steps. No oversized data exception is requested anymore.
Permanent check_full.py / _plots.py approvals from the earlier request remain
pending; no new named module is created by this proposal. Figure and docs
work remains after the reviewed evidence steps.

---

# SIM.8a — Full-study evidence and independent artifact audit (2026-11-05)

Start / previous simulation final: a3792a7703ed539db777472d876bd00ac2d1568e
(SIM.7b.2 round 2: merge). Maintainer merge base:
d2469e0a7e719dbcb276277e6fd80a447626ad4e. Original worktree base:
8757c0fda78e34e6e818249adedd2a144a33c33d.
Maintainer says "ok get to it". Remain on release-simulation; no merge/push.
Preserve all unfinished plan history. Existing modules are near their size
limits; no new plotting module is created without its proposed approval.

Authorized readiness run (no new source code): the reviewed CLI was run from
clean committed a3792a7 into /private/tmp/survjax-sim-full-a3792a7 before this
plan edit. Metadata confirms that exact source and dirty-at-start=false,
CPU Apple M4 Max, x64, both levels together per n/model. Root key/starts/
BFGS/acceptance unchanged. Data generation/fitting remained JAX batched.
Full: n=100,300,1000; R=1000 per scenario/model; 12,000 raw fit records.

Pre-code audit plan (about 300 handwritten changed lines including checks):
- examples/weibull_simulation/check_full.py: standalone explicit optional
  full-artifact audit, with --results DIR. Check 6 scenarios, 12 summaries,
  12,000 unique records, exact metadata/settings/counts/timing correspondence.
  Reconstruct independent uniform draws with the prescribed fold_in/split
  keys and NumPy inverse-CDF algebra, not the demo generator. Recompute pooled
  censoring and no-event estimability. Use analytical PH score/Hessian algebra
  for both models to check raw gradients/Hessians; no temporary distribution
  implementation or replacement fitter. Compare with existing x64 tolerances.
  Check exact acceptance/reason policy from saved diagnostics, recompute all
  accepted-fit measures/MCSEs using NumPy, and enforce the n=1000 Weibull
  95% +/-3 observed MCSE gate. Check rejected inference stays NaN.
  Explain host loops as independent CSV/reference validation, never fitting.
- Investigate every rejected fit without changing thresholds or dropping rows:
  gradient range, finite/SPD status, and Newton gamma correction H^-1 g;
  Exponential closed-form log-rate/log-HR MLE gives an additional independent
  diagnostic. These comparisons do not substitute fits or reclassify failure.
- Small dedicated hand cases or existing analytical cases validate audit
  math where useful; do not mirror the exporter or write toy timing tests.
- Write REVIEW.md with exact output map, commands, checksums, coverage gate,
  failure investigation, versions/CPU/censoring/shared timings and limitations.

Artifact boundary: performance.csv is 13 lines, metadata.json is 200 lines,
replicates.csv is 12,001 lines / about 4.6 MiB. Before committing generated
artifacts, present the audited concrete results and request the working-
agreement exception for a separate generated-data commit over 300 lines.
Keep handwritten audit/plot code within normal review budgets. No artifact
copy into this repo until that approval; no format/dependency workaround.

Figure follow-up proposal: examples/weibull_simulation/_plots.py, demo-only
Matplotlib writer for the three specified SVG figures (true/fitted curves
n=300,target=.55, accepted-fit HR distributions, coverage with MCSE bars).
Uses saved fit parameters/flags and public survjax predictions; no refitting.
Request approval for this unnamed helper before creating it. It avoids
adding plotting responsibilities to the already ~300-line numerical modules.
Figures and executable docs are separate reviewable steps after audited data.

Initial full results: both n=1000 Weibull levels are within 3 MCSE of .95;
127 rejected fits across the study, all gradient-only. Investigate before
claiming release evidence. No new dependency/API/protected path/docs edits.
No R/accelerator/reference patient data or public sampling API.

---

# SIM.7b.2 — Runtime provenance and metadata.json (2026-11-05)

Start / previous simulation final: f4cfbfcc498c727ca95258fc1638b88309992d22
(SIM.7b.1 round 2: merge). Maintainer merge base:
d2469e0a7e719dbcb276277e6fd80a447626ad4e. Original worktree base:
8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-simulation; maintainer says "ok on to next bit". Preserve all
unfinished/merged plan history. Existing approved _results.py only; no new
module/dependency/library API/protected path or docs edits.

One host-only provenance step, about 300 changed code/check lines:
- _results.py: capture_metadata(config, root_seed, log_hr) snapshots Python
  process argv/cwd, code HEAD and worktree-dirty-at-start, lock SHA-256 and
  spec-seen, tested package versions, OS/CPU identity/logical CPU count,
  actual JAX backend/devices/x64, and curated thread environment settings.
  Capture before calibration/fitting/CSV writing so new output files do not
  create a misleading dirty-at-start flag. CPU model via Darwin sysctl or
  /proc/cpuinfo/platform.processor fallback; report missing info explicitly.
  Thread env unset is JSON null; effective worker count is not measured.
- Record exact root seed and fold_in/split scheme; DGP k=1.38,b0=11.42,
  HR=.7/log HR, fixed equal allocation and independent Uniform(0,c). Record
  solver BFGS rtol/atol=1e-8,max_steps=1000,throw=False, exact start policy,
  strict gradient<1e-5/finite/success/Cholesky acceptance and no-event policy.
  Parameter order, flag semantics and raw CSV unused-blank vs NaN conventions
  identify how to interpret the approved records, without duplicating fitting.
- write_metadata(out, snapshot, limits, batches) adds output path, once-per-
  target calibrated c and all scenario indices/n/R/realized censoring. Store
  timing once per (n,model) group with C/R and shared first/warm time/rate;
  median(3), synchronized whole results, first call may include compilation,
  C*R/warm as approved D-034. No invented per-scenario time allocation/cache
  clearing. JSON uses allow_nan=False. Write only after both CSVs succeed.
  Invalidate old metadata just before CSV replacement; failures must not leave
  stale provenance attached to partly replaced CSVs.
- run.py: snapshot after config/x64/output-directory validation, before work;
  write metadata after CSVs. Keep all numerical batching and CSV values
  unchanged. Captured command is actual Python sys.orig_argv, not an invented
  uv parent-shell invocation; config/output fields describe direct calls too.
- check_metadata.py: mocked CPU fallback probes, exact JSON round-trip
  with distinct two-size/two-level provenance/timing values, capture-at-start
  ordering and strict nonfinite rejection. Add real smoke cross-file checks
  in check_cli.py for grid/c/realized censoring/timing/runtime/source/backend/seeds
  and Python argv. No new timing thresholds beyond existing smoke timeout.
- REVIEW.md: overwrite, include requirement/check map and actual results.
  Numerical code and the shared generator contract stay untouched.

Spec/use-case: 00-common metadata versions/seeds/settings/timings; 01 main
and details DGP, Seeds, Models, Timing, Files; 10.3, 11.4, 12.2 release item 4
and 12.3. Test current CPU/macOS environment only; mocked host branches are
not a claim of a tested cross-platform runtime matrix. No full-study run or
coverage claim here. Run explicit x64 metadata/export/CLI/fit/scenario checks,
generators in both dtype processes, Ruff/mypy/lock and unchanged fresh-wheel
through-A10 gate. Commit and exact independent-review SIM.7b.2, at most three
rounds with fix-ups, then stop/report.
Implemented metadata snapshot/final JSON in existing helper. Source identity
is captured before work, final group facts after CSV success, stale markers
invalidated before CSV replacement. No numerical code changed. Helper 301
lines, runner 299; code/check diff about 346 lines (mostly JSON/fixture fields).
Full optional suite 51 passed (55.51s), smoke 6.62s; generators 17 per dtype
(3.87s/3.82s), static checks clean. Initial sandbox sysctl denial was resolved
by rerunning with host access, not by weakening CPU provenance. Fresh-wheel
through-A10 exit 0: own 1282/1287 passed (8/3 skips), doctests 4 per mode,
independent 2333/2324 passed (27/36 skips). Exact commands/times in REVIEW.md.
Commit and exact independent review next, then stop after this step.
Round 1 fix-up: F-1 shares fixed study settings across actual execution and
metadata, with a real Optimistix call spy to detect drift. N-2 makes denied
CPU-model probes consistently report null. Values/algorithms stay unchanged;
no new module or configuration abstraction. Full optional checks 54 passed
(55.79s), final metadata checks 5 passed (2.87s); generator 17 per dtype
(3.50s each). Static checks clean. Record exact evidence in REVIEW.md, commit
and request round 2; no new implementation step before that verdict.
Remaining: full six-scenario evidence/results and figures, executable docs.
Docs to-do: provenance fields, actual Python argv vs outer invocation,
unset thread settings/effective count, shared group timings and CSV schema.

---

# SIM.7b.1 — Complete per-replicate CSV export (2026-11-04)

Start / previous simulation final: 49f57462583259606abca2ecceed2ef846392c03.
Reviewed batching implementation: 877daf120aa43a4d11b49f04fd04b7ccb39af2d6.
Maintainer merge base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-simulation. The maintainer accepts shared-batch timing (D-034)
and says "on to next bit" after the concrete _results.py export proposal.
Proceed with that proposed demo-only helper under the next-step authorization;
no further module/library/dependency expansion. Preserve all plan history.

Split before code: SIM.7b.1 extracts CSV serialization and adds complete raw
replicate records; SIM.7b.2 adds metadata/provenance. This keeps the combined
source/check diff near 300 lines without hiding numerical code in a helper.
No full study/figures or complete artifact claim before metadata exists.

Files/interfaces and invariants:
- run.py: keep nested C/R vmaps and last-axis statistics unchanged. Include
  failure_columns in the compiled _summarize output and ModelRun so all masks
  are computed on device, together with counts/performance. Transfer once per
  n, collect host records, then call _results.write_results. No Python fitting
  or numerical scenario/replicate loop. Create output directory after config/
  x64 guards and before calibration (SIM.7a N-1); fail early for a file path.
- New examples/weibull_simulation/_results.py: the previously proposed private
  demo serializer, not a library API. Two CSVs: performance.csv retains its
  reviewed fields, counts and shared timing scope with explicit schema order
  (fix batching N-1); replicates.csv contains every (scenario,model,replicate),
  including rejected/unestimable, with theta, objective, gradient, Hessian,
  SE, HR interval, raw solver/Hessian/estimability/acceptance flags and every
  separate reason plus overall failed. Use numeric theta_i/gradient_i/
  hessian_i_j columns, parameter_count and log_hr; unused Exponential parameter
  columns are empty, not fake zeros. Real NaN/inf diagnostics remain numeric
  CSV spellings, never blank/substituted. Reason/flag values are 0/1.
  Every Python loop is explained as host serialization of computed records;
  no estimator, numerical reduction or failure-policy duplication in helper.
- check_results.py: explicit optional-group host CSV checks. Hand fit results
  include accepted, overlapping failures, theta/objective/gradient/Hessian
  NaNs and unestimable placeholders for both model dimensions and two levels.
  Round-trip every vector/matrix/flag/reason and verify unique row identifiers,
  count conservation and stable summary headers. No tolerance changes.
- check_cli.py: adapt fake runner to include raw result fields/masks, extend
  real smoke to require all 80 raw rows, reconstruct failure-count totals from
  them and cross-check summaries. Add unusable-output early-rejection case.
- DEVIATIONS.md D-034: record the maintainer decision, leave status open.
  REVIEW.md: overwrite for this step, exact requirement/check mapping and
  independent-review count. Shared study.py/core/dependencies unchanged.

Use-case 01 main/details/common: all failed fits recorded, seeds/outputs;
spec 7.2, 10.3, 11.4, 12.2 release item 4. Dedicated x64 export/CLI/fit/
scenario/performance checks, generators in both dtype processes, Ruff/mypy/
lock and unchanged fresh-wheel through-A10 gate. Commit and run exact
independent-review SIM.7b.1 (up to 3 rounds), then stop/report.
Implemented: write_results stores every raw diagnostic/flag and fixes summary
field order; failure masks are returned by the existing compiled summary.
Runner now 297 lines, helper 130; code/check diff 336 lines including 39
removed old serialization lines (297 added). Metadata was split out before
code. Combined focused suite 49 passed (40.22s), real smoke 4.90s with 80 raw
records; strengthened matrix/gradient round-trip fixture passes (1.45s).
Generators 17 per dtype (3.40s each), static checks clean. Fresh-wheel
through-A10 exit 0: own 1282/1287 passed (8/3 skips), doctests 4 per mode,
independent 2333/2324 passed (27/36 skips). Exact outputs in REVIEW.md.
Commit and exact independent review next, then stop after this step.
Round 1 fix-up (tests only): F-1 checks every unused Exponential matrix/vector
cell is blank, N-1 pins all measure-column ordering, N-2 reconciles real smoke
raw flags and summary totals. No exporter/numerical changes. Round 2 next.
Remaining: provenance/metadata, full results and figures, executable docs.
Docs to-do: exported parameter order, numeric blank-vs-NaN meaning, complete
failure columns, shared batch timing and results-vs-live smoke labels.

---

# SIM.7a.batch — Batch same-size censoring scenarios in JAX (2026-11-04)

Start / previous step final: 6a9c96b7c2935f18d8467d3266625bd12aaaba6e.
Maintainer merge base: d2469e0a7e719dbcb276277e6fd80a447626ad4e.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-simulation. The maintainer explicitly corrects the Python
scenario-dispatch loop: batch same-size scenarios, keep only justified Python
loops and explain them in code. This direction supersedes the SIM.7a
one-scenario-at-a-time dispatch plan. Preserve all unfinished plan history.

One correction, about 300 changed implementation/check lines:
- run.py: prepare_scenario accepts a scalar pair or equal-shaped 1D arrays
  of scenario indices/censor limits. Nested vmaps generate (C,R,N) data with
  the exact original root -> fold_in(index) -> split(R) -> U/C keys. Scalar
  calls retain (R,N) behavior. Host validation remains before fitting.
- fit_batch supports (R,N) or (C,R,N) with nested vmaps inside one module-level
  jit per model. No Python numerical iteration across scenarios/replicates.
  Summaries/failure counts reduce only the last replicate axis, retaining C;
  compile their joint computation once. Keep strict acceptance/MCSE formulas,
  overlapping reasons and unique failed; no fitter or library changes.
- run_config loops only over n (changing shapes require separate compilation),
  calling the grouped runner once per n with indices [0,1], [2,3], [4,5].
  Calibration stays once per target. Device-to-host transfer happens before
  CSV-only loops, whose comments explain that they do no numerical fitting.
  Explain every remaining Python loop/comprehension in the demo runner:
  host calibration, static model/schema dispatch, wall-clock timing and CSV.
- Timings cover the actual C*R-fit batch, without inventing per-scenario
  allocation. Repeat shared times with timed_scenarios=C on each CSV row;
  throughput numerator is C*R. Per-scenario counts/statistics remain separate.
  This is a consequence of the explicit batching correction; D-034's future
  full-study compile-report label/cache choice remains open. No cache clearing.
- Existing check_cli.py: adapt mocked dispatch to assert one batched call per
  n, original index/target order and per-scenario output. Actual script smoke
  continues independently checking keys/censoring and verifies timing scope.
- check_scenario.py: grouped generation/fits vs existing scalar path, plus
  actual trace-count reuse for changed values at fixed C/R/N.
- check_performance.py: independent hand expectations for grouped reductions
  with differing accepted counts. Existing scalar/timing tests must stay green.
- No new module, dependency, library API, protected path or docs edits. study.py
  numerical code and generator contract stay unchanged; only comments may
  explain its existing host-only calibration loops if needed.

Read use-case 01 main/details/common and handoffs; spec 7.2–7.3, 10.3,
11.4 and 12.2 release item 4. Ordinary-range fitting remains unchanged.
Focused explicit x64 demo checks, generators in both dtype processes,
Ruff/mypy/lock and fresh-wheel through-A10 gate. Write REVIEW.md, commit and
run independent-review SIM.7a.batch (max three rounds with fix-ups), then stop.
Implementation: nested C/R generation and fitting, last-axis summaries/counts,
compiled joint summaries and a single grouped dispatch per n. Every remaining
Python loop has a host/static justification. Counts are JAX arrays until the
single device_get before CSV output. Code/check diff is about 295 lines;
run.py 321 lines, near the approximate file-size target; no new module.
Combined optional checks 44 passed (43.28s), smoke 5.20s. After four added
invalid grouped-input cases, scenario suite 20 passed (18.57s); other files
are unchanged. Generators 17 per dtype (3.62s/3.66s), static checks clean.
Initial CSV ordering failures due to device_get's dict sorting were fixed by
explicit model order. No numerical/tolerance change. Fresh-wheel through-A10
exit 0: own 1282/1287 passed (8/3 skips), doctests 4 per mode, independent
2333/2324 passed (27/36 skips). REVIEW.md has exact commands/timings. Commit
and exact independent review next; stop after this correction's verdict.
Remaining: complete per-replicate/metadata exports (new helper proposal still
awaits approval), full results/figures and executable docs. Docs to-do: group
axes, justified host loops and shared timing scope, alongside existing CLI.

---

# SIM.7a — CLI configurations and smoke performance CSV (2026-11-04)

Step start / current integration base:
d2469e0a7e719dbcb276277e6fd80a447626ad4e (maintainer completed main merge).
Previous simulation final: 8d0fb5edb248038dc7bcc9a9af1e6326f120da15.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-simulation; maintainer says "merge done; on to next bits".
Preserve all merged/unfinished plan history below. SIM.6 review round 2 had
no remaining code findings; its maintainer verdict concerned timing labels
for full-run output, not the implemented first_call_seconds field.

Split before code: SIM.7a adds CLI/configuration and performance.csv only;
SIM.7b will add per-replicate CSV and complete provenance/metadata before
any full-study evidence is recorded. SIM.8 adds full results and figures in
separate reviewable commits as needed; final step is executable docs.
No full-study/figure/metadata deliverable claim in this first CLI step.

Files and demo interfaces (about 300 changed implementation/check lines):
- examples/weibull_simulation/run.py: support direct-script and package
  imports without changing study.py or the shared generator contract. Add
  run_config(config, out) for smoke (n=100,R=20) and full (n=100,300,1000;
  R=1000), each with censoring targets .25/.55. Calibrate exactly once per
  target using equally weighted arm LPs [0,log(.7)] and baseline k/b0.
  Enumerate n ascending then targets, pass those exact indices to the reviewed
  runner (same explicit key scheme). No optimizer/acceptance formula changes.
- Write performance.csv with scenario/n/target/calibrated/realized censoring,
  model, attempted/accepted/unique failed and every separate overlapping
  reason, all accepted-only measures/MCSEs and reviewed timing fields. Keep
  first_call_seconds honest; no compile-time label/cache-policy decision.
  The raw per-replicate export is SIM.7b, not silently claimed here.
- CLI: --config {smoke,full} and required --out directory. Only the CLI main
  sets x64, as a runner; importing the module/run_config does not change JAX
  precision. Reject invalid config and disabled x64 before fitting/output.
  Full configuration wiring is checked with a fake runner; do not execute
  the full study in this step. Source file stays about 300 lines.
- examples/weibull_simulation/check_cli.py: explicit optional-group checks for
  both configuration grids/order/R, one calibration per target with correct
  pooled PH inputs, lossless summary CSV fields/failure totals/timing labels,
  invalid arguments/configuration and a real direct-script smoke subprocess
  starting with x64 disabled. Check its rows/counts/calibration independently.
  No numerical tolerance additions or standard runner/protected-file edits.
- REVIEW.md: overwrite for this step, record invariants, exact checks and
  independent-review round count. No new dependency/public library API/module,
  core, docs, protected path or other-agent implementation edits.

Spec: use-case 01 main/details (DGP, Seeds, Performance, Files and commands,
smoke/full sizes), common conventions; 10.3, 11.4, 12.2 release item 4.
Existing summary/fit numerical checks plus generator tests in both processes,
Ruff/mypy/lock checks and the unchanged fresh-wheel through-A10 gate.
Commit and run independent-review SIM.7a (up to 3 rounds with fix-ups), then
stop/report. No merge, rebase, push or publication.

Tracking: D-034 remains the future full-run table/cache question; current CSV
uses the existing reviewed first_call_seconds field. The merged ledger has
two D-033 headings (CI guard and simulation failure policy); references here
mean the simulation heading, whose maintainer choice is already recorded.
Do not change another workstream's ID/status or reopen the settled policy.
Implemented: run.py totals 297 lines (+73/-2); dedicated checks 222 lines.
Smoke direct script passes in 4.54s; combined optional suites 41 passed in
30.99s. Generators: 17 per dtype (3.42s/3.44s). Ruff/mypy/lock checks clean.
An initial gate stopped at test dictionary-style C408 errors; fixed without
numerical changes and reran the fresh-wheel gate. Final gate exit 0, all
gates passed through A10: own 1282/1287 passed (8/3 skips), doctests 4 per
mode, independent 2333/2324 passed (27/36 skips). Exact outputs in REVIEW.md.
Commit and request independent review, then stop after its verdict.

Next-step proposal (not implementation/approval): SIM.7b needs demo-only
examples/weibull_simulation/_results.py to keep the already 297-line runner
and 306-line shared study module from growing. It would serialize all raw
replicate estimates/objective/gradient/Hessian/flags and Wald inference to
CSV, and metadata.json with exact grid, versions/CPU/thread settings,
seeds/key scheme, solver policy, calibrated/realized censoring and timings.
Move CSV assembly out of run.py into that helper, adding no dependency or
library API. Tests would round-trip an overlapping/rejected fixture and a
real smoke run, cross-check totals and metadata, and ensure undefined
inference is represented explicitly. Keep that step under about 300 lines
or split again. AGENTS.md requires maintainer approval for an unnamed module;
request that approval before creating it. No metadata/full-study claim here.

Docs to-do: exact CLI invocation, partial-vs-complete output scope until
SIM.7b, summary-column definitions and imported-runner layout.

---

# SIM.6 — Accepted-fit performance and synchronized timings (2026-11-02)

Clean start and previous simulation final:
458ebbd559fb5540b4eb7a32784d93143fe4d79b (SIM.5 round 2: merge).
Integration base: 95f453279ab9bffad1c67dc234a56d46b534a677.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
SIM.5 F-1 fixed; N-1 direct-script imports/layout deferred to CLI/docs,
N-2 optimizer-status fragility documented, N-3 theta-only nonfinite test is a
nonblocking remaining nit. SIM.5 reviewer raw gate ended with wheel-fixture setup errors, not code
failures; its exact results are recorded separately in REVIEW.md. Preserve all plan history.
The maintainer explicitly authorized SIM.5 and this next bit in one request.

One concept, about 300 changed source/check lines: accepted-only summaries and
synchronized timing/orchestration. No CLI or saved results in this step.
Use-case 01 details Performance measures/Timing; spec 3.3, 7.2, 10.3,
11.4, 12.2 release item 4 and 12.3. Read main/details/common and handoffs.

Files and demo interfaces:
- examples/weibull_simulation/run.py: add performance(fits, truth), pure JAX
  reductions over accepted fits only. Bias log HR, empirical SD (ddof=1),
  root-mean-square model SE, inclusive log-Wald coverage, strict |z|>1.96
  power. MCSE formulas exactly as details, no MCSE for model SE. Cast counts
  to the float dtype before division. Rejected NaN placeholders are substituted
  before arithmetic. With zero accepted, statistics are undefined (NaN); with
  one, empirical SD and its/bias MCSE are undefined. Defined singleton measures
  remain defined. No additional acceptance filters or silent failure dropping.
- Keep study.py (306 lines) and its frozen generator import contract untouched;
  use the existing spec-named run.py as an importable engine. CLI/docs will
  deliberately import the engine and resolve direct-script imports (SIM.5 N-1).
- Add typed ModelRun/ScenarioRun result containers and run_scenario(index,n,R,c).
  Prepare/validate once; realized pooled censoring uses floating counts. Fit
  both models. Synchronize prepared inputs before measuring. First call includes
  block_until_ready on the whole FitResult; warm time is median of 3 subsequent
  synchronized calls. Rate is requested R / warm seconds. Report first-call
  seconds honestly: an existing cached executable need not compile again. No
  cache clearing. Return raw fits, counts and summaries with timings.
- examples/weibull_simulation/check_performance.py: explicit optional-group
  checks, with hand-computed estimates/MCSEs, rejected NaN exclusion, zero/one
  accepted edge cases and eager/jit agreement. Spy on fit/clock/synchronization
  to check first+three warm calls and median without timing thresholds; one
  real small scenario checks both models' orchestration and returned totals.
- REVIEW.md: reading order, invariants, requirement/check mapping, exact checks
  and review rounds. No library API, new dependency, config, protected script,
  docs or other agent files. No differentiation promise through the optimizer.

Checks: dedicated x64 performance/scenario/fit suites, generator suites in
separate x64/x32 processes, Ruff, mypy, lock consistency, unchanged fresh-wheel
through-A10 gate. Commit then exact independent-review SIM.6 (max 3 rounds),
fix blocker/should-fix findings between rounds, then stop/report as requested.
Implemented: run.py 226 lines, adds 115 lines; dedicated checks 170 lines.
Hand measures/MCSEs, rejected huge/NaN rows, zero/singleton accepted and timing
order/median/requested-throughput all pass. Combined optional suites: 35
passed in 68.80s; after stronger timing assertions, performance 6 passed in
8.60s. Generators 17 per dtype (8.65s/8.15s); Ruff/mypy/lock clean.
Fresh-wheel through-A10 exit 0: own 1282/1287 passes (8/3 skips), doctests 4
per mode, independent 2333/2324 passes (27/36 skips). Exact outputs in
REVIEW.md. Added a rejected NaN-theta/finite-objective check for masking and
SIM.5 nonblocking N-3. Commit and independent-review next, then stop.
Remaining: CLI smoke, full six-scenario study/results/figures, executable docs.
SIM.6 round 1 F-1: add a three-accepted hand case distinguishing the two
MCSE formulas (1/sqrt(3) vs .5); production code unchanged. S-1 is D-034,
asked of the maintainer and deferred to full-run labeling; no cache policy
change now. Full-run metadata must record CPU model, JAX version, thread
settings, calibrated/realized censoring and command explicitly (N-1).
Docs to-do: importable runner layout, overlap reason columns/unique failed,
accepted-only denominators, undefined small-count summaries, timing convention.

---

# SIM.5 — Compiled scenario runner and failure accounting (2026-11-02)

Clean start/base and previous simulation step final:
`d3ceb0fadb5052564079a46ac8e799bf0a7fbbc3` (SIM.4 round 2: merge).
Integration base: 95f453279ab9bffad1c67dc234a56d46b534a677; original worktree
base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Preserve all unfinished and merged plan history below.

D-033 is settled by the maintainer: record all reasons separately, allowing
multiple reasons per rejected fit, and an overall failed column for the unique
total. Unestimable rows get that reason; placeholder diagnostics are not actual
solver attempts. The maintainer explicitly requests "implement this and then
the next bit as well": complete/review SIM.5, report its verdict in progress,
then implement/review SIM.6 performance/MCSE summaries without another approval.
Stop after SIM.6; still stop on a maintainer verdict or unresolved spec question.

Self-contained step, under 300 code/check/decision-record lines:
- examples/weibull_simulation/run.py (named by use case 01): scenario/batch
  functions in this separate module, leaving the 306-line study.py unchanged.
  Caller passes the pre-calibrated c; do not recalibrate per replicate/n.
  Fixed equal allocation, k=1.38,b0=11.42,gamma=log(.7), explicit root/fold_in/
  split key scheme. Generate a (R,N) cohort and validate on the host before
  fitting with one compiled vmap per model.
- Return raw FitResult batches, per-replicate reasons and accepted/failed
  counts under D-033. Timing and high-level orchestration move to SIM.6
  alongside performance summaries to keep both commits small.
  No parameter/MLE/solver changes, no failure dropping or retries.
- examples/weibull_simulation/check_scenario.py: dedicated optional checks for
  key mapping and allocation, generation versus primitive draws, compiled fits
  versus reviewed single-fit values, hand failures/count conservation, and
  overlapping failure columns and unique-total counts. Add a real locked-BFGS
  failure regression to cover SIM.4 N-3 solver-status wiring; this is a failure
  accounting check, not an extreme-range derivative-accuracy claim. Keep checks explicit like
  check_fit.py; no protected gate edits or silent dependency skips.
- REVIEW.md to be overwritten only when implementation begins; record D-033's
  decision, requirement mapping, exact checks and remaining scope. No library
  API or dependency changes. No new unnamed library module (run.py is demo code).

Readiness probe (existing reviewed fitter only, no implementation edits):
valid four exact times equal to 1, two per arm, Weibull fit returns
estimable=True, solver_success=True, accepted=False, hessian_ok=False;
theta=[709,0,0], objective=-2832, gradient=[0,0,0]. This verifies why raw
solver success is insufficient; it does not broaden the no-events definition
of unestimable or claim tail-derivative accuracy. Not a full-study run.

After the decision: use explicit examples-group float64 scenario/fit checks,
generator checks in both dtype processes, Ruff/mypy/lock consistency and the
unchanged fresh-wheel through-A10 gate. Commit, exact independent-review SIM.5
up to three rounds, then report and proceed to the explicitly authorized SIM.6.
Remaining subsequent steps after SIM.6:
CLI smoke, full study/results/figures and executable docs.
Docs to-do: explain the selected reason policy and timing convention there.


Implemented: run.py 113 lines, dedicated checks 147 lines, D-033 record 24
lines. Compiled fits for both models agree with reviewed single-fit values;
closed-form scores are checked at each fitted theta (not different fits'
small residuals). No tolerance change. Added real jitted optimizer-failure
plumbing check, closing SIM.4 N-3 for a locked-environment failure case.
Final scenario suite 10 passed (11.26s); combined pre-boundary fit/scenario
suite 23 passed (23.87s); unchanged fit suite contributes 15 cases. Generators
17 per dtype (3.11s/3.03s), Ruff/mypy/lock clean. Fresh-wheel through-A10:
exit 0, own 1282/1287 passed (8/3 skips), doctests 4 per mode, independent
2333/2324 passed (27/36 skips). Exact outputs/investigations in REVIEW.md.
Round 1 F-1: add four isolated reason checks (14 scenario checks pass); no
production change. Request round 2; after merge verdict proceed to SIM.6 as
explicitly requested, then stop after that step's review.

---

# SIM.4 — Fit one replicate with the integrated PH models (2026-11-01)

Clean start/base: 95f453279ab9bffad1c67dc234a56d46b534a677 on release-simulation.
Previous simulation step final: afc266a684be990406ec6152b7eb035b6bb1aa26.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Maintainer integrated main/PH/RMST and requests resuming implementation.
Preserve all merged/unfinished plan history verbatim below. This continues
use case 01; no archive of other workstreams' preserved integration records.

One concept: a usable single-replicate fit for each specified model, with
likelihood verification and exact-Hessian Wald inference. Spec revision 41,
sections 4.4, 6.3, 7.1–7.3, 10.3, 12.2; use-case 01 main and details Models
and fitting. Ordinary-range derivatives only, per current PH limitation.

Files and demo interfaces:
- examples/weibull_simulation/study.py: add typed FitResult,
  negative_log_likelihood(theta, data, arm, *, model), and
  fit_replicate(data, arm, *, model). Model is static 'weibull'/'exponential'.
  Both build the real public ProportionalHazards wrapper, with lp=gamma*arm
  on its own line. Optional Optimistix import stays inside the fit function,
  preserving generator-only imports without optional dependencies.
- BFGS rtol=atol=1e-8, max_steps=1000, throw=False; starts exactly as details.
  float64 fitting only, explicit error for other precision. Preconditions:
  validated nonempty 1D exact/right untruncated data and matching binary arm.
  Detect no-event arms before fitting. Return raw objective/gradient/Hessian,
  solver and estimability/acceptance flags; do not impose a priority among
  overlapping failure reasons. Scenario bookkeeping remains the next step.
  Acceptance uses solver success, finite theta/objective, max|grad|<1e-5,
  successful Cholesky. Inverse-Hessian gamma SE and exp(gamma +/-1.96 SE),
  with undefined inference explicitly NaN for rejected/unestimable fits.
- examples/weibull_simulation/check_fit.py: dedicated optional-environment
  pytest checks, deliberately outside ordinary test filename discovery.
  Verify analytical Weibull/Exponential PH objectives/derivatives, eager/jit/
  vmap agreement and check_grads order 2; exact exponential MLE/Hessian/Wald
  algebra; an accepted seeded Weibull fit, unestimable data, and mixed batching.
  Reuse value/AD bounds in tests/impl_tolerances.py. Add a solver-only
  log-parameter accuracy bound 2e-5 before running checks, derived from the
  specified score threshold and analytic H^-1 at the Exponential MLE; no
  existing tolerance is loosened.
  Explicit invocation JAX_ENABLE_X64=1 uv run --locked --group examples pytest
  -q examples/weibull_simulation/check_fit.py. No conditional dependency skips.
- examples/weibull_simulation/DEPENDENCIES.md: fix the trivial SIM.3 N-2
  pointer to its durable probe source in REVIEW.md at commit afc266a.
- Overwrite REVIEW.md with reading order, invariants, evidence and limits.

No core API/source, dependency, shared docs/navigation/config or protected
script changes. Keep study.py and this step's diff about 300 lines or less;
split before growing further. Docs to-do: show these model/fit functions on
the future simulation page. Remaining: compiled scenarios/failure accounting,
performance/MCSE summaries, CLI smoke, full study/results and executable docs.

Checks: optional fit suite in float64, existing generator suites in separate
x64/float32 processes, Ruff/mypy/lock checks, unchanged fresh-wheel through-A10
gate (includes integrated PH/RMST own tests). Commit, exact independent-review
skill SIM.4 up to three rounds, then stop/report; no next sub-step, merge or push.


Implementation: 308 changed source/check/tolerance/bookkeeping lines, record files excluded;
study.py is 292 lines. Optional fit checks pass all six cases in float64
(35.40s), including both models on the same seeded replicate. Existing
generator suites pass 17 per dtype (5.42s/3.79s). Ruff/mypy/lock checks pass.
The solver-only 2e-5 log-parameter bound was derived before testing from the
analytic Exponential H^-1 and the specified 1e-5 score acceptance threshold;
no existing numerical tolerance was changed. Fresh-wheel through-A10 gate
exited 0: own suites 1282/1287 passed (8/3 skipped), doctests 4 per mode,
independent 2333/2324 passed (27/36 skipped). REVIEW.md has exact timings,
commands and limits. Commit and independent review follow, then stop.


Round 1 (1005ac7) returned fix-first, F-1: rejected-but-estimable fits have
no coverage and removing gradient/Cholesky gates survives. Review-directed
fix-up: extract the post-solve policy into private _assess_fit with unchanged
BFGS settings and inference arithmetic. Add hand-input cases for solver
failure, over/boundary/NaN gradient, nonfinite theta/objective, indefinite/NaN
Hessian, and a positive control. Assert raw diagnostic preservation, flags and
NaN rejected inference, eagerly and jitted. This is a separate small review
fix-up commit; no new concept/API/dependency/module. N-2: align the check-file
invocation with --locked. N-1 remains a maintainer-owned protected-gate decision.
Fix-up checks: 15 optional checks passed (18.63s), generators 17 per dtype
(2.72s/2.69s), Ruff/mypy/lock checks clean. No core changes; initial full gate
not repeated for the example-only extraction. Review fix-up is a separate
~120-line diff (mostly code movement); cumulative implementation scope is
374 lines, disclosed in REVIEW.md. study.py is 306 lines. Commit and request
round 2, then stop after the skill's verdict.

---
# Preserved incoming main plan (963c7fc, verbatim)

# A9.close — Reconcile completed A9 records (2026-11-04)

Maintainer requested A9 bookkeeping on the existing a9 branch at a351952.
Files: PLAN.md, plans/A9.md, plans/README.md and REVIEW.md. Public API,
code, tests and dependencies touched: none. No protected-path changes.

Pre-edit plan: add a dated closure note to the A9 archive, preserving its
historical text verbatim; retain the completed A9.d1 gate/narrative map in
that archive verbatim from 14155ed:REVIEW.md; correct the archive index's
stale pending labels. The existing integrated plans below remain verbatim.
Cite A9.4 round-1 and A9.d1 round-2 merge verdicts and the later combined
gate evidence without claiming new execution or whole-Slice-A acceptance.
The maintainer's bookkeeping instruction authorizes this record closure.

Spec context: 7.3, 10.3, 12.1–12.3. No new tests: this step only reconciles
records. Verify historical-text and gate-map byte preservation, referenced
paths/commits, diff scope and whitespace. Do not repeat numerical tests,
docs builds or benchmark measurements for unchanged executable inputs.
Commit once, run independent-review as A9.close (at most three rounds),
address required findings in fix-up commits, then stop/report.

---

# CI.integration — Finish maintainer-started main merge (2026-11-01)

Branch ci; merge already in progress when requested. Finish this existing
merge only, without rebasing, merging into main, pushing or worktree cleanup.
CI parent / previous CI endpoint: 1299a88fbf444fe1717a50f896ea024121f96f29.
Incoming main: 95f453279ab9bffad1c67dc234a56d46b534a677.
Original worktree base: 84d2943e788440db60504afb49a13360d43ad95e.

Conflict-resolution plan, written before resolution:
- PLAN.md: this integration first; preserve both complete parent plans verbatim
  below as history. Neither unfinished workstream is silently discarded.
- DEVIATIONS.md: retain incoming main's complete entry sequence and statuses.
  Its D-030 is the previously reconciled RMST zero-boundary entry; rename
  only CI's colliding guard entry to D-033 and record the historical mapping.
  Preserve both entry bodies and all decisions; no new spec question/decision.
- REVIEW.md: overwrite for CI.integration with reading order, preservation
  invariants, exact checks, limitations and independent-review round 1/3.
  Prior reviews remain available verbatim in the two parents.
- All other paths are already merged. Preserve main's source/docs/test changes
  exactly, with the sole existing CI interpreter-path selection retained in
  tests/install/test_wheel.py; the auto-merge includes PH exports/RMST checks.
  Keep .github/workflows/ci.yml exactly the CI parent. No protected/spec edit.

Verify byte equality of all other paths to the appropriate parent; both
complete plans as exact substrings; unique deviation IDs and unchanged entry
bodies; no conflict markers or unmerged paths. Run YAML, Ruff/mypy, lock,
whitespace checks and fresh-wheel through-A10 gate in both dtype processes.
No new test for record reconciliation; existing installed-wheel tests cover
both contributions. Imported docs are unchanged, so no docs authoring/build
step is introduced. Commit the merge, run independent-review as
CI.integration (up to three rounds), then stop/report.

Validation complete: both parent plans/entry bodies preserved, unique IDs,
all other paths byte-exact as planned. YAML, lock, Ruff/mypy and whitespace
checks pass. Fresh-wheel A10 gate exits 0: own 1282/8 x64 and 1287/3 x32;
doctests 4 per mode; existing acceptance 2332/28 and 2322/38 passed/skipped.
Exact commands/results and platform limits are in REVIEW.md. Commit and
independent CI.integration review remain before handoff.

Remaining: hosted CPU runs after maintainer push; maintainer's ci-to-main
integration; optional GPU provisioning/environment/workflow. Existing PH/RMST
handoffs and pending independent acceptance remain owned as recorded below.

---

# Preserved CI parent plan (verbatim)

# CI.1 — Three-platform CPU GitHub Actions (2026-11-01)

## CI.1 round 2 plan — approved PR guard policy

Maintainer agreed to use the PR source branch name for guard exemptions,
retain merge-commit tests, and ensure the guard runs. In workflow only:
create local main from origin/main when absent; on PRs attach a temporary
local branch at the existing merge commit with the source branch name.
The source name main must map to ci-pr-main so it cannot overwrite the
baseline; both names have the same ordinary no-exemption policy. Pass the
source name as an environment value, never interpolate it into shell code.
No protected script edit, new privilege, commit/content/checkout change.

Record the decision in D-030 without changing its open status, update
REVIEW.md for round 2/3, validate YAML/embedded shell and actual guard in
throwaway repositories: ordinary/main source blocked on protected changes,
meta-* and spectests-* get exactly their existing path exemptions, clean
ordinary PR passes. Check merge SHA preservation and baseline creation.
Rerun static/whitespace checks; initial full numerical gate remains valid
because this fix changes only checkout/ref metadata. Commit and request
independent-review round 2, then follow its verdict and stop.
Validation complete: all ten guard scenarios pass, preserving the merge SHA,
two parents and main baseline. YAML/shell syntax, Ruff/mypy and whitespace
checks pass. The full initial wheel-gate evidence remains in REVIEW.md.

## CI.1 round 1 follow-up plan

Review 2166045 returned fix-first: F-1 Windows CRLF checkout; F-2 silently
skipped guard. S-1 asks which PR branch policy to apply; record D-030 and
ask the maintainer before dependent changes. S-2 is already settled by the
explicit request for CPU macOS/Linux/Windows. Fix F-1 by setting autocrlf
false and eol lf before checkout on the ephemeral runners. No protected
script change. Revalidate YAML and whitespace; full numerical rerun is
unnecessary for checkout configuration after the passing initial gate.
N-1/N-2/N-3 are nonblocking nits; retain the existing gate/triggers/action
choices for now. Record review limits and remaining work in REVIEW.md.

Start on maintainer-created branch ci at
84d2943e788440db60504afb49a13360d43ad95e (worktree starting base).
Previous implementation endpoint: aae68920769855aa08227f4b7c4818ce7a820923
(RMST.integration); subsequent commits are inherited maintainer metadata.
Earlier unfinished RMST/A9 records below remain
owned by their original workstreams; preserve them rather than archiving
unfinished work. This CI step adds no numerical feature or public API.

Files and behavior, planned before code:
- .github/workflows/ci.yml: push, pull_request and manual CPU checks on
  ubuntu-24.04, macos-15 (Apple silicon), windows-2022 (x64), Python 3.12,
  uv 0.11.19 and the unchanged lock. Read-only token, independent job failures,
  bounded runtime and cancellation of superseded runs. Force JAX_PLATFORMS=cpu.
- Provision locked dev/test dependencies and build a wheel while online;
  export its native absolute path with Python so Windows Git Bash does not
  leak an MSYS path. Run the unchanged scripts/check.sh --through A10 offline
  with two workers. Its own suite, source doctests and independent tests each
  run in separate float64/float32 processes. Keep tests/spec read-only.
- tests/install/test_wheel.py: select Scripts/python.exe on Windows and
  bin/python elsewhere. Existing out-of-tree wheel tests verify this path
  through actual installation on each matrix platform; no mock-only tests.
- REVIEW.md: overwrite for CI.1 with invariant/check map and honest platform
  limits. No docs/, protected paths, spec, dependencies or lock changes.

Spec covered: 10.1 (existing static checks), 10.3 (locked CPU baseline,
separate dtype processes, installed package, optional accelerator evidence),
12.1/12.2 (existing through-A10 acceptance gate, no new slice).
Validate YAML, Ruff/mypy, lock consistency, full local wheel gate, whitespace
and scope. Commit then use independent-review, at most three rounds; stop.
Hosted Linux/Windows execution requires pushing, which is outside this step.

GPU assessment / subsequent optional step (not enabled in this commit):
- Prefer manually dispatched Linux/NVIDIA verification. GitHub larger runners
  offer a T4, or use an ephemeral self-hosted/cloud GPU. Runner availability,
  account eligibility and billing must be chosen before adding a working job.
- Pin a separate accelerator environment to the CPU JAX version plus compatible
  CUDA plugins/libraries and record the NVIDIA driver. Do not mutate uv.lock
  or add GPU dependencies to core. Current JAX guidance offers CUDA 12/13;
  choose against actual GPU and driver (Linux minima 525/580 respectively).
- Set JAX_PLATFORMS=cuda, assert GPU devices, log versions/driver/devices and
  run applicable numerical/transformation/spec suites per dtype, initially
  serially to avoid GPU memory contention. Installation smoke tests retain
  their separate CPU-only environment. Record backend failures explicitly.
- Require float32 and float64 evidence before NVIDIA support claims; measure
  performance separately. Metal remains experimental/float32 only, outside
  required CPU CI. Do not execute untrusted PRs on a persistent GPU runner.
Sources checked 2026-11-01:
https://docs.github.com/en/actions/reference/runners/larger-runners
https://docs.jax.dev/en/latest/installation.html

Local validation completed: unchanged fresh-wheel A10 gate exits 0; own
suites 1097/7 x64 and 1103/1 x32 passed/skipped; source doctests 4 each;
independent suites 2332/28 x64 and 2322/38 x32. YAML/Ruff/mypy/lock and
whitespace checks pass. REVIEW.md records commands and platform limits.

Remaining sub-steps: CI.1 review and hosted CPU runs after maintainer push;
GPU provisioning, separate lock and optional workflow require a later step.

---

# RMST.integration — Resolve the maintainer-started main merge (2026-10-31)

Branch: release-rmst. Merge already in progress when requested; finish that
merge only, without rebasing, publishing, or merging this branch into main.
Parents: ddd7f1f808eea0c802848bf0489135b547fed67f (RMST) and
4f170654fdc5bb2840f89232e439e33d7e5fc3ff (incoming main).

Conflict-resolution plan, before edits:
- tests/impl_tolerances.py: preserve both independently added blocks verbatim:
  measured RMST factors and SIMULATION_SIGMAS=6. No tolerance value changes.
- PLAN.md: keep this active integration/RMST handoff first, then preserve both
  pre-merge plan versions verbatim in clearly labelled records below. Do not
  discard or archive unfinished history. Main's existing plans/A10.md remains.
- REVIEW.md: write this integration's reading order, preservation invariants,
  checks and limitations. The prior RMST and README reviews remain available
  verbatim at the two parent commits; no extra review/archive module needed.
- All other staged paths came from main. Preserve them exactly, including
  CLAUDE.md and .workmux.yaml; do not edit their protected contents. No new
  numerical code, public API, dependency or spec change in this integration.

Initial static check found an inherited main issue: README.md's fenced
Python example needs two blank lines before its top-level function. Apply
Ruff's formatting-only correction, preserving the example's behavior; this
is the only inherited-file exception, with no docs/ or protected-path edit.

Validate source equality against the RMST parent; imported-file equality
against main; both plan records as exact substrings; both tolerance blocks;
absence of conflict markers/unmerged paths; Ruff/mypy, locked environment,
focused RMST+generator tests in separate dtype processes, and unchanged
fresh-wheel through-A10 gate. Commit the existing merge, invoke the required
independent-review as RMST.integration (round 1/3), then stop/report.

Remaining work owned by this worktree after integration:
1. Core RMST.1/.2/.3 and math appendix are complete and independently reviewed
   (RMST.3 round 3 merge). Spec revisions 38/39 are now on design main; no
   new spec decision is needed. Deviation status remains maintainer-owned.
2. Handoff base.py/exponential.py/weibull.py closure edits to release-ph once
   this RMST branch is integrated into code main. This local main snapshot
   does not contain wrappers.py or ProportionalHazards yet. PH owns wrappers,
   exports and its own closure/RMST identity/derivative tests; do not duplicate.
3. Then one separate executable docs/rmst.md topic: staged surface; explicit
   cutoff axes rather than rmst_grid; units; invalid values; first-order-only
   zero boundary; unspecified NaN-own tangents; native higher-shape restriction
   and supported nonshape derivatives. Coordinate navigation/API ownership.
4. Recheck the combined PH+RMST documentation/examples and both-dtype release
   evidence after separately reviewed PH integration. No new core RMST feature
   is planned; generic quadrature/sampling/options stay outside this workstream.

Validation complete: 49 focused RMST/generator tests pass per dtype;
Ruff/mypy/lock/preservation/whitespace checks pass. The fresh-wheel A10 gate
exits 0: own 1097/7 x64 and 1103/1 x32, doctests four per mode, independent
2332/28 x64 and 2322/38 x32 passed/skipped. Finish the maintainer-started
merge and obtain the independent RMST.integration review before reporting.

---

# Preserved RMST-side plan (verbatim pre-merge version)

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Other sub-slices are archived verbatim in `plans/` (index: `plans/README.md`).
Still open there: A9 (`plans/A9.md`, "A9 Docs to-do" and "Final evidence
audit and planning checks") and A8 (`plans/A8.md`, "Closure still required").

---

# RMST.3 — Weibull derivative and isolation evidence (2026-10-30)

Start release-rmst at b2150e1fe10df97d7ac2d4e9926a2b8a803f90c4;
original launch base remains 8757c0fda78e34e6e818249adedd2a144a33c33d.
The maintainer explicitly requested the next implementation step. Appendix
review is running in its separate checkout against the committed snapshot;
its findings will be handled before this step is committed. Preserve history.

Status: implementation resumed from appendix fix-up
dc48b24cd89a571a082099f4055cd1669c8ff3cd. The maintainer explicitly
approved unspecified own derivatives for a NaN RMST result (2026-10-30),
settling S-1b. Appendix round 2 returned VERDICT: merge, all findings fixed.

One bounded numerical-verification concept, about 250 changed test lines:

- New tests/unit/test_weibull_rmst_derivatives.py: independent 60-digit
  mpmath quadrature of survival and its shape/scale derivative integrands;
  integrate the scale kernel directly to avoid cancellation in m-r*S,
  and derive the cutoff derivative from survival. Use actual dtype-rounded
  inputs for references. Cross
  k={0.2,0.5,1,2,10,50}, tau/b={0.01,0.1,0.5,1,10,100,1e5},
  b={1e-4,1,1e4}; straddle both branch thresholds at four shapes too.
  Compare eager/jit/vmap forward and reverse natural Jacobians; express
  parameter errors in log-coordinate units k*dM/dk and b*dM/db, divided
  by |M|, so the section 12.1 value-scaled exception respects time units.
  The cutoff derivative uses absolute scale 1 and the S(tau) identity.
- tests/impl_tolerances.py: add only RMST-specific absolute error factors,
  zero relative tolerance. Pre-test full-range probes (142 cases, both
  dtypes) found maxima [shape,scale,cutoff] of
  [1.72e-14,2.46e-15,1.56e-15] x64 and
  [1.29e-5,1.55e-6,6.45e-7] x32 under the above normalization.
  Fix parameter factors 2e-13/5e-5 and cutoff factors 2e-13/5e-6,
  allowing rounding margin without changing any existing tolerance.
- Verify finite supported (b,tau) Hessians across the grid and independent
  analytical Hessian entries on lower-series and gamma interior cases.
  Preserve explicit shape-Hessian NotImplementedError coverage; extend it
  across forward/reverse combinations in the gamma middle branch.
- Shared shape/scale parameters: valid entries beside invalid parameters
  or cutoffs retain the isolated call's first derivatives in both modes;
  supported scale Hessians stay isolated too. Do not assert a guarantee
  about an invalid result's own tangent. Extend zero first-derivative checks
  over shapes/scales, with eager/jit/vmap and debug_nans for valid entries.

Spec 4.5, 7.2–7.3, 8.7 revision 38, 12.1–12.2 and the release handoff were
read in full. No numerical algorithm/public API, wrappers, exports, modules
in src/, dependencies, quadrature tangent, grids or ordinary docs changes.
No new higher-order boundary or shape guarantee. Docs to-do: executable
RMST topic after PH integration, with first-order-only zero boundary,
unspecified NaN-own tangents, and the native higher-shape restriction. Explain
keeping shape outside differentiated arguments for supported Hessians;
packing it with scale can materialize a zero shape tangent in native JAX.

Run focused RMST tests in separate x64/float32 processes, Ruff/mypy,
whitespace checks, and unchanged scripts/check_wheel.sh --through A10
(fresh wheel supplies the standard gate's required install artifact).
Write REVIEW.md with exact outputs, commit, invoke independent-review
exactly as RMST.3, handle up to three rounds and stop/report. PH owns
wrappers.py/exports and must wait for maintainer core integration and an
explicit ownership handoff before built-in closure edits. No merge/push.

## RMST.3 approved tolerance clarification

Maintainer explicitly answered “Confirm log-parameter scaling” to S-1/D-029.
Approved clarification: change spec 12.1's exception to name k*dM/dk and b*dM/db, with cutoff error
still at absolute scale 1. Bump specification revision 38 to 39, commit alone
on design release-rmst, then mark this worktree via spec-since.sh
dev-release-rmst --mark and commit .spec-seen plus decision/review records.
Committed separately in design as
1389af4da8cd23f7c0d38da047ba14b5ac0801f2; .spec-seen now records it.
No numerical code, tests or factor changes are needed. Leave D-029's status
maintainer-owned. Finish round 2's source-fix review before invoking round 3
on the approved contract/records; do not exceed three rounds for RMST.3.

## RMST.3 round 2 preparation

Round 1 at ab7afb1 returned VERDICT: merge. Record S-1 as D-029 and ask
whether to clarify 12.1 in log-parameter units; do not edit the spec until
answered. N-1: narrow the old Weibull invalid-cutoff tangent assertion to
valid entries only, matching the confirmed unclaimed NaN-own-tangent rule.
Apply the same correction to the analogous Exponential test. The independent
precision-failure fix is already committed at 106078d. Run those focused
isolation tests per dtype, record round-1 outcome and final gate evidence,
commit records/test corrections and invoke round 2 on the fix and N-1.

## RMST.3 precision-failure isolation fix-up

A post-commit probe found a spec 8.7 isolation violation outside the accuracy
range: float32 k=[2,0.01], b=1.25 shared, tau=[0.75,0.01] gives a valid
first value and a NaN failed second value, but the first reverse scale derivative
is NaN (forward is finite). The gamma probability underflows; the inactive
exp(log_value) then overflows because its substituted probability was one.
Protect the exponentiation operand before it runs in src/survjax/_numerics.py;
keep NaN as the failed value. Add a dtype-specific regression for failed-entry
shared-scale isolation (k=0.005 in x64). No floors or tangent rule changes.
The extra source/test lines keep this step below about 300 numerical lines.
Validation: failed-entry first/second derivatives pass in both dtypes; the
fresh-wheel A10 gate passes with own 1080/7 and 1086/1, doctests four
per dtype, acceptance 2332/28 and 2322/38 passed/skipped. Record results,
commit a fix-up, then request
round 2 only after the current round 1 report is written.

Initial implementation evidence at ab7afb1: 198 new derivative-test lines, 10 RMST tolerance
lines and nine changed lines in the existing zero test (217 numerical-test/
tolerance lines total). Focused tests pass 31 per dtype. The fresh-wheel A10
gate exits 0: own 1079/7 x64 and 1085/1 x32 passed/skipped, doctests four
per mode, independent 2332/28 x64 and 2322/38 x32. No algorithm change.
Commit and independent review remain required before core handoff.

# RMST.appendix — Permanent mathematical appendix and conservative guarantees (2026-10-30)

## Round 2 fix-up plan

Maintainer explicitly confirmed S-1b: a NaN RMST result's own derivatives may
be unspecified. Record the confirmation verbatim in D-028 and REVIEW.md;
keep valid-entry isolation mandatory and leave deviation status maintainer-owned.
F-1: extend both existing RMST Docs to-do records with the zero-boundary and
NaN-own-tangent limits. F-2: add the missing gamma-middle section pointer.
N-1: restore the mathematical reason the 24b example needs a wider dtype;
rebuild and visually inspect the adjacent PDF. N-2: label the old report
reference historical. No numerical/spec/API changes; current revision 38
already says what the maintainer approved. S-1a's relation is symmetric;
no further spec wording change is needed to preserve the approved meaning.
Run static/whitespace checks and docs builds, commit the fix-up and invoke
round 2/3 exactly before continuing with the authorized RMST.3 step.

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
higher-derivative limit, first-order-only zero boundary, unspecified own
derivatives for NaN results, and absolute-scale cutoff-derivative accuracy.
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
state staged surface, Weibull shape-derivative limitation, first-order-only
zero boundary and unspecified own derivatives for NaN results. A9/A10 docs
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


---

# Preserved incoming main plan (verbatim pre-merge version)

# README.1 — Draft the alpha README (2026-10-30)

Rewrite README.md at the maintainer's request on add-readme. Preserve the
unfinished existing plan history below. One documentation topic; no API,
dependencies, protected paths, or spec changes. Spec 4.1–4.2, 6.1, 7.3,
9.2, 10.3, 12.2. Exponential/Weibull and concordance are the first built-ins;
more families and metrics are planned, not advertised as available.

Files: README.md, PLAN.md, REVIEW.md. Present current supported likelihood
scope, extension requirements, a core-only runnable gradient example, local
installation, principles and documentation. No new tests: execute the README
snippet against the source in both dtype modes with an analytical check, run
scripts/build_docs.sh directly, and check whitespace. Overwrite REVIEW.md,
commit, then run independent-review for README.1 and stop after its verdict.
Target under 300 changed content lines, excluding existing review records.

---

# SIM.3 — Complete the approved examples diagnostics environment (2026-10-30)

Start clean on release-simulation at
`ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d`; SIM.2 round 1 returned merge.
Preserve the unfinished SIM/A10 plan history below. The maintainer approved
ArviZ 0.23.0 in the existing dependency proposal and now requests the next step.
One concept: finish the examples dependency group for Bayesian tail ESS.

Files and scope:
- `pyproject.toml` / `uv.lock`: add exact `arviz==0.23.0` to examples;
  resolve without upgrading existing packages or changing runtime dependencies.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark the six-package setup
  complete, record the actual installed diagnostics probe and its command.
- Overwrite `REVIEW.md`: reading order, dependency isolation, actual evidence
  and limits; preserve the short temporary probe source for reproducibility.

Spec 10.3; shared use-case 00 dependency-group rule and Bayesian details
Sampling/Diagnostics: NumPyro split R-hat/ESS, ArviZ tail ESS. No source API,
shared generator, ordinary tests, docs/navigation/config or protected scripts.
Docs to-do: the future Bayesian page should use this group for tail ESS.
Target below 300 changed implementation/config/lock lines, records excluded.

Checks: `uv lock --check`, `uv sync --locked --group examples`, compare old/new
lock name/version mappings. Explicit optional-environment probe uses seeded
4x1000 iid Normal draws (known equal-chain law) and deliberately shifted
chains to verify the chain/draw axes, NumPyro summary and ArviZ bulk/tail ESS
and R-hat; this is compatibility evidence, not posterior convergence evidence.
The pinned ArviZ import writes a daily stamp under Path.home()/arviz_data, so
run that probe with required sandbox escalation rather than modifying HOME.
No new ordinary tests for the dependency-only change. Run the existing focused
generator suite in both dtype processes, Ruff, mypy, and unchanged fresh-wheel
through-A10 gate. Record versions/results, write REVIEW.md, commit, invoke the
exact independent-review skill for SIM.3, follow at most 3 rounds and stop.
PH fitting still waits for maintainer integration of reviewed prerequisites.


Implementation: added the approved ArviZ pin and six lock packages. All 165
prior package versions unchanged; runtime dependencies remain jax/numpy.
Actual installed-group CPU probe passes named chain/draw axes, iid gates and
shifted-chain detection: NumPyro R-hat 1.0000270083 / ESS 3976.58; ArviZ
R-hat 1.0001031616 / bulk ESS 3979.33 / tail ESS 3931.29. ArviZ's expected
import warning was emitted; sandbox escalation allowed its home-cache stamp.
Existing generator suites: 17 passed per mode (3.60s/3.18s), Ruff/mypy/lock
checks pass. Fresh-wheel through-A10 gate exited 0: own suites 1065/1071
passed (7/1 skipped), doctests 4 per mode, independent 2333/2324 passed
(27/36 skipped). ArviZ's pandas dependency enabled the existing x32 pandas
conversion test. Wheel hash is identical to SIM.2 and runtime METADATA still
requires only jax/numpy. REVIEW.md records exact commands, outputs and limits.
Commit and exact independent review follow; stop after the verdict.

---

# SIM.2 — Address shared foundation questions and enable examples (2026-10-29)

Start on `release-simulation` at `3556d12c1abca553f77607c1e1739df2bd1911f0`.
The only initial dirty file is study.py with three maintainer question comments.
SIM.1 round 2 returned merge. Preserve all unfinished history below.
Maintainer approves the proposed dependencies in this turn; no renewed
approval is needed for those exact pins. Keep PH fitting deferred.

Files and scope:
- `pyproject.toml` / `uv.lock`: examples group with exact tested Optimistix
  0.1.0, Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0.
  Keep existing runtime/dev/docs versions unchanged. The five-package lock
  preview is 160 changed lines; full six-package setup is 295 lock lines
  alone. Approved ArviZ 0.23.0/tail ESS is a separate later step so this step
  including generator checks remains about 300 lines. No protected scripts.
- `examples/weibull_simulation/study.py`: answer/consume the maintainer notes,
  define quadrature rules before their closure, and explain calibration as
  once-per-target host setup outside the compiled replicate loop. PH input
  preference is asked asynchronously (baseline/log-HR versus PH rate m).
  Preserve the current call contract; any PH convenience will be optional,
  demo-only, and checked against the existing scale formulation.
- `tests/unit/test_simulation_generator.py`: if adding PH input, analytical
  PH/scale and calibration identities, row broadcasts and transformed values/
  derivatives. Reuse existing tolerances. Existing numerical checks also
  exercise the quadrature block reorder.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark approval, staged pins,
  remaining approved ArviZ stage and exact optional-environment invocation.
- Overwrite `REVIEW.md` with reading order, input-contract updates, evidence
  and limits, preserving SIM.1's outcome in this plan and Git history.

Spec 4.4/4.6/7.3/10.3; use-cases/00-common and 01-details DGP/calibration.
No library API, shared docs/config or other workstream files. Docs to-do:
future simulation page should show PH inputs and setup versus compiled work.
Run generator checks in separate x64/float32 processes, approved-environment
compatibility probe, Ruff, mypy, lock consistency and fresh-wheel through-A10
gate. Inspect size before committing; split if needed. Commit, invoke exact
independent-review skill for SIM.2 (max 3 rounds), then stop/report.


Implementation: optional log_hazard_ratio=0.0 added to both generator and
calibration, with unchanged no-keyword calls and U/C key split. Rules now
precede the fraction closure; calibration explicitly remains host-only setup.
No response to the optional PH-form preference arrived during environment
work; used the stated baseline/log-HR convention from 01-details and spec 4.4.
Approved five-package group resolves/installs; all 155 existing locked
versions unchanged, 10 additions. Optional compiled BFGS/NN-update/NUTS/SVG
probe passes in the actual group environment. ArviZ remains approved for its
separate lockfile step. Total implementation/test/config/proposal diff is
303 changed lines, including 160 lockfile lines.

Focused final suites: 17 passed per dtype (10.17s/10.56s), Ruff/mypy/lock
checks pass. A joint three-parameter check_grads initially failed a float32
second-order projection near zero; no tolerance was changed. Retained the
original k/b finite-difference checks and verified all PH gradient/Hessian
entries against inverse-CDF algebra, including cross-partials. The full
fresh-wheel through-A10 gate exited 0: own suites 1064/1071 passed (7/1
skips), doctests 4 per mode, independent 2333/2323 passed (27/37 skips).
NumPyro enables one acceptance check previously skipped. REVIEW.md records
exact commands, times, initial numerical failure and final focused evidence.
Commit and exact independent review follow, then stop/report.

---

# SIM.1 — Shared Weibull generator and censoring calibration (2026-10-29)

Work on `release-simulation` from full base
`8757c0fda78e34e6e818249adedd2a144a33c33d` (clean launch snapshot).
The matching reviewer checkout already exists. Preserve the unfinished A10
history below; its docs and A9/PH/RMST work belong to other implementers.

One concept: reusable demo-only inverse-CDF generation and host-side numerical
calibration, without PH, fitting, core changes or optional dependencies.
Spec revision 37: 3.3, 4.4 (generating PH-equivalent scales only), 7.3,
10.3, 11.4, 12.1–12.3; use-cases/00-common.md and both 01 files.

Files and contract to implement:
- `examples/weibull_simulation/study.py`: `generate_weibull(key, shape, scale,
  censor_limit, *, sample_shape=())`, returning a named tuple with `time`,
  `event`, `event_time`, `censor_time`. Parameters broadcast to row/batch axes;
  sample_shape prefixes those axes. Exactly split the supplied key into U/C
  keys; callers derive scenario/replicate keys with the specified fold/split
  scheme. JAX generation supports jit/vmap and selected process precision.
- Same module: `calibrate_censoring(shape, scale, target) -> float` uses NumPy
  float64 quadrature and bracketed bisection of the mean integrated survival.
  Concrete positive finite broadcast parameters only; calibration is host-only,
  not an RMST implementation or differentiable library API. Check quadrature
  convergence; fail explicitly rather than return an unverified calibration.
- `tests/unit/test_simulation_generator.py`: analytical exponential calibration,
  independent mpmath Weibull quadrature, heterogeneous broadcasting, key mapping,
  transformation agreement and fixed-seed large-sample survival/HR/censoring
  recovery with justified statistical bounds. Centralize any new test bounds
  in `tests/impl_tolerances.py`. Ordinary checks remain offline.
- `examples/weibull_simulation/DEPENDENCIES.md` and a temporary isolated environment
  probe (record the exact invocation in REVIEW.md): concrete exact-version proposal for the later examples group, test
  Optimistix/Matplotlib/Equinox/Optax/NumPyro against locked Python/JAX/NumPy.
  No pyproject/lock edits or added diagnostics/loader packages in this step.
- `REVIEW.md`: reading order, invariants, evidence and limitations; freeze the
  reviewed consumer import contract and calibrated simulation c values.

Target at most about 300 implementation/test/proposal lines (plan/review
records separate); split fitting, scenario runner, results and docs into later
steps. Docs to-do: executable simulation page and recorded full results after
reviewed PH integration; propose fresh docs command with examples selected.

Checks: focused tests in separate JAX_ENABLE_X64=1/0 processes, Ruff formatting
and lint, mypy src, unchanged `scripts/check.sh --through A10` cumulative gate.
Record exact versions, commands and outputs. Commit the step, invoke the exact
independent-review skill command for SIM.1, follow up to three rounds, then stop.
Dependency installation into the project still awaits maintainer approval;
only an isolated proposal probe is authorized here. Never merge/push/publish.


Implementation: 300 added source/test/proposal lines, excluding plan/review
records; no library API, dependencies, lockfile or protected-file edits.
Reviewed consumer contract is recorded in REVIEW.md pending round 1.
Both simulation targets independently match 40-digit mpmath quadrature;
focused final suites: 12 passed in each dtype, including second derivatives.
Ruff formatting/lint and mypy pass. Calibrated c values: 47.793149053990135
and 19.17253543619806 years; pilot empirical censoring .249355/.550485.
Dependency versions proposed and tested in isolation: Optimistix 0.1.0,
Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0 against the
existing locked JAX/jaxlib 0.11.2 and NumPy 2.5.3; group/lock approval pending.
Cumulative fresh-wheel through-A10 gate: exit 0, all gates passed.
Own suites 1059/1066 passed (7/1 existing skips); independent 2332/2322
passed (28/38 existing skips); source doctests 4 per mode. REVIEW.md records
exact timings, commands and the focused derivative-test evidence. Commit and
independent round 1 now follow; stop after the review verdict.

Round 1 on 1e3f5fc returned fix-first for F-1: add explicit quadrature-failure
coverage. Add the k=20/b=5/target=.5 RuntimeError regression, and clarify N-1
underflow and N-2 time units in docstrings; N-3 import-helper suggestion stays
conditional. Fix-up is 7 net source/test lines (307 total), no behavior or
contract change. Focused suites: 13 passed per mode (4.32s/4.05s), Ruff and
mypy clean. Commit separately and request round 2 after round 1 completes.

---

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; all incoming A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.d1 — Using and verifying the installed Slice A subset (2026-10-29)

Authorized by "the current work is approved; if more work needed, get to it".
Work on the existing a9b branch from clean bd32ff9. The maintainer completed
the main merge, manually reran `scripts/check_wheel.sh --through A9` without
errors, and cleaned the reviewer-owned throwaway worktree. Record those as
maintainer-reported checks/cleanup, not as runs performed by this agent.
A9.4 is accepted; optional review nits N-1/N-2 do not block this docs step.

One topic: explain the installed subset's verification and limitations.
Spec: 6.1–6.2, 7.2–7.3, 10.3, 10.5, 12.1–12.3. Public API unchanged.
At most 300 changed docs lines. No dependency, source, test, fixture,
benchmark, protected-path or spec edits. PLAN.md keeps only current A9.

Files to create/change:

- Create docs/slice-a-verification.md as executable MyST. Give a compact
  synthetic exact/right example with construction and diagnostic access inside
  jit, value/parameter gradient/Hessian checked against hand algebra, eager
  and prepared/raw agreement, and vmap over rates. Document the smooth
  interior parameter-only guarantee and explicit host checkpoints.
- Explain the current subset and unsupported likelihood modes; link existing
  construction, likelihood, numerical, distribution/grid and concordance
  guides. Describe the fresh-wheel runner, installed out-of-tree custom-family
  checks, pinned synthetic R evidence and the exact CPU reproduction commands.
  Explain first-call versus warmed costs, prepared/traced flags, returned
  diagnostics, logical bytes versus unmeasured peak memory, and optional
  benchmark nits without silently changing accepted measurements.
- Add the new page to docs/index.md's prose and toctree. Audit the remaining
  executable pages and playground. Correct only stale capability statements
  within this verification topic; leave unrelated teaching work for separate
  steps. Document the locked JupyterLab command.
- The rendered audit found docs/survival-data.md's opening note still claims
  likelihoods, distributions and evaluation are unimplemented. Correct that
  one scope note and link current likelihood/concordance guides. This narrow
  factual correction belongs to the installed-subset verification topic;
  it adds no new tutorial or runtime behavior. Total docs remain below 300
  changed lines. Record the discovery before making the edit.
- Overwrite REVIEW.md with reading order, invariants, evidence mapping and
  remaining gate status. Record every Slice A gate item with concrete tests
  or artifacts and honest pass/pending status; this is not an A10/release
  completion claim. The agreed README/use-case example replacement is still
  separate from this small compiled-verification recipe.

Validation ownership: implementer owns fresh `scripts/build_docs.sh` in both
modes, Ruff and rendered-page/browser inspection. The maintainer's single-
runner preference means the independent reviewer should inspect source and
recorded outputs instead of repeating the same build/gate. No full wheel-gate
repeat for docs-only edits: the maintainer just reran it successfully on the
merged baseline, and no runtime/packaging change follows. Inspect links,
outputs, formulas, navigation and tables across the site in both builds;
inspect the new page carefully and record any existing warning/limitation.
Verify committed notebooks still have no outputs/counts. No new unit tests:
the executable page's assertions check its example, and existing suites
already cover the numerical and packaging behavior.

After successful build/source/visual checks, record exact commands/results,
commit this docs step on a9b with "Independent review: round 1/3" in REVIEW.md,
invoke the exact independent-review skill command for step A9.d1 in
../review/code-a9b, address blocker/should-fix findings in separate fix-ups
within three rounds, then stop and report. Do not start another step.

Docs/closure to-do: agreed replacement README/use-case teaching material and
any gaps from this rendered-site/evidence audit remain explicit; don't claim
Slice A or A10 complete until their separate gate items are accepted.

## A9.d1 evidence and audit conclusion

Implemented the verification page, index navigation and one stale data-guide
scope note; no code or notebook changes. Both final docs builds pass, nine
executed documents and 137 code cells per mode. The three new example cells
check exact analytical values, gradients/Hessians, input forms and jit/vmap.
All 1,048 local file/fragment links per rendered site pass. The committed
playground remains output-free. Ruff passes for the repository and the
extracted three-cell notebook; JupyterLab CLI reports 4.6.4. Native Brave
inspection covers all 11 content pages in float64, all pages' accessible
content in float32 and detailed views of the new page, corrected note,
playground and representative grids/formulas/tables. REVIEW.md records final
execution times, initial sandbox/interruption errors, checks and limits.

The final evidence map is included in this step's REVIEW.md rather than
requiring a separate metadata-only audit step. The new three-observation
verification recipe provides tiny synthetic smoke evidence; the larger
agreed README/use-case teaching replacement remains separate work and is
not claimed complete. The maintainer's post-merge fresh-wheel pass supplies
current regression evidence; no runtime/packaging changes follow it.
A9's verification evidence is now ready for independent review and maintainer
acceptance. No claim that A10, Slice A as a whole or the release is complete.
Stop after this docs step's review; no additional implementation starts.

## A9.d1 round-1 follow-up

Round 1 on 2c8cd60 returned fix-first for should-fix F-1: cite the explicit
unsupported-mode NaN/preserved-neighbor requirement and executed triage
recipe in the gate map, and map each narrative item in spec 12.2's paragraph
to its docs page. Existing code/tests already cover this behavior; no new
implementation or test is required. Also fix trivial N-1 (R paragraph line
wrap) and N-2 (clarify that tests/doctests, rather than static checks, run
in separate dtype processes). Code cells remain byte-identical.

The reviewer reran the build because its role requires it, despite the
initial ownership request. Its build passed in both modes (new page 3.12s /
3.07s); Ruff passed. No wheel gate or mutations were rerun. For this prose/
metadata fix-up, the reviewer alone owns round-2 builds/lint. The implementer
checks only the diff and unchanged cells, records F-1/N-1/N-2, commits a
separate fix-up and requests round 2. No duplicate build/full-gate run by
implementer; maximum three rounds, then stop.

---

# A9.4/main merge resolution (2026-10-29)

Authorized by "merged main into the branch; fix conflicts". The maintainer's
merge is in progress on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflict; resolve these metadata files. No public
API, code, docs, dependency, spec or test behavior changes are needed.

- Preserve A9.4's complete plan and evidence below. Independent review round
  1 on 51319c8 returned merge, with optional N-1/N-2; those nits remain open.
- Preserve all main's A10 plan sections verbatim in plans/A10.md, including
  A10.d2, A10.d1 and A10.4. Its older A10 history remains the unchanged suffix
  of that archive. PLAN.md keeps only the current A9 sub-slice. A10's incoming
  review record is preserved in REVIEW.md alongside the A9.4 record. These
  are historical records; this merge starts no new implementation step.
  This corrects the first resolution, which incorrectly kept incoming A10
  sections in PLAN.md; the maintainer explicitly checked the current-slice
  format before completing the merge.
- Preserve every original stage-0 index entry except the intentional
  plans/A10.md archive extension and plans/README.md index description.
  Incoming docs, concordance fixtures/tests, release-agent plans and
  .workmux.yaml remain untouched. The last is an incoming protected-path
  change, not an agent edit.
- Verify both parents' records are retained, no conflict markers/unmerged
  entries remain, whitespace is clean and every other staged object is
  unchanged from the captured merge index, apart from the two archive/index
  changes above. No runtime/build reruns: neither
  conflicted file affects execution, and no manual executable edit is needed.
  Preserve incoming A10.d2's single-runner validation ownership.
- Stage the two resolutions and archive/index corrections; leave the merge commit to the
  maintainer under the working agreement. Do not create a merge commit,
  start the next implementation step or rerun independent review for this
  metadata-only resolution.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.

---

# Preserved incoming main plan (verbatim)

# RMST.final — Combined evidence and maintainer handoff (2026-11-01)

Integration sub-step 3, after RMST.d3 round 1 verdict merge. Step start /
previous step final: 4f8f2dbfa3a39f95e11b1efc34a8675d7509994a.
Original dedicated-worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-rmst; PH.8 through 8acfb9f is already integrated. Retain all
unfinished plan history verbatim below. Do not archive it while this sub-slice
is still under review or begin another sub-slice during this step.

One final documentation/evidence topic, target under 250 changed content lines:
- docs/rmst.md: resolve RMST.d3 N-1 by saying native methods evaluate their
  family's RMST formulas directly; PH constructs the equivalent distribution
  and delegates. Replace only the ambiguous 'independently' sentence.
- plans/release-agents/rmst-evidence.md (new, beside ph-evidence.md): concrete
  requirement-to-test map for spec 8.7 /12.1 /12.2 item 3, combined PH closure
  evidence and links to the PH handoff, public guide/API and appendices.
  Separate independent quadrature references in implementer tests from
  maintainer-owned independent acceptance, which is still pending delivery.
  Record fresh gate/docs checks, retained/new wheel and source-archive paths,
  byte identity/install evidence and measured tolerance coordinates/factors.
  Hand off D-028/D-029/D-030/D-031/D-032 with the existing decisions, never
  change formal statuses. Include named remaining owners and release limits.
- plans/release-agents/rmst.md: link the final evidence record, preserving
  original launch instructions and ownership/history.
- REVIEW.md: overwrite for RMST.final round 1/3, reading order, map validation,
  actual checks, N-1 correction and explicit remaining acceptance/release work.

Current AGENTS/CLAUDE and release handoffs read; spec 8.7,10.5,12.1–12.2
re-read in full, plus D-028–D-032 and acceptance TESTMAP/Q-049. Spec remains
revision 41, design 00e7225b6b10b0d4e3137f7cf85c0b76e377d821, matching
.spec-seen. No source/API/test/tolerance/dependency/appendix/protected change.
No new unit tests for this records/prose step; executable guide assertions
and unchanged regression gate provide verification.

Validation: scripts/check_wheel.sh --through A10 on the final combined tree;
scripts/build_docs.sh directly, no redirect/env prefix/wrapper, using its
approved escalation. Keep two dtype processes, test selection and assertions
unchanged. Inspect both rendered guides, all 3 RMST API entries, outputs,
links/anchors and exact appendix download. Retained figure/browser limitations
from RMST.d3 remain unless newly inspected. Build a fresh offline sdist,
safely unpack it, build the archive wheel via the documented built-in backend
from its directory (no new dependency). Compare all 23 source files and all
31 wheel members/complete bytes; install archive wheel in fresh external
runtime-only environments in both dtype processes via existing install tests.
Retain artifacts/hashes and the prior installation evidence. Do not claim
isolated PEP-517 archive builds pass offline. Verify every mapped test name,
source/test/decision-file identity to step start and full plan preservation;
Ruff/mypy, whitespace, source playground output-free. No new benchmark/R run.

Commit and invoke independent-review exactly, at most three rounds; stop/report.
After a merge verdict, no RMST implementer sub-step is planned. Maintainer owns
merging release-rmst into main, formal deviation disposition and independent
PH/RMST acceptance delivery/sync plus subsequent cumulative run. Release-wide
demo/audit owners retain their work; publication is separate. No merge, rebase,
push, publication, protected test/spec edit or acceptance weakening here.

Implementation/evidence: N-1 clarified; permanent rmst-evidence.md links from
the original handoff and maps all staged requirements, all five decisions,
installation evidence and pending owners. All 33 explicit test references,
Markdown links, prior plan preservation and source/test/decision/protected
identity checks pass. Fresh wheel-backed A10 gate exits 0: implementer
1282 /8 and 1287 /3; doctests 4 each; existing acceptance 2332 /28 and
2322 /38 (float64/float32). Direct strict docs build passes both modes,
16 RMST cells; HTML/API/output/download checks pass. Fresh archive installation
3 passed per dtype; source, 31 wheel members and full bytes match, prior
artifacts retained and hashes unchanged. Ruff/mypy/whitespace pass. Exact
commands, counts, paths/hashes and limitations in handoff and REVIEW.md.
No additional RMST implementer sub-step planned after this final review.

---

# RMST.d3 — Complete PH RMST teaching and availability (2026-11-01)

Integration sub-step 2, after RMST.integration3 round 1 verdict merge.
Step start /previous step final: 393c4e93aba42501c3eb622f525095b1eed4291d.
Original dedicated-worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Work only on clean release-rmst. Preserve all unfinished parent plans below
verbatim; this remains the same unfinished RMST sub-slice, with no archival
commit. PH integration is complete; no source/exports/helper changes needed.

One documentation topic, target under 300 changed implementation/doc lines:
- docs/rmst.md: retain native Exponential/Weibull explanation, add executable
  PH closure identities for both baselines and the (3,1) baseline x (3,2) eta
  batch with explicit cutoff axes. Explain native methods are independent;
  PH delegates to equivalent built-ins. Show first derivatives in baseline
  parameters/eta/tau in both AD modes, supported fixed-shape Hessians, zero
  values/first derivatives, and valid-entry isolation with shared parameters.
  Extend the higher-shape restriction and first-only zero boundary to PH.
  Add a small registered custom baseline showing native and PH RMST raise
  under jit, with no generic fallback. Cover invalid eta/parameter/cutoff
  policies and the accepted D-032 unrepresentable-closure NaN with clean
  original diagnostics outside the equivalent-scale accuracy domain, using
  one all-dtype executable example. D-031 direct/likelihood derivative
  deferral does not relax RMST. Do not expose new APIs or promise full Hessians.
- README.md: advertise PH availability with its guide link and RMST for
  native built-ins and their PH wrappers. Preserve the minimal example.
- docs/api.md: complete now-unblocked RMST.d2 N-3 /integration3 N-2 by adding
  rmst to both native member lists; add guide links. Fix integration3 N-1's
  obsolete PH-planned sentence in the RMST guide.
- .spec-seen: full revision 40->41 diff read (MNIST-only, no RMST change).
  Mark this worktree using ../scripts/spec-since.sh dev-release-rmst --mark;
  no spec edit, no changes in another implementer's checkout.
- REVIEW.md: overwrite for RMST.d3 round 1/3 with reading order, example and
  requirement map, actual checks/results, limitations and remaining handoff.

Read current AGENTS/CLAUDE, release handoffs, PH evidence, spec 4.4–4.5,5,
7.2–7.3,8.7,10.5,12.1–12.2 and D-028–D-032 decisions. Public surface unchanged.
No source/test/tolerance/dependency/build-config/index/appendix/protected edits.
Assertions accompany the committed executable examples; no redundant unit test
for this docs-only step. Use analytical PH closure/derivative identities as
comparison origins and unchanged repository tolerances for regression runs.

Validation: run scripts/build_docs.sh directly (no redirect/env prefix/wrapper),
executing separate dtype processes with warnings/errors fatal. Inspect generated
RMST page/output/links/appendix download, home/sidebar navigation and all three
RMST autodoc entries in both modes. Visually inspect the retained figure; only
claim full-browser inspection if it succeeds. Run focused API-reference,
native RMST and PH RMST regressions in both dtype processes, Ruff/mypy and
whitespace checks. No full numerical/wheel rerun for unchanged code: prior
393c4e9 REVIEW.md retains full combined A10 and wheel/source-archive evidence.
Commit, invoke independent-review exactly (max three rounds), then stop/report.

Remaining sub-step 3: final combined evidence map and maintainer decision
handoff for D-028/D-029/D-030/D-031/D-032, with formal statuses unchanged.
Retain installation artifact paths/hashes and the independent PH/staged RMST
acceptance-delivery gap. No tests/spec edits; no release-wide demo claims.
Maintainer merges release-rmst into main after reviewed completion. No push
or publication; generic summaries/quadrature and new core work are out of scope.

Implementation/evidence: guide adds PH closure/batch/derivative/isolation/
zero/custom-failure examples and all agreed limitations; README/API gaps
N-1/N-2 fixed. Direct strict docs build passes both modes, 16 RMST cells;
HTML/figure/download inspection passes within the recorded browser limitation.
Focused native/PH/API regressions: 72 passed per dtype. README analytical
checks, Ruff/mypy, whitespace and complete plan/source preservation checks
pass. Content diff remains under 300 lines; historical installation artifacts
retained and rehashed, with exact evidence in REVIEW.md. Commit/review then stop.

---

# RMST.integration3 — Integrate reviewed PH through PH.8 (2026-11-01)

Sub-step 1: resolve the maintainer-started PH merge, validate the combined
tree and retain installation evidence; then commit, independently review,
and stop/report. Do not begin the narrative follow-up in this step.

Commit checkpoints (AGENTS.md rules 11–12):
- Step start /previous RMST step final:
  907d5cbaa0feed36f73554a0c483a5a2bffc47b6 (RMST.d2).
- Incoming reviewed PH.8 tip /existing MERGE_HEAD:
  8acfb9f866e6e530b762724ed4ec4c65d3cf1230 (PH.8 verdict merge).
- Original dedicated-worktree base:
  8757c0fda78e34e6e818249adedd2a144a33c33d.
Work only on release-rmst; finish this existing merge, with no additional
branch merge, rebase, push or publication. Current agreement and full
plans/release-agents/ph-evidence.md were read before editing.

Conflict-resolution plan (no new numerical implementation):
- PLAN.md: retain both complete parent versions verbatim below this active
  section, labelled by workstream. Do not discard/archive unfinished history.
- REVIEW.md: overwrite for RMST.integration3 round 1/3, with parent/scope
  invariants, combined checks and evidence. Both earlier complete records
  remain in Git at the parent commits (RMST.d2 and PH.8).
- docs/index.md: start with PH's current feature description and navigation;
  restore the RMST guide prose link and toctree entry from this workstream.
  Preserve both guides without claiming a final release gate is complete.
- Preserve all other staged changes exactly as reviewed PH.8, including the
  protected CLAUDE.md import. No authored protected-file/spec/test changes.
  Keep _numerics.py removed and _rmst.py as the numerical helper; keep both
  permanent mathematical notes/PDFs in appendices/ unchanged.

Validation: assert complete plan preservation and exact source/test/appendix/
imported-file equality to the appropriate parent; no unmerged entries or
conflict markers; Ruff/mypy and whitespace checks. Run the unchanged
scripts/check_wheel.sh --through A10 on this combined tree and invoke
scripts/build_docs.sh directly with no redirect/env prefix/wrapper. Retain
the direct wheel, source archive and archive-built wheel, verify source/member
identity, and run tests/install/test_wheel.py against the archive-built wheel
in separate dtype processes. Use the PH.8 documented offline unpacked-directory
backend route; add no dependency or committed helper/build script. Record
artifact paths/hashes, exact results and the isolated PEP-517 route's known
uncached-backend limitation. Do not claim independent PH/RMST acceptance:
maintainer-owned delivery/sync remains pending; tests/spec/ stays read-only.

Remaining sub-steps after this integration is reviewed:
2. Finish executable docs/rmst.md for native Exponential/Weibull plus PH
   delegation: both closures, cutoff axes, zero/parameter derivatives,
   valid-entry isolation, custom-baseline failure and higher-shape limits.
   Explain D-032's clean-original-diagnostics /unrepresentable-closure NaN
   outside the accuracy domain, and that D-031 does not relax RMST. Update
   README availability/links for PH and RMST. Complete the now-unblocked
   docs/api.md handoff: add rmst to both native family member lists; rendered
   PH.rmst is present but native RMST methods are currently absent. Preserve
   small-step limits.
3. Final combined evidence/decision handoff: D-028 (PH constructor), D-029
   (log-parameter derivative-error scale), D-030 (Weibull zero-boundary
   guarantee), D-031 (deferred extreme direct/likelihood derivatives), D-032
   (unrepresentable PH closure). Leave formal statuses maintainer-owned.
   Retain wheel/source-archive evidence and the acceptance-delivery gap;
   maintainer then merges release-rmst into main. Release-wide demos,
   independent acceptance delivery and publication remain separate actions.

Known transitional prose: docs/rmst.md still calls PH planned and README
currently lists only native RMST availability. They are deliberately handled
in sub-step 2, after this clean integration is reviewed, rather than mixing
new examples with conflict resolution. Native guide examples remain runnable.
The HTML audit also found RMST.d2 N-3 still incomplete: PH added its RMST
API entry but neither native family member list. This is queued explicitly
with sub-step 2, not claimed as verified in the integration evidence.

Validation completed: wheel-backed A10 gate exits 0. Implementer counts
float64 1282 passed /8 skipped, float32 1287 /3; doctests 4 per mode;
existing acceptance 2332 /28 and 2322 /38. Direct strict docs build exits 0
for both modes (13 pages). Fresh archive-built wheel installation: 3 passed
per mode; source files and complete wheel bytes match, with paths/hashes in
REVIEW.md. All imported paths and both parent plans preserved as planned.
Native API-member omission is recorded for sub-step 2. No protected/spec
changes authored, no new acceptance coverage claimed. Commit/review, then stop.

---

# Preserved RMST parent plan (verbatim 907d5cb)

# RMST.d2 — Resolve guide nits and record API-reference ownership (2026-10-31)

Current branch release-rmst is clean. Maintainer approves RMST.d1 and requests
this next step, noting updated AGENTS.md. Rule 11 now requires final reports
to identify this step's final tip, previous step tip and worktree base. Record:
- Step start: 6d5e19a5a983fda6f236ceb9c15235af26953a2d (maintainer merged
  main instructions into this worktree; do not merge/rebase again).
- Previous step final: 4440f1b6dc09d9b7ae707b341ff39b32cbb7c730 (RMST.d1).
- Original dedicated-worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Preserve unfinished RMST and all inherited plan history below.

One small documentation follow-up, well under 300 changed content lines:
- docs/rmst.md N-1: replace the vague broadcasting warning with the exact
  (3,) cutoff against (3,2) batch TypeError, and show that a (2,) cutoff
  pairs with arms and returns (3,2). Add one executable example checking the
  error plus each paired arm against the previously computed explicit grid.
- docs/rmst.md N-2: use a user-facing support heading and availability text;
  remove branch/merge/release-step wording. RMST support for PH models remains
  planned in this snapshot; do not import another workstream's code or claim
  its pending integration is complete.
- N-3 Docs to-do / named handoff: release-ph owns docs/api.md. In its API-reference
  update, add rmst to BOTH Exponential and Weibull :members: lists so their
  public rmst docstrings render, and document ProportionalHazards.rmst when it
  becomes available. RMST must not edit that shared file concurrently. The
  subsequent combined PH/RMST evidence audit must check those rendered entries.
- .spec-seen: read the new spec revision 40 diff (4.4/7.3/12.1/12.2 PH-only
  extreme-scaling derivative deferral; staged RMST contract unchanged), then
  mark this dedicated worktree against the read design HEAD. No spec edit.
- REVIEW.md: overwrite for RMST.d2 round 1/3, reading order, N-1/N-2 fixes,
  N-3 ownership record, executed examples, checks and remaining integration.

Spec 5/8.7 broadcasting and unsupported-family surface, 10.5 executable docs;
review RMST.d1 N-1 to N-3. No source, test, tolerance, public API, dependency,
protected-file, docs/conf.py, docs/api.md or README changes. No new standalone
unit tests for a reversible docs-only change; notebook assertions are committed
with the examples. Run scripts/build_docs.sh directly, separately executing
both dtype modes via its unchanged runner (approved escalation as needed).
Check generated HTML/outputs/links, Ruff/mypy, API-reference regression and
whitespace; no full numerical rerun for unchanged code. Browser access was
unavailable in RMST.d1; do not claim full-browser layout verification without
an actual successful inspection. Commit and invoke independent-review exactly
as RMST.d2; follow at most three rounds, then stop/report. No push/publication.

Remaining after this step: release-ph's closure implementation/API reference,
then PH RMST examples and combined dtype/docs/release checks. RMST numerical
core and the original mathematical appendix are complete; no new core feature
is planned. Deviation status changes remain maintainer-owned.

Implementation/evidence: all three prior nits addressed within the planned
scope. Direct strict docs build exits 0 in both dtype modes, executing nine
RMST cells. The new error/paired-arm messages and identities pass; generated
HTML/anchors/local links/download checks pass. Ruff/mypy/API-reference check
pass; whitespace clean. No code/test/API-reference file changes or new core
feature. Commit and independent review follow, then stop/report.

---

# RMST.d1 — Executable RMST guide and current-scope README (2026-10-31)

Start clean release-rmst at aae68920769855aa08227f4b7c4818ce7a820923,
now also main. Maintainer confirms core integration into main/release-ph and
explicitly requests item 2. RMST remains the unfinished sub-slice; preserve
all plan history below rather than archive it. Integration review round 1
returned merge, with N-1 (README says RMST planned) assigned to this step.

One documentation topic, target about 250 changed content lines, no public
API/code/test/tolerance/dependency/spec changes:
- docs/rmst.md: new executable MyST-NB guide with current Exponential/Weibull
  examples. Define restricted expectation/area and units; explain cutoff choice
  and model dependence; stable closed forms, accuracy range, explicit cutoff
  axes/draw-arm contrasts, invalid value/type policies, native first derivatives,
  supported nonshape Hessians, higher-shape NotImplementedError, first-order-only
  Weibull zero boundary, and unclaimed NaN-own tangents with valid isolation.
  Keep shape separate for supported higher derivatives. State unsupported
  summary methods/families and absence of grids/options/info/sampling APIs.
  Link the existing mathematical appendix PDF as a download, without moving
  or duplicating its mathematical source into docs/. No new numerical proof.
- README.md: mark core RMST as available, retain means/median/quantile as future;
  link the guide. Resolves RMST.integration N-1 without advertising PH as merged.
- docs/index.md: one guide link and toctree entry plus a narrow current-RMST
  scope sentence. Maintainer's requested guide includes discoverability; A9
  navigation changes have merged. No docs/conf.py or build-tool edits, and no
  API-reference/export entry edits owned by PH.
- REVIEW.md: overwrite for RMST.d1 round 1/3 with spec/API scope, executable
  evidence, rendered-page assessment, checks and remaining PH-integration gate.

Read spec 4.5, 7.3, 8.7, 10.5, 12.1–12.2 and release handoffs. Examples use
only current public API and synthetic parameters, execute in separate dtype
processes via scripts/build_docs.sh directly (approved escalation if kernels
need process access), and assert shape/boundary/analytic identities. No new
standalone tests for a reversible docs-only change. Run Ruff/mypy, whitespace,
API-reference check and inspect rendered HTML in both modes. Existing numerical
regression already passed on this exact core; do not rerun a full numerical
suite without new code/failure concerns. Commit and invoke independent-review
exactly for RMST.d1, at most three rounds; then stop/report.

PH is not in this checkout yet: no wrapper imports or unmerged examples.
Label it as a separate pending release integration. Remaining after this step:
PH closure integration by release-ph, then combined PH/RMST executable evidence
and release checks (item 3); publication remains a separate maintainer action.

Implementation/evidence: docs/rmst.md is 216 lines; README changes five and
index adds four content lines. Eight new executable cells pass in both dtype
processes under the direct strict docs runner. The final build exits 0 in
both modes; HTML/anchors/outputs/local links and the exact appendix download
check pass. Browser host unavailable: full viewport layout not claimed;
rendered figure inspected instead. Ruff/mypy and API-reference test pass.
Commit and independent-review are the remaining actions for this step.

---

# RMST.integration — Resolve the maintainer-started main merge (2026-10-31)

Branch: release-rmst. Merge already in progress when requested; finish that
merge only, without rebasing, publishing, or merging this branch into main.
Parents: ddd7f1f808eea0c802848bf0489135b547fed67f (RMST) and
4f170654fdc5bb2840f89232e439e33d7e5fc3ff (incoming main).

Conflict-resolution plan, before edits:
- tests/impl_tolerances.py: preserve both independently added blocks verbatim:
  measured RMST factors and SIMULATION_SIGMAS=6. No tolerance value changes.
- PLAN.md: keep this active integration/RMST handoff first, then preserve both
  pre-merge plan versions verbatim in clearly labelled records below. Do not
  discard or archive unfinished history. Main's existing plans/A10.md remains.
- REVIEW.md: write this integration's reading order, preservation invariants,
  checks and limitations. The prior RMST and README reviews remain available
  verbatim at the two parent commits; no extra review/archive module needed.
- All other staged paths came from main. Preserve them exactly, including
  CLAUDE.md and .workmux.yaml; do not edit their protected contents. No new
  numerical code, public API, dependency or spec change in this integration.

Initial static check found an inherited main issue: README.md's fenced
Python example needs two blank lines before its top-level function. Apply
Ruff's formatting-only correction, preserving the example's behavior; this
is the only inherited-file exception, with no docs/ or protected-path edit.

Validate source equality against the RMST parent; imported-file equality
against main; both plan records as exact substrings; both tolerance blocks;
absence of conflict markers/unmerged paths; Ruff/mypy, locked environment,
focused RMST+generator tests in separate dtype processes, and unchanged
fresh-wheel through-A10 gate. Commit the existing merge, invoke the required
independent-review as RMST.integration (round 1/3), then stop/report.

Remaining work owned by this worktree after integration:
1. Core RMST.1/.2/.3 and math appendix are complete and independently reviewed
   (RMST.3 round 3 merge). Spec revisions 38/39 are now on design main; no
   new spec decision is needed. Deviation status remains maintainer-owned.
2. Handoff base.py/exponential.py/weibull.py closure edits to release-ph once
   this RMST branch is integrated into code main. This local main snapshot
   does not contain wrappers.py or ProportionalHazards yet. PH owns wrappers,
   exports and its own closure/RMST identity/derivative tests; do not duplicate.
3. Then one separate executable docs/rmst.md topic: staged surface; explicit
   cutoff axes rather than rmst_grid; units; invalid values; first-order-only
   zero boundary; unspecified NaN-own tangents; native higher-shape restriction
   and supported nonshape derivatives. Coordinate navigation/API ownership.
4. Recheck the combined PH+RMST documentation/examples and both-dtype release
   evidence after separately reviewed PH integration. No new core RMST feature
   is planned; generic quadrature/sampling/options stay outside this workstream.

Validation complete: 49 focused RMST/generator tests pass per dtype;
Ruff/mypy/lock/preservation/whitespace checks pass. The fresh-wheel A10 gate
exits 0: own 1097/7 x64 and 1103/1 x32, doctests four per mode, independent
2332/28 x64 and 2322/38 x32 passed/skipped. Finish the maintainer-started
merge and obtain the independent RMST.integration review before reporting.

---

# Preserved RMST-side plan (verbatim pre-merge version)

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Other sub-slices are archived verbatim in `plans/` (index: `plans/README.md`).
Still open there: A9 (`plans/A9.md`, "A9 Docs to-do" and "Final evidence
audit and planning checks") and A8 (`plans/A8.md`, "Closure still required").

---

# RMST.3 — Weibull derivative and isolation evidence (2026-10-30)

Start release-rmst at b2150e1fe10df97d7ac2d4e9926a2b8a803f90c4;
original launch base remains 8757c0fda78e34e6e818249adedd2a144a33c33d.
The maintainer explicitly requested the next implementation step. Appendix
review is running in its separate checkout against the committed snapshot;
its findings will be handled before this step is committed. Preserve history.

Status: implementation resumed from appendix fix-up
dc48b24cd89a571a082099f4055cd1669c8ff3cd. The maintainer explicitly
approved unspecified own derivatives for a NaN RMST result (2026-10-30),
settling S-1b. Appendix round 2 returned VERDICT: merge, all findings fixed.

One bounded numerical-verification concept, about 250 changed test lines:

- New tests/unit/test_weibull_rmst_derivatives.py: independent 60-digit
  mpmath quadrature of survival and its shape/scale derivative integrands;
  integrate the scale kernel directly to avoid cancellation in m-r*S,
  and derive the cutoff derivative from survival. Use actual dtype-rounded
  inputs for references. Cross
  k={0.2,0.5,1,2,10,50}, tau/b={0.01,0.1,0.5,1,10,100,1e5},
  b={1e-4,1,1e4}; straddle both branch thresholds at four shapes too.
  Compare eager/jit/vmap forward and reverse natural Jacobians; express
  parameter errors in log-coordinate units k*dM/dk and b*dM/db, divided
  by |M|, so the section 12.1 value-scaled exception respects time units.
  The cutoff derivative uses absolute scale 1 and the S(tau) identity.
- tests/impl_tolerances.py: add only RMST-specific absolute error factors,
  zero relative tolerance. Pre-test full-range probes (142 cases, both
  dtypes) found maxima [shape,scale,cutoff] of
  [1.72e-14,2.46e-15,1.56e-15] x64 and
  [1.29e-5,1.55e-6,6.45e-7] x32 under the above normalization.
  Fix parameter factors 2e-13/5e-5 and cutoff factors 2e-13/5e-6,
  allowing rounding margin without changing any existing tolerance.
- Verify finite supported (b,tau) Hessians across the grid and independent
  analytical Hessian entries on lower-series and gamma interior cases.
  Preserve explicit shape-Hessian NotImplementedError coverage; extend it
  across forward/reverse combinations in the gamma middle branch.
- Shared shape/scale parameters: valid entries beside invalid parameters
  or cutoffs retain the isolated call's first derivatives in both modes;
  supported scale Hessians stay isolated too. Do not assert a guarantee
  about an invalid result's own tangent. Extend zero first-derivative checks
  over shapes/scales, with eager/jit/vmap and debug_nans for valid entries.

Spec 4.5, 7.2–7.3, 8.7 revision 38, 12.1–12.2 and the release handoff were
read in full. No numerical algorithm/public API, wrappers, exports, modules
in src/, dependencies, quadrature tangent, grids or ordinary docs changes.
No new higher-order boundary or shape guarantee. Docs to-do: executable
RMST topic after PH integration, with first-order-only zero boundary,
unspecified NaN-own tangents, and the native higher-shape restriction. Explain
keeping shape outside differentiated arguments for supported Hessians;
packing it with scale can materialize a zero shape tangent in native JAX.

Run focused RMST tests in separate x64/float32 processes, Ruff/mypy,
whitespace checks, and unchanged scripts/check_wheel.sh --through A10
(fresh wheel supplies the standard gate's required install artifact).
Write REVIEW.md with exact outputs, commit, invoke independent-review
exactly as RMST.3, handle up to three rounds and stop/report. PH owns
wrappers.py/exports and must wait for maintainer core integration and an
explicit ownership handoff before built-in closure edits. No merge/push.

## RMST.3 approved tolerance clarification

Maintainer explicitly answered “Confirm log-parameter scaling” to S-1/D-029.
Approved clarification: change spec 12.1's exception to name k*dM/dk and b*dM/db, with cutoff error
still at absolute scale 1. Bump specification revision 38 to 39, commit alone
on design release-rmst, then mark this worktree via spec-since.sh
dev-release-rmst --mark and commit .spec-seen plus decision/review records.
Committed separately in design as
1389af4da8cd23f7c0d38da047ba14b5ac0801f2; .spec-seen now records it.
No numerical code, tests or factor changes are needed. Leave D-029's status
maintainer-owned. Finish round 2's source-fix review before invoking round 3
on the approved contract/records; do not exceed three rounds for RMST.3.

## RMST.3 round 2 preparation

Round 1 at ab7afb1 returned VERDICT: merge. Record S-1 as D-029 and ask
whether to clarify 12.1 in log-parameter units; do not edit the spec until
answered. N-1: narrow the old Weibull invalid-cutoff tangent assertion to
valid entries only, matching the confirmed unclaimed NaN-own-tangent rule.
Apply the same correction to the analogous Exponential test. The independent
precision-failure fix is already committed at 106078d. Run those focused
isolation tests per dtype, record round-1 outcome and final gate evidence,
commit records/test corrections and invoke round 2 on the fix and N-1.

## RMST.3 precision-failure isolation fix-up

A post-commit probe found a spec 8.7 isolation violation outside the accuracy
range: float32 k=[2,0.01], b=1.25 shared, tau=[0.75,0.01] gives a valid
first value and a NaN failed second value, but the first reverse scale derivative
is NaN (forward is finite). The gamma probability underflows; the inactive
exp(log_value) then overflows because its substituted probability was one.
Protect the exponentiation operand before it runs in src/survjax/_numerics.py;
keep NaN as the failed value. Add a dtype-specific regression for failed-entry
shared-scale isolation (k=0.005 in x64). No floors or tangent rule changes.
The extra source/test lines keep this step below about 300 numerical lines.
Validation: failed-entry first/second derivatives pass in both dtypes; the
fresh-wheel A10 gate passes with own 1080/7 and 1086/1, doctests four
per dtype, acceptance 2332/28 and 2322/38 passed/skipped. Record results,
commit a fix-up, then request
round 2 only after the current round 1 report is written.

Initial implementation evidence at ab7afb1: 198 new derivative-test lines, 10 RMST tolerance
lines and nine changed lines in the existing zero test (217 numerical-test/
tolerance lines total). Focused tests pass 31 per dtype. The fresh-wheel A10
gate exits 0: own 1079/7 x64 and 1085/1 x32 passed/skipped, doctests four
per mode, independent 2332/28 x64 and 2322/38 x32. No algorithm change.
Commit and independent review remain required before core handoff.

# RMST.appendix — Permanent mathematical appendix and conservative guarantees (2026-10-30)

## Round 2 fix-up plan

Maintainer explicitly confirmed S-1b: a NaN RMST result's own derivatives may
be unspecified. Record the confirmation verbatim in D-028 and REVIEW.md;
keep valid-entry isolation mandatory and leave deviation status maintainer-owned.
F-1: extend both existing RMST Docs to-do records with the zero-boundary and
NaN-own-tangent limits. F-2: add the missing gamma-middle section pointer.
N-1: restore the mathematical reason the 24b example needs a wider dtype;
rebuild and visually inspect the adjacent PDF. N-2: label the old report
reference historical. No numerical/spec/API changes; current revision 38
already says what the maintainer approved. S-1a's relation is symmetric;
no further spec wording change is needed to preserve the approved meaning.
Run static/whitespace checks and docs builds, commit the fix-up and invoke
round 2/3 exactly before continuing with the authorized RMST.3 step.

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
higher-derivative limit, first-order-only zero boundary, unspecified own
derivatives for NaN results, and absolute-scale cutoff-derivative accuracy.
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
state staged surface, Weibull shape-derivative limitation, first-order-only
zero boundary and unspecified own derivatives for NaN results. A9/A10 docs
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


---

# Preserved incoming main plan (verbatim pre-merge version)

# README.1 — Draft the alpha README (2026-10-30)

Rewrite README.md at the maintainer's request on add-readme. Preserve the
unfinished existing plan history below. One documentation topic; no API,
dependencies, protected paths, or spec changes. Spec 4.1–4.2, 6.1, 7.3,
9.2, 10.3, 12.2. Exponential/Weibull and concordance are the first built-ins;
more families and metrics are planned, not advertised as available.

Files: README.md, PLAN.md, REVIEW.md. Present current supported likelihood
scope, extension requirements, a core-only runnable gradient example, local
installation, principles and documentation. No new tests: execute the README
snippet against the source in both dtype modes with an analytical check, run
scripts/build_docs.sh directly, and check whitespace. Overwrite REVIEW.md,
commit, then run independent-review for README.1 and stop after its verdict.
Target under 300 changed content lines, excluding existing review records.

---

# SIM.3 — Complete the approved examples diagnostics environment (2026-10-30)

Start clean on release-simulation at
`ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d`; SIM.2 round 1 returned merge.
Preserve the unfinished SIM/A10 plan history below. The maintainer approved
ArviZ 0.23.0 in the existing dependency proposal and now requests the next step.
One concept: finish the examples dependency group for Bayesian tail ESS.

Files and scope:
- `pyproject.toml` / `uv.lock`: add exact `arviz==0.23.0` to examples;
  resolve without upgrading existing packages or changing runtime dependencies.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark the six-package setup
  complete, record the actual installed diagnostics probe and its command.
- Overwrite `REVIEW.md`: reading order, dependency isolation, actual evidence
  and limits; preserve the short temporary probe source for reproducibility.

Spec 10.3; shared use-case 00 dependency-group rule and Bayesian details
Sampling/Diagnostics: NumPyro split R-hat/ESS, ArviZ tail ESS. No source API,
shared generator, ordinary tests, docs/navigation/config or protected scripts.
Docs to-do: the future Bayesian page should use this group for tail ESS.
Target below 300 changed implementation/config/lock lines, records excluded.

Checks: `uv lock --check`, `uv sync --locked --group examples`, compare old/new
lock name/version mappings. Explicit optional-environment probe uses seeded
4x1000 iid Normal draws (known equal-chain law) and deliberately shifted
chains to verify the chain/draw axes, NumPyro summary and ArviZ bulk/tail ESS
and R-hat; this is compatibility evidence, not posterior convergence evidence.
The pinned ArviZ import writes a daily stamp under Path.home()/arviz_data, so
run that probe with required sandbox escalation rather than modifying HOME.
No new ordinary tests for the dependency-only change. Run the existing focused
generator suite in both dtype processes, Ruff, mypy, and unchanged fresh-wheel
through-A10 gate. Record versions/results, write REVIEW.md, commit, invoke the
exact independent-review skill for SIM.3, follow at most 3 rounds and stop.
PH fitting still waits for maintainer integration of reviewed prerequisites.


Implementation: added the approved ArviZ pin and six lock packages. All 165
prior package versions unchanged; runtime dependencies remain jax/numpy.
Actual installed-group CPU probe passes named chain/draw axes, iid gates and
shifted-chain detection: NumPyro R-hat 1.0000270083 / ESS 3976.58; ArviZ
R-hat 1.0001031616 / bulk ESS 3979.33 / tail ESS 3931.29. ArviZ's expected
import warning was emitted; sandbox escalation allowed its home-cache stamp.
Existing generator suites: 17 passed per mode (3.60s/3.18s), Ruff/mypy/lock
checks pass. Fresh-wheel through-A10 gate exited 0: own suites 1065/1071
passed (7/1 skipped), doctests 4 per mode, independent 2333/2324 passed
(27/36 skipped). ArviZ's pandas dependency enabled the existing x32 pandas
conversion test. Wheel hash is identical to SIM.2 and runtime METADATA still
requires only jax/numpy. REVIEW.md records exact commands, outputs and limits.
Commit and exact independent review follow; stop after the verdict.

---

# SIM.2 — Address shared foundation questions and enable examples (2026-10-29)

Start on `release-simulation` at `3556d12c1abca553f77607c1e1739df2bd1911f0`.
The only initial dirty file is study.py with three maintainer question comments.
SIM.1 round 2 returned merge. Preserve all unfinished history below.
Maintainer approves the proposed dependencies in this turn; no renewed
approval is needed for those exact pins. Keep PH fitting deferred.

Files and scope:
- `pyproject.toml` / `uv.lock`: examples group with exact tested Optimistix
  0.1.0, Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0.
  Keep existing runtime/dev/docs versions unchanged. The five-package lock
  preview is 160 changed lines; full six-package setup is 295 lock lines
  alone. Approved ArviZ 0.23.0/tail ESS is a separate later step so this step
  including generator checks remains about 300 lines. No protected scripts.
- `examples/weibull_simulation/study.py`: answer/consume the maintainer notes,
  define quadrature rules before their closure, and explain calibration as
  once-per-target host setup outside the compiled replicate loop. PH input
  preference is asked asynchronously (baseline/log-HR versus PH rate m).
  Preserve the current call contract; any PH convenience will be optional,
  demo-only, and checked against the existing scale formulation.
- `tests/unit/test_simulation_generator.py`: if adding PH input, analytical
  PH/scale and calibration identities, row broadcasts and transformed values/
  derivatives. Reuse existing tolerances. Existing numerical checks also
  exercise the quadrature block reorder.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark approval, staged pins,
  remaining approved ArviZ stage and exact optional-environment invocation.
- Overwrite `REVIEW.md` with reading order, input-contract updates, evidence
  and limits, preserving SIM.1's outcome in this plan and Git history.

Spec 4.4/4.6/7.3/10.3; use-cases/00-common and 01-details DGP/calibration.
No library API, shared docs/config or other workstream files. Docs to-do:
future simulation page should show PH inputs and setup versus compiled work.
Run generator checks in separate x64/float32 processes, approved-environment
compatibility probe, Ruff, mypy, lock consistency and fresh-wheel through-A10
gate. Inspect size before committing; split if needed. Commit, invoke exact
independent-review skill for SIM.2 (max 3 rounds), then stop/report.


Implementation: optional log_hazard_ratio=0.0 added to both generator and
calibration, with unchanged no-keyword calls and U/C key split. Rules now
precede the fraction closure; calibration explicitly remains host-only setup.
No response to the optional PH-form preference arrived during environment
work; used the stated baseline/log-HR convention from 01-details and spec 4.4.
Approved five-package group resolves/installs; all 155 existing locked
versions unchanged, 10 additions. Optional compiled BFGS/NN-update/NUTS/SVG
probe passes in the actual group environment. ArviZ remains approved for its
separate lockfile step. Total implementation/test/config/proposal diff is
303 changed lines, including 160 lockfile lines.

Focused final suites: 17 passed per dtype (10.17s/10.56s), Ruff/mypy/lock
checks pass. A joint three-parameter check_grads initially failed a float32
second-order projection near zero; no tolerance was changed. Retained the
original k/b finite-difference checks and verified all PH gradient/Hessian
entries against inverse-CDF algebra, including cross-partials. The full
fresh-wheel through-A10 gate exited 0: own suites 1064/1071 passed (7/1
skips), doctests 4 per mode, independent 2333/2323 passed (27/37 skips).
NumPyro enables one acceptance check previously skipped. REVIEW.md records
exact commands, times, initial numerical failure and final focused evidence.
Commit and exact independent review follow, then stop/report.

---

# SIM.1 — Shared Weibull generator and censoring calibration (2026-10-29)

Work on `release-simulation` from full base
`8757c0fda78e34e6e818249adedd2a144a33c33d` (clean launch snapshot).
The matching reviewer checkout already exists. Preserve the unfinished A10
history below; its docs and A9/PH/RMST work belong to other implementers.

One concept: reusable demo-only inverse-CDF generation and host-side numerical
calibration, without PH, fitting, core changes or optional dependencies.
Spec revision 37: 3.3, 4.4 (generating PH-equivalent scales only), 7.3,
10.3, 11.4, 12.1–12.3; use-cases/00-common.md and both 01 files.

Files and contract to implement:
- `examples/weibull_simulation/study.py`: `generate_weibull(key, shape, scale,
  censor_limit, *, sample_shape=())`, returning a named tuple with `time`,
  `event`, `event_time`, `censor_time`. Parameters broadcast to row/batch axes;
  sample_shape prefixes those axes. Exactly split the supplied key into U/C
  keys; callers derive scenario/replicate keys with the specified fold/split
  scheme. JAX generation supports jit/vmap and selected process precision.
- Same module: `calibrate_censoring(shape, scale, target) -> float` uses NumPy
  float64 quadrature and bracketed bisection of the mean integrated survival.
  Concrete positive finite broadcast parameters only; calibration is host-only,
  not an RMST implementation or differentiable library API. Check quadrature
  convergence; fail explicitly rather than return an unverified calibration.
- `tests/unit/test_simulation_generator.py`: analytical exponential calibration,
  independent mpmath Weibull quadrature, heterogeneous broadcasting, key mapping,
  transformation agreement and fixed-seed large-sample survival/HR/censoring
  recovery with justified statistical bounds. Centralize any new test bounds
  in `tests/impl_tolerances.py`. Ordinary checks remain offline.
- `examples/weibull_simulation/DEPENDENCIES.md` and a temporary isolated environment
  probe (record the exact invocation in REVIEW.md): concrete exact-version proposal for the later examples group, test
  Optimistix/Matplotlib/Equinox/Optax/NumPyro against locked Python/JAX/NumPy.
  No pyproject/lock edits or added diagnostics/loader packages in this step.
- `REVIEW.md`: reading order, invariants, evidence and limitations; freeze the
  reviewed consumer import contract and calibrated simulation c values.

Target at most about 300 implementation/test/proposal lines (plan/review
records separate); split fitting, scenario runner, results and docs into later
steps. Docs to-do: executable simulation page and recorded full results after
reviewed PH integration; propose fresh docs command with examples selected.

Checks: focused tests in separate JAX_ENABLE_X64=1/0 processes, Ruff formatting
and lint, mypy src, unchanged `scripts/check.sh --through A10` cumulative gate.
Record exact versions, commands and outputs. Commit the step, invoke the exact
independent-review skill command for SIM.1, follow up to three rounds, then stop.
Dependency installation into the project still awaits maintainer approval;
only an isolated proposal probe is authorized here. Never merge/push/publish.


Implementation: 300 added source/test/proposal lines, excluding plan/review
records; no library API, dependencies, lockfile or protected-file edits.
Reviewed consumer contract is recorded in REVIEW.md pending round 1.
Both simulation targets independently match 40-digit mpmath quadrature;
focused final suites: 12 passed in each dtype, including second derivatives.
Ruff formatting/lint and mypy pass. Calibrated c values: 47.793149053990135
and 19.17253543619806 years; pilot empirical censoring .249355/.550485.
Dependency versions proposed and tested in isolation: Optimistix 0.1.0,
Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0 against the
existing locked JAX/jaxlib 0.11.2 and NumPy 2.5.3; group/lock approval pending.
Cumulative fresh-wheel through-A10 gate: exit 0, all gates passed.
Own suites 1059/1066 passed (7/1 existing skips); independent 2332/2322
passed (28/38 existing skips); source doctests 4 per mode. REVIEW.md records
exact timings, commands and the focused derivative-test evidence. Commit and
independent round 1 now follow; stop after the review verdict.

Round 1 on 1e3f5fc returned fix-first for F-1: add explicit quadrature-failure
coverage. Add the k=20/b=5/target=.5 RuntimeError regression, and clarify N-1
underflow and N-2 time units in docstrings; N-3 import-helper suggestion stays
conditional. Fix-up is 7 net source/test lines (307 total), no behavior or
contract change. Focused suites: 13 passed per mode (4.32s/4.05s), Ruff and
mypy clean. Commit separately and request round 2 after round 1 completes.

---

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; all incoming A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.d1 — Using and verifying the installed Slice A subset (2026-10-29)

Authorized by "the current work is approved; if more work needed, get to it".
Work on the existing a9b branch from clean bd32ff9. The maintainer completed
the main merge, manually reran `scripts/check_wheel.sh --through A9` without
errors, and cleaned the reviewer-owned throwaway worktree. Record those as
maintainer-reported checks/cleanup, not as runs performed by this agent.
A9.4 is accepted; optional review nits N-1/N-2 do not block this docs step.

One topic: explain the installed subset's verification and limitations.
Spec: 6.1–6.2, 7.2–7.3, 10.3, 10.5, 12.1–12.3. Public API unchanged.
At most 300 changed docs lines. No dependency, source, test, fixture,
benchmark, protected-path or spec edits. PLAN.md keeps only current A9.

Files to create/change:

- Create docs/slice-a-verification.md as executable MyST. Give a compact
  synthetic exact/right example with construction and diagnostic access inside
  jit, value/parameter gradient/Hessian checked against hand algebra, eager
  and prepared/raw agreement, and vmap over rates. Document the smooth
  interior parameter-only guarantee and explicit host checkpoints.
- Explain the current subset and unsupported likelihood modes; link existing
  construction, likelihood, numerical, distribution/grid and concordance
  guides. Describe the fresh-wheel runner, installed out-of-tree custom-family
  checks, pinned synthetic R evidence and the exact CPU reproduction commands.
  Explain first-call versus warmed costs, prepared/traced flags, returned
  diagnostics, logical bytes versus unmeasured peak memory, and optional
  benchmark nits without silently changing accepted measurements.
- Add the new page to docs/index.md's prose and toctree. Audit the remaining
  executable pages and playground. Correct only stale capability statements
  within this verification topic; leave unrelated teaching work for separate
  steps. Document the locked JupyterLab command.
- The rendered audit found docs/survival-data.md's opening note still claims
  likelihoods, distributions and evaluation are unimplemented. Correct that
  one scope note and link current likelihood/concordance guides. This narrow
  factual correction belongs to the installed-subset verification topic;
  it adds no new tutorial or runtime behavior. Total docs remain below 300
  changed lines. Record the discovery before making the edit.
- Overwrite REVIEW.md with reading order, invariants, evidence mapping and
  remaining gate status. Record every Slice A gate item with concrete tests
  or artifacts and honest pass/pending status; this is not an A10/release
  completion claim. The agreed README/use-case example replacement is still
  separate from this small compiled-verification recipe.

Validation ownership: implementer owns fresh `scripts/build_docs.sh` in both
modes, Ruff and rendered-page/browser inspection. The maintainer's single-
runner preference means the independent reviewer should inspect source and
recorded outputs instead of repeating the same build/gate. No full wheel-gate
repeat for docs-only edits: the maintainer just reran it successfully on the
merged baseline, and no runtime/packaging change follows. Inspect links,
outputs, formulas, navigation and tables across the site in both builds;
inspect the new page carefully and record any existing warning/limitation.
Verify committed notebooks still have no outputs/counts. No new unit tests:
the executable page's assertions check its example, and existing suites
already cover the numerical and packaging behavior.

After successful build/source/visual checks, record exact commands/results,
commit this docs step on a9b with "Independent review: round 1/3" in REVIEW.md,
invoke the exact independent-review skill command for step A9.d1 in
../review/code-a9b, address blocker/should-fix findings in separate fix-ups
within three rounds, then stop and report. Do not start another step.

Docs/closure to-do: agreed replacement README/use-case teaching material and
any gaps from this rendered-site/evidence audit remain explicit; don't claim
Slice A or A10 complete until their separate gate items are accepted.

## A9.d1 evidence and audit conclusion

Implemented the verification page, index navigation and one stale data-guide
scope note; no code or notebook changes. Both final docs builds pass, nine
executed documents and 137 code cells per mode. The three new example cells
check exact analytical values, gradients/Hessians, input forms and jit/vmap.
All 1,048 local file/fragment links per rendered site pass. The committed
playground remains output-free. Ruff passes for the repository and the
extracted three-cell notebook; JupyterLab CLI reports 4.6.4. Native Brave
inspection covers all 11 content pages in float64, all pages' accessible
content in float32 and detailed views of the new page, corrected note,
playground and representative grids/formulas/tables. REVIEW.md records final
execution times, initial sandbox/interruption errors, checks and limits.

The final evidence map is included in this step's REVIEW.md rather than
requiring a separate metadata-only audit step. The new three-observation
verification recipe provides tiny synthetic smoke evidence; the larger
agreed README/use-case teaching replacement remains separate work and is
not claimed complete. The maintainer's post-merge fresh-wheel pass supplies
current regression evidence; no runtime/packaging changes follow it.
A9's verification evidence is now ready for independent review and maintainer
acceptance. No claim that A10, Slice A as a whole or the release is complete.
Stop after this docs step's review; no additional implementation starts.

## A9.d1 round-1 follow-up

Round 1 on 2c8cd60 returned fix-first for should-fix F-1: cite the explicit
unsupported-mode NaN/preserved-neighbor requirement and executed triage
recipe in the gate map, and map each narrative item in spec 12.2's paragraph
to its docs page. Existing code/tests already cover this behavior; no new
implementation or test is required. Also fix trivial N-1 (R paragraph line
wrap) and N-2 (clarify that tests/doctests, rather than static checks, run
in separate dtype processes). Code cells remain byte-identical.

The reviewer reran the build because its role requires it, despite the
initial ownership request. Its build passed in both modes (new page 3.12s /
3.07s); Ruff passed. No wheel gate or mutations were rerun. For this prose/
metadata fix-up, the reviewer alone owns round-2 builds/lint. The implementer
checks only the diff and unchanged cells, records F-1/N-1/N-2, commits a
separate fix-up and requests round 2. No duplicate build/full-gate run by
implementer; maximum three rounds, then stop.

---

# A9.4/main merge resolution (2026-10-29)

Authorized by "merged main into the branch; fix conflicts". The maintainer's
merge is in progress on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflict; resolve these metadata files. No public
API, code, docs, dependency, spec or test behavior changes are needed.

- Preserve A9.4's complete plan and evidence below. Independent review round
  1 on 51319c8 returned merge, with optional N-1/N-2; those nits remain open.
- Preserve all main's A10 plan sections verbatim in plans/A10.md, including
  A10.d2, A10.d1 and A10.4. Its older A10 history remains the unchanged suffix
  of that archive. PLAN.md keeps only the current A9 sub-slice. A10's incoming
  review record is preserved in REVIEW.md alongside the A9.4 record. These
  are historical records; this merge starts no new implementation step.
  This corrects the first resolution, which incorrectly kept incoming A10
  sections in PLAN.md; the maintainer explicitly checked the current-slice
  format before completing the merge.
- Preserve every original stage-0 index entry except the intentional
  plans/A10.md archive extension and plans/README.md index description.
  Incoming docs, concordance fixtures/tests, release-agent plans and
  .workmux.yaml remain untouched. The last is an incoming protected-path
  change, not an agent edit.
- Verify both parents' records are retained, no conflict markers/unmerged
  entries remain, whitespace is clean and every other staged object is
  unchanged from the captured merge index, apart from the two archive/index
  changes above. No runtime/build reruns: neither
  conflicted file affects execution, and no manual executable edit is needed.
  Preserve incoming A10.d2's single-runner validation ownership.
- Stage the two resolutions and archive/index corrections; leave the merge commit to the
  maintainer under the working agreement. Do not create a merge commit,
  start the next implementation step or rerun independent review for this
  metadata-only resolution.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.


---

# Preserved PH parent plan (verbatim 8acfb9f)

# PH.8 — Final PH evidence and acceptance audit (2026-11-01)

Authorized by "ok do the final step". Start /previous PH.7 final:
eeeb47a74e734acbfd40e240496b02c04c974c53, release-ph; only user-owned
ph2.md is untracked. Original launch:
8757c0fda78e34e6e818249adedd2a144a33c33d. Current AGENTS.md/CLAUDE.md,
release packet and PH handoff read. Preserve all unfinished/merged history.

One final PH verification topic; no runtime/API changes expected. Before edits:
- Audit construction, nested diagnostics, all seven direct/grid methods,
  exact/right likelihoods, equivalent-family identities, required ordinary
  baseline/eta/mixed derivatives, invalid-entry isolation, PH RMST closure,
  public exports/install and executable documentation against spec
  4.1–4.7,5–7,8.7,10.2–10.5,12.1–12.2 and the handoff.
- Create plans/release-agents/ph-evidence.md (under 300 changed implementation
  lines) as a durable PH requirement-to-test map with exact current run results,
  artifact identity, limits and remaining integration obligations. Add a link
  to this evidence record in plans/release-agents/ph.md. Do not claim
  this PH audit certifies the whole release or other agents' demos/docs.
- Run unchanged scripts/check_wheel.sh --through A10 once, preserving its fresh
  wheel and output. It includes own tests in both dtype processes, source
  doctests, Ruff/mypy, installed core/custom-family checks and available
  independent acceptance through A10. Confirm dedicated PH tests are included;
  no independent PH acceptance files currently exist, so do not imply otherwise.
- Build/inspect a fresh source archive and wheel from that archive offline,
  outside the source tree, checking PH source/export/type-marker parity. Reuse
  installed-core tests against this artifact in both modes if needed to verify
  the archive route; do not add a generic packaging framework or dependency.
- Run scripts/build_docs.sh DIRECTLY in both modes; inspect final PH outputs,
  navigation/API and math appendix/PDF presence. Do not rebuild unchanged PDF.
- If the audit finds a small PH coverage gap, record the concrete test addition
  here before implementing; no speculative code changes. Split before the
  300-line budget; do not change tests/spec, spec, protected paths or tolerances.

Overwrite REVIEW.md with reading order, invariants, every relevant completion
item/test, exact checks and honest gaps; commit the evidence step and invoke
independent-review PH.8 exactly (up to three rounds), then stop/report full
final/previous/original SHAs. No merge/rebase/push/publish. Leave ph2.md intact.
No new feature, AFT, generic summary, sampling, demo work or RMST-core edit.

PH.8 results: no further PH implementation/coverage gap identified. Fresh
wheel through-A10 gate exits 0: own tests 1282/8 x64 and 1287/3 x32, source
doctests 4/4, independent acceptance 2332/28 x64 and 2322/38 x32 (passed/skipped).
Direct strict docs build exits 0 both modes; five PH outputs, navigation/API,
plot and clean source playground checked. README example independently checked
in both modes. Source archive package bytes match checkout; wheel built from
its unpacked root equals direct wheel in all 31 members/full SHA256. Archive
wheel isolated install/custom-family checks: 3 passed per mode. Initial direct
offline archive build lacked uncached uv_build; unpacked-root fast path worked
without dependency changes. Artifacts/versions/limits recorded in evidence.
Independent TESTMAP/Q-049 confirms PH/RMST acceptance not supplied. README's
stale RMST-planned sentence is included in RMST documentation handoff. User
ph2.md was later absent; this audit never edited/staged/removed it. No library,
test, user-doc, dependency, tolerance, protected path or appendix changes.
Commit final evidence and invoke independent-review PH.8 round 1/3 exactly;
stop/report. Preserve all prior PH/merged history below.

Remaining ownership: independent PH acceptance tests/sync stay with maintainer;
RMST narrative (including PH closure and restrictions) stays with RMST owner;
release-wide integrated demos/evidence/publication are separate. Formal
DEVIATIONS statuses remain maintainer-owned; honor recorded decisions D-028,
D-030, D-031 (spec rev40) and D-032 without reopening or resolving them.

---

# PH.7 — Apply maintainer prose edits and hand off RMST docs (2026-11-01)

Before edits, authorized by the maintainer's request to remove RMST from this
PH guide and incorporate their ph2.md edits. Start /previous PH.6 final:
5c2354e3539668a927c5b355c173bffaf0d63c1c, release-ph. Original launch:
8757c0fda78e34e6e818249adedd2a144a33c33d. Current AGENTS.md/CLAUDE.md,
spec 4.4,7.3,8.7,10.5 read. Preserve unfinished PLAN history below.

One docs topic, below 300 changed documentation lines:
- docs/proportional-hazards.md: apply the two body edits from the maintainer's
  untracked ph2.md: positive eta means higher hazard thus shorter survival;
  remove the survival-probability-ratio sentence. Preserve current navigation
  rather than restoring the draft's stale orphan marker.
- Remove the RMST section, closure table, cutoff examples, summary derivative
  restrictions and RMST overflow example/caveats. Remove the now-dangling RMST
  cross-reference from the direct/likelihood derivative limitation. Keep that
  limitation, PH diagnostics and all five remaining executable examples.
- docs/index.md and docs/api.md: adjust only guide descriptions so they no
  longer advertise RMST content in the PH page. Keep the home capability
  statement, PH toctree/prose links and rmst API autodoc member unchanged.
- Leave ph2.md byte-for-byte untouched and uncommitted (user-owned input;
  initial SHA256 985dc60936bf7504afdfc1e2761c78e97f7d2ec08f3a342a05725ef57e374b82).

Public API unchanged. No source/tests/dependency/protected/spec/deviation or
appendix changes. This transfers narrative documentation ownership; it does
not alter the RMST contract or tests. No new unit tests for prose removal.
Run scripts/build_docs.sh directly in both dtype modes with fresh execution
and strict warnings. Inspect generated guide content, outputs and navigation;
run Ruff/mypy, strict API-reference test and whitespace/scope checks. No
numerical/wheel rerun for this docs-only step. Write REVIEW.md, commit exact
owned paths, invoke independent-review PH.7 exactly, address findings within
three rounds, then stop/report. User ph2.md remains untracked deliberately.

PH.7 results: 94 changed docs lines; both draft body edits applied verbatim,
user draft hash unchanged, five retained code cells byte-identical. Direct
strict docs build exits 0 in both modes, five executed outputs each; PH runs
4.53s x32 /4.25s x64. Rendered content, dtypes, AD/diagnostic outputs and home /
API navigation checked; plot hash unchanged. Ruff/mypy/whitespace clean;
strict API-reference test 1 passed in 0.53s. No code/test/API/protected change.
Exact commands and evidence in REVIEW.md. Commit PH.7 and invoke independent
review round 1/3 exactly; stop/report after review.

Docs to-do /remaining: RMST implementer now owns the separate RMST guide,
including PH closure, derivative restrictions and numerical-failure material.
Final PH evidence/acceptance audit and maintainer guide review remain separate.
No merge, push or publish.

---

# PH.6 — Integrate the PH guide into docs-home navigation (2026-11-01)

The maintainer confirms A9 was merged and its worktree removed, and explicitly
assigns this follow-up HERE: describe the exact changes, make a plan and fix
the missing docs-home link while the guide is still under human review.
This supersedes the old packet's deferred A9 navigation ownership. No A9
work remains assigned to another active implementer for this change.

Start /previous PH.5 final: 414fa845e121eed9ff1cc09f991b920193bdef08,
clean release-ph. Original worktree launch:
8757c0fda78e34e6e818249adedd2a144a33c33d. Current AGENTS.md/CLAUDE.md and
spec 10.5 /12.2 read. Keep all unfinished PH/merged PLAN history below.

Before edits, one small docs-discoverability topic, well below 300 lines:
1. docs/index.md intro: include the existing built-in PH wrapper and staged
   rmst on Exponential/Weibull and their PH wrappers, keeping other summaries
   and unavailable features planned. Remove the unverified old Slice A gate
   status sentence, without claiming A9/Slice A/the release is complete.
2. Add a direct home-page guide-list link to proportional-hazards.md, with a
   concise description of the built-in wrapper, batch/grid use, likelihoods,
   RMST and initial-release derivative limits. Distinguish the custom-family
   recipe link so users can find the real built-in wrapper guide directly.
3. Add proportional-hazards to the root toctree immediately after
   builtin-families, giving home contents, sidebar and ordinary page relations.
4. docs/proportional-hazards.md: remove ONLY orphan: true. Preserve all other
   front matter and the entire body/8 executable cells byte-for-byte, since
   the maintainer is still reviewing it. This addresses PH.5 N-2.

No code/API, test, dependency, config, playground, spec/deviation/protected
path or other guide changes. No new test merely mirrors a Markdown link.
Run scripts/build_docs.sh DIRECTLY, no redirect/env assignment/shell wrapper,
forcing notebook execution in both dtype modes with warnings fatal. Inspect
both generated home main articles for the direct PH link AND contents link,
sidebar presence and guide/home relations; check the guide body/code unchanged
and its eight executed outputs/figure retained. Run applicable static/API
checks and whitespace/scope audit. No core/wheel rerun for this docs-only edit.

Overwrite REVIEW.md, record the public-capability claims against spec 4.4 /
8.7 /12.2 and the exact checks, commit, invoke independent-review PH.6 exactly
(up to three rounds) and stop/report. Report final/previous/original SHAs and
remaining work. Never merge, push, publish or recreate an A9 worktree.

PH.6 results: 15 changed docs lines (index +10/-4, guide orphan -1).
Guide bytes equal the reviewed source with ONLY the orphan line removed;
all prose/cells unchanged. Direct strict docs build exits 0 in both modes.
Both generated HOME main articles contain a direct prose link AND toctree
link to PH, plus the global sidebar entry. Guide relations are prev built-in
families /next likelihoods. All eight executed outputs retained; figure hash
unchanged. Ruff/mypy/whitespace clean; API-reference test 1 passed (0.30s).
No code/dependency/protected/config/playground/body edits or generated output
committed. Exact commands/scope/outputs in REVIEW.md. Commit and invoke PH.6
independent-review round 1/3 exactly; stop/report. PH.5 N-2 is addressed here,
not handed back to A9. No A9 task or worktree needs reopening.

Remaining: final PH evidence/acceptance audit (independent PH spec tests still
not supplied) and the maintainer's ongoing review of the guide. Navigation
will be completed here, not handed back to the retired A9 workstream.

---

# PH.5 — Executable proportional-hazards guide (2026-11-01)

Authorized by “ok on to next part”. Clean release-ph start /previous PH.4 final:
2a341c87b870f5f0baba8269e432db01aedc9f6a. Original worktree launch:
8757c0fda78e34e6e818249adedd2a144a33c33d. Current AGENTS.md/CLAUDE.md,
PH handoff, existing MyST-NB guides and spec 4.4,5,6,7.3,8.7,10.5,12.1–12.2
read. Keep unfinished PH and merged PLAN histories here; do not archive them.

Before docs edits: one PH topic, about 250 changed documentation lines:
- Create docs/proportional-hazards.md as an executable MyST-NB guide. Public
  API only, synthetic examples, existing locked docs dependencies. Cover
  positive eta /hazard direction, positional baseline and keyword-only eta,
  caller-computed predictors, a baseline (S,1) x eta (S,2) with K=S time-first
  grids and indexing, exact/right likelihood and an ordinary eta Hessian.
- Show RMST on Exponential/Weibull via same-family PH closures, scalar and
  explicit cutoff grids, units, unsupported custom closures and remaining
  summaries. State the distinct shape-Hessian and first-order-only Weibull
  zero-boundary restrictions, without weakening nonshape interior AD.
- Document revision-40 extreme-scaling DIRECT/LIKELIHOOD parameter derivative
  deferral accurately: baseline/eta/mixed derivatives may fail despite finite
  true derivatives and accurate values; no mapped safe boundary or blanket
  tail exemption. RMST retains its separate contract.
- Document accepted D-032 behavior: original diagnostics can be clean while
  unrepresentable natural PH closure produces NaN outside mandatory RMST
  accuracy range. Include the explicit float32 example in both dtype builds,
  without changing global x64 mode, issue flags, implementation or tolerances.
- Include a small survival plot contrasting baseline and hazard ratio 2, with
  labeled month/probability axes. Use Matplotlib already in the docs group.
- docs/api.md: add a two-line link from the owned PH section to the guide.
  This is a docs-only PH-topic step, not a broader code-step exception.
- Keep docs/index.md, docs/conf.py, playground and other implementers' pages
  unchanged. Mark this page orphan in its front matter for the pending shared
  index/TOC integration; the API link makes it reachable now. Hand off the
  root navigation entry to maintainer/A9 as the release packet requires.

Tests with the docs: executable cells assert shapes, zero boundaries,
finite ordinary derivatives, declared-issue masks and documented NaN failure.
Analytical expected values in prose come from exp(-t) and the hand-derived
NLL=3*exp(eta)-eta, not the implementation. Do not introduce ad hoc numeric
assertion tolerances or import tests helpers into user examples. Existing
numerical suites already check accuracy; no unrelated library/test changes.

Run scripts/build_docs.sh DIRECTLY, without redirects/environment assignments
or shell wrappers, on both dtype modes. Inspect generated HTML, executed
notebook outputs, API link and rendered figure in both builds. Run applicable
Ruff/mypy/whitespace and API-reference checks. No code change, dependency,
protected path, spec/deviation edit, source/PDF appendix rebuild or publishing.
Record actual results/invariants in REVIEW.md, commit, invoke independent-review
PH.5 exactly (up to three rounds), stop/report with final/previous/original
SHAs and remaining integration/audit work.

PH.5 results: new guide 237 lines plus 3 API-link lines = 240 changed docs
lines. Initial and final direct docs builds exit 0 in both dtype modes, with
12 source pages and fresh notebook execution. Final source wording scopes
ordinary derivative guarantees to built-ins, explicitly states unsupported
likelihood rows yield NaN, and keeps shape out of nonshape Hessian arguments.
HTML outputs checked: eight cells each mode, loss (3,2,3), RMST near0.864665,
leading (3,3,2) grids and explicit f32 NaN failure with clean original issues
in BOTH modes. Plot inspected, same pixels across modes; links work both ways.
Ruff/mypy/whitespace clean; strict API-reference test 1 passed (0.41s). No code,
dependency, test, shared-nav/config/playground or protected-path edit. No full
core/wheel rerun for this docs-only change; same library passed PH.4's gate.
Exact commands, scope, mathematical references and outputs in REVIEW.md.
Commit and invoke independent-review PH.5 round 1/3 exactly; stop/report.

PH.5 round-1 response, planned before edits: verdict fix-first. F-1 identifies
an obsolete two-line PH/AFT future-availability note in docs/distributions.md.
Inspected the ownership handoffs: A10 owns concordance teaching; A9 owns its
verification/playground/navigation work. This already-committed extension
recipe's PH-availability paragraph is not an exclusive remaining assignment.
Narrowly extend this PH topic to that paragraph ONLY: clarify its local class
is a nesting example, link the real built-in PH guide and leave AFT planned.
Do not alter any executable recipe, R-reference material or other sections.
N-1: say direct methods/likelihoods in the general second-AD sentence, so it
cannot be read as overriding the later RMST shape restriction. N-2: retain
orphan/root-navigation handoff; no index/conf/playground change. Rebuild docs
DIRECTLY in both modes, reinspect outputs/links/figure, update REVIEW round
2/3 and commit a fix-up before invoking the skill again. Stay under 300 lines.

Round-2 results: F-1 fixed with a four-line replacement of the two stale
PH/AFT lines; the existing recipe and its other sections remain untouched.
N-1 now explicitly scopes general second AD to direct methods/likelihoods.
Direct strict docs build exits 0 in both modes; new links/prose inspected in
both HTML outputs, all eight guide outputs retained with correct dtypes and
failure case. Figure unchanged, same 490x290 PNG/hash as first inspection.
Ruff/mypy/whitespace clean; API-reference test 1 passed (0.26s). Total docs
diff 247 lines (238 guide +3 API +4 additions/2 removals recipe note).
N-2 stays with the root-nav owner. Commit fix-up, invoke PH.5 round 2/3
exactly and stop/report; no code/dependency/protected change or publication.

Docs to-do /remaining: maintainer/A9 root navigation integration after this
page exists; final PH evidence/acceptance audit, including the independent PH
spec tests when supplied. Existing D-031/D-032 formal status stays open and
maintainer-owned; this guide records the already accepted/spec-aligned limits.

---

# PH.4 — Public exports and installed API integration (2026-11-01)

Start / previous PH.3d final: 5e3868094391e0c5464c92d3775c42902c8ab2cb,
clean release-ph. Original worktree launch base:
8757c0fda78e34e6e818249adedd2a144a33c33d. Reread current AGENTS.md and PH
handoff, spec 4.4,8.7,10.2,12.1–12.2. The PH sub-slice remains unfinished;
preserve all prior PLAN sections without archiving them.

Authorization: after preparing and presenting the exact seven-line
/private/tmp/ph4-api-reference.patch, asked whether it could accompany the
public exports and tests. After the intervening wrappers.py naming question,
the maintainer replied “ok good; now on to the next bit”. Proceed with that
specific next-step approval and its described narrow docs/api.md exception.
No broader docs/ editing permission, spec change or publishing is inferred.
Keep wrappers.py: AFT is assigned there for later Slice D, not this release.

Before code, one small public-surface integration concept, below 300 changed
code/test/implementation lines:
- src/survjax/distributions/__init__.py: import the implemented registered
  ProportionalHazards class and add it to the exact public export list.
- src/survjax/__init__.py: re-export the SAME class at top level, as spec 4.4
  examples and 10.2 require. No aliases/new wrappers or numerical changes.
- tests/unit/test_import.py: retain exact equality of the root export set,
  adding only ProportionalHazards; verify root/distribution/defining-module
  imports are identical, with an exact distribution export set too.
- tests/install/test_wheel.py: retain exact installed root exports, add PH,
  verify both import paths agree and use a positional baseline. Eager and
  jit construction/RMST smoke through the installed core, with only runtime
  deps. Expected rate=0.5*exp(log(2))=1, M(2)=-expm1(-2), independently from
  the exponential integral. Compare outside the isolated environment using
  unchanged tests/impl_tolerances.py constants, not ad hoc tolerances.
- docs/api.md: ONLY the seven-line Proportional hazards autodoc block shown
  for approval, beside built-ins. Leave test_api_reference.py unchanged and
  strict; no missing-export exceptions, skips, dynamic expected sets or
  conditional autodoc. No other docs/ page or build config is touched.

Use the already-prepared syntax/lint-checked proposal only after this plan.
Run focused import/API checks in both dtype processes; unchanged fresh-wheel
A10 gate (all unit/numerical/installed/acceptance evidence); Ruff/mypy and diff
scope checks. Run scripts/build_docs.sh DIRECTLY as the updated agreement
requires, with no redirect, wrapper or environment assignment. Inspect both
rendered API pages for the new class/signature and methods. No new tests of
unchanged numerical internals, dependencies, fixture regeneration, protected
path or generic summaries/AFT/sampling. Record actual results in REVIEW.md,
commit and invoke independent-review PH.4 exactly; up to three rounds, then
stop/report final/previous/original commit IDs and remaining sub-steps.

PH.4 first checks: import/API-reference tests pass, four per dtype (1.80s
both). The first direct docs build fails in float32 on ONE warning from the
newly exposed PH class docstring: inline emphasis is opened by the unquoted
(K, *batch_shape) shape. Before correction, extend this plan narrowly to
wrappers.py: render that existing shape as an RST code literal. No numerical
behavior or API change, no warning suppression or other docs/ edit. Rebuild
docs directly and use a final fresh wheel after the documentation fix.

PH.4 results: root and distribution export the SAME PH class, fixed exact
sets updated by only that name, installed-core eager/jit PH RMST checked
against the rate-1 integral. Focused import/API checks: 4 passed per dtype,
1.80s each. Docs initially exposed the unquoted shape warning; after fixing
that docstring literal, direct scripts/build_docs.sh exits 0 for both modes.
Both generated API pages inspected: positional baseline /keyword-only eta
signature, all six selected method IDs and intact shape literal. No other
docs/ source or config edit. Ruff/mypy/whitespace clean. Final-source fresh-
wheel A10 gate exits 0: own 1282/8 x64,1287/3 float32; doctests 4 per mode;
independent acceptance 2332/28 x64,2322/38 float32 (passed/skipped).
Final wheel export/docstring bytes match source; full commands, timings and
hash/artifact recorded in REVIEW.md. Final implementation diff is 51 lines.
Commit and invoke independent-review PH.4 round 1/3 exactly, then stop/report.

Docs to-do: the separate executable PH topic covering broadcasting, eta>0
hazard direction, exact/right likelihood, RMST, the shape/zero boundary
restrictions, revision-40 extreme direct/likelihood derivative limitation and
accepted D-032 failure behavior. Final PH acceptance/evidence audit follows.
Appendix/PDF alignment is already complete in PH.3d. Optional numerical/
coverage nits stay recorded, not mixed into this export-only step. PH.3d S-1
(spec layout old helper name) remains maintainer/designer-owned; no spec edit.

---

# PH.3d — Give the RMST helper a focused module name (2026-11-01)

Explicit maintainer request: rename _numerics.py because its responsibility
is exclusively RMST, then proceed to the next PH step. Choose _rmst.py for
shared cutoff handling and staged Weibull RMST. Start / previous PH.3c final:
295834c612351c8ea4cff106f7b0208c446674e2, clean release-ph. Original worktree
base: 8757c0fda78e34e6e818249adedd2a144a33c33d. AGENTS.md reread; unfinished
PH PLAN history retained verbatim below. Spec 10.2's illustrative old private
filename is superseded only by this explicit naming instruction; no spec edit,
new abstraction, module responsibility or public API is introduced.

Before code, one bounded rename/alignment concept:
- Move src/survjax/_numerics.py unchanged to src/survjax/_rmst.py. Update
  imports in distributions/exponential.py, weibull.py and wrappers.py. No
  compatibility alias for an unexported private module; leave _log_math alone.
- Update appendices/README.md and appendices/weibull-rmst.md source links.
  Complete the tracked appendix alignment: explain nearby log1p evaluation,
  exact shared power-of-two normalization, derivative safety and inactive
  operands; retain existing tail bounds. Rebuild companion PDF with its
  established Pandoc/XeLaTeX command and inspect rendered pages. Keep this
  permanent appendix outside standard docs/ as the maintainer requested.
- No new tests for a path-only refactor; existing native/PH RMST value,
  transformation, boundary and invalid-entry suites exercise every importer.
  Run Ruff/mypy and the unchanged fresh-wheel A10 gate in both dtype modes;
  inspect wheel contents and confirm byte-identical moved implementation.
- Write REVIEW.md, commit, invoke independent-review exactly for PH.3d and
  report verdict/commits/remaining steps. The maintainer already requested
  continuation after this naming correction; do not treat that as permission
  for the still-unapproved docs/api.md code-step exception. Prepare the exact
  export-reference edit for a maintainer decision before adding public exports.

Spec 8.7,10.2,12.1–12.2. Well under 300 changed implementation lines after
rename detection; no dependencies, tolerance, protected-path, tests/spec or
standard-docsite changes. Historical PLAN/archive references to the old path
are preserved as evidence, not rewritten. PDF source editing uses the
pandoc-safe-markdown and PDF skills; no dependency installation is needed.

PH.3d results: implementation bytes unchanged, SHA256
6fd938fe83c43fab859601b59842900e50c94df1bdee55fd07326b4b0d9a6030.
All live imports/appendix links use _rmst.py; the new wheel contains it and
omits _numerics.py. Ruff/mypy/whitespace checks clean. Standalone PDF builds
and final four pages visually inspected; obsolete forced page break removed
to avoid a mostly empty extra page. No source text extends beyond page bounds.
Fresh-wheel A10 gate exits 0: own 1281/8 x64,1286/3 float32; doctests 4 per
mode; independent acceptance 2332/28 x64,2322/38 float32. Exact commands,
versions, timings, source/PDF/wheel hashes and artifacts in REVIEW.md.
Commit and invoke PH.3d independent-review round 1/3 exactly.

Docs to-do / next PH step: public exports/install/API reference integration,
requiring the explicitly authorized narrow docs/api.md edit or an approved
passing split; retain exact-export/reference assertions. Then executable PH
topic and final PH evidence/acceptance audit. PH.3c optional N-1/N-2 remain
recorded numerical/test improvements; no additional numerical change here.

---

# PH.3c — Preserve RMST log-ratio precision near unit cutoff (2026-11-01)

Authorized by “ok on to the next bit”; reread the current AGENTS.md (symlink
of CLAUDE.md), handoff and numerical spec before writing. Clean release-ph
start / previous PH.3b final: e630d1c721600b6554b406cd8b78309b446a4847.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
The PH sub-slice is unfinished; retain its PLAN history here, without archiving.

One numerical correction, well below 300 code/test lines:
- src/survjax/_numerics.py: for comparable positive cutoff and scale, evaluate
  log(tau/b) as log1p((tau-b)/b), retaining the log-difference route for separated
  scales. Choose equivalent expressions, with safe operands in the discarded
  route before subtraction/division/log1p. No clipping or tolerance change.
  This is needed by both native and PH-Weibull, whose RMST delegates here;
  the reviewed RMST core is already integrated and no parallel core work is
  imported. No public API, dependency, module or protected-path changes.
- tests/unit/test_weibull_rmst_near_scale.py: first reproduce the reported
  float32 cutoff failure, then independent mpmath survival-integral values and
  first derivatives at neighboring cutoffs, several shapes/scales and both
  native/PH laws. Cover either side of the expression switch, eager/jit/vmap,
  all four supported nonshape Hessian compositions and zero/invalid/tail safety
  through the unchanged existing RMST suites. Keep shape outside Hessian args.
- Spec 7.2–7.3, 8.7, 12.1–12.2: ordinary-domain cutoff error 1.06e-5 exceeds
  the unchanged 5e-6 float32 tolerance at k=50,b=10000,eta=0,
  tau=10000.005859375. This is a settled-contract bug, outside neither accuracy
  range nor representability; D-031/D-032 do not excuse it. No new deviation.

Overwrite REVIEW.md, map requirements to tests, record the failing before /
passing after evidence and actual validation. Run focused native/PH suites in
separate dtype processes, Ruff/mypy, unchanged fresh-wheel A10 gate and diff
scope checks. Commit, invoke independent-review exactly for PH.3c (up to three
rounds), then stop/report with final/previous/original commits and remaining
sub-steps. PH.3b nits N-4/N-5 and its D-032 wording question stay outside this
single-concept correction; no spec/deviation interpretation is changed.

PH.3c results: the pre-fix float32 regression failed (four of 30 cutoff
mismatches, maximum 1.06080507e-5). With the correction, the reported forward /
reverse errors are 6.11807e-8 /9.09831e-8. Final focused native/PH suites:
69 passed per mode (184.45s float32 /182.42s x64), including all eight new
cases, 30 native /48 PH oracle points and inactive-route extreme-scale checks.
Ruff format/check and mypy clean; whitespace clean. Final-source fresh-wheel
A10 gate exits 0: own 1279/8 x64 and 1284/3 float32, doctests 4 per mode,
independent acceptance 2332/28 x64 and 2322/38 float32 (passed/skipped).
The implementation diff is 206 lines (196 test +9 source additions /1 removal).
Exact commands, oracle derivations, timings and retained wheel in REVIEW.md.
Commit and invoke PH.3c independent-review round 1/3; then stop/report.

PH.3c round-1 response, planned BEFORE fixes: review verdict fix-first. F-1
requires points at log(tau/b)=+/-1/k,+/-2/k, for k=20,50 at b=1e4, in the same
independent physical-law oracle. Verify that narrowing the switch to 1e-3 now
fails. N-1 (existing extreme-scale float32 round-trip error) stays recorded;
do not loosen tolerances or change gamma assembly in this step.

A separate implementer probe while review ran found a NEW first-derivative
regression at k=2,b=tau=1e-200 float64: the direct relative-difference division
creates reciprocal-square overflow under AD. The old log-difference gives
finite gradients. Fix this within the SAME log-ratio concept: normalize the
already-safe near operands by an exact shared power of two using frexp/ldexp,
then subtract/divide their O(1) mantissas. Its integer exponent carries no
tangent; the real mantissa/cutoff retain their derivatives. Add native/PH
value/first-derivative regressions at scales 1e+/-200 x64 /1e+/-30 float32.
No custom derivative, stop_gradient, clipping, tolerance or new module.
Keep total implementation diff below 300 lines. Update REVIEW.md round 2,
run focused suites in both modes plus the fresh-wheel A10 gate on final code,
commit a fix-up and invoke the skill exactly. Defer appendix/PDF wording
alignment to its docs step as already tracked; no appendix source edit here.

Round-2 results: retain c=+/-1,+/-2 and add +/-0.2,+/-0.4 after the initially
suggested four points still let M4 pass. Final oracle has 46 native /64 PH
points. Threshold mutant now fails (2.64s); unscaled-division mutant fails
(1.09s, NaN gradient). Production restored before checks. Final ten new cases
pass per dtype (33.17s float32 /32.15s x64). Static checks clean. Final-source
fresh-wheel A10 gate exits 0: own 1281/8 x64,1286/3 float32; doctests 4 per
mode; independent acceptance 2332/28 x64,2322/38 float32. Final implementation
diff: 256 lines (243 tests +12 source additions /1 removal), under budget.
Exact commands/artifact/timings in REVIEW.md. Commit fix-up and invoke round
2/3 exactly; stop/report after verdict. N-1 remains an optional recorded
extreme-scale precision investigation, not a relaxed tolerance.

Docs to-do: the existing PH topic, API reference/export integration and final
PH evidence audit remain pending. No new public behavior or approximation is
introduced here; the permanent RMST appendix's log-ratio identity remains
valid. A later appendix refinement may explain the cancellation-safe evaluation.

---

# PH.3b — RMST range, broadcast derivatives and supported Hessians (2026-10-31)

Authorized by “ok agreed and on to the next bit”, resumed after the explicit
request to reread the updated AGENTS.md/CLAUDE.md (the former symlinks to the
latter). Clean release-ph start: 61327fd4cb3a7b8d4f67c4a95b8c09d910c28bb8,
the maintainer's agreement merge. Previous PH step final:
b49638e2aec294bcaa0bab0c07443e2bf249aedd. Original worktree launch base:
8757c0fda78e34e6e818249adedd2a144a33c33d. No interrupted mutation/process was
started: the interrupted turn only read files. Preserve all unfinished history.

PH.3a round 2 left no code/test fix, only D-032; record the maintainer's
acceptance of a detectable NaN outside closure representability/accuracy range
without marking its status resolved. Keep mandatory-domain accuracy and zero
boundary/invalid-neighbor isolation. Spec revision 40 remains read-only;
.spec-seen already records 765c65b. Do not imply the direct/likelihood derivative
exception changes staged RMST. No public-export/docsite integration here.

One numerical-verification concept, about 300 changed implementation/test lines:
- Create tests/unit/test_proportional_hazards_rmst_derivatives.py. No public API.
- Broadcast (S,1) baseline fields × (S,2) eta: first-derivative closure identities
  for rate or shape/scale and eta, scalar cutoff and explicit leading cutoff
  axes (m=S), eager/jit/vmap and forward/reverse Jacobians.
- Independent 60-digit physical PH survival quadrature, with derivative
  integrands, across Weibull k={0.2,0.5,1,2,10,50}, tau/b_PH in
  {0.01,0.1,0.5,1,10,100,1e5}, baseline scales {1e-4,1,1e4}, both eta signs.
  Include both sides of the native lower/gamma/mean branch thresholds. Include
  analogous Exponential rate/eta/cutoff values and derivatives across scales.
  References use actual dtype-rounded inputs, never code-under-test outputs.
- Reuse unchanged native RMST value-scaled parameter and absolute-cutoff bounds;
  eta is already a log coordinate. State normalization and chain-factor scales.
  Measure rather than loosen any tolerance if composition exposes a failure.
- Supported Hessians: all four fwd/rev combinations, full Exponential and
  Weibull scale/eta/tau with shape OUTSIDE differentiated arguments. Check
  finiteness across the grid and independent integral Hessians at interior
  points; include check_grads order=2 in supported directions.
- Restore a shared-eta invalid-neighbor case (PH.3a N-4 defence in depth), plus
  ordinary representable Exponential zero-boundary Hessian evidence if space
  permits. Accepted outside-domain failures must not be mistaken for invalid
  parameters or generalized into allowed in-domain NaNs.
- Anticipate a small closure-precision correction in exponential.py/weibull.py
  or wrappers.py ONLY if these checks expose a settled-contract failure in the
  mandatory domain. Keep all work within about 300 changed code/test lines;
  split optional evidence before exceeding the budget. No new module,
  dependency, protected path, tolerance change, generic summary or sampling.

Read spec 4.4–4.5, 5, 7.2–7.3, 8.7, 12.1–12.2 and the PH handoff. Write
REVIEW.md with oracle derivations, invariants, requirement/test mapping and
actual commands/results. Run focused both-dtype tests, Ruff/mypy, unchanged
fresh-wheel A10 gate, whitespace/scope audit. Commit and invoke independent-review
exactly for PH.3b round 1/3; fix blockers/should-fix within three rounds, then
stop/report. Report final, previous PH-step, step-start and original worktree
base commits, plus remaining sub-steps, under the updated agreement.

PH.3b implementation findings: the first 142-point Weibull /21-point
Exponential grid alternated eta signs and passed all 10 new cases per mode
(46.16s/46.10s). Expand to BOTH eta signs at every shape/ratio/scale tuple,
not just alternating signs. A targeted pre-fix probe then found an in-domain
float32 cutoff derivative error 1.777321e-5 at k=50,b=1e4,eta=0.3 and
physical tau/b_PH approximately 1, versus the unchanged 5e-6 bound. The log(b)
then exp round trip in the private closure, amplified by large k, caused it.
This is a settled-contract ordinary-domain precision failure, not covered by
D-031/D-032. Correct only Weibull._ph_equivalent: use b*exp(-eta/k) when both
factor and final parameter are normal and finite; retain the guarded log-space
route otherwise, substituting operands in BOTH branches before operations.
No clipping, epsilon or tolerance change. After correction, forward/reverse
errors on that point are 1.8775e-8 /4.8577e-8.

The numerical fix adds a direct-product / guarded-log fallback branch.
Replace the optional new shared-eta isolation cases (already passed twice)
with positive-cutoff fallback regressions: the multiplier over/underflows,
but effective scale and RMST remain representable. Existing PH.3a isolation
regressions remain unchanged; N-4 defence-in-depth stays a future coverage item.
This prioritizes regression evidence for BOTH branches of the new code.

Final range grid: 252 Weibull Cartesian rows plus 16 threshold neighbors;
42 Exponential rows. New test module 266 lines; source fix 22 additions/3
removals =25, total 291 changed implementation/test lines. Shared-eta
first/second isolation (PH.3a N-4) and ordinary Exponential zero-Hessian evidence
is left to existing/native tests and later integration rather than exceed
this step's budget. The pre-swap 29 focused cases passed (67.00s float32,66.34s x64), including
shared-eta first/second isolation on the corrected production code. The final
29-case selection with both guarded fallback regressions passes too:
70.89s x64 /71.64s float32. Static checks clean; fresh-wheel A10 gate exits 0
on the final production source: own 1271/8 and 1276/3, doctests four per mode,
independent 2332/28 and 2322/38 (passed/skipped). The gate began before the
optional shared-eta test was exchanged for fallback cases; the production
source did not change and the final focused selection verifies those cases.
Record exact commands/timings/wheel in REVIEW.md, commit and request PH.3b
independent round 1/3. Stop after its verdict.

Docs to-do: executable PH topic with RMST examples, shape-Hessian and zero
boundary restrictions, revision-40 extreme derivative limitation and accepted
D-032 numerical-failure behavior. Exports/install/API-reference integration
still needs the narrow docs/api.md code-step exception; exact export/API tests
remain unchanged. No unfinished sub-slice is archived or called complete.

---

# PH.3a — Built-in PH closure and delegated RMST (2026-10-31)

Authorized by “get on with the PH implementation”. Complete the maintainer's
already-resolved RMST merge as prerequisite commit 2e6b55a9610e6733bf539d420d7a2f6d52ee355e;
start this implementation from that clean release-ph snapshot. No new merge
is initiated. RMST core ownership now passes to PH for closure-only edits,
per the release packet and the maintainer's instruction to continue after
bringing in that work. Preserve the unfinished PH/RMST planning histories.

The maintainer has chosen to defer extreme-range PH derivative improvements
until after the initial release; the designer is preparing the spec exception.
Record F-1/F-2/S-1 and that decision in DEVIATIONS.md (D-031). Do not change
spec, the existing PH direct formula, or the pending derivative tests. Current
revision 39's closure/RMST contract is settled independently of that exception.
During implementation the designer committed revision 40 at
765c65bba9efb3af8ef4a13c1ecb8e302a47f346. Read the full specification diff;
it narrows only extreme-scaling PH direct/likelihood derivatives and leaves
staged RMST unchanged. Run spec-since.sh dev-release-ph --mark for .spec-seen.
D-031 records alignment without changing its maintainer-owned status.

One bounded code concept, <=300 changed code/test/implementation lines (plan,
review and deviation records excluded):
- base.py: a private PH-equivalent hook, default None (no generic summaries).
- exponential.py/weibull.py: implement that hook using log-space transformed
  rate/scale. The internal invalid-entry mask includes original parameters,
  eta and cutoff violations; substitute neutral parameter operands BEFORE
  logarithms, division or exp so invalid rows cannot poison shared derivatives.
  Mask shape E broadcasts across the existing batch and explicit cutoff axes;
  built-in scalar fields have no component axes. No change to RMST numerics
  or tolerance constants. The hook is private, not an extension-interface API.
- wrappers.py: rmst(tau) converts/broadcasts cutoffs through the existing helper,
  computes original issue union, asks the baseline for its PH closure, raises
  NotImplementedError for unsupported families, delegates to closure.rmst and
  restores original invalid rows to NaN. Direct methods/likelihood never
  dispatch through the closure. No exports, other summaries, rmst_grid or sampling.
- New tests/unit/test_proportional_hazards_rmst.py: both closure laws, independent
  mpmath integral and first derivatives, eager/jit/vmap, (S,1)x(S,2) batch
  identities and explicit cutoff grids, zero values/first derivatives, invalid
  rows and shared-parameter isolation, native Weibull shape-Hessian
  restriction and custom-baseline NotImplementedError. Keep shape outside the
  differentiated arguments for supported nonshape Hessians. Use existing
  ordinary/RMST tolerances, with explicit scale justification. Split broader
  RMST tail/full-range evidence into PH.3b if it does not fit; no oversized step.

Spec 4.4–4.5, 5, 7.2–7.3, 8.7 and 12.1–12.2. No dependencies, protected paths,
public export integration or docs-site edits. Docs to-do: executable PH topic,
RMST closure examples, shape-Hessian/zero-boundary restriction and the coming
extreme-derivative limitation. Later export step needs the narrow docs/api.md
code-step exception; do not weaken API-reference or exact-export tests.
Post-release numerical roadmap: map extreme PH derivative failure regimes and
stabilize first/second baseline/eta derivatives; preserve the stress evidence.

After formatting, the first draft reached 337 code/test lines. Split its
(S,1)x(S,2) / explicit cutoff-grid test into PH.3b alongside the broader RMST
range/identity evidence. Simplify the independent Hessian order construction
and defer the full supported-Hessian integral test to PH.3b too: explicit
zero-value/first-derivative handling for an unrepresentable natural closure
is required now. The smaller broadcast/grid test fits again after deferring
the larger Hessian test, so retain the broadcast/grid checks in PH.3a.
Guard closure exp overflow before evaluating it; restore the
nonfinite closure parameter, with a zero-cutoff fallback preserving the exact
value and first derivatives. Ordinary representable closures keep their
native boundary derivatives. Promote copied baseline/eta leaves with the
cutoff dtype before closure arithmetic; original stored leaves are unchanged.
The final step is exactly 300 changed code/test lines: 62 production additions
and 238 test lines. Deferred supported-Hessian/full-range coverage is not
claimed complete in this step.

Run focused PH/RMST in separate dtype processes, Ruff/mypy, unchanged fresh-wheel
through-A10 gate, whitespace and scope checks. Write REVIEW.md, commit, invoke
independent-review exactly as PH.3a round 1/3, address findings within three
rounds, then stop/report this step. Do not start export/docs or PH.3b afterward.

PH.3a execution: implemented 62 source additions and 238 test lines, exactly
300 code/test lines. Final focused tests pass 19 cases in each dtype (15.95s /
16.01s); full fresh-wheel A10 exits 0: own 1261/8 and 1266/3, doctests 4 each,
independent acceptance 2332/28 and 2322/38 (passed/skipped). Ruff format/lint,
mypy and whitespace clean. REVIEW.md records prototype failures, fixes, split,
commands, timings, numerical invariants, wheel identity and deferred evidence.
Revision 40 is read and .spec-seen aligned; no spec/dependency/protected edit.
Commit this step, then invoke independent PH.3a review round 1/3 and stop.


PH.3a independent round 1 on 97dd11a returned fix-first:
- F-1: strengthen the existing seven invalid-row cases so a finite shared
  scalar feeds positive/negative/zero baseline entries via q*sign. Inject
  NaN/inf constants via where; do not create upstream q*NaN/inf arithmetic.
  Rate, shape and scale substitutions must each be observable in reverse AD.
- N-2: reject the base-default (unsupported) hook by class before diagnostics,
  cutoff conversion or casting. The custom test now has a raising diagnostic
  condition to prove that it is not called.
- N-1 / PH.3b release-gate list: first-derivative closure identities for every
  baseline field and eta under (S,1)x(S,2) broadcasting, plus independent
  supported nonshape Hessians (scale,eta,tau; Exponential full Hessians),
  all fwd/rev combinations, eager/jit/vmap, broad RMST range and scale changes.
  Keep Weibull shape outside differentiated arguments for supported Hessians.
- N-3: acknowledge Weibull's 311 lines (301 inherited +10 closure); no unrelated
  refactor/module is added to this step.
- S-1: log D-032 without choosing a policy or changing numerical code. It is
  a separate RMST value question, not covered by revision 40's derivative
  exception. The reviewer marked it non-blocking for this round.
Round-2 scope is 302 code/test lines (64 source +238 tests), approximately the
300-line target: the extra two source lines are the optional early type gate.
Run both-dtype focused tests, local process-only mutants, static checks and a
fresh final-wheel A10 gate. Commit this fix-up, request independent round 2/3,
then follow its verdict and stop. No extra reviewer prompt/context or edits.
Round-2 validation: 19 focused cases pass per mode (12.89s/13.00s); all three
missing baseline-substitution mutants are caught in BOTH modes. Fresh-wheel
A10 exits 0: own 1261/8 and 1266/3; doctests 4 each; independent 2332/28 and
2322/38 (passed/skipped). Ruff/mypy/whitespace clean; records include the final
wheel hash and timings. D-032 remains an open value-policy question; no
numerical source change answers it. Commit fix-up and invoke round 2/3.

---

# PH / RMST merge conflict resolution (2026-10-31)

The maintainer started merging reviewed RMST work into release-ph and asked
for conflict resolution. Parents: PH `c2b9c80c63435b3e36c9ef41e57ca1e6438fd2c9` and
RMST `aae68920769855aa08227f4b7c4818ce7a820923`. This task resolves and stages the four
record/index conflicts; it does not create new numerical behavior or finish
the merge commit.

Resolution plan:
- Preserve both complete parent PLAN.md contents verbatim below, with branch
  labels. No unfinished history is archived or declared complete.
- Preserve both complete parent REVIEW.md records beneath the merge-resolution
  record. PH.2c's independent round-1 verdict remains maintainer: F-1/F-2/S-1
  require a decision about finite extreme-tail derivative guarantees. Importing
  RMST does not resolve or bypass that question.
- DEVIATIONS.md: keep common entries once, retain PH's D-028, import RMST's
  D-029, and renumber RMST's independently assigned D-028 to D-030 with an
  explicit historical alias. Preserve all entry bodies, decisions and statuses.
- appendices/README.md: retain both appendix/PDF rows and both build commands.
- All other staged blobs, including RMST numerical code/tests/tolerances and
  .spec-seen, stay byte-identical to the pre-resolution index. No dependency,
  protected-path, API/export, spec or numerical change is made by resolution.

Verify both parent plan/review records, all deviation bodies, all unaffected
index blobs and appendix targets; check whitespace/conflict markers, Ruff,
mypy and combined PH/RMST tests in separate dtype processes. Keep this
maintainer-started merge staged for completion; no merge commit/push/publish.
PH closure/delegated RMST and the open PH.2c decision remain separate work.

Completed verification (this resolved tree):
- Both full parent plans and reviews are exact substrings of the resolved
  records. Every deviation entry body/decision/status is preserved; IDs are
  unique after the explicitly recorded RMST D-028 -> D-030 reconciliation.
- Both appendix rows, original build commands and linked sources/PDFs exist.
  Their mathematical source/PDF files were not edited or rebuilt.
- Every unaffected index blob matches the saved pre-resolution index; no
  unresolved conflict markers remain in the four files. Whitespace is clean.
- uv run ruff format --check .: 132 files already formatted.
- uv run ruff check .: all checks passed.
- uv run mypy src: no issues found in 22 source files.
- JAX_ENABLE_X64=1 uv run --group test pytest -q -n 6
  tests/unit/test_proportional_hazards*.py tests/unit/test_rmst.py
  tests/unit/test_weibull_rmst*.py: 177 passed, 1 skipped (59.47s).
- Separate JAX_ENABLE_X64=0 process with the same selection:
  176 passed, 2 skipped (64.38s).
The full wheel/acceptance gate and PDF compilation were not repeated for this
record/index-only resolution; the focused run checks combined PH/RMST
compatibility. These passes do not settle PH.2c's open derivative question.

---

# Preserved release-ph plan (verbatim)

# PLAN.md: current PH work and preserved integration records

The maintainer-started merge brings main 14155ed into release-ph 68bd461.
Both parents' planning records are retained below. No unfinished sub-slice
is archived or declared complete by this resolution. New PH work goes first.

---

# PH.2c — High-precision tails and direct time-value policy (2026-10-31)

Authorized by “ok on to next part”. Clean release-ph base:
796085ec9c000996d467609b94090235a2a4548f (maintainer's agreement merge).
Original launch base remains 8757c0fda78e34e6e818249adedd2a144a33c33d below.
Same unfinished PH sub-slice; retain its history. PH.2b round 1 was merge,
with two optional nits. Its reviewer independently checked eager second-order
isolation (92 comparisons per mode) and caught five mutants in both modes.
The current spec is revision 39; the revision-39 RMST tolerance clarification
adds no PH direct-method work. No core RMST implementation/handoff is in this
checkout, so closure remains deferred and the RMST-owned files stay untouched.

One concept: numerical PH evidence outside the ordinary interior. Target
<=300 changed code/test/other implementation lines; PLAN/REVIEW are excluded
under the updated working agreement. Create
`tests/unit/test_proportional_hazards_tails.py` (no public API change):

- Use mpmath at 100 decimal digits, with actual dtype-rounded input values,
  for independent Exponential and Weibull PH formulas. Compare all seven
  direct methods in lower-tail recovery, predictor-exponential overflow,
  predictor-exponential underflow with a finite scaled hazard, and a late tail
  with representable tiny survival/log CDF. Eager/jit/vmap agreement, no
  matching-NaN acceptance and zero absolute tolerance for nonzero tails.
- Compare eta first and all four second-order AD compositions of log survival
  against -H, eager and jit/vmap, on those finite smooth cases. Check natural
  baseline first derivatives on the lower-tail recovery cases. Do not assert
  unrepresentable derivatives or arbitrary extreme natural-parameter Hessians.
- Verify direct/grid values at negative/infinite/zero/NaN times using spec 4.5
  and hand boundary limits, plus finite policy values' zero parameter gradients.
  The additional mixed exact/right Weibull debug_nans check is deferred
  below to keep the formatted implementation diff under the budget.
- Use existing central/tail tolerances without changes; explain scales and
  references. If a numerical failure calls for production changes, keep those
  in wrappers.py within the same budget or split before exceeding it. Never
  edit base.py/exponential.py/weibull.py or RMST tolerances.

Spec: 3.3, 4.4–4.5, 5, 6.1, 7.1–7.3, 12.1–12.2. No new dependencies,
exports, docs-site edits, protected paths, AFT, summaries or sampling. Docs
to-do remains the later PH topic/export integration; appendix stays unchanged.
Clarify PH.2b N-1 in this record: its committed derivative loop used vmap and
jit(vmap); the independent reviewer separately verified unmapped eager AD.

Split after the initial formatted draft reached 315 test lines: defer the
extra 28-line Weibull mixed-batch debug_nans test. The current test module
is 287 lines before any numerical fixes. If a fix needs more room, split
the entire time-policy group into the next step rather than exceed budget.

Overwrite REVIEW.md with reading order, oracle derivations, safeguards,
requirement/test mapping, actual commands/results and remaining limits. Run
focused tests in separate x64/float32 processes, Ruff/mypy, then unchanged
fresh-wheel through-A10 gate. Commit and invoke independent-review exactly
for PH.2c round 1/3, address blocker/should-fix findings within three rounds,
then stop/report. No merge, push or publishing; no next step in this launch.


PH.2c execution: final 288 test lines, 34 cases, no production changes. The
first draft's eta test rebuilt the parameter vector and exposed a NaN in
forward-over-forward at an underflowing baseline. The final test varies eta
as a distinct constructor argument; all six AD orders pass there. REVIEW.md
records the reproduction and remaining rebuilt-vector limitation explicitly.
No tolerance was changed. Final focused tests: 34 passed x64 (16.32s),
34 passed float32 (17.13s). Ruff format/check and mypy clean; fresh-wheel
through-A10 gate exits 0: own 1210/8 and 1215/3, doctests 4 each, independent
acceptance 2332/28 and 2322/38 (passed/skipped). Full commands, wheel hash,
timings and limits are in REVIEW.md. Commit, then PH.2c independent round
1/3 and stop. No unfinished sub-slice was archived or declared complete.

---

# PH.2b — Declared-invalid row isolation and custom-baseline composition (2026-10-30)

Authorized by “ok on to next bit”. Clean release-ph base 813ee36d5ec51649386b633f0c131d1e39b220b7.
Same PH sub-slice; preserve the earlier PH and integration records. PH.2a
round 1 was merge, with its optional scale-comment nit fixed without changing
its test AST. This step targets <=300 changed test lines, one concept: PH's
public likelihood composition preserves valid rows when declared parameters
are invalid, including the custom-baseline path and runtime issue-mask changes.
Spec 4.2/4.4/4.7, 5, 6.1, 7.1–7.3 and 12.1–12.2.

Create `tests/unit/test_proportional_hazards_isolation.py`:

- Exponential rate and Weibull shape/scale invalid at 0, -1, NaN and ±inf;
  PH eta invalid at NaN and ±inf over both built-ins (zero/negative eta
  remain valid). Include shared healthy baseline parameters or shared eta,
  so reverse-mode poisoning cannot hide behind independent parameter rows.
- A tiny registered custom baseline has rate sqrt(upper-lower), positive
  declared fields and a named nonpositive-gap condition. Zero/negative gaps
  violate only that condition. Default density/hazard/CDF methods exercise
  extension composition. Its valid PH law is Exponential with the scaled rate.
- Check eager/jit/vmap likelihood values, exact prefixed field/condition masks,
  union polarity and NaN only at the bad row. Expected valid contributions
  come from explicit Exponential/Weibull PH formulas with invalid operands
  removed before every calculation, not from another likelihood call.
- Differentiate only valid contribution outputs. For every violation class,
  compare first derivatives and all four forward/reverse second-order
  combinations to the isolated formula, eager and under jit/vmap. Do not
  assert the invalid contribution's own derivatives or upstream exp overflow.
  Use unchanged central tolerances, justified by O(1–10) AD/analytic entries
  and rounding-only error; no finite differences across invalid coordinates.
- On valid custom-baseline parameters, compare all seven grid methods and
  their first derivatives with the analytical Exponential equivalent, eagerly,
  jit and vmap, at positive interior times. No unsupported generic hazard
  boundary derivative is asserted.
- Trace counters in compiled raw construction/diagnostics/likelihood calls
  change numeric parameters, valid/invalid masks, times and events while
  retaining shapes/dtypes/weak types/structure. Require a single trace.

Overwrite REVIEW.md with reading order, invariants, formula origins and
actual commands/limits. Run focused PH tests in separate dtype processes,
Ruff/mypy and the unchanged fresh-wheel through-A10 gate; audit whitespace
and scope, commit and invoke independent-review exactly (PH.2b, round 1/3).
Address findings within three rounds, then stop/report. No exports, docs,
production expressions, dependencies, protected paths, tolerance changes,
RMST-owned files or closure work. Split if new test text exceeds the budget.
Docs to-do: existing PH topic/export integration remains later; no new user
API or docs page is required by this test-only step. Extended-tail/range
coverage and RMST integration remain separate; do not claim the PH gate done.


PH.2b execution: added 259 test lines / 55 cases, no production changes.
Final float32 focused run: 55 passed (274.00s). Initial x64 version: 54 passed,
1 failed due to bit-equality of a compiled/eager value differing by 1.1e-16;
fixed to analytical-oracle comparison with unchanged central tolerances and
separate NaN check. Corrected x64 value/diagnostics family: 6 passed (19.76s).
Final full fresh-wheel A10 gate also reran all final tests in both modes:
own 1159/8 x64 and 1164/3 float32; doctests 4 each; independent acceptance
2332/28 x64 and 2322/38 float32 (passed/skipped). Gate exit 0. Ruff format,
Ruff check, mypy and whitespace clean. Full commands, wheel identity, timings
and limits are in REVIEW.md. Commit then PH.2b independent review round 1/3;
stop after its verdict. No unfinished history archived.

---

# PH.2a — Built-in PH identities and interior derivatives (2026-10-30)

Explicit continuation authorized by “after this: go and implement the next
bit”. Start clean release-ph at 88360950ecdbe7ecca2c5aff38162bdafe0a35c2.
PH.appendix round 2 returns merge: F-1/N-1 fixed, no new findings, all three
overflow mutants caught in both modes. Its reviewer ran clean own suites
1071/8 and 1076/3 (passed/skipped), with the known unsupplied-wheel setup
limit on its acceptance gate. The successful implementer fresh-wheel gate
and all appendix records remain above/below and in Git; preserve PH history.

One test-only concept, target <=300 changed test lines. No API/export,
production expression, dependency, protected-path, RMST-owned file or tolerance
changes. Spec 4.4–4.5, 5, 6.1, 7.3 and 12.1–12.2. Keep broader PH.2 evidence
and the closure/RMST handoff separate; this does not close the PH release gate.

Create `tests/unit/test_proportional_hazards_identities.py`:

- Express the analytical equivalences r_PH=r*exp(eta) and
  b_PH=b*exp(-eta/k), with k unchanged. Compare every available direct method,
  grid twin and exact/right likelihood; summaries are unavailable here.
- Scalar baseline/predictor inputs: eager/jit/vmap values for the seven direct
  methods at positive interior times; compare first derivatives in eta and
  each natural baseline parameter using jit of forward/reverse Jacobians.
- Baseline (3,1) × predictor (3,2): paired direct values and shared grid values
  (K,3,2), including K=N, zero/infinite times and Weibull k<1, k=1 and k>1
  limits. Interior grid Jacobians compare each original baseline/predictor
  field; no derivatives at density/hazard discontinuities are asserted.
- Exact/right likelihood: prepared/raw forms and first parameter Jacobians
  against the equivalent family with the same broadcast inputs. Add direct
  analytic Hessians for scalar mixed likelihood means in both families,
  plus order-2 forward/reverse check_grads on smooth interior inputs.
  Expected Hessians come from differentiating the PH likelihood formula,
  never from the implementation. Keep existing central tolerances unchanged.
- Tests name the equivalence/broadcast/derivative behavior and state the spec.
  No helper layers outside this dedicated PH test module and no redundant
  tests for implementation details. Split evidence if it would exceed budget.

Overwrite REVIEW.md for PH.2a with reading order, math/invariants, coverage,
commands and limits. Run focused PH suites in separate dtype processes,
Ruff/mypy, fresh-wheel through-A10 gate, then whitespace/scope audit. Commit,
invoke independent-review exactly for PH.2a round 1/3, address findings within
three rounds, then stop/report this next implementation step. No PH.2b start.
Implemented: 260 new test lines / 33 cases. Initial 32 new cases passed in
both modes; a subsequent mechanical edit caused a syntax/collection error,
corrected before final checks. Final combined PH suites: 56 passed/1 skipped
float64 (28.00s), 55 passed/2 skipped float32 (28.61s). Ruff/mypy clean.
Fresh-wheel through-A10 exits 0: own suites 1104/8 and 1109/3; doctests 4 each;
independent acceptance 2332/28 and 2322/38 (passed/skipped). Only the new test
module and plan/review records change; production code and tolerances remain
unchanged. REVIEW.md records commands, timings, wheel identity and limitations.
Commit and independent round 1/3 follow; stop after this next step's review.

Independent review completed on 4194ac0: round 1 verdict merge; no blockers
or should-fix findings. Hand Hessians independently match mpmath to <5e-41;
all five targeted mutants are caught in both dtypes, including a zero-time
Weibull hazard mutant missed by the earlier PH tests. Optional N-1 is
addressed by one O(1)/rounding-scale comment, with identical test AST and no
assertion/tolerance changes. Record the result and stop after PH.2a; no PH.2b.

Docs to-do: the existing executable PH topic/export integration remains later;
this test-only step introduces no new user-facing documentation requirement.

---

# PH.appendix — Explain the scaling helper and retain its mathematical appendix (2026-10-30)

The maintainer approves PH.1 apart from naming/shape-documentation nits and
requests a permanent appendix outside the normal docs, matching release-rmst,
then explicitly authorizes the next PH implementation step after this one.

One bounded clarity/appendix step, about 300 changed text lines. No numerical
expression, public API, parameter handling or export change. Spec 3.3,
4.4–4.5, 5 and 7.2–7.3; the new RMST-only revision-38 changes are read but do
not change these PH contracts. Retain all unfinished/incoming plan history.

Round-1 follow-up: PH.appendix review confirms the expressions are unchanged,
all mathematical claims/examples/shapes checked are correct, and stable code
anchors resolve. It returns fix-first for F-1: the finite-time overflow restore
has no direct regression test. Add above/below-log(max) finite-time cases to
the PH guard module; no numerical source change. N-1 clarifies in the helper
docstring that only log_s==0 CDF entries are read; others may be reference-time
values. This is under 30 additional source/test lines and keeps the complete
clarity/appendix step near 300 text lines. Run focused tests in both dtypes,
Ruff/mypy and whitespace checks, commit separately and request round 2/3.
The prior full fresh-wheel gate is retained as initial-commit evidence; this
small fix-up adds only assertions and docstring wording. Focused PH suites
pass: 23/1 float64 and 22/2 float32 (passed/skipped), 11.37s each; Ruff/mypy
and whitespace clean. Commit the fix-up and request PH.appendix round 2/3.

- `src/survjax/distributions/wrappers.py`: rename the ambiguous baseline and
  scaled log cumulative hazard locals; name the optional argument as baseline
  log CDF. Add the private helper's Args/Returns docstring with evaluated shape
  E, field batch shape B, scalar-time/infinity B, grid (K,*B), promotion and
  the None infinity-hook path. Add stable appendix references at the branches.
- `appendices/ph-log-survival.md` plus adjacent PDF: permanent mathematical
  appendix derived from the temporary reading note. Keep definitions, input
  shapes, guard meanings, real-arithmetic tail/derivative bounds and explicit
  information-loss limits; remove snapshot/review/check history. Use code-
  aligned names. No docs/ page or navigation change. The old output note is
  a temporary artifact, not another authoritative source.
- `appendices/README.md`: follow the RMST worktree's common table convention,
  listing this appendix, code and PDF with a reproducible Pandoc command.
  This branch lists PH only; integration must retain the RMST row too.
- If needed, retain the tiny pagination Lua filter alongside the appendix;
  no package dependency or shared build-tool changes. Keep combined text diff
  near 300 lines; split before implementing any additional concept.
- Record this step in REVIEW.md after the existing integration is committed.
  Focused PH tests run separately in both dtypes; Ruff/mypy, Pandoc compilation
  (normal and formatted), all-page Poppler inspection and whitespace audit.
  Existing tests already cover the unchanged scalar/grid/infinity shapes;
  no redundant tests for local renames. Commit and independent-review this
  new clarity step, not a fourth PH.1 numerical-review round.

The main merge was already in progress before this request. Both conflicted
record files are now resolved by preserving both parents verbatim; every
other staged incoming entry is unchanged. The merge-only index is prepared.
The maintainer explicitly authorized finishing this prepared merge (“jok
agreed”). Commit 245219d records the staged integration only; PH changes
remain separate. The older temporary output notes are backed up under
/private/tmp after migration to the canonical appendices/ source and PDF.

Implementation verified: private naming/docstrings/comments plus the permanent
appendix total 249 changed text lines (excluding plan/review records and the
PDF binary). Full-wrapper AST equivalence, focused PH tests in both modes,
Ruff/mypy, both Pandoc builds and all-page PDF inspection pass. Fresh-wheel
through-A10 exits 0: own suites 1069/8 and 1074/3, doctests 4 each, independent
2332/28 and 2322/38 (passed/skipped). Exact commands and times in REVIEW.md.
Commit PH.appendix and request independent round 1; then continue to PH.2
under the maintainer's explicit “after this” instruction, subject to findings.

Next authorized implementation: PH.2, split into <=300-line test portions.
Start with the built-in direct/grid/likelihood PH identities in values and
first derivatives, including scalar and (S,1) baseline × (S,2) eta batches,
plus representative second-order interior evidence. A concrete file/test
plan will precede that code after this appendix step's review; additional
invalid-row isolation/tail coverage and closure/RMST remain separate portions.

---

# Release-ph planning record (68bd461)

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Other sub-slices are archived verbatim in `plans/` (index: `plans/README.md`).
Still open there: A9 (`plans/A9.md`, "A9 Docs to-do" and "Final evidence
audit and planning checks") and A8 (`plans/A8.md`, "Closure still required").

---

# PH.1 — Construction, diagnostics and direct PH composition (2026-10-29)

Authorized by the release-ph workmux launch. Clean branch `release-ph` starts
at full base SHA `8757c0fda78e34e6e818249adedd2a144a33c33d` (8757c0f).
Keep the unfinished A9/A10 history below; no archive or integration commit.
Spec: 3.3, 4.1–4.2, 4.4–4.7, 5–7, 8.7, 10.2, 12.1–12.2.

## PH.1 round-2 follow-up (2026-10-29)

Round 2 on 13b8b58 returns fix-first. S-1 is settled by the maintainer's
positional-baseline decision; F-1/F-2/N-1/N-2 are independently confirmed fixed.
One small final fix-up, <=150 changed source/test lines; next review is 3/3.

- F-3 / `wrappers.py`: normalize eta in the PH adapter with the existing dtype
  helper and a caller-level stack depth before forwarding the converted array
  to registration. Registration then receives its ordinary normalized dtype,
  so it emits no duplicate warning. Do not catch/re-emit warnings or change
  global filters, shared registration, or RMST-owned files.
- Add `tests/unit/test_proportional_hazards_guards.py`: a float32-only warning
  test exercises positional/keyword forms at two distinct user lines, repeats
  each line, and requires exactly two warnings at those lines. A custom family
  with only log survival/reference_time checks the baseline tangent when the
  generic log-CDF also underflows (F-4), in both dtypes, forward and reverse.
- Extend the existing cure-baseline tests with p=1 and eta=1000: survival at
  infinity is 1 and its eta derivative is zero. This pins F-4's linear-branch
  overflow bound without asserting an unrepresentable baseline-p derivative.
- A follow-up probe found that the linear branch still exponentiates eta in
  its own dtype. Promote eta with baseline log survival before scaling, so a
  float64 baseline retains a representable exp(90) tangent with float32 eta.
  Add an x64-only mixed-dtype zero-limit regression. This belongs to the same
  F-4 guard correction and keeps the fix-up below 150 changed source/test lines.
- N-3: add a short case map next to the scaling masks. Retain lower-tail
  recovery because the first-derivative/value tests require its precision;
  make no performance claim or new optimization in this correctness fix-up.
- Update REVIEW.md by finding ID; run focused PH tests in both processes,
  Ruff/mypy and a fresh-wheel through-A10 gate. Commit, invoke the exact
  independent-review command for round 3/3, then stop and report its verdict.
  D-028 remains formally open for the maintainer's spec disposition; no spec
  edits, further public API, exports, new dependencies or protected-path changes.

Final fix-up verified 2026-10-30: 122 changed source/test lines (120 added,
2 removed), below the 150-line budget. Final focused PH suites: 21 passed,
1 skipped / 20 passed, 2 skipped. Fresh-wheel through-A10 exits 0: own suites
1069 passed, 8 skipped / 1074 passed, 3 skipped; doctests 4 passed per mode;
acceptance 2332 passed, 28 skipped / 2322 passed, 38 skipped. Ruff/mypy and
scope/whitespace clean. REVIEW.md distinguishes the earlier intermediate gate
from the successful final-source rerun and records commands, times and hashes.
Commit this fix-up and invoke round 3/3, then stop after the verdict.

Round 3 completed on a4ca299: no open code findings, all five relevant mutants
caught, and the reviewer states the code is ready to merge. Final verdict is
maintainer solely for the formal D-028 spec exception. The constructor choice
is already implemented; no repeat decision is needed. Spec/protected paths
remain untouched, and D-028 remains formally open for maintainer disposition.
Record the outcome without code changes, then stop. No fourth review or PH.2.

## PH.1 round-1 follow-up (2026-10-29)

Maintainer decision: “allow for positional baseline in PH wrapper”. Resume the
same step from clean b0b0f8b on release-ph; next review is round 2/3. Record the
decision under D-028 without marking that entry resolved or editing the spec.
This fix-up targets <=300 changed source/test lines relative to b0b0f8b; the
initial 292-line implementation remains its own committed review unit.

- `wrappers.py`: adapt only PH's registered initializer to accept baseline
  positionally or by keyword, with log_hazard_ratio still keyword-only. Keep
  base.py and other RMST-owned files untouched. Update the constructor docstring.
- F-1: for zero baseline log survival at positive times, recover log H0 from
  the baseline's log-CDF lower tail (log F0 = log H0 to rounding precision in
  this underflow regime), then apply the PH formula in log space. Preserve
  exp(eta) times the baseline tangent on true zero-limit hook values using a
  safe linear branch. Do not add a new family hook or dispatch on family type.
- F-2: add a small registered improper baseline in PH-specific tests; test
  infinity values and derivatives, including a zero log-limit, and NaN eta's
  hook/direct outputs. The tests accompany the already implemented hook.
- N-1: return the baseline reference in the promoted baseline/predictor dtype;
  test float64 baseline with float32 predictor. N-2: extend boundary tests to
  a predictor just above log(dtype.max), pinning the overflow guard's band.
- Constructor tests assert positional/keyword equivalence under tracing and
  reject a second positional argument, duplicate baseline and missing fields.
  F-1 regressions use hand-derived PH values and baseline/eta first derivatives
  in both dtypes, eagerly and under jit, forward and reverse modes. Existing
  tolerances remain unchanged. No weakened export-reference tests.
- Update REVIEW.md by finding ID. Run focused tests in separate dtype processes,
  Ruff/mypy, fresh-wheel through-A10 regression and whitespace/scope audit;
  commit, invoke the independent-review skill exactly for round 2, then follow
  its verdict with at most three total rounds. No next PH step or export work.

Follow-up implemented: 191 changed source/test lines, within the fix-up budget.
Focused PH: 18 passed float64 / 17 passed, 1 x64-only skip float32; Ruff/mypy
clean. Fresh-wheel through-A10 exits 0: own suites 1066 passed, 7 skipped /
1071 passed, 2 skipped; doctests 4 passed per mode; independent acceptance
2332 passed, 28 skipped / 2322 passed, 38 skipped. Exact commands, timings,
wheel hash and finding mapping are in REVIEW.md. Commit and round 2 follow.

## Initial plan and round-1 evidence (retained history)

One concept: the registered frozen PH distribution, used through its defining
module until a separate export step. Target <=300 source/test changed lines.

- Create `src/survjax/distributions/wrappers.py`: `ProportionalHazards` with
  required keyword-only `baseline` nested distribution and real-constrained
  `log_hazard_ratio`. Reuse registration for dtype/broadcast/indexing/diagnostics.
  Implement PH log survival, log density and log hazard; inherit the remaining
  direct methods, seven grids and exact/right likelihood evaluator. Delegate
  reference time to the baseline, converted to the predictor dtype with a
  finite-positive fallback used only for inactive evaluations. Scale finite
  baseline log survival in log space; guard zero and infinity before arithmetic.
  Scale the infinity hook to retain improper-baseline limits/derivatives.
- Create `tests/unit/test_proportional_hazards.py`: frozen/pytree construction,
  nested dtype preservation, `(S,1)` × `(S,2)` broadcasting, indexing, traced
  construction and prefixed diagnostics; static misuse; all seven direct/grid
  methods vs analytical exponential expectations; exact/right raw and prepared
  likelihood; eager/jit/vmap agreement, interior forward/reverse order-2 checks;
  zero/infinity policy and extreme finite predictor.
  Reuse existing central ordinary/derivative/tail tolerances, unchanged.
- Overwrite `REVIEW.md` with reading order, invariants, tests/spec mapping,
  commands/results and limits. Commit this step, invoke independent-review
  exactly for PH.1, follow at most three rounds, then stop/report.

Public exports remain unchanged: no docs/code exception assumed and no export
assertions weakened. No RMST-owned files, dependencies, protected paths, AFT,
summaries or sampling. Later PH steps own the full Weibull/Exponential identity,
transformation, improper custom-baseline, invalid-row and tail matrix;
closure/RMST waits for reviewed RMST integration. Split additional evidence before exceeding this step's budget.

Checks: focused PH tests in separate x64/float32 processes, Ruff format/lint,
mypy and fresh-wheel unchanged cumulative `scripts/check_wheel.sh --through A10`.
If a check fails, record the actual failure; no unsupported gate-pass claim.
Docs to-do: separately authorize public export/API-reference integration; later
executable PH topic documents eta>0 as higher hazard and the release subset.

Initial implementation committed: 118 source + 174 test lines (292 total). Focused PH:
12 passed per mode (10.05s float64 / 9.99s float32); Ruff formatting/lint and
mypy clean (21 source files). Initial focused boundary-gradient failure was
fixed with guarded exponential overflow, leaving the test unchanged.
Fresh-wheel through-A10 gate exits 0: own suites 1060 passed, 7 skipped /
1066 passed, 1 skipped; source doctests 4 passed per mode; independent suites
2332 passed, 28 skipped / 2322 passed, 38 skipped. Exact commands/timings,
wheel identity, limits and requirement map are in REVIEW.md. Scope and staged
whitespace audit clean. Commit 965cbeb is reviewed: round 1 returns maintainer
for S-1 (constructor contract; logged as open D-028). F-1 underflowed baseline
tangents and F-2 missing improper/NaN-hook evidence remain should-fix; N-1/N-2
remain nits. REVIEW.md records the independent probes and gate limits. Stop
for the maintainer decision, with no code fix-up or next step begun.

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


---

# Incoming main planning record (14155ed)

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; all incoming A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.d1 — Using and verifying the installed Slice A subset (2026-10-29)

Authorized by "the current work is approved; if more work needed, get to it".
Work on the existing a9b branch from clean bd32ff9. The maintainer completed
the main merge, manually reran `scripts/check_wheel.sh --through A9` without
errors, and cleaned the reviewer-owned throwaway worktree. Record those as
maintainer-reported checks/cleanup, not as runs performed by this agent.
A9.4 is accepted; optional review nits N-1/N-2 do not block this docs step.

One topic: explain the installed subset's verification and limitations.
Spec: 6.1–6.2, 7.2–7.3, 10.3, 10.5, 12.1–12.3. Public API unchanged.
At most 300 changed docs lines. No dependency, source, test, fixture,
benchmark, protected-path or spec edits. PLAN.md keeps only current A9.

Files to create/change:

- Create docs/slice-a-verification.md as executable MyST. Give a compact
  synthetic exact/right example with construction and diagnostic access inside
  jit, value/parameter gradient/Hessian checked against hand algebra, eager
  and prepared/raw agreement, and vmap over rates. Document the smooth
  interior parameter-only guarantee and explicit host checkpoints.
- Explain the current subset and unsupported likelihood modes; link existing
  construction, likelihood, numerical, distribution/grid and concordance
  guides. Describe the fresh-wheel runner, installed out-of-tree custom-family
  checks, pinned synthetic R evidence and the exact CPU reproduction commands.
  Explain first-call versus warmed costs, prepared/traced flags, returned
  diagnostics, logical bytes versus unmeasured peak memory, and optional
  benchmark nits without silently changing accepted measurements.
- Add the new page to docs/index.md's prose and toctree. Audit the remaining
  executable pages and playground. Correct only stale capability statements
  within this verification topic; leave unrelated teaching work for separate
  steps. Document the locked JupyterLab command.
- The rendered audit found docs/survival-data.md's opening note still claims
  likelihoods, distributions and evaluation are unimplemented. Correct that
  one scope note and link current likelihood/concordance guides. This narrow
  factual correction belongs to the installed-subset verification topic;
  it adds no new tutorial or runtime behavior. Total docs remain below 300
  changed lines. Record the discovery before making the edit.
- Overwrite REVIEW.md with reading order, invariants, evidence mapping and
  remaining gate status. Record every Slice A gate item with concrete tests
  or artifacts and honest pass/pending status; this is not an A10/release
  completion claim. The agreed README/use-case example replacement is still
  separate from this small compiled-verification recipe.

Validation ownership: implementer owns fresh `scripts/build_docs.sh` in both
modes, Ruff and rendered-page/browser inspection. The maintainer's single-
runner preference means the independent reviewer should inspect source and
recorded outputs instead of repeating the same build/gate. No full wheel-gate
repeat for docs-only edits: the maintainer just reran it successfully on the
merged baseline, and no runtime/packaging change follows. Inspect links,
outputs, formulas, navigation and tables across the site in both builds;
inspect the new page carefully and record any existing warning/limitation.
Verify committed notebooks still have no outputs/counts. No new unit tests:
the executable page's assertions check its example, and existing suites
already cover the numerical and packaging behavior.

After successful build/source/visual checks, record exact commands/results,
commit this docs step on a9b with "Independent review: round 1/3" in REVIEW.md,
invoke the exact independent-review skill command for step A9.d1 in
../review/code-a9b, address blocker/should-fix findings in separate fix-ups
within three rounds, then stop and report. Do not start another step.

Docs/closure to-do: agreed replacement README/use-case teaching material and
any gaps from this rendered-site/evidence audit remain explicit; don't claim
Slice A or A10 complete until their separate gate items are accepted.

## A9.d1 evidence and audit conclusion

Implemented the verification page, index navigation and one stale data-guide
scope note; no code or notebook changes. Both final docs builds pass, nine
executed documents and 137 code cells per mode. The three new example cells
check exact analytical values, gradients/Hessians, input forms and jit/vmap.
All 1,048 local file/fragment links per rendered site pass. The committed
playground remains output-free. Ruff passes for the repository and the
extracted three-cell notebook; JupyterLab CLI reports 4.6.4. Native Brave
inspection covers all 11 content pages in float64, all pages' accessible
content in float32 and detailed views of the new page, corrected note,
playground and representative grids/formulas/tables. REVIEW.md records final
execution times, initial sandbox/interruption errors, checks and limits.

The final evidence map is included in this step's REVIEW.md rather than
requiring a separate metadata-only audit step. The new three-observation
verification recipe provides tiny synthetic smoke evidence; the larger
agreed README/use-case teaching replacement remains separate work and is
not claimed complete. The maintainer's post-merge fresh-wheel pass supplies
current regression evidence; no runtime/packaging changes follow it.
A9's verification evidence is now ready for independent review and maintainer
acceptance. No claim that A10, Slice A as a whole or the release is complete.
Stop after this docs step's review; no additional implementation starts.

## A9.d1 round-1 follow-up

Round 1 on 2c8cd60 returned fix-first for should-fix F-1: cite the explicit
unsupported-mode NaN/preserved-neighbor requirement and executed triage
recipe in the gate map, and map each narrative item in spec 12.2's paragraph
to its docs page. Existing code/tests already cover this behavior; no new
implementation or test is required. Also fix trivial N-1 (R paragraph line
wrap) and N-2 (clarify that tests/doctests, rather than static checks, run
in separate dtype processes). Code cells remain byte-identical.

The reviewer reran the build because its role requires it, despite the
initial ownership request. Its build passed in both modes (new page 3.12s /
3.07s); Ruff passed. No wheel gate or mutations were rerun. For this prose/
metadata fix-up, the reviewer alone owns round-2 builds/lint. The implementer
checks only the diff and unchanged cells, records F-1/N-1/N-2, commits a
separate fix-up and requests round 2. No duplicate build/full-gate run by
implementer; maximum three rounds, then stop.

---

# A9.4/main merge resolution (2026-10-29)

Authorized by "merged main into the branch; fix conflicts". The maintainer's
merge is in progress on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflict; resolve these metadata files. No public
API, code, docs, dependency, spec or test behavior changes are needed.

- Preserve A9.4's complete plan and evidence below. Independent review round
  1 on 51319c8 returned merge, with optional N-1/N-2; those nits remain open.
- Preserve all main's A10 plan sections verbatim in plans/A10.md, including
  A10.d2, A10.d1 and A10.4. Its older A10 history remains the unchanged suffix
  of that archive. PLAN.md keeps only the current A9 sub-slice. A10's incoming
  review record is preserved in REVIEW.md alongside the A9.4 record. These
  are historical records; this merge starts no new implementation step.
  This corrects the first resolution, which incorrectly kept incoming A10
  sections in PLAN.md; the maintainer explicitly checked the current-slice
  format before completing the merge.
- Preserve every original stage-0 index entry except the intentional
  plans/A10.md archive extension and plans/README.md index description.
  Incoming docs, concordance fixtures/tests, release-agent plans and
  .workmux.yaml remain untouched. The last is an incoming protected-path
  change, not an agent edit.
- Verify both parents' records are retained, no conflict markers/unmerged
  entries remain, whitespace is clean and every other staged object is
  unchanged from the captured merge index, apart from the two archive/index
  changes above. No runtime/build reruns: neither
  conflicted file affects execution, and no manual executable edit is needed.
  Preserve incoming A10.d2's single-runner validation ownership.
- Stage the two resolutions and archive/index corrections; leave the merge commit to the
  maintainer under the working agreement. Do not create a merge commit,
  start the next implementation step or rerun independent review for this
  metadata-only resolution.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.

---

# Incoming main simulation planning records

Preserved from the incoming main parent during this merge. The shared older
plan history is retained above; PH remains the active branch work.

# SIM.3 — Complete the approved examples diagnostics environment (2026-10-30)

Start clean on release-simulation at
`ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d`; SIM.2 round 1 returned merge.
Preserve the unfinished SIM/A10 plan history below. The maintainer approved
ArviZ 0.23.0 in the existing dependency proposal and now requests the next step.
One concept: finish the examples dependency group for Bayesian tail ESS.

Files and scope:
- `pyproject.toml` / `uv.lock`: add exact `arviz==0.23.0` to examples;
  resolve without upgrading existing packages or changing runtime dependencies.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark the six-package setup
  complete, record the actual installed diagnostics probe and its command.
- Overwrite `REVIEW.md`: reading order, dependency isolation, actual evidence
  and limits; preserve the short temporary probe source for reproducibility.

Spec 10.3; shared use-case 00 dependency-group rule and Bayesian details
Sampling/Diagnostics: NumPyro split R-hat/ESS, ArviZ tail ESS. No source API,
shared generator, ordinary tests, docs/navigation/config or protected scripts.
Docs to-do: the future Bayesian page should use this group for tail ESS.
Target below 300 changed implementation/config/lock lines, records excluded.

Checks: `uv lock --check`, `uv sync --locked --group examples`, compare old/new
lock name/version mappings. Explicit optional-environment probe uses seeded
4x1000 iid Normal draws (known equal-chain law) and deliberately shifted
chains to verify the chain/draw axes, NumPyro summary and ArviZ bulk/tail ESS
and R-hat; this is compatibility evidence, not posterior convergence evidence.
The pinned ArviZ import writes a daily stamp under Path.home()/arviz_data, so
run that probe with required sandbox escalation rather than modifying HOME.
No new ordinary tests for the dependency-only change. Run the existing focused
generator suite in both dtype processes, Ruff, mypy, and unchanged fresh-wheel
through-A10 gate. Record versions/results, write REVIEW.md, commit, invoke the
exact independent-review skill for SIM.3, follow at most 3 rounds and stop.
PH fitting still waits for maintainer integration of reviewed prerequisites.


Implementation: added the approved ArviZ pin and six lock packages. All 165
prior package versions unchanged; runtime dependencies remain jax/numpy.
Actual installed-group CPU probe passes named chain/draw axes, iid gates and
shifted-chain detection: NumPyro R-hat 1.0000270083 / ESS 3976.58; ArviZ
R-hat 1.0001031616 / bulk ESS 3979.33 / tail ESS 3931.29. ArviZ's expected
import warning was emitted; sandbox escalation allowed its home-cache stamp.
Existing generator suites: 17 passed per mode (3.60s/3.18s), Ruff/mypy/lock
checks pass. Fresh-wheel through-A10 gate exited 0: own suites 1065/1071
passed (7/1 skipped), doctests 4 per mode, independent 2333/2324 passed
(27/36 skipped). ArviZ's pandas dependency enabled the existing x32 pandas
conversion test. Wheel hash is identical to SIM.2 and runtime METADATA still
requires only jax/numpy. REVIEW.md records exact commands, outputs and limits.
Commit and exact independent review follow; stop after the verdict.

---

# SIM.2 — Address shared foundation questions and enable examples (2026-10-29)

Start on `release-simulation` at `3556d12c1abca553f77607c1e1739df2bd1911f0`.
The only initial dirty file is study.py with three maintainer question comments.
SIM.1 round 2 returned merge. Preserve all unfinished history below.
Maintainer approves the proposed dependencies in this turn; no renewed
approval is needed for those exact pins. Keep PH fitting deferred.

Files and scope:
- `pyproject.toml` / `uv.lock`: examples group with exact tested Optimistix
  0.1.0, Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0.
  Keep existing runtime/dev/docs versions unchanged. The five-package lock
  preview is 160 changed lines; full six-package setup is 295 lock lines
  alone. Approved ArviZ 0.23.0/tail ESS is a separate later step so this step
  including generator checks remains about 300 lines. No protected scripts.
- `examples/weibull_simulation/study.py`: answer/consume the maintainer notes,
  define quadrature rules before their closure, and explain calibration as
  once-per-target host setup outside the compiled replicate loop. PH input
  preference is asked asynchronously (baseline/log-HR versus PH rate m).
  Preserve the current call contract; any PH convenience will be optional,
  demo-only, and checked against the existing scale formulation.
- `tests/unit/test_simulation_generator.py`: if adding PH input, analytical
  PH/scale and calibration identities, row broadcasts and transformed values/
  derivatives. Reuse existing tolerances. Existing numerical checks also
  exercise the quadrature block reorder.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark approval, staged pins,
  remaining approved ArviZ stage and exact optional-environment invocation.
- Overwrite `REVIEW.md` with reading order, input-contract updates, evidence
  and limits, preserving SIM.1's outcome in this plan and Git history.

Spec 4.4/4.6/7.3/10.3; use-cases/00-common and 01-details DGP/calibration.
No library API, shared docs/config or other workstream files. Docs to-do:
future simulation page should show PH inputs and setup versus compiled work.
Run generator checks in separate x64/float32 processes, approved-environment
compatibility probe, Ruff, mypy, lock consistency and fresh-wheel through-A10
gate. Inspect size before committing; split if needed. Commit, invoke exact
independent-review skill for SIM.2 (max 3 rounds), then stop/report.


Implementation: optional log_hazard_ratio=0.0 added to both generator and
calibration, with unchanged no-keyword calls and U/C key split. Rules now
precede the fraction closure; calibration explicitly remains host-only setup.
No response to the optional PH-form preference arrived during environment
work; used the stated baseline/log-HR convention from 01-details and spec 4.4.
Approved five-package group resolves/installs; all 155 existing locked
versions unchanged, 10 additions. Optional compiled BFGS/NN-update/NUTS/SVG
probe passes in the actual group environment. ArviZ remains approved for its
separate lockfile step. Total implementation/test/config/proposal diff is
303 changed lines, including 160 lockfile lines.

Focused final suites: 17 passed per dtype (10.17s/10.56s), Ruff/mypy/lock
checks pass. A joint three-parameter check_grads initially failed a float32
second-order projection near zero; no tolerance was changed. Retained the
original k/b finite-difference checks and verified all PH gradient/Hessian
entries against inverse-CDF algebra, including cross-partials. The full
fresh-wheel through-A10 gate exited 0: own suites 1064/1071 passed (7/1
skips), doctests 4 per mode, independent 2333/2323 passed (27/37 skips).
NumPyro enables one acceptance check previously skipped. REVIEW.md records
exact commands, times, initial numerical failure and final focused evidence.
Commit and exact independent review follow, then stop/report.

---

# SIM.1 — Shared Weibull generator and censoring calibration (2026-10-29)

Work on `release-simulation` from full base
`8757c0fda78e34e6e818249adedd2a144a33c33d` (clean launch snapshot).
The matching reviewer checkout already exists. Preserve the unfinished A10
history below; its docs and A9/PH/RMST work belong to other implementers.

One concept: reusable demo-only inverse-CDF generation and host-side numerical
calibration, without PH, fitting, core changes or optional dependencies.
Spec revision 37: 3.3, 4.4 (generating PH-equivalent scales only), 7.3,
10.3, 11.4, 12.1–12.3; use-cases/00-common.md and both 01 files.

Files and contract to implement:
- `examples/weibull_simulation/study.py`: `generate_weibull(key, shape, scale,
  censor_limit, *, sample_shape=())`, returning a named tuple with `time`,
  `event`, `event_time`, `censor_time`. Parameters broadcast to row/batch axes;
  sample_shape prefixes those axes. Exactly split the supplied key into U/C
  keys; callers derive scenario/replicate keys with the specified fold/split
  scheme. JAX generation supports jit/vmap and selected process precision.
- Same module: `calibrate_censoring(shape, scale, target) -> float` uses NumPy
  float64 quadrature and bracketed bisection of the mean integrated survival.
  Concrete positive finite broadcast parameters only; calibration is host-only,
  not an RMST implementation or differentiable library API. Check quadrature
  convergence; fail explicitly rather than return an unverified calibration.
- `tests/unit/test_simulation_generator.py`: analytical exponential calibration,
  independent mpmath Weibull quadrature, heterogeneous broadcasting, key mapping,
  transformation agreement and fixed-seed large-sample survival/HR/censoring
  recovery with justified statistical bounds. Centralize any new test bounds
  in `tests/impl_tolerances.py`. Ordinary checks remain offline.
- `examples/weibull_simulation/DEPENDENCIES.md` and a temporary isolated environment
  probe (record the exact invocation in REVIEW.md): concrete exact-version proposal for the later examples group, test
  Optimistix/Matplotlib/Equinox/Optax/NumPyro against locked Python/JAX/NumPy.
  No pyproject/lock edits or added diagnostics/loader packages in this step.
- `REVIEW.md`: reading order, invariants, evidence and limitations; freeze the
  reviewed consumer import contract and calibrated simulation c values.

Target at most about 300 implementation/test/proposal lines (plan/review
records separate); split fitting, scenario runner, results and docs into later
steps. Docs to-do: executable simulation page and recorded full results after
reviewed PH integration; propose fresh docs command with examples selected.

Checks: focused tests in separate JAX_ENABLE_X64=1/0 processes, Ruff formatting
and lint, mypy src, unchanged `scripts/check.sh --through A10` cumulative gate.
Record exact versions, commands and outputs. Commit the step, invoke the exact
independent-review skill command for SIM.1, follow up to three rounds, then stop.
Dependency installation into the project still awaits maintainer approval;
only an isolated proposal probe is authorized here. Never merge/push/publish.


Implementation: 300 added source/test/proposal lines, excluding plan/review
records; no library API, dependencies, lockfile or protected-file edits.
Reviewed consumer contract is recorded in REVIEW.md pending round 1.
Both simulation targets independently match 40-digit mpmath quadrature;
focused final suites: 12 passed in each dtype, including second derivatives.
Ruff formatting/lint and mypy pass. Calibrated c values: 47.793149053990135
and 19.17253543619806 years; pilot empirical censoring .249355/.550485.
Dependency versions proposed and tested in isolation: Optimistix 0.1.0,
Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0 against the
existing locked JAX/jaxlib 0.11.2 and NumPy 2.5.3; group/lock approval pending.
Cumulative fresh-wheel through-A10 gate: exit 0, all gates passed.
Own suites 1059/1066 passed (7/1 existing skips); independent 2332/2322
passed (28/38 existing skips); source doctests 4 per mode. REVIEW.md records
exact timings, commands and the focused derivative-test evidence. Commit and
independent round 1 now follow; stop after the review verdict.

Round 1 on 1e3f5fc returned fix-first for F-1: add explicit quadrature-failure
coverage. Add the k=20/b=5/target=.5 RuntimeError regression, and clarify N-1
underflow and N-2 time units in docstrings; N-3 import-helper suggestion stays
conditional. Fix-up is 7 net source/test lines (307 total), no behavior or
contract change. Focused suites: 13 passed per mode (4.32s/4.05s), Ruff and
mypy clean. Commit separately and request round 2 after round 1 completes.

---

---

# Preserved release-rmst plan (verbatim)

# RMST.integration — Resolve the maintainer-started main merge (2026-10-31)

Branch: release-rmst. Merge already in progress when requested; finish that
merge only, without rebasing, publishing, or merging this branch into main.
Parents: ddd7f1f808eea0c802848bf0489135b547fed67f (RMST) and
4f170654fdc5bb2840f89232e439e33d7e5fc3ff (incoming main).

Conflict-resolution plan, before edits:
- tests/impl_tolerances.py: preserve both independently added blocks verbatim:
  measured RMST factors and SIMULATION_SIGMAS=6. No tolerance value changes.
- PLAN.md: keep this active integration/RMST handoff first, then preserve both
  pre-merge plan versions verbatim in clearly labelled records below. Do not
  discard or archive unfinished history. Main's existing plans/A10.md remains.
- REVIEW.md: write this integration's reading order, preservation invariants,
  checks and limitations. The prior RMST and README reviews remain available
  verbatim at the two parent commits; no extra review/archive module needed.
- All other staged paths came from main. Preserve them exactly, including
  CLAUDE.md and .workmux.yaml; do not edit their protected contents. No new
  numerical code, public API, dependency or spec change in this integration.

Initial static check found an inherited main issue: README.md's fenced
Python example needs two blank lines before its top-level function. Apply
Ruff's formatting-only correction, preserving the example's behavior; this
is the only inherited-file exception, with no docs/ or protected-path edit.

Validate source equality against the RMST parent; imported-file equality
against main; both plan records as exact substrings; both tolerance blocks;
absence of conflict markers/unmerged paths; Ruff/mypy, locked environment,
focused RMST+generator tests in separate dtype processes, and unchanged
fresh-wheel through-A10 gate. Commit the existing merge, invoke the required
independent-review as RMST.integration (round 1/3), then stop/report.

Remaining work owned by this worktree after integration:
1. Core RMST.1/.2/.3 and math appendix are complete and independently reviewed
   (RMST.3 round 3 merge). Spec revisions 38/39 are now on design main; no
   new spec decision is needed. Deviation status remains maintainer-owned.
2. Handoff base.py/exponential.py/weibull.py closure edits to release-ph once
   this RMST branch is integrated into code main. This local main snapshot
   does not contain wrappers.py or ProportionalHazards yet. PH owns wrappers,
   exports and its own closure/RMST identity/derivative tests; do not duplicate.
3. Then one separate executable docs/rmst.md topic: staged surface; explicit
   cutoff axes rather than rmst_grid; units; invalid values; first-order-only
   zero boundary; unspecified NaN-own tangents; native higher-shape restriction
   and supported nonshape derivatives. Coordinate navigation/API ownership.
4. Recheck the combined PH+RMST documentation/examples and both-dtype release
   evidence after separately reviewed PH integration. No new core RMST feature
   is planned; generic quadrature/sampling/options stay outside this workstream.

Validation complete: 49 focused RMST/generator tests pass per dtype;
Ruff/mypy/lock/preservation/whitespace checks pass. The fresh-wheel A10 gate
exits 0: own 1097/7 x64 and 1103/1 x32, doctests four per mode, independent
2332/28 x64 and 2322/38 x32 passed/skipped. Finish the maintainer-started
merge and obtain the independent RMST.integration review before reporting.

---

# Preserved RMST-side plan (verbatim pre-merge version)

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Other sub-slices are archived verbatim in `plans/` (index: `plans/README.md`).
Still open there: A9 (`plans/A9.md`, "A9 Docs to-do" and "Final evidence
audit and planning checks") and A8 (`plans/A8.md`, "Closure still required").

---

# RMST.3 — Weibull derivative and isolation evidence (2026-10-30)

Start release-rmst at b2150e1fe10df97d7ac2d4e9926a2b8a803f90c4;
original launch base remains 8757c0fda78e34e6e818249adedd2a144a33c33d.
The maintainer explicitly requested the next implementation step. Appendix
review is running in its separate checkout against the committed snapshot;
its findings will be handled before this step is committed. Preserve history.

Status: implementation resumed from appendix fix-up
dc48b24cd89a571a082099f4055cd1669c8ff3cd. The maintainer explicitly
approved unspecified own derivatives for a NaN RMST result (2026-10-30),
settling S-1b. Appendix round 2 returned VERDICT: merge, all findings fixed.

One bounded numerical-verification concept, about 250 changed test lines:

- New tests/unit/test_weibull_rmst_derivatives.py: independent 60-digit
  mpmath quadrature of survival and its shape/scale derivative integrands;
  integrate the scale kernel directly to avoid cancellation in m-r*S,
  and derive the cutoff derivative from survival. Use actual dtype-rounded
  inputs for references. Cross
  k={0.2,0.5,1,2,10,50}, tau/b={0.01,0.1,0.5,1,10,100,1e5},
  b={1e-4,1,1e4}; straddle both branch thresholds at four shapes too.
  Compare eager/jit/vmap forward and reverse natural Jacobians; express
  parameter errors in log-coordinate units k*dM/dk and b*dM/db, divided
  by |M|, so the section 12.1 value-scaled exception respects time units.
  The cutoff derivative uses absolute scale 1 and the S(tau) identity.
- tests/impl_tolerances.py: add only RMST-specific absolute error factors,
  zero relative tolerance. Pre-test full-range probes (142 cases, both
  dtypes) found maxima [shape,scale,cutoff] of
  [1.72e-14,2.46e-15,1.56e-15] x64 and
  [1.29e-5,1.55e-6,6.45e-7] x32 under the above normalization.
  Fix parameter factors 2e-13/5e-5 and cutoff factors 2e-13/5e-6,
  allowing rounding margin without changing any existing tolerance.
- Verify finite supported (b,tau) Hessians across the grid and independent
  analytical Hessian entries on lower-series and gamma interior cases.
  Preserve explicit shape-Hessian NotImplementedError coverage; extend it
  across forward/reverse combinations in the gamma middle branch.
- Shared shape/scale parameters: valid entries beside invalid parameters
  or cutoffs retain the isolated call's first derivatives in both modes;
  supported scale Hessians stay isolated too. Do not assert a guarantee
  about an invalid result's own tangent. Extend zero first-derivative checks
  over shapes/scales, with eager/jit/vmap and debug_nans for valid entries.

Spec 4.5, 7.2–7.3, 8.7 revision 38, 12.1–12.2 and the release handoff were
read in full. No numerical algorithm/public API, wrappers, exports, modules
in src/, dependencies, quadrature tangent, grids or ordinary docs changes.
No new higher-order boundary or shape guarantee. Docs to-do: executable
RMST topic after PH integration, with first-order-only zero boundary,
unspecified NaN-own tangents, and the native higher-shape restriction. Explain
keeping shape outside differentiated arguments for supported Hessians;
packing it with scale can materialize a zero shape tangent in native JAX.

Run focused RMST tests in separate x64/float32 processes, Ruff/mypy,
whitespace checks, and unchanged scripts/check_wheel.sh --through A10
(fresh wheel supplies the standard gate's required install artifact).
Write REVIEW.md with exact outputs, commit, invoke independent-review
exactly as RMST.3, handle up to three rounds and stop/report. PH owns
wrappers.py/exports and must wait for maintainer core integration and an
explicit ownership handoff before built-in closure edits. No merge/push.

## RMST.3 approved tolerance clarification

Maintainer explicitly answered “Confirm log-parameter scaling” to S-1/D-029.
Approved clarification: change spec 12.1's exception to name k*dM/dk and b*dM/db, with cutoff error
still at absolute scale 1. Bump specification revision 38 to 39, commit alone
on design release-rmst, then mark this worktree via spec-since.sh
dev-release-rmst --mark and commit .spec-seen plus decision/review records.
Committed separately in design as
1389af4da8cd23f7c0d38da047ba14b5ac0801f2; .spec-seen now records it.
No numerical code, tests or factor changes are needed. Leave D-029's status
maintainer-owned. Finish round 2's source-fix review before invoking round 3
on the approved contract/records; do not exceed three rounds for RMST.3.

## RMST.3 round 2 preparation

Round 1 at ab7afb1 returned VERDICT: merge. Record S-1 as D-029 and ask
whether to clarify 12.1 in log-parameter units; do not edit the spec until
answered. N-1: narrow the old Weibull invalid-cutoff tangent assertion to
valid entries only, matching the confirmed unclaimed NaN-own-tangent rule.
Apply the same correction to the analogous Exponential test. The independent
precision-failure fix is already committed at 106078d. Run those focused
isolation tests per dtype, record round-1 outcome and final gate evidence,
commit records/test corrections and invoke round 2 on the fix and N-1.

## RMST.3 precision-failure isolation fix-up

A post-commit probe found a spec 8.7 isolation violation outside the accuracy
range: float32 k=[2,0.01], b=1.25 shared, tau=[0.75,0.01] gives a valid
first value and a NaN failed second value, but the first reverse scale derivative
is NaN (forward is finite). The gamma probability underflows; the inactive
exp(log_value) then overflows because its substituted probability was one.
Protect the exponentiation operand before it runs in src/survjax/_numerics.py;
keep NaN as the failed value. Add a dtype-specific regression for failed-entry
shared-scale isolation (k=0.005 in x64). No floors or tangent rule changes.
The extra source/test lines keep this step below about 300 numerical lines.
Validation: failed-entry first/second derivatives pass in both dtypes; the
fresh-wheel A10 gate passes with own 1080/7 and 1086/1, doctests four
per dtype, acceptance 2332/28 and 2322/38 passed/skipped. Record results,
commit a fix-up, then request
round 2 only after the current round 1 report is written.

Initial implementation evidence at ab7afb1: 198 new derivative-test lines, 10 RMST tolerance
lines and nine changed lines in the existing zero test (217 numerical-test/
tolerance lines total). Focused tests pass 31 per dtype. The fresh-wheel A10
gate exits 0: own 1079/7 x64 and 1085/1 x32 passed/skipped, doctests four
per mode, independent 2332/28 x64 and 2322/38 x32. No algorithm change.
Commit and independent review remain required before core handoff.

# RMST.appendix — Permanent mathematical appendix and conservative guarantees (2026-10-30)

## Round 2 fix-up plan

Maintainer explicitly confirmed S-1b: a NaN RMST result's own derivatives may
be unspecified. Record the confirmation verbatim in D-028 and REVIEW.md;
keep valid-entry isolation mandatory and leave deviation status maintainer-owned.
F-1: extend both existing RMST Docs to-do records with the zero-boundary and
NaN-own-tangent limits. F-2: add the missing gamma-middle section pointer.
N-1: restore the mathematical reason the 24b example needs a wider dtype;
rebuild and visually inspect the adjacent PDF. N-2: label the old report
reference historical. No numerical/spec/API changes; current revision 38
already says what the maintainer approved. S-1a's relation is symmetric;
no further spec wording change is needed to preserve the approved meaning.
Run static/whitespace checks and docs builds, commit the fix-up and invoke
round 2/3 exactly before continuing with the authorized RMST.3 step.

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
higher-derivative limit, first-order-only zero boundary, unspecified own
derivatives for NaN results, and absolute-scale cutoff-derivative accuracy.
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
state staged surface, Weibull shape-derivative limitation, first-order-only
zero boundary and unspecified own derivatives for NaN results. A9/A10 docs
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


---

# Preserved incoming main plan (verbatim pre-merge version)

# README.1 — Draft the alpha README (2026-10-30)

Rewrite README.md at the maintainer's request on add-readme. Preserve the
unfinished existing plan history below. One documentation topic; no API,
dependencies, protected paths, or spec changes. Spec 4.1–4.2, 6.1, 7.3,
9.2, 10.3, 12.2. Exponential/Weibull and concordance are the first built-ins;
more families and metrics are planned, not advertised as available.

Files: README.md, PLAN.md, REVIEW.md. Present current supported likelihood
scope, extension requirements, a core-only runnable gradient example, local
installation, principles and documentation. No new tests: execute the README
snippet against the source in both dtype modes with an analytical check, run
scripts/build_docs.sh directly, and check whitespace. Overwrite REVIEW.md,
commit, then run independent-review for README.1 and stop after its verdict.
Target under 300 changed content lines, excluding existing review records.

---

# SIM.3 — Complete the approved examples diagnostics environment (2026-10-30)

Start clean on release-simulation at
`ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d`; SIM.2 round 1 returned merge.
Preserve the unfinished SIM/A10 plan history below. The maintainer approved
ArviZ 0.23.0 in the existing dependency proposal and now requests the next step.
One concept: finish the examples dependency group for Bayesian tail ESS.

Files and scope:
- `pyproject.toml` / `uv.lock`: add exact `arviz==0.23.0` to examples;
  resolve without upgrading existing packages or changing runtime dependencies.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark the six-package setup
  complete, record the actual installed diagnostics probe and its command.
- Overwrite `REVIEW.md`: reading order, dependency isolation, actual evidence
  and limits; preserve the short temporary probe source for reproducibility.

Spec 10.3; shared use-case 00 dependency-group rule and Bayesian details
Sampling/Diagnostics: NumPyro split R-hat/ESS, ArviZ tail ESS. No source API,
shared generator, ordinary tests, docs/navigation/config or protected scripts.
Docs to-do: the future Bayesian page should use this group for tail ESS.
Target below 300 changed implementation/config/lock lines, records excluded.

Checks: `uv lock --check`, `uv sync --locked --group examples`, compare old/new
lock name/version mappings. Explicit optional-environment probe uses seeded
4x1000 iid Normal draws (known equal-chain law) and deliberately shifted
chains to verify the chain/draw axes, NumPyro summary and ArviZ bulk/tail ESS
and R-hat; this is compatibility evidence, not posterior convergence evidence.
The pinned ArviZ import writes a daily stamp under Path.home()/arviz_data, so
run that probe with required sandbox escalation rather than modifying HOME.
No new ordinary tests for the dependency-only change. Run the existing focused
generator suite in both dtype processes, Ruff, mypy, and unchanged fresh-wheel
through-A10 gate. Record versions/results, write REVIEW.md, commit, invoke the
exact independent-review skill for SIM.3, follow at most 3 rounds and stop.
PH fitting still waits for maintainer integration of reviewed prerequisites.


Implementation: added the approved ArviZ pin and six lock packages. All 165
prior package versions unchanged; runtime dependencies remain jax/numpy.
Actual installed-group CPU probe passes named chain/draw axes, iid gates and
shifted-chain detection: NumPyro R-hat 1.0000270083 / ESS 3976.58; ArviZ
R-hat 1.0001031616 / bulk ESS 3979.33 / tail ESS 3931.29. ArviZ's expected
import warning was emitted; sandbox escalation allowed its home-cache stamp.
Existing generator suites: 17 passed per mode (3.60s/3.18s), Ruff/mypy/lock
checks pass. Fresh-wheel through-A10 gate exited 0: own suites 1065/1071
passed (7/1 skipped), doctests 4 per mode, independent 2333/2324 passed
(27/36 skipped). ArviZ's pandas dependency enabled the existing x32 pandas
conversion test. Wheel hash is identical to SIM.2 and runtime METADATA still
requires only jax/numpy. REVIEW.md records exact commands, outputs and limits.
Commit and exact independent review follow; stop after the verdict.

---

# SIM.2 — Address shared foundation questions and enable examples (2026-10-29)

Start on `release-simulation` at `3556d12c1abca553f77607c1e1739df2bd1911f0`.
The only initial dirty file is study.py with three maintainer question comments.
SIM.1 round 2 returned merge. Preserve all unfinished history below.
Maintainer approves the proposed dependencies in this turn; no renewed
approval is needed for those exact pins. Keep PH fitting deferred.

Files and scope:
- `pyproject.toml` / `uv.lock`: examples group with exact tested Optimistix
  0.1.0, Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0.
  Keep existing runtime/dev/docs versions unchanged. The five-package lock
  preview is 160 changed lines; full six-package setup is 295 lock lines
  alone. Approved ArviZ 0.23.0/tail ESS is a separate later step so this step
  including generator checks remains about 300 lines. No protected scripts.
- `examples/weibull_simulation/study.py`: answer/consume the maintainer notes,
  define quadrature rules before their closure, and explain calibration as
  once-per-target host setup outside the compiled replicate loop. PH input
  preference is asked asynchronously (baseline/log-HR versus PH rate m).
  Preserve the current call contract; any PH convenience will be optional,
  demo-only, and checked against the existing scale formulation.
- `tests/unit/test_simulation_generator.py`: if adding PH input, analytical
  PH/scale and calibration identities, row broadcasts and transformed values/
  derivatives. Reuse existing tolerances. Existing numerical checks also
  exercise the quadrature block reorder.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark approval, staged pins,
  remaining approved ArviZ stage and exact optional-environment invocation.
- Overwrite `REVIEW.md` with reading order, input-contract updates, evidence
  and limits, preserving SIM.1's outcome in this plan and Git history.

Spec 4.4/4.6/7.3/10.3; use-cases/00-common and 01-details DGP/calibration.
No library API, shared docs/config or other workstream files. Docs to-do:
future simulation page should show PH inputs and setup versus compiled work.
Run generator checks in separate x64/float32 processes, approved-environment
compatibility probe, Ruff, mypy, lock consistency and fresh-wheel through-A10
gate. Inspect size before committing; split if needed. Commit, invoke exact
independent-review skill for SIM.2 (max 3 rounds), then stop/report.


Implementation: optional log_hazard_ratio=0.0 added to both generator and
calibration, with unchanged no-keyword calls and U/C key split. Rules now
precede the fraction closure; calibration explicitly remains host-only setup.
No response to the optional PH-form preference arrived during environment
work; used the stated baseline/log-HR convention from 01-details and spec 4.4.
Approved five-package group resolves/installs; all 155 existing locked
versions unchanged, 10 additions. Optional compiled BFGS/NN-update/NUTS/SVG
probe passes in the actual group environment. ArviZ remains approved for its
separate lockfile step. Total implementation/test/config/proposal diff is
303 changed lines, including 160 lockfile lines.

Focused final suites: 17 passed per dtype (10.17s/10.56s), Ruff/mypy/lock
checks pass. A joint three-parameter check_grads initially failed a float32
second-order projection near zero; no tolerance was changed. Retained the
original k/b finite-difference checks and verified all PH gradient/Hessian
entries against inverse-CDF algebra, including cross-partials. The full
fresh-wheel through-A10 gate exited 0: own suites 1064/1071 passed (7/1
skips), doctests 4 per mode, independent 2333/2323 passed (27/37 skips).
NumPyro enables one acceptance check previously skipped. REVIEW.md records
exact commands, times, initial numerical failure and final focused evidence.
Commit and exact independent review follow, then stop/report.

---

# SIM.1 — Shared Weibull generator and censoring calibration (2026-10-29)

Work on `release-simulation` from full base
`8757c0fda78e34e6e818249adedd2a144a33c33d` (clean launch snapshot).
The matching reviewer checkout already exists. Preserve the unfinished A10
history below; its docs and A9/PH/RMST work belong to other implementers.

One concept: reusable demo-only inverse-CDF generation and host-side numerical
calibration, without PH, fitting, core changes or optional dependencies.
Spec revision 37: 3.3, 4.4 (generating PH-equivalent scales only), 7.3,
10.3, 11.4, 12.1–12.3; use-cases/00-common.md and both 01 files.

Files and contract to implement:
- `examples/weibull_simulation/study.py`: `generate_weibull(key, shape, scale,
  censor_limit, *, sample_shape=())`, returning a named tuple with `time`,
  `event`, `event_time`, `censor_time`. Parameters broadcast to row/batch axes;
  sample_shape prefixes those axes. Exactly split the supplied key into U/C
  keys; callers derive scenario/replicate keys with the specified fold/split
  scheme. JAX generation supports jit/vmap and selected process precision.
- Same module: `calibrate_censoring(shape, scale, target) -> float` uses NumPy
  float64 quadrature and bracketed bisection of the mean integrated survival.
  Concrete positive finite broadcast parameters only; calibration is host-only,
  not an RMST implementation or differentiable library API. Check quadrature
  convergence; fail explicitly rather than return an unverified calibration.
- `tests/unit/test_simulation_generator.py`: analytical exponential calibration,
  independent mpmath Weibull quadrature, heterogeneous broadcasting, key mapping,
  transformation agreement and fixed-seed large-sample survival/HR/censoring
  recovery with justified statistical bounds. Centralize any new test bounds
  in `tests/impl_tolerances.py`. Ordinary checks remain offline.
- `examples/weibull_simulation/DEPENDENCIES.md` and a temporary isolated environment
  probe (record the exact invocation in REVIEW.md): concrete exact-version proposal for the later examples group, test
  Optimistix/Matplotlib/Equinox/Optax/NumPyro against locked Python/JAX/NumPy.
  No pyproject/lock edits or added diagnostics/loader packages in this step.
- `REVIEW.md`: reading order, invariants, evidence and limitations; freeze the
  reviewed consumer import contract and calibrated simulation c values.

Target at most about 300 implementation/test/proposal lines (plan/review
records separate); split fitting, scenario runner, results and docs into later
steps. Docs to-do: executable simulation page and recorded full results after
reviewed PH integration; propose fresh docs command with examples selected.

Checks: focused tests in separate JAX_ENABLE_X64=1/0 processes, Ruff formatting
and lint, mypy src, unchanged `scripts/check.sh --through A10` cumulative gate.
Record exact versions, commands and outputs. Commit the step, invoke the exact
independent-review skill command for SIM.1, follow up to three rounds, then stop.
Dependency installation into the project still awaits maintainer approval;
only an isolated proposal probe is authorized here. Never merge/push/publish.


Implementation: 300 added source/test/proposal lines, excluding plan/review
records; no library API, dependencies, lockfile or protected-file edits.
Reviewed consumer contract is recorded in REVIEW.md pending round 1.
Both simulation targets independently match 40-digit mpmath quadrature;
focused final suites: 12 passed in each dtype, including second derivatives.
Ruff formatting/lint and mypy pass. Calibrated c values: 47.793149053990135
and 19.17253543619806 years; pilot empirical censoring .249355/.550485.
Dependency versions proposed and tested in isolation: Optimistix 0.1.0,
Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0 against the
existing locked JAX/jaxlib 0.11.2 and NumPy 2.5.3; group/lock approval pending.
Cumulative fresh-wheel through-A10 gate: exit 0, all gates passed.
Own suites 1059/1066 passed (7/1 existing skips); independent 2332/2322
passed (28/38 existing skips); source doctests 4 per mode. REVIEW.md records
exact timings, commands and the focused derivative-test evidence. Commit and
independent round 1 now follow; stop after the review verdict.

Round 1 on 1e3f5fc returned fix-first for F-1: add explicit quadrature-failure
coverage. Add the k=20/b=5/target=.5 RuntimeError regression, and clarify N-1
underflow and N-2 time units in docstrings; N-3 import-helper suggestion stays
conditional. Fix-up is 7 net source/test lines (307 total), no behavior or
contract change. Focused suites: 13 passed per mode (4.32s/4.05s), Ruff and
mypy clean. Commit separately and request round 2 after round 1 completes.

---

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; all incoming A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.d1 — Using and verifying the installed Slice A subset (2026-10-29)

Authorized by "the current work is approved; if more work needed, get to it".
Work on the existing a9b branch from clean bd32ff9. The maintainer completed
the main merge, manually reran `scripts/check_wheel.sh --through A9` without
errors, and cleaned the reviewer-owned throwaway worktree. Record those as
maintainer-reported checks/cleanup, not as runs performed by this agent.
A9.4 is accepted; optional review nits N-1/N-2 do not block this docs step.

One topic: explain the installed subset's verification and limitations.
Spec: 6.1–6.2, 7.2–7.3, 10.3, 10.5, 12.1–12.3. Public API unchanged.
At most 300 changed docs lines. No dependency, source, test, fixture,
benchmark, protected-path or spec edits. PLAN.md keeps only current A9.

Files to create/change:

- Create docs/slice-a-verification.md as executable MyST. Give a compact
  synthetic exact/right example with construction and diagnostic access inside
  jit, value/parameter gradient/Hessian checked against hand algebra, eager
  and prepared/raw agreement, and vmap over rates. Document the smooth
  interior parameter-only guarantee and explicit host checkpoints.
- Explain the current subset and unsupported likelihood modes; link existing
  construction, likelihood, numerical, distribution/grid and concordance
  guides. Describe the fresh-wheel runner, installed out-of-tree custom-family
  checks, pinned synthetic R evidence and the exact CPU reproduction commands.
  Explain first-call versus warmed costs, prepared/traced flags, returned
  diagnostics, logical bytes versus unmeasured peak memory, and optional
  benchmark nits without silently changing accepted measurements.
- Add the new page to docs/index.md's prose and toctree. Audit the remaining
  executable pages and playground. Correct only stale capability statements
  within this verification topic; leave unrelated teaching work for separate
  steps. Document the locked JupyterLab command.
- The rendered audit found docs/survival-data.md's opening note still claims
  likelihoods, distributions and evaluation are unimplemented. Correct that
  one scope note and link current likelihood/concordance guides. This narrow
  factual correction belongs to the installed-subset verification topic;
  it adds no new tutorial or runtime behavior. Total docs remain below 300
  changed lines. Record the discovery before making the edit.
- Overwrite REVIEW.md with reading order, invariants, evidence mapping and
  remaining gate status. Record every Slice A gate item with concrete tests
  or artifacts and honest pass/pending status; this is not an A10/release
  completion claim. The agreed README/use-case example replacement is still
  separate from this small compiled-verification recipe.

Validation ownership: implementer owns fresh `scripts/build_docs.sh` in both
modes, Ruff and rendered-page/browser inspection. The maintainer's single-
runner preference means the independent reviewer should inspect source and
recorded outputs instead of repeating the same build/gate. No full wheel-gate
repeat for docs-only edits: the maintainer just reran it successfully on the
merged baseline, and no runtime/packaging change follows. Inspect links,
outputs, formulas, navigation and tables across the site in both builds;
inspect the new page carefully and record any existing warning/limitation.
Verify committed notebooks still have no outputs/counts. No new unit tests:
the executable page's assertions check its example, and existing suites
already cover the numerical and packaging behavior.

After successful build/source/visual checks, record exact commands/results,
commit this docs step on a9b with "Independent review: round 1/3" in REVIEW.md,
invoke the exact independent-review skill command for step A9.d1 in
../review/code-a9b, address blocker/should-fix findings in separate fix-ups
within three rounds, then stop and report. Do not start another step.

Docs/closure to-do: agreed replacement README/use-case teaching material and
any gaps from this rendered-site/evidence audit remain explicit; don't claim
Slice A or A10 complete until their separate gate items are accepted.

## A9.d1 evidence and audit conclusion

Implemented the verification page, index navigation and one stale data-guide
scope note; no code or notebook changes. Both final docs builds pass, nine
executed documents and 137 code cells per mode. The three new example cells
check exact analytical values, gradients/Hessians, input forms and jit/vmap.
All 1,048 local file/fragment links per rendered site pass. The committed
playground remains output-free. Ruff passes for the repository and the
extracted three-cell notebook; JupyterLab CLI reports 4.6.4. Native Brave
inspection covers all 11 content pages in float64, all pages' accessible
content in float32 and detailed views of the new page, corrected note,
playground and representative grids/formulas/tables. REVIEW.md records final
execution times, initial sandbox/interruption errors, checks and limits.

The final evidence map is included in this step's REVIEW.md rather than
requiring a separate metadata-only audit step. The new three-observation
verification recipe provides tiny synthetic smoke evidence; the larger
agreed README/use-case teaching replacement remains separate work and is
not claimed complete. The maintainer's post-merge fresh-wheel pass supplies
current regression evidence; no runtime/packaging changes follow it.
A9's verification evidence is now ready for independent review and maintainer
acceptance. No claim that A10, Slice A as a whole or the release is complete.
Stop after this docs step's review; no additional implementation starts.

## A9.d1 round-1 follow-up

Round 1 on 2c8cd60 returned fix-first for should-fix F-1: cite the explicit
unsupported-mode NaN/preserved-neighbor requirement and executed triage
recipe in the gate map, and map each narrative item in spec 12.2's paragraph
to its docs page. Existing code/tests already cover this behavior; no new
implementation or test is required. Also fix trivial N-1 (R paragraph line
wrap) and N-2 (clarify that tests/doctests, rather than static checks, run
in separate dtype processes). Code cells remain byte-identical.

The reviewer reran the build because its role requires it, despite the
initial ownership request. Its build passed in both modes (new page 3.12s /
3.07s); Ruff passed. No wheel gate or mutations were rerun. For this prose/
metadata fix-up, the reviewer alone owns round-2 builds/lint. The implementer
checks only the diff and unchanged cells, records F-1/N-1/N-2, commits a
separate fix-up and requests round 2. No duplicate build/full-gate run by
implementer; maximum three rounds, then stop.

---

# A9.4/main merge resolution (2026-10-29)

Authorized by "merged main into the branch; fix conflicts". The maintainer's
merge is in progress on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflict; resolve these metadata files. No public
API, code, docs, dependency, spec or test behavior changes are needed.

- Preserve A9.4's complete plan and evidence below. Independent review round
  1 on 51319c8 returned merge, with optional N-1/N-2; those nits remain open.
- Preserve all main's A10 plan sections verbatim in plans/A10.md, including
  A10.d2, A10.d1 and A10.4. Its older A10 history remains the unchanged suffix
  of that archive. PLAN.md keeps only the current A9 sub-slice. A10's incoming
  review record is preserved in REVIEW.md alongside the A9.4 record. These
  are historical records; this merge starts no new implementation step.
  This corrects the first resolution, which incorrectly kept incoming A10
  sections in PLAN.md; the maintainer explicitly checked the current-slice
  format before completing the merge.
- Preserve every original stage-0 index entry except the intentional
  plans/A10.md archive extension and plans/README.md index description.
  Incoming docs, concordance fixtures/tests, release-agent plans and
  .workmux.yaml remain untouched. The last is an incoming protected-path
  change, not an agent edit.
- Verify both parents' records are retained, no conflict markers/unmerged
  entries remain, whitespace is clean and every other staged object is
  unchanged from the captured merge index, apart from the two archive/index
  changes above. No runtime/build reruns: neither
  conflicted file affects execution, and no manual executable edit is needed.
  Preserve incoming A10.d2's single-runner validation ownership.
- Stage the two resolutions and archive/index corrections; leave the merge commit to the
  maintainer under the working agreement. Do not create a merge commit,
  start the next implementation step or rerun independent review for this
  metadata-only resolution.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.

---

# Preserved incoming alignment plan (d1825f0)

# ALIGN.REVIEW — Independent Codex Astra review (complete)

Starting/previous step: 91a4503; worktree starting point: 3cb4cd7.
Maintainer explicitly requests a fresh Codex Astra session, low reasoning,
replacing the earlier reviewer waiver for this review. Spawned gpt-6-astra
with reasoning low and no inherited conversation history. Round 1/3 reviewed
23e8c3e..91a4503, including design cleanup and merged author corrections.
Reviewer wrote ASTRA-REVIEW.md; author did not alter that report. Verdict:
merge, no actionable findings. Reviewer independently passed the fast gate
through A12 in both modes and audited 46 docs/example Python blocks across
7 files. Full Tier 2/builds/release certification remain outside that review.
No code changes or fix-ups. Record report and status only, then stop.

---

# ALIGN.3 — Final alignment verification (complete)

Starting commit: 3b00d10 (maintainer's correction merge); previous completed
substep: 97d473f (ALIGN.2c); worktree starting point: 3cb4cd7.
D-043/D-044 acceptance corrections are now integrated. .spec-seen equals main
and design HEAD 213cfb0 (revision 47, fixed simulation censoring limits),
which includes the approved D-042 example cleanup. No implementation change
is proposed. Run scripts/check.sh --fast --through A12 in both modes through
the normal runner, inspect any failure, record exact evidence in REVIEW.md
and update handoff status notes. Prior touched-file and docs-build evidence
remains valid; no redundant rebuilds. No new API, dependency, code, test,
spec or protected-path changes. Independent review remains waived by the
maintainer while reviewer tokens are unavailable. Tier 2 remains the
maintainer/CI merge gate. Stop after this verification-only step.

Completed: scripts/check.sh --fast --through A12 exits 0 and reports all
gates passed. Implementation tests: x64 1059 passed/8 skipped; x32 1064
passed/3 skipped. Source doctests: 4 passed per mode. Acceptance: x64 3397
passed/32 skipped; x32 3391 passed/38 skipped, with two existing NumPy
reference warnings per mode. Static checks pass. D-043/D-044 integration
evidence recorded without changing formal deviation statuses. REVIEW.md holds
commands/timings and limits. All alignment substeps complete; no further
implementation/docs corrections identified. Full Tier 2 and broader release
obligations remain separate; independent review remains waived as requested.

---

# ALIGN.2c — Approved D-044 author handoff (complete)

Starting/previous step: 3a6d013; worktree starting point: 3cb4cd7.
Maintainer approves restricting upper-tail relative comparisons to the spec's
normal-survival range while preserving shape/full-grid agreement and existing
tolerances. Saved SPEC-TEST-D044.txt; refreshed the D-043 note's cross-reference.
Checked the instruction against spec 4.1/4.2 and the affected A11 grid test.
No spec, implementation, acceptance-test or dependency change; no tests rerun
for this handoff-only step. Approval recorded without changing deviation
statuses. All three decisions now have maintainer confirmation; acceptance
corrections still belong to the independent author. ALIGN.3 final verification
awaits their normal synchronization. No additional decision pending here.

---

# ALIGN.2b — Approved D-043 author handoff (complete)

Starting/previous step: 3580ff2; worktree starting point: 3cb4cd7.
Maintainer confirms the expected condition mask is [False,True,True] and asks
for a short instruction note. Saved SPEC-TEST-D043.txt for the independent
author; no acceptance tests, spec or implementation changed. Checked note
against the failing test, custom predicate and spec 4.7. No tests rerun for
this planning/handoff-only step. D-043 approval recorded; formal status stays
with the maintainer. D-044 is the next decision; final gate remains pending.

---

# ALIGN.2a — Approved D-042 spec example cleanup (complete)

Starting/previous step: a4c8b42; worktree starting point: 3cb4cd7.
Maintainer explicitly approves example-only cleanup of positional PH snippets.
Change spec/SPECIFICATION.md (4.4, 11.3) and use-cases/01-weibull-simulation,
01-weibull-simulation-details, 02-bayesian-survival and
02-bayesian-survival-details Markdown. Preserve requirements/revision 46;
make constructor spelling consistent with revision 42. No public API, code,
acceptance-test or dependency changes. Inspect exact diff, scan all PH snippets
and AST-check the Python model/route sketches. Commit spec separately in the
design repo with the prescribed spec(...) subject. Then update this worktree's
.spec-seen and progress/review records in a separate implementation-repo commit.
The spec-since helper accepts a repo-directory argument: use rmsttest rather
than dev so it marks this worktree, not the unrelated dev checkout.
Independent review remains waived. No executable change warrants numerical
or docs rebuilds. Stop after cleanup and present D-043 as decision 2/3.

Completed: design commit 54bbc62 corrects examples in five spec files;
revision 46 and requirements unchanged. Exact diff/whitespace and PH snippet
scan pass; both Bayesian Python sketches retain identical ASTs after keyword
normalization. Ran ../scripts/spec-since.sh rmsttest --mark and commit the
updated marker here. D-042 approval recorded without changing its status.
No numerical or docs rebuild; no executable change. D-043 is next for decision.

---

# ALIGN.d1 — Keyword-only PH examples in executable docs (complete)

Starting/previous step commit: e8e9a51 (ALIGN.1); worktree starting point:
3cb4cd7. ALIGN.2's owner decisions are pending; this independent docs work
can proceed. One topic across four pages: correct PH constructor argument
spelling in docs/proportional-hazards.md, docs/rmst.md,
docs/technical-notes/rmst.md and docs/use-cases/bayesian-rmst.md. Replace the
obsolete positional-baseline promise. No public API, formulas, dependencies,
parameters, posterior data or docstring changes. Spec 4.4, 8.7, 10.5, 11.3.
No new tests for these source edits. Validate code-cell AST equivalence after
normalizing argument spelling; scripts/build_docs.sh directly for both dtype
modes; rendered-page checks; Ruff/whitespace. Under 300 changed lines.
Independent review remains waived by the maintainer while reviewer tokens
are unavailable. Stop after this documentation step and report.

Completed: 15 constructor calls across four pages and one obsolete prose
promise corrected (32 changed docs lines). All 33 Python/code-cell blocks
retain identical ASTs after keyword normalization. scripts/build_docs.sh
succeeded directly in both dtype modes; each rendered site has 23 HTML pages,
1844 valid local links/assets and all 15 calls/prose verified. Ruff and
whitespace checks pass. No implementation/test/spec/dependency changes.
Independent review waived as above. Stop here; only ALIGN.2 owner decisions
and ALIGN.3 verification after those decisions remain. There are no remaining
constructor/doc migrations found by the source audit.

---

# ALIGN.1 — Keyword-only PH construction (complete)

Starting commit: fc484d3; previous completed substep: fc484d3 (ALIGN.0).
Previous docs step/worktree starting point: 3cb4cd7. Implement the ALIGN.1
scope below, under 300 changed implementation/test lines. The maintainer
waives independent review while the reviewer is out of tokens (2026-11-06).
Stop after this step and report the remaining work. Protected files stay
read-only; D-043/D-044 remain owner questions. Docs to-do: ALIGN.d1 below.

Completed: removed the positional-baseline adapter; ordinary family registration
owns initialization. Migrated eight unit files, both Bayesian Python modules
and three notebook cells; retained explicit rejection tests. Implementation,
test and example changes total 137 lines. Touched suites pass x64 210 / x32
209 tests; Bayesian factor/trial/4000-draw RMST-route checks pass. Fast-gate
implementation suites and doctests pass both modes. Acceptance remains red
only for D-043 (both modes) and D-044 (x32); standalone x32 wheel setup errors
were corrected by supplying SURVJAX_WHEEL, then all 5 wheel tests passed.
REVIEW.md records exact commands/counts and the review waiver. No docs/spec
or acceptance-test edits. Stop here: ALIGN.2, ALIGN.d1 and ALIGN.3 remain.

---

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
