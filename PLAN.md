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
