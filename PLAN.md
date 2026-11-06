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
