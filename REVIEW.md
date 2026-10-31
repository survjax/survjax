Independent review: round 1/3

# RMST.d2 — Guide clarification and explicit API-reference handoff

Read PLAN.md RMST.d2 and RMST.d1 review N-1/N-2/N-3, then the narrow
docs/rmst.md diff and the Docs to-do ownership statement in PLAN.md.
This is one small docs-only follow-up (28 content lines changed).

Commit checkpoints required by updated AGENTS.md rule 11:
- Step start: 6d5e19a5a983fda6f236ceb9c15235af26953a2d.
- Previous step final: 4440f1b6dc09d9b7ae707b341ff39b32cbb7c730 (RMST.d1).
- Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
The final commit for this step will be reported after committing/reviewing it.
The maintainer, not this step, merged updated main instructions at the start.

Findings addressed and invariants:
- N-1: the page now distinguishes the actual (3,) vs (3,2) TypeError from
  a (2,) cutoff pairing with arms. A new code cell catches the static error,
  asserts paired output (3,2), and verifies each paired arm
  against the corresponding time/arm slice of the explicit (3,3,2) grid.
  These are broadcasting identities from spec 5/8.7, not a new tolerance.
- N-2: replaced the process-oriented scope heading and merge/branch wording
  with Available summaries and plain current/planned support statements.
  PH RMST stays labelled planned because it has not landed in code main.
- N-3: PLAN.md explicitly assigns docs/api.md's Exponential AND Weibull
  rmst member entries, and the subsequent PH rmst entry, to release-ph.
  Combined PH/RMST evidence must check these rendered entries. This records
  the known reference gap without conflicting with that owner's work;
  docs/api.md is intentionally unchanged in this step.
- Read the full revision-39-to-40 specification diff. It defers extreme PH
  direct-method/likelihood parameter derivative stabilization; the spec says
  staged RMST is unchanged. .spec-seen now records the read design HEAD,
  765c65bba9efb3af8ef4a13c1ecb8e302a47f346. No spec change was authored.
- No src/, tests/, tolerance, API/export, dependency, protected path, README,
  docs/index.md or docs/conf.py changes. Preserve all unfinished plan history.

Checks actually run:
- scripts/build_docs.sh directly, using approved escalation, without env
  assignments, redirects or shell wrappers: exit 0; both strict -E -n -W
  dtype builds succeeded. Kernel TCP startup warnings were emitted, with no
  Sphinx warning or execution error. Nine RMST code cells execute per mode.
- The rendered x32 and x64 pages include the expected TypeError-catch message
  and "One cutoff per arm: (3, 2)" output. The completed notebook has nine
  execution counts and no error outputs. Original RMST(12) results remain
  4.54641056060791 (x32) and 4.546410233552938 (x64).
- Generated HTML inspection: updated Available summaries anchor, retained
  section anchors/navigation/local links, new paired-cutoff outputs, figure
  assets and byte-identical mathematical appendix download all pass in both
  modes. The unchanged plot and full-page browser layout were not rechecked;
  RMST.d1 recorded the unavailable browser host and its figure inspection.
- uv run ruff format --check .: 126 files already formatted.
  uv run ruff check .: all checks passed.
  uv run mypy src: no issues in 21 source files.
  uv run --group test pytest tests/unit/test_api_reference.py -q: 1 passed.
  git diff --check: clean. No full numerical suite rerun for unchanged code.

No unresolved mathematical or implementation choice is introduced. PH's
closure, API reference and combined PH/RMST docs/tests remain its integration
handoff; this step makes no PH availability or full-release completion claim.
Commit, invoke independent-review for RMST.d2 (max 3 rounds), then stop/report.
