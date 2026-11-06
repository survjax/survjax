# DOCS.4 — Remove internal citations

Read the diff for src/survjax docstrings, docs/simulation-guide.md,
docs/technical-notes/verification.md and appendices/ph-log-survival.md.
Internal implementation comments are intentionally outside this user-facing
cleanup. Preserve all behavior descriptions, mathematical content and the
maintainer's uncommitted callout-title edit. No runtime changes intended.

Previous step/start: 300c368; original worktree base: 2093ca2.
Validation complete; independent review remains waived. No further substeps
beyond cleanup, verification and commit.

## Completed checks

- Removed internal citations from 46 package/config docstrings. AST-guided edits
  preserve their remaining text. `python /tmp/audit_citation_cleanup.py` passes:
  23 Python files have identical executable ASTs after removing docstrings and
  identical comment tokens; remaining docstrings have no target references.
- Prose scan covers docs Markdown, root README, examples and appendix Markdown;
  no decision-log citations found in these surfaces. Internal source comments
  and historical plans/reviews remain intentionally unchanged.
- PH appendix: removed only the final contract-citation sentence. Rebuilt with
  `pandoc appendices/ph-log-survival.md --standalone --pdf-engine=xelatex
  --lua-filter=appendices/ph-log-survival-pages.lua -o appendices/ph-log-survival.pdf`:
  exit 0. `pdftotext -layout` checks both appendix PDFs: no target citations.
  `pdftoppm -scale-to 1100 -png` rendered all five PH pages; inspected each PNG:
  equations, tables, page breaks and final paragraph intact, no clipping.
  Weibull RMST appendix source/PDF unchanged.
- `uv run ruff format --check .`: `166 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 22 source files`.
- `git diff --check`: clean.
- First direct escalated `scripts/build_docs.sh` passed both modes, but the
  rendered-text audit found spec and D-011 comments exposed in an expected-error
  traceback on SurvivalData. Changed that single demonstration to catch and
  print the same TypeError/message, without implementation traceback details.
  Final build and rendered audit follow below.

No numerical test suite or independent review run: runtime ASTs are unchanged.
The maintainer's callout-header edit stays unmodified and unstaged; the build
includes it as it exists in the working tree.

Final direct escalated `scripts/build_docs.sh`: exit 0, both modes report
`build succeeded.` (17 executable pages, 145 code cells per mode).
`python /tmp/audit_rendered_citations.py`: all rendered HTML in both modes free
of internal contract citations; downloaded PH appendix matches rebuilt PDF;
changed source copies and intended TypeError output verified.
`python /tmp/audit_docs.py`: 22 HTML pages, 1,739 local links/assets per mode,
zero failures. Final whitespace check clean. No remaining substeps.
