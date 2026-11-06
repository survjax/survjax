# ALIGN.d1 — Keyword-only PH documentation

Independent review waived by maintainer while reviewer tokens are unavailable
(2026-11-06). Starting/previous step: e8e9a51; worktree starting point: 3cb4cd7.

Read first: PLAN.md ALIGN.d1; docs/proportional-hazards.md constructor prose;
then the PH calls in docs/rmst.md, docs/technical-notes/rmst.md and
docs/use-cases/bayesian-rmst.md. Spec 4.4, 8.7, 10.5, 11.3.

Invariants: one topic, constructor argument spelling only. Preserve executable
cell meaning, formulas, model settings, parameters and stored results. Correct
all executable PH examples together so the full docs build can execute.
No new public API, dependencies or tests. Validation is recorded below.

D-042/D-043/D-044 remain owner resolutions. Do not touch spec or acceptance
tests. This docs step does not close the red numerical acceptance gate.

## Validation and results

- `scripts/build_docs.sh`, executed directly: exit 0; float32 and float64
  Sphinx builds both reported `build succeeded`, with forced notebook
  execution and warnings-as-errors configuration. All changed executable
  pages ran successfully in each mode. Kernel startup notices were printed;
  neither Sphinx build reported a build warning/failure.
- Source AST audit, `uv run --no-sync python /tmp/align-docs-audit.py`:
  33 code blocks retain identical ASTs after baseline keyword normalization;
  all 15 PH calls are keyword-only. The audit includes executable code-cell
  blocks and displayed Python examples, and scans the rest of docs/ too.
- Rendered HTML audit,
  `uv run --no-sync python /tmp/align-docs-html-audit.py`:
  x32 and x64 each contain 23 HTML pages and 1844 valid local link/asset
  references, including fragment targets. Each site contains all 15 corrected
  rendered PH constructor calls and the corrected keyword-only prose.
  This is an HTML content/link audit; no browser visual inspection was run.
- `uv run ruff format --check .`: 173 files already formatted.
- `uv run ruff check .`: All checks passed!
- `git diff --check`: exit 0.
- `git diff --exit-code -- src tests examples pyproject.toml uv.lock spec`:
  exit 0. Only four docs pages and PLAN.md/REVIEW.md changed. Documentation
  source diff is 32 changed lines, within the one-topic step limit.

No new tests for argument spelling; the docs build executes the examples.
No numerical regression rerun, Tier 2 or independent review for this docs-only
step. Existing numerical evidence remains in e8e9a51:REVIEW.md; the maintainer
waived independent review while the reviewer is unavailable.

ALIGN.d1 is complete. ALIGN.2 still requires owner decisions on D-042/D-043/
D-044, and ALIGN.3 remains final verification after those resolutions. The
acceptance gate remains red on the known discrepancies; this step does not
claim merge readiness. Stop after this step.
