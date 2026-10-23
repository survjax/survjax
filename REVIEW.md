# Review: A7.d1 follow-up — Zero-censoring diagnostics

## What to read, in order

1. `docs/numerical-behaviour.md`, section "Zero censoring is an exact boundary
   value": the added paragraph and executable DataIssues display.
2. The follow-up entry at the top of PLAN.md.

The maintainer requested this narrow correction directly. Existing A7 in
`/private/tmp/survjax-a7-docs`, starting at f7236c4; shared A10 checkout untouched.
Only the tutorial and plan/review metadata change. No source/API/test changes.

## Invariants and evidence

For right censoring at 0 with default truncation, the observed set is the
whole inclusion window. The row is valid but adds no parameter information:
`issues.ok=True`, `issues.invalid_rows=False`, `issues.uninformative=True`.
The last flag is non-fatal and does not enter the invalid-row union (spec 2,
3.3, 6.2). The text also explains that explicit host validation passes.

The added cell constructs SurvivalData(0.0, event=False) and displays:

    DataIssues: all 1 row valid
      uninformative             1 row, first index ()  (not an error)

This uses the actual public issues property and its readable representation;
no separate data_issues API or new diagnostic flag is introduced.

## Checks actually run

- Separate JAX_ENABLE_X64=1/0 `.venv/bin/python -` diagnostic probes: each prints
  `ok: True invalid_rows: False uninformative: True` and
  `raise_if_invalid returned: None`, without an exception.
- `UV_CACHE_DIR=/private/tmp/survjax-a7-docs-uv-cache scripts/build_docs.sh`:
  exit 0, `build succeeded.` for both float32 and float64, warnings fatal.
  The page executes in 2.90s / 2.82s respectively. Full log:
  `/private/tmp/survjax-a7d1-zero-diagnostics-build.log`.
- `.venv/bin/python /private/tmp/survjax-a7d1-zero-inspect.py`: both builds
  pass, with 8 outputs, 6 headings, 2 math blocks, 5 table rows and 30 valid
  local links/anchors. Read the new diagnostic output in both generated pages;
  it matches the display above. Result artifact:
  `/private/tmp/survjax-a7d1-zero-html-inspection.json`.
- Final Ruff formatting/lint and git diff checks run after this review edit;
  results are recorded in PLAN.md and the completion message. No full code
  gate, R, installation or accelerator checks were rerun for this docs edit.

The previously recorded browser file-URL restriction still prevents visual
layout inspection; it was not retried. Local HTML/output inspection is complete.
No numerical/spec ambiguity found. Commit on isolated A7, then stop for review.
