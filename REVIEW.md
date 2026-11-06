# DOCS.1 — Documentation restructure

Independent review waived by the maintainer for this restructuring. The full
scope and size were explicitly approved, including playground removal.

## Reading order

1. DOCS-MIGRATION.md: section-level moves, consolidation, deletions and deferred work.
2. docs/index.md and docs/working-with-distributions.md: navigation and entry path.
3. docs/distributions.md and docs/proportional-hazards.md: basic extension tutorial
   and built-in PH wrapper progressing from exponential to piecewise baseline.
4. Shortened built-in, likelihood and RMST guides; reorganized SurvivalData.
5. docs/technical-notes/: extracted standalone examples, provenance and math links.

## Invariants and scope

No source API, dependencies, tests, mathematical appendices, example code or
results changed. SurvivalData representation remains distinct from implemented
likelihood support. RMST restrictions and PH extreme-scaling limits remain in
the main guides. Executable extractions have their own imports and setup.
MNIST page and tentative simulation-guide.md are byte-identical to 2093ca2.
Simulation stays excluded; unmerged Bayes v2 is not imported. Existing MNIST
notebook supplies interactive exploration after playground removal.

The entire approved restructure is implemented. Future Bayes/simulation merges
are separate maintainer work, not unfinished restructuring substeps.

## Validation

- `scripts/build_docs.sh`: final direct escalated run exited 0; both float32
  and float64 report `build succeeded.`
  Builds use fresh notebook execution and warnings-as-errors, including MNIST's
  existing float32 smoke example. 17 executable pages / 147 code cells per mode.
- `uv run ruff format --check .`: `165 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `git diff --check`: clean.
- `/tmp/audit_docs.py`: stdlib HTML audit of local href/src targets and fragment
  IDs. Both modes: 22 HTML pages (19 source pages plus generated indexes/search),
  1,744 local links/assets each, zero failures. All 19 rendered source copies
  per mode match the current source files byte-for-byte.
- Native Chrome rendered inspection: overview grouped navigation and executed
  example; Working with distributions method/formula table; piecewise PH
  hazard/survival figure and explanation; RMST support/derivative table; mathematical-appendix download table.
  Visual inspection sampled float32 pages; both modes received the complete
  HTML link/source audit, not a claim of visual inspection of every page.
  These checks found no clipping, broken mathematical rendering or layout issue.
- No numerical regression gate, wheel installation suite or R regeneration run:
  no library/packaging/example implementation changed. Existing executable
  documentation checks are exercised by the full build; no new unit tests.

Earlier attempts: sandboxed Jupyter kernels hit psutil/sysctl shutdown permission
errors; interrupted and reran with approved escalation. First completed float32
pass executed every page but failed warnings-as-errors for four H1-to-H3 heading
jumps in the extracted parameter-conventions page. Fixed those to H2. A second
accidental sandbox invocation was interrupted immediately, then the final direct
script invocation ran outside the sandbox. No warning suppression added.
The UI tool also stalled while acquiring Chrome; no build success was inferred
from that delay. Browser inspection resumed once the tool returned.

## Commit trace

Previous step / starting HEAD / original worktree base: 2093ca2 (MNIST.20).
Archive-only commit: bfa2a3a. Final implementation commit follows this record.
The maintainer's untracked docsstructure.md remains untouched and unstaged.

An optional output-free notebook assertion failed: the existing MNIST notebook
contains saved outputs. Its bytes match 2093ca2 exactly; these are pre-existing
and untouched. No notebook cleanup or result regeneration was performed.

Final staged diff check caught extra EOF blank lines in two newly tracked notes
that the unstaged check had not included. Removed in a whitespace-only follow-up
to cbd2f02; no executable cells or rendered content changed, so no build repeat.
The exact-source audit above precedes this trailing-whitespace cleanup.
