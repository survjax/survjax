# Independent alignment review

Date: 2026-11-06. Reviewer: fresh Codex Astra session (`gpt-6-astra`),
reasoning setting `low`, as requested. This report is independent of the
implementation author's conclusions.

Reviewed HEAD: `91a45034ad430173da3b76c40ab814cdfcb8fa3f`.
Inspection range: `23e8c3ef033e9c6c6e93c3fde21356626923c684..91a45034ad430173da3b76c40ab814cdfcb8fa3f`.
Included implementation `e8e9a51`, documentation `a4c8b42`, design example
cleanup `54bbc62e3880dbbe5544cf913fb95291e78904f9`, merged acceptance
corrections and final alignment records. Current normative specification is
revision 47 at design `213cfb0b7167b9b833e7fae8898de50ae2e83321`, which exactly
matches `.spec-seen`.

## Findings and assessment

**No actionable issue found.** No blocker or should-fix finding emerged in
the reviewed change set.

- `src/survjax/distributions/wrappers.py`: removing the positional adapter
  correctly exposes the existing registered-family constructor. Inspection of
  `src/survjax/distributions/base.py:450` confirms it rejects positional
  arguments before initialization, normalizes own parameter leaves with the
  caller-warning stack level, validates nested registered distributions, and
  broadcasts leaves. This matches spec 3.3, 4.2 and 4.4 without adding an API
  or changing PH numerical formulas. Rejection and warning-attribution tests
  cover the affected behavior.
- `tests/spec/test_a11_ph_parameter_issues.py:136`: the corrected condition
  mask is consistent with the independently inspected predicate
  `self.rate > self.cap` in `tests/spec/spec_families.py:490`. The third
  comparison is `1 > -1`; its field violation does not suppress its condition
  violation under spec 4.7.
- `tests/spec/test_a11_ph_direct_methods.py:419`: the corrected log-CDF
  reference comparison uses independent NumPy reference survival to identify
  the active dtype's normal range. It retains full-grid direct/grid agreement,
  shape checks and original relative tolerances in the guaranteed range,
  including every entry in float64. This follows spec 4.2 and does not hide
  an in-range accuracy failure.
- Changed documentation and executable examples consistently use keyword
  construction. Independent AST comparison found no other Python semantic
  changes after normalizing this spelling. Design commit `54bbc62` likewise
  repairs examples rather than changing requirements.
- Scope is appropriately narrow. No dependencies or numerical algorithms
  changed. Final records distinguish alignment completion from the full
  merge/release and demo gates. Open formal deviation statuses remain with
  the maintainer rather than being silently closed.

## Checks actually run

Read repository `AGENTS.md`, `PLAN.md`, `REVIEW.md`, deviation records, current
normative specification and relevant implementation/test context; inspected
the complete changed-file diff and design cleanup diff. Confirmed HEAD and
spec marker using Git. Initial working tree was clean.

Ran `scripts/check.sh --fast --through A12` independently; exit status **0**:

| Check | Actual output |
| --- | --- |
| Protected-path guard | `guard: ok (spectests-rmst)` |
| Ruff format | `173 files already formatted` |
| Ruff lint | `All checks passed!` |
| Mypy | `Success: no issues found in 22 source files` |
| Implementation, x64 | 1059 passed, 8 skipped, 13.94s |
| Implementation, x32 | 1064 passed, 3 skipped, 15.35s |
| Source doctests, x64 | 4 passed, 0.46s |
| Source doctests, x32 | 4 passed, 0.43s |
| Acceptance through A12, x64 | 3397 passed, 32 skipped, 2 warnings, 34.96s |
| Acceptance through A12, x32 | 3391 passed, 38 skipped, 2 warnings, 36.73s |

The gate built its wheel and used its normal installation-test setup. Its
final output was `check: all gates passed through A12 (fast: slow tests skipped)`.
Both warning pairs are NumPy reference `log(0)` and zero-times-infinity at
`tests/spec/test_a11_ph_direct_methods.py:78`, not assertion failures.

Also ran `git diff --check 23e8c3e 91a4503`: passed without output. A read-only
Python AST comparison of all changed docs/example files passed for **46 code
blocks across 7 files**, after normalizing positional PH baseline arguments
to keywords and excluding cell metadata/magic lines. The first attempt failed
to parse a MyST `:tags:` directive; the corrected parser excluded such lines
and completed successfully. Inspected constructor occurrences in docs and
examples with `rg`.

## Limitations and verdict

This is a review of the alignment delta, not a fresh audit of every existing
numerical algorithm or a release certification. The fast gate deliberately
omits slow-listed tests. I did not independently rerun the full Tier 2 gate,
build/render documentation, run NUTS or regenerate R fixtures. Previous
reported build/full touched-file results are not presented here as my own
verification. No repository source, tests, spec, plans, review records or Git
state were edited; this report is the only intentional file change.

**Verdict: merge** for the reviewed alignment changes, subject to the
repository's normal maintainer-run Tier 2 merge gate. No fix is requested by
this review; broader release and demo obligations remain separate.
