# A8.d1 review fix-up — S-1 and S-2

Approved by "yes do it" on 2026-10-29 after reviewing
../review/slices/A8.d1.md for 726896d. Branch: a8-1-r-generator.

## Read in this order

1. docs/builtin-families.md, final paragraph: S-1 replaces the maintainer
   handoff/status paragraph with the acceptance-schema copy and the invariant
   that ordinary regeneration never writes into tests/spec/.
2. docs/survival-data.md, Comparison with R: S-2 retains the syntax comparison
   and links to numerical evidence, scoped to exact/right without truncation.
3. PLAN.md: approved fix-up and result record.

## Invariants and scope

Spec 10.4–10.5, 11.1 and 12.1: user-facing documentation matches the committed
reference evidence. No unimplemented likelihood or broader R-agreement claim.
Pending protected-fixture delivery remains a development-record concern:
through-A8 is still unverified; this commit does not change that status.

Only these two prose edits and plan/review records change. No public API,
source, tests, fixtures, generators, dependencies, tolerances, spec or protected
paths change. Optional reviewer nits N-1/N-2 remain outside this approved fix.
No new tests are appropriate for two prose edits; existing executable docs
provide the build check.

## Actual checks

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh` exited 0:
build succeeded in separate float32 and float64 processes, fresh notebook
execution with Sphinx warnings fatal. Escalated execution was used for local
Jupyter kernel port binding, as required by the previously observed sandbox
restriction. Full log: /private/tmp/survjax-a8d1-fix-docs.log.

| Changed page execution | Float32 | Float64 |
| --- | --- | --- |
| builtin-families.md | 2.68s | 2.44s |
| survival-data.md | 2.76s | 2.49s |

The build executed all existing runnable docs pages. No new code examples;
R regeneration commands were not run. The log includes the existing local
IPKernelApp TCP-without-encryption notices; no Sphinx build warning/error.

`python3 /private/tmp/check_a8d1_fix_html.py` passed in both modes:
corrected paragraphs present, obsolete claims/status absent, new cross-page
anchor resolves, all 85 local links/anchors across both changed pages resolve.
Checks ran after both final HTML trees completed.

Browser inventory returned no available connector browsers. No new Safari
interaction was attempted after the concurrent-user conflict recorded in the
previous review. Full visual layout inspection remains unverified; final
HTML text, links and structure were checked directly in both modes.

Final checks with the same UV_CACHE_DIR:
- uv run ruff format --check .: 77 files already formatted.
- uv run ruff check .: All checks passed!
- git diff --check: exit 0, no output.

No full code gate, numerical-suite rerun, R regeneration or new deviations.
Stop after this fix-up commit for review.
