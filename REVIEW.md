# Review: A10.2b2 follow-up — N-5 and N-8

## Reading order

1. `tests/unit/test_concordance_entry_points.py`: the new three-case
   `test_unsupported_bounds_cohort_isolated_from_valid_neighbor` (N-8).
2. `tests/unit/test_concordance_counts.py`: the three-line ClosedJaxpr
   descent in the existing workspace guard (N-5).
3. PLAN.md's current follow-up and results. The external review is
   `../review/slices/A10.md`, round 4 at `8109470`, verdict merge.

Test-only follow-up on A10-cindex from clean `8109470`, requested by
"check the nits" within the standing review-and-address instruction.
Only the two own test files, PLAN.md and this review change. No new API,
production logic, docs, fixtures, dependencies or spec interpretation.

## Invariants and requirement mapping

Spec 9.1/N-8: finite right truncation, interval observations and left
censoring are valid data outside concordance's supported observation modes.
Each new case constructs bounds explicitly and asserts `data.issues.ok`.
The unsupported cohort must give NaN, unsupported=True, invalid=False,
no_comparable_pairs=False and zero high/low words for all four counts.
A neighboring exact/right cohort remains valid. Its comparable pairs are
C,T,D,D, so the hand-derived counts are (1,2,1,4) and C=(1+1/2)/4=3/8.
Both eager and jit paths must obey these invariants. The existing own
left-truncation case remains. No rows are dropped and no expected value
comes from the implementation.

Spec 9.2/N-5: the workspace guard walks distinct ClosedJaxpr wrappers via
`.jaxpr`, in addition to direct Jaxpr, dict and sequence parameters. The
Jaxpr case stays first: locked JAX 0.11.2 aliases ClosedJaxpr to Jaxpr,
so testing ClosedJaxpr first could repeatedly visit the same Jaxpr.
The existing N=7 guard still checks loop bodies for N-by-N intermediates.
No workspace algorithm or numerical behavior changed.

## Checks actually run

Working directory dev-a10; UV_CACHE_DIR=/private/tmp/survjax-uv-cache.
Focused command, separately with JAX_ENABLE_X64=1 and =0:
`uv run --group test pytest -q tests/unit/test_concordance_entry_points.py tests/unit/test_concordance_results.py tests/unit/test_concordance_counts.py tests/unit/test_api_reference.py tests/unit/test_import.py`.
Float64: **67 passed in 6.56s**. Float32: **67 passed in 5.83s**.
No skips or failures; includes the three new cases and the workspace guard.

The round-4 review's 63-versus-64 difference is the file selection:
its four-file run omitted `test_api_reference.py`, which contributes one
test. The original five-file evidence was 64; these three cases make 67.
No historical count was changed.

Ruff reformatted one test file, left the other unchanged. Final commands:
`uv run ruff format --check .`: **78 files already formatted**;
`uv run ruff check .`: **All checks passed!**;
`uv run mypy src`: **Success: no issues found in 20 source files**.
`git diff --check`: clean. Final metadata/staged/scope checks follow.

Mutation checks used isolated copies of `src/` in /private/tmp and verified
that Python imported each copied metrics module. Each reported term in the
unsupported expression was replaced with False; the working source was
never modified. Ran only the new parametrized test via
`.venv/bin/python -m pytest -q tests/unit/test_concordance_entry_points.py -k test_unsupported_bounds_cohort_isolated_from_valid_neighbor`
in separate dtype processes:

| Mutation | Float64 | Float32 |
| --- | --- | --- |
| E1: disable right-truncation term | 1 failed, 2 passed, 23 deselected, 1.48s | 1 failed, 2 passed, 23 deselected, 1.25s |
| E2: disable interval/left-censor term | 2 failed, 1 passed, 23 deselected, 1.12s | 2 failed, 1 passed, 23 deselected, 1.16s |

Failures are exactly right-trunc for E1 and interval/left-censored for E2,
at the expected NaN-mask assertion. All four runs exited 1 as required.
Optional logs: `/private/tmp/survjax-a10-nits-olixhji5`; the table is durable.

## Limits and remaining work

The installed JAX aliases the Jaxpr classes. The distinct-wrapper branch is
not dynamically exercised, and other JAX versions were not run. No broader
version compatibility claim is made.

No full regression gate or independent tests were rerun for this test-only
follow-up. Production source is unchanged from `8109470`, whose complete
through-A7 gate passed in both dtype modes (recorded in PLAN.md).
The 38 added test lines are within the review budget.

No merge/rebase, spec/.spec-seen/sync, protected tests, likelihood, docs or
dependency changes. S-1/S-2 and D-027 disposition remain maintainer-owned.
R fixtures, narrative docs, installed-wheel evidence and the full Slice A
gate remain pending. Stop after this fix-up commit for review.
