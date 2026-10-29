# Review: A10 integration with main/A9

Maintainer-initiated merge: A10-cindex f71140a with main ff9786f.
Requested: check merge conflicts and bring A10 up to date after A9 merged.
This review replaces the two conflicting handoff records. Their full content
remains at `git show f71140a:REVIEW.md` and `git show ff9786f:REVIEW.md`.
Both branches' planning records are retained in PLAN.md below the current
integration record. No new feature or spec decision is introduced.

## Reading order

1. src/survjax/__init__.py and tests/unit/test_import.py: current exports
   retain A10's two metrics and main's absence of top-level log_likelihood.
   Keep main's regression test that imports cannot restore the removed API.
2. src/survjax/likelihood.py, distributions/base.py, then
   _observation_inputs.py: one shared input resolver, the incoming private
   numerical evaluator and main's public distribution method.
3. tests/install/test_wheel.py: the existing strict export assertion now
   includes the two reviewed concordance functions; all installed checks stay.
4. PLAN.md's integration record and the unchanged incoming A8/A9 tests,
   fixtures, wheel runner, aligned docs and spec-test sync.

## Resolution invariants and scope

The five conflicts were PLAN.md, REVIEW.md, __init__.py, likelihood.py and
the own import smoke test. Main removed the standalone likelihood wrapper;
A10 moved input normalization and introduced the public metrics. Resolve by
keeping the method-only likelihood API and both metrics, with exactly one
_resolve_data implementation in _observation_inputs.py. The method imports
it directly. Keep every incoming numerical evaluator function unchanged.
Spec 6.1, 7.1-7.3, 9.1-9.2 and 10.2 govern these choices.

The two additional integration edits are the distribution method's import
and the installed wheel's exact expected export set. Only these seven paths
may differ from the initial conflict/index snapshot. Preserve all other
173 stage-0 entries byte-for-byte in the index. Incoming protected tests,
scripts, dependencies, fixture data, docs and .spec-seen must match main;
the API reference retains only the previously approved metric additions.
Concordance code and its own tests, including N-5/N-8 fixes, are unchanged.
The withdrawn standalone A9 example and its tests must stay absent.

The aligned spec marker be26e50 and incoming fixtures supersede the earlier
A10 records' missing A8/A9 baseline and spec-alignment blockers. They do not
prove combined-tree acceptance until the checks below finish. No R fixture
is generated and no independent acceptance file is edited.

## Checks actually run

Working directory dev-a10; UV_CACHE_DIR=/private/tmp/survjax-uv-cache.
`uv sync --locked --offline --dev`: succeeded using the incoming lockfile;
installed its execnet 2.1.2 and pytest-xdist 3.8.0. Runtime versions remain
Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. No dependency files changed
beyond main's existing staged versions. Ruff reformatted one resolution file.
Final static checks: **89 files already formatted**, **All checks passed!**;
mypy: **Success: no issues found in 20 source files**.

Focused own command, separately with JAX_ENABLE_X64=1 and =0:
`uv run --group test pytest -q tests/unit/test_observation_inputs.py tests/unit/test_likelihood_entry_points.py tests/unit/test_concordance_counts.py tests/unit/test_concordance_results.py tests/unit/test_concordance_entry_points.py tests/unit/test_import.py tests/unit/test_api_reference.py`.
**120 passed in 12.49s** (float64), **120 passed in 11.93s** (float32).
These cover shared input forms, method-only import behavior, counts/results,
shape and failure isolation, score precision, jit/vmap and API coverage.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check_wheel.sh --through A9`
exited **0**, ending **check: all gates passed through A9**. The unmodified
incoming runner builds a fresh wheel, supplies it to both own/independent
installation tests, and uses the default six workers for numerical suites.

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Own suite | 1036 passed, 7 skipped, 56.62s | 1042 passed, 1 skipped, 59.24s |
| Source doctests | 4 passed, 1.09s | 4 passed, 1.10s |
| Independent through A9 | 2211 passed, 28 skipped, 60.17s | 2202 passed, 37 skipped, 53.97s |

This includes incoming A8 fixture/reference checks and A9 compiled
composition, derivative, installed core and external typed-family checks.
It does not imply completion of the withdrawn/replacement example,
A9.4 performance or A9.d1 documentation work.

Retained wheel:
`/private/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.qky0dw/survjax-0.0.1.dev0-py3-none-any.whl`.
SHA-256: `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9`.
Optional full output: `/private/tmp/survjax-a10-main-a9-gate.log`.

A10 focused independent command, separately with JAX_ENABLE_X64=1/0 and
SPEC_THROUGH=A10:
`uv run --group test pytest -q -n 6 tests/spec/test_a10_concordance_api.py tests/spec/test_a10_concordance_values.py tests/spec/test_a10_concordance_diagnostics.py tests/spec/test_a10_concordance_shapes.py tests/spec/test_a10_concordance_properties.py`.
**113 passed in 28.26s** (float64); **112 passed, 1 skipped in 21.81s**
(float32). Existing skip requires float64 scores beside float32 data.
No independent test edits, skip additions or xfails. This is the five
non-fixture modules, not a full A10 gate. Optional logs:
`/private/tmp/survjax-a10-main-acceptance-x64.log` and
`/private/tmp/survjax-a10-main-acceptance-x32.log`.

## Merge audit

All conflict markers and unmerged index entries are gone; the worktree
matches the staged resolutions. AST comparison confirms the incoming
likelihood module is unchanged except removal of its moved resolver and
unused ArrayLike import. That resolver's AST matches _observation_inputs.py;
there is exactly one implementation. The method's import is the only base
change relative to main. Main's extra method-only import regression is kept.

All **173 other initial stage-0 index entries** retain their original
mode/blob identity. Protected tests/scripts, spec marker, dependencies,
fixtures and references equal main; docs differ from main only by the seven
previously approved API-reference lines. Both branches' PLAN history is
retained, including the A9 standalone example withdrawal. The withdrawn
example and its tests remain absent. HEAD f71140a and MERGE_HEAD/main
ff9786f are unchanged; no merge commit, rebase or new merge was performed.

`git diff --cached MERGE_HEAD --check` and resolution-only whitespace checks
are clean. The full staged `git diff --cached --check` exits 2 solely for
incoming EOF blank lines in tests/spec/API_ASSUMPTIONS.md:644 and
tests/spec/test_a6_likelihood_shapes.py:121. `git diff HEAD MERGE_HEAD --check`
reports the identical two warnings. Both read-only files are byte-identical
to main and deliberately preserved; these are not resolution regressions.

The imported spec marker remains be26e50 (revision 37). Design HEAD 3c9ce10
has subsequent use-case companion-link changes in sections 11/12 and the
header, with no likelihood/concordance contract change. No marker sync or
spec edit was made here. Final metadata restaging/index audit follows;
no source or test changes followed the passing runs.

Leave this resolved/staged merge for the maintainer's final commit under
AGENTS.md's "never merge" rule. Do not start the next A10 subslice.

## Remaining limits

The concordance R fixture, narrative topic/playground/held-out example and
full A10 gate remain separate work. No R regeneration, docs build, CPU
benchmark, accelerator result, A9.4/A9.d1 completion, or complete Slice A
claim is implied. S-2 and formal deviation resolution remain maintainer-owned.
