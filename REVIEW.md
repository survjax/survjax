Independent review: round 2/3

# A10.3 — Public concordance transformations and permutation evidence

Approved after the detailed PLAN.md. Work on maintainer-created `a10b` from
`a5cfea7`. Two new test modules, now 260 lines; no production/API changes.
The previous integration review is retained at `git show a5cfea7:REVIEW.md`.

## Reading order

1. tests/unit/test_concordance_transformations.py: two independently
   enumerated tied-time cohorts crossed with three score draws; then failed
   cohorts and a raw-input compile counter.
2. tests/property/test_concordance.py: three reproducible public identities,
   each with 40 Hypothesis examples (small tied integer cohorts, sizes 1–8).
3. PLAN.md's A10.3 approval, scope, validation and later-work record.
   Existing metrics.py and _concordance.py supply background only; unchanged.

## Invariants and requirement mapping

- Spec 6.1, 9.1–9.2, 12.1: the parametrized crossed-cohort test checks raw
  and prepared input forms against hand-enumerated expectations. Data
  `(2,1,4)` and scores `(3,4)` give `(2,3)`, so dataset and draw axes cannot
  accidentally align. Eager, jit, nested vmap and jit-of-vmap check each
  value, all eight uint32 count words, all three boolean masks and shapes.
- Spec 6.2–6.3, 9.1–9.2, 12.1: the transformed-failures test constructs
  observations inside each evaluator. Five cohorts distinguish valid,
  invalid raw-event code, left truncation, both failures and all censored.
  Check both score orders under all four evaluators. Failed cohorts retain
  overlapping failure masks and zero counts; only evaluated zero-pair
  cohorts set no_comparable_pairs. Valid neighbors retain all three pairs.
- Spec 6.1/7.2: the raw compile-count test checks composition with raw
  construction inside a trace; it does not exercise host-derived metadata.
- Spec 3.2/7.2/12.1: the prepared-data compile-count test passes two
  host-constructed SurvivalData arguments with different times/events/scores
  but the same exact/right-censoring flags. Both hand-counted outputs are
  checked and one trace is required, exercising static metadata reuse.
  A third, all-exact cohort changes the flags and must produce a second trace.
- Spec 9.2, 12.1: generated joint row permutations preserve the complete
  value/info result, including NaNs. Python integer reconstruction of the
  public hi/lo words checks the count partition and the zero-pair mask.
  Score negation swaps concordant/discordant words, preserves ties, totals
  and masks, and complements the value whenever pairs exist. Undefined
  values stay NaN. Inputs and expectations use no production numerical
  helpers and no independent-test helpers.

Counts and masks compare exactly; non-exact fractions use only existing
ATOL/RTOL from tests/impl_tolerances.py. No new tolerances or score-gradient
claims. Test dtype comes from the process environment; library configuration
is unchanged. No spec ambiguities encountered; no new deviation needed.

## Checks actually run for the initial implementation, fc7f82c

All commands from dev-a10b with
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Versions checked: Python 3.12.9,
JAX/jaxlib 0.11.2, NumPy 2.5.3; import resolves to dev-a10b/src/survjax.
No dependency edits or environment-version matrix claim.

Focused command, separately with `JAX_ENABLE_X64=1` and `=0`:
`uv run --group test pytest -q tests/unit/test_concordance_counts.py tests/unit/test_concordance_results.py tests/unit/test_concordance_entry_points.py tests/unit/test_concordance_transformations.py tests/property/test_concordance.py`.
**72 passed in 21.99s** (float64), **72 passed in 22.07s** (float32), no skips.
The final naming-only lint correction is also included in the full gate.

`uv run ruff format --check .`: **92 files already formatted**.
`uv run ruff check .`: **All checks passed!**
`uv run mypy src`: **Success: no issues found in 20 source files**.
Initial default-cache uv calls exited 2 because the sandbox blocked the user
cache. Retried formatting outside the sandbox with the temporary cache;
subsequent checks use that cache. Initial lint flagged E741 for a test
argument `l`; renamed it to `left_trunc`, then reran lint successfully.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check_wheel.sh --through A9`
exited **0**, ending **check: all gates passed through A9**. The unchanged
runner built a fresh wheel and used six workers for own/independent suites.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own suite, including installed wheel | 1043 passed, 7 skipped, 62.65s | 1049 passed, 1 skipped, 56.33s |
| Source doctests | 4 passed, 1.09s | 4 passed, 0.99s |
| Independent through A9 | 2211 passed, 28 skipped, 61.89s | 2202 passed, 37 skipped, 50.55s |

Retained wheel:
`/private/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.TXntu5/survjax-0.0.1.dev0-py3-none-any.whl`.
SHA-256: `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9`.
Full optional log: `/private/tmp/survjax-a103-wheel-gate.log`.

Selected independent A10 command, separately with `JAX_ENABLE_X64=1/0`
and `SPEC_THROUGH=A10`:
`uv run --group test pytest -q -n 6 tests/spec/test_a10_concordance_api.py tests/spec/test_a10_concordance_values.py tests/spec/test_a10_concordance_diagnostics.py tests/spec/test_a10_concordance_shapes.py tests/spec/test_a10_concordance_properties.py`.
**113 passed in 22.55s** (float64); **112 passed, 1 skipped in 30.07s**
(float32). Existing precision skip only; no skips/xfails added. These are
the five non-fixture modules, not the full A10 gate. Optional logs:
`/private/tmp/survjax-a103-acceptance-x64.log` and
`/private/tmp/survjax-a103-acceptance-x32.log`.

## Round 1 finding and follow-up for round 2

Read round 1 of `../review/slices/A10.3.md` for fc7f82c: **fix-first**,
one should-fix **F-1**, no blockers or nits. Claude independently verified
the hand-counted cohorts statically but could not run gates, focused tests
or mutants because its headless Bash permissions refused those commands.
No reviewer settings or permissions were changed by the implementer.

**F-1 addressed:** add a prepared-data compilation test with two equal-flag
mixed exact/right cohorts whose values differ, then an all-exact cohort with
different flags. Check all hand-counted results and trace counts 1 then 2.
Retain the raw test as tracing/composition coverage and correct its docstring
and requirement mapping; it is not static-metadata evidence.

Reran the exact five-module focused command above on the final follow-up,
separately in both modes: **73 passed in 12.16s** (float64),
**73 passed in 12.22s** (float32), no skips. Ruff format: **92 files already
formatted**; lint: **All checks passed!** The earlier full wheel/A9 gate,
A10 selection and mypy results apply to fc7f82c; they were not rerun for
this test-only follow-up. Production code, property tests and all existing
checks remain unchanged. No mutation check was run by the implementer.

## Scope and limits

Only the two new own-test modules, PLAN.md and this review may change.
Spec, .spec-seen, independent tests, scripts, dependencies, production code,
fixtures, docs and other branches remain unchanged. `git diff --cached
--check` exits 0 for the initial commit. Its staged path set was exactly the
four authorized files. The follow-up changes only the unit test and the
two handoff records; the staged path set matches those three paths and
`git diff --cached --check` exits 0. No unstaged changes.

A10.4's concordance R fixture, executable topic/playground and held-out
example remain separate work. No full A10, docs build, R generation,
performance measurement or complete Slice A claim. Round 2 is pending;
invoke the exact skill command after the fix-up commit, then stop according
to its verdict. No open uncertainty about this test-only change.
