# Review: A10.1 — Private concordance pair counts

## What to read, in order

1. `src/survjax/_concordance.py`: `_add_count`, then `_pair_counts` and its
   event-time ordering, partner-index mask and four reductions.
2. `tests/unit/test_concordance_counts.py`: hand-enumerated mixed cohorts,
   tie cases, permutations, transformations and word-carry arithmetic.
3. The A10 plan at the top of `PLAN.md`; D-027 in `DEVIATIONS.md` describes
   public integration questions left for a later step.

Authorized by "Approved go for it" on maintainer branch `A10-cindex`.
This step adds a private core and tests only: no public metric entry point,
ratio, result container or export yet. Source plus tests total 230 lines.
Existing A7 docs edits remain uncommitted and are excluded from this commit.
The A10 plan is staged separately from the existing A7 docs plan insertion.

## Invariants and spec coverage

- Inputs to the private core are aligned one-dimensional finite scores/times
  and boolean exact/right status. Validation and unsupported-mode handling
  are integration work; callers must not pass unchecked cohorts here.
- Spec 9.2: each unordered pair is visited only with partner index greater
  than row index. The earlier time must be an event; at equal times, only
  event/censor pairs qualify and the event precedes censoring. Higher earlier
  event score is concordant. Equal scores are counted separately for later
  half-credit calculation. No tie tolerance, time repair or clipping.
- Spec 9.1: each row's reduction explicitly uses int32. Totals are uint32
  `(hi, lo)` scalar arrays, representing `hi * 2**32 + lo`, with low-word
  wrap detected by unsigned comparison and carried to the high word. No
  int64 availability or floating count approximation is assumed. With cohort
  length representable by int32, the pair total fits comfortably in two words.
- Spec 9.2: a JAX loop holds only cohort-sized masks and four scalar word
  pairs. Work is O(N squared), pair workspace O(N). This is established by
  reading the implementation; no memory benchmark or scaling claim was run.
- The static empty-shape return avoids tracing invalid indexing into an empty
  array. It defines private count behavior only, not the public empty-input API.
- No NumPy conversion occurs inside the core or transformed functions. Host
  conversions in tests inspect completed counts. JAX configuration is unchanged.

The mixed-cohort test explicitly lists C/T/D/D pairs; the permutation test
lists all six pairs, including the excluded one. The three two-row score/tie
cases in both orders cover event/censor time ties; separate same-status tests
exclude event/event and censor/censor ties. Earlier censoring is excluded in
both orders. Complete-event cases demonstrate perfect, reversed and all-tied
scores. Adjacent representable floats demonstrate exact time/score comparison.
Empty, singleton and all-censored cases have zero counts. These cover spec 9.2.

All 24 permutations of a mixed-time/score-tie cohort preserve its hand-counted
totals. Eager, jit, vmap and jit-of-vmap results match independent pair lists
for two separate cohorts (spec 9.1/12.1). Carry tests compare against Python
integer arithmetic beyond signed-int32 capacity, across low-word wrap,
repeated carries and existing high words (spec 9.1). All comparisons are exact;
no numerical tolerance or expensive quadratic large-cohort test is needed.

## Checks actually run before the full gate

- Initial uv invocation failed before executing checks because its default
  cache was sandbox-blocked. Subsequent commands use
  `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.
- Initial focused Ruff check found B023 on a loop-index lambda; bound the
  index explicitly and reran successfully. No numerical test failure occurred.
- Focused Ruff format/check passed; mypy reported
  `Success: no issues found in 18 source files`.
- After adding the adjacent-float case, ran
  `JAX_ENABLE_X64=1 UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run --group test pytest -q tests/unit/test_concordance_counts.py`:
  **24 passed in 1.37s**. Same command with `JAX_ENABLE_X64=0`:
  **24 passed in 1.34s**. No skips.

## Final gate record

The full gate runs against the finalized source/tests:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`.
Its full output is saved in
[the A10.1 gate log](/private/tmp/survjax-a101-final-gate.log).
The observed exit status, counts and timings are recorded in PLAN.md after
completion. The line-count note in this review was corrected while the gate ran;
REVIEW.md is left unchanged after completion. Final formatting and both
working-tree/staged diff checks validate all review/plan metadata edits.

## Remaining work and uncertainties

D-027 remains open: public exact-count representation, field names, masks and
undefined input behavior need maintainer resolution before A10.2. This private
core does not settle those public choices. Existing acceptance tests stop at
A8; no independent A10 tests are synced and this through-A7 run is regression
evidence, not a full A10 or Slice A completion gate. No spec edits or edits to
independent tests/gate scripts. No score derivatives promised or tested.

No R generation/comparison, installed-artifact check, docs build, performance
benchmark or accelerator check was run. The remaining public metric, ratio,
validation/broadcasting, diagnostics, R fixtures and executable docs belong to
later individually approved steps. Stop after this commit for review.
