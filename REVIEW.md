# A8.5 — Synthetic Weibull R reference comparisons

Approved by "implement A8.5, plan is approved"; branch a8-1-r-generator.
One reference-test step, then stop for review.

## Reading order

1. tests/reference/test_weibull_bc_like_r.py: AFT parameter construction,
   per-row comparisons, then the time-first grid and supplemental totals.
2. PLAN.md: approved A8.5 scope and separate A8.d1 docs topic.

## Invariants and requirement coverage

- Spec 3.3/10.4: all fixture access uses r_fixture. R expectations stay host
  float64; inputs and parameters use the process's active JAX dtype.
  Every numerical output has explicit shape and dtype assertions.
- Spec 4.6/11.1: for fitted, unit_shape and alternative, derive scales from
  baseline times exp(group coefficient), ordered Good/Medium/Poor. Both
  natural and from_log constructors are checked against the same R values.
  Scalar common shape is supplied to constructors, which broadcast it to
  match scales; vmap then slices both stored parameter arrays.
- Spec 4.7/6.2: fixture data and all row/group distributions have clear
  diagnostics. No library behavior, public API or tolerance changes.
- Spec 6.1/12.1: all 686 ordered likelihood contributions are compared with
  each R source, through function/method and raw/prepared entry points.
  Each runs eagerly, under jit with dynamic distributions, and row-wise vmap.
  Separate R package totals supplement the per-row comparisons.
- Spec 4.6/12.1: observed-time survival and log survival compare every row
  against both R sources under eager/jit/vmap, for both constructors.
- Spec 5/11.1: all (141, 3) survival-grid entries compare with both stored
  sources under eager/jit and vmap over time. Use stored R grid times to
  preserve their rounding. Grid labels come from the checked shared loader.

Numerical expectations are committed flexsurv/survreg outputs, never generated
by survjax. These related package paths are not fully independent oracles.
This step covers finite synthetic reference cases; A7/A9 own broader tail and
transformation/derivative evidence. No R regeneration, docs build, source,
loader, payload, protected-path, dependency or tolerance edits.

## Focused checks

With UV_CACHE_DIR=/private/tmp/survjax-uv-cache,
`uv run pytest -q tests/reference/test_weibull_bc_like_r.py`:

- JAX_ENABLE_X64=1: 102 passed in 8.34s.
- JAX_ENABLE_X64=0: 102 passed in 8.44s.

Logs: /private/tmp/survjax-a85-focused-x64.log and
/private/tmp/survjax-a85-focused-x32.log. An initial setup assertion expected
stored scalar shape; it was corrected to the constructor's broadcast shape
before these successful runs. No numerical assertion or tolerance changed.

Sensitivity control: temporary pytest plugin /private/tmp/survjax_a85_control.py
rolls public-function likelihood rows (preserving their sum) and reverses
survival-grid columns. No committed implementation or fixture is edited.
Both dtypes reject all 24 function-row and 12 grid cases: 36 expected failures,
66 passes (6.44s x64 / 6.42s x32). The remaining tests, including scalar totals,
pass. Logs: /private/tmp/survjax-a85-control-x64.log and
/private/tmp/survjax-a85-control-x32.log. These expected exit-1 mutation runs
are distinct from the passing unmodified suite.

## Gate and limits

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
exited 0: check: all gates passed through A7.
Full output: /private/tmp/survjax-a85-gate.log.

- Guard OK; Ruff: 77 files already formatted / All checks passed.
- mypy: no issues in 17 source files.
- Own float64: 982 passed, 7 skipped in 181.75s.
- Own float32: 988 passed, 1 skipped in 181.70s.
- Source doctests: 4 passed per dtype in 1.11s / 1.08s.
- Independent float64: 2123 passed, 28 skipped, 223 deselected in 122.71s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected in 113.45s.

Final Ruff formatting/lint and git diff --check also pass after result recording.
The protected tests/spec/fixtures/bc_synthetic_weibull.json is still absent,
so the through-A8 gate is pending durable maintainer delivery and sync.
No full A8 or Slice A completion claim. No new deviation.
Stop after this commit; A8.d1 remains a separate docs step.
