# A5.6 — Base-owned boundary values (spec revision 27)

Implemented the approved code/tests/docstrings step on branch `A5.6`, based on
`9146251`. Step 2 (narrative docs) has not started. The source/test/docstring
change is 299 lines, including the new 83-line unit module; planning/review
records are additional. No spec sources, independent tests, docs pages,
dependencies, top-level exports or deviation statuses changed.

## Read first, in order

1. `src/survjax/distributions/_survival_methods.py:apply_policy`: direct-method
   names select the base-owned boundaries even over explicit/inherited
   overrides; the old raw primitive probes at infinity are gone.
2. `src/survjax/distributions/base.py:log_survival_at_infinity` and registration:
   proper default, own-parameter dtype (or nested-leaf promotion for a
   nested-only family), exactly batch_shape, no component axes. Public method
   docstrings now state the boundary ownership and exceptions.
3. `tests/unit/test_family_boundaries.py`: interior-only exponential and plain
   softplus log-logistic primitives, zero/infinity parameter gradients and
   finite grid-entry derivatives under debug_nans.
4. Own infinity/direct/batching test diffs: explicit cure hooks, changed
   override/NaN precedence, generic density's raw differentiation at zero,
   existing cure Hessians and nested dtype checks.

## Invariants and spec mapping

- Sections 4.2/4.5/7.2: one stopped reference input per direct call. Base-owned
  zero/infinity rows reach raw formulas only through finite positive
  substitutes for valid parameters. Stop no parameter or active hook value.
- Four survival/CDF/H methods select the base's zero constants and the
  infinity hook's values/derivatives. The hook is masked to a safe log value
  on other rows before boundary exp/log1mexp operations. Density at infinity
  is -inf; generic log hazard/hazard at infinity is NaN, including cure
  families. The hook is consulted by every direct method at infinity.
- NaNs from substituted family values and the infinity hook take precedence.
  No validity domain, floor, clipping or epsilon is introduced.
- Density/log-hazard overrides still receive zero; analytic hazard limits
  remain active at infinity. Generic defaults differentiate the raw primitive
  at zero without a new boundary guarantee. Existing raw override unwrapping,
  dtype promotion, batch broadcasting and grids remain in place.
- `test_family_boundaries` demonstrates sections 4.5/5/7.2 for primitives with
  unsafe boundary values or gradients, in eager, jit and vmap transformations.
  `test_family_mixed_infinity` retains mixed-entry Jacobians/Hessians and cure
  log-survival/H first and second derivatives; `test_family_infinity` checks
  cure survival/log-CDF first derivatives and override/NaN policy.
- Independent `test_a4_base_boundaries` covers hook shape/dtype, proper/cure
  values and gradients, unsafe primitives, NaN precedence and transformations
  (2/4.2/4.5/12.1/12.2). The through-A5 suite also covers the revision-25
  generic/built-in tail modules, dtype/dispatch regressions, built-in limits,
  second-order interior derivatives and time-first grids.

## Checks actually run

Commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Focused pytest used
`uv run --locked --group test pytest ... -q --tb=short` in separate processes
with `JAX_ENABLE_X64=1` and `0`, and `SPEC_THROUGH=A5`.

- First focused attempts exited 2 (four collection errors in each mode): the
  initial edit placed the hook inside a doctest. Corrected before evaluation.
- Focused boundary acceptance plus old direct/infinity modules then exited 1
  in each mode: 13 failed, 205 passed. All 84 boundary acceptance cases passed;
  failures were own assertions for the superseded contract, migrated here.
- Focused new/changed own modules plus boundary acceptance: 250 passed in
  each mode (11.22s x64, 11.48s x32). This run included a temporary standalone
  hook-dtype case subsequently folded into the existing nested-dtype test.
- A focused Ruff check found B023 on a loop closure; mypy found one reused
  local's Array/float assignment. Bound the loop variable and named the
  infinity constant separately; the full gate below verifies both fixes.
- `scripts/check.sh --through A5` exited 0, ending with
  `check: all gates passed through A5`. Guard passed on A5.6; Ruff reported
  53 files already formatted and all checks passed; mypy found no issues in
  13 source files. Own tests: x64 **463 passed, 7 skipped** (37.48s), x32
  **469 passed, 1 skipped** (38.20s). Source doctests: **4 passed** per mode.
  Independent tests: x64 **1256 passed, 26 skipped** (25.49s), x32
  **1261 passed, 21 skipped** (20.93s). Existing dtype-specific skips retained.
- Removed a redundant value assertion with default NumPy tolerances after the
  gate; its exact gradient assertion remains. Final focused verification is
  recorded below. No numerical source changed after the full gate.
- Final focused pytest of the generic zero-time density gradient and nested
  hook-dtype assertions: x64 **2 passed** (0.38s), x32 **2 passed** (0.36s),
  exit 0 in each process. `uv run ruff format --check .` exited 0,
  `53 files already formatted`; `git diff --check` exited 0 with no output.

## Remaining work and uncertainty

Narrative docs still describe primitive-owned boundaries and guarded cure/log-
logistic recipes; the approved second step will update those guides and the
API listing and run scripts/build_docs.sh. No docs build was run in this code
step. Built-in and grid docstrings were audited and required no ownership
correction. No new spec ambiguity was identified; D-016 closure stays with the
maintainer. This is through-A5 evidence, not a claim that Slice A is complete.
