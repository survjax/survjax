# Review: A3 fix-ups after `791517a`

Branch: `A3`. Implements handoff items 1–8 in an extra commit.

## Read first

1. `_data_metadata.py`: standard pytree tracer detection, right-before-left
   precedence, device reductions and one transfer of six static booleans.
2. `data.py`: bounds issue calculations now inside compile-time evaluation;
   indexing docstring explains JAX's out-of-range clamping.
3. `tests/unit/test_data_metadata.py`: new edge cases and transformation evidence.
4. `DEVIATIONS.md`, D-012: overlap of kind predicates; status remains open.

## Invariants and evidence

- Concrete inputs: both constructors evaluate all six stored leaves at trace
  time inside jit and match eager construction. Any traced observation input
  forces all flags true; calculations involving those inputs remain traced.
  Tests: `test_concrete_construction_inside_jit_has_no_traced_leaves`,
  extended `test_concrete_inputs_inside_jit_keep_eager_flags`, and
  `test_vmap_construction_has_all_presence_flags` (both forms).
- Diagnostics decoded inside jit remain traced, even for concretely built
  data. The checkpoint raises host-only `ValueError`, not `InvalidDataError`.
  Test: `test_concrete_data_checkpoint_inside_jit_remains_host_only` (both forms).
- Spec 3.3 gives `(0, +inf]` right precedence. Test:
  `test_unbounded_zero_lower_is_right_censored_only`; existing normalized-kind
  tests continue to cover other kinds and rows with issues.
- Presence flags are not validity checks. All normalized rows participate;
  NaN truncation and negative entry can leave truncation flags false while
  issue bits report invalidity. Tests:
  `test_nan_truncation_is_invalid_despite_false_presence_flags` and
  `test_negative_entry_is_invalid_despite_false_presence_flags` (3.2/6.2).
- Concrete metadata uses jnp reductions on device and one `device_get` of
  six booleans, inside compile-time evaluation only. Static Python flags need
  one host synchronization. Slices inherit flags without metadata synchronization.
  Behavior is covered by existing and new metadata tests; no accelerator
  timing or transfer profiling was performed.

## A6 handoff

- Reuse normalized kind definitions from `_data_metadata.py` for row dispatch
  so static flags and likelihood classification agree. `may_contain` returns
  Python flags through a host transfer; it is not a jit-safe row-mask API.
  Plan any shared pure-JAX classifier at A6; none is added in this fix-up.
- Issue bits must cause NaN contributions and `invalid_data` regardless of
  which branches metadata removes. A6 must test NaN truncation with false
  truncation flags.
- Resolve unsupported-mode detection on traced data in the A6 plan. All-true
  metadata cannot establish that unsupported modes are actually present;
  address row values and explicit failure under transformations (12.2).

## Checks

All uv commands use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- Focused metadata tests, x64: **23 passed in 2.65s**.
- Ruff: **All checks passed!**; mypy: **Success: no issues found in 5 source files**.
- Initial mypy check reported an incompatible fixed-length return annotation
  for the tuple comprehension. Updated the helper annotation to
  `tuple[bool, ...]`; subsequent mypy checks passed.
- `scripts/check.sh --through A3`: **check: all gates passed through A3**.
  Guard passed; **30 files already formatted**; Ruff and mypy passed.
  Own tests: x64 **117 passed, 3 skipped in 6.37s**; x32 **120 passed in
  5.33s**. Independent tests: x64 **372 passed, 15 skipped in 8.39s**;
  x32 **382 passed, 5 skipped in 8.27s**. Own x64 skips cover float32-only
  conversion warnings and unavailable-dtype behavior.
- `git diff --check`: passed with no output.

## Open items

D-012 and D-010 remain open for maintainer resolution. No tests/spec or spec
files changed. Existing notebook edits and the maintainer's handoff are excluded
from staging. No R, docs build, install smoke test or CPU timing was run.
