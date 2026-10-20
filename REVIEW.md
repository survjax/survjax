# A6.3 review fix-up — Mask batch shapes

Status: approved fix-up implemented on `A6`, based on `b0b44a5`.
Addresses S1/S2 in `../review/slices/A6.3.md`; stop after this commit.
Spec revision 30 and independent snapshot f527f75 are unchanged.

## Read first

1. `src/survjax/distributions/_parameter_issues.py`: own mask storage.
2. `tests/unit/test_parameter_issues.py`: rebuilt-own and per-leaf vmap tests,
   then the strengthened rebuilt-nested test.
3. PLAN.md: approved fix-up scope and the optional findings left unchanged.

## Invariants and test/spec mapping

- Spec 4.7 requires every field/condition mask to have current batch_shape.
  Construction usually broadcasts leaves, but reconstruction and per-leaf
  vmap axes can give own leaves different broadcast-compatible batch shapes.
  Own masks now broadcast after component reduction, before storage, matching
  nested masks. Union broadcasting alone did not enforce individual shapes.
- `test_rebuilt_smaller_own_batch_broadcasts_field_masks` checks eager/jit
  leaves with batch shapes (1,) and (3,): both masks have shape (3,), rate is
  all True and cap all False, from the declared positive constraints.
- `test_per_leaf_vmap_broadcasts_mapped_and_shared_field_masks` maps rate while
  sharing cap: every per-example mask has shape (3,), stacked as (3,3).
  Its explicit expected rows follow rate [-1,2,3] and positive cap [1,1,1].
- `test_rebuilt_smaller_nested_batch_broadcasts_diagnostics` now asserts (3,)
  separately for nested fields and conditions before checking their values.
  NumPy's scalar/vector equality shortcut cannot hide missing broadcasts.
- API, keys, polarity, component reductions and aggregate logic are unchanged.
  Assertions use exact boolean values and shapes; no tolerance is involved.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Separately with JAX_ENABLE_X64=1 and 0, SPEC_THROUGH=A6:
  `uv run --locked --group test pytest tests/unit/test_parameter_issues.py tests/unit/test_parameter_issue_container.py tests/unit/test_parameter_constraints.py tests/spec/test_a6_constraints.py tests/spec/test_a6_parameter_issues.py -q --tb=short`.
  Each exit 0: 176 passed, 1 skipped in 4.54s. The existing optional NumPyro
  interoperability check skips because NumPyro is absent.
- Separately with JAX_ENABLE_X64=1 and 0:
  `uv run --locked python /private/tmp/survjax_a63_broadcast_mutations.py`.
  Each exit 0: four baseline regression checks pass. Removing own broadcasting
  is rejected by eager, jit and per-leaf vmap shape assertions. Removing nested
  field broadcasting fails its field shape assertion; removing nested condition
  broadcasting fails its condition shape assertion. All three mutants are
  compiled in memory and the class binding is restored; no source file changes.
- `scripts/check.sh --through A5`: exit 0; guard ok (A6); Ruff format:
  59 files already formatted; lint: All checks passed!; mypy:
  Success: no issues found in 15 source files.
  Own x64: 534 passed, 7 skipped (45.63s); x32: 540 passed, 1 skipped
  (45.31s). Source doctests: 4 passed per mode (1.06s/1.14s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 308 deselected (29.56s);
  x32: 1261 passed, 21 skipped, 308 deselected (24.04s).
  Final output: "check: all gates passed through A5".
- `git diff --check`: exit 0, no output before delivery.

## Limits and uncertainties

No new spec ambiguity identified; DEVIATIONS.md unchanged. N1's optional
nonboolean-constraint validation is deferred: such results violate the current
protocol, and this fix-up adds no new error/coercion policy. N2's pre-existing
base.py length is unchanged; no registration refactor.
No docs/spec/independent-test/dependency/export edits. No docs build (docs and
public docstrings unchanged), complete A6 gate, R, install or benchmark run.
No A6.4 work; D-023 remains the later likelihood step's open question.
