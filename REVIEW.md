# A2 constructor API migration review

Branch: `a2-constructors`. This step completes the revision 9 A2 constructor
surface requested by the maintainer. A3 indexing and concrete metadata remain
separate work. The diff is larger than the usual 300-line target because the
public constructor change requires migrating existing tests in the same commit,
and the previously uncommitted plan has been replaced with a current handoff.

## Read first

1. `src/chronox_jax/data.py`: ordinary flag and bounds constructors now return
   data for invalid observation values; `data.issues` decodes the stored bits;
   the issue-returning convenience method is gone.
2. `src/chronox_jax/_data_validation.py`: the obsolete constructor host
   checkpoint is removed; normalization and issue calculations remain shared.
3. `tests/unit/test_data_construction.py`, `test_data_issues.py`,
   `test_data_boundaries.py` and `test_data_tracing.py`: own checks migrated to
   the public API, including traced inputs and tree reconstruction.
4. `DEVIATIONS.md` D-010: one focused A3 acceptance test conflicts with the
   specified scalar shape of `data.issues.ok`.

## Invariants and requirement evidence

- The data PyTree still has five normalized fields and one `int32` issue-bit
  leaf. `data.issues` adds no stored leaves. The property decodes raw-input
  faults after eager construction, `jit`, `vmap`, unflattening and `jax.tree.map`
  slicing; see `test_data_issues.py` and `test_data_tracing.py`.
- Invalid observation values set issue masks and do not raise during either
  constructor. Explicit concrete checkpoints raise `InvalidDataError` with
  counts and first indices; valid and uninformative rows pass. Construction,
  boundary and issue tests cover the hand-derived cases.
- Argument misuse still raises `TypeError`, including shape misuse under
  tracing. Explicit unavailable float64 requests still raise `ValueError`.
  Construction and dtype tests cover these exception distinctions.
- `SurvivalData.with_issues` and `from_bounds_with_issues` are absent. The only
  diagnostic accessor is the `data.issues` property. No own source or tests
  call a removed method. Historical deviation records retain their wording.

## Checks

- `scripts/check.sh --through A2` with `UV_CACHE_DIR=/private/tmp/chronox-uv-cache`:
  guard passed; Ruff format `20 files already formatted`; Ruff lint `All checks
  passed!`; mypy `Success: no issues found in 4 source files`; own tests float64
  `77 passed, 3 skipped`, float32 `80 passed`; independent A2 tests float64
  `178 passed, 13 skipped, 189 deselected`, float32 `186 passed, 5 skipped,
  189 deselected`; `check: all gates passed through A2`.
- Focused A3 independent tests (`test_a3_data_issues.py`,
  `test_a3_bounds_issues.py`, `test_a3_tracing.py`) in float64: `1 failed, 136
  passed, 2 skipped`. The failure is D-010; the test indexes the scalar
  `batched.issues.ok` as though it were a per-batch vector. Those files were
  not edited, skipped or xfailed. Float32 focused A3 tests were not run.
- `git diff --check`: passed with no output.

## Remaining questions

The A3 test mismatch awaits maintainer and independent test author review.
Spec section 13 still leaves the traced host-checkpoint exception subtype and
legacy diagnostic compatibility open. This step preserves explicit failure for
a traced host checkpoint and adds no compatibility alias. Concrete static
metadata, indexing and compile-count checks remain A3 work.
