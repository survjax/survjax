# Slice A, step A1 — SurvivalData representation and dtype policy

Status: **A1 and its review fix-up implemented on `a1-survivaldata`; awaiting maintainer review**.
The maintainer approved one A1 commit despite the code-and-test diff exceeding
the preferred roughly 300 lines, because splitting the coupled constructor
and dtype work would leave the A1 acceptance gate failing between commits.

## Scope and step boundary

Build the normalized `SurvivalData` storage object: five public observation-shaped
fields, one private `int32` issue-bitfield leaf, immutable PyTree reconstruction,
and the time/bound dtype policy. The A1 acceptance tests construct data through
both public constructors, so this step must also provide their valid-input paths
and row-wise normalization. A2 will add complete indicator, endpoint, and
truncation validation, descriptive eager errors, and issue-bit setting. A3 will
add indexing/slicing and value-derived static metadata. Until A3, auxiliary
metadata will conservatively say every kind/truncation may be present.

## Files and public API

- `src/chronox_jax/data.py` (new): frozen, typed `SurvivalData`; the specified
  flag and `from_bounds` signatures for valid inputs; shape-preserving
  normalization; common floating dtype selection; six-leaf PyTree flattening
  and reconstruction that bypasses public construction. Keep normalization
  helpers private in this module.
- `src/chronox_jax/__init__.py`: export `SurvivalData` and list it in `__all__`.
- `tests/conftest.py` (new): read the dtype mode from the process environment
  for A1 tests; do not modify JAX's global configuration.
- `tests/unit/test_data_representation.py` (new): check each normalized kind,
  absent and supplied truncation, scalar and multidimensional shapes, six leaf
  dtypes/shapes, immutability, PyTree round-trip, `jit`, `vmap`, and bypass of
  normalization on unflatten. Expected field values are direct consequences of
  the table in spec 3.2.
- `tests/unit/test_data_dtype.py` (new): check integer/default, float16 and
  bfloat16, mixed float/int and float32/float64 promotion, explicit dtype,
  float64 conversion and caller-site warning in 32-bit mode, and rejection of
  boolean, complex, string and object time/bound arrays.
- `tests/unit/test_import.py`: update the A0 smoke test to expect the newly
  exported public name.
- `DEVIATIONS.md`: record D-001 concerning inactive `time2` entries in mixed
  observations; leave it open for maintainer resolution.
- `REVIEW.md`: after implementation, overwrite with reading order, invariants,
  requirement-to-test mapping, exact check results, and uncertainties.

No dependency or configuration changes are planned. `tests/spec/`, `spec/`,
`AGENTS.md`, and `CLAUDE.md` remain untouched.

## Implementation invariants

- The first time/lower input fixes the observation shape; other arrays may
  broadcast to it but never enlarge it. All six leaves retain that shape.
- `exact` is bool; all four time/bound fields share one JAX-promoted floating
  dtype of at least float32; the issue leaf is `int32` and initially zero for
  valid inputs. Integer-only inputs use JAX's configured default float dtype.
- In 32-bit mode, implicit float64 input converts once with one warning at the
  caller's line; explicitly requesting float64 raises. Library code never
  changes `jax_enable_x64` or converts tracers to NumPy.
- PyTree unflattening sets stored fields directly, without running the public
  constructor. Static metadata is conservative until A3.
- Inactive `time2` values are ignored during normalization, pending D-001.

## Specification and checks

Primary sections: 3.2 (fields, shapes, storage, PyTree) and 3.3 (dtype policy).
The valid-input normalization path also uses the signatures and kind meanings
from 3.1. Section 10.2 governs the public export; section 12.2 defines Slice A.

After implementation, run `uv run ruff format --check .`, `uv run ruff check .`,
`uv run mypy src`, `scripts/run_tests.sh`, and
`scripts/check.sh --through A1`. Report each exact result in `REVIEW.md`.
If the step exceeds roughly 300 handwritten lines including tests, stop and
propose a narrower split before committing.

## A1 review fix-up (maintainer requested 2026-09-25)

The independent review was found at
`~/dev/chronox-dev-review/.review/a.md` (`a1.md` does not exist there).
The maintainer's instruction to implement it authorizes this fix-up on the
same branch as a separate commit. Keep the existing A1 public constructor
signatures and export unchanged.

Files to change:

- `src/chronox_jax/data.py`: follow JAX weak typing for bare Python scalars;
  give lists/tuples JAX's default array dtype; warn only for typed float64
  inputs in 32-bit mode; disable dataclass array equality; simplify field
  assignment; document the conservative metadata tuple's positions.
- `tests/unit/test_data_dtype.py`: cover weak scalar bounds in both modes,
  omitted versus explicitly default truncation, list and scalar defaults,
  typed NumPy float64 scalars, and one warning for several float64 fields.
- `tests/unit/test_data_representation.py`: reject a field that enlarges the
  observation shape; normalize an exact bounds row whose supplied upper is
  NaN to `upper == lower`.
- `DEVIATIONS.md`: add D-002 for the maintainer-decided weak typing and warning
  rules pending the proposed section 3.3 spec change. Leave status open.
- `REVIEW.md`: overwrite with the fix-up reading order, invariants, test map,
  exact results, and remaining A2/A3 boundaries.

Spec coverage: 3.2 (shape and normalized exact bounds), 3.3 (JAX promotion and
warning policy), 10.2 (unchanged export), and 12.2 (A1 gate). Run Ruff format
and lint, mypy, `scripts/run_tests.sh`, and `scripts/check.sh --through A1` in
both configured dtype processes. No dependencies or protected paths change.
