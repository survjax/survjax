# Slice A, step A1 — SurvivalData representation and dtype policy

Status: **implemented and committed on `a1-survivaldata`; awaiting maintainer review**.
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
