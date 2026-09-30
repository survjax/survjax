# Specification revision 9 — ordinary constructors and diagnostic property

Status: completed under the maintainer's follow-up request on `spec-constructor`;
ready for review.

- `spec/SPECIFICATION.md`: remove both issue-returning convenience methods
  from the public API, metadata/validation wording, acceptance criteria and
  Slice A deliverable. Document migration to ordinary construction followed by
  `data.issues`; bump the revision and mark prior decisions as superseded.
- `PLAN.md` and `REVIEW.md`: record this documentation fix-up and checks.

Public API: only `SurvivalData(...)` and `SurvivalData.from_bounds(...)` construct
data; diagnostics use `data.issues`. Spec coverage: 3.1–3.2, 6.2, 10.2,
11.4, 12.1–12.2 and 13–15. No implementation, dependencies or independent
tests change; no executable tests are added. Existing examples already use the
retained API. Verify all active references, parse the example blocks, and run
`git diff --check`. The previous A1 gate covers unchanged code; do not repeat
it or claim it verifies this new API. Commit this fix-up and stop for review.

## Previous revision plan (historical; constructor surface superseded by revision 9)

# Specification revision 8 — pure SurvivalData construction

Status: documentation revision completed; ready for maintainer review on the
existing `spec-constructor` branch. This request authorizes editing
the specification for this step despite the usual implementation-only rule.

Files and public contract:

- `spec/SPECIFICATION.md`: revise sections 3, 6, 10–15 for pure ordinary and
  convenience constructors, the `data.issues` property, explicit host raising,
  unchanged static misuse and likelihood rules, metadata, compatibility, and
  acceptance evidence. Add strict and simulation workflows with an illustrative
  compiled Newton solve using second parameter derivatives.
- `DEVIATIONS.md`: record any remaining exception-lifecycle question as open.
- `PLAN.md` and `REVIEW.md`: record scope, reading order, invariants, checks,
  and implementation/test follow-up. Retain previous plans as history.

Supporting user guidance belongs beside the normative API and in section 11.4;
no new documentation framework is needed. No implementation, dependencies,
independent tests, or project goals change. No executable tests are added.
This is one coherent documentation step; the contract replacement and example
may exceed the approximate 300-line target and will be reviewed as one revision.

Verification: audit constructor/validation references, parse the illustrative
Python, check Newton usage against primary Optimistix documentation, run
`git diff --check` and the existing `scripts/check.sh --through A1` gate.
The current implementation cannot execute the future-contract example; do not
claim that this gate verifies revision 8. Commit this step, write `REVIEW.md`,
and stop for review.

## Previous plans (historical; superseded where revision 8 changes validation)

# Metadata step — define Slice A steps for the blind spec tester

Status: approved by the maintainer in the request to create a new branch and
make the proposed role-contract changes. Branch: `meta-spectester-steps`.

## Files and scope

- `roles/TESTER.md`: give the blind tester the A0–A9 delivery schedule that is
  currently available only in `CLAUDE.md`. Correct the reference to spec 12.2,
  which defines slices but does not number Slice A steps. State how to assign
  step markers when a test needs an API introduced in a later step.
- `PLAN.md`: record this documentation step.
- `REVIEW.md`: record the reading order, schedule invariants, checks, and the
  separate correction needed in the independent test repository.

No public API, implementation, dependencies, or executable tests change. Do
not edit `tests/spec/` on this branch; the tester must correct its Q-001 step
assumption, markers, and test map on a `spectests-*` branch.

## Contract and evidence

- `CLAUDE.md`, Slice A suggested step order: source of the A0–A9 schedule.
- Specification sections 3.1–3.3, 4.1–4.2, 4.5, 5, 6.2–6.3, 7,
  7.1–7.3, 10.1–10.4, 12.1–12.2: requirements assigned to steps; the
  specification remains normative for test assertions.
- The A1 plan refines its boundary: valid constructor paths may support A1
  representation and dtype tests; full constructor validation and `DataIssues`
  belong to A2.

Verification: inspect the role diff against the step table, run
`git diff --check`, and run the current `scripts/check.sh --through A1` gate.
This is documentation only, so no new test is justified.

## Previous A1 plan (retained for history)

# Slice A, step A1 — SurvivalData representation and dtype policy

Status: **A1 and its round-two review fix-up implemented on `a1-survivaldata`; awaiting maintainer review**.
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

## A1 maintainer docstring fix-up (requested 2026-09-25)

The maintainer requested clearer constructor documentation, including the
relationship to R's `survival::Surv`. This review instruction authorizes a
separate documentation-only fix-up commit on `a1-survivaldata`.

Files to change:

- `src/chronox_jax/data.py`: expand the class, flag-constructor and
  `from_bounds` docstrings with argument meanings, shape/broadcast rules,
  endpoint and truncation conventions, short examples, and the precise mapping
  to R's `Surv(time, event)`, `type="interval"`, and `type="interval2"`.
- `REVIEW.md`: overwrite with the reading order, documentation invariants,
  exact checks, and any uncertainty.
- `PLAN.md`: record this fix-up and its public API boundary.

Public API touched: documentation of `SurvivalData(...)` and
`SurvivalData.from_bounds(...)`; signatures and runtime behavior remain the
same. Spec coverage: 3.1 (constructor semantics), 3.2 (stored fields and
shapes), 3.3 (bounds and dtype), and 10.2 (public export). No new tests are
needed for documentation-only changes. Run Ruff format/lint, mypy, and
`scripts/check.sh --through A1` before the fix-up commit.

## A1 round-two review fix-up (requested 2026-09-25)

The independent re-review is at `~/dev/chronox-dev-review/.review/a-r2.md`.
Fix R1 now rather than carry it into A3. This is a small additional commit on
`a1-survivaldata`; public constructor signatures and exports do not change.

Files to change:

- `src/chronox_jax/data.py`: preserve JAX's `weak_type` for `jax.Array` inputs,
  including tracers, when selecting a common dtype; continue rejecting
  nonnumeric time dtypes and warning only for strong typed float64 inputs.
- `tests/unit/test_data_dtype.py`: add float32 time plus eager weak JAX bound,
  a `jit`ted Python-float bound, and Python-bool scalar rejection.
- `DEVIATIONS.md`: update D-002 to cite the landed section 3.3 spec commit
  `3041ebd`, while leaving its status `open` for maintainer resolution.
- `REVIEW.md`: overwrite with the fix-up reading order, invariants, exact
  checks, and remaining A2/A3 work.
- `PLAN.md`: record this fix-up.

Spec coverage: section 3.3 JAX promotion and bool-time rejection, plus 3.2
observation storage. Run Ruff format/lint, mypy, both dtype modes of own tests,
and `scripts/check.sh --through A1`. Do not edit the protected spec or spec
tests; the review's section 4.2 formatting note is for a future `spec-*` branch.
