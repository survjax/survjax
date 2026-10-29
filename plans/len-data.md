<!-- Moved verbatim from PLAN.md lines 1612-1721 (dev commit a5cfea7) on 2026-10-29. -->

# SurvivalData size interface — revision 37 (2026-10-26)

Status: **implemented and verified; ready for review**. Approval: "ok i switched
branches you do it". The maintainer-created branch is `len_data`; the only
pre-existing change was this plan. Planning started on clean `main`.
This entry is the active plan; older entries below are historical.
One code step, one commit, then a review stop. Target under 300 changed
source/test lines.

## Contract and current evidence

- Spec section 3.2 (revision 37) adds `data.shape`, `len(data)` and an explicit
  prohibition on truth testing. Sections 3.2 and 12.1 cover the shape,
  indexing, PyTree and transformation checks for this step.
- `shape` is a tuple of Python ints obtained from stored leaves, static under
  jit and per-example under vmap. It must not enter PyTree auxiliary data.
- `len(data)` counts the leading axis, including zero for empty data. Scalar
  data raise TypeError. Multidimensional length is not the total element count.
- `bool(data)` raises TypeError for every shape, including scalar and empty
  data; implementing this is required to prevent Python falling back to len.
  The contract adds neither `size` nor `ndim`.
- At planning time `src/survjax/data.py` had indexing and PyTree reconstruction but none
  of these three interfaces. Existing indexing and metadata tests provide
  regression coverage for the six leaves and inherited may-contain flags.
- `.spec-seen` already records `be26e50cc35db1c2affc3dafcb4bf4cf4baf1ebf`;
  `../scripts/spec-since.sh dev` reports `spec unchanged since be26e50`.
  No spec or marker change is needed. No contract ambiguity was found.

## Implementation after approval

Public API touched: `SurvivalData.shape`, `SurvivalData.__len__` and
`SurvivalData.__bool__`. No new package exports.

- `src/survjax/data.py`: add a read-only `shape` property annotated
  `tuple[int, ...]`, returning `self.lower.shape`; add `__len__ -> int` with
  an explicit scalar TypeError and leading-axis result; add `__bool__ -> bool`
  that always raises TypeError with a clear truth-value message. Document
  shapes, return values and errors in their docstrings and mention the size
  interface in the class summary. Keep construction, leaves, auxiliary data
  and reconstruction unchanged. No new helper or dependency is needed.
- Create `tests/unit/test_data_size.py` with small cases and spec 3.2
  docstrings. Expected shapes and lengths come directly from explicit input
  dimensions, independently of the new methods:
  - Both constructors: scalar `()`, vector `(3,)`, empty `(0,)` and matrix
    `(2, 3)`; verify shape elements and eager lengths are Python ints, scalar
    len raises TypeError, and matrix len is 2 rather than 6. Include `(2, 0)`
    to distinguish a nonempty leading axis from zero total elements.
  - Integer indexing, slices and tree-map slicing: shape and len follow the
    selected leaves; integer selection of a vector produces scalar data.
  - Eager/jit queries agree; inspect shape types inside tracing because jit
    converts returned Python integers to scalar arrays. Under vmap over a
    matrix, the body sees `(3,)` and len 3; over a vector, it sees `()` and
    scalar len raises TypeError. Include jit-of-vmap coverage.
  - Truth testing raises TypeError for scalar, empty, vector and matrix data,
    eagerly and while tracing under jit/vmap. This includes `if data:`.
  - Tree flatten/unflatten preserves the six leaves and auxiliary flags;
    differently shaped slices retain the parent's tree structure. Assert
    `size` and `ndim` are absent. No derivative checks apply to this static
    shape interface; no tolerances are needed.
- Update this PLAN.md with actual evidence and overwrite REVIEW.md with
  reading order, invariants, spec-to-test mapping, exact checks and doubts.
  Keep `tests/spec/`, the spec, docs and protected process files unchanged.

## Verification and completion

1. Run the new size tests plus existing data indexing, representation,
   metadata and tracing tests in separate float64/float32 processes:
   `JAX_ENABLE_X64=1` and `=0`, using `uv run pytest`.
2. Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
   on the final review snapshot. This runs Ruff, mypy, own tests, source
   doctests and independent acceptance tests through the implemented gate in
   both dtypes. Do not edit, skip or xfail independent tests if they fail.
3. Run `git diff --check`; report exact results, failures and skips. Tests
   and static checks have not been run during planning. No R checks are
   needed for a shape-only interface.
4. Commit code, tests, PLAN.md and REVIEW.md together on the maintainer-created
   branch with an imperative subject and rationale. Stop for review; do not
   start another step or claim Slice A completion.


### Implementation evidence

Implemented the three interfaces and 22 new own test cases. Source/test diff:
34 added source lines and 139 test lines, below the 300-line budget.
Focused own tests passed in both processes: 65 passed in 5.31s (float64),
65 passed in 4.30s (float32), no skips or failures.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
exited 0: `check: all gates passed through A7`. Ruff: 72 files already
formatted / All checks passed! Mypy: no issues found in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 774 passed, 7 skipped, 157.24s | 780 passed, 1 skipped, 150.46s |
| Source doctests | 4 passed, 0.95s | 4 passed, 1.03s |
| Independent through A7 | 2114 passed, 28 skipped, 218 deselected, 103.10s | 2118 passed, 24 skipped, 218 deselected, 102.64s |

Full log: `/private/tmp/survjax-len-data-gate.log`. No new skips or exclusions;
independent tests were not modified. No R or docs build was run. REVIEW.md
records invariants and the test mapping. Final Ruff/diff checks follow this
record update; commit on `len_data` and stop for review.

### Docs to-do

A later approved docs step can explain shape, leading-axis len, scalar errors
and explicit truth-value rejection in the observation-data guide. Public
API docstrings ship with this code step; do not edit docs/ here.

---

