# A2a review — validated eager constructors

Branch: `a2-constructors`. This is the A2a step in `PLAN.md`; stop here for
maintainer review before A2b.

## Read in this order

1. `PLAN.md`, especially “Proposed review steps” and “Settled A2 decisions”.
2. `src/chronox_jax/_data_validation.py`: dtype/shape helpers, raw indicator
   checks, the ten issue bits, and eager error summaries.
3. `src/chronox_jax/data.py`: both public constructors now normalize, assemble
   issue bits, and reject invalid concrete data. PyTree structure remains six
   leaves.
4. `tests/unit/test_data_construction.py`, `tests/unit/test_data_boundaries.py`,
   and the updated traced dtype test in `tests/unit/test_data_dtype.py`.

## Invariants

- Raw indicators are checked for exactly 0/1 before selecting kinds. Supplying
  `event` with any censoring flag sets `conflicting_indicators` on every row.
- The first time/lower input fixes the shape. Only active interval `time2`
  values are endpoints (D-001). All time and truncation arrays share the A1
  dtype policy.
- Endpoint presence and domains precede interval order. Malformed endpoints
  do not acquire `invalid_interval`; independent problems may overlap.
- Exact membership is `L < t <= U`. A normalized right-censored row with
  `c=L` is valid; `c<L` is invalid. Censored sets must intersect `(L,U]`.
- Eager construction reports counts and first indices. Traced ordinary
  construction raises at trace time and points to the planned `with_issues`.
  Tree reconstruction still bypasses validation.

## Requirements and evidence

- Spec 3.1: event/flag codes and conflicts, active `time2`, exact bounds upper,
  mixed-kind normalization, and unsupported outside-interval mode are checked
  by `test_data_construction.py` and independent A2 constructor tests.
- Spec 3.2/3.3: shape, dtype, endpoints, truncation, and observation-window
  rules are checked by existing A1 tests, `test_data_boundaries.py`, and
  independent A2 boundary tests.
- Spec 6.2: issue-bit assembly, descriptive eager errors, and traced-call
  rejection are checked by the new unit tests. Full `DataIssues` decoding and
  `with_issues` behavior belong to A2b.
- Spec 12.1: both-constructor parity is covered by the independent A2 tests,
  including their reproducible property test.

## Checks run

All commands used `UV_CACHE_DIR=/private/tmp/uv-cache` because the sandbox
cannot write the default uv cache.

- `uv run ruff format --check .`: `16 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 3 source files`.
- `scripts/run_tests.sh`: x64 `39 passed, 3 skipped`; x32 `42 passed`.
- `scripts/check.sh --through A2`: guard OK; same static and own-test results;
  independent x64 `179 passed, 13 skipped, 109 deselected`; independent x32
  `187 passed, 5 skipped, 109 deselected`; `check: all gates passed through A2`.
- `git diff --check`: passed with no output.

## Open items

The A2a commit does not include `DataIssues` or `SurvivalData.with_issues`;
they are the separate A2b review step. D-001 remains formally open in
`DEVIATIONS.md`; no new spec ambiguity arose in this step.

The textual diff exceeds the roughly 300-line review target: it moves the
existing A1 dtype and shape helpers into the private validation module to
keep `data.py` at 286 lines, and it replaces the A1 plan and review handoff.
The new behavior is confined to eager constructor validation and its tests.
