# A2 review — constructors, boundaries, and data diagnostics

Branch: `a2-constructors`. This handoff covers A2a, A2b, and the independent
review fix-up. Stop after this fix-up commit for maintainer review.

## Read in this order

1. `PLAN.md`, especially “A2 independent-review fix-up” and the two A2 steps;
   `DEVIATIONS.md` for decisions still open.
2. `src/chronox_jax/_data_validation.py`: input dtype/shape rules, raw
   indicators, `row_issues`, shared flag normalization, and the eager host
   checkpoint.
3. `src/chronox_jax/data.py`: the eager flag and bounds constructors, pure
   `with_issues`, six-leaf PyTree, and constructor-specific traced errors.
4. `src/chronox_jax/diagnostics.py`: bit decoding, `uninformative`, `ok`,
   PyTree reconstruction, and explicit host checkpoint.
5. `tests/unit/test_data_construction.py`, `test_data_boundaries.py`,
   `test_data_validation.py`, `test_data_issues.py`, and `test_data_dtype.py`.
   The independent A2 and targeted A3 diagnostic tests are read-only.

## Invariants and requirement-to-test map

- Spec 3.1: raw indicators must be exactly 0/1 before kind selection; NaN,
  fractional codes, and other codes are invalid. `event` with any supplied
  flag is static misuse and raises `TypeError` in both flag entry points
  (D-006, maintainer decision). Own construction tests cover these and
  row-wise true-flag conflicts. Inactive `time2` is ignored (D-001).
- Spec 3.2–3.3: the first time/lower input fixes all six leaf shapes and the
  common floating dtype. Exact times are finite and positive, censored first
  endpoints finite and nonnegative, `0 <= L < U`, and exact membership is
  `L < t <= U`. Own dtype, representation, and boundary tests cover these,
  including left censoring at `+inf` and caller-site float64 warnings.
- Spec 2/6.2: endpoint presence and domain precede interval ordering;
  independent faults may overlap. A right censor at `c=L` is valid and
  uninformative; `c<L` is invalid. Censored sets must intersect `(L,U]`, while
  sets covering the whole window are uninformative outside `invalid_rows`.
  Direct `row_issues` tests and the own diagnostic tests cover the bitfield,
  every rule class, the finite `upper=U` boundary, overlap, and scalar `ok`.
  The new bounds NaN-upper test checks `missing_time2` specifically.
- Spec 6.2–6.3: eager constructors report counts and first indices; the pure
  flag form returns data and masks under `jit`/`vmap`. `DataIssues` is a PyTree
  with ten rule masks, `uninformative`, `invalid_rows`, and scalar `ok`.
  Unit and targeted independent diagnostic tests cover these transformations.
  A traced bounds error now says `with_issues` accepts flag inputs only.

## Checks run

Commands used `UV_CACHE_DIR=/private/tmp/uv-cache` because the sandbox cannot
write the default uv cache.

- `uv run ruff format --check .`: `19 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 4 source files`.
- `scripts/run_tests.sh`: x64 `65 passed, 3 skipped`; x32 `68 passed`.
- `scripts/check.sh --through A2`: guard OK; same static and own-test results;
  independent x64 `179 passed, 13 skipped, 109 deselected`; independent x32
  `187 passed, 5 skipped, 109 deselected`; `check: all gates passed through A2`.
- Targeted `tests/spec/test_a3_data_issues.py` with `SPEC_THROUGH=A3`:
  `81 passed` in x64 and `81 passed` in x32.
- `git diff --check`: passed with no output.

## Open items and limits

- D-007 / review F-2: ordinary constructors still reject concrete observation
  inputs captured inside `jit`. I recommend accepting them, while rejecting
  traced inputs, but the maintainer has not decided. No F-2 behavior changed.
- D-008–D-009 / S-2–S-3: the maintainer will update the normative spec before
  a pure bounds API or a bounds unsupported-keyword change is implemented.
- D-001 and D-002 remain formally open. D-006 records the maintainer's S-1
  decision but remains formally open for resolution under the working
  agreement. D-003–D-005 are already resolved in spec revision 5.
- The review's N-1–N-3 and N-5 are optional cleanup nits left for a separate
  small step, especially N-3 after the pure bounds API is specified.
- The four new own tests target the review's M3–M6 mutation gaps. I did not
  rerun the mutation harness. The full A3 gate was not run because indexing
  and static metadata are A3 scope and remain unimplemented.
