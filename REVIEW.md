# A2-R1 review — exception contract

Branch: `a2-constructors`. This commit implements only A2-R1 from `PLAN.md`.
The separately modified `PLAN.md` is left unstaged; it was present before this
implementation. A2-R2 and A2-R3 remain for later approval and commits.

## Read first

1. `src/chronox_jax/diagnostics.py` and `__init__.py`: the explicit checkpoint
   now raises exported `InvalidDataError(ValueError)` for concrete invalid data.
2. `src/chronox_jax/_data_validation.py`: malformed shapes, including JAX
   broadcast errors, become `TypeError`.
3. `src/chronox_jax/data.py`: the removed outside-interval keyword follows the
   unknown-keyword `TypeError` path.
4. Changed tests in `tests/unit/`: exception type, shape, keyword and tracing
   cases. The implementation and test diff is 83 insertions, 21 deletions.

## Invariants and evidence

- Observation-value issues still occupy the same ten per-row bits; no
  normalization rule or data leaf changed. The explicit host checkpoint counts
  each rule and reports the first offending index. The issue tests cover
  invalid, valid and uninformative diagnostics.
- Shape checks use only array shapes, so malformed calls raise under tracing.
  Construction tests cover enlarging and incompatible time/bounds/indicator
  shapes, including flag and bounds misuse under `jit`.
- Event-plus-flags, nonnumeric bounds/indicators and unknown keywords are
  argument misuse. Construction tests cover these. The dtype test shows an
  unavailable explicit float64 request stays `ValueError`, outside the new
  `InvalidDataError` class. Import test checks the public export.

## Checks

- `uv run ruff format --check .` (with `UV_CACHE_DIR=/private/tmp/chronox-uv-cache`):
  `19 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 4 source files`.
- `scripts/run_tests.sh`: float64 `73 passed, 3 skipped`; float32 `76 passed`.
- `scripts/check.sh --through A2`: guard, static checks and both own-test modes
  passed. The float64 independent acceptance run ended with `55 failed, 123
  passed, 13 skipped, 189 deselected`; float32 independent tests were not run
  because the gate stopped. These failures are the still-automatic ordinary
  constructor checks: revision 9 requires invalid rows to return data and
  diagnostics. That migration is A2-R2.
- `git diff --check`: passed with no output.

## Uncertainties and next step

No new spec ambiguity arose. The traced host-checkpoint exception subtype and
legacy diagnostic compatibility remain open as recorded in `PLAN.md` and
`DEVIATIONS.md`; this step preserves the traced-checkpoint behavior. Ordinary
constructors still raise automatically and reject traced observations until
A2-R2. `with_issues` remains temporarily until A2-R3. Do not treat this
intermediate checkpoint as completion of A2.
