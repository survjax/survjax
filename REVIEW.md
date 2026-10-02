# A4.1 review fix-up

Implements `../review/slices/A4.1.md` F-1, N-1, N-2 and its template cleanup.

## Reading order

1. `src/survjax/_dtypes.py`: explicit None rejection before dtype selection;
   warning noun supplied by the caller.
2. `src/survjax/_data_validation.py`: supplies "time" to preserve observation
   warning text. No warning stack levels change.
3. `tests/unit/test_parameter_dtype.py`: None rejection and mixed float32/
   float64 warning coverage.
4. `DEVIATIONS.md`: remove the obsolete D-001 format example, retaining the
   real entry. No deviation statuses change.

## Invariants and tests

- Spec 3.3/4.2: None is not an omitted parameter. The nonreal-parameter test
  requires TypeError with the field name, in both dtype modes. Observation
  inputs still allow None to mean omitted.
- Spec 3.3: one float64 leaf must trigger the x32 warning even when another
  leaf is float32. The warning test checks count, filename and line, then
  confirms all own leaves are float32. This catches an any-to-all regression.
- Promotion, shapes, weak typing and tracing behavior are unchanged. The
  existing parameter/data tests and prior acceptance suite all pass.
- Warning wording no longer depends on comparison with an input-kind string;
  callers explicitly provide the noun. No public API changes.

## Checks actually run

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- `scripts/check.sh --through A3`: **all gates passed through A3**.
  Guard passed; **29 files already formatted**; Ruff **All checks passed!**;
  mypy **Success: no issues found in 6 source files**.
  Own x64: **136 passed, 4 skipped in 9.85s**.
  Own x32: **139 passed, 1 skipped in 7.99s**.
  Independent x64: **372 passed, 15 skipped, 203 deselected in 12.96s**.
  Independent x32: **382 passed, 5 skipped, 203 deselected in 11.00s**.
  Own skips remain dtype-specific: x64 excludes x32 warning/unavailable-dtype
  cases; x32 excludes the parameter float64 promotion case.
- `scripts/build_docs.sh`: **build succeeded** in x32 and x64, executing both
  documentation notebooks per mode. Ran with approved escalation for the
  Jupyter local kernel ports. Kernel startup printed its TCP encryption warning.
- `git diff --check`: passed with no output.

## Remaining scope

No unresolved questions from these findings. No spec or tests/spec edits.
The pre-existing D-013 planning entry remains unstaged; only the obsolete
format-template deletion is included in this commit. A4.2 constructor warning
attribution and nested dtype integration still require their later step tests.
The 203 A4 acceptance tests remain deselected by the prescribed A3 gate.
No R, install smoke test, performance baseline or derivative suite was run.
