# A2b review — pure data diagnostics

Branch: `a2-constructors`. A2b is complete; stop here for maintainer review
before A3 indexing and static metadata.

## Read in this order

1. The A2b scope in `PLAN.md`.
2. `src/chronox_jax/_data_validation.py`: `flag_fields` now supplies the same
   normalized fields and issue bits to eager and pure construction.
3. `src/chronox_jax/data.py`: eager construction keeps its host checkpoint;
   `SurvivalData.with_issues(...)` returns data and diagnostics without it.
4. `src/chronox_jax/diagnostics.py`: frozen `DataIssues`, bit decoding,
   `uninformative`, scalar `ok`, PyTree handling, and host checkpoint.
5. `tests/unit/test_data_issues.py` and the read-only independent
   `tests/spec/test_a3_data_issues.py`.

## Invariants and test map

- Both flag paths use `flag_fields`, so the five stored arrays and the int32
  issue leaf normalize identically. The valid-input parity unit test and
  independent eager-versus-pure tests check this (spec 3.2/6.2).
- Invalid numeric rows remain data plus issue bits under `with_issues`;
  nonnumeric inputs and impossible shapes still raise. The structural and
  invalid-row unit tests, plus the independent diagnostic suite, check this
  distinction (spec 6.2).
- The ten masks decode independent bits, `invalid_rows` is their union, and
  `ok` reduces over observation axes. `uninformative` is a separate valid
  mask for censored sets covering the full truncation window. Unit tests cover
  overlapping faults, endpoint attribution, truncation boundaries, and full
  window coverage; the independent suite covers every named rule (spec 2/6.2).
- `DataIssues` is a PyTree whose per-row masks match the observation shape and
  whose `ok` value is scalar before batching. Unit and independent `jit`/`vmap`
  tests check this (spec 6.2–6.3). `raise_if_invalid()` is an explicit host
  checkpoint; unit tests cover invalid, valid, and traced calls.
- The A1 float64 warning still points to the caller after adding the shared
  normalization call. The existing dtype tests catch this in 32-bit mode.

## Checks run

Commands used `UV_CACHE_DIR=/private/tmp/uv-cache` because the sandbox cannot
write the default uv cache.

- `uv run ruff format --check .`: `19 files already formatted`.
- `uv run ruff check .`: `All checks passed!`.
- `uv run mypy src`: `Success: no issues found in 4 source files`.
- `scripts/run_tests.sh`: x64 `59 passed, 3 skipped`; x32 `62 passed`.
- `scripts/check.sh --through A2`: guard OK, same static/own-test results;
  independent x64 `179 passed, 13 skipped, 109 deselected`; independent x32
  `187 passed, 5 skipped, 109 deselected`; `check: all gates passed through A2`.
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A3 uv run pytest
  tests/spec/test_a3_data_issues.py -q`: `81 passed`.
- The same targeted command with `JAX_ENABLE_X64=0`: `81 passed`.
- `git diff --check`: passed with no output.

## Open items

No new spec ambiguity arose. D-001 remains formally open in `DEVIATIONS.md`.
The full `--through A3` gate was not run: A3 indexing and static metadata are
still unimplemented. The textual diff exceeds the roughly 300-line review
target because the shared normalization was moved out of `data.py` and a new
diagnostic container and tests were added in one coupled API step. Each source
module remains under 300 lines.
