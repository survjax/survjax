# A4.1 review: shared floating-dtype normalization

## Read first

1. `src/survjax/_dtypes.py`: extracted promotion and input-kind checks, followed
   by the private parameter normalizer.
2. `src/survjax/_data_validation.py`: observation adapter preserving its errors,
   warning text and caller attribution.
3. `tests/unit/test_parameter_dtype.py`: independent dtype and transformation
   cases; existing `test_data_dtype.py` protects observation behavior.
4. `docs/survival-data.md`: shared precision policy, with registration marked
   as planned. `PLAN.md` records approval for A4.1 only.

## Invariants and spec coverage

- Spec 3.3: each family caller supplies only its own parameter leaves. Nested
  distributions are excluded by that caller and retain their own dtype; public
  integration is A4.2/A4.3 and is not verified by this step.
- Dtype inspection uses metadata, never host copies of traced values. Weak
  scalars participate in JAX promotion; integers/scalars alone use the configured
  default; strong floating inputs impose a float32 minimum. Component shapes
  are preserved. Tests cover mixed leaves, float16, float64 and weak inputs.
- Bool, complex, string and object inputs raise TypeError. Parameter domains
  remain unchecked (spec 4.1); a test preserves negative, NaN and infinite values.
- Conversion emits one warning per normalization call, located at the caller;
  Python's ordinary warning filter deduplicates repeated calls at that line.
  The x32 test checks count, filename and line number. A future constructor
  must supply a stack level accounting for its additional frames.
- jit/vmap tests check traced weak normalization and preserved component axes.
  No global JAX configuration is changed. Public exports are unchanged.

## Checks actually run

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- `uv run ruff format src/survjax/_dtypes.py src/survjax/_data_validation.py
  tests/unit/test_parameter_dtype.py`: two files reformatted, one unchanged.
  Subsequent formatting of `_dtypes.py`: one file unchanged.
- Final `scripts/check.sh --through A3`: **all gates passed through A3**.
  Guard passed; **29 files already formatted**; Ruff **All checks passed!**;
  mypy **Success: no issues found in 6 source files**.
  Own tests: x64 **135 passed, 4 skipped in 8.89s**;
  x32 **138 passed, 1 skipped in 8.13s**.
  Independent tests: x64 **372 passed, 15 skipped, 203 deselected in 12.73s**;
  x32 **382 passed, 5 skipped, 203 deselected in 12.19s**.
  New dtype skips: x64 skips the x32 conversion warning; x32 skips float64
  promotion. Other skips are the existing mode-specific cases.
- Earlier gate attempts exposed a mypy invariant-dict annotation and an x32
  warning pointing into pytest (**1 failed, 137 passed, 1 skipped**). Fixed
  with a read-only Mapping annotation and the correct default stack level;
  the final full gate above passed.
- `scripts/build_docs.sh`: x32 and x64 **build succeeded**, executing both
  documentation notebooks in each process. Initial sandbox execution failed
  because Jupyter could not bind a local port; the approved escalated rerun
  succeeded. Kernel startup printed its TCP encryption warning in each build.
  Confirmed the new precision prose appears in both generated HTML pages.
- `git diff --check`: passed with no output.

## Remaining scope and working tree

No new deviations or uncertain spec choices. Existing open deviations remain
for maintainer resolution. A4 registration, nested dtype integration and direct
methods are unimplemented; all 203 A4 acceptance tests remain deselected by the
A3 gate. No R, wheel smoke test, numerical derivative suite or performance
baseline was run; those belong to later steps.

The pre-existing A4 planning text is included in PLAN.md with approval/branch
status updated. The pre-existing DEVIATIONS.md edit is left unstaged and intact.
The implementation and tests change about 280 lines; the prior planning text
and this review account for the additional documentation in the commit.
