# Review: A1 constructor docstrings

Read in this order:

1. `PLAN.md`, final section, for the maintainer's documentation request and
   the public API boundary.
2. `src/chronox_jax/data.py`: the `SurvivalData` class docstring, `__init__`
   docstring, and `from_bounds` docstring. No executable code changed.
3. R's [`survival::Surv` manual](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html)
   for the comparison used in those docstrings.

## Documentation invariants

- The flag constructor explains that the second positional argument is the
  event indicator, as in `Surv(time, event)`: 1/True means exact event,
  0/False means right censoring. It explains the alternative per-row
  censoring flags, the role of `time2`, shape/broadcast rules, truncation
  defaults, and dtype selection.
- R's `Surv(time, time2, event)` defaults to counting-process data when all
  three arguments are present. Chronox's `time2` instead names an interval
  upper endpoint selected by `interval_censored`.
- `from_bounds` is related to `Surv(..., type="interval2")` but requires an
  explicit `exact` mask. It explains `(lower, upper]`, `0`/`+inf` censored
  endpoints, exact-row normalization, and that an equal endpoint pair alone
  does not mark an exact event. R accepts NA for an unbounded endpoint;
  Chronox documents the numeric bounds in spec 3.1.
- The class docstring describes Chronox's five aligned arrays and issue leaf
  alongside R's matrix-with-type representation. All observations retain the
  shape of the first time argument; there is no time-grid axis.

## Checks and limits

No tests were added because this fix-up changes docstrings only. The existing
tests remain the evidence for spec 3.1–3.3 behavior; the full A1 gate was
rerun with `UV_CACHE_DIR=/private/tmp/chronox-uv-cache`:

- Ruff format: `13 files already formatted`; Ruff lint: `All checks passed!`.
- mypy: `Success: no issues found in 2 source files`.
- Own tests: x64 enabled `19 passed, 3 skipped`; x64 disabled `22 passed`.
- Independent A1 tests: x64 enabled `41 passed, 8 skipped, 196 deselected`;
  x64 disabled `45 passed, 4 skipped, 196 deselected`.
- Final output: `check: all gates passed through A1`.
- `git diff --check`: passed with no output.

The docstrings describe the specified valid-input contract. A1's current
constructors do not yet enforce every indicator, endpoint, and truncation
rule; A2 adds those checks and the issue bitfield behavior. A3 adds indexing
and value-derived static metadata. No R runtime, fixture, wheel, or install
check was run for this documentation-only fix-up.
