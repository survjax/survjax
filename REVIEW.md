# A4.2 review fix-up

Addresses `../review/slices/A4.2.md` F-1 and N-1–N-4, plus the maintainer's
request for clearer public docstrings. One extra commit on `A4.2`, then stop.

## Reading order

1. `src/survjax/distributions/base.py`: public docstrings, declaration checks,
   inherited-initializer rejection and field-named static hashability errors.
2. New tests at the end of `tests/unit/test_family_registration.py`.
3. `DEVIATIONS.md` D-014: previously omitted plain fields and inherited
   constructors; approved implementation policy, status still open.
4. `PLAN.md`: approved fix-up and added docstring scope.

## Invariants and tests

- F-1 / spec 4.2, D-014: every dataclass field has an available role. Plain
  fields raise a field-named TypeError before registration; tests cover both
  ordinary defaults and default factories. They cannot silently reset or
  disappear after JAX reconstruction. The earlier REVIEW.md claim that no
  registration ambiguity existed was wrong; D-014 records the gap.
- N-1 / spec 4.2: component counts are non-negative Python ints, excluding
  bool. Separate tests cover wrong kinds and negative values.
- N-2 / spec 3.3, D-014: the initializer must belong to the class being
  registered. An undecorated subclass raises TypeError instead of wrapping
  its parent's registered initializer. A separately decorated subclass remains
  supported, with an x32 test checking warning count and caller filename/line.
- N-3 / spec 4.2: unhashable static configuration raises TypeError naming the
  field. The new test requires the explanatory message.
- N-4 / spec 4.1–4.2, 5: tests now require rejection of non-frozen dataclasses
  and scalar inputs lacking a declared component axis. Removing either guard
  would fail its explicit rejection test; mutation testing was not rerun.
- No parameter-value checks added. Normalization, broadcasting, PyTree children,
  auxiliary data and reconstruction stay unchanged for valid declarations.
  Existing unit and focused acceptance cases cover these paths in both modes.
- Public docstrings explain usage, arguments, shapes, returns and errors for
  `param`, `static`, `register_family`, `SurvivalDistribution`, `batch_shape`,
  `log_survival` and `reference_time`. Development/spec bookkeeping was removed
  from those docstrings. Documentation pages remain for the maintainer's later
  review; no page rewrite or new public names in this fix-up.

## Checks actually run

All uv commands used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

- `scripts/check.sh --through A3`: **all gates passed through A3**.
  Guard passed; **33 files already formatted**; Ruff **All checks passed!**;
  mypy **Success: no issues found in 8 source files**.
  Own x64: **158 passed, 6 skipped in 9.84s**.
  Own x32: **163 passed, 1 skipped in 8.77s**.
  Prior independent x64: **372 passed, 15 skipped, 203 deselected in 12.44s**.
  Prior independent x32: **382 passed, 5 skipped, 203 deselected in 12.34s**.
  Own x64 skips are x32 conversion/unavailable-dtype cases, including the new
  subclass warning test; the x32 skip is float64 promotion.
- Focused A4 acceptance, separately for mode=1 and mode=0:
  `JAX_ENABLE_X64=<mode> SPEC_THROUGH=A4 uv run --locked --offline pytest -q $(cat /private/tmp/survjax-a42-nodes.txt)`.
  Same 60 nodes as commit `6ea4814`'s REVIEW.md: registration and parameter
  tests excluding the nine tests that depend on later A4 scope, plus
  `test_a4_indexing.py::test_tree_map_slicing_updates_batch_shape` and
  `test_a4_indexing.py::test_batch_shape_inside_vmap_is_the_per_entry_shape`.
  x64: **58 passed, 2 skipped in 0.43s**.
  x32: **59 passed, 1 skipped in 0.36s**.
  These skips cover x32 warnings in x64 and float64 promotion in x32.
- `scripts/build_docs.sh`, with approved escalation for local Jupyter kernel
  ports: **build succeeded in x32 and x64**, executing all three example
  notebooks per process. Kernel startup printed its TCP encryption warning.
  No visual browser inspection was run in this fix-up; local HTML browser
  access was blocked during the original step.
- Final docstring wording adjustment was followed by Ruff format/lint and
  mypy checks: **33 files already formatted**, **All checks passed!**,
  **Success: no issues found in 8 source files**. Runtime tests were not
  repeated for prose only.
- `git diff --check`: passed with no output.

## Remaining scope

D-014 remains open for maintainer resolution; D-013 stays open for A6.
No spec or tests/spec changes. No R, wheel smoke test, benchmark, derivative
suite, mutation run or full A4 gate was run. Nested fields/indexing and numerical
methods remain later A4 steps. Stop after this fix-up commit.
