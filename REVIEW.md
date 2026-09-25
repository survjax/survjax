# Review: A1 round-two weak-typing fix-up

Read in this order:

1. `~/dev/chronox-dev-review/.review/a-r2.md`, especially R1–R3. The
   review inspected `f663fd9`; this branch also includes the later
   constructor-docstring commit `92a5d63`.
2. `PLAN.md`, final section, for this fix-up's limited scope.
3. `src/chronox_jax/data.py` (`_input_dtype` and `_time_dtype`) and the new
   cases in `tests/unit/test_data_dtype.py`.
4. `DEVIATIONS.md` D-002, now citing the landed spec change `3041ebd`.

## Invariants and disposition

- **R1 fixed:** A JAX array or tracer with `weak_type=True` contributes its
  value to `jnp.result_type`, preserving JAX's weak promotion. A strong JAX
  array still contributes its dtype. The check rejects boolean and other
  nonnumeric time dtypes before returning a weak JAX value. No tracer is
  converted to NumPy.
- **R2 fixed:** A bare Python `True` time raises, rather than being treated as
  weak integer 1. The existing rejected-input test now covers it.
- **R3 fixed:** D-002 cites section 3.3 as changed in `3041ebd`. Its status
  remains `open`; only the maintainer resolves deviations.
- The review's section 4.2 formatting note concerns protected spec text. No
  `spec/` or `tests/spec/` file changed in this fix-up.

## Tests and checks

| Requirement | Evidence |
| --- | --- |
| Weak eager JAX bound adopts float32 time (spec 3.3) | `test_weak_jax_bound_preserves_float32_time_dtype` |
| Python float becomes a weak tracer under `jit` (spec 3.3) | `test_jitted_weak_float_bound_preserves_float32_time_dtype` |
| Boolean scalar time is rejected (spec 3.3) | `test_nonreal_time_input_is_rejected` includes `True` |
| Earlier A1 contract remains intact | `scripts/check.sh --through A1`, including independent A1 tests in both modes |

Final `scripts/check.sh --through A1` output, with
`UV_CACHE_DIR=/private/tmp/chronox-uv-cache`:

- Guard: `ok (a1-survivaldata)`.
- Ruff format: `13 files already formatted`; Ruff lint: `All checks passed!`.
- mypy: `Success: no issues found in 2 source files`.
- Own tests: x64 enabled `22 passed, 3 skipped`; x64 disabled `25 passed`.
- Independent A1 tests: x64 enabled `41 passed, 8 skipped, 196 deselected`;
  x64 disabled `45 passed, 4 skipped, 196 deselected`.
- Final line: `check: all gates passed through A1`.
- `git diff --check`: passed with no output.

The own `jit` construction test verifies A1 promotion now. A3 will require
ordinary traced construction to raise and will introduce `with_issues`; move
this transformation check to that route when A3 is implemented. The
independent A1 suite predates the updated section 3.3 wording, so the new
R1/R2 cases are currently evidenced by implementer tests. No R, fixture,
wheel or install check was run. A2 validation and A3 indexing/metadata remain
outside this fix-up.
