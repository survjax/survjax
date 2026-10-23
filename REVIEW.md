# Review: A7.2 — Boundary and inactive-branch safety

## What to read, in order

1. `tests/unit/test_likelihood_debug_nans.py`: RootHazard and the S1 regression
   `test_custom_density_at_zero_is_inactive`, then mixed-batch built-in/cure checks.
2. `tests/unit/test_likelihood_boundaries.py`: smallest normal time, nonzero
   right contributions and zero censoring beside an event.
3. The current A7.2 section in `PLAN.md`, then the mutation evidence below.

This step adds 19 cases in 299 test lines (113 boundary, 186 debug). There are
no repository source, API, numeric tolerance, dependency, narrative-doc or
independent-test changes. The small custom families are local test probes.

## Invariants and expected values

- Supported exact/right rows only, without truncation. JAX dtype mode is
  supplied by separate runner processes and is never changed in the tests.
- At the smallest positive normal time, 100-digit mpmath evaluates the defining
  exponential/Weibull density and its hand-derived parameter derivatives on
  dtype-rounded inputs. The reference uses log(t)-log(b), not a subnormal ratio.
  Only right contributions whose cumulative hazard is normal get relative-only
  tiny-value comparisons. No backend-specific subnormal assertion is added.
- Zero censoring contributes exactly zero and has exactly zero parameter
  derivatives. A neighbouring event retains its analytical value/derivatives,
  in both AD modes, for Weibull shapes below/at/above one.
- Mixed built-in references use log L=event*log h-H and the hand derivatives.
  NumPy evaluates the independent algebra on concrete inputs; from_log uses
  exp of the dtype-rounded log parameters and the chain rule. Reference
  rounding differences must fit the unchanged tolerances.
- The NaN-debug batches cover shared/per-row parameters, early/interior/late
  events, zero/tiny/interior censoring, both observation constructors and public
  entry points, plus from_log. They check values and first derivatives of the
  summed contributions eagerly; debugging alone is not correctness evidence.
- RootHazard supplies only log S(t)=-rate*sqrt(t) and reference_time=1. Its
  generic density is deliberately unsafe at zero. At rate 2, [right at 0,
  exact at 1] gives [0,-2], sum derivative -0.5 and zero derivative on row 0.
  These anchors are checked ordinarily and under NaN debugging, through the
  function/method, eager/jit/vmap, with forward/reverse zero-row checks.
- Cure uses S(t)=p+(1-p)*exp(-r*t), f(t)=(1-p)*r*exp(-r*t). Right censoring
  includes the cure mass, including the far-tail log(p) limit; exact density
  comes from the generic default where the hazard is representable. Its
  reference derivatives come from differentiating these identities.
- Ordinary and relative-only tolerances come exclusively from
  tests/impl_tolerances.py and are unchanged. NumPy/mpmath conversions run only
  on concrete reference inputs or completed results, never tracers.

## Spec-to-test map

| Spec | New tests | Evidence |
| --- | --- | --- |
| 3.3, 4.1, 7.1, 7.3 | `test_exponential_exact_at_smallest_normal_time`, `test_weibull_exact_at_smallest_normal_time` | Finite smallest-normal-time values and first parameter derivatives, with independent references, both AD modes |
| 7.1, 12.1 | `test_tiny_right_censoring_keeps_nonzero_mass` | Normal tiny log survival is negative/nonzero with zero absolute tolerance |
| 4.5, 7.2 | `test_zero_censoring_keeps_zero_derivatives_beside_an_event` | Structural zero value/derivatives and preserved event derivatives |
| 4.1, 7.2, 12.1–12.2 | `test_exponential_mixed_values_and_gradients_under_debug_nans`, `test_weibull_mixed_values_and_gradients_under_debug_nans` | Five shared/per-row/from-log cases with independent values and gradients under eager debug mode |
| 4.2, 7.2, reviewer S1 | `test_custom_density_at_zero_is_inactive` | Four function/method and ordinary/debug cases detect missing inactive-density substitution |
| 2, 4.2, 7.2 | `test_cure_mixed_values_and_gradients_under_debug_nans` | Improper-family right/zero/exact rows preserve cure mass and mathematical derivatives |

A7.1 tail/overflow tests and existing direct-method boundary tests remain
unchanged. Existing likelihood transformation and parameter-isolation suites
retain their second-order/interior and invalid-parameter coverage.

## Checks run

Python 3.12.9; JAX 0.11.2; NumPy 2.5.3; mpmath 1.4.1; pytest 9.1.1;
`[CpuDevice(id=0)]`. Verified the ordinary environment imports
`/Users/wamster3/dev/survjax/dev/src/survjax/__init__.py`.

- Ruff initially reported B023 on the loop's gradient lambda; binding run as a
  default argument fixes it. Final command `.venv/bin/ruff format
  tests/unit/test_likelihood_boundaries.py tests/unit/test_likelihood_debug_nans.py`
  reports `2 files left unchanged`; `ruff check` on those two files reports
  `All checks passed!`.
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A7 .venv/bin/python -m pytest -q -ra
  tests/unit/test_likelihood_boundaries.py tests/unit/test_likelihood_debug_nans.py
  tests/spec/test_a7_likelihood_boundaries.py tests/spec/test_a7_debug_nans.py`:
  **53 passed in 11.77s**. Same with `JAX_ENABLE_X64=0`:
  **53 passed in 11.81s**. Includes all 19 new cases; no focused skips.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
  passed. Guard `ok (A7)`; Ruff
  `70 files already formatted` / `All checks passed!`; mypy
  `Success: no issues found in 17 source files`; own float64 suite
  **777 passed, 7 skipped in 141.67s**, own float32 suite
  **783 passed, 1 skipped in 140.21s**; source doctests **4 passed in 0.86s**
  (float64) and **4 passed in 0.83s** (float32). Independent suites through A7:
  **2123 passed, 28 skipped, 44 deselected in 93.45s** (float64),
  **2127 passed, 24 skipped, 44 deselected in 92.71s** (float32).
  Final output: `check: all gates passed through A7`. No skips or exclusions
  were added; the gate deselects later-step tests.
- `git diff --check`: clean; `git diff -- src tests/spec`: empty.

## S1 mutation check — expected failures

Copied src/ into `/private/tmp/survjax-a72-mut-ta0ge86q`, changing exactly one
statement in the copy: density_time selects the actual time on every row,
instead of substituting reference_time on inactive rows. Repository source
was never modified. Each process asserted that survjax.__file__ is exactly
`/private/tmp/survjax-a72-mut-ta0ge86q/src/survjax/__init__.py` before pytest.

Commands used separate `JAX_ENABLE_X64=1` and `=0` processes,
`PYTHONPATH=/private/tmp/survjax-a72-mut-ta0ge86q/src`,
`PYTHONDONTWRITEBYTECODE=1`, and `.venv/bin/python -`. After the import
assertion, the stdin wrapper called:

```python
pytest.main(['-q', '-p', 'no:cacheprovider',
             'tests/unit/test_likelihood_debug_nans.py',
             '-k', 'custom_density_at_zero_is_inactive', '--tb=short'])
```

Both processes exited 1: **4 failed, 6 deselected in 1.14s** each, as required.
Function and method ordinary cases fail the gradient assertion: actual NaN,
expected -0.5. Their debug cases raise
`FloatingPointError: invalid value (nan) encountered in mul`.
The same four cases pass on the repository source in each focused run above.
Thus the new regression catches the missing substitution in both dtypes;
this is selected mutation evidence, not exhaustive mutation coverage.
`git diff -- src tests/spec` is empty.

## Limits and next review stop

No new deviation or unresolved numerical failure was found. This step does
not test data-time derivatives, arbitrary custom families or the derivatives
of invalid/infinite likelihoods. No tolerance was loosened or exclusion added.
The separate numerical-documentation step is still pending; no docs build,
R fixture, installation check, GPU run or performance measurement was run.
A7 and Slice A are not declared complete. Commit this step, then stop for review.
