# D.LOGNORMAL.1 handoff

Independent review: round 1/3

## Read first

1. [Plan](PLAN.md): authorized family-only scope and deferred API documentation.
2. `src/survjax/distributions/lognormal.py`: natural leaves, symmetric stable
   normal tails, analytic density, inverse-Mills-ratio hazard.
3. `tests/unit/test_lognormal.py`: independent mpmath references and transformations.
4. Both package initializers and implementer unit/install export assertions.

## Invariants and coverage

- Spec revision 47, sections 3.3/4.1/4.7: registration owns dtype normalization,
  broadcasting, indexing and diagnostics. meanlog is real; sdlog is positive.
  `from_log` preserves meanlog and stores exp(log_sdlog), including overflow.
  `test_registration_grid_and_log_constructor` and
  `test_diagnostics_and_likelihood` exercise these paths.
- Sections 4.1/7.2/12.1: log survival/CDF use the prescribed symmetric
  log1p/ndtr versus log_ndtr branches, with safe operands in both branches.
  `test_tails_and_transformations` compares all seven methods to 100-digit
  normal-integral identities over standardized [-10,10] / [-7,7], with zero
  absolute tail tolerance, eagerly, jitted and vmapped.
- Sections 4.2/4.5: constant reference time one stays positive and finite for
  every valid parameter. Base machinery supplies survival/CDF/H boundaries;
  family density/hazard limits are zero at zero and infinity. Negative times
  follow the base policy; NaN operands retain precedence.
  `test_boundaries_and_zero_gradients` checks limits, parameter Jacobians,
  mixed-batch Hessians under debug_nans, and extreme reference locations.
- Section 7.3: `test_parameter_derivatives` checks forward/reverse gradients
  and Hessians against mpmath, at both normal-CDF and hazard branch switches
  and in both tails; second-order check_grads uses existing repository
  derivative tolerances and finite-difference steps. No tolerances changed.
- The log hazard uses a 128-term Laplace continued fraction for the inverse
  normal Mills ratio above standardized time two. This avoids tail-Hessian
  cancellation in log f-log S. The other branch uses the analytic difference;
  discarded continued-fraction operands are safe constants. Fixed loop bounds
  support reverse-mode differentiation. A separate 100-digit mpmath probe
  measured relative continued-fraction truncation error 7.0596174e-19 at z=2.
  This changes no shared machinery.
- Sections 5/7.1: time-first equal-length grids, vmap over distributions,
  invalid likelihood rows and mixed exact/right likelihoods are exercised.

## Validation

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3; no dependency changes.

- `JAX_ENABLE_X64=<1,0> uv run --group test pytest -n 6
  tests/unit/test_lognormal.py tests/unit/test_import.py -q`: 20 passed in each
  mode on the final numerical implementation (32.41s / 39.34s). The subsequently
  added extreme inactive-branch regression also passed separately in x64
  (`test_boundaries_and_zero_gradients`: 1 passed); it was included in x32.
  The final boundary assertions check only finite-valued boundary Jacobians;
  that test was rerun in each mode: 1 passed each (4.22s / 3.94s).
- `uv run ruff format --check .`: 201 files already formatted;
  `uv run ruff check .`: all checks passed;
  `uv run mypy src`: success, no issues in 23 source files;
  `git diff --check`: no output.
- `scripts/check.sh --fast --through A12`: guard/static checks passed, then
  final run: 1 failed, 1077 passed, 8 skipped in x64 (13.12s). Failure:
  `test_api_reference_lists_every_export` reports `['LogNormal']`, because
  the maintainer explicitly deferred docs edits. The existing test is unchanged.
  No claim of a green cumulative gate is made.
- Supplemental x32 implementer run: `JAX_ENABLE_X64=0 uv run --group test pytest
  -n 6 --ignore=tests/spec --ignore=tests/unit/test_api_reference.py
  -m 'not slow' -q`: 1093 passed, 3 skipped (227.83s). This exclusion exposes
  remaining tests, and does not turn the gate green.
- `JAX_ENABLE_X64=<1,0> SPEC_THROUGH=A12 uv run --group test pytest tests/spec
  -q -n 6 -m 'not slow'`: x64 3392 passed/33 skipped/4 setup errors;
  x32 3385 passed/40 skipped/4 setup errors. I omitted SURVJAX_WHEEL in
  these manual invocations. Built a wheel with `uv build --wheel -q -o
  /private/tmp/survjax-D.LOGNORMAL.1-wheel`, then reran
  `tests/spec/test_a9_wheel_smoke.py` in both modes with SPEC_THROUGH=A12 and
  SURVJAX_WHEEL naming that wheel: 5 passed each. The spec runs emitted two
  pre-existing NumPy divide/invalid warnings in the PH reference calculation.
- `JAX_ENABLE_X64=<1,0> uv run --group test pytest src --doctest-modules -q`:
  4 passed each (3.15s / 2.57s).
- `scripts/update_slow_tests.sh --through A12`: completed, 967 measured slow
  entries. Preserved all baseline entries and programmatically added only the
  11 measured LogNormal entries, avoiding unrelated timing churn. Updated the
  generated header to describe that provenance. The script ignores pytest exit
  statuses, so its completion is timing evidence only.
  Its earlier x64 implementer snapshot reported the docs failure and a stale
  install export assertion, subsequently updated and passing in the
  supplemental implementer run; full x64 spec timing reported 3580 passed,
  33 skipped, 2 warnings. Full x32 implementer timing reported 1333 passed,
  3 skipped, 1 failure (the deferred API-doc entry).

## Integration and remaining work

Starting/base commit: 3abefc5683a3aee8169eb2deffe57602897a3af3.
Preceding step checkpoint: ff07cf9f7942db0ce0e5d3a27525e0c1ca3d3431
(A12 acceptance), merged by the starting commit.

Overlap with the independent LogLogistic branch: `src/survjax/__init__.py`,
`src/survjax/distributions/__init__.py`, `tests/unit/test_import.py`,
`tests/install/test_wheel.py`, and the regenerated `tests/slow_tests.txt`.
Integration must retain both families' imports/exports, expected exports and
slow-test entries. No shared numerical code or other agent records changed.

No additional family implementation substeps are planned. Follow-up docs must
add LogNormal's API entry and parameterization/tail guidance; the roadmap may
then mark this family implementation complete, but Slice D as a whole remains
incomplete. No integration, merge, publication or unrelated Slice D work here.
