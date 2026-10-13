# A5.4 review fix-up — clear notation and visible formulas

Implemented on the existing A5.4 branch after commit 63e4171, authorized by
the maintainer's request. Same step; no public API or numerical policy change.

## Reading order

1. src/survjax/distributions/weibull.py: the opening class sentence defines
   k=shape and b=scale. Whitespace and spec-linked comments distinguish safe
   inputs, the interior cumulative-hazard/log-hazard formulas and boundary
   restoration. The density identity is identified where it is evaluated.
2. Weibull.reference_time: the WA question is replaced with its explanation.
   A valid stored scale is already a positive finite time; returning it needs
   no reciprocal calculation. Exponential's preferred reference is 1/rate,
   whose representability must be guarded. Both follow spec 4.2.
3. src/survjax/distributions/exponential.py: the product rate*safe_time is
   named value in a separate formula block before restoring the infinity limit.
   Constant-hazard and density comments identify the analytic formulas; the
   reference comment explains that its fallback affects only inactive inputs.
4. PLAN.md's A5.4 review fix-up records the requested scope and validation.

## Invariants and spec evidence

- All masks, substitutions, boundary values, logarithms, products, dtype rules,
  broadcasting and dispatch are unchanged. The sole executable edit names
  Exponential's existing product before passing it to the same jnp.where.
- No real parameter/result is stopped or modified. The reference safeguard
  changes only substituted inputs, never a rate or probability (4.2/7.2).
- Existing test_construction_and_reference (Exponential) and
  test_extreme_ratios_and_reference_scales (Weibull) verify the reference
  shape/dtype and positive finite values across extreme valid parameters.
- Existing built-in value, boundary, raw-broadcasting and mixed-batch derivative
  tests cover 4.1/4.5/5/7.2/7.3. No new tests mirror comments or the equivalent
  temporary assignment. No tests, spec, exports or docs pages were edited.

## Checks actually run

    .venv/bin/ruff format --check .
    .venv/bin/ruff check .
    .venv/bin/mypy src

All exited 0: **44 files already formatted**, **All checks passed!**,
**Success: no issues found in 11 source files**.

    JAX_ENABLE_X64=1 .venv/bin/python -B -m pytest tests/unit/test_weibull.py tests/unit/test_exponential.py -q -p no:cacheprovider

Repeated with mode 0. x64: **32 passed, 1 skipped** (10.92s).
x32: **33 passed** (11.22s). The x64 skip is the existing x32-only
float64-conversion-warning case. git diff --check passed with no output.
The targeted WA search returned no matches. Diff inspection confirmed the
unchanged formulas/masks and the single named-intermediate edit.

## Uncertainty and stop

No new ambiguity or numerical behavior is introduced. The existing D-020
independent acceptance discrepancy remains open; this fix-up does not address
it. The full regression gate and independent acceptance suite were not rerun,
since their formulas and tests are unchanged. Their previous A5.4 evidence is
in commit 63e4171: through-A4 passed; non-grid A5 had one failure per mode.
No docs build, benchmark, R, wheel or mutation check was run.

Commit this requested review fix-up and stop before any next step.
