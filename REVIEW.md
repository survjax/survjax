# A4.5 review fix-up — WA naming comments

Implemented on `A4.5` after `e2c08af`, authorized by the maintainer request to
address the inline WA comments. One naming-only fix-up commit, then stop.
Specification revision 21; no public API or numerical behavior changes.

## Reading order and addressed comments

1. `src/survjax/distributions/_survival_methods.py`: read the two private helpers
   and their calls from the density/log-hazard defaults.
2. `tests/unit/test_family_density.py`: existing constant and power identities,
   zero-hazard active derivatives and parameter-derivative checks.
3. `PLAN.md`: the review fix-up scope. Prior full A4.5 evidence is in `e2c08af`.

All three WA comments were addressed:

- `_survival_and_derivative` is now `_log_survival_and_derivative`, because
  the JVP returns log survival and its time derivative.
- `_log_rate` is now `_log_hazard`; its argument is `hazard_value`, and safe
  operands are called `safe_hazard` throughout both defaults.
- Its docstring states that it takes the log of a linear hazard value and
  preserves zero hazard as minus infinity. This is a general hazard helper:
  `h(t) = -d log S(t)/dt`, not an Exponential/Weibull parameter helper and not
  the density. Generic density combines this log hazard with log survival.

The addressed WA comments were removed. No other user edits were present.

## Invariants and spec-to-test mapping

The numerical expressions, zero-hazard substitution and stop-gradient placement
are unchanged. The raw primitive JVP, time/shape policy, override dispatch and
public methods are unchanged. No new tests are needed for private renames.

- Spec 4.2: existing constant/power density and hazard identities demonstrate
  that the helper is family-independent.
- Spec 7.2: `test_zero_hazard_does_not_poison_active_derivatives` demonstrates
  the protected zero-hazard logarithm; direct safe-reference tests cover
  inactive time substitutions.
- Spec 7.3: `test_interior_parameter_derivatives` checks parameter derivatives
  to order two with jit/vmap and forward/reverse check_grads.
- Spec 4.5/5: the unchanged direct policy/broadcast tests and full independent
  A4 acceptance suite exercise all method call sites.

## Verification

Used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.
`scripts/check.sh --through A4` exited 0:

- Guard: `guard: ok (A4.5)`.
- Ruff format: `38 files already formatted`; lint: `All checks passed!`.
- mypy: `Success: no issues found in 9 source files`.
- Own x64: **282 passed, 6 skipped** (10.76s).
- Own x32: **287 passed, 1 skipped** (9.85s).
- Independent x64: **676 passed, 25 skipped** (14.36s).
- Independent x32: **690 passed, 11 skipped** (12.19s).
- Final output: `check: all gates passed through A4`.

Skips are unchanged: mode-specific conversion/promotion checks, optional pandas,
plus the independent finite-difference density checks restricted to x64.
No protected tests were changed, skipped or xfailed.

The whole-word search for `WA`, `_survival_and_derivative`, `_log_rate` and
`safe_rate` across src/tests/docs (excluding generated documentation) found no
matches (rg exit 1). `git diff --check` passed with no output.

Docs builds, combined own/spec sessions and source doctests were not repeated:
only private names and a helper docstring changed; their original A4.5 results
remain in the preceding commit. No new uncertainty or deviation was identified.
D-016 remains open and deferred to A5. No A5 work or publishing was performed.
