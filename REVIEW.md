# ALIGN.1 — Keyword-only PH construction

Independent review waived by maintainer on 2026-11-06 while reviewer tokens
are unavailable. Starting commit: fc484d3. Worktree starting point: 3cb4cd7.

Read first: PLAN.md ALIGN.1, src/survjax/distributions/wrappers.py, construction
and warning tests, then executable Bayesian callers. Spec 3.3, 4.2, 4.4, 11.3.

Invariants: standard registration owns keyword-only initialization, nested
broadcast/dtype handling and warning attribution. Only constructor spelling
changes in numerical tests/examples; formulas, trial data and draws stay fixed.
Tests: explicit rejection in eager/jit/vmap; existing construction, diagnostics,
warning and numerical regressions in both modes. Bayesian factor and RMST
route checks use existing committed draws without sampling again.

D-043/D-044 remain unresolved acceptance discrepancies;
no protected tests or specification will be changed.

## Completed validation

- `uv sync --locked --group test --group examples`: succeeded; installed only
  the already-declared optional example dependencies, with no lockfile change.
- Full touched suites, no slow filter, both processes:
  `JAX_ENABLE_X64=<1,0> uv run --no-sync --group test pytest -n 6
  tests/unit/test_proportional_hazards*.py
  tests/unit/test_weibull_rmst_near_scale.py
  tests/spec/test_a11_ph_construction.py -q --tb=short`.
  x64: 210 passed, 2 skipped in 179.63s; x32: 209 passed, 3 skipped in 200.66s.
  This includes explicit positional rejection eagerly and under jit/vmap,
  keyword construction/broadcasting/diagnostics, x32 warning attribution,
  PH numerical regressions and touched RMST tail/derivative tests.
- `scripts/check.sh --fast --through A12`: guard passes; Ruff reports
  173 files already formatted / All checks passed; mypy reports no issues
  in 22 source files. Implementation tests: x64 1059 passed, 8 skipped in
  42.10s; x32 1064 passed, 3 skipped in 50.02s. Source doctests: 4 passed in
  each mode (1.81s / 2.32s). Acceptance x64: 1 failed, 3396 passed, 32 skipped,
  2 warnings in 106.81s. Sole failure is D-043; gate exits 1 and consequently
  does not reach its float32 acceptance phase. No claim of a green gate.
- Bayesian one-off float64 check using make_trial, run.check_factor and
  NumPyro Predictive over all 4000 committed draws: factor wiring passes
  atol=1e-10, rtol=0; regenerated trial time/event/arm arrays equal the
  committed arrays exactly. Native batch vs Predictive RMST max differences:
  1.3322676295501878e-14 at tau=5 and 2.842170943040401e-14 over all 101
  cutoffs, both below 1e-12. No NUTS sampling or output regeneration.
- AST audit of model.py/run.py and all notebook code: only baseline argument
  spelling changes. Three notebook cells changed; metadata/outputs unchanged.
  No positional PH calls remain in package or Python examples. Intentional
  rejection cases remain in unit tests. Formulas and reference values unchanged.
- No changes in pyproject.toml, uv.lock, tests/spec/, spec or docs/.
  Implementation/test/example diff is 137 changed lines, below 300.
- Logs: /tmp/align1-touched-x64.log, /tmp/align1-touched-x32.log,
  /tmp/align1-fast-gate.log and /tmp/align1-bayes.log.

Float32 acceptance follow-up command:
`JAX_ENABLE_X64=0 SPEC_THROUGH=A12 uv run --no-sync --group test pytest
tests/spec -q -n 6 -m 'not slow' --tb=short`.
Result: 2 failed, 3385 passed, 38 skipped, 2 warnings, 4 errors in 148.85s.
The only failures are D-043 and D-044. The four errors are invocation setup:
I omitted SURVJAX_WHEEL, which the standard gate normally supplies. Rebuilt the
wheel and reran that wheel file separately with SURVJAX_WHEEL set; result below.
Log: /tmp/align1-spec-fast-x32.log.
 Tier 2 and docs builds are not run in this code
step. Docs still contain positional examples; ALIGN.d1 must migrate those.

Corrected wheel invocation:
`JAX_ENABLE_X64=0 SPEC_THROUGH=A12
SURVJAX_WHEEL=/tmp/survjax-align1-wheel/survjax-0.0.1.dev0-py3-none-any.whl
uv run --no-sync --group test pytest tests/spec/test_a9_wheel_smoke.py -q
--tb=short`: 5 passed in 1.18s, including the outside-source import check.
Log: /tmp/align1-wheel-x32.log. No wheel/spec-test changes were necessary.

Final whitespace check passes. Independent review remains waived explicitly;
no reviewer tool was run. ALIGN.1 is complete with a red acceptance gate on
existing owner discrepancies; it is not declared ready to merge. Remaining:
ALIGN.2 owner resolutions (D-042/D-043/D-044), ALIGN.d1 docs migration and
ALIGN.3 final verification. No subsequent implementation step is started.
