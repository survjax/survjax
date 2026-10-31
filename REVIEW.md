Independent review: round 1/3

# BAYES.1 — Fixed trial, NumPyro wiring and sampling pilot

Read first:
1. PLAN.md BAYES.1 and plans/release-agents/bayesian.md.
2. Both spec/use-cases/02-bayesian-survival files, spec 11.3 and shared 00.
3. examples/bayesian_rmst/model.py, check_model.py, then run.py.

The maintainer explicitly says to assume incoming PH/RMST are present and
approves smoke = four chains with 100 warmup + 100 retained draws per chain.
This commit targets those specified APIs without adding temporary core code.
The current checkout does not yet export ProportionalHazards or provide rmst.
Consequently this is implementation prepared for integration, not a successful
pilot or verified factor wiring. Runtime/docs-budget decisions remain pending.

Invariants:
- Fixed balanced 150/150 synthetic trial; float64 is required. Shared generator
  alone owns U/C splitting and host censoring calibration.
- Root 20261025, use-case fold 2, separate folds 0–3 exactly as designed.
- Three scalar Normal priors and one (300,) likelihood factor inside the
  observations plate. No sum inside that plate. Per-arm deterministic sites
  remain outside it and disabled during NUTS.
- Runner configures x64 and four devices before backend initialization,
  preserves default NUTS settings/init, and reports actual chain method,
  synchronized sampling timing and per-chain divergences. Full remains
  4x(1000+1000); its eventual bulk/tail ESS and truth audit are a later step.
- Optional checks live outside ordinary pytest collection; core dependencies,
  source, docs, shared generator and lockfile are unchanged.

Evidence (2026-10-31):
- `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.check_model`: exit 1. Trial reproducibility, exact
  150/150 arms, dtype, validity and all four key mappings passed;
  c=19.1725354361981 years. Factor check then stops with AttributeError:
  survjax has no ProportionalHazards. Factor equality/plate structure and
  optional deterministic-site checks are implemented but NOT verified.
- `uv run --group examples python -m examples.bayesian_rmst.run --config smoke`:
  exit 1 at the same missing PH API during parallel MCMC initialization.
  No retained draws, timing or diagnostic claims. Full run not attempted.
- `uv run ruff check .`: All checks passed!
- `uv run mypy src`: Success: no issues found in 20 source files.
- `uv run ruff format --check examples/bayesian_rmst`: 3 files already formatted.
- `git diff --check`: exit 0.
- `uv run ruff format --check .`: exit 1, pre-existing README.md embedded
  Python requires blank lines at 67 and 72; 121 files already formatted.
- `scripts/check.sh --through A10`: exit 1 at that same README formatter
  failure after `guard: ok (release-bayes)`; later gate checks did not run.
- `scripts/run_tests.sh`: exit 0; float64 1065 passed, 7 skipped in
  91.22s; float32 1071 passed, 1 skipped in 73.05s.
- Fresh optional-process device probe: 4 devices, cpu backend, x64 True.
- Installed optional environment: JAX/jaxlib 0.11.2, NumPy 2.5.3,
  NumPyro 0.22.0; no dependency edits.

Coverage mapping: fixed-trial/key checks cover 02-details Data; density and
plate assertions target spec 11.3's no-overcounting recipe; Predictive shapes
and site absence target its optional derived-quantity wiring. No claim of
route A/B numerical agreement, full convergence, posterior truth coverage,
KM verification, plots, predictive checks or executable docs yet.

Uncertainty: integrated PH/RMST execution and actual pilot time are unknown.
Do not merge prerequisites or modify reviewer checkout from this session.
