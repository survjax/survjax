Independent review: round 2/3

# BAYES.1 follow-up — Direct execution and independent trial oracle

Round 1 returned maintainer. The maintainer requests continuation and reports
`uv run examples/bayesian_rmst/run.py` fails to import examples. This fix-up
supports that command and addresses F-002 and N-001 without changing core.

Follow-up reading order: PLAN.md follow-up, run.py bootstrap, check_model.py
independent trial oracle. Prior round context and evidence are preserved below.

Changes and findings:
- Direct-file execution now inserts the resolved checkout root into sys.path
  before importing examples. Module execution keeps its existing search path.
  Configuration of device count/x64 remains before backend initialization.
- F-002: expected c comes from the equal-weight two-arm calibration, and
  expected observations use the literal data key and fixed design values.
  Compare time, exact/event and c instead of comparing make_trial to itself.
- N-001: deterministic wiring check uses literal key(0), leaving reserved
  predictive streams unambiguous.
- F-001 and S-001 remain pending prerequisite integration, including PH RMST.

Follow-up checks:
- `uv run examples/bayesian_rmst/run.py --help` and
  `uv run --group examples python -m examples.bayesian_rmst.run --help`:
  both exit 0 with smoke/full choices.
- `uv run examples/bayesian_rmst/run.py` and the module smoke command:
  both now reach MCMC model initialization and exit 1 because survjax has no
  ProportionalHazards. No examples import error. No sampling/timing evidence.
- Explicit x64 optional-process check_trial passes, c=19.1725354361981.
  In the same process unittest.mock.patch replaces model.calibrate_censoring
  with target=0.25, model.generate_weibull with negated eta, then
  model.use_case_key with stream+1: all three raise AssertionError as expected.
- `uv run ruff check .`: All checks passed!
- `uv run mypy src`: Success: no issues found in 20 source files.
- Format check of run.py/check_model.py: 2 files already formatted.
- `git diff --check`: exit 0.
- No repeat core suite or cumulative gate for this isolated demo fix; earlier
  results and the unchanged README gate failure are recorded below.

Maintainer committed all three Normal(0,5) priors in 3c554be and requests
continuation. The checkout is now clean before this follow-up. This explicitly
authorizes retaining those priors; the factor-check prior term now matches the
committed model and run metadata records distribution/loc/scale for each site.
D-028 records the departure from the delegated design (open, not resolved here).
Prior predictive assessment is still pending. No spec file is edited.

Current follow-up checks: Ruff format (3 files) and lint pass; mypy passes
for 20 source files; both help entry points exit 0; whitespace check passes.
Explicit x64 trial oracle passes with c=19.1725354361981. At the specified
factor-check parameters, the three Normal(0,5) terms sum to
-7.6936321221273225 and agree with the independent analytical Normal formula
to atol=1e-12, rtol=0. This verifies only the prior arithmetic, not NumPyro's
full factor density (still unexecuted without PH). Ordinary core suites and
full gate are not repeated for this seven-line example-only alignment.

Round 2 includes b92ccf4's direct-entrypoint and independent-trial fixes, the
maintainer's 3c554be prior choice, and this alignment/metadata follow-up.
F-002 and N-001 are addressed. F-001 and S-001 still require integrated PH/RMST
execution; authorization to proceed against their specified APIs remains the
maintainer's original instruction. Please assess the prepared implementation
with those runtime limitations, which are not claimed as passing evidence.

---

# Prior BAYES.1 implementation record

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
