Independent review: round 2/3

# BAYES.3 — Full posterior audit and reproducible output

Round 1 returned fix-first. F-001: independent 40-digit mpmath quadrature
now checks all five recorded true values, including the arm order/sign.
F-002: a separate wrong delta interval must fail; shift only chain 0's derived
delta, then compare recorded R-hat against NumPyro summary of an independently
grouped (4,1000) chain matrix. An initial assertion of failure alone did not
catch order=F (split R-hat also detects time trends); the exact comparison does.
N-002: docstring explicitly distinguishes hard malformed/nonfinite input
errors before an audit exists from acceptance failures, which are saved.
N-001 is a nonblocking duplication nit; retain fixed design literals rather
than add a configuration layer in this step.

Current checks: optional full check_model passes with truth/delta additions;
standalone enhanced audit/artifact check passes, and the order=F mutant is
detected by the exact R-hat comparison. Truth-arm and dropped-delta mutants are detected. Ruff format,
lint, mypy and whitespace pass. No numerical/sampling behavior change, no
repeated full run or core gate. Cumulative source/check changes are 310 lines
(308 additions, 2 removals), about the 300-line step target; initial review
already covered the full audit/persistence capability, not just this fix-up.

Read PLAN.md BAYES.3; run.py audit_posterior/save_results and main integration;
check_model.py check_audit_and_artifacts; 02-details Sampling/Truth/Files.
Previous step/head aeb249d; original worktree base b1291a8. Same Bayesian
sub-slice. Maintainer explicitly requests meaningful combined work before
review, excluding plan/review lines: 262 changed source/check lines (260 added,
2 removed), no new module, API, dependency, protected file, core or docs edit.

Invariants:
- Four explicit chain axes preserved; flattened derived draws reshape into
  NumPyro's chain-major order. Diagnose all three log parameters and all five
  reported quantities. NumPyro summary supplies split R-hat/n_eff as required
  by 02-details; ArviZ supplies tail ESS. No rank/bulk method substitution.
- Strict R-hat<1.01, both ESS>400, zero divergences. Thresholds apply to full;
  smoke records failures but remains explicitly exploratory. Nonfinite metrics
  fail and serialize as null, not NaN. Nonfinite draws or malformed axes raise.
- True RMST uses verified PH-Weibull at k=1.38,b0=11.42,HR=.7. Record all
  coverage indicators; require HR and delta RMST truth within equal-tail 95%.
- Full failure outputs are saved before nonzero exit; never reseed or tune.
  Existing output directory is rejected before sampling and by save_results.
- Optional output includes JSON settings/diagnostics/truth/timing and NPZ
  observations, chains, flags, grids and per-draw curves; no pickle. No default
  output path or committed generated-file churn. Core remains optional-free.
- Priors remain the maintainer's Normal(0,5); D-033 remains open. Nothing here
  claims prior-predictive justification or resolves that decision.

Checks (2026-11-02):
- `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.check_model`: exit 0. Trial/grid/aggregation,
  factor/plate/sites, analytical routes (error 0.0), audit/artifact checks pass.
  New checks: seeded iid Normal chains pass; a shifted gamma chain and a
  divergence fail; wrong HR interval fails; transposed flags fail; mocked
  r_hat=1.01 and ESS=400 fail all 24 metrics; NaN r_hat fails but JSON remains
  valid. Temporary output reload checks arrays/metadata; overwrite rejected.
- First full attempt stopped at ArviZ's daily home-cache stamp PermissionError.
  Reran same seed/settings with required escalation, passed. ArviZ FutureWarning
  and Matplotlib temporary-cache warnings observed, not hidden.
- Final command: `uv run --group examples examples/bayesian_rmst/run.py
  --config full --output /private/tmp/bayes3-final-full-20261102`, stdout in
  /private/tmp/bayes3-final-full-20261102.log: exit 0. Four parallel CPU chains,
  1000 warmup+1000 retained each, x64. Zero divergences; max R-hat 1.003555,
  min NumPyro ESS 2158.578, min tail ESS 2250.707 (across eight quantities).
  HR 95% [.5941096753,1.1688700067] includes .7; delta RMST 95%
  [-.0802711574,.2743181956] includes true .1701579210. All five truths inside
  intervals; P(delta>0)=.84725. Full audit passed, no failures.
- Final runtime before output 20.981s, sampling 2.249s/postprocessing 5.085s
  including compilation; imports/audit included in the total. This is measured
  on this CPU only, not a docs-build/platform runtime guarantee. Full live docs
  is feasible here; actual executable page build still needs verification.
- Independent reload of final NPZ: chains (4,1000), survival (201,4000,2),
  RMST curve (101,4000,2), event (300,), allow_pickle=False; checks pass.
- Smoke with --output /private/tmp/bayes3-smoke-20261102: exit 0; short-chain
  audit explicitly exploratory. Full was run twice with identical settings
  (second after adding runtime/JSON-failure telemetry), no reseeding/tuning.
- Ruff format/lint and mypy src pass (142 formatted, 22 typed source files);
  git diff --check passes. Initial lint lambda assignment fixed to def.
- No repeated core gate or docs build: source/core/tests/docs unchanged;
  prior integration wheel-backed acceptance and core runs remain historical.

Artifacts are outside git for review of code without generated-data volume:
/private/tmp/bayes3-final-full-20261102/{metadata.json,arrays.npz}.
Installed pinned optional environment is unchanged: NumPyro .22.0, ArviZ .23.0,
JAX/jaxlib .11.2, NumPy 2.5.3. No new dependencies.

Remaining: committed full results/figures and prior predictive assessment,
predictive KM plus external validation, executable docs. No claim those gates
are done; no clinical data, external messages, library sampling or new API.
