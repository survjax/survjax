Independent review: round 1/3

# BAYES.2 — Batched posterior RMST and per-draw summaries

Read first:
1. PLAN.md BAYES.2; spec 11.3 and 02-details Route A, Sampling, Grids.
2. model.py PosteriorDraws, time_grids, posterior_draws, check_rmst_routes,
   summarize_draws (the existing model/priors are unchanged).
3. check_model.py check_aggregation_and_grids and check_posterior_routes.
4. run.py postprocessing and output metadata.

Scope: 234 changed example/check lines (plan/review records excluded), no new
module, public library API, dependencies, core, protected paths or docs.
Maintainer approves the earlier BAYES.1 work and explicitly requests this next
step. Same Bayesian sub-slice: its PLAN records remain in place. Previous step
head 12bc641; worktree original base b1291a8. Prior independent review is in
review/slices/BAYES.1.md, round 2. Its integration findings F-001/S-001 remain;
D-028 remains open for the maintainer, and the committed Normal(0,5) priors are
preserved. No claim that they match the delegated design.

Invariants:
- Flatten chains only through sampler.get_samples(); no hand flatten order.
- Route A: base (S,1), PH log-HR (S,2), RMST(5) (S,2), RMST grid (m,S,2),
  survival grid (K,S,2). Arms are always control then treatment.
- Route B uses conditioned parameters and deterministic sites outside the
  observation plate, enabled only after NUTS. Its (S,m,2) curve is swapped
  to (m,S,2). Require identical shapes and finite equality at absolute-only
  1e-12 (equal_nan=False). Return the maximum absolute discrepancy.
- Differences are paired per draw before mean/SD/quantiles/P(delta>0).
  SD uses ddof=1; intervals are equal-tail 2.5/97.5 percentiles. Curves stay
  draw-level for later plotting rather than evaluating at mean parameters.
- Host grid setup uses observed lower (valid for this exact/right-only demo):
  p90 rounded down to half-years; reject tau_max<5 or nonfinite. Grids are
  201 points on [0,20] and 101 points on [0,tau_max], including zero.
- JAX distribution evaluation is followed by host NumPy aggregation. These
  host helpers are demo-only; no jit/derivative promise is made for them.
- Sampling timing still measures sampling/compile only. New postprocessing
  timing includes both summary routes/aggregation and synchronizes output.

Evidence (2026-10-31):
- Explicit `JAX_ENABLE_X64=1 uv run --group examples python` process called
  check_trial(), check_aggregation_and_grids(), time_grids(make_trial().data):
  exit 0. Trial c=19.1725354361981; arithmetic and grid checks passed;
  actual trial grids (201,)/(101,), tau_max=13.0 years.
- Hand arithmetic check: deltas [1,-1,0], mean 0, sample SD 1,
  95% interval [-.95,.95], P(delta>0)=1/3; k draws [1,3,5] have sample SD 2.
  This checks pairing, strict positivity and aggregation independently of PH.
- Grids: p90([1,6])=5.5 and p90([1,6.4])=5.86 both give tau_max=5.5;
  short follow-up [1,4] raises. Exact expected linspaces check all points.
- `JAX_ENABLE_X64=1 uv run --group examples python -c 'from
  examples.bayesian_rmst.check_model import check_posterior_routes;
  check_posterior_routes()'`: exit 1, AttributeError: survjax has no
  ProportionalHazards, before RMST/survival evaluation. These analytical
  exponential and route-agreement assertions are implemented, NOT verified.
- `uv run ruff format --check examples/bayesian_rmst`: 3 files formatted.
- `uv run ruff check .`: All checks passed! (initial import-order lint fixed).
- `uv run mypy src`: Success: no issues found in 20 source files.
- `git diff --check`: exit 0.
- Direct-file and module runner --help both exit 0 with smoke/full choices.
- `scripts/check.sh --through A10`: exit 1 at unchanged README.md:67/72
  embedded-code format failure after guard ok. Later checks did not execute.
- Ordinary core suites are not repeated: no source/core/test changes; the
  previous step's suites were 1065/1071 passing in x64/x32. This is historical
  evidence, not a run for this commit. Float32 demo checks are not applicable:
  the design requires float64.

Requirements mapped to checks:
- 02-details Grids: full expected linspaces, exact/rounded endpoint and
  insufficient follow-up failure.
- Spec 11.3 per-draw effects: paired differences and independently derived
  summary statistics. RMST/survival batch axes: analytical shape-one cases
  with distinct scales/HRs (pending PH execution).
- Spec 11.3 route agreement: Predictive alignment and strict absolute 1e-12,
  with zero in the grid (pending PH/RMST execution).

Unknown / later steps:
- All PH-dependent execution, the approved smoke/full posterior, route errors
  and timings remain unverified until integration. No substitute core is added.
- Full convergence/truth audit, per-summary diagnostics, prior/posterior plots,
  posterior predictive KM, external KM validation and executable docs/results
  are later work. This step records means/intervals/probability, not those gates.
- Full smoke/module execution is not repeated: the PH API is still absent and
  the explicit posterior check confirms that boundary. No NUTS result claimed.
