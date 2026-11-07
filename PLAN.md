# AFT — Direct distribution wrapper

Starting commit / previous work / original worktree base: cc0e30f.
Plan archival commit: 64b8393. Branch: aft.
Maintainer scope: this worktree targets AFT only. Generic RMST, other families,
summary derivative infrastructure and later overrides are owned elsewhere.
This is planning only; no implementation step has started.

## AFT.1 — Registered wrapper and direct methods

Files:
- src/survjax/distributions/wrappers.py
- src/survjax/distributions/__init__.py and src/survjax/__init__.py
- tests/unit/test_aft.py
- PLAN.md and REVIEW.md

Public API: AcceleratedFailureTime(baseline=..., log_time_ratio=...).
Spec: 4.1–4.2, 4.4–4.5, 4.7, 5, 7.1–7.3.
Implement a frozen registered family with a nested baseline and a real predictor.
Construction is keyword-only; broadcast nested leaves and predictor using the
existing registration machinery, preserving dtype policy and issue prefixes.
Provide reference_time and the baseline infinity limit, plus:
- log_survival(t) = baseline.log_survival(t * exp(-eta));
- log_prob(t) = baseline.log_prob(t * exp(-eta)) - eta;
- log_hazard(t) = baseline.log_hazard(t * exp(-eta)) - eta.
Use safe time rescaling rather than unguarded products; preserve baseline
boundary limits, invalid-entry isolation and the direct-method value policy.
Inherit other direct methods, time-first grids, indexing and likelihood.
Docstrings explain positive eta means longer times (versus higher hazard in PH).

Tests accompanying implementation: keyword-only construction, broadcast shape,
pytree behavior, nested diagnostics, Weibull and Exponential value identities,
zero/negative/infinite/NaN times, eager/jit/vmap agreement and smooth-interior
parameter derivatives. Expected values come from mathematical identities.
No RMST override: retain the inherited explicit unsupported behavior until the
shared generic backend lands. No mean, quantile, inverse-hazard or sampling work.
Keep wrappers.py around 300 lines and the implementation diff around 300 lines;
if it cannot fit, propose a smaller step or module split before proceeding.

## AFT.2 — AFT numerical and integration verification

Files: wrappers.py if corrections are required; focused implementation-owned
AFT test files; tests/impl_tolerances.py only for justified new tolerances;
PLAN.md and REVIEW.md. No additional public API.
Spec: 4.4–4.5, 5, 6.3, 7.1–7.3, 12.1.
Extend coverage to:
- large finite predictors and rescaling tails, including representable results
  where forming exp(-eta) alone overflows or underflows;
- a custom baseline, including an improper infinity limit;
- time-first grids, nested batch shapes and mixed valid/invalid entries;
- likelihood equivalence to Weibull with scale b * exp(eta), and Exponential
  with rate r * exp(-eta), in numerically suitable identity cases;
- first/second baseline and predictor derivatives, including mixed Hessians,
  checked against analytical identities or mpmath, with jit/vmap agreement.
AFT.1 must already obey the numerical contract; AFT.2 broadens evidence.
Do not extend the PH-specific derivative exception to AFT. Log any unresolved
spec issue and stop rather than introducing clipping or weaker guarantees.
Keep this step within approximately 300 changed lines; split if needed.

## AFT.3 — AFT documentation

One bounded topic in the existing distribution documentation, plus PLAN.md and
REVIEW.md; correct docstrings only if wrong. Choose the exact existing page at
step start. Cover construction, sign convention, caller-owned predictors,
Weibull equivalence, broadcasting, grids, likelihood and currently unsupported
summaries. No generic-RMST design or new family documentation in this worktree.
Spec: 4.4, 5, 10.5. Run scripts/build_docs.sh directly and inspect the rendered
changed topic. Commit, independent review, then stop.

## Validation and review for each code step

Run all touched test files in both dtype modes (Tier 0), then
scripts/check.sh --fast --through A10 (Tier 1). Record exact commands/results;
the existing A10 gate alone does not establish AFT coverage.
Update tests/slow_tests.txt if new tests exceed one second.
Write REVIEW.md, commit, run independent-review, address findings within at
most three rounds, and stop after each step. Planning needs no review.
No protected paths, dependencies, spec edits or publishing are in scope.

## Handoff to the generic RMST worktree

AFT direct methods can be developed independently of generic quadrature.
The receiving worktree can integrate AFT survival through the shared backend
without an AFT-specific rmst override. Validate the integral against
exp(eta) * baseline.rmst(tau * exp(-eta)) for supported baseline references,
and differentiate with respect to both nested baseline parameters and eta.
That integration, its tests, SummaryOptions/SummaryInfo, the Weibull quadrature
tangent, log-logistic and any later AFT summary scaling override are outside
this worktree. AFT completion does not claim full Slice D or summary support.

## Docs to-do

AFT.3 closes the direct-wrapper documentation topic after the code steps.

Remaining: AFT.1, AFT.2 and AFT.3. No implementation or checks run yet.
