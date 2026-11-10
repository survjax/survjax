# AFT — Direct distribution wrapper

Starting commit / previous work / original worktree base: cc0e30f.
Plan archival commit: 64b8393. Branch: aft.
Maintainer scope: this worktree targets AFT only. Generic RMST, other families,
summary derivative infrastructure and later overrides are owned elsewhere.
AFT.1 starts from approved-plan commit 7b5b70f; previous completed work is
cc0e30f, also the original worktree base. AFT.1 implementation and validation are complete. Three independent review
rounds concluded with maintainer disposition of F-2: remove the redundant
boundary substitution; follow-up checks passed.

## AFT.3 — AFT documentation

Documentation implemented and validated; independent review pending.
Start and previous-step commit 607a11b, original
worktree base cc0e30f. Files: docs/accelerated-failure-time.md (one tutorial
in the existing distribution guides), docs/index.md (navigation/status),
docs/api.md (link from the existing AFT entry), PLAN.md and REVIEW.md.
No API/docstring/test changes. Follow the PH guide's mathematical exposition,
with a synthetic Weibull time-ratio example, one survival plot, caller-owned
regression predictors, right-censored likelihood and draw/arm broadcasting.
Explain the Weibull PH relationship and theoretical quantile scaling without
calling unsupported summary APIs. Avoid assertions and property-test-style
examples. Keep docs within 300 changed lines; execute both dtype builds with
scripts/build_docs.sh, inspect rendered pages, then independent review.


One bounded topic in the existing distribution documentation, plus PLAN.md and
REVIEW.md; correct docstrings only if wrong. Choose the exact existing page at
step start. Cover construction, sign convention, caller-owned predictors,
Weibull equivalence, broadcasting, grids, likelihood and currently unsupported
summaries. No generic-RMST design or new family documentation in this worktree.
Spec: 4.4, 5, 10.5. Run scripts/build_docs.sh directly and inspect the rendered
changed topic. Commit, independent review, then stop.

## AFT.2 — AFT numerical and integration verification

Active step, starting/previous-step commit 09148c5; original worktree base
cc0e30f. Implementation complete: 283 lines of tests, no library changes.
Full timing regeneration changes 450 generated lines (36 additions/414
removals); keep that generated metadata in this step, following the maintainer's
previous preference to include the regenerated list in the same commit.
 Implement focused tests in tests/unit/test_aft_integration.py (about
300 lines); wrappers.py only if these expose a settled-contract defect.
Test large finite predictors with representable transformed times, a custom
improper baseline and generic derivatives, nested wrapper grids, mixed
invalid likelihood rows, and built-in likelihood values/Jacobians/mixed
Hessians against explicit formulas. Reuse existing tolerances. Add no API,
dependencies or documentation. Record validation and review in REVIEW.md;
regenerate tests/slow_tests.txt if newly measured cases exceed one second.

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

## AFT.1 — Registered wrapper and direct methods

Maintainer follow-up (2026-11-09, F-2): remove the redundant pre-log time
and predictor substitutions from _rescale_time. Keep boundary restoration,
the pre-exp overflow guard and numerical-failure detection. Files: wrappers.py,
test_aft.py (correct the helper test's explanation; keep every assertion),
PLAN.md and REVIEW.md. No API changes. Spec 4.4–4.5 and 7.1–7.3.
Run all AFT tests in both dtype modes and the fast A10 gate. All three review
rounds are used; this implements the maintainer's explicit disposition of F-2.

Files:
- src/survjax/distributions/wrappers.py
- src/survjax/distributions/__init__.py and src/survjax/__init__.py
- tests/unit/test_aft.py
- tests/unit/test_import.py and tests/install/test_wheel.py (public export lists)
- docs/api.md (single autodoc entry explicitly authorized during AFT.1)
- tests/slow_tests.txt (required full timing regeneration; maintainer explicitly
  allows its 310 changed lines in the single AFT.1 commit)
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

AFT.3 now covers the direct-wrapper tutorial, with minimal navigation and
API-guide links. No remaining documentation topic in this AFT plan.

AFT.1 implemented and validated; review rounds complete, F-2 disposed by
maintainer-directed removal with passing follow-up checks.
AFT.2 complete; independent review round 1 verdict merge (607a11b).
AFT.3 written and validated; independent review pending.
No further implementation sub-steps remain after AFT.3.
