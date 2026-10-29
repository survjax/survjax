# Agent: Bayesian survival with posterior RMST

Read the packet README, spec 8.7, 11.3, 12.1–12.2, shared use-case conventions
and BOTH 02-bayesian-survival files. Own examples/bayesian_rmst/, dedicated
optional-environment checks and docs/use-cases/bayesian-rmst.md.
Start final implementation after generator/environment, PH and RMST merge.
Do not copy temporary PH/RMST implementations or modify core library files.

1. Model/factor wiring and pilot: fixed synthetic n=300 two-arm trial, exact
   keys/priors from the details, generator's 55% censoring calibration. Verify
   prior log density + summed survjax likelihood equals NumPyro log_density
   to 1e-10. Keep row factors inside the observation plate; never sum there.
   Pilot timing decides live full sampling versus live smoke + committed full
   results. Resolve exact smoke settings in PLAN.md before implementation;
   seek a maintainer decision for choices not settled by the design.
2. NUTS full run: four chains, 1000 warmup + 1000 draws; record chain method,
   versions/settings/timing. Require R-hat<1.01, bulk/tail ESS>400 and zero
   divergences. Investigate truth outside specified intervals; never re-seed
   or silently tune to obtain a pass. Request dependencies from foundation owner.
3. Batched posterior summaries: route A on (S,2) distributions; route B through
   Predictive deterministic sites behind compute_rmst=False during NUTS.
   Match RMST routes to atol=1e-12 after axis alignment. Compute curves and
   summaries per draw before aggregation; enforce the specified grids/tau bound.
4. Prior/posterior plots and predictive KM check with 100 replicated datasets.
   Demo-only KM/event generation; cross-check KM against R survfit or lifelines
   once with recorded evidence. Include the posterior predictive check required
   by the spec; dropping it requires a maintainer/spec decision.
5. Executable docs topic, diagnostics/truth table and specified figures; live
   run within 120 seconds or clearly labelled committed full fallback. Record
   actual thresholds and investigate failures. Review rendered HTML.

No NumPyro adapter, public KM, medians, library sampling or clinical data.
No A9 benchmark/example edits; no dependency or shared generator ownership.
