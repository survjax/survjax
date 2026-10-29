# Agent: demo foundation and Weibull simulation

Read the packet README, spec 4.4, 7.3, 10.3, 11.4, 12.2–12.3 and
spec/use-cases/00-common.md plus BOTH 01-weibull-simulation files.
Own examples/weibull_simulation/, its dedicated checks and
docs/use-cases/weibull-simulation.md. A9 owns the mini README example;
coordinate that excerpt but do not take over its benchmark or evidence audit.

1. Environment proposal: obtain approval for the examples dependency group,
   exact tested package versions and lockfile changes. Own the single group
   for Optimistix/Matplotlib, Equinox/Optax, NumPyro and diagnostics tooling
   actually needed; stage additions as approved. No new core dependency.
   Coordinate docs execution environment with the maintainer; current docs-only
   command does not select examples. Do not change protected build scripts.
2. Shared generator/calibration in study.py: Weibull inverse-CDF event times,
   independent Uniform(0,c) censoring and numerical calibration. Keep it demo
   code with the shared use-case semantics, accepting row-broadcast k/b and
   explicit keys. Test generation against the analytical survival law and
   censoring target, including empirical recovery. Freeze the reviewed import
   contract, communicate it to both demo agents and merge before they consume it.
3. After PH lands: fit one replicate for both models with Optimistix BFGS and
   exact Hessians; check analytical likelihood agreement and fit acceptance.
4. Compiled vmapped scenario runner, failure accounting, MCSE/performance
   summaries and smoke configuration. Follow the details' starts, tolerances,
   key scheme, failure categories and accepted-fit denominators exactly.
5. Full-run evidence, split into reviewable commits if outputs exceed budget:
   six scenarios, 1000 replicates/model; coverage at n=1000 within 3 MCSEs.
   Record every failed/unestimable fit, calibrated/realized censoring, versions,
   CPU and synchronized first-call/warm timings. Investigate failed thresholds.
6. Separate executable docs page: live smoke (n=100,R=20,two censoring levels)
   and labelled committed full results/figures, using the specified run.py
   --config {smoke,full} --out interface. Inspect the rendered HTML.

No real bc rows, R benchmark, library sampling API or public generator module.
Shared generator changes after handoff require coordination with both consumers.
