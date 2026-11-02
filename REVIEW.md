Independent review: round 1/3

# BAYES.4 — Survival and treatment-effect figures

Read PLAN.md BAYES.4 and both 02 use-case files; figures.py KM/prior/band helpers
then render_figures; check_figures.py numerical and rendered-figure checks.
Previous step/head 69629c9; original worktree base b1291a8. Maintainer explicitly
approves figures.py/check_figures.py because the existing files are near 300
lines. Two approved demo modules, 335 actual source/check lines excluding
plan/review, roughly the requested 300-line review unit. No library API, new
dependency, core, protected path, docs or spec edits. Generated output is not
committed here. Same Bayesian sub-slice, plan history retained.

Invariants:
- Figures use the passing full saved run, not a new posterior fit or mean
  parameters. Require config=full, audit.passed and exact committed prior
  metadata. No output overwriting; create a new directory only.
- Prior Predictive uses root=20261025, fold 2 then 2; 200 unconditioned prior
  sample sites from the existing model. Only survival evaluated, not extreme
  prior RMST. All curves must be finite; no dropping, clipping or reseeding.
- Figure 1: 200 thin curves per arm and observed KM; label Normal(0,5) on the
  three log parameters. D-033 remains open, not resolved by plotting.
- Figure 2: time-first saved survival (201,4000,2), reduce draw axis 1 for
  .025/.5/.975 quantiles, truth and observed per-arm KM. Pointwise bands,
  not simultaneous. Both survival panels use 0–20 years and common y scale.
- Figure 3: normalized posterior histogram of paired delta RMST(5), 95%
  equal-tail interval, truth and zero; honest histogram label rather than KDE.
- Figure 4: paired treatment-control RMST for every draw before quantiles;
  median/band on the specified tau grid, true delta and zero.
- Host KM uses distinct event times; all tied censorings remain in the risk
  set for event deaths. Step coordinates start at (0,1), extend flat to the
  arm's last observation and no farther. All-censored cohorts remain at one.
  Demo-only NumPy computation, not a differentiable public estimator.
- Matplotlib Agg backend; each figure closes after SVG/PNG output. Metadata
  records source run, prior key mapping/200 draws and Matplotlib version;
  prior_curves.npz retains the exact evaluated prior curves with grid.

Evidence (2026-11-02):
- `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.check_figures /private/tmp/bayes3-final-full-20261102`:
  exit 0: KM, quantile, figure and overwrite checks passed. Hand KM tied
  case has two deaths among four at t=1 (S=.5), last death at t=2 (S=0);
  all-censored [1,3] extends to 3 at S=1. Crossed-arm analytical band checks
  paired differences, axis and quantiles. Captured figures check 200 prior
  lines plus KM, density axis, delta lines; posterior/delta median ordinates
  compared against independent direct reductions of saved arrays. Four SVG
  and PNG files emitted, all nonempty; existing destination rejected.
- Final render command: `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.figures /private/tmp/bayes3-final-full-20261102
  /private/tmp/bayes4-final-figures-20261102`: exit 0. All four PNGs visually
  inspected (posterior/delta rechecked after style cleanup): no clipping,
  readable axes/legends and visible truth/intervals. PNG/SVG output retained
  there; source full arrays remain outside git as in BAYES.3.
- Prior probe: (201,200,2), no nonfinite curves; prior shape range approximately
  1.94e-6 to 1.66e5. Plot shows many nearly flat/step-like/extreme curves under
  chosen broad priors. No inference that these priors are justified; prior
  assessment remains a maintainer decision in D-033 and the eventual page.
- External one-off KM comparison (not fixture regeneration): R version 4.4.2,
  survival 3.7.0. Export full saved time/event/arm via np.savetxt to
  /private/tmp/bayes4-trial.csv. R command:
  `Rscript -e 'd <- read.csv("/private/tmp/bayes4-trial.csv"); library(survival);
  out <- do.call(rbind,lapply(0:1,function(g) { f <- survfit(Surv(time,event) ~ 1,
  data=d[d$arm==g,]); data.frame(arm=g,time=f$time,survival=f$surv) }));
  write.csv(out,"/private/tmp/bayes4-r-km.csv",row.names=FALSE)'`.
  Both arms compared at all 150 sorted original observed times: max absolute
  error 4.4408920985e-16 /5.5511151231e-16, atol=1e-14, rtol=0.
- First comparison used rounded R CSV times directly in searchsorted and
  landed before some exact jumps. That comparison failed (not a KM value
  failure). Verified R times against original sorted times at atol=1e-12,
  then evaluated each step at its original observed time; no epsilon, altered
  time or probability adjustment, nor tolerance relaxation. R outputs remain
  one-off evidence, not hand-written or approximated fixture values.
- `uv run ruff format --check .`: 144 files already formatted;
  `uv run ruff check .`: All checks passed!; `uv run mypy src`: 22 files clean;
  `git diff --check`: exit 0. Matplotlib temporary-cache/font warnings emitted.
- No repeated core gate, full NUTS or docs build for example-only additions.
  Dedicated checks run in x64 only because the Bayesian design requires it.

Remaining: posterior predictive KM figure 5, committed full artifact delivery,
executable page/rendered HTML and prior interpretation. This step does not
claim those gates or change D-033. No patient data, network acquisition,
library event-time sampling or publishing.
