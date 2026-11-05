# Bayesian survival demo

All observations are synthetic. Core survjax installation does not need the
optional examples environment. Run these commands from the repository root.

## Generate a full posterior

```sh
uv run --group examples examples/bayesian_rmst/run.py --config full --output /tmp/bayes-full
```

Use a new output directory. `--config smoke` uses four chains with 100 warmup
and 100 retained draws each; full uses 1000 of each. Smoke diagnostics are
exploratory. Full diagnostics and truth-coverage failures produce a nonzero
exit after saving valid audit evidence. Malformed/nonfinite inputs are hard
errors before an audit exists. Do not reseed or tune until a failure disappears.

`metadata.json` records priors, seed, versions, timings, chain method,
per-quantity diagnostics and truth coverage. `arrays.npz` contains chain-grouped
parameters, observed time/event/arm, grids, draw-level RMST/survival and
NUTS divergence flags. Load numerical output with `allow_pickle=False`.

## Render the five figures

```sh
JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.figures /tmp/bayes-full /tmp/bayes-figures
```

The renderer requires a passing full run and a new destination. It emits five
SVG/PNG pairs, `figures.json`, `prior_curves.npz`, and `predictive_trials.npz`.
Figure 1 displays all 200 prior draws; figures 2–4 use posterior draw-level
curves and paired treatment effects, not calculations at mean parameters.

Figure 5 overlays observed per-arm Kaplan–Meier curves on 100 replicated
trials. Replicate `i` selects flattened posterior draw `floor(i*S/100)`, using
the original chain-major order. With 4000 draws, 25 replicates come from each
chain. The exact indices are saved with the replicated observations.

Keys start at root 20261025: fold 2 for this use case, then 3 for posterior
prediction, then the replicate index. The shared generator splits this key
into independent event and censor uniforms. The observed arm assignment stays
fixed, while event times and censoring are regenerated for every replicate.

The censor limit is the **known synthetic** calibration from the original
trial, not an estimated censoring model. This simplification cannot be assumed
for real data. All 100 curves per arm are shown through their last observed
follow-up. This is a visual predictive check, not a calibrated p-value or a
proof of model correctness. Random tail variation is expected with censoring.

Run the explicit optional checks with the same examples environment:
`JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.check_figures /tmp/bayes-full`.

The three log parameters currently have the maintainer's Normal(0,5) priors.
Their often extreme prior curves are retained for assessment; D-033 remains
open. Recorded full artifacts are in `results/`; the executable page is
`docs/use-cases/bayesian-rmst.md`. The page labels live smoke separately from
the recorded full tables and figures.
