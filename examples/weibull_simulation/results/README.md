# Committed full-study evidence

Six scenarios, 1000 replicates per model, root seed 20261025. Censoring
limits are prespecified at 50 and 20 years (spec revision 47). The numerical
run used clean source `ac0fa01337ff37c05505292b3c7584e3f2f11cb1`; original
commands, versions, prespecified limits, realized censoring and shared batch timings
are preserved in `metadata.json`. Its original output path is historical.

Under maintainer-approved D-044, Git stores compact evidence:

- `performance.csv`: all 12 scenario/model summaries and failure counts.
- `failures.csv`: every rejected ID, separate reason flags, overall `failed`,
  maximum absolute gradient and fitted log HR. Accepted IDs are the complement
  within each group’s `0..attempted-1` replicate IDs.
- `sampling_histograms.json`: exact integer counts for every accepted HR,
  with common stored edges (about .025 wide); bins are left-closed/right-open,
  except the final bin includes its right edge. There is no subsampling.
- `illustration.csv`: the first replicate accepted by both models at n=300,
  censoring limit 20 years (scenario 3, replicate 0), with complete fit parameters.
- `evidence.json`: source/output checksums and the reducer’s commit, path,
  file-dirty status and exact source hash.

All 137 rejections fail the strict max-absolute-gradient <1e-5 gate, despite
successful solver status and positive-definite Hessians. They stay rejected.
Weibull coverage at n=1000 is 94.26% and 93.60%, both within three observed
Monte Carlo SEs of 95%; denominators include accepted fits only.

Full per-fit diagnostics remain local and are ignored at this results path.
To produce a fresh run and compact bundle from the repository root:

```bash
uv run --locked --group examples python examples/weibull_simulation/run.py --config full --out /tmp/survjax-weibull-full
uv run --locked --group examples python examples/weibull_simulation/_evidence.py --results /tmp/survjax-weibull-full --out examples/weibull_simulation/results
JAX_ENABLE_X64=1 uv run --locked --group examples python -c "from pathlib import Path; from examples.weibull_simulation._plots import write_figures; p = Path('examples/weibull_simulation/results'); write_figures(p, p)"
```

Figures: [survival](fig1_survival.svg), [HR sampling](fig2_sampling.svg),
[coverage](fig3_coverage.svg). Survival colours identify arms; other panels
use colours for models, as their legends state.

The provenance check reads the reducer commit from local Git history.
Timings change on reruns. Bitwise equality across machines is not promised.
All data are synthetic.
