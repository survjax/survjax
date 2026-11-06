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

The saved bundle retains its original provenance manifest. The simplified
runner writes performance, failures, illustration parameters, histograms,
metadata and figures directly; it does not export bulk optimizer diagnostics
or build a separate audit manifest. To reproduce the study in a new directory:

```bash
uv run --group examples python -m examples.weibull_simulation.run --config full --out /tmp/my-weibull-study
```

Figures: [survival](fig1_survival.svg), [HR sampling](fig2_sampling.svg),
[coverage](fig3_coverage.svg). Survival colours identify arms; other panels
use colours for models, as their legends state.

Timings change on reruns. Bitwise equality across machines is not promised.
All data are synthetic.
