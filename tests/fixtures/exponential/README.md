# Synthetic exponential R reference (A8.1)

This directory currently contains the generator handoff only. `data.csv` and
`expected.json` are produced and reviewed separately in A8.3; no numerical
agreement with survjax is claimed yet. No patient data are loaded.

From the repository root, use R 4.4.2, flexsurv 2.3.2 and survival 3.7-0
(reported by R as `3.7.0`). No additional R packages are needed for export:

```sh
Rscript --vanilla scripts/fixtures/generate_exponential.R --check-only
Rscript --vanilla scripts/fixtures/generate_exponential.R
# Optional destination for independent regeneration (overwrites payload files):
Rscript --vanilla scripts/fixtures/generate_exponential.R /tmp/exponential-run1
```

Unknown `--` options fail before simulation or output. `--check-only` checks
the references and case labels; a full run also checks the export: CSV columns
and decimal text must read back identically, JSON numbers must match 17-digit
formatting, and the written JSON must match the encoded payload. The pinned
macOS R decimal reader can lose one ULP on correct 17-digit text, so numeric
round trips are checked separately against R hexadecimal values with Python
during validation. Python is not required by the generator.

Seed 8101 with Mersenne-Twister/Inversion/Rejection generates 16 event times
from an exponential rate of 0.35/year and independent Uniform(0.5, 5) year
censoring. Simulation order is preserved; `event=1` means exact and `0` right
censored. There is no truncation. Rates `slow=0.1`, `generating=0.35` and
`fast=1.2` are evaluated without optimization. For survreg, `mu=-log(rate)`
is log mean time, and its exponential scale is fixed at 1.

Installed documentation establishes the evaluation paths: `?flexsurvreg`
documents `inits`, `fixedpars`, `logliki` and `loglik`; `?summary.flexsurvreg`
documents survival/cumulative-hazard summaries. `?dsurvreg` documents density
and CDF evaluation in survreg's parameterization; `?survreg` and
`?survreg.control` document initial values and the iteration limit. See also
the [flexsurv reference](https://chjackson.github.io/flexsurv/reference/flexsurvreg.html)
and [survival manual](https://therneau.r-universe.dev/survival/doc/manual.html).
The generator asserts unchanged parameters and zero iterations, compares
density/CDF contributions with singleton and full-data survreg totals, then
compares both packages before writing. These are not fully independent oracles.
CDF complements are used only for these checked finite interior cases.

CSV columns are `row_id,time,event`. Schema 1 JSON has `header`, `row_id` and
named `cases`. Each case records natural/log/AFT parameters and separate
`flexsurv`/`survreg` objects with row-aligned `log_likelihood`, `log_survival`,
`survival` vectors of length 16 and a scalar `total_log_likelihood`. The header
records RNG, versions, platform, units, realized counts, evaluation paths and
MD5 checksums of the CSV and generator. Numbers use 17 significant digits;
nonfinite values fail checks. There are no timestamps or destination paths.
In A8.3, generate twice into separate destinations and byte-compare both files
on the same platform before committing. Normal tests will only read outputs.
