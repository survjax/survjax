# Synthetic bc-like Weibull R reference

The committed `data.csv`/`expected.json` were generated and verified in A8.3.
No patient rows or `flexsurv::bc` are loaded. No survjax agreement is claimed;
that comparison is A8.5.
From the repository root, use R 4.4.2, flexsurv 2.3.2 and survival 3.7-0
(R reports `3.7.0`), with no extra export dependencies:

Running the generator without arguments overwrites `data.csv`, `expected.json`
and `bc_synthetic_weibull.json` in this directory.

The A8.3a generator now also writes the acceptance-schema JSON. Its committed
payload and updated provenance/hashes are deferred to A8.3b. Until then,
`expected.json` below differs in generator/exporter provenance; the observations
and reference numbers are preserved. The current manifest still verifies the
two original committed payloads.

```sh
Rscript --vanilla scripts/fixtures/generate_bc_like.R --check-only
Rscript --vanilla scripts/fixtures/generate_bc_like.R /tmp/bc-like-run1
Rscript --vanilla scripts/fixtures/generate_bc_like.R /tmp/bc-like-run2
cmp /tmp/bc-like-run1/data.csv /tmp/bc-like-run2/data.csv
cmp /tmp/bc-like-run1/expected.json /tmp/bc-like-run2/expected.json
cmp /tmp/bc-like-run1/bc_synthetic_weibull.json /tmp/bc-like-run2/bc_synthetic_weibull.json
cmp /tmp/bc-like-run1/data.csv tests/fixtures/bc_like/data.csv
cmp /tmp/bc-like-run1/expected.json tests/fixtures/bc_like/expected.json
(cd tests/fixtures/bc_like && shasum -a 256 -c SHA256SUMS)
```

Seed 8201, Mersenne-Twister/Inversion/Rejection: 686 Weibull draws from the
recorded fitted vector in spec 11.1, then independent Uniform(0.5, 7.5) year
censoring. Good/Medium/Poor blocks have 229/229/228 rows. Realized exact/right
counts are recorded, not forced to match the cohort; no truncation is used.

Cases `fitted`, `unit_shape` and `alternative` are the spec 11.1 fitted vector,
`(1,10,-0.5,-1)` and `(1.5,12,-0.6,-1.2)`, in shape/baseline-scale/Medium/Poor
order. Each stores natural parameters, `scale_by_group`, `mu_by_group` and
`sigma`: `b_g=b_Good*exp(beta_g)`, `mu_g=log(b_g)`, `sigma=1/shape`.

Installed `?flexsurvreg` documents natural `inits`, `fixedpars` and likelihood
outputs; `?summary.flexsurvreg` documents paired predictions (`cross=FALSE`).
`?survreg`, `?survreg.control` and `?dsurvreg` document fixed scale, zero
iterations and AFT density/CDF evaluation. The survreg fit uses `offset(mu)`
with a zero intercept. Singleton and full-data totals check the composed
density/CDF contributions. The two reference paths are not independent.

CSV: `row_id,time,event,group`, years, event 1 exact/0 right. Schema 1 JSON:
`header`, `row_id`, `grid`, `cases`. Each case has separate flexsurv/survreg
log-likelihood, log-survival and survival vectors (686), a total, and a nested
`survival_grid` array (141 time rows, three Good/Medium/Poor columns).
`grid` records times 0–7 by 0.05, groups and shape. Provenance includes RNG,
versions/platform, generator/CSV MD5 hashes, simulation and realized counts.
Each column is checked against a separate prediction for its exported label.
Use the stored `grid.time` values in reference tests: rebuilding R's `seq`
grid with JAX arange/linspace can change some times by one ULP.

Grid survreg values come directly from `psurvreg` at supplied AFT locations.
They are cross-checked with flexsurv; singleton-fit checks cover observed-time
likelihoods, not grid cells.

Unknown flags fail before writing. Check-only validates references and labels;
full export also validates exact CSV/JSON text and 17-digit numeric formatting.
As in A8.1, the pinned R decimal parser can lose one ULP: exact binary numeric
read-back is verified separately with Python during validation, never required
at generator runtime. Both payload files were byte-identical across two runs
on the recorded platform. Repeat the commands above for regeneration. Normal
tests only read committed files.

## Independent acceptance export

`scripts/fixtures/bc_spec_export.R` maps the evaluated R objects to the schema
defined in `tests/spec/fixtures/README.md`; it does not reparse decimal files
or calculate new reference values. The seed remains 8201. `unit_shape`,
`alternative` and `fitted` become `fixed_a`, `fixed_b` and `r_optimum`.
The file embeds observations, each source's per-row likelihood and survival,
and flexsurv curves with time rows and Good/Medium/Poor columns. R checks the
fixed-vector labels independently of the mapping. Check-only also builds and
checks the acceptance payload without writing files. The original header's
checksums now include the exporter as well as the generating script.

After A8.3b verifies and commits the candidate under this directory, the
maintainer supplies that exact file to the authoritative tester repo for the
normal spec-test sync. The generator never writes into `tests/spec/`; the
independent loader and tests remain unchanged. Neither this export addition
nor its R-side checks claim survjax agreement.

## Generation provenance

Generated on 2026-10-21 from the unchanged generators at commit
`8e8a96a8302f5706ce3434152ddabab146b9881a` on `aarch64-apple-darwin20`.
Rows: 686; exact: 293; right-censored: 393.
Good/Medium/Poor events: 52/91/150, simulated counts rather than cohort counts.
R/package versions and generator/CSV MD5 hashes are also stored in the JSON
header. SHA-256 hashes of both committed payloads are recorded in
[SHA256SUMS](SHA256SUMS). The command above verifies both files. The A8.4
loader will verify this manifest before parsing either payload.
