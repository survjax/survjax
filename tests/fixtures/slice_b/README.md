# Slice B synthetic likelihood reference generator

B3.1 supplies the generator; payload integration and Python reference tests
are a separate step. No patient data, fitted parameters or derivatives appear.

From the repository root, with R 4.4.2, flexsurv 2.3.2 and survival 3.7-0:

```sh
Rscript --vanilla scripts/fixtures/generate_slice_b.R --check-only
Rscript --vanilla scripts/fixtures/generate_slice_b.R /tmp/survjax-slice-b
```

The second command writes `data.csv` and `slice_b_likelihood.json`. Six fixed
parameter vectors share fifty seeded rows: five in each of the ten comparable
cells in spec 10.4 revision 60. Boundaries include `t=U` and `b=U` where allowed.
All 300 exported values come from `flexsurvreg(..., fixedpars=TRUE)$logliki`.
Before writing, the generator checks row order, fixed parameters, totals,
section-2 algebra, hazard <=15, minimum mass >=1e-6 and relative width >=1e-4.
Warnings abort generation. Base R exports 17 significant digits without a JSON
dependency; metadata records the seed, versions, platform and generator MD5.

The JSON follows the independent tester's schema on main at c1a6a8d, documented
in `tests/spec/fixtures/README.md`; its eventual acceptance destination is
`tests/spec/fixtures/slice_b_likelihood.json`. That protected copy belongs to the
maintainer/spec-test integration workflow. This step writes only temporary
outputs; it does not modify `tests/spec/`. After merging main, all 73 independent
fixture tests passed against the staged JSON via a temporary fixture-directory
override. This does not establish that the committed-fixture gate passes.

Excluded comparisons: left/interval censoring with left truncation, right
censoring under finite U, sets beyond U, censoring at/before entry, observations
outside the window and exact zero. flexsurv's conventions differ there; use
independent mathematics. Compare these R outputs with survjax in float64 only.
