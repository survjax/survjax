# Slice B synthetic likelihood reference generator

B3.1 supplies the generator; B3.2 commits its outputs and Python reference tests.
No patient data, fitted parameters or derivatives appear.

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

The JSON follows the independent tester's schema documented in
`tests/spec/fixtures/README.md`. The acceptance copy at
`tests/spec/fixtures/slice_b_likelihood.json` is byte-identical to this directory's
JSON; the reference test checks that identity. `SHA256SUMS` records both payload
hashes. The shared `r_fixture("slice_b")` loader verifies those hashes, pinned
provenance and CSV/JSON alignment before any comparison. Regular tests need no R.

The committed outputs came from two byte-identical runs of the B3.1 generator
under the pinned environment. To regenerate, run the command above into a fresh
temporary directory, compare the outputs, then replace both payloads, update
`SHA256SUMS`, and install the identical acceptance JSON with maintainer approval.
Run `tests/reference/test_slice_b_r.py` and the independent
`tests/spec/test_observation_censoring_flexsurv_fixture.py` with all three family
features enabled in float64 before accepting new outputs. The metadata's
`survival` spelling `3.7-0` follows the acceptance schema; R reports `3.7.0`.

Excluded comparisons: left/interval censoring with left truncation, right
censoring under finite U, sets beyond U, censoring at/before entry, observations
outside the window and exact zero. flexsurv's conventions differ there; use
independent mathematics. Compare these R outputs with survjax in float64 only.
