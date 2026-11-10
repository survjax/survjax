# B3.1 — Slice B fixed-parameter R generator

Independent review: round 1/3

Read work/B3.1/PLAN.md; spec 10.4 (fixture constraints, rev 60; unchanged by
rev 61); tests/spec/fixtures/README.md (schema); the generator; then its README.
Implementation base f9c9b0c, merge/previous checkpoint 6fafbaa, original base
cc0e30f. Only generator, README and this step's records change. No public API,
library code, dependency, protected file or independent assertion changes.

Invariants: one shared dataset (50 synthetic rows, seed 20261110), five rows in
each of the ten comparable cells, six fixed natural parameter vectors across
three families (including Weibull shape 1). Exported values come only from
flexsurvreg(fixedpars=TRUE)$logliki. The generator asserts fixed parameters,
row order, totals, positive masses, width/hazard bounds and required boundary
coverage before export. R warnings abort. Its section-2 algebra checks are not
substituted for flexsurv results. Base R exports 17 significant digits; missing
time2 and infinite U become JSON null. No patient data or fitted parameters.

Checks performed:
- R 4.4.2, flexsurv 2.3.2 and survival 3.7.0 were verified. Initial sandboxed
  R loading failed on OpenMP shared memory; the approved unsandboxed run worked.
- R parsed and executed the generator successfully. Two generation runs to
  /tmp/survjax-slice-b-run1 and run2 each passed all generator checks and
  produced byte-identical CSV/JSON. No R-derived values were written by hand.
- Independent 80-digit mpmath audit of all 300 values: maximum absolute error
  4.972287541523497e-15, within 1e-10 + 1e-8*abs(expected). It also checked
  row counts, hazard/mass/width constraints and exact CSV/JSON input agreement.
- Incoming tests/spec/test_observation_censoring_flexsurv_fixture.py: 73 passed
  in float64 with both observation and both family features enabled. A temporary
  pytest plugin supplied /tmp/survjax-slice-b-run1 as FIXTURE_DIR through the
  existing spec loader. Assertions/markers/files were untouched. This validates
  the staged artifact, not the committed-fixture gate. Float32 reference
  comparisons are intentionally outside the spec 10.4 fixture contract.
- JSON SHA256: 38bfe13e202c71c7f8ad46b50cf314534d3ab0254dac8fea1dfa71addd22118b.
  CSV SHA256: 22bbf2b9b04fce88561ae193d2eff26066302612568f3a0172dfa183fbe093dc.

Limits: payloads are temporary, not part of this generator-only commit. The
protected tests/spec/fixtures/slice_b_likelihood.json installation and normal
fixture loading are pending integration work. Main's newly enabled fixture
acceptance therefore cannot pass against the committed tree yet. Do not infer
Slice B completion from the staged tests. No docs build for this generator step.
Remaining: B3 payload/loader/acceptance integration and any mathematical matrix
gaps, then B4 pruning/docs. No new spec ambiguity identified.

Pre-review scripts/check.sh --fast --through A12 exited 1. Formatting (249
files), lint and mypy (25 source files) passed. The gate advanced through both
implementation modes and source doctests to x64 acceptance; x64 implementation
reported 1126 passed/8 skipped. Acceptance: 3508 passed, 32 skipped, two existing
PH reference warnings, 49 setup errors, all because the committed Slice B JSON
is absent. The gate stops before x32 acceptance. No feature flag was removed
and no missing-fixture test was skipped to hide this integration dependency.
git diff --check passed. The full non-fast gate was not run.
