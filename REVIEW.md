# A8.1 — Small exponential generator

Implemented on `a8-1-r-generator`; stop after this commit for review.
The approved A8 plan was already an uncommitted 187-line addition at handoff;
this commit retains it. No public API, dependency, spec or protected-test edits.

## Read in this order

1. `scripts/fixtures/generate_exponential.R`: pins/simulation, package checks,
   then fixed-schema export. Assertions are the generator's accompanying tests.
2. `tests/fixtures/exponential/README.md`: commands, schema and provenance.
3. `PLAN.md`: A8.1 outcome and remaining independently approved steps.

## Invariants and requirement evidence

- Spec 10.3: exact R 4.4.2 / flexsurv 2.3.2 / survival 3.7.0 checks precede
  package loading. R's `3.7.0` denotes the required release 3.7-0.
- Spec 10.4: seed 8101, explicit RNG settings, 16 independent simulated rows,
  positive finite times, both event kinds, simulation order, no patient input.
- Spec 4.6: three fixed rates map to `mu=-log(rate)`, exponential sigma=1.
  Checks require unchanged package parameters and zero survreg iterations.
- Spec 10.4/12.1: flexsurv's documented `fixedpars=1` path supplies `logliki`
  and `loglik`; its summary supplies survival and cumulative hazard. survreg's
  documented density/CDF functions supply row contributions; zero-iteration
  full-data and singleton evaluations independently check their alignment/sums.
  Assertions check shapes, finite values, row identity and cross-path agreement
  at the ordinary float64 targets, before export. These finite interior cases
  avoid CDF-complement tail loss; these packages are not independent oracles.
- Export uses base R and 17 significant digits; schema records natural/log/AFT
  parameters, units, event coding, row IDs, package paths, realized counts,
  versions/platform and MD5 hashes. No volatile timestamp or destination path.

## Commands and results

- `../scripts/spec-since.sh dev-a8`: exit 0; reports design commits `0683e35`
  (revision 28 parameter diagnostics) and `9d7483c` (PDF helpers). Read the
  relevant current contract; inherited `.spec-seen` remains unchanged.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv sync --dev --locked`: exit 0;
  Python 3.12.9, 32 locked packages installed, no lockfile changes.
- `Rscript --vanilla scripts/fixtures/generate_exponential.R --check-only`:
  sandbox attempt exited 134, OpenMP SHM error 179. Approved unsandboxed retry
  exited 0: `Checks passed: 16 synthetic rows, 3 fixed rates, both R reference
  paths`. No fixture payloads were written.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`:
  exit 0, `check: all gates passed through A5`. Guard ok; Ruff: 54 files already
  formatted, all checks passed; mypy: no issues in 13 source files.
  Own tests: x64 466 passed/7 skipped; x32 472 passed/1 skipped.
  Doctests: 4 passed in each mode. Spec tests: x64 1256 passed/26 skipped;
  x32 1261 passed/21 skipped. Log: `/private/tmp/survjax-a8-1-check.log`.
  No tests were edited, skipped or xfailed by this step.
- `git diff --check`: exit 0, no output.

## Remaining limits

CSV/JSON writing and two-run byte comparisons are not executed here: A8.3
assigns output generation/review to the maintainer. Only the README is present
in the fixture directory. No R/survjax comparison, through-A8 gate or Slice A
completion claim; A8.2 and A8.4/A8.5 are untouched. Narrative docs wait for A8.d1.
No unresolved package-evaluation-path gap was found and no deviation was added.
