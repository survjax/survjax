# A8.1 review follow-up — R structure and spacing

Read `scripts/fixtures/generate_exponential.R` first, then this validation.
The maintainer requested this fix-up on the existing A8.1 branch.

Calls and checks have more space; section breaks separate setup, simulation,
reference checks and export. `evaluate_survreg(rows, mu)` is defined once.
The JSON schema is a named R list, with a small base-R encoder for its types.
Parameters, RNG order, row alignment, assertions and schema values are preserved.

Validation (no fixture files written; R checks ran outside the OpenMP sandbox):
- `Rscript --vanilla scripts/fixtures/generate_exponential.R --check-only`:
  exit 0; 16 rows, 3 fixed rates, both reference paths passed.
- `/private/tmp/survjax-a8-1-structure-check.R` piped to the companion Python
  JSON comparison: CSV bytes and parsed JSON values match old/new versions.
  Writes were intercepted, hashes normalized; 0.35 now uses 17-digit formatting.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`:
  exit 0; all gates passed, pass/skip counts match the original results below.
  Log: `/private/tmp/survjax-a8-1-structure-gate.log`.
- `git diff --check`: exit 0, no output. No test files or dependencies changed.

Actual filesystem export/two-run regeneration remain A8.3. Original evidence:

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
