# A8.3b — Verified payloads and acceptance-fixture handoff (2026-10-25)

Authorized by "all good; next bit of A8" after the A8.3a review fix-up.
Parent/source commit: ef4a038d9b272c0fbd576cc891bc159cfcfe24d8.
One generated-data step on a8-1-r-generator, then stop for review/handoff.

## Read in this order

1. tests/fixtures/bc_like/SHA256SUMS: three verified payloads.
2. tests/fixtures/bc_like/README.md: passing regeneration recipe, provenance,
   acceptance-schema mapping and pending protected-fixture handoff.
3. bc_synthetic_weibull.json: the added tester-schema candidate. Its values
   map exactly to the original data/R references; review via the checks below.
4. expected.json: checksum provenance only changes; all prior values remain.
5. PLAN.md: A8.3b result and the four remaining implementation/docs steps.

## Invariants and spec mapping

Spec 10.4/11.1: synthetic data only, seed 8201, 686 rows, group blocks
229/229/228, 293 exact and 393 right. No patient data or real-cohort totals.
Data.csv is unchanged. All existing expected.json values are unchanged except
checksums.generator and the added checksums.exporter. No R generator, library,
public API, dependency, tolerance, spec or independent-test changes.

The additional JSON embeds the observations and three fixed vectors, with
separate flexsurv/survreg likelihood and survival arrays and flexsurv grids.
Its values come directly from the approved R evaluations. Python checks and
copies those values; it does not generate or approximate expected numbers.
The two R routes remain related numerical evidence, as documented.

## Checks actually run

- `python3 /private/tmp/survjax-a83b-validate.py`: exit 0, run outside the
  sandbox for flexsurv's OpenMP shared memory. Actual pinned environment:
  R 4.4.2, flexsurv 2.3.2, survival 3.7.0 (spec spelling 3.7-0),
  aarch64-apple-darwin20. Log: /private/tmp/survjax-a83b-validation.log;
  summary: /private/tmp/survjax-a83b-summary.json.
- Two fresh R generations: all three files byte-identical. CSV matches the
  committed A8.3 CSV. Structured comparison against the parent JSON proves
  every other field/value unchanged. Generator/exporter checksum fields match
  the exact source bytes; CSV checksum and algorithm remain unchanged.
- Exact schema/metadata/row/parameter/source/grid validation passes. An R
  wrapper emits original %a doubles without changing real export bytes.
  Exact Python numeric read-back: 11,027 acceptance-JSON numbers,
  15,770 original-JSON numbers and 686 CSV times; all pass.
- Installed the verified candidate only into an unchanged temporary copy of
  tests/spec, with every source file compared byte-for-byte to the real suite.
  Ran `uv run pytest -q` on that copy's test_a8_bc_fixtures.py with
  SPEC_THROUGH=A8 and UV_CACHE_DIR=/private/tmp/survjax-uv-cache:
  JAX_ENABLE_X64=1: 44 passed in 1.84s;
  JAX_ENABLE_X64=0: 44 passed in 1.85s.
  Logs: /private/tmp/survjax-a83b-spec-x64.log and
  /private/tmp/survjax-a83b-spec-x32.log. This covers spec 11.1/12.1 comparisons
  and transformations with the unchanged acceptance assertions.
- Extracted the exact final README shell block; sh -n exits 0. Ran the full
  block with sh (set -eu): exit 0. Check-only and both R generations succeed,
  all six cmp commands succeed, all three SHA256SUMS entries report OK.
  Log: /private/tmp/survjax-a83b-readme-recipe.log. This closes review N-3.

- Final Ruff format check: exit 0, 73 files already formatted.
- Final Ruff lint: exit 0, All checks passed!
- git diff --check: exit 0, no output. Preservation checks confirm no change
  to generators, library, independent tests, tolerances or data.csv.
No full through-A7/A8 gate, mypy, own suites or docs build rerun for this
payload/manifest/README-only step. The last through-A7 gate at ef4a038 passed;
the targeted acceptance tests above are this step's numerical evidence.

## Handoff ready for the maintainer

Source: tests/fixtures/bc_like/bc_synthetic_weibull.json (214,860 bytes).
SHA-256: 4edbdee917ae335da11911b39ec60442e21998687d27c62494f54e7eb1d0e450.
This agrees with the reviewed A8.3a candidate exactly.

Supply that exact file to tests/spec/fixtures/bc_synthetic_weibull.json in the
authoritative tester repo, commit there, then use the normal spectests-* sync
and maintainer merge. Do not install it only into this worktree's tests/spec:
the next wholesale sync would remove it. No protected-path write was performed.
After sync, compare its bytes/hash to this candidate and run the actual A8
module in both dtypes. A full through-A8 claim waits for the integrated gate.

## Remaining work

A8.4a shared loader and integrity/corruption tests; A8.4b exponential reference
comparisons; A8.5 Weibull comparisons and full through-A8 gate after handoff;
A8.d1 reference-evidence docs after reviewed code merges. Stop for this step's
review and the maintainer handoff. No new deviation or numerical/API decision.
