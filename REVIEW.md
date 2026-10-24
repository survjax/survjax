# A8.3a — R export for the independent acceptance schema (2026-10-25)

Approved by the maintainer's instruction to implement the next planned step.
Branch: a8-1-r-generator; parent: d1f1f1a. One code step, then stop.

## Read in this order

1. scripts/fixtures/bc_spec_export.R: explicit case/field mappings from the
   evaluated R objects, with independent fixed-vector label and grid checks.
2. scripts/fixtures/generate_bc_like.R: source the exporter; build/serialize
   its payload before the check-only exit; write its JSON alongside the old
   files; record exporter checksum in the original provenance header.
3. tests/fixtures/bc_like/README.md: the additional output, overwrite warning,
   interim provenance drift, and durable acceptance-fixture handoff.
4. PLAN.md: completed A8.3a scope and the separate A8.3b payload step.

## Invariants

- No numerical values are fitted or computed by this exporter. It maps the
  same R evaluations used by the reviewed A8.3 reference files. Fixed cases
  map unit_shape/alternative/fitted to fixed_a/fixed_b/r_optimum.
- Seed 8201, 686 synthetic rows and block order Good/Medium/Poor remain.
  No patient data, real-cohort totals or Python-generated references.
- Each source retains its own observed-time likelihood and survival. Grid
  curves are flexsurv values, time first, columns Good/Medium/Poor.
- Check-only validates and serializes the additional schema without writing.
  Unknown flags fail before package loading or output. Full output stays
  under the supplied directory; no write under tests/spec.
- Generator runtime remains R-only. Exact text/17-digit formatting is checked
  in R; separate Python validation compares parsed doubles to original R %a.
- Committed data, JSON payloads and manifests remain unchanged. On future
  regeneration, old expected.json changes only generator/exporter checksum
  provenance. The README marks that interim committed-file comparison drift.

## Checks and spec mapping

Spec 10.4/11.1/12.1: the generator's R assertions exercise finite dimensions,
case/parameter identities, row/group order, grid preservation and precision.
The additional exporter is 85 lines; the generator grows from 343 to 351
lines, keeping the new responsibility in a separate file. No public API,
dependency, tolerance, library source or independent-test changes.

Actual scratch validation command:
`python3 /private/tmp/survjax-a83a-validate.py`
(exit 0; run outside sandbox because flexsurv's OpenMP cannot open shared
memory in the sandbox). Log: /private/tmp/survjax-a83a-validation.log;
summary: /private/tmp/survjax-a83a-summary.json. Pinned R 4.4.2,
flexsurv 2.3.2, survival 3.7.0 (spec spelling 3.7-0).

- Check-only exits 0, and a before/after snapshot of all tests/fixtures bytes
  is unchanged. Misspelled --check-onyl exits 1 with Unknown option and
  also leaves every committed fixture unchanged.
- Two full exports into temporary directories: all three files byte-identical.
  CSV equals committed CSV exactly. Original JSON matches every committed
  value/field apart from generator/exporter checksum provenance.
- Independent Python validation checks the exact schema, metadata, embedded
  observations, all fixed parameters, every source/row quantity and every
  flexsurv grid cell against the original R output.
- An instrumented R wrapper emits %a sidecars without changing the real
  output bytes. Python numeric round-trip equality: 11,027 additional-schema
  numbers, 15,770 original-JSON numbers and 686 CSV times; all pass exactly.
- Mutation: swap fixed_a/fixed_b mappings. Check-only and export both fail
  the independent fixed-vector assertion before creating an output directory.
- Mutation: reverse exported grid columns. Shape and S(0) still hold, but
  check-only and export both fail exact grid-preservation assertion.
- Mutation: serialize with six digits. R fails the independent formatC
  17-digit check before creating an output directory.
- Scratch outputs are uncommitted; they are validation evidence only. The
  new acceptance JSON is 214,860 bytes, SHA-256
  4edbdee917ae335da11911b39ec60442e21998687d27c62494f54e7eb1d0e450.

## Final gate

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`:
exit 0, check: all gates passed through A7.
Complete log: /private/tmp/survjax-a83a-gate.log.

- Guard: OK; Ruff: 73 files already formatted, All checks passed!
- mypy: Success: no issues found in 17 source files.
- Own float64: 777 passed, 7 skipped in 148.16s.
- Own float32: 783 passed, 1 skipped in 151.72s.
- Source doctests: 4 passed per mode, 0.91s / 0.96s.
- Independent float64: 2123 passed, 28 skipped, 223 deselected in 107.74s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected in 99.24s.

The review was prepared before the full gate; only this result section and
plan status are updated afterward. Final Ruff/diff checks accompany the
finished review. No through-A8 claim.

## Remaining work and limits

A8.3b separately regenerates/commits payloads and updates manifests. The
maintainer then supplies the verified JSON to the authoritative tester repo
and syncs it; the real acceptance fixture is still absent. No A8 acceptance
comparison was run or claimed in this step. A8.4/A8.5 loaders and Python/R
comparisons remain unimplemented; narrative docs stay in A8.d1.
No docs build, publication or spec edit. No new deviation or unresolved
numerical/API decision. Stop after this commit for review.
