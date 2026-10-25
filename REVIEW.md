# A8.3a review nits — 2026-10-25

Authorized by the maintainer's request to address ../review/slices/A8.3a.md.
Parent: 691edc3; branch: a8-1-r-generator. One fix-up commit, then stop.

## Read in this order

1. scripts/fixtures/bc_spec_export.R: final assembled-output checks.
2. tests/fixtures/bc_like/README.md: preservation versus independent
   grid-orientation evidence; shared R-route limitation remains explicit.
3. PLAN.md: N-3's concrete A8.3b passing-recipe requirement.

## Findings addressed

- N-1: compare assembled time/event/group arrays, parameter vectors, grid
  times/curves and each labelled source's loglik/survival directly with the
  original evaluated R objects before returning the fixture for serialization.
- N-2: remove identity immediately after assigning curves; check the final
  fixture's curves instead. Comments and README call this preservation,
  not independent orientation evidence. The independent per-group prediction
  checks remain upstream, unchanged.
- N-3: keep the documented temporary provenance drift; A8.3b must replace
  payload provenance and extend the committed-file comparisons to the new
  JSON, run the full recipe successfully and remove the interim notice.
  No payload/manifest update is smuggled into this fix-up.
- N-4: numerical independence is still not claimed. The new exact output
  checks also happen to detect the source-label swap on these R objects;
  tolerance-based agreement alone would still not distinguish those routes.

## Invariants and spec-to-test mapping

Spec 10.4/11.1/12.1: export the same synthetic observations, case vectors,
per-row R references and time-first grid. No numerical reevaluation,
patient data, Python-generated references or runtime Python dependency.
The main generator, all committed payloads/manifests, independent tests,
public API, library source, dependencies and tolerances remain unchanged.
Only the small exporter, fixture README and process records change.
A8.3b and later steps are not started.

## Checks actually run

- Base-R syntax parse: exit 0.
- `python3 /private/tmp/survjax-a83a-nits-validate.py`: exit 0 in the pinned
  R 4.4.2 / flexsurv 2.3.2 / survival 3.7.0 environment. Run outside sandbox
  for flexsurv's OpenMP shared memory. Log:
  /private/tmp/survjax-a83a-nits-validation.log.
- Check-only succeeds without changing any tests/fixtures bytes. Unknown
  --check-onyl fails before writing. Two full scratch exports are identical
  for all three files. CSV equals committed CSV; existing expected.json
  differs only in generator/exporter checksum provenance.
- Exact schema/row/case/source/grid comparison against the original R output
  passes. Original R %a numeric equality: 11,027 acceptance numbers,
  15,770 original-JSON numbers and 686 CSV times, all exact.
- E2 wrong loglik field, E4 reversed groups, reversed times, reversed events,
  wrong survival field and E6 swapped sources each fail check-only and full
  export before creating their output directory. This directly covers N-1.
- Existing case-swap and grid-column-swap mutants fail in both modes;
  shortened-decimal serialization fails the R 17-digit check.
- Additional acceptance JSON is unchanged: 214,860 bytes; SHA-256
  4edbdee917ae335da11911b39ec60442e21998687d27c62494f54e7eb1d0e450.
  Outputs remain scratch validation evidence, not committed payload delivery.

## Final gate

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`:
exit 0, check: all gates passed through A7.
Log: /private/tmp/survjax-a83a-nits-gate.log.

- Guard OK; Ruff: 73 files formatted and All checks passed!
- mypy: no issues in 17 source files.
- Own float64: 777 passed, 7 skipped, 137.88s.
- Own float32: 783 passed, 1 skipped, 132.99s.
- Source doctests: 4 passed in each mode, 0.75s each.
- Independent float64: 2123 passed, 28 skipped, 223 deselected, 88.23s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected, 84.52s.

Only result records are updated after the gate; final Ruff formatting/lint
and git diff --check pass on the finished records. No through-A8 claim.

## Limits and next review stop

No through-A8 claim, docs build, publication, spec edit or new deviation.
The committed independent fixture is still absent; its durable handoff and
N-3 recipe completion belong to A8.3b. The source routes remain numerically
related. Stop after the review fix-up commit.
