# A8.4a.1 — Manifest-first R fixture loading (2026-10-25)

Approved as the next A8 step after 1c82bba. The approved plan permits splitting
additional schema coverage to keep code/tests below 300 changed lines.
This implements the common part; A8.4a.2 remains a separate review stop.

## Read in this order

1. tests/unit/test_reference_fixtures.py: valid committed sets, exact precision
   anchors, missing/malformed manifests/files, tampering before any parse,
   and correctly rehashed malformed/misaligned primary JSON.
2. tests/conftest.py: private manifest verifier, common loader, r_fixture
   callback. All implementation reference tests will use that one entry point.
3. PLAN.md: A8.4a.1/A8.4a.2 split and the deferred bc-specific checks.

## Invariants and interface

r_fixture(name, root=...) returns primary header/cases/grid plus data arrays;
root is a fixtures directory, optionally a temporary copy for negative tests.
Supported names are exponential and bc_like. All required payloads (two or
three respectively) must be listed exactly once and must pass SHA-256 before
any JSON or CSV parsing. Missing files fail; nothing skips or silently falls
back to unchecked parsing.

CSV time and per-row reference quantities are NumPy float64 independently of
JAX mode. Row IDs/events are integer host arrays. No JAX configuration changes,
R, network access, extra dependencies or library API changes are introduced.
Results are freshly loaded per call, without cached mutable parsed fixtures.

Common schema checks cover pinned seed/package/R provenance, generating path,
units/event coding, CSV provenance hash, row count/ID alignment, valid positive
finite times, event counts, case labels, scalar parameters/totals and finite
per-row arrays of the expected shape. Historical generator hashes are not
compared to current source bytes: committed references may come from older
reviewed generator revisions.

## Scope split and remaining checks

Code/test delta is 286 added lines (141 loader/shared-fixture lines plus
145 test lines). The complete bc group/grid/acceptance comparison would push
this beyond the review budget, so A8.4a.2 will add those checks and their
negative cases. This step hashes the additional acceptance JSON but does not
parse it; bc grid values remain in their original JSON form until A8.4a.2.
Do not use this partial milestone to claim completed bc schema validation.

No payloads, manifests, generators, tolerances, src/, tests/spec/ or docs/
are changed. Protected fixture handoff remains pending as recorded in A8.3b.
No distribution comparisons, documentation step or later-slice work starts.

## Checks and spec mapping

Spec 10.4/12.1: focused loader tests in separate JAX_ENABLE_X64=1 and 0
processes, using UV_CACHE_DIR=/private/tmp/survjax-uv-cache:
`uv run pytest -q tests/unit/test_reference_fixtures.py`:
20 passed in 0.20s per mode. Every known payload is independently modified;
a patched JSON parser fails if called before hash verification, so these tests
also guard the ordering of integrity checks.

Spec 3.3/10.4: exact expected anchors are original R %a values from the
A8.3/A8.3b numeric audits, for row 1's time and likelihood in each fixture.
Tests check exact equality as well as host dtype. In a temporary source/test
copy, force a float32 intermediate and cast back to float64; selecting the
precision tests yields 2 expected assertion failures / 18 deselected per mode
(0.08s x64 / 0.07s x32). No committed source/fixture is mutated.
Logs: /private/tmp/survjax-a84a1-mutation-x64.log and
/private/tmp/survjax-a84a1-mutation-x32.log.

Test JSON parsing outside the loader occurs only to construct deliberately
malformed temporary inputs and rehash them, never to obtain reference
expectations. All committed reference consumption uses r_fixture.

## Final gate

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`:
exit 0, check: all gates passed through A7.
Complete log: /private/tmp/survjax-a84a1-gate.log.

- Guard OK; Ruff: 74 files already formatted / All checks passed!
- mypy: no issues in 17 source files.
- Own float64: 797 passed, 7 skipped, 158.10s.
- Own float32: 803 passed, 1 skipped, 157.89s.
- Source doctests: 4 passed per mode, 0.96s / 0.90s.
- Independent float64: 2123 passed, 28 skipped, 223 deselected, 119.00s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected, 117.34s.

Only result records and plan status are updated after the gate. Final Ruff
formatting/lint and git diff --check pass on the completed records.

## Limits and stop

No through-A8 claim; no R execution or docs build. A8.4a.2 must finish the bc
schema before A8.4b/A8.5 comparisons. No new spec ambiguity or deviation.
One implementation commit on a8-1-r-generator, then stop for review.
