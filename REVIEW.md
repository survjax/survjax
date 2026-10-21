# A8.3 — Generated synthetic R fixture payloads

Generated data step on `a8-1-r-generator`, generators from `8e8a96a`.
No merge; stop after the A8.3 commit for review. The generated-data size
exception was explicitly approved in the final commit request.

## Read in this order

1. `tests/fixtures/exponential/README.md` and `tests/fixtures/bc_like/README.md`:
   regeneration commands, pinned environment, counts and SHA-256 manifests.
2. The JSON headers, case labels and grid metadata; inspect payload structure
   through the checks below rather than reading long generated JSON lines.
3. `PLAN.md`: data-only scope and deferred A6/A7/A8.4/A8.5 integration.

## Payloads and provenance (spec 10.3/10.4/11.1)

R 4.4.2, flexsurv 2.3.2, survival 3.7.0 (release 3.7-0), platform
`aarch64-apple-darwin20`. Both generators loaded and executed successfully
outside the sandbox; source code remains unchanged. No patient data were
loaded. All fixture values come from R; Python only inspects the exports and
compares them with original R hexadecimal numbers.

| Fixture | Seed | Rows | Exact/right | JSON numbers checked |
| --- | --- | --- | --- | --- |
| exponential | 8101 | 16 | 11/5 | 333 |
| bc_like | 8201 | 686 | 293/393 | 15770 |

The bc-like Good/Medium/Poor row counts are 229/229/228, events 52/91/150.
Its reference grids are 141 time rows by three labeled columns, using the
stored 0–7 year R grid. The realized censoring counts are not forced to match
the real cohort. These are newly computed synthetic-data totals:

| Fixture/case | flexsurv total | survreg total |
| --- | --- | --- |
| exponential/slow | -27.278095797078223 | -27.278095797078223 |
| exponential/generating | -18.371852578988488 | -18.371852578988488 |
| exponential/fast | -21.390380164991171 | -21.390380164991171 |
| bc_like/fitted | -799.28812804665461 | -799.2881280466554 |
| bc_like/unit_shape | -835.15401196535299 | -835.15401196535277 |
| bc_like/alternative | -798.05898311787723 | -798.05898311787735 |

Four generated files total 706 lines and 341,548 bytes (about 334 KiB).
The CSV files contain 17 and 687 lines; each JSON is one generated line.
Only these payload files use the planned size exception. Handwritten changes
are fixture provenance and process notes; no generator or library edits.

## Commands and verification

- `../scripts/spec-since.sh dev-a8`: exit 0, reports through `cf4fef9`, spec
  revision 32. Read current 10.4/11.1; revisions 31–32 concern likelihood
  handling, not these finite exact/right reference requirements. `.spec-seen`
  remains unchanged because this checkout has not integrated A6.
- `python3 /private/tmp/survjax-a83-prepare.py`: exit 0 outside the sandbox
  for the known R OpenMP issue. Python 3.12.13, standard library only. For
  each fixture, the script invokes the unchanged generator twice as:
  `Rscript --vanilla scripts/fixtures/generate_<name>.R <temporary-destination>`
  (actual script names are `generate_exponential.R` and `generate_bc_like.R`).
  Every R invocation passes its built-in version, numerical, alignment,
  case-label, grid-column and export checks as applicable.
- The two independent runs produce byte-identical CSV/JSON. Schema, row IDs,
  lengths, groups, counts, units, RNG, parameter vectors, finite interior
  values, grid shape/labels/S(0) and both checksum entries are verified.
  Per-row and total flexsurv/survreg values agree at existing float64 targets.
- One additional scratch run per fixture records original R `%a` values;
  its actual CSV/JSON must byte-match the original runs. Python reads all
  16,103 JSON numbers and all 702 CSV times back exactly to those R values.
  Instrumentation and hex sidecars are never imported into this checkout.
- Copied only the original run-1 payloads into the fixture directories, then
  byte-compared them with run 2. READMEs record SHA-256 of each imported file;
  JSON headers contain actual generator and CSV MD5 hashes. No timestamps or
  destination paths affect the deterministic payloads.
- Generation log: `/private/tmp/survjax-a83-generation.log`; full file hashes,
  sizes and totals: `/private/tmp/survjax-a83-summary.json`.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`:
  exit 0, `check: all gates passed through A5`. Ruff: 55 files already
  formatted, all checks passed; mypy: no issues in 13 source files.
  Own tests: x64 466 passed/7 skipped; x32 472 passed/1 skipped.
  Doctests: 4 passed per mode. Spec tests: x64 1256 passed/26 skipped;
  x32 1261 passed/21 skipped. Log: `/private/tmp/survjax-a83-gate.log`.
- `git diff --check`: exit 0, no output. Direct whitespace checks of all four
  new payloads are clean (`git diff --no-index --check /dev/null <payload>`
  returns the normal no-index difference status 1, without diagnostics).
  The final staged check in the commit command also covers these payloads.

## Limits and stop

This verifies genuine R fixture generation and export, not survjax/R
agreement. A8.4/A8.5 still add the loader and library comparisons after A6/A7
integration. No through-A8 gate or Slice A completion claim. No docs build,
source/test changes, new dependencies, spec changes or new DEVIATIONS entry.
Do not merge into main here; stop for review after the approved data commit.
