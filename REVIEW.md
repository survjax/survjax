# A8.1 review corrections — 2026-10-20

Requested against `../review/slices/A8.1.md` (reviewed commit `697ab6f`).
One fix-up on `a8-1-r-generator`; stop after committing for review.

## Read in this order

1. `scripts/fixtures/generate_exponential.R`: CLI guard, shared generating
   rate, label/rate assertions, then export text checks.
2. `tests/fixtures/exponential/README.md`: check-only/full-run distinction
   and the pinned R parser limitation.
3. `PLAN.md`: SF-1/N-1/N-3 corrections and updated future likelihood API.

## Findings addressed and invariants

- SF-1 (spec 10.4/12.1): CSV is read back as character columns and compared
  exactly with the original row IDs, independently formatted times and events.
  Each encoded JSON number must match 17-digit `formatC` text. Written JSON
  must read back identically to the validated payload. Case names/order and
  each label's fixed rate are checked before the check-only exit.
- Exact numeric R read-back initially failed: three observed times changed by
  one ULP in `read.csv`, despite correct 17-digit output. The pinned R build
  reports no long-double support; `as.numeric` also changes those decimals.
  The maintainer selected R-only exact text checks plus separate Python numeric
  validation. No reference tolerance was loosened and no dependency was added.
- N-1: an unrecognized `--` argument stops before loading R packages or writing.
- N-3: `generating_rate` controls simulation, the generating case and metadata.
- N-2/N-4: retain defensive time matching and the honest density/CDF composition
  description, with singleton survreg totals checking each contribution (4.6).
  The planned A8.d1 documentation retains that evaluation-path explanation.
- S-1: later A8 planning now names `survjax.log_likelihood` and
  `dist.log_likelihood`, with `data.issues`/`dist.parameter_issues()` diagnostics.
  The current spec is revision 30; revision 29 changed the likelihood API,
  revision 30 changed parameter-constraint comparisons, neither changes this
  R generator. `.spec-seen` remains at the inherited implementation baseline:
  no claim that this checkout implements the unmerged A6 requirements.

## Validation

- `../scripts/spec-since.sh dev-a8`: exit 0; reports design changes through
  `fd150db` (revision 30), including `be8010c` (revision 29). Read current
  sections 6.1–6.3/7.1/10.4; no spec or marker edits.
- `python3 /private/tmp/survjax-a81-review-check.py`: exit 0, run outside the
  sandbox for the known R OpenMP issue, using R 4.4.2 / flexsurv 2.3.2 /
  survival 3.7.0. It invokes the actual generator in check-only and export
  modes, using temporary directories and scratch copies for mutations:
  - Check-only and two full exports exit 0. Both payload files byte-identical
    across runs; JSON parses, case order and both MD5 checksums verified.
  - A scratch copy records source numbers as R `%a` hex strings before
    decimal parsing. Python matches all 333 JSON numbers and all 16 CSV times
    exactly to these original binary values; no approximate expected numbers.
  - `--chek-only` exits nonzero with `Unknown option`; no directory created.
  - Changing the shared generating rate to 0.42 keeps simulation, exported
    header and generating case consistent; full export still exits 0.
  - All scratch exports/mutants removed after success.

| Reviewer mutant | Reproduced edit | Result |
| --- | --- | --- |
| M6 | CSV `data$event` replaced by `1L - data$event` | nonzero exit at exact CSV read-back |
| M7 | JSON `sprintf("%.17g", value)` replaced by `%.6g` | nonzero exit at `identical(encoded, expected)` |
| M8 | `references$fast <- references$slow` before label validation | nonzero exit at label/rate identity |

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`:
  exit 0, `check: all gates passed through A5`. Ruff: 54 files formatted,
  all checks passed; mypy: no issues in 13 source files. Own tests: x64
  466 passed/7 skipped, x32 472 passed/1 skipped. Doctests: 4 passed per mode.
  Spec tests: x64 1256 passed/26 skipped, x32 1261 passed/21 skipped.
  Log: `/private/tmp/survjax-a81-review-gate.log`. Python/protected tests unchanged.
- `git diff --check`: exit 0, no output.

## Limits

The generator itself requires only the pinned R packages; Python was used for
validation, not added to its runtime. Actual R numeric read-back equality is
not claimed. Temporary exports test the new checks; no fixture payloads are
committed or left in this checkout. A8.3 generation/acceptance and A8.4/A8.5
survjax comparisons remain separate. No docs build or through-A8 gate run.
No new spec ambiguity or DEVIATIONS entry; no A8.2 work started.
