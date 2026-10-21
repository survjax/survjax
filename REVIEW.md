# A8.2 review corrections — 2026-10-20

Requested against `../review/slices/A8.2.md`, reviewed at `b87dcc1`.
Review this fix-up with `git diff b87dcc1..HEAD` on `a8-1-r-generator`.
A8.1/A8.2 remain unmerged; one fix-up commit, then stop for review.

## Read in this order

1. `scripts/fixtures/generate_bc_like.R`: expanded per-case loop, especially
   the new per-column prediction check and the three label-based scale lookups.
2. `tests/fixtures/bc_like/README.md`: stored grid times and grid-path limits.
3. `PLAN.md`: review corrections and A8.5/A8.d1 handoff.

## Findings addressed and invariants

- SF-1 (spec 11.1, PLAN grid orientation): each grid column is compared with
  a separate flexsurv prediction for its exported `group_names` label. This
  expectation is independent of shared `grid_groups`, so permuting both
  reference grids no longer preserves the checks. Existing cross-path, shape,
  S(0)=1 and serialization reconstruction checks remain.
- N-1 (spec 4.6): named scale vectors are indexed using character labels in
  simulation, row locations and conversion checks, rather than factor codes.
- N-2: expand the reviewed calls, case constants, conditions and result fields;
  separate simulation and reference phases with space. The standalone script
  is now 343 lines rather than 254. This fix-up's code/README diff is 162 lines;
  no helper framework or dependency was introduced to reduce the file length.
- N-3: fixture consumers must use stored `grid.time`, preserving R's actual
  seq doubles. README and A8.5 plan state this explicitly.
- N-4: README and A8.d1 handoff describe psurvreg grid evaluation at supplied
  AFT locations, cross-checked with flexsurv. Singleton-fit checks apply to
  observed contributions, not grid cells; no stronger independence claim.

## Commands and results

- `python3 /private/tmp/survjax-a82-review-fixes-check.py`: exit 0 outside the
  sandbox for the known R OpenMP issue, with the pinned R 4.4.2 / flexsurv
  2.3.2 / survival 3.7.0 environment. It runs the prior focused suite and:
  - B4 replaces `rep(group_names, each = length(eval_times))` with
    `rep(rev(group_names), each = length(eval_times))`. Both check-only and
    full export exit nonzero at `close(flex_grid[, column], expected_curve)`.
  - Before/after CSV bytes are identical. Parsed JSON values are identical
    after excluding the changed generator checksum; schema/parameters unchanged.
  - Check-only also passes with `options(warn = 2)` in a scratch copy.
  - Two full exports byte-identical; JSON valid, hashes and case labels checked.
  - All 15,770 JSON numbers and all 686 CSV times match original R `%a` values
    exactly on Python read-back. Closed-form checks retain max absolute error
    7.96e-13 and max relative error 5.58e-14, using unchanged tolerances.
  - Existing six mutants remain rejected: wrong AFT sigma, reversed survreg
    grid groups, transposed matrix export, flipped CSV events, six-digit JSON
    and duplicated case label. Unknown flags still fail before writing.
  - All temporary exports/mutants removed; no fixture payloads in the checkout.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`:
  exit 0, `check: all gates passed through A5`. Ruff: 55 files already
  formatted, all checks passed; mypy: no issues in 13 source files.
  Own tests: x64 466 passed/7 skipped; x32 472 passed/1 skipped.
  Doctests: 4 passed per mode. Spec tests: x64 1256 passed/26 skipped;
  x32 1261 passed/21 skipped. Log: `/private/tmp/survjax-a82-review-fixes-gate.log`.
- `git diff --check`: exit 0, no output.

## Limits and stop

No new dependencies, public API, Python/library, spec, independent-test or
narrative-doc changes. The R generator remains R-only; Python verifies exact
numeric read-back separately under the A8.1 decision. No R parser exactness,
survjax/R agreement or through-A8 completion claim. A8.3 committed generation
and acceptance remain pending. No new ambiguity or DEVIATIONS entry.
