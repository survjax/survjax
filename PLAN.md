# Slice A, step A2 — constructors, boundaries, and DataIssues

Status: **A2a implemented on `a2-constructors`; stop for maintainer review
before A2b**.
Branch: `a2-constructors`, at the A2 spec-test merge (`5729f6b`).
Specification revision 5 now includes the first-endpoint and infinite-interval
clarifications (`022d5eb`, `3f83d91`, `f7d2fff`). The corresponding independent
tests were synced in `99f7629`.

## A2a review fix-up: direct `row_issues` coverage

The A2a tests exercise `row_issues` through eager constructor errors, but do
not assert its per-row bitfield directly. Error-message checks can pass while
one mask is assigned to the wrong bit or while independent masks are lost.
Add a small follow-up commit on `a2-constructors` before A2b. The maintainer
approved this fix-up in the current conversation; implementation and both-mode
checks are complete.

- `tests/unit/test_data_validation.py` (new): call the private `row_issues`
  helper with hand-chosen arrays. Assert the ten rule masks from spec 6.2,
  their observation shape and `int32` dtype, and simultaneous independent
  faults. Cover interval order only after valid endpoints, invalid truncation,
  exact `L < t <= U`, right censoring at versus before entry, and empty versus
  nonempty censored intersections. Expected rows come from the stated set and
  window inequalities, not from constructor output.
- `src/chronox_jax/_data_validation.py`: change only if those tests expose a
  real defect. No planned public API change or new dependency.
- `REVIEW.md`: replace with the fix-up reading order, test-to-rule map, exact
  results, and any remaining uncertainty. `PLAN.md` records this proposal.

Run Ruff format/lint, mypy, `scripts/run_tests.sh`, and
`scripts/check.sh --through A2`, in both configured dtype processes. Keep
`tests/spec/` and `spec/` read-only. Stop after one fix-up commit and review.

## Scope and public API

A1 supplied the two constructor signatures, valid-input normalization, dtype
policy, and six-leaf PyTree. A2 will make construction enforce the full data
contract. It will preserve the five stored public fields and all existing
named constructor parameters. The flag constructor will accept `**kwargs`
solely to recognize `outside_interval_censored` and reject it explicitly;
other unknown keywords remain errors.

- `SurvivalData(...)`: validate raw `event` and censoring flags before casting;
  reject `event` combined with any supplied flag and conflicting true flags;
  require `time2` only on active interval rows; reject invalid times, intervals,
  truncation windows, and empty observation/window intersections. The ordinary
  constructor raises with counts and first offending indices. Under traced
  observation values it raises at trace time with a pointer to `with_issues`.
  Any `outside_interval_censored` argument raises `NotImplementedError`, even
  when all its flags are false (section 3.1).
- `SurvivalData.from_bounds(...)`: validate `exact` before casting; check the
  supplied upper endpoint on exact rows before normalization; report NaN upper
  on non-exact rows as `missing_time2`; attribute invalid first endpoints to
  `invalid_time`; then apply the same endpoint, truncation, and intersection
  rules as the flag form.
- `SurvivalData.with_issues(...)`: add the section 6.2 pure flag-constructor
  counterpart, returning `(SurvivalData, DataIssues)` under eager, `jit`, and
  `vmap` calls. Invalid rows set issue bits without value-dependent exceptions.
  Inputs that cannot form an object (for example, wrong shapes or string
  indicators) still raise.
- `chronox_jax.diagnostics.DataIssues`: add a frozen, typed PyTree result with
  the ten named invalid-rule masks, `uninformative`, `invalid_rows`, scalar
  `ok`, and host-only `raise_if_invalid()`. Keep it discoverable in its module
  as section 10.2 specifies; do not add a top-level export.

The `int32` issue bitfield remains the sixth `SurvivalData` leaf and records
raw indicator conflicts and missing active `time2` even after normalization.
No indexing, slicing, or value-derived static metadata yet; those are A3.
No distributions, likelihoods, or later Slice A features are included.

## Files

- `src/chronox_jax/data.py`: route both eager constructors and `with_issues`
  through shared normalization and validation; preserve A1 dtype and PyTree
  behavior.
- `src/chronox_jax/_data_validation.py` (new, to keep modules near 300 lines):
  existing private dtype and shape rules, indicator checks, pure array rules,
  bit assembly, and eager error reporting, with no new public API.
- `src/chronox_jax/diagnostics.py` (new): `DataIssues` decoding, reduction,
  PyTree registration, and host error reporting.
- `tests/unit/test_data_construction.py` (new): indicator codes, flag conflicts,
  active and inactive `time2`, bounds exactness, missing values, and parity of
  the two constructors, including mixed rows and broadcasting. Cover the
  explicit unsupported argument.
- `tests/unit/test_data_boundaries.py` (new): exact, censoring, interval, and
  truncation boundaries; partial/full/empty intersections; right censoring at
  entry under both constructors (including positive entry and finite `U`);
  right censoring strictly before entry; interval `time2=+inf` before/at
  entry; valid `uninformative` rows.
- `tests/unit/test_data_dtype.py`: keep the A1 traced weak-type promotion
  check at the private dtype helper, because A2 makes traced ordinary
  construction reject observation values at trace time.
- `tests/unit/test_data_issues.py` (new): each bit and decoded mask, row shape,
  missing versus invalid upper endpoints in the flag path and the shared pure
  validator used by `from_bounds` (which has no public `with_issues` form),
  `invalid_time` for NaN or out-of-domain first endpoints on exact, one-sided,
  interval, and bounds rows; interval order only after both endpoints pass
  presence/domain checks; independent overlapping issue masks, invalid union
  and scalar `ok`, PyTree preservation, `jit`/`vmap` behavior, eager error
  summaries, and traced ordinary-constructor rejection.
- `DEVIATIONS.md`: read the maintainer-resolved D-003/D-004/D-005 entries; make no
  status changes. Add an entry only if implementation exposes a new ambiguity.
- `REVIEW.md`: overwrite only after implementation with reading order,
  invariants, requirement-to-test map, exact check output, and uncertainties.

The existing `tests/spec/`, `spec/`, and working-agreement files are read-only.
No dependency or configuration changes are planned.

## Invariants and implementation order

1. Inspect indicator dtype and values before boolean conversion, preserving
   evidence in the issue bitfield. A selected kind alone controls which time
   endpoints are active; inactive `time2` is ignored per D-001.
2. Compute issue masks from raw inputs and normalized fields using JAX array
   operations. Structural failures are distinguished from invalid rows: the
   former raise in both paths; the latter remain data plus flags in
   `with_issues`.
3. Check both endpoint domains before interval order. A NaN or out-of-domain
   active flag `time` or bounds `lower` sets `invalid_time`, including interval
   lower endpoints (D-005). Absent/NaN active uppers set `missing_time2`;
   negative or `-inf` active uppers and mismatched exact bounds uppers set
   `invalid_time2`; otherwise equal/reversed censored intervals set
   `invalid_interval` (D-004). A malformed endpoint does not also set
   `invalid_interval`. An exact bounds row may supply NaN upper, which
   normalizes to its lower.
4. Apply `0 <= L < U`, exact membership `L < t <= U`, and nonempty censored
   intersection with `(L, U]`. Any normalized right-censored row with `c=L`
   is valid and `uninformative`, including `L>0` and finite `U`; `c<L` is
   invalid and sets `censored_before_entry` (D-003). This includes interval
   flags with `time2=+inf`; they normalize to right censoring and use the same
   before/at-entry rule. Other full-window censored coverage is valid and
   `uninformative`, outside `invalid_rows`.
5. The eager path reduces masks on the host, reports counts and first indices,
   and raises. The pure path never performs Python truth tests on tracers.
   PyTree reconstruction bypasses both paths.

## Proposed review steps

The implementation and tests described above exceed the roughly 300-line
review target. Split at the constructor-validation versus diagnostic-interface
boundary, with one commit and a `REVIEW.md` update after each step. Stop after
each step for maintainer review.

1. **A2a — validated eager constructors.** Add shared private array rules and
   issue-bit assembly; use them in both ordinary constructors. Cover raw
   indicators, active endpoints, normalization, truncation, and observation
   intersections in `test_data_construction.py` and `test_data_boundaries.py`.
   The constructors report counts and first offending indices; traced calls
   point to the planned `with_issues` method. Keep the six-leaf PyTree. Run
   Ruff, mypy, `scripts/run_tests.sh`, and `scripts/check.sh --through A2`.
   The synced A2 acceptance tests cover eager construction. Record that the
   A2 diagnostic API remains incomplete at this intermediate commit.
2. **A2b — pure diagnostics.** Add `DataIssues` in `diagnostics.py` and
   `SurvivalData.with_issues(...)`, using the same normalization and bitfield
   rules. Cover each issue mask, unions, reductions, PyTree behavior,
   `jit`/`vmap`, host error reporting, and constructor parity in
   `test_data_issues.py`. Run all checks listed below, including A2 and the
   independent A3 diagnostic file, once per dtype mode. Record any actual
   spec-test disagreement in `DEVIATIONS.md`; do not edit those tests.

The split and this updated A2a implementation were approved by the maintainer.
Stop after the A2a commit and review handoff.

## Specification and checks

Coverage: section 2 structural observation cases; 3.1 constructors; 3.2
normalization and issue-bit storage; 3.3 boundaries; 6.2 validation and
`DataIssues` and revision 5 endpoint issue attribution; 6.3 diagnostic container shape;
10.2 module/public API policy; 12.1 construction verification; and 12.2 Slice A
scope. The independent suite labels `DataIssues` tests A3, while the working
agreement assigns this API to A2. This plan follows the working agreement and
runs both A2 and relevant A3 diagnostic tests once the API exists.

After each implementation step, run `uv run ruff format --check .`,
`uv run ruff check .`, `uv run mypy src`, `scripts/run_tests.sh`, and
`scripts/check.sh --through A2`. For A2b, also run
`JAX_ENABLE_X64=1 SPEC_THROUGH=A3 uv run pytest tests/spec/test_a3_data_issues.py -q`
and the same command with `JAX_ENABLE_X64=0`. Report every command's exact
result. The synced A2 and A3 diagnostic tests now cover revision 5; any
failure requires investigation, not an assumed stale-test exception. Do not
edit, skip, or xfail independent tests.

## Settled A2 decisions

- **D-003:** Right censoring at `c=L` is valid and uninformative for any
  entry time; `c<L` is invalid.
- **D-004:** Missing active upper endpoints set `missing_time2`; negative or
  `-inf` active uppers and mismatched exact bounds uppers set `invalid_time2`;
  otherwise valid but equal/reversed non-exact bounds set `invalid_interval`.
  Presence/domain checks precede order checks; independent problems may set
  multiple masks. See specification revision 5, section 6.2.
- **D-005:** NaN and other invalid active flag `time` or bounds `lower` values
  set `invalid_time`, including interval lower endpoints. Invalid first or
  second endpoints do not also set `invalid_interval`. See specification
  revision 5, section 6.2. The separate section 3.1 clarification makes
  interval `time2=+inf` follow right-censoring entry rules.
