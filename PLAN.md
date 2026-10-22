# A6.4 replanning — Exact/right likelihood (2026-10-21)

## Current step — A6.4c public likelihood integration (2026-10-22)

Status: **implemented, ready for review**, authorized by "do it".
Planning starts on clean `A6` at
`6a35b7a`, after the A6.4b2 review fixes and N3 rename. Spec revision 32 is
unchanged since `.spec-seen` (`cf4fef9`); the public integration outline below
still applies. This entry refines its files, test budget and validation order.
One implementation commit with REVIEW.md on the existing A6 branch, then stop.

### Files and public API

- `src/survjax/likelihood.py`: add typed
  `log_likelihood(dist, data=None, *, time=None, event=None) -> jax.Array`.
  One private resolver accepts SurvivalData unchanged, or constructs it from
  both raw keywords using the existing pure checked constructor. Reject mixed
  or missing forms and positional raw arrays with TypeError; point positional
  array users to `time=`/`event=`. Delegate evaluation to `_log_likelihood`.
  Keep all existing numerical preparation and row checks intact.
- `src/survjax/distributions/base.py`: add the same typed base method without
  the dist argument, forwarding directly to the public function. Use a local
  import for the function and TYPE_CHECKING for the data annotation if needed
  to avoid a cycle. No registration/declaration or direct-method changes.
- `src/survjax/__init__.py` and `tests/unit/test_import.py`: add exactly the
  `log_likelihood` export. No result wrapper, reduction or new diagnostic API.
- `docs/api.md`: add one generated function entry under the standing minimal
  API-reference authorization. Public function/method docstrings describe
  inputs, trailing-axis broadcasting (no time-grid axis), unsummed output,
  static TypeErrors and invalid/unsupported-row NaNs. No narrative docs here.

### Own tests and specification coverage

Create `tests/unit/test_likelihood_entry_points.py`, aiming for 130–180 lines;
total source/test/docstring changes should stay around 250–300 lines. Replan
before exceeding the budget; do not copy the existing core/masking matrices.

- Check raw/prepared and function/method values against hand-derived
  Exponential exact/right contributions. Confirm a custom family inherits the
  method using existing test coverage in the independent suite (4.2/6.1/7.1).
- Small named cases check mixed/missing forms, positional arrays, extra
  constructor keywords and incompatible raw shapes, through both entry
  points. Include a traced misuse case; signature checks remain static (6.1).
- Invalid raw indicator values must retain their constructor issue evidence
  and yield NaN only on the affected rows, eager/jit/vmap, through both forms.
  Reuse the core suites for normalized-field reconstruction cases (6.2/7.1).
- Compare values, Jacobians and Hessians in both entry points/forms with
  analytical Exponential anchors, using named transform cases and existing
  tolerances. Exercise jit and vmap without duplicating the full isolation
  matrix; independent method/isolation suites provide broader coverage
  (3.3/5/7.2–7.3/12.1).

### Verification, limits and delivery

Use UV_CACHE_DIR=/private/tmp/survjax-uv-cache and separate x64/x32 processes.
First run the new own tests and the independent entry-point, method,
gradient-isolation and unsupported-mode modules with SPEC_THROUGH=A6.
Then run `scripts/check.sh --through A6`: it covers every other synced A6
module plus Ruff, mypy, the full own suite and source doctests in both modes.
Run `scripts/build_docs.sh`, inspect the generated function entry in both
dtype builds, and finish with `git diff --check`. Record exact results and
any optional skips in REVIEW.md; independent tests and gate scripts stay
read-only. Planning used read-only inspection; implementation evidence is
recorded below and in REVIEW.md.

D-026 remains an open recipe-wording question; this step preserves the
specified stored-issue and current-field behavior without editing the spec.
Docs to-do: likelihood usage, diagnostic triage, method reference and examples
remain separate docs work. A6.5 composition checks and A7–A9/remaining Slice A
completion evidence remain later steps. Approval covers this integration
only; stop after its implementation commit and REVIEW.md.

Evidence: 52 own API cases and 544 focused independent cases pass in each
dtype. The complete through-A6 gate passes (own 700/706 passed; independent
2023/2027 passed), as do both executable docs builds. REVIEW.md records exact
counts, skips and the sandbox-only docs retry. The source/test/docstring/API
entry diff is 234 lines. No A6.5 or Slice A completion is claimed.

## A6.4b2 nit N3 — Name the current-field check (2026-10-22)

Status: **implemented, ready for review**, explicitly requested by the maintainer.
Rename the private helper to `current_field_issues` in `_data_validation.py`
and its import/call in `likelihood.py`; update the review notes. No logic,
public API or spec changes. Spec 3.3/6.2/7.1 coverage remains unchanged.
Run Ruff, mypy and the existing likelihood-core tests in both dtype modes;
no new test is needed for this rename. Commit on A6, then stop.
Docs to-do: unchanged; D-026 remains open.

## A6.4b2 review follow-up — Readability (2026-10-22)

Status: **implemented, ready for review**, authorized by
"check the review in ../review/slices and update",
on A6 at `401ce87`. Read all of `../review/slices/A6.4b2.md`. One follow-up
commit, then stop; no A6.4c or constructor rewrite.

- R1: give likelihood evaluation named private stages for unusable data,
  mode support and safe evaluation, leaving the entry point as orchestration.
- R2: move normalized-field checks into `current_field_issues` in
  `_data_validation.py`. Explain reconstruction bypassing constructors,
  normalized exact upper=lower, and unavailable raw indicator evidence;
  retain the existing rules and shared row_issues packing.
- R3/R4/R6: build one bad-row mask, use its complement for substitutions,
  name the broadcast exact mask separately, and read data.issues.invalid_rows
  rather than the private bitfield.
- R5: cite dtype promotion; explain the reference, data-time and output
  casts, including references/overrides with fixed precision. Add a fixed-float32 custom family
  test that demonstrates why the output cast remains necessary.
- SQ1: record the recipe-wording gap as D-026, with a proposed clarification;
  no spec or diagnostic behavior change without the maintainer's decision.
- Run core and core-matrix focused tests in both dtypes, then the full
  through-A5 gate and git diff checks. Overwrite REVIEW.md with actual
  results and outstanding SQ1. No public API, docs/ or spec-test changes.
- Docs to-do: clarify the recipe's numerical remainder after edited fields,
  pending SQ1 approval; existing likelihood usage docs remain a later step.

Evidence: 62 focused core cases pass in both dtypes; the fixed-precision
regression also passes after pinning its infinity limit. An in-memory
output-cast deletion fails that regression in x64 as expected. The full
through-A5 gate passes in both dtypes (own suites 648/654 passed with
existing 7/1 skips). REVIEW.md records exact results. R1–R6 are implemented;
D-026/SQ1 remains open pending the separate spec-clarification decision.
The source/test diff is 172 lines; files remain below 300 lines.

## Current step — A6.4b2 implementation

Status: **implemented, ready for review**, approved by "now do the next bit
of slice A6". Starting at
`523532a`; returned from main to the existing authorized A6 branch at the
same commit. Implement the private core and tests below, then commit and
stop. Reuse the existing parameter-isolation matrix by running it against
both the helper and the core; no copied family/test scaffolding. A6.4c is
still separate. No new public API or docs/ edits in this step.

## A6.4b1 review fix-up — Two nested levels and precise test diagnostics

Status: **implemented, ready for review**; authorized by the explicit review request, based on
`2e7ae42` on `A6`. One fix-up commit, then stop. No A6.4b2 work.

- Add `tests/unit/test_likelihood_parameters_depth.py` with a custom wrapper
  nested twice around a Weibull. A NaN deepest scale sits between valid
  scales; both wrapper levels have shared valid parameters. Compare valid-row
  values and first/second derivatives with the hand-derived isolated formula,
  eager and jit, naming transforms via pytest ids. Spec 7.1/7.3.
- N1: retain the explicit broadcast in the helper. Correct REVIEW.md:
  implicit where broadcasting makes deleting that line behaviorally equivalent;
  structural tests catch skipped masking of rebuilt leaves, not that deletion.
- N2: add transform names to the existing loop's assertions, including the
  jit-vmap check. Keep the existing case builder and coverage intact.
- Run focused tests in both dtypes, Ruff/mypy and the through-A5 gate.
  Record actual results in REVIEW.md; no spec-test/public-API/docs changes.
- Docs to-do: unchanged from A6.4b1; no narrative documentation in this fix-up.

Evidence: all 14 depth cases pass in each dtype. An in-memory mutation that
skips the deepest masking is caught by four reverse-mode cases per dtype.
The full through-A5 gate passes (own suites 586/592 passed, existing 7/1
skips). REVIEW.md records exact results and the corrected N1 claim. Library
source is unchanged; stop after this review fix-up commit.

Status: **A6.4b1 implemented; ready for review**.
The maintainer's "ok now do the actual next step" authorizes A6.4b1 on `A6`
at `9099310`. Finish the helper and its tests, commit and stop; A6.4b2/c
remain proposed. A6.4a rework is complete.
Initial alignment on maintainer-selected branch `A6` at `21c2632` read the complete normative
diff from `be7b253` to `cf4fef9`, current sections 7.1–7.3/12.1, decision 0045,
and the A6.4a review and designer request. Ran spec-since.sh dev --mark;
.spec-seen remains `cf4fef9`. After the maintainer's sync/merge, planning is
now based on `f07c560`, with independent snapshot `08d97a6` (revision 32).
Read all seven synced-file changes; collection is not conformance evidence.

Review B1 demonstrates that A6.4a's family-local guards isolate only zeros;
NaN, infinity and negative violations still contaminate reverse derivatives.
Revision 32 settles the mechanism: the likelihood broadcasts every parameter
leaf to the current distribution batch shape, then stops its gradients only
at entries in parameter_issues().invalid_entries, before any family code.
Nested leaves and component axes are included. Parameter values stay unchanged;
final invalid-row contributions are NaN. Valid rows retain values, gradients
and Hessians in forward and reverse modes. Invalid-row gradients remain
unspecified, and undeclared invalidity has no isolation guarantee.

This expressly authorizes the invalid-entry exception to the older
stop_gradient wording in CLAUDE.md; valid parameters and active valid
calculations still must not be stopped. No edit to that protected file or
new design decision is needed here. Direct-method policy remains section 4.5.
The superseded zero-only guards are now removed. Rework and core integration
are separate reviewable steps below; the initial alignment pass changed no
code. A6.4a rework and the separately approved A6.4b1 are complete.

The expanded A6.4a–c plan below replaces the obsolete revision-28 likelihood
outline. Revision 29 requires one array-returning function plus the base
method; no LikelihoodIssues, likelihood _with_info variant, extra result tuple
or new issue container. The A6.5 outline is updated to match this split.

D-023's policy was approved after the maintainer requested an explanation:
NaN only on unsupported rows, eagerly and under transformations, preserving
supported rows and guaranteed derivatives. Such rows acquire no data/parameter
issue. The section 7.1 recipe now checks mode support after the diagnostics,
before attributing a remaining supported NaN to normalization/numerical failure.
Spec commit `be7b253` and decision 0044 (`9cc2ab2`) record the agreement.
D-023's formal status stays with the maintainer; no implementation is claimed.

The work is split up front to keep each implementation/test/docstring diff
around 300 lines or less. Each part needs separate approval, a maintainer-
selected branch, one commit, an overwritten REVIEW.md and a stop. Stop for
review of A6.4b1; A6.4b2/c remain proposed. D-023 and review B1's
design decisions are settled by revisions 31 and 32 respectively. Do not
rewrite the published A6.4a commit or treat its checks as revision-32 evidence.

Standing maintainer authorization (2026-10-20): "exception approved in
general". Code steps may add the minimal generated API-reference entries
needed alongside new exports, keeping the existing export-coverage check
passing. This is a general exception to the working agreement's code-step
docs/ restriction; do not request it again for each symbol or step. Narrative
guides, worked examples and broader reference changes remain separate docs
steps. A6.4c's one function autodoc entry is covered by this approval.

## Synced revision-31/32 acceptance coverage (2026-10-21)

The sync removes the pending acceptance-test prerequisite. It does not
change the chosen mechanism, step order or implementation approval status.
tests/spec/ remains read-only; no acceptance tests are copied into our suite.

- `test_a6_likelihood_gradient_isolation.py`: 41 violation/family cases;
  finite valid-row values and derivatives equal evaluation alone as functions
  of the same flat parameter vector, including shared leaves and zero cross
  derivatives. Checks jacfwd/jacrev and all four second-derivative combinations
  (fwd/rev over fwd/rev), plus selected jit/method checks and hand-derived
  Exponential, shared-shape Weibull and nested shared-log-ratio anchors.
  The matrix covers five violation classes, component leaves, shared leaves,
  unguarded custom families and base/outer named conditions.
- `test_a6_unsupported_modes.py`: seven modes across flag/bounds construction,
  truncation and censoring. Requires NaN without a value-dependent exception
  in both entry points; supported neighbors retain values, gradients and
  Hessians with per-row/shared parameters, including jit/traced construction
  and vmap value checks. Diagnostics stay clear; triage uses unsupported_nan
  rather than the numerical remainder. Q-037's exception alternative is gone.
- `helpers.py`, `test_a6_likelihood_parameter_issues.py`, TESTMAP.md and
  API_ASSUMPTIONS.md now reflect revisions 31/32. No API change is needed.
- Retain own helper/evaluator checks for current shapes after reconstruction,
  smaller leaves, input immutability, unconstrained sibling masking, draws
  and per-leaf vmap; the new isolation module does not exercise these paths.
  Own valid-interior transformation and analytical checks remain required.
- A6.4a rework and A6.4b1/b2 still gate through A5 while the public entry
  points are absent. A6.4c must explicitly run both modules above in each
  dtype, then the complete through-A6 gate. No tests are edited or skipped
  to make an intermediate step pass.

## Historical revision-31 planning evidence, superseded by revision 32

Read-only probes actually run with UV_CACHE_DIR=/private/tmp/survjax-uv-cache,
JAX_ENABLE_X64=1 and 0, `uv run --locked python`; all exited 0. No likelihood
code was saved. For exact/exact/right rows at times [1,1,3], a candidate
composition of direct methods followed by parameter-issue NaN selection gave:

- Exponential rates [0.5,0,2]: valid-row forward Jacobian is [[1,0,0],[0,0,-3]],
  but reverse mode contains NaN in the invalid rate's column of both valid
  rows, instead of zero.
- Weibull shape [2,0,2], scale=2: valid-row reverse derivatives in the shape
  column for the invalid row are NaN. The same happens for scale [2,0,2]
  with shape=2. Valid diagonal scale derivatives are -0.75 and 2.25.

These are observed prerequisite gaps in zero-cotangent arithmetic, not spec
ambiguities or evidence that an evaluator exists. No pytest, runtime gate,
static checks or docs build was run for planning. No branch or commit created.
`git diff --check` exited 0 with no output after the planning edits. Only
PLAN.md and DEVIATIONS.md changed; the read-only design diff from .spec-seen
to design HEAD has no changes to SPECIFICATION.md or GOALS.md.

Approved-policy alignment: specification revision 31 is committed separately
in design (`be7b253`); supporting decision/index links are in `9cc2ab2`.
Ran ../scripts/spec-since.sh dev --mark, then its read-only check: "dev: spec
unchanged since be7b253". Executed the revised recipe in both dtype processes
against supplied synthetic likelihood outputs; data/parameter issues,
unsupported modes and the numerical remainder separate with (2,6) masks.
This checks only the recipe, not an unimplemented evaluator. No runtime gate
or docs build was run for the spec/metadata change; git diff --check is clean.

# A6.3 review fix-up — Mask batch shapes (historical)

Status: **implemented; ready for review**. Read
`../review/slices/A6.3.md` against clean `A6` at `b0b44a5`. Both should-fix
findings S1/S2 are valid under spec 4.7's requirement that every mask has
batch_shape, including when reconstruction changes current leaf shapes.
One small follow-up commit on the same maintainer-selected `A6` branch,
followed by an overwritten REVIEW.md and a stop. No A6.4 work.

- S1: in `src/survjax/distributions/_parameter_issues.py`, broadcast each own
  constrained field mask to the current batch_shape after component reduction
  and before storing it, matching the existing nested path. Preserve values,
  polarity, ordering and aggregate calculation. No API/signature/export change.
- In `tests/unit/test_parameter_issues.py`, add a two-positive-field regression
  whose reconstructed leaves have batch shapes (1,) and (3,). Check eager/jit
  masks have exactly batch_shape and hand-enumerated values. Add a per-leaf
  vmap in_axes regression (one parameter mapped, the other shared), asserting
  the full stacked shapes and values so union broadcasting cannot hide a
  malformed individual mask.
- S2: strengthen the existing smaller-nested-batch regression with explicit
  `.shape == (3,)` assertions for both the nested field and condition masks.
  Keep its value assertions. Demonstrate the regressions reject removal of
  own, nested-field and nested-condition broadcasting, each separately, using
  in-memory mutants without changing the checkout or independent tests.
- N1: leave nonboolean custom constraint results outside this fix-up. Such
  results violate the existing bool-array protocol; the review explicitly
  leaves additional validation to the maintainer. Do not silently cast them
  or introduce a new exception policy. N2: retain the private-module split;
  no unrelated base.py refactor. No deviation entry or status change needed.
- Files: the private module and own test file above, PLAN.md and REVIEW.md.
  No docs/, spec/, tests/spec/, dependency, constructor or direct-method edit.
  Existing A6 docs to-do remains sufficient for this shape-contract repair.
- Validation after approval: UV_CACHE_DIR=/private/tmp/survjax-uv-cache;
  separate JAX_ENABLE_X64=1/0 focused own diagnostics/declaration suites plus
  complete independent A6 constraints/parameter-issues modules with
  SPEC_THROUGH=A6; mutation checks in both modes; then
  `scripts/check.sh --through A5` and `git diff --check`. Record exact results
  and any failures in REVIEW.md. No docs build or complete A6 gate is claimed.

Read-only probes actually run: JAX_ENABLE_X64=1 and 0, `uv run --locked python`.
Both exited 0. Eager and jit report batch_shape (3,), own field shapes
rate=(1,), cap=(3,), invalid_entries=(3,). In both processes,
`np.testing.assert_array_equal(jnp.array(True), [True]*3)` passes, confirming
the missing shape assertion. No pytest suite or gate was rerun for this plan.

Implementation evidence: own field masks now broadcast after reduction; the
new reconstructed-own and per-leaf vmap cases and the strengthened nested
case assert exact shapes. Focused tests pass in both modes (176 passed,
1 optional skip); all three broadcast-removal mutants are rejected by shape
assertions in both modes. The full through-A5 gate passes. Exact commands,
counts and scope limits are in REVIEW.md. Stop after this fix-up commit.

# A6.3 planning — Base-owned parameter diagnostics (historical)

Status: **implemented; ready for review**. A6.2 is merged; planning
starts from clean `main` at `a715d3e`. The local spec marker remains
`fd150db3e82ad744277caecd1c68b2a524ad24df` (revision 30), and the synced
acceptance snapshot remains `f527f75325db5cac62c87c5bd28c7213cce51d8f`.
The expanded A6.3 step below supersedes its revision-28 outline. A6.1/A6.2
branch, approval and review statuses below are historical. A6.4/A6.5 still
need likelihood-plan alignment before approval; D-023 does not block A6.3.

Read current GOALS, spec 4.1–4.2/4.5/4.7/6.3/7.2/10.2/10.5/12.1–12.2,
the merged container and family registration/batch-shape code, existing own
declaration tests, and the synced parameter-issues acceptance module. This
initial planning pass changed only PLAN.md. The maintainer approved A6.3 and
selected branch `A6`, currently at `a715d3e`; use that branch for this step.
Implementation ends in one commit, REVIEW.md and a stop. No later step is
authorized.

Planning verification: `git diff --check` exited 0 with no output; only
PLAN.md changed. The read-only design diff from .spec-seen to design HEAD
contains no changes to SPECIFICATION.md or GOALS.md. No new spec ambiguity
was identified; existing deviation statuses are unchanged.

# A6.2 planning — ParameterIssues container and host checkpoint (historical)

Status: **implemented; ready for review** (2026-10-20). Planning starts
from clean `main` at `7274dd0`, after A6.1 merged. The maintainer approved the
plan including the minimal API-reference entry, and explicitly authorized
creating branch `A6.2`; that branch is now selected. The expanded A6.2
step below is the current proposal and supersedes its revision-28 outline.
Earlier A6.1 status and test-sync notes below are historical: the local
`tests/spec/SOURCE_COMMIT` now records `f527f75325db5cac62c87c5bd28c7213cce51d8f`,
and `.spec-seen` matches revision 30 at
`fd150db3e82ad744277caecd1c68b2a524ad24df`.

Read current GOALS and specification sections 4.7, 6.2–6.3, 10.2, 10.5 and
12.1–12.2, existing DataIssues/checkpoint code, public-export checks, and the
synced A6 parameter-diagnostic acceptance tests. The initial planning pass
changed only PLAN.md; approval covers A6.2 only. It does not authorize or
implement A6.3–A6.5. Their original
likelihood plans still need revision-29 alignment before approval.

Approval of A6.2 includes the single API-reference exception
specified below: the new top-level exception needs an autodoc entry to satisfy
the existing export-coverage test. The working agreement otherwise defers
docs/ edits to docs steps; the earlier A6.1 exception was specific to A6.1.

# Spec alignment — revision 30 (2026-10-20; historical record)

Status: read and marked in `b26a461`; the maintainer resumed implementation on
2026-10-20. **A6.1 plain-comparison follow-up implemented, ready for review.**
The alignment commit made no source/test edits. The separately approved
implementation follow-up is recorded below; later A6 steps remain unapproved.

Reviewed the full normative diff from marker `0683e35` (revision 28) through
`fd150db3e82ad744277caecd1c68b2a524ad24df` (revision 30), and current GOALS.

- Revision 29 (`be8010c`), sections 4.1–4.2, 6.1–6.3, 7.1, 11.4 and 12:
  add the base-owned dist.log_likelihood(data=None, *, time=None, event=None),
  identical to the top-level function in values and derivatives. Remove
  log_likelihood_with_info and LikelihoodIssues; explain NaN rows using
  data.issues and parameter_issues(), broadcast to the contribution shape.
  Preserve invalid-row NaNs, genuine zero likelihood -inf and gradient
  isolation. Later A6 likelihood plans below remain historical until replanned.
- Revision 30 (`fd150db`), section 4.7 and decision 0042: use ordinary JAX
  comparisons in the parameter dtype, converting bounds to that dtype.
  Backend-flushed subnormal values/bounds and values within rounding of a bound
  may be classified either way. Add no extra handling, including at zero.
  The table still specifies finite x>=0 for nonnegative; zero remains valid.
  This settles the outstanding boundary clarification and the normative
  questions behind D-024/D-025; formal entry closure remains the maintainer's.
- After the independent tests are synced and the maintainer resumes work,
  update the plain-comparison follow-up below to the synchronized acceptance
  suite: remove frexp and the own strict subnormal-sign guarantee, retain
  signed-zero inclusion, ordinary domains, exact simplex threshold and bound
  TypeError coverage. No implementation change is made now.
- Run ../scripts/spec-since.sh dev --mark and confirm no unread spec changes.
  The marker records reading, not conformance. Commit alignment bookkeeping
  on A6.1 so scripts/sync_spec_tests.sh can start from a clean dev checkout.
  No runtime checks are needed or claimed for this metadata-only step.

---

# A6 — Revision 28 parameter diagnostics and exact/right likelihood

Status: **A6.1 plain-comparison follow-up implemented, ready for review** (2026-10-19). This section supersedes
older forward plans for A6; the A5 records below remain historical.
Planning starts on clean `main` at `9623594`. The maintainer approved A6.1
and created branch `A6.1`. Only that code/test step is authorized; later steps
still need approval. The maintainer creates/selects each step branch; do not
commit on main. Every implementation step needs its own
approval, tests, commit, overwritten REVIEW.md and stop.

## A6.1 follow-up — Plain comparisons without frexp (approved 2026-10-20)

Requested after `142b778`: remove jnp.frexp for transparent comparisons.
The maintainer explicitly resumed this work after the revision-30 spec and
independent-test updates. Same `A6.1` branch, one small follow-up commit,
overwrite REVIEW.md and stop. Later A6 steps remain unapproved.

Revision-30 tests are at tester HEAD `f527f75`, with matching .spec-seen.
This branch still records tests/spec/SOURCE_COMMIT `1d87b72`; the maintainer
has a spectests-A6r1 branch, but its snapshot has not arrived on A6.1. Run the
updated constraint module directly from ../spectest without merging/syncing
or changing protected paths.

- Revision 30 confirms inclusive `nonnegative` as finite x>=0. Signed zeros,
  zero weights and unit_interval(0) remain valid. No clarification is pending.
- `src/survjax/constraints.py`: replace all four frexp expressions with plain
  JAX comparisons: finite x>=0 for nonnegative, finite x>bound for
  _GreaterThan, and direct lower/upper comparisons in _Interval. No zero-bound
  special case or replacement decomposition/bit-level workaround. Keep the
  exact simplex tolerance and finite-bound error normalization unchanged.
- `tests/unit/test_constraints.py`: replace the own mathematical-subnormal-sign
  assertion with explicit signed-zero boundary inclusion/exclusion cases.
  Keep the existing ordinary domain, nonfinite, transformation, simplex exact
  threshold and huge-bound coverage. Do not assert eager/jit identity for
  subnormal values when the chosen backend does not provide it.
- Public API: same signatures/exports. Values at subnormal boundaries inherit
  JAX/backend comparisons, per the maintainer's preference. No docs, spec,
  dependency or later-step changes. D-025 records the pre-revision-30 acceptance
  conflict; the revised tester module now avoids those subnormal assertions.
  Entry statuses remain the maintainer's responsibility.
- Validation: focused own constraints/declarations/import tests and the
  independent A6 constraint module in both dtype processes, then the full
  through-A5 gate, Ruff/mypy and git diff --check. Run against the newly synced
  revision-30 tester module and report all results; do not change/skip/xfail it.
  No docs build needed if docstrings and narrative docs remain unchanged.

Read-only probe (JAX_ENABLE_X64=1/0, uv run --locked python): on
[-nextafter(0,1), -0, 0, nextafter(0,1)], finite x>=0 gives
[True,True,True,True] eagerly/vmap and [False,True,True,True] under jit,
for both dtypes in this CPU environment. This is observed backend behavior,
not a verified cross-backend guarantee. The external independent
`test_unit_interval_includes_zero_and_one` expects False for the first input.
No pytest or gate was rerun during this planning check.

## A6.1 review fix-up — S1 and small nits (approved)

Reviewed `../review/slices/A6.1.md` against `98be832` on `A6.1`.
The maintainer approved this fix-up and chose JAX parameter-dtype/backend
comparison behavior for Q1. One small fix-up commit on this same branch,
then overwrite REVIEW.md and stop.

- S1: add one focused simplex tolerance regression to
  `tests/unit/test_constraints.py`, exercised in both separate dtype processes.
  With n=10, all entries zero except x[0]=1+k*eps, the sum is exact. Check
  k=30 and 40 accepted, k=41 and 50 rejected, eagerly/jit/vmap. This covers
  the review's 3n/5n cases and pins the inclusive 4n endpoint and its next
  representable neighbour. Expected masks come directly from spec 4.7's
  abs(sum-1)<=4*n*eps; no assertion tolerance or new tolerance constant.
  Demonstrate that the review's M2 tenfold-tolerance mutant fails this test
  in both modes, using an in-memory mutant rather than editing the checkout.
- N1: in `src/survjax/constraints.py`, translate overflow from the finite-bound
  check to TypeError with the existing bounds message. Add a small regression
  for 10**400 through all three public bound factories, including both
  interval endpoints. No other domain/value-policy change is proposed.
- N2: overwrite REVIEW.md after implementing the fix-up; state precisely that
  the callable dataclass instances are frozen, while module functions have
  mutable Python attributes. Do not change function representations.
- N3: no registration refactor in this fix-up. A6.3 still puts the new
  diagnostic composition in a separate private module as planned.
- Q1: record the maintainer's decision in D-024: follow JAX parameter-dtype
  and backend comparison behavior. Preserve the current implementation; no
  new mathematical Python-bound comparison or spec edit. Formal deviation
  closure stays with the maintainer.
- Public API touched: no signatures or exports; N1 changes the exception for
  an overflowing bound check. Files: constraints.py, test_constraints.py,
  PLAN.md and REVIEW.md; deviation bookkeeping in DEVIATIONS.md. No docs,
  independent-test, dependency or later-slice changes.
- Validation: focused own constraints/declarations/import tests plus the
  independent A6 constraint module directly from ../spectest in both modes;
  M2 rejection in both modes; scripts/check.sh --through A5; git diff --check.
  Record exact outputs, including mutation failures, in REVIEW.md. No docs
  rebuild is needed for this test/exception fix with unchanged docstrings.

Read-only review probes actually run with UV_CACHE_DIR=/private/tmp/survjax-uv-cache
and JAX_ENABLE_X64=1/0, using uv run --locked python: both modes report current
simplex masks [True, True, False, False] for k=[30,40,41,50], versus
[True, True, True, True] for an in-memory M2; 10**400 raises OverflowError.
No pytest suite or runtime gate was rerun while checking this review.
The maintainer's previous "ok make the minimal entry" explicitly authorized
A6.1's docs/api.md exception; the review's scope question needs no new approval.

Revision-29 follow-up: read `../scripts/spec-since.sh dev` and the complete
0683e35..be8010c normative diff. A6.1's sections 4.2/4.7 are unchanged.
Do not implement the stale revision-28 likelihood references below (A6.2 is
now separately replanned against revision 30):
remove LikelihoodIssues/log_likelihood_with_info and add base-owned
`dist.log_likelihood`, equal to the function in values/derivatives, in a
separate planning/alignment pass before the next step is approved. D-023 has
an open revision-29 note. Leave .spec-seen at revision 28 for this review
check; re-mark it as part of that explicit alignment pass.

## Contract and independent tests reviewed

- Spec revision 28, design commit `0683e358a312a8a1f8264af8f60eb61b0dba7a69`.
  Read the normative diff from the previous marker `b92334a`, current GOALS,
  and the relevant current sections 4–7, 10.2 and 12. The intervening
  `9d7483c` adds PDF helpers, with no runtime contract change.
- Revision 28 adds declared constraints, base-owned `parameter_issues()`,
  `parameter_conditions()`, `ParameterIssues`, `InvalidParameterError`,
  built-in declarations and likelihood invalid-parameter attribution. Direct
  methods retain their existing value policy; constructors do not check values
  or transform parameters. Slice A explicitly includes this contract.
- Reviewed the new independent A6 modules and Q-024/Q-034–Q-038 in
  `../spectest/tests/spec`, commits `b9bce65` and `e22001c` (tester HEAD).
  They cover constraints, parameter diagnostics, likelihood parameter issues,
  entry points, values, shapes, stored data issues and unsupported modes.
- Dev's `tests/spec/SOURCE_COMMIT` is still `1d87b72` (revision 27).
  The new tests have **not** been synced here. Alignment of `.spec-seen` is
  requested now; syncing requires the maintainer-created `spectests-*` branch
  and `scripts/sync_spec_tests.sh ../spectest`, separately from code work.
  Do not edit/copy tests into that protected directory while implementing.
- D-013's return-contract question is answered by revision 28. Leave its formal
  status to the maintainer. D-023 below records the remaining unsupported-mode
  contract gap identified by tester Q-037; approval of the early parameter
  steps need not wait for that likelihood decision.

## Step A6.1 — Constraint vocabulary and declarations

Spec: 4.1, 4.2, 4.7, 10.2, 12.1. Target about 300 changed lines including tests.

- Create `src/survjax/constraints.py`: typed pure callable constraints `real`,
  `positive`, `nonnegative`, `unit_interval`, `ordered`, `simplex`, and bound
  factories `greater_than`, `interval`, `closed_interval`. Use plain functions
  and small frozen callable dataclasses as needed, with `event_dim` metadata.
  Check finite Python bounds and ordering with TypeError for misuse. Preserve
  the specified NaN/infinity domains; simplex uses exactly `4*n*eps` of the
  parameter dtype, permits zeros, and reduces only the final axis.
- Extend `param(component_ndim=0, constraint=None)` in
  `src/survjax/distributions/base.py`: store constraints in field metadata;
  validate callable/event-axis compatibility at registration, without running
  constraints at construction. Add declaration-ordered read-only
  `Family.parameter_constraints`, excluding nested fields. Preserve constraint
  object identity and the existing PyTree leaves and construction behavior.
- Declare positive rate/shape/scale in `exponential.py` and `weibull.py`.
  Export the constraints module from `src/survjax/__init__.py`.
- Update the explicit export set in `tests/unit/test_import.py`.
- Add `tests/unit/test_constraints.py` and
  `tests/unit/test_parameter_constraints.py`: boundary/domain masks,
  malformed bounds/declarations, metadata order/immutability/identity,
  component compatibility, no added leaves or constructor checks, built-ins,
  and eager/jit/vmap. Check simplex softmax lengths through 1000 and rejection
  of a length-10 sum of 1.001. Expected masks follow the section 4.7 domains.
  Keep focused cases compact; split before exceeding the step budget.
- Maintainer approved the minimal constraints autodoc entry in docs/api.md
  during A6.1 so the existing public-export documentation check stays valid.
- Docs to-do: vocabulary and declaration recipe after code review.

## Step A6.2 — ParameterIssues container and host checkpoint

Status: **implemented, ready for review**. One container/checkpoint concept and one
implementation commit, targeting at most about 300 changed lines including
tests and review notes. Keep diagnostics.py below about 300 lines; if the
implementation or step budget grows beyond this, stop and propose a split.

Spec coverage: 4.7 (container fields, mask polarity/order and checkpoint),
6.2 (checkpoint behavior mirrored from DataIssues), 6.3 (PyTree result),
10.2 (module/export), 10.5 (public docstrings and minimal reference), and the
applicable container/checkpoint requirements of 12.1–12.2. No likelihood or
family aggregation is required for this step; D-023 does not block it.

### Files and public API

- `src/survjax/diagnostics.py`: add a small typed frozen `ParameterIssues`
  dataclass with four explicit fields: `fields: dict[str, jax.Array]`,
  `conditions: dict[str, jax.Array]`, `invalid_entries: jax.Array`, and
  `ok: jax.Array`. Document True-as-violation, batch-shaped masks and scalar
  ok. Like DataIssues, it stores supplied arrays; A6.3's producer will compute
  the union and scalar reduction. No new validation/normalization framework,
  derived properties, custom repr, or deep-immutability promise for the dicts.
- Register the container as a PyTree with explicit flatten/unflatten methods.
  Store the two ordered key tuples as static auxiliary data; expose their
  mask values, invalid_entries and ok as array leaves. Reconstruct dicts in
  that order, including dotted names, empty mappings and nonalphabetical
  keys. Do not return raw dict children to JAX's sorted-dict flattening.
  Unflatten only reconstructs: no array operations, host conversions or
  recomputation of ok, including when vmap stacks per-example scalar ok.
- Add `InvalidParameterError(ValueError)` and
  `ParameterIssues.raise_if_invalid(self) -> None`. Reject tracing before
  any NumPy conversion by checking all array leaves, even if ok is concrete;
  raise a plain ValueError containing "host-only" for both valid and invalid
  traced diagnostics. No callbacks. For concrete masks, reuse `_rule_counts`
  without changing DataIssues behavior, returning None when no violation
  fires and otherwise naming all violated fields and conditions, their counts
  and first batch-index tuples, with dotted names intact. Qualify the two
  categories so matching field/condition names remain distinguishable.
- `src/survjax/__init__.py`: re-export InvalidParameterError beside
  InvalidDataError and add it to __all__. ParameterIssues remains accessible
  as `survjax.diagnostics.ParameterIssues`, following DataIssues; no top-level
  container alias. Public docstrings cover attributes, return and exceptions.
- `tests/unit/test_import.py`: extend the explicit export assertion and
  check that the exception is the diagnostics class and subclasses ValueError.
- `tests/unit/test_parameter_issue_container.py`: create the focused tests
  below. No changes to tests/spec/, spec/, dependencies or check scripts.
- `docs/api.md`, **only if the requested exception is approved**: add one
  `autoexception:: survjax.InvalidParameterError` block. No narrative guide or
  family-method documentation in A6.2. Keep the existing export-coverage test.
- PLAN.md: record approval/completion; REVIEW.md: overwrite after the approved
  implementation with reading order, invariants, test/spec mapping, exact
  check output and uncertainties. Leave the A6.1 review intact while planning.

### Tests and evidence

Construct containers directly from hand-enumerated boolean masks, with explicit
expected invalid_entries/ok. This verifies storage and transformations;
calculation of those aggregates from family declarations belongs to A6.3.
Use exact boolean/structural assertions, without numerical tolerances.

- Spec 4.7/6.3: scalar, multidimensional and empty-mapping cases retain their
  masks, shapes and scalar ok through tree flatten/unflatten. Assert frozen
  field assignment and that leaves are arrays, while keys remain metadata.
- Spec 4.7/6.3: deliberately unsorted own/dotted field and condition keys
  retain insertion order and values after tree round-trip, jit and vmap.
  Jitted identity must preserve even a vmap result's vector ok; vmap of scalar
  examples returns the expected vector of per-example ok values.
- Spec 4.7/7.2: changing only same-shaped mask values reuses a jitted function
  (trace-count assertion), with fixed keys and structure and updated results.
- Spec 4.7/6.2: all-valid and empty mappings return normally; field-only,
  condition-only and overlapping violations raise InvalidParameterError.
  Assert all names/categories, counts and first indices: scalar `()`, vector
  `(i,)`, and multidimensional `(i, j)`. Expected counts/indices are enumerated
  in the tests, not computed by `_rule_counts` or the container under test.
- Spec 4.7/6.2/6.3: a concrete container returned by jit can be checked on the
  host. Calling the checkpoint inside jit/vmap with valid or invalid traced
  masks raises exactly plain ValueError with "host-only". Include a case
  with concrete ok and a traced mask to catch incomplete tracer guards.

### Validation and stop

After approval, use UV_CACHE_DIR=/private/tmp/survjax-uv-cache. Run focused
container/import/DataIssues regression tests in separate JAX_ENABLE_X64=1
and 0 processes. Run the independent
`tests/spec/test_a6_parameter_issues.py::test_invalid_parameter_error_is_a_public_value_error`
with SPEC_THROUGH=A6 in both modes; it is the independent A6.2-only check.
The other tests in that module call the A6.3 producer and are not yet an A6.2
gate. Do not edit, skip or xfail them.

Run `scripts/check.sh --through A5`: it includes Ruff format/lint, mypy src,
the entire own suite in both modes, source doctests and independent A0–A5
tests. If the minimal API-reference exception is approved, also run
`scripts/build_docs.sh` clean. Finish with `git diff --check`. Do not claim
the complete A6 gate, numerical derivatives, R fixtures, install checks or
benchmarks. No gradient tests are needed for a boolean-only result container.

Write REVIEW.md with exact commands, exit codes, pass/skip counts and any
failures, commit this step only on the maintainer's A6.2 branch with an
imperative subject and a body explaining why, then stop for review. Do not
start A6.3.

Planning verification actually run: `git diff --check` exited 0 with no
output. Only PLAN.md changed. No runtime tests, static checks or docs build
were run for this planning-only edit; no branch or commit was created.

Docs to-do (A6 docs step): document ParameterIssues fields and polarity,
ordered/dotted names, explicit host checkpoint and traced rejection, and
clarify that no reported issue is not a validity guarantee. Add the container
reference and examples using the A6.3 producer after that producer exists.

Implementation evidence (2026-10-20): 17 container cases added; focused own
regressions plus the independent exception check pass (32 tests per dtype).
`scripts/check.sh --through A5` passes both modes; `scripts/build_docs.sh`
passes after a sandbox port restriction required an escalated rerun. Exact
commands, counts, invariants and remaining scope are in REVIEW.md. No A6.3
producer or likelihood work started. Stop after the A6.2 implementation commit.

## Step A6.3 — Base-owned diagnostic composition

Status: **implemented, ready for review**. One concept: compose the declared
parameter diagnostics into the A6.2 result. Target about 250–300 changed
implementation/test/docstring lines: roughly 75 for the private module,
25 for the base hooks and 150–180 for focused own tests. If this cannot fit,
propose a split before coding beyond the approved scope. No registration
refactor or general diagnostic framework.

Spec: 4.1–4.2 (pure methods and extension hooks), 4.5 (direct values unchanged),
4.7 (composition, order, shapes and hook errors), 6.3 (PyTree results), 7.2
(compiled execution and reuse), and relevant 12.1–12.2 acceptance checks.

### Files and public API

- `src/survjax/distributions/base.py`: expose typed base methods
  `parameter_conditions() -> Mapping[str, jax.Array]` and
  `parameter_issues() -> ParameterIssues`, following the existing method
  attachment pattern where useful. The default condition hook returns a fresh
  empty mapping. Families override that hook; parameter_issues is base-owned.
  Include public docstrings describing violation polarity, exact batch shapes,
  ordering, return type and TypeError for malformed hook results. No new
  top-level exports, constructor arguments or distribution PyTree leaves.
- Create `src/survjax/distributions/_parameter_issues.py` for aggregation;
  use TYPE_CHECKING imports to avoid a base/private-module import cycle.
  The 449-line base module gets only the small hook integration, without
  moving existing registration, dtype, broadcasting or value-policy code.
- For each constrained own parameter in declaration order, evaluate its
  constraint, invert the valid-domain mask, and reduce exactly the remaining
  trailing component axes (`component_ndim - event_dim`) with any. Preserve
  all batch axes, including scalar batches. Use declaration metadata and
  current leaf shapes; never infer an undeclared parameter's validity domain.
- Recursively obtain each nested distribution's issues in field declaration
  order, prefix all field/condition names with `field_name + "."`, and
  broadcast masks to the current outer batch_shape. Retain deeper dotted
  prefixes. Fields contain own entries before nested entries; conditions
  likewise contain own hook entries before nested entries. Preserve false
  masks and unsorted insertion order so values cannot change the result tree.
- Validate the own condition hook using Mapping, key types/content, array
  dtype and exact shape only. Reject non-mappings, non-string/empty/dotted
  names, nonboolean values and any shape other than batch_shape with TypeError,
  also while tracing. Accept other Mapping implementations and preserve their
  iteration order. Do not coerce invalid values or broadcast condition masks.
  Fixed keys are the hook author's contract; add no key registry or cache.
- Start invalid_entries as a boolean all-False array of batch_shape, OR every
  field/condition mask into it, then compute scalar ok as negated any. Empty,
  undeclared and nested-only families use the same path. Return ParameterIssues
  directly; its reconstruction and explicit host checkpoint stay unchanged.
- Create `tests/unit/test_parameter_issues.py` for the evidence below.
  PLAN.md records approval/completion and docs to-do; overwrite REVIEW.md only
  after the implementation. No changes to diagnostics.py, docs/, spec/,
  tests/spec/, dependencies, public exports or gate scripts are planned.

### Tests and invariants

Use small frozen custom families and explicit expected boolean masks derived
from their inequalities. Do not copy the independent fixture families or
repeat A6.1's entire constraint-domain suite. Exact mask assertions need no
new tolerance; use existing tolerances for numerical identity comparisons.

- Spec 4.7: own fields with no constraint, scalar constraints on component
  arrays, and event_dim smaller than/equal to component_ndim. Hand-enumerate
  failures in different batch/component positions to distinguish any from all
  and catch accidental batch reduction. Verify union, boolean dtype and
  scalar ok, including no declarations, conditions only and scalar batches.
- Spec 4.7: nested and nested-only families, own-first unsorted key order in
  both mappings, dotted prefixes, nested conditions and outer broadcasting.
  Include reconstruction with a smaller nested batch shape to exercise
  diagnostic broadcasting independently of constructor broadcasting.
- Spec 4.7/6.3: compare eager, jit, vmap and traced-construction diagnostics
  against explicit masks; check per-example ok under vmap and the recomputed
  reduction after indexing. Verify keys/tree structure stay fixed between
  valid/invalid parameter values, with a one-trace compilation-reuse check.
  Exercise compiled diagnostics under a device-to-host transfer guard after
  input placement; host checkpoints remain outside that guard.
- Spec 4.7: malformed conditions raise TypeError eagerly and while tracing;
  include key, dtype and exact-shape errors and an accepted non-dict Mapping.
  The independent module supplies the complete eight-case error matrix.
- Spec 4.1/4.5: from_log overflow/underflow is diagnosed on the stored natural
  parameters. Constrained/undeclared twins retain matching direct values,
  including a finite formula at a declared violation; a hook that fails if
  called demonstrates constructors/direct methods never invoke diagnostics.
  Include an explicit host checkpoint on diagnostics from the new producer;
  the full host-error matrix is already covered by A6.2 and acceptance tests.

### Validation and stop

After approval use UV_CACHE_DIR=/private/tmp/survjax-uv-cache. In separate
JAX_ENABLE_X64=1 and 0 processes run focused own parameter-issue/container/
declaration tests, then with SPEC_THROUGH=A6 run the complete synced
`tests/spec/test_a6_constraints.py` and
`tests/spec/test_a6_parameter_issues.py`. Do not edit, skip or xfail acceptance
tests. NumPyro interoperability may retain its existing explicit skip when
that optional package is absent; do not install a new dependency.

Run `scripts/check.sh --through A5` for the protected-path guard, Ruff
format/lint, mypy src, the full own suite in both modes, source doctests and
independent A0–A5 checks. Finish with `git diff --check`. Boolean diagnostic
masks need no gradient suite. No docs build is planned because docs/ and
executable examples are unchanged. No complete through-A6 gate is claimed:
likelihood work and its independent tests remain A6.4/A6.5.

Write REVIEW.md with reading order, these invariants, test/spec mapping,
exact commands/counts/outputs and remaining concerns. Commit only on the
maintainer-selected `A6` branch with an imperative subject and a body explaining why,
then stop. No A6.4 work, unsupported-mode decision or deviation closure.

Docs to-do (A6 docs step): document both new methods and the hook contract,
including violations-as-True versus constraints' validity-as-True; component
reductions; own/nested ordering and dotted prefixes; fixed keys; no-issue
limitations; and custom-condition, nested-family and compiled-host-checkpoint
examples using the merged producer. No API reference exception is needed:
the export set stays unchanged.

Implementation evidence (2026-10-20): producer and hook integrated on `A6`;
299 production/test lines changed. The focused own/independent suite passes
in both dtypes (173 passed, 1 optional skip per mode). The through-A5 gate
passes; exact counts and commands are in REVIEW.md. No spec/acceptance-test
edits, new dependency or A6.4 work. Stop after this step's commit.

## Step A6.4a rework — Withdraw the superseded family-local guards

Status: **implemented, ready for review** (2026-10-21).
Replace the approach in `21c2632` with a follow-up commit on `A6`, preserving
history. Spec 4.5/7.1–7.3/12.1, revision 32; public API unchanged.

- Restore only `src/survjax/distributions/exponential.py` and `weibull.py`
  to their pre-A6.4a implementations at `3b0c3bf`. Retain all A5 guards for
  valid-parameter time boundaries. No parameter-domain repair or replacement
  direct-method policy is introduced.
- Remove `tests/unit/test_likelihood_operands.py`, whose test-only evaluator
  and zero-only isolation checks served the rejected mechanism. Replacement
  isolation tests belong with the actual parameter preparation in A6.4b1
  and its evaluator integration in A6.4b2. Existing direct-method, boundary
  and valid-interior derivative suites remain and must pass in both dtypes.
- Verify the two source files exactly match `3b0c3bf`, then run focused
  Exponential/Weibull and A6 constraint/parameter-issue suites, followed by
  scripts/check.sh --through A5. This rollback makes no isolation claim.
- Target about 200 changed source/test lines, mainly deletions. No new
  tests of the discarded behavior, helper module or export in this part.
- Docs to-do: no direct-method documentation change; explain the central
  likelihood mechanism with its implementation in later parts.

Implementation evidence: both distribution files exactly match `3b0c3bf`;
the obsolete operand-test module is removed. Focused own/A6 diagnostic
suites pass (176/177 passed, 2/1 existing skips), and the full through-A5
gate passes in both dtypes, including A5 boundary/derivative regressions.
194 source/test lines changed, mainly deletions. REVIEW.md records exact
commands and counts. No new helper, likelihood API or A6.4b1 work in this
step; stop after committing for review.

## Step A6.4b1 — Prepare parameter leaves for likelihood evaluation

Status: **implemented, ready for review** (2026-10-21).
One concept: section 7.1's parameter-gradient barrier. Spec 4.1/4.7/5/7.1–7.3/
12.1, revision 32. Target at most 300 implementation/test/docstring lines.

- Create private `src/survjax/_likelihood_parameters.py`: a typed helper
  accepts a distribution and its declared invalid-entry mask, reconstructs
  a temporary distribution without invoking constructors, and preserves
  types/static configuration. No public API, family-specific dispatch,
  custom derivative rule or registry change.
- Use field metadata to broadcast own and recursively nested parameter
  leaves to the outer distribution's current batch_shape plus each leaf's
  component_shape, including leaves rebuilt smaller than their siblings.
  Expand the outer union mask across component axes. Apply
  `where(mask, stop_gradient(leaf), leaf)` only after broadcasting; do not
  substitute numerical values, alter diagnostics or stop valid gradients.
- Add `tests/unit/test_likelihood_parameters.py`: assert unchanged leaf
  values (including NaN/inf), exact shapes/static fields and input immutability;
  compare selected valid contributions and their first/second derivatives
  with those rows evaluated alone, plus hand-derived Exponential checks.
  Exercise zero, negative, NaN and both infinities in built-ins, an unguarded
  custom family and a nested custom family, plus a named-condition violation.
- Cover component axes, an unconstrained sibling leaf in an invalid entry,
  outer-only condition masks reaching nested leaves, reconstructed smaller
  leaves, a shared leaf, per-leaf vmap and a draws batch shaped (S,1).
  Use eager/jit/vmap, forward/reverse Jacobians and all four compositions
  of forward/reverse differentiation to second order, matching revision-32
  acceptance coverage. Include a shared valid leaf beside a per-row violation.
  Make no assertion about invalid-row gradients. These helper-level checks
  are not yet evidence for the complete likelihood contract.
- Run focused tests and the through-A5 gate in both dtypes. Keep tolerances
  in tests/impl_tolerances.py; no tolerance loosening. If the full matrix
  exceeds the budget, replan before adding more code or tests.
- Docs to-do: declared constraints/conditions suffice for likelihood
  isolation; undeclared invalidity and direct-method calls are outside it.

Original implementation evidence (`2e7ae42`): the helper and 38 tests add 300 lines
(41 source, 259 tests). All 35 violation/family cases preserve valid-row
values and derivatives through second order, including jit/vmap and shared
leaves. Structural tests cover smaller rebuilt nested leaves, two component
axes, per-leaf vmap and draws columns. The full through-A5 gate passes in
both dtypes (own suites 572/578 passed, existing 7/1 skips). Initial test-hook
shape errors and their correction are recorded honestly in REVIEW.md.
No public likelihood or core integration yet; stop after this commit.

## Step A6.4b2 — Private exact/right likelihood core

Status: **implemented, ready for review**. D-023's policy is
approved in revision 31; parameter isolation follows revision 32.
One concept: evaluate prepared SurvivalData. Spec 3.2–3.3/4.2/4.5/5/6.2/7.1–7.3/
12.1–12.2. Target at most 300 implementation/test/docstring lines; no public
entry point until A6.4c, so no partially functional public export.

- Create `src/survjax/likelihood.py` with a typed private prepared-data core
  and only the private validation support it needs. Reuse row_issues from
  `_data_validation.py` for normalized endpoint/window checks, combining them
  with stored raw-input bits. Do not rebuild through a constructor, erase
  evidence, mutate the input, or recompute concrete metadata during evaluation.
  Newly invalid values after tree reconstruction must also produce NaN.
- Compute the trailing-axis broadcast of data shape and current batch_shape;
  incompatible shapes raise TypeError. Promote time arrays and temporary
  distribution leaves to the common data/parameter float dtype, including
  nested leaves; preserve original objects, component axes and static fields.
  Diagnostics and mode masks must explicitly have the contribution shape.
- Obtain parameter_issues().invalid_entries and apply A6.4b1's helper before
  calling reference_time, log_prob, log_survival or any other family code.
  Broadcast the same declared mask to the contribution shape for final NaN
  selection. Do not recompute it from repaired values or infer extra domains.
- Support only exact rows and non-exact rows with upper=+inf, all with
  left_trunc=0 and right_trunc=+inf. Determine this from current array fields;
  all-true traced metadata must not reject supported inputs. Derived-kind
  precedence treats (0,+inf] as right censoring, not unsupported left censoring.
- Dispatch through dist.log_prob and dist.log_survival, including custom
  overrides and generic defaults. Before inactive density evaluation use the
  stopped reference_time; before inactive set probability use the untruncated
  window's zero lower endpoint through the base boundary method. Substitute
  inactive/invalid/unsupported observation operands before operations. Valid
  parameters remain differentiable; the only stopped parameter entries are
  those already declared invalid under revision 32. No family-type dispatch.
- Exact rows contribute log density; right-censored rows log survival. At
  censoring time zero the base's log S(0) yields structural zero with zero
  parameter gradient. Preserve true zero density/survival as -inf. Do not
  compute truncation normalization, interval probabilities or Slice B pruning.
- Force final NaN on data-issue rows, parameter_issues().invalid_entries and
  unsupported rows per the approved D-023 policy. Do not invent parameter
  validity domains or assign unsupported observations a data/parameter issue.
  There is no numerical_failure mask or result container; unexplained family
  NaNs remain NaN. Invalid or infinite rows have no derivative guarantee.
- Add `tests/unit/test_likelihood_core.py`: hand-derived Exponential/Weibull
  values, exact/right mixtures, custom dispatch, scalar/paired/draw-column
  broadcasting with explicit shape assertions, dtype promotion, stored bits
  and changed normalized values after indexing/reconstruction, -inf zero mass,
  and zero-censoring value/gradient. Include supported neighbors of invalid
  or unsupported rows and the approved eager/jit unsupported-mode behavior.
- Verify eager/jit/vmap and valid-row values/Jacobians/Hessians against
  isolated valid rows through the actual core, reusing A6.4b1's family cases
  and violation matrix. Cover exact and right-censored contributions and
  all four second-derivative compositions. For unsupported rows separately,
  verify supported neighbors' gradients and Hessians with per-row and shared
  parameters; parameter-issue masking must not be used to flag unsupported
  modes. Run a valid mixed batch eagerly with
  jax_debug_nans. The public independent modules are not an A6.4b2 gate:
  the public API arrives
  in A6.4c. Do not edit or skip tests to accommodate this intermediate step.
- Docs to-do: supported subset, return shape/dtype, stored-issue propagation,
  zero-mass and zero-censoring behavior, D-023 policy and triage limitation.

Implementation evidence: a 69-line private evaluator, 219 lines of focused
core tests and 9 changed lines reusing the existing matrix (297 source/test
lines total). The 35 core isolation cases and 26 new core cases pass in
both dtypes through the full through-A5 gate; own suites report 647/653
passed with existing 7/1 skips. Mixed valid likelihoods pass eager
jax_debug_nans. REVIEW.md records exact counts and the corrected NumPy
expected-value expression. No public function/method/export or A6.4c work;
stop after this commit for review.

## Step A6.4c — Public function, method and shared input convention

Status: **implemented, ready for review**; see the current A6.4c entry at the top.
Follows reviewed A6.4b2. One integration
concept, targeting 150–250 changed implementation/test/docstring lines.
Spec 4.2/5/6.1–6.3/7.1–7.3/10.2/10.5/12.1–12.2.

- Add `log_likelihood(dist, data=None, *, time=None, event=None) -> jax.Array`
  to likelihood.py. One private resolver accepts prepared SurvivalData or
  constructs it with the pure checked raw time/event constructor, with both
  keywords required. Mixed/missing forms, positional raw arrays, constructor
  keywords and incompatible shapes raise TypeError, eagerly and under tracing.
  Preserve constructor dtype/warning behavior and raw invalid-indicator bits.
- Add the matching typed base method to distributions/base.py, delegating
  directly to the function. Use a local/TYPE_CHECKING import as needed to
  avoid a cycle. No alternative evaluator or copied constructor signature.
- Re-export exactly log_likelihood in src/survjax/__init__.py and extend
  tests/unit/test_import.py. No log_likelihood_with_info, LikelihoodIssues,
  reduction, weights, opt-out validation or convenience aliases.
- Add `tests/unit/test_likelihood_entry_points.py`: raw/prepared equality;
  malformed argument combinations and invalid raw values; traced construction;
  function/method values, gradients and Hessians against hand-derived smooth
  exponential expectations. Pin unchanged stored issue evidence through both
  forms. Public docstrings cover args, shapes, returns, errors and Slice A's
  unsupported-mode policy, with longer examples deferred to docs.
- Use the standing docs/api.md exception recorded at the top: one
  `autofunction:: survjax.log_likelihood` block, keeping the export-coverage
  test unchanged. No narrative guide or method-reference expansion here.
- The complete through-A6 gate runs every synced A6 likelihood module,
  including method, shapes, values, invalid-data, parameter-issue and
  entry-point tests in both dtypes.
  Explicitly run `SPEC_THROUGH=A6 uv run --locked --group test pytest
  tests/spec/test_a6_likelihood_gradient_isolation.py
  tests/spec/test_a6_unsupported_modes.py -q --tb=short` in separate
  JAX_ENABLE_X64=1/0 processes with the cache setting below. These public-API
  suites are available now; do not defer isolation to A6.5 or A9.
  Then run scripts/check.sh --through A6 and scripts/build_docs.sh.
  Record actual skips and failures without modifying the independent tests.
  A passing A6 gate is not completion of Slice A or the later A6.5 verification.
- Docs to-do: public usage, raw/prepared equivalence and errors, unsummed
  contributions, function/method interchangeability, diagnostics/triage and
  the D-023 limitation. Full API-method reference and examples follow merged
  code in the A6 docs steps.

### Validation and delivery for each A6.4 part

Use UV_CACHE_DIR=/private/tmp/survjax-uv-cache and separate JAX_ENABLE_X64=1/0
processes. Run focused own tests and applicable existing independent modules,
then scripts/check.sh --through A5 for A6.4a rework/b1/b2 and through A6 for A6.4c.
Those gates include Ruff format/lint, mypy src, the entire own suite, source
doctests and independent acceptance checks in both modes. Finish with
git diff --check. Keep tests/spec/ and the gate scripts unchanged.

Each approved part updates PLAN.md, overwrites REVIEW.md with reading order,
invariants, test/spec mapping, exact commands/counts/outputs and uncertainties,
then makes one imperative-subject commit (body explains why) on the
maintainer-selected step branch. Stop after every part; do not start the next
part without instruction. Replan before exceeding the code/test budget.

## Step A6.5 — Likelihood composition and derivative regressions

Status: forward outline, not approved. Spec 4.7/6.3/7.1–7.3/12.1. Broader
verification after A6.4c; revision-32 parameter isolation belongs to
A6.4b1/b2 and must not be deferred to this step.
Replan against the reviewed public implementation before approval.

- Add `tests/unit/test_likelihood_transformations.py`: eager/jit/vmap agreement,
  prepared and traced construction, stored issues, compilation reuse with new
  same-shaped values/minibatches, and device transfer guard after warmup.
- Extend the A6.4 baseline Jacobian isolation cases across both families,
  method/function and batching patterns. Do not invent validity-repair
  transformations for arbitrary custom families.
- Include second-order check_grads/Hessians on valid interior mixed likelihoods
  with analytical exponential derivatives and justified central tolerances in
  tests/impl_tolerances.py; a valid mixed eager jax_debug_nans run. A7/A9 retain
  broader tail/transformation verification; this step must verify new code.
- Run the complete through-A6 gate with the expanded own tests. Later A7–A9,
  fixtures, install and CPU benchmark evidence remain required for Slice A
  completion.
- Docs to-do: transformation and diagnostic composition examples.

## Validation and delivery for every approved code step

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run focused own and available
independent modules in separate JAX_ENABLE_X64=1 and 0 processes, Ruff format
and lint, mypy src, scripts/run_tests.sh and git diff --check. Run
`scripts/check.sh --through A5` while A6 is partial; after A6.3 run the focused
independent constraint/diagnostic modules with SPEC_THROUGH=A6. At completion
run `scripts/check.sh --through A6` without changing/skipping future tests.
Record exact counts, skips, failures and commands in REVIEW.md for each step.
Do not add NumPyro; its optional independent interoperability check may skip
when it is absent. No new runtime/test dependencies or later-slice features.

Narrative docs follow reviewed code in separate topic-sized docs steps
(at most 300 changed lines each): parameter declarations/conditions and host
checks first, likelihood usage/subset second. Edit docs/api.md and the relevant
existing guide or new focused guide; review need for playground updates.
Run scripts/build_docs.sh and inspect both dtype renders for each docs step.
Public API docstrings accompany code rather than waiting for docs.

Planning checks: read-only git/spec/test/source inspection only; no pytest,
lint, typing, runtime gate or docs build run. The marker records that the spec
was read, not implementation conformance. Mark using the existing hyphenated
`../scripts/spec-since.sh dev --mark` (the requested underscore path does not
exist). Leave this planning change uncommitted on main for maintainer review.

---

# A5.6 — Base boundary policy, then A5.d2 docs refresh

Status: code and review fix-up are merged by the maintainer. Step 2 is
implemented on the maintainer-created `docs-A5.6` branch on 2026-10-19,
based on `282dcca`, and ready for review. The maintainer explicitly requested
direct execution without a further planning/approval stop and asked to keep
the existing writing style. Only `docs/distributions.md` and `docs/api.md`
changed (188 documentation lines); the tutorial structure is retained.
Both dtype docs builds pass; exact checks and output inspection are in
REVIEW.md. No source/docstring, test, spec or dependency changes were needed.
Each step ends with its own commit, REVIEW.md and a stop.

Contract: specification revision 27, `.spec-seen` at `b92334a`. The newly synced
independent suite records author commit `1d87b726a15072f6e7223a60fe1afefa5bb999be`
(dev sync `645479e`, merged in `9146251`). The only later design/spec change,
`9d7483c`, adds PDF conversion helpers; it changes no normative requirement.
Leave the marker and spec sources unchanged during implementation.

## A5.6 review fix-up — F-1/F-2 and nits (2026-10-19)

Authorized by the maintainer's request to address `../review/slices/A5.6.md`,
including its explicit F-2 decision. Same step and `A5.6` branch; one fix-up
commit, then stop. No new public API, spec edit or narrative-doc work.

- F-1: add an own `log_cdf(+inf)` regression for Cure at fraction=1e-12,
  expecting analytical `log1p(-fraction)` with the existing zero-atol tail
  tolerances, in eager/jit/vmap and both dtype modes. Demonstrate that reviewer
  mutant M4 (naive log(1-exp)) fails this test in both modes, then restore.
- F-2: restrict hook-NaN precedence in `_survival_methods.py` to the four
  base-boundary methods and log density. Add a NaN-hook family with an
  analytic log-hazard override: log hazard/hazard keep log(rate)/rate at
  infinity; log survival/log density remain NaN. Run both modes and
  eager/jit/vmap. Keep every method's own NaN result intact.
- N-1: simplify Cure primitives in both own infinity modules and remove
  NaNLimit's dead primitive infinity branch; their hooks declare the limits.
- N-2: clarify the hook docstring's role and explicitly state that its default
  does not raise for parameter values; do not invent a raised exception.
- Files: these two own infinity test modules, `_survival_methods.py`,
  `base.py` (docstring only), PLAN.md and REVIEW.md. Cover spec 4.2/4.5/7.2/12.1.
  Correct the superseded every-method hook-NaN assertion below and in REVIEW.md.
  No independent-test or deviation-status changes.
- Checks: focused own infinity modules in both dtype processes, M4 regression
  evidence, `scripts/check.sh --through A5`, and `git diff --check`. Record
  exact outputs in REVIEW.md and stop before the narrative-docs step.

## Step 1 — Code, tests and public API docstrings (A5.6)

Outcome: every registered family receives the section 4.5 base boundary values,
even when its primitive has NaN values/gradients at zero or infinity. Improper
families declare their limit through a differentiable hook. Preserve interior
formulas, existing log-CDF tail precision, dtype/broadcasting rules, inherited
overrides, time-first grids and finite-entry derivatives in mixed batches.

Public addition: `SurvivalDistribution.log_survival_at_infinity(self) ->
jax.Array`, inherited by registered families and the built-ins. Its default is
`-inf`, exactly `batch_shape`, in the parameter dtype; component axes are
excluded. Custom overrides return log cure fractions with active parameter
derivatives. No new top-level export, constructor option or stored field.
Covered spec: 2, 3.3, 4.1–4.2, 4.5–4.6, 5, 7.1–7.3, 12.1–12.2.

Files and approach:

- `src/survjax/distributions/base.py`: add the typed default hook with summary,
  arguments, return-shape/dtype and error documentation. Derive its output dtype
  from normalized parameters, including nested-only families; do not use the
  transient time dtype. Keep it an ordinary overridable method, separate from
  the seven policy-wrapped pointwise methods. Pass the direct method's explicit
  name to the existing wrapper so base-owned boundaries also cover overrides.
- `src/survjax/distributions/_survival_methods.py`: replace the raw-primitive
  infinity probe/zero-limit mechanism with explicit boundary masks. For log S,
  S, log F and H, substitute at zero and infinity before calling any raw family
  formula, including inherited overrides. At zero select 0, 1, -inf, 0. At
  infinity derive values from the hook using exp, the existing safe log1mexp,
  or negation. Base constants must retain zero parameter gradients; hook values
  must retain their genuine derivatives. Substitute safe operands before
  boundary transformations as needed, never mask only their final results.
- For log density, substitute before evaluating at infinity and select -inf.
  For generic log hazard and generic hazard based on it, substitute at infinity
  and select NaN even for a nonzero cure limit. Analytic log-hazard overrides
  still receive zero/infinity, with matching hazard behavior; retain existing
  explicit hazard dispatch. At zero the generic density/log-hazard defaults
  must differentiate the raw primitive, preserving the specified exception;
  they acquire no value or parameter-gradient guarantee there.
- Use one stopped finite positive reference per direct call for valid
  parameters. Propagate NaNs from the family's substituted-input evaluation
  and, on base-supplied infinity values, from the hook, as section 4.5 requires.
  The maintainer's 2026-10-19 F-2 decision excludes analytic hazard overrides
  from hook-NaN precedence. Eliminate probes of the primitive at actual infinity. Preserve finite-support generic hazard behavior
  at finite times, negative/NaN-time policy and inherited raw-method unwrapping.
  No floors, clipping, inferred parameter validity or stopped active values.
- Update the base class, `log_survival`, `reference_time` and affected generic
  method docstrings in those same files. Audit `exponential.py`, `weibull.py`
  and `_grid_methods.py` docstrings; correct only obsolete ownership/gradient
  claims. Built-in formulas remain unless a demonstrated conformance failure
  requires a scoped correction. No `docs/` edits in this step.

Own tests, kept alongside the code in the same commit:

- Add `tests/unit/test_family_boundaries.py` with a small naive log-logistic
  primitive (log S=-softplus(beta*(log t-log alpha))) and an interior-only
  primitive (log S=-r*t*(t/t)). Pin base values at zero/infinity, gradients at
  zero, proper survival/log-CDF gradients at infinity, and mixed finite-entry
  derivatives. Cover eager/jit/vmap, shared parameters and time-first grids;
  use valid mixed batches under `jax_debug_nans` for methods with defined
  non-NaN outputs. Do not demand debug-nans success for deliberately undefined
  generic hazard outputs or invalid inputs.
- Extend the two existing infinity modules rather than duplicate their cure
  coverage: give Cure the explicit log-fraction hook. Check log S=log p,
  S=p, log F=log(1-p), H=-log p and their first/second parameter derivatives;
  rate derivatives at infinity are zero. Reuse the mixed-batch Hessian tests.
  Add hook-default scalar/batch/component/nested shape and dtype cases, plus
  finite overridden formulas whose boundary values must come from the base.
- Migrate obsolete own assertions in `test_family_infinity.py`,
  `test_family_mixed_infinity.py` and `test_family_direct.py`: survival/log-CDF/
  cumulative-hazard overrides no longer supply their own boundary values;
  raw primitive NaN at infinity no longer determines density NaN status;
  NaN precedence comes from substituted values/the hook. Move NaNLimit's
  declared NaN limit into its hook. Preserve positive-time override dispatch
  and analytical density/hazard boundary checks. Replace the public
  log-survival time-derivative-at-zero assertion with coverage of the generic
  density's raw-primitive differentiation; direct-time boundary derivatives
  are not part of section 7.3's guarantee.
- Expected values come from the algebra above or analytical identities stated
  in each test. Reuse `tests/impl_tolerances.py`; zero absolute tolerance for
  tiny log/derivative comparisons. New justified tolerances belong only there.
  Keep one behavior per test and do not edit, skip or xfail independent tests.

Independent evidence to require:

| Requirement | Independent coverage |
| --- | --- |
| Hook shape/dtype, proper and improper boundaries, NaN precedence, unsafe primitives and boundary gradients | `test_a4_base_boundaries.py` |
| Negative/NaN rows, raw defaults, finite zero survival, overrides, time dtypes | A4 policy, defaults, hazard, NaN-precedence and time-dtype modules |
| Built-in limits and gradients, unchanged closed forms | `test_a5_boundaries.py`, `test_a5_closed_forms.py`, `test_a5_derivatives.py` |
| Generic and built-in two-tail precision and derivatives (revision 25) | `test_a4_log_cdf_upper_tail.py`, `test_a5_log_cdf_tails.py` |
| Time-first boundary values, shapes and transformations | `test_a5_survival_grids.py` |

Validation: use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run new/changed
own tests and the focused boundary acceptance module in separate x64/x32
processes, then `scripts/check.sh --through A5` once after focused checks pass.
That gate includes Ruff, mypy, all own tests, source doctests and the complete
through-A5 independent suite in both modes. Finish with `git diff --check`.
Do not build narrative docs until step 2; their old boundary recipes are
intentionally refreshed there. Record exact failures/skips and any scope issue.

Aim for about 300 changed code/test/docstring lines; reuse current coverage and
remove superseded policy logic. If the cohesive change exceeds the working
agreement's size cap, stop and propose a code split before expanding it,
keeping all code steps before narrative docs. No new dependency, helper module,
registration protocol, custom derivative rule or later-slice feature is planned.
Overwrite REVIEW.md with reading order, safety/dispatch/gradient invariants,
spec-to-test mapping and exact checks. Commit on the maintainer's step branch,
then stop for review. Leave deviation statuses to the maintainer.

Docs to-do for step 2: replace primitive-owned boundary guidance with the new
base contract and hook; keep custom density/hazard responsibilities explicit.

## Step 2 — Narrative docs after code review (A5.d2)

Status: completed on `docs-A5.6` (2026-10-19), following the maintainer's
direct implementation request. Built-in/grids guides and the playground were
audited and needed no changes. The following scope describes the completed
boundary-contract documentation update.

One topic: explaining and using the revised family boundary contract. Public API
and runtime behavior stay as approved in step 1. Cover spec 2, 4.2, 4.5, 5,
7.2–7.3, 10.5 and the boundary evidence in 12.1–12.2.

- `docs/distributions.md`: update the edges table and custom-family checklist;
  describe finite-positive primitive inputs and base-supplied limits. Simplify
  the log-logistic example so naive log survival now works for survival/CDF/H
  boundaries. Retain the separate guarded analytic density/hazard extension
  example and its NaN handling. Replace the cure primitive's infinity branch
  with `log_survival_at_infinity() = log(fraction)`. Execute boundary values,
  parameter-gradient checks and mixed valid/NaN entries; explain the generic
  density/hazard exception at zero and generic NaN hazard at infinity.
- `docs/api.md`: list/document the optional hook and distinguish required
  primitive/reference methods from this override. The generated method help
  comes from step 1's docstrings.
- Audit `docs/builtin-families.md`, `docs/grids.md` and
  `docs/playground.ipynb`; edit only statements/examples that become wrong.
  Their built-in results and grid shapes should stay the same. Notebook source
  inspection during planning found no boundary/hook/primitive recipe to
  migrate, so no notebook edit is planned without a concrete need.

Keep this one topic within 300 changed lines; propose a split before exceeding
that cap. No unrelated tutorial/theme work or new notebook feature. Source
edits are limited to docstrings found to be wrong; a numerical defect goes
back to a separately reviewed code fix.

Run `scripts/build_docs.sh`: it executes and builds all docs afresh, with
warnings as errors, in separate x32/x64 processes. Inspect rendered guide/API
sections and their outputs in both modes. Run `git diff --check`; rerun a
runtime gate only if a source change or discovered failure warrants it.
Overwrite REVIEW.md with reading order, executed examples/spec mapping, exact
build results and remaining uncertainty. Commit on the maintainer's docs
branch, then stop. This step does not claim Slice A's remaining gates complete.

### Tutorial nuance follow-up (2026-10-19)

The maintainer directly requests clarifying that hazard/density endpoint
limits are optional, preserving their concurrent edit. On `docs-A5.6`, add
this explanation at "The hazard and density need more" and make nearby
instructions/checklist entries conditional on actually needing endpoint
values. Preserve the maintainer's "In many medical settings..." sentence in
the piecewise-hazard introduction. No executable examples or library code
change. Run `scripts/build_docs.sh` in both modes, inspect generated text,
update REVIEW.md, commit this small docs follow-up, then stop.

## Planning checks actually run

No source, test, spec or narrative-doc changes were made during planning.
Ran the following in separate processes with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`,
`SPEC_THROUGH=A5` and `JAX_ENABLE_X64=1` / `0`:

`uv run --locked --group test pytest tests/spec/test_a4_base_boundaries.py -q --tb=short`

- x64: exit 1, **32 failed, 52 passed in 2.11s**.
- x32: exit 1, **29 failed, 55 passed in 2.19s**.

Failures include the absent default hook, raw boundary NaNs, boundary gradients,
and improper-family generic hazard behavior. These are pre-implementation
results, not regressions caused by this plan. The full gate, tail modules and
docs builds were not run during planning. Historical records follow below.

# Spec alignment — revisions 25–27 (2026-10-18)

The maintainer requested reading `../scripts/spec-since.sh dev`, recording the
changes, then running it with `--mark`. This bookkeeping step changes only
`PLAN.md`, `DEVIATIONS.md`, `REVIEW.md` and `.spec-seen`. No public API or
runtime changes, no new tests. It supersedes the historical revision-24
alignment statement below; marking records review of the contract, not proof
that the implementation satisfies revision 27.

Reviewed committed range: `b917ebe` (revision 24) to
`b92334a1a4b584a7970039e321a6b10fe7c176ff` (revision 27).

- **Revision 25, `1e9011c` — sections 4.1–4.3, 7.2, 12.1:** specify the
  private two-branch `log1mexp` helper, with safe operands in both branches.
  Generic `log_cdf` and built-in standard branches use it; Exponential and
  Weibull retain the lower-tail calculation from `log H`. Both tails must
  keep relative precision over the specified representable ranges. Acceptance
  checks include mpmath upper tails, non-unit Weibull parameters, generic and
  grid paths, first/second derivatives around the branch split and in tails,
  and mixed boundary/interior calls under `jax_debug_nans`. Tiny log values
  and derivatives use zero absolute tolerance. The spec records the decisions
  behind D-021/D-022; formal deviation closure stays with the maintainer.
- **Revision 26, `0008c82` — sections 2, 4.2–4.5, 7.1, 8.2, 12:** the base
  supplies log survival, survival, log CDF and cumulative hazard at zero and
  infinity. Add the optional `log_survival_at_infinity()` extension method:
  default `-inf`, exactly `batch_shape`, parameter dtype; improper-family
  overrides retain parameter derivatives. These methods' family formulas and
  overrides need only handle finite positive times. Density/hazard limits
  remain family-supplied; generic autodiff at zero has no value/gradient
  guarantee, and generic log hazard at infinity is NaN. Preserve NaN
  precedence from substituted evaluations and the infinity hook. Base boundary
  gradients now cover every registered family, including naive primitives
  with unsafe boundary derivatives. Tests must cover proper and improper
  custom families. Section 2 also clarifies that sets closed at infinity
  include mass at infinity; future evaluator/solver work uses the declared
  limit. HazardFamily, solver and `check_family` changes stay in later slices.
- **Revision 27, `b92334a` — sections 4.5, 12.2:** valid-parameter substituted
  inputs must be finite and positive. Slice A must honor infinity-hook
  overrides in the base, including improper-family values and gradients;
  supplying only a hard-coded proper-family default is insufficient.

Implementation follow-up requires a separate approved plan on a
maintainer-created step branch. `rg -n 'log_survival_at_infinity' src
tests/unit docs` found no matches (exit 1), so the new hook is not present in
those files. No conformance gate was run in this bookkeeping step.

Docs to-do: refresh the extension recipe, boundary table, API reference and
playground for base-supplied limits and the infinity hook in a separate docs
step after implementation.

The script warns about uncommitted spec changes because design has an
untracked `spec/SPECIFICATION.pdf`; the tracked specification matches HEAD.
Do not incorporate or commit that generated file. The dev checkout is on
`main`; AGENTS.md prohibits commits there and reserves branch creation for
the maintainer. Leave this bookkeeping ready for the step-branch commit.

# A5 — Exponential, Weibull and time-first grids

Status: A5.1 (`3f50962`) was approved by the reviewer and merged into main by
the maintainer. A5.2 (`a58c57c`) received a merge verdict on 2026-10-12.
The F-2/F-3 review nits are corrected; A5.2b (`71403c3`) and its review fix-up
(`63874e8`) are merged into main. A5.3 (`cc2f85d`) and its broadcasting fix-up
(`cd3f213`) are present on main. A5.4 and both log-CDF review fix-ups
(`c477d31`, `4f867cb`) are merged into main. The maintainer selected branch
`A5.5`; the approved implementation from `861ac71` passes the full through-A5
gate in both dtype modes on 2026-10-15 and awaits review.
The spec and `.spec-seen` agree at revision 24 (`b917ebe`). The synchronized
tests correct D-020's density expectation. The maintainer's D-022 upper-tail
note is committed in `861ac71`; its numerical fix is implemented below as a
separate step on the same `A5.5` branch and awaits review.
A5 docs follow the remaining code work.
The original planning started from a
clean `A5` checkout at `3f50962`, then the main tip. No branch was created
or switched by this planning request; implementation uses the maintainer's
selected step branch after approval. Based on
specification revision 22, design commit `9323cb5`, and the independent tester
commit `7f1d6df`. Implementation starts on the maintainer-selected branch `A5`
at `89cb9f4`, after the maintainer aligned the marker and synchronized tests.
Commit this approved planning/deviation record separately from the A5.1
implementation so each diff remains reviewable. The A4 plans below are history.

## What changed after the spec/test review

Read the complete revision-21-to-22 diff, the six new A5 test modules,
Q-027–Q-033 and their TESTMAP entries, plus the three new A4 NaN-precedence
cases. Revision 22 replaces the earlier log-leaf proposal: `from_log` simply
exponentiates inputs and stores the resulting natural parameters. The previous
requirement to retain extreme log inputs is removed. No log storage, private
representation flag or registration redesign is needed (D-019).

Other settled requirements:

- Built-ins are registered frozen subclasses of `SurvivalDistribution`, with
  natural `rate` or `shape`/`scale` as their only dynamic leaves (4.1–4.2).
- `reference_time()` returns an array of exactly `batch_shape`, in the parameter
  dtype, positive and finite for valid parameters; invalid cases are unspecified.
- `log_prob(+inf)` is NaN when the family's raw `log_survival(+inf)` is NaN;
  otherwise it is -inf, including when a density formula yields inf-inf.
  Testing the raw density for NaN alone cannot distinguish these cases (4.5).
- At zero and infinity, finite built-in values have the actual parameter
  derivatives of their limits. Derivatives across a discontinuity are excluded:
  do not require the Weibull shape derivative of log density at zero and k=1.
  The scale derivative there is required. Custom-family guarantees depend on
  the author's primitive; the extension recipe must explain this (D-016/4.5).
- Revision 24 expands the base surface to seven methods: `survival_grid(t)`,
  `log_survival_grid(t)`, `log_prob_grid(t)`, `log_cdf_grid(t)`,
  `cumulative_hazard_grid(t)`, `log_hazard_grid(t)` and `hazard_grid(t)`.
  Every registered family inherits them.
  The first positional argument is named `t` and also accepts `t=...`; it must
  be one-dimensional and follows the direct-time dtype rules (5/8.1).
- The zero-time grid example is self-contained: Weibull shape 1, scale 10.
  The section 11.1 fixed vector additionally gives group scales
  `10 * exp([0, -0.5, -1])`; it requires no R fixture.

These decisions supersede the old pending choices; no new spec ambiguity was
identified in this review. Record revision-22 evidence in DEVIATIONS.md and
leave formal deviation closure to the maintainer.

## Scope, staging and independent-test availability

Public additions: `survjax.Exponential(rate=...)`,
`survjax.Weibull(shape=..., scale=...)`, keyword-only `from_log` classmethods,
all seven direct methods on both families, and their seven grid base methods.
Covered spec: 3.3, 4.1–4.2, 4.5–4.6, 5, 7.2–7.3, 8.1 (grids only),
10.2–10.5 and applicable family/boundary requirements of 12.1–12.2.
No likelihoods, interval/truncation methods, parameter diagnostics (D-013),
summaries, sampling, wrappers, `check_family` or new dependencies. R fixtures,
installation checks and the CPU likelihood baseline remain later Slice A work.

There are five commit-sized steps below, separating the newly specified policy
correction from the generic gradient work. Aim for roughly 300 changed
source/test/docstring lines per code step, at most 300; split before expanding
if necessary. Each step requires its own plan approval, tests and public API
docstrings with the code, one commit and an
updated REVIEW.md, then a stop. This proposal does not authorize proceeding
through all five steps. Use the maintainer-selected branch; never commit on main.

From A5.2b onward, narrative docs follow specification revision 23 (decision
0038): code steps don't edit `docs/`; record their documentation needs under
"A5 docs" below. Docs steps start after the last A5 code step merges, and A6
starts only after the last A5 docs step merges. Earlier A5 docs are drafts;
the A5.1/A5.2 instructions and verification evidence below retain their history.

The maintainer aligned `.spec-seen` to `9323cb5` in `96fc32d` and synchronized
the independent suite from `7f1d6df` in `89cb9f4` before implementation. The
current through-A4 gate includes the three revision-22 regressions. The
revision-24 suite now collects 438 A5 cases, including 116 in the grid module.
Both independent suites and the alignment marker remain untouched by the
implementation agent. Never edit, skip or xfail acceptance tests to obtain a pass.

## A5.1 — Correct NaN precedence for density at infinity

- Change `src/survjax/distributions/_survival_methods.py`'s density policy to
  consult the raw primitive at +inf. Return NaN iff that primitive value is NaN;
  otherwise preserve the unconditional -inf density limit. Leave negative/NaN
  time handling, normal broadcasting, override dispatch and non-density methods
  unchanged. No exports, signature changes or generic validity-domain inference.
- Add `tests/unit/test_family_infinity.py`: constant-hazard and power custom
  families with NaN parameters; mixed NaN/valid entries where the valid generic
  density formula is NaN at infinity; an explicit density override whose result
  differs from the primitive's NaN status. Exercise eager/jit/vmap. Expected
  constants follow the revision-22 policy, not the implementation.
- Preserve the existing one-reference-call invariant. Keep any extra primitive
  evaluation restricted to the density rule; do not double-wrap methods.
- Update the infinity rule in `docs/distributions.md` and relevant base-method
  docstrings. Cover 4.2/4.5/7.2. Run the three new acceptance tests in
  `test_a4_nan_precedence.py`, then the complete through-A4 regression gate
  once the suite is synced. Stop with REVIEW.md and the step commit.

## A5.2 — Generic finite boundary gradients and extension recipe (D-016)

Outcome: generic survival and log-CDF give their correct finite values and
parameter gradients at +inf for proper primitives such as `log S=-r*t`, while
preserving finite, parameter-dependent limits of custom cure families. Public
API, exports, constructors, dtypes and broadcasting are unchanged. Covered spec:
4.2, 4.5, 5, 7.2–7.3, 10.5 and relevant transformations in 12.1. No built-ins,
grids, density/hazard changes, dependency additions or custom derivative rules.

### Numerical change — `src/survjax/distributions/_survival_methods.py`

- Integrate the guard in the existing policy wrapper, where its already-computed
  reference input can be reused. This refines the earlier defaults-local sketch:
  calling `reference_time()` again inside each default would break the existing
  one-reference-call invariant. Keep the change in this module; registration
  and `base.py` need no change.
- After resolving the existing raw-method marker, recognize only the generic
  `survival` and `log_cdf` functions using function identity, a static Python
  decision. Explicit and inherited family overrides retain their dispatch.
  No new marker protocol, cache, parameter specialization or wrapper stack.
- Retain the single stopped reference input in a named local. Evaluate raw
  `log_survival` on the current safe times to obtain the true limit. The guard
  is elementwise: original time is positive infinity AND raw log survival is
  negative infinity. Use this result only as a Boolean selection condition.
- Before the default's numerical operations, replace those guarded times with
  the stopped reference input. Evaluate the default on these safe times, then
  select the exact finite boundary value 0 for survival or log-CDF. The raw
  primitive has still been called at the actual infinity to establish its limit.
  Stop only the substituted reference, never real parameters or active results.
- Leave finite raw limits at +inf active: if S(+inf)=p, gradients in p must
  survive. NaN raw limits must remain NaN. The guard must not activate at finite
  times, even when linear survival underflows or log survival overflows to -inf;
  no precision loss is reclassified as zero mass. Preserve zero-time behavior,
  A5.1's density rule and negative/NaN-time policy.
- A second primitive evaluation is expected for these generic methods, including
  finite calls. Record that cost without claiming a measured speedup or adding
  an optimization layer. Custom authors remain responsible for primitive
  differentiability and safe behavior at their chosen reference inputs.

### Own tests — `tests/unit/test_family_infinity.py`

Extend the existing 111-line module, keeping it below roughly 300 lines.
Expected values/derivatives come from algebra written in each test; run both
dtype modes with `tests/impl_tolerances.py`'s existing appropriate tolerances.
Add a scale only with a separate numerical justification in that file.

- Existing Constant and Quadratic primitives: survival(+inf)=0 and
  log-CDF(+inf)=0; first parameter derivatives are exactly 0 in forward and
  reverse modes, eagerly and composed with jit/vmap.
- Scalar rate with times [1,+inf]: sum survival=e**(-r), derivative=-e**(-r)
  and second derivative=e**(-r). Sum log-CDF=log(1-e**(-r)), derivative
  1/expm1(r) and second derivative=-e**r/expm1(r)**2. Cover per-entry parameter
  batches as well, so an inactive infinity entry cannot poison active entries.
- A guarded cure primitive S(t)=p+(1-p)*e**(-r*t), with 0<p<1 and r>0:
  at infinity survival=p and log-CDF=log(1-p). Their p derivatives are 1 and
  -1/(1-p), and their r derivatives are 0. Preserve these under jit/vmap and
  check a finite-time case; the primitive itself guards its infinity operands.
- NaN parameter entries beside valid entries at +inf retain NaNs only in their
  positions; finite-time underflow and active lower-tail values retain current
  behavior. Do not assert derivative guarantees for invalid parameters.
- Explicit/inherited survival and log-CDF overrides remain honored. Extend
  reference-call coverage to these methods at infinity, checking one call.
  Existing finite/negative-time, zero-CDF, primitive time-derivative, density
  and second-order interior tests remain regressions, not targets for alteration.
- Demonstrate before/after evidence with the new infinity-gradient cases.
  After the fix, temporarily remove only the safe-time substitution: the
  constant-family gradient cases must detect NaN despite final output masking.
  Restore the source before final checks; never mutate independent tests.

### Documentation — `docs/distributions.md`

Preserve the maintainer's tutorial. Add a short executable guarded-boundary
extension recipe and parameter-gradient assertions, explain the proper-family
zero limit and the parameter-dependent cure limit, and state the author's
responsibility for safe primitive operands/NaN propagation. Update method
docstrings with the same scope. Amend PiecewiseHazard's `reference_time()` to
return exactly `batch_shape` in `rates.dtype`, and revise the checklist's
current "broadcastable" promise to the revision-22 exact-shape contract.
No new page, notebook or later-slice feature is needed.

The separately requested Gompertz notation correction is applied during this
planning turn: display `-(b/c)*(e**(c*t)-1)` mathematically, with `jnp.expm1`
still explained in prose and used by executable code. It changes no computation.
Planning verification: `UV_CACHE_DIR=/private/tmp/survjax-uv-cache
scripts/build_docs.sh` exited 0, reporting `build succeeded` for x32 and x64
and executing the existing examples afresh. `git diff --check` passed with no
output. No numerical implementation tests were run during this planning turn.

### Validation, commit and stop

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run the focused own suite in
separate x64/x32 processes, then `scripts/check.sh --through A4`,
`scripts/build_docs.sh` (both modes), and `git diff --check`. The independent
author does not guarantee arbitrary custom-family boundary gradients; the new
own tests supply that evidence. All 370 A5 cases remain outside this step's
gate, since built-ins/grids are not yet implemented. Do not edit or skip them.

Aim for roughly 300 changed source/test/doc lines; split before expanding if
necessary. Overwrite REVIEW.md with reading order, safe-input/dispatch invariants,
spec-to-test mapping, mutation evidence, exact results/skips and the unmeasured
extra-evaluation cost. Leave D-016 open for maintainer closure. Commit the
approved step on the maintainer-selected branch, then stop before A5.3.

## A5.2b — Isolate finite-entry derivatives from infinity entries (review F-1)

### Review fix-up (2026-10-13)

Authorized by the maintainer's request to implement the A5.2b review comments.
Same step and branch; no new public API or numerical policy.

- `tests/unit/test_family_mixed_infinity.py`: add a density override to the
  existing cure family whose formula is -r*t. Pin [1,+inf] values [-2,-inf]
  at r=2, p=0.25, and the finite-component rate derivative -1 in forward and
  reverse modes, including jit/vmap. Spec 4.5/7.2; expected numbers are algebra.
- `_survival_methods.py`: consolidate each generic method's guard flag and
  boundary value in one static conditional chain; retain the hazard check
  against its underlying log-hazard and rename that local `log_hazard_method`.
  No registration-table redesign or new configuration layer is needed.
- F-3 remains a structural argument: `isposinf(time)` explicitly excludes
  finite times. Do not assert unsupported NaN-gradient behavior to test it.
- Demonstrate that reviewer mutant M1 (density substitutes only zero_limit)
  fails the new test in both dtype modes; restore before the full through-A4
  gate. Run focused tests, Ruff/mypy through that gate and `git diff --check`.
  Overwrite REVIEW.md, commit one review fix-up, then stop. Full docs stay
  deferred to the end of A5; spec and independent tests remain read-only.

Maintainer update (2026-10-13): code steps include tests and public API
docstrings, with no `docs/` edits. Narrative docs and their builds are separate
A5.d1, A5.d2, … steps after the last A5 code step merges; see "A5 docs" below.

Status: implemented, reviewed and merged with its fix-up on 2026-10-13.
This is a separate numerical
step before A5.3, on the maintainer-selected branch. It addresses the review's
shared-parameter mixed-batch leak; it does not promise derivatives of infinite
or NaN outputs. Covered spec: 4.2, 4.5 and 7.2–7.3; the extension recipe in
10.5 is updated and verified at the end of A5.
No public API, exports, dependencies, built-ins, grids or spec-test changes.

Files and numerical approach:

- `src/survjax/distributions/_survival_methods.py`: retain the single stopped
  reference and probe the primitive at the original policy-safe times before
  any infinity substitution. Use probe results only as Boolean conditions.
  For density, substitute all positive-infinity times before running the raw
  density, including overrides. Select -inf or NaN using the original probe's
  NaN status; a substituted reference must not determine that status.
- For the policy-wrapped log-survival primitive and the generic cumulative
  hazard, extend the existing zero-limit guard (positive infinite time AND
  negative infinite raw log survival); substitute before evaluation and select
  -inf/+inf respectively. Finite cure limits stay active. Cumulative-hazard
  overrides retain their own dispatch.
- Guard generic log-hazard with the same zero-limit condition and select its
  existing NaN boundary value. Guard generic hazard only when its underlying
  log-hazard is also generic. Analytic log-hazard/hazard overrides, including
  inherited overrides, must preserve their infinity values and derivatives.
  This prevents the generic hazard default from reintroducing the same leak.
- `src/survjax/distributions/base.py`: if needed, pass an explicit private
  primitive flag from registration to the existing policy wrapper. The
  log-survival primitive has no generic function identity; do not infer its
  role from an author's function name or introduce a new marker protocol.
  Other generic defaults continue to use static raw-function identity.
- Preserve NaN precedence, negative/NaN and zero-time behavior, finite tails,
  and reference-call count. Stop only substituted inputs, never active values
  or parameters. Reuse the probe for density instead of probing at the newly
  substituted time. Record extra evaluation cost without optimizing it here.

Tests: create `tests/unit/test_family_mixed_infinity.py` to keep the existing
infinity module below 300 lines. Use small explicit constant/quadratic/cure
families and hand-derived expectations, with existing test tolerances.

- Shared scalar rate r=2 at times [1,+inf]: finite-entry derivatives are -1
  for log survival, +1 for cumulative hazard, 1/r-1 for log density, 1/r for
  log hazard and 1 (constant) or 2 (quadratic) for hazard. Check the finite
  output component with jacfwd/jacrev, eager/jit and jit/vmap compositions;
  do not differentiate a sum containing a nonfinite output.
- Check second derivatives of the finite output component against algebra:
  log density/log hazard -1/r**2, other listed methods zero. Independent
  parameter batches must keep finite-entry derivatives separate too.
- Preserve A5.1 density values for mixed NaN/valid parameters, including
  finite and NaN overrides. Preserve nonzero fraction derivatives of finite
  log-survival/cumulative-hazard cure limits and analytic hazard overrides.
- Pin one reference call and unchanged finite values. Demonstrate pre-fix
  failures in both dtype modes, then remove only the new input substitutions
  as a mutation: finite-component reverse derivatives must fail again.

Update function docstrings with mixed-entry isolation and its limits, including
that derivatives of nonfinite outputs and arbitrary unsafe custom overrides
are not guaranteed. Do not update full docs or run `scripts/build_docs.sh`
during A5.2b. Record this behavior under "A5 docs" for the docs steps to cover
and verify with the full documentation build in both dtype modes.
`DEVIATIONS.md`: extend D-016 evidence without resolving it.
Overwrite `REVIEW.md` with invariants, test/spec mapping and exact check results.
Run focused tests in both processes, `scripts/check.sh --through A4` and
`git diff --check`. Record the deferred documentation work in REVIEW.md.
Aim for about 300 source/test/docstring lines, split if needed, commit one
numerical step, then stop before A5.3.

## A5.3 — Exponential

### Reviewer broadcasting fix-up

Authorized by the maintainer's request to implement the reviewer suggestion.
On the existing `a5.3-exponential` branch, update `log_hazard` and `hazard`
in `src/survjax/distributions/exponential.py` to compute the broadcast shape
of `t` and `self.batch_shape` explicitly before broadcasting their values.
No public API or time-policy change; this removes the raw methods' dependency
on the registration wrapper pre-broadcasting time. Covers spec 4.1/4.5/5.

Add an own regression in `tests/unit/test_exponential.py` for raw and wrapped
hazard methods with scalar time and singleton-axis time arrays against batched
rates, eagerly and with jit/vmap. Expected values are rate and log(rate).
Demonstrate the new regression fails before the source change. Run the focused
suite in both dtype modes, the through-A4 gate, the existing Exponential-only
A5 acceptance selection, and `git diff --check`. Overwrite REVIEW.md, commit
one fix-up and stop. Full docs remain deferred to the end of A5.

Status: approved and implemented on 2026-10-13 from main at `7a2fef6`, on
`a5.3-exponential`; awaiting review. This is the code step after merged A5.2b,
before Weibull (A5.4) and grids (A5.5). Approved planning and spec-marker
alignment were committed separately in `9337969` before implementation.

Spec: 3.3, 4.1–4.2, 4.5–4.6, 5 (pointwise broadcasting only), 7.2–7.3,
10.2–10.5 and the applicable 12.1 checks. Revision 23 changes documentation
scheduling only; the synced revision-22 numerical acceptance tests remain
applicable. `.spec-seen` was aligned from `9323cb5` to `a212ddd` after reading
the diff, in the separate bookkeeping commit. Spec tests and their revision
labels remain unchanged.

### Files and public API

- Create `src/survjax/distributions/exponential.py`: registered frozen dataclass
  subclass of `SurvivalDistribution`, with `rate: jax.Array = param()` as its
  only leaf. Constructor is keyword-only `Exponential(rate=...)`; no default,
  dtype keyword, value validation, log-storage flag or new dependency.
- Add `Exponential` imports/exports in `src/survjax/distributions/__init__.py`
  and `src/survjax/__init__.py`; update the exact public export assertion in
  `tests/unit/test_import.py`.
- Add typed `from_log(*, log_rate)` returning the same class. Reuse
  `_dtypes.normalize_parameters` before exponentiation so boolean/complex/
  string/object inputs cannot become valid parameters through `exp`. Keep
  weak/default dtype promotion and the float64-to-float32 warning policy.
  Store only `exp(log_rate)`; overflow yields an infinite natural rate.
- Implement `log_survival=-r*t`, `cumulative_hazard=r*t`,
  `log_prob=log(r)-r*t`, `log_hazard=log(r)` and `hazard=r` analytically.
  Inherit `survival` and stable `log_cdf` from the existing base. Preserve
  pointwise output broadcasting even for time-independent hazard methods.
  Use the existing policy wrappers; no registration or generic-wrapper changes.
- `reference_time()` returns exactly `batch_shape` in `rate.dtype`, positive
  and finite for valid rates. Use `1/r` where representable; substitute safe
  operands before division and choose a finite positive reference if reciprocal
  overflow or underflow would make it unusable. Invalid-rate references have
  no value guarantee. This fallback never changes active rates or probabilities.

### Numerical invariants and docstrings

- Negative/NaN times use the base policy. At zero: log S=H=0, S=1,
  log F=-inf, log density/log hazard=log(r), hazard=r. At +inf: log S=-inf,
  H=+inf, S=0, log F=0, log density=-inf, log hazard=log(r), hazard=r.
- Substitute safe times before inactive infinity arithmetic in analytic
  overrides where needed. Retain NaN-rate propagation at every time, including
  negative/zero/infinite times; don't hide it behind boundary constants.
  Invalid finite rates are accepted and follow the formula; add no domain mask.
- At finite-valued boundaries, derivatives are the actual limit derivatives:
  S(0), H(0), S(+inf) and log F(+inf) have zero rate derivative;
  log density/log hazard at zero and log hazard at infinity have derivative
  1/r; hazard at infinity has derivative 1. Mixed finite/infinite batches
  must preserve finite-output derivatives for shared and per-entry rates.
- Class and method docstrings state purpose, arguments, returns/shapes and
  dtype/shape misuse errors. Include defining formulas, positive finite rate
  domain, inverse-time units, rate=1/mean, log-rate/log-mean conversions and a
  short conversion example. `from_log` states natural leaves and overflow
  limits. No narrative `docs/` edits or documentation build in this code step;
  tutorial/navigation work remains in "A5 docs" below.

### Own tests and diff budget

Create `tests/unit/test_exponential.py`, with algebra or independently evaluated
NumPy formulas as expectations and existing `tests/impl_tolerances.py` values.

- Seven-method values, paired/multiaxis broadcasting, all time-policy rows,
  NaN isolation, large-H analytic hazard and underflowed survival/log density.
  At H=2e-30 assert relative precision with zero atol; check the lower-tail
  log CDF against log(2e-30), without accepting a zero-H result.
- Only natural rate as a PyTree leaf; keyword requirements, no value checks,
  no `.shape` alias, batch indexing/tree slicing and construction inside jit/vmap.
  Test ordinary/from_log integer/scalar/array dtype behavior and rejected kinds,
  plus reference shape/dtype and finite positive extreme-rate references.
- Finite log-input values and gradients against exp-parameterized algebra:
  d log f/d log r = 1-r*t. Test from_log keyword-only misuse and natural-rate
  overflow without promising a specific invalid-rate method result.
- Eager/jit/vmap agreement; interior forward/reverse derivative 1/r-t and
  Hessian -1/r**2, finite boundary derivatives and mixed-batch finite-component
  Jacobians. Use x64 order-two `check_grads`; use algebra in both modes.

Budget: about 90–110 source/docstring lines, 150–170 own-test lines and under
15 export/import-test lines; keep code/tests/docstrings at most 300 changed
lines. Reuse existing generic registration/policy tests and the independent
suite rather than copying their full matrices. Split before exceeding the
limit; don't defer correctness tests just to fit the diff.

Implementation accounting: 103 family lines, 183 own-test lines, 13 changed
export/import-test lines: **299 changed lines**. Complete boundary matrices,
registration behavior and x64 order-two finite differences reuse the independent
suite. Own tests focus on additional extreme-reference, input-dtype and
mixed-infinity cases. No coverage is deferred beyond the stated step scope.

### Checks, evidence and stop

After implementation, run focused own tests in separate x64/x32 processes,
then `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`
and `git diff --check`. Run the following read-only A5 selection in each mode
with `JAX_ENABLE_X64=1` and `0`, `SPEC_THROUGH=A5`:

```bash
.venv/bin/python -B -m pytest \
  tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py \
  tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py \
  tests/spec/test_a5_derivatives.py \
  -k '(exponential and not weibull) or nan_rate or boundary_values_agree_under_vmap_over_time' \
  -q -p no:cacheprovider
```

Planning evidence: this selection collected successfully with `--collect-only
-qq` in both modes (exit 0): construction 10, from_log 4, closed forms 13,
boundaries 54, derivatives 14 — **95 cases**. No implementation tests or full
gate ran during planning. The broader `-k exponential` selection also includes
seven Weibull identity cases, so it is not the A5.3 selection. Mixed-family
tests, Weibull identities and grid tests wait for their implementing steps;
report these exclusions and the existing x32 finite-difference skips explicitly.

After code validation, overwrite `REVIEW.md` with reading order, invariants,
spec/test mapping, exact commands/results and deferred docs. Commit this one
approved code step on the selected branch and stop before A5.4. Don't merge,
publish, edit the spec/independent tests or resolve existing deviations.

## A5.4 — Weibull

### Review fix-up — parameter notation and visible formulas

Status: implemented and verified; see REVIEW.md for the focused results.

Authorized by the maintainer's request to clarify the docstring, visually
separate the mathematics in both built-ins and address the inline WA question.
Same A5.4 step and branch; no new public API or numerical policy.

- `src/survjax/distributions/weibull.py`: define k=shape and b=scale in the
  first docstring sentence. Add whitespace and spec-linked comments that guide
  readers from safe operands to the interior formula, then boundary limits.
  Replace WA's reference-time question with its explanation: scale is already
  a positive finite time for valid Weibull parameters, whereas Exponential's
  inverse rate needs representability guards.
- `src/survjax/distributions/exponential.py`: visibly separate H=rate*t
  from input substitution and infinity restoration, naming its intermediate
  value; identify the log-density and constant-hazard formulas with comments.
  Clarify that the reciprocal guard concerns inactive reference inputs only.
- No tests added for comments/docstrings or an equivalent named intermediate.
  Run existing built-in own tests in both dtype modes, Ruff format/lint, mypy
  and git diff --check. Check the diff for unchanged operations and masks.
  Record exact results and the existing D-020 acceptance failure in REVIEW.md;
  do not rerun unchanged acceptance or edit its tests, spec or docs pages.
- Commit one review fix-up on A5.4, then stop before grids or docs steps.

Status: approved and implemented on 2026-10-13 from `cd3f213`, on the
maintainer-selected branch `A5.4`; awaiting review. Planning changed only
PLAN.md and created no branch or commit. Specification revision
23 and `.spec-seen` agree with design commit `a212ddd`; the synchronized
revision-22 numerical tests remain applicable. No spec ambiguity identified.

### Review fix-up — preserve tiny built-in CDF tails

Authorized by the maintainer's request to implement the review's proposed
tail-stable `log_cdf` branches for both built-ins. This is an A5.4 follow-up
on the same branch, with no public API or export changes. The current spec is
revision 24 (`b917ebe`); its update adds grid twins and does not change the
CDF-tail formulas. See D-021 for the numerical-policy deviation.

- `weibull.py`: add `_log_cumulative_hazard` from `shape * log(t / scale)`;
  use safe time/scale operands before logs and restore the zero/infinity limits.
  Override `log_cdf` to switch to `log_H` when `log_H < log(eps)`. Substitute
  before evaluating the `expm1` branch, including the infinite upper limit and
  values at which `exp(log_H)` would overflow.
- `exponential.py`: add the matching `log_H = log(rate) + log(t)` helper and
  `log_cdf` override, with the same lower-tail switch and safe branch operand.
- Add own tests for hand-derived lower-tail values (including a float32 case
  where forming H underflows), continuity around the dtype-specific switch,
  and finite shape/rate gradients under `jax.debug_nans` on both sides.
  Do not edit the independent revision-24 spec tests.
- Cover spec 4.2, 4.3, 4.5, 7.1 and 7.2. Run
  `scripts/check.sh --through A5`; report its current revision-24 A5.5 grid-test
  failures separately, and run the non-grid A5 acceptance modules in both dtype
  modes. Then overwrite REVIEW.md with mappings, exact results and this gate
  boundary. Commit this fix-up on A5.4 and stop.

### Review fix-up — assert switch gradients with shared tolerances

Authorized by the maintainer's test review. This A5.4 follow-up changes only
the two own `log_cdf` switch tests and REVIEW.md; no runtime code or public API.

- In both tests, compare values using `ATOL[x64_enabled]` and
  `RTOL[x64_enabled]` from `tests/impl_tolerances.py`. Keep `eps` only for
  constructing points on opposite sides of the algorithm's threshold.
- Assert analytic gradients, not just finiteness: Weibull's shape derivative
  is approximately `log(t)` on both sides; Exponential's rate derivative is
  approximately `1` at rate one. Add a short algebra comment explaining the
  O(H) difference on the standard branch. The existing RTOL covers that
  rounding-scale difference, so no tolerance constant is added.
- Run both targeted unit modules in x64 and x32, Ruff format/lint, mypy and
  `git diff --check`. Overwrite REVIEW.md with exact results and commit this
  test-only review fix-up on A5.4, then stop.

Outcome: keyword-only `survjax.Weibull(shape=..., scale=...)` and
`Weibull.from_log(log_shape=..., log_scale=...)`, with seven pointwise direct
methods, analytic hazard limits and safe finite boundary parameter derivatives.
Covered spec: 3.3, 4.1–4.2, 4.5–4.6, 5 (pointwise broadcasting), 7.2–7.3,
10.2–10.3, 10.5 (code docstrings) and applicable 12.1/12.2 family checks.
Grids remain A5.5. Likelihoods, summaries, sampling, parameter diagnostics,
R fixtures, benchmarks and installation checks remain later steps.

### Files and public API

- Create `src/survjax/distributions/weibull.py`: a registered frozen dataclass
  subclass of `SurvivalDistribution`, with `shape: jax.Array = param()` and
  `scale: jax.Array = param()` as its only dynamic leaves. Reuse registration
  for keyword requirements, common dtype, parameter broadcasting, indexing
  and pytree behavior. `.shape` is the natural parameter; `.batch_shape` is
  the array shape. No defaults, dtype keyword or parameter-value checks.
- Export `Weibull` from `src/survjax/distributions/__init__.py` and
  `src/survjax/__init__.py`; update `tests/unit/test_import.py`'s exact export
  expectation. Existing public names and signatures stay as they are.
- Type `from_log` with keyword-only `ArrayLike` inputs and `Self` return.
  Normalize both log inputs together with `_dtypes.normalize_parameters`
  before exponentiation, preserving input-kind rejection, weak promotion and
  stored-float64 warning behavior. Construct with natural `exp(log_shape)`
  and `exp(log_scale)` leaves. Overflow stores infinity; retain no log state.
- `reference_time()` returns scale directly, exactly `batch_shape` and the
  common parameter dtype. For valid parameters it is positive and finite;
  no guarantee for invalid parameters. The base stops substituted references.
- Override `log_survival`, `cumulative_hazard`, `log_prob` and `log_hazard`
  analytically; inherit `survival`, stable `log_cdf` and `hazard`. The inherited
  hazard honors the analytic log-hazard override, including at infinity.
  Keep existing base/registration/policy modules unchanged. Any need to change
  them is a scope expansion to explain before implementation.

### Numerical invariants

- At positive finite time let `u=log(t)-log(b)` and `H=exp(k*u)`.
  Use log S=-H, log h=log(k)-log(b)+(k-1)*u, and log f=log h-H.
  Computing the log ratio without first dividing avoids losing a representable
  H when t/b underflows, or overflowing t/b before a small shape is applied.
  Keep helpers private and local to this family; introduce no numerical layer.
- For zero/infinite time, substitute a safe positive finite input before logs,
  products and exponentiation in each helper. Select the actual boundary
  values afterward, using explicit shape regimes and xlogy-style treatment of
  the zero coefficient at k=1. A call to xlogy alone does not establish safe
  gradients: guard operands before operating. Stop only substituted inputs;
  all active shape/scale expressions, including -log(b), remain differentiable.
- With valid parameters, zero gives log S=H=0, S=1, log F=-inf.
  Log density/log hazard give +inf for k<1, -log(b) for k=1, -inf for k>1;
  hazard gives +inf, 1/b, 0 respectively. Positive infinity gives log S=-inf,
  H=+inf, S=0, log F=0, log density=-inf; log hazard gives -inf, -log(b),
  +inf, and hazard gives 0, 1/b, +inf in those same shape regimes.
- The base supplies negative/NaN-time policy and infinity density precedence.
  Raw log survival must preserve NaN shape/scale at zero and infinity so a
  boundary constant cannot conceal it. Analytic log hazard must also preserve
  NaNs. Accept invalid finite parameters without exceptions or a blanket
  validity mask; their values remain formula-driven as specified in 4.5.
- Mixed [zero, finite, +inf] entries must preserve finite-component first and
  second derivatives for shared and per-entry parameters. Never take a sum
  containing infinite outputs as a derivative test. Do not treat finite-time
  underflow/overflow as an exact time boundary or claim its gradients verified.
  No floors, epsilon, clipping, log-parameter storage or custom AD rules.

### Docstrings and docs to-do

Include summary, arguments, returns/shapes and static misuse errors for public
APIs. The class docstring gives S(t), positive finite domain, dimensionless k,
time units of b, broadcasting and all spec 4.6 mappings: R/flexsurv shape/scale,
NumPyro concentration/scale, survreg mu=log(b) and sigma=1/k, WeibullPH
m=b**(-k), log inputs and inverse scale b=1/r. Include the common-shape AFT/PH
relation gamma=-k*beta, natural versus logged flexsurv coefficients, and a
short conversion example. State from_log overflow behavior and derivative
limits at discontinuous boundaries. Narrative examples remain in A5 docs;
this code step edits no `docs/` and runs no documentation build.

### Own tests, independent coverage and budget

Create `tests/unit/test_weibull.py`. Prefer small named tests with stated
algebra/analytical expectations, using `tests/impl_tolerances.py`; no oracle
comes from running survjax. Reuse the independent suite's complete seven-method
value/boundary matrices, construction/indexing/dtype cases, shape-one identity,
lower/upper tails and x64 order-two forward/reverse `check_grads`.
Own tests add evidence beyond that suite:

- Full two-parameter log-density gradient and Hessian, including the mixed
  entry. For u=log(t/b) and H=(t/b)**k: gradient
  `(1/k+(1-H)*u, k*(H-1)/b)`; Hessian entries
  `-1/k**2-H*u**2`, `(H-1+k*H*u)/b`, and
  `k*(1-(k+1)*H)/b**2`. Check both differentiation modes and jit/vmap in
  both dtype processes. These are smooth positive-time tests, including k=1.
- from_log chain-rule gradients in both parameters: multiply the above
  natural gradients by k and b respectively. Check natural leaves, common
  dtype before exp, rejected input kinds and overflow without prescribing
  a particular invalid-parameter method value.
- Mixed-boundary finite-component Jacobians/Hessians with shared parameters
  and independent parameter batches, including analytic hazard. Pin zero
  k/b derivatives of S(0), log S(0), H(0), S(+inf), log F(+inf).
  At k=1 pin scale derivatives -1/b for finite log density/log hazard and
  -1/b**2 for hazard at zero/infinity; log density at infinity is excluded.
  Do not require shape derivatives at the discontinuous k=1 hazard limits.
- NaN shape/scale isolation beside valid entries at zero/finite/infinite time;
  raw analytic methods broadcast scalar and multiaxis times against batches
  without depending on wrapper pre-broadcasting (A5.3 review regression).
- Positive finite references across small/large valid scales and shapes;
  safe finite evaluations at t=b. Add independent power expectations for
  representable H despite t/b underflow/overflow (e.g. b=1e30, t=1e-30,
  k=0.25 gives H=1e-15, with the reciprocal case H=1e15). Compare tiny H
  with zero atol and existing rtol; compare underflowed probabilities in logs.

Budget: approximately 135–145 source/docstring lines, 140–145 own-test lines
and under 10 export/import-test lines, at most 300 changed code/test/docstring
lines. Tests ship with the code. If safe formulas and meaningful own coverage
cannot fit, propose a split and obtain approval before implementation; do not
omit correctness tests or compress code to force the budget.

Implementation accounting: 145 family/docstring lines, 150 own-test lines,
and 5 export/import-test lines: **300 changed code/test/docstring lines**.
The own mixed-boundary tests also cover the full interior two-parameter
Jacobian/Hessian in both differentiation orders and jit/vmap at k=0.7, 1, 2;
this avoids duplicating a standalone density-Hessian test. Scale derivatives
of the finite log-density/log-hazard limits are demonstrated by the independent
suite; own tests add hazard's -1/b**2 derivative at zero/infinity. No required
coverage was deferred. The sole non-grid acceptance failure is D-020, an
expected value missing log(shape); keep the correct formula and test untouched.

### Validation, commit and stop

After approval and branch selection, run the own Weibull module in separate
`JAX_ENABLE_X64=1` and `0` processes, then
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A4`.
Run all five non-grid A5 acceptance modules in each mode, with no `-k` filter:

```bash
JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest \
  tests/spec/test_a5_construction.py tests/spec/test_a5_from_log.py \
  tests/spec/test_a5_closed_forms.py tests/spec/test_a5_boundaries.py \
  tests/spec/test_a5_derivatives.py -q -p no:cacheprovider
```

Repeat with `JAX_ENABLE_X64=0`. This includes Exponential regressions,
Weibull/Exponential identities and mixed-family construction. Only the 48 grid
cases remain outside this A5 selection; full through-A5 acceptance waits for
A5.5. Report existing dtype skips exactly; never edit/skip/xfail spec tests.

Planning evidence: the above five-module command with `--collect-only -q`
exited 0 and reported **322 tests collected in 0.21s** in each dtype process.
Collection verifies selection only; no implementation tests or gate were run
in this planning turn. Planning `git diff --check` exited 0 with no output;
only PLAN.md is modified. Repeat the diff check after implementation.

After implementation, overwrite REVIEW.md with reading order, safe-input and
NaN/dispatch invariants, spec-to-test mapping, exact commands/results/skips
and deferred docs. Record any newly discovered ambiguity in DEVIATIONS.md and
ask; leave existing entries open. Commit one approved implementation step on
the maintainer-selected branch, never main, then stop before A5.5.

## A5.5 — Seven time-first grid methods

### Review fix-up — boundary Jacobians, tracing claims and docstrings

Authorized by the maintainer's review of `2d0c22d`; same A5.5 branch and API.
Implemented and verified: 42 own grid cases pass in each dtype mode; Ruff,
mypy and whitespace checks pass. REVIEW.md records the current fix-up evidence.

- `tests/unit/test_survival_grid.py`: differentiate the full mixed-boundary
  vector in both forward/reverse modes and assert analytical derivatives at
  the interior and every finite-valued boundary. Mark the nonfinite-valued
  rows outside spec 4.5's derivative guarantee explicitly in the expectation
  table. This makes the forward-mode part check actual boundary behavior.
- Keep dynamic tracing checks for JAX-compatible rank/bool/complex misuse.
  Capture string/object inputs as constants inside a zero-argument jitted call
  and match the grid conversion error, so JAX's argument rejection cannot
  satisfy the test before the grid method is traced.
- `_grid_methods.py`: replace the compressed Args/Returns/Raises lines with
  concise prose, retaining input/rank rules, output shapes and direct-method
  conversion/value policy. No executable implementation changes.
- Run the own grid module in both dtype processes, Ruff format/lint, mypy and
  `git diff --check`. No full gate rerun is needed for this test/docstring-only
  fix-up; retain the parent commit's gate evidence as historical. Overwrite
  REVIEW.md with actual results, commit this fix-up and stop for review.

Status: implemented and verified on 2026-10-15; awaiting review. Started from
`861ac71` on the
maintainer-selected `A5.5` branch. Planning used `4f867cb`, after A5.4 merged;
the maintainer has since committed D-022 in `861ac71`. The marker
`.spec-seen` records revision-24 content commit `b917ebe`. Design HEAD is now
`addc1cc` (archive snapshot); SPECIFICATION.md and GOALS.md have no diff from
the marker. The local acceptance tests are already synchronized. No branch,
implementation or spec/test synchronization was part of the planning request.

Outcome: all seven inherited grid methods use a single rank/reshape helper and
delegate to public registered direct methods. Own grid coverage has 42 cases in
each dtype mode, including analytical derivatives through `jit(vmap(...))`.
The full `scripts/check.sh --through A5` passes: own x64 415 passed/7 skipped,
x32 421 passed/1 skipped; independent x64 1116 passed/26 skipped, x32 1121
passed/21 skipped; source doctests 1 passed in each mode. Ruff and mypy pass.
Exact commands, reading order, invariants and the D-022 limitation are in
REVIEW.md. No spec, independent tests, docs pages or exports were changed.

### Public API and shape contract

Add the seven inherited methods to `SurvivalDistribution`:
`survival_grid`, `log_survival_grid`, `log_prob_grid`, `log_cdf_grid`,
`cumulative_hazard_grid`, `log_hazard_grid`, `hazard_grid`.
Each has signature `(self, t: ArrayLike) -> jax.Array`, accepts positional or
keyword `t`, and is available on built-ins and registered custom families.
No new top-level exports or family-specific grid implementations are needed.

- Convert a one-dimensional grid with JAX, then reshape it to
  `(K,) + (1,) * len(self.batch_shape)`. Evaluate the corresponding public
  direct method, returning `(K, *batch_shape)`. K=N still evaluates every time
  against every batch entry. Batch rank comes from current leaves, including
  inside `vmap`; component axes are excluded.
- Reject non-1-D grids with `TypeError`, including during tracing. Normalize
  array-conversion misuse to `TypeError`; let the registered direct method
  apply its existing dtype, promotion and value policy. This preserves integer
  inputs, silent float64-time conversion in x32, overrides and NaN precedence.
- Preserve input order and duplicates; support empty grids and empty batches
  through the same reshape/broadcast operations. Shapes are inspected statically;
  grid values never drive Python branching, sorting or host conversion.
- Grid wrappers delegate to the public direct methods, so their safe-input
  calculations and boundary derivatives remain active. No second registration
  wrapper, raw-method bypass, numerical formula or derivative rule is added.

### Files to create or change

- Create `src/survjax/distributions/_grid_methods.py`: seven explicit typed
  thin methods and one private time-conversion/rank/reshape helper. Use a
  type-checking-only base import to avoid a circular runtime import. Each
  public method docstring gives its quantity, 1-D `t`, time-first return shape,
  conversion/value policy and `TypeError` conditions.
- Change `src/survjax/distributions/base.py`: import the private grid module
  and bind its seven methods on the base. The existing registration list
  continues to wrap only direct methods. This file is already 339 lines;
  keeping the grid functions separate limits its addition to import/bindings
  and avoids an unrelated registration refactor.
- Create `tests/unit/test_survival_grid.py`: own analytical and structural
  checks described below. Prefer a small explicit expectation table over a
  generated method factory or a testing helper layer.
- Change `tests/impl_tolerances.py` only to document the rounding/truncation
  scale for applying the existing derivative tolerances to unit-scale smooth
  grid functions. Reuse its ATOL/RTOL for analytical values and derivatives;
  keep the existing finite-difference tolerance values unchanged.
- Update this plan's outcome and overwrite REVIEW.md after implementation.
  Spec files, `tests/spec/`, exports, dependencies and `docs/` stay untouched.

### Own tests and independent acceptance

Keep own tests focused; the read-only grid acceptance module already covers
all seven names, ordinary values, shapes, keyword arguments and policy rows.

- Hand-derived Exponential values for all seven grids with K=N and an unsorted
  repeated time vector: use `S=exp(-r*t)`, `H=r*t`, `h=r`, `log f=log(r)-r*t`
  and `log F=log(-expm1(-r*t))`. Assert full matrices and row order.
- Scalar batches, several batch axes, empty grids and empty batches: assert
  exact time-first shapes. A small custom family reduces its own parameter
  components and verifies they never become grid batch axes; a direct override
  verifies dispatch through the public registered method rather than a default.
- Rank and dtype misuse in eager/jit calls; integer inputs, explicit float32
  inputs in x64, and warning-free NumPy float64 grids in x32. Exercise the
  positional and keyword `t` forms.
- Eager/jit/vmap agreement for every grid, including `vmap` over distributions
  whose per-entry batch rank differs from the original. Count tracing for
  changed parameter/grid values with unchanged shape/dtype/PyTree structure.
- Assert analytical first and second rate derivatives for all seven
  Exponential grids on a positive finite grid. Use order-two `check_grads` in
  both forward/reverse modes on unit-scale interior cases with shared derivative
  tolerances. Expected derivatives come from the same stated algebra, never
  another library call. No time or discontinuous boundary derivative is promised.
- Weibull shape-one grid from zero with the section 11.1 scales
  `10 * exp([0, -0.5, -1])`: verify analytical survival, density and hazard,
  plus the zero shape derivative of `S(0)`. Mixed zero/interior/infinity
  entries retain the existing finite-entry derivative isolation; NaN times
  and parameters remain confined to the corresponding rows/columns.

Covered spec: 4.2, 4.5, 5, 7.2–7.3, 8.1, 10.2–10.3, 10.5 (docstrings),
12.1 and the grid part of Slice A in 12.2. The 116-case independent module is
`tests/spec/test_a5_survival_grids.py`; do not edit, skip or xfail any case.
The full A5 gate must now pass, including all seven grid twins.

### Validation, review and scope boundary

After approval, target roughly 300 changed source/test/docstring lines in one
code commit. Keep the private module and own test file below about 300 lines
each. If the tests exceed that step budget, propose a split before expanding.
Run focused grid tests in separate x64/x32 processes, then
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`
and `git diff --check`. The full gate includes own suites, Ruff, mypy, source
doctests and independent tests in both dtype modes. Report exact skips/results.
Write REVIEW.md with reading order, shape/dispatch invariants, spec-to-test
mapping, actual commands/results and remaining limitations. Commit on A5.5
and stop for review.

D-022 is the maintainer's existing upper-tail issue, committed in `861ac71`. This plan
preserves that file and keeps grid semantics equal to the current direct
methods, including their known log-CDF upper-tail limitation. Propose a
separate numerical step after A5.5 for the shared `log1mexp` fix, with its own
plan approval, mpmath and derivative checks. The grid gate alone does not
establish upper-tail accuracy or complete Slice A. Likelihoods, summaries,
sampling, fixtures, benchmarks and installation checks keep their later scope.

Docs to-do:

- Extend the later grids topic to all seven methods, paired-vector versus
  shared-grid examples (including K=N), posterior batches, order/duplicates,
  component axes and zero-time Weibull examples. Update API navigation and
  playground examples in the separate A5 docs steps.

Planning evidence (collection only; no tests or full gate executed):

- In each dtype process, `SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  tests/spec/test_a5_survival_grids.py --collect-only -q -p no:cacheprovider`
  reports **116 tests collected**.
- In each dtype process, the same collection against `tests/spec` with
  `-k test_a5` reports **438/1142 tests collected (704 deselected)**.
- Only PLAN.md is changed by this planning turn. The pre-existing
  DEVIATIONS.md change is preserved; REVIEW.md remains the last code review.
- Planning `git diff --check` passes with no output. No implementation,
  numerical test execution, docs build or commit was performed.

## A5.5 follow-up — D-022: preserve log-CDF upper tails

Status: approved on 2026-10-15, from clean branch `A5.5` at `ace2f4e`.
Implemented and verified on 2026-10-15; full through-A5 gate passes in both
dtype modes. REVIEW.md records the regression failure and final results.
The maintainer confirmed D-022 as the target and authorized using this branch.
This is one numerical step after the grid work, with its own implementation
commit and review stop. The maintainer approved implementation in conversation.
Specification revision 24 remains read-only. D-022 records the proposed change
to the section 4.2 default formula; leave its status open for the maintainer.

Outcome: generic, Exponential and Weibull `log_cdf` retain small negative
upper-tail values while representable, instead of rounding to zero when the
intermediate CDF rounds to one. `log_cdf_grid` inherits the correction through
its existing delegation. Public names, signatures, shapes and exports do not
change. Covered spec: 4.1–4.3 (numerical recipe only), 4.5, 5, 7.1–7.3 and
12.1. Do not implement HazardFamily, interval methods or later-slice features.

### Files and numerical invariants

- Create `src/survjax/distributions/_log_probability.py` with one typed private
  `log1mexp(x)` helper, taking floating log probabilities and preserving dtype
  and shape. For x <= 0 compute log(1-exp(x)): `log(-expm1(x))` for
  x > -log(2), otherwise `log1p(-exp(x))`. Form the split constant in the
  input dtype. At x=0 return -inf; at x=-inf return zero; NaN and positive
  inputs give NaN. Substitute safe operands in both inactive branches before
  exp/expm1/log/log1p, including the zero-mass case. Stop gradients only on
  substituted values. No floors, probability clipping or new epsilon.
- In `src/survjax/distributions/_survival_methods.py`, delegate the default
  `log_cdf` calculation to this helper, replacing its local zero-mass guard.
  Keep `_raw_log_survival`, the policy wrappers and infinity/cure guards
  unchanged. Update the public docstring to describe both tail branches and
  the remaining lower-tail limitation when the primitive rounds log S to zero.
- In `src/survjax/distributions/exponential.py` and `weibull.py`, replace only
  the standard log-CDF branch with the shared helper applied to -H. Retain
  D-021's direct log-H lower-tail switch and safe log-H substitution before
  exponentiation, including the overflow/upper-limit guard. This preserves
  the existing tiny-tail behavior and boundary parameter derivatives.
- Add `tests/unit/test_log_cdf_tails.py` for the focused public-API regressions
  below. Define one minimal registered exponential custom family locally to
  exercise the generic default independently of the built-in overrides.
- In `tests/impl_tolerances.py`, add named tail tolerances with zero absolute
  tolerance and relative targets 1e-8 (x64), 1e-4 (x32). Explain why absolute
  tolerance would accept an erroneous zero for these tiny values and why
  these targets cover rounding in log-H reconstruction and transcendental
  operations on the chosen normal-range cases. Use the same relative targets
  for tiny analytic derivatives; retain existing finite-difference derivative
  tolerances for `check_grads`. Report failures without loosening tolerances.
- Overwrite `REVIEW.md` after implementation with reading order, invariants,
  spec-to-test mapping, exact commands/results, skips and remaining limits.
  Update this plan's status and D-022's evidence without resolving the entry.

### Tests accompanying the code

Keep the new test module below about 250 lines. Each behavior states its spec
section and expected-value origin. Use the process-selected dtype throughout;
mpmath references use at least 100 decimal digits and the actual rounded input
values converted on the host, never values computed by library methods.

- **Upper-tail values (4.2/7.1/12.1):** compare the generic family and both
  built-ins with mpmath `log1p(-exp(-H))`. Include H=20.1, 33.1 and 54.6
  from D-022, an ordinary H, and deeper normal-range cases (e.g. H=70 in x32,
  H=600 in x64). For Weibull include a nonunit shape/scale case using
  H=(t/b)**k. Require negative, nonzero results and relative accuracy; compare
  eager, jit and vmap. These tests must fail against the original formula.
- **Branch switch (7.2–7.3):** use times on both sides of H=log(2), plus
  the split itself, against independent values and first/second parameter
  derivatives. Test smooth interior points with
  `check_grads(order=2, modes=("fwd", "rev"))`; compare analytical or mpmath
  derivatives at the split rather than finite differences through a branch.
- **Upper-tail derivatives (7.3):** compare forward/reverse parameter Jacobians
  and Hessians with analytical or mpmath derivatives at an upper-tail point,
  including both Weibull parameters. At H=rate*t the rate derivatives are
  t/expm1(H) and -t**2*exp(H)/expm1(H)**2. Use relative-only assertions so
  a zero derivative cannot pass. Compose derivative calls with jit and vmap.
- **Boundary and inactive operands (4.5/7.2):** exercise public log-CDF on a
  mixed vector with zero, ordinary interior, upper tail and +inf. Check full
  forward/reverse Jacobians at finite-valued rows and finite-component
  Hessians, including zero derivatives at +inf. Log CDF at zero is -inf;
  do not assert its derivative as a spec guarantee. Run valid mixed calls
  and derivative checks eagerly under `jax.debug_nans(True)` to detect
  unselected unsafe operations. Check NaN times/parameters separately outside
  that context. Preserve parameter-dependent finite cure limits through the
  existing infinity tests; the generic wrapper must still dispatch unchanged.
- **Grid integration (5/12.1):** one batched, time-first `log_cdf_grid` tail
  comparison against independent references for all three family paths;
  include K=N to distinguish grid shape from paired broadcasting. Existing
  grid tests retain responsibility for the other six adapters.
- Rerun the existing D-021 lower-tail/threshold regressions and generic
  zero-mass, boundary, cure-limit and NaN-policy tests. No edits to
  `tests/spec/`, no new dependency, and no R fixture generation.

### Validation, budget and stop

After plan approval, first run the new value regression against the old code
and record the expected failure. After the fix, run in separate processes:

```bash
JAX_ENABLE_X64=1 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
JAX_ENABLE_X64=0 .venv/bin/pytest tests/unit/test_log_cdf_tails.py tests/unit/test_exponential.py tests/unit/test_weibull.py tests/unit/test_family_direct.py tests/unit/test_family_infinity.py tests/unit/test_survival_grid.py -q
UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5
git diff --check
```

The full gate covers Ruff format/lint, mypy, own tests, source doctests and
independent acceptance in both dtype modes. Report actual counts and outputs,
including any failures; the previous grid gate is not evidence for this fix.
Aim for about 300 changed lines including tests and review; keep source changes
small and split the step for approval if the diff exceeds that budget. Commit
the approved implementation on `A5.5` with an imperative subject and a body
explaining the lost upper-tail precision, then stop for review before A5 docs.

Docs to-do: explain both-tail log-CDF evaluation and D-021's retained lower-tail
switch in the later direct-method/custom-family docs topic. State the generic
primitive's representability limitation. Do not edit `docs/` in this code step.

Planning checks: only PLAN.md changes in this turn. `git diff --check` passes
with no output. No implementation, numerical tests, full gate, docs build or commit at this
planning stop. REVIEW.md remains the last completed grid review.

## A5.5 review follow-up — consolidate guarded log arithmetic

Status: approved on 2026-10-15 at `0369331`, branch `A5.5`. The maintainer
selected `_log_math.py`: log and log1mexp also apply outside probability
calculations and approved implementing this refactor in conversation.
Implemented and verified on 2026-10-15: 36 tail cases pass per dtype mode,
and the full through-A5 gate passes. REVIEW.md records exact commands/results.

### Files, responsibilities and API

- Rename `src/survjax/distributions/_log_probability.py` to `_log_math.py`.
  Name its existing guarded kernel `safe_log1mexp(x)`, explicitly documenting
  the convention log(1-exp(x)) for x<=0, dtype/shape preservation, boundaries
  and NaN behavior. Keep the currently verified guarded formulas in this
  organizational step. Reusing jax.nn with a corrected custom JVP changes the
  derivative implementation and should have its own numerical plan.
- Move `_safe_log` from `_survival_methods.py` here as `safe_log`, preserving
  its zero substitution and existing behavior. "Safe" means protecting
  inactive/boundary operands; it does not mean clipping or validating values.
- Add `log_cdf_from_log_hazard(log_h)` in `_log_math.py`, moving the identical
  dtype-epsilon switch, overflow guard and safe exponentiation out of the two
  built-in log-CDF overrides. Its standard branch calls `safe_log1mexp`.
- In `_survival_methods.py`, import `safe_log` and `safe_log1mexp`; keep the
  distribution-aware primitive/derivative helpers and policy wrappers here.
- In `exponential.py` and `weibull.py`, reduce each log-CDF body to a call to
  `log_cdf_from_log_hazard(self._log_cumulative_hazard(t))`. Each family keeps
  its distinct analytic log-H calculation. Future log-H families can reuse the
  shared function without adding a family base class in this step.
- Public API and exports are unchanged. No dependency or docs-page changes.
  Covered spec: 4.1–4.2, 4.5, 7.2–7.3, 10.2 and 12.1. The current spec is
  revision 25; `.spec-seen` still records revision 24. Preserve both during
  this planning/refactor step; spec alignment remains maintainer bookkeeping.

### Tests, checks and review stop

- Extend `tests/unit/test_log_cdf_tails.py`'s derivative checks to H=54.6
  and the existing deep-tail values (H=70 in x32, H=600 in x64), against
  independent mpmath references with the existing relative-only tolerances.
- Add the analytical mixed objective discussed in review as a regression:
  L(eta)=exp(eta)-eta-log F(c), at eta=log(2), with c=100 in x32 and c=1000
  in x64. Check finite value, gradient and Hessian eagerly, under jit and
  debug_nans. Its expected derivatives follow from H=c*exp(eta); evaluate
  the tiny correction with mpmath, rather than asserting an approximate zero.
  This is a synthetic mathematical composition, not a Slice B likelihood API
  or an optimizer implementation. The unmodified JAX rule gives a NaN Hessian
  in the local review probes; retain the useful finite curvature of this sum.
- Rerun existing generic density/hazard, D-021 lower-tail switch, boundary,
  cure-limit and grid coverage. No private helper is exported for testing.
- Run focused log-CDF/family tests in separate x64/x32 processes, then
  `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`
  and `git diff --check`. Report exact results; do not modify spec tests.
- Aim for about 300 changed lines. Update REVIEW.md with reading order,
  numerical invariants, spec mappings and actual checks; commit the approved
  refactor on A5.5 and stop for review. Keep deviations open for the maintainer.

Docs to-do: explain the guarded arithmetic and its limits in the later A5
numerical/custom-family topic. The helper names remain private implementation
details, not a user-facing API.

Planning changed only PLAN.md. Implementation changes the files above and
REVIEW.md; the spec, marker, docs pages and independent tests remain untouched.

## A5.5 review follow-up — WA log-math notes and cumulative-hazard naming

Status: approved in conversation on 2026-10-15 at `7208c81`, branch `A5.5`.
Implemented and verified on 2026-10-15: the three WA notes in `_log_math.py`
are replaced as approved. Targeted x64: 72 passed/1 skipped; x32: 73 passed;
static and whitespace checks pass. REVIEW.md records exact commands/results.

- In `src/survjax/distributions/_log_math.py`, replace the math-note request
  with F=H-H**2/2+O(H**3), hence log F=log H-H/2+O(H**2). This explains
  the existing tiny-H branch without changing its threshold or formula.
- Answer the gradient note beside the substitution: `zeros_like(log_h)` is
  a constant with zero tangent, so an explicit stop_gradient is redundant.
  Keep the real log-H branch differentiable and retain the existing guards.
- Rename `standard` to `log_cdf_from_h` and `value` to `log_cdf` in that
  function, resolving the request for more informative intermediate names.
- Rename the private function to `log_cdf_from_log_cumulative_hazard`, because
  it takes log H, not the instantaneous log hazard. Update imports/calls in
  `exponential.py` and `weibull.py`; keep their log-CDF overrides. The generic
  default already calls safe_log1mexp(log S). Computing log H from log S
  would lose the built-ins' lower-tail guarantee after H underflows to zero.
- Public API and numerical behavior are unchanged; cover spec 4.1–4.2, 7.2
  and 12.1. No docs pages, new abstractions, new tests or tolerance changes.
  Existing independent-reference lower-tail, switch and transformation tests
  cover these naming/docstring edits; retain them without modification.
- Run the log-CDF tail, Exponential and Weibull unit modules in both dtype
  processes, Ruff format/lint, mypy and git diff --check. Record exact results
  and the existing full-gate evidence in REVIEW.md; do not rerun or claim a new
  full gate for this naming/docstring-only fix-up unless a failure warrants it.
- Keep the diff below about 300 lines. Update REVIEW.md, commit the approved
  review fix-up on A5.5 and stop. Spec, marker, independent tests and deviation
  statuses stay unchanged. Docs to-do: use H for cumulative hazard and h for
  instantaneous hazard consistently in the later numerical-methods topic.

Planning evidence: float32 probes of the unchanged public built-ins give
Exponential(rate=1e-30), t=1e-30: log S=-0, generic log CDF=-inf, built-in
log CDF=-138.1551055908203; Weibull(shape=10, scale=1), t=1e-5: log S=-0,
generic log CDF=-inf, built-in log CDF=-115.1292495727539. Only PLAN.md is
changed by the planning turn; the maintainer's WA notes remained intact then.

## A5.5 review follow-up — explicit cumulative-hazard names and helper placement

Status: approved in conversation after `9a8bb2a` on branch `A5.5`.
Implemented and verified: focused x64 249 passed/1 skipped; x32 250 passed.
Ruff, mypy and whitespace checks pass; REVIEW.md records the exact results.
The naming audit found cumulative quantities abbreviated in the shared
conversion and three own test modules. Instantaneous hazard API names are
correct. Starting working tree was clean; this planning turn changes PLAN.md.

### Files and changes

- Move `log_cdf_from_log_cumulative_hazard` from `_log_math.py` into
  `_survival_methods.py`, alongside the generic log-CDF method. Leave only
  general guarded log operations in `_log_math.py`. Update its two imports in
  `exponential.py` and `weibull.py`; retain their analytic log-H overrides.
- In the moved helper, rename `log_h` to `log_cumulative_hazard`,
  `safe_log_h` to `safe_log_cumulative_hazard`, and `log_cdf_from_h` to
  `log_cdf_from_cumulative_hazard`. Keep the formulas, thresholds, guards,
  mathematical H notation and gradient behavior unchanged.
- In `tests/unit/test_log_cdf_tails.py`, use `cumulative_hazard` and
  `cumulative_hazards` for H quantities, including the parametrized derivative
  test argument and its mpmath references.
- In `tests/unit/test_exponential.py` and `test_weibull.py`, rename lower-tail
  switch locals to `log_cumulative_hazard`; rename the Weibull switch's
  `hazard` local to `cumulative_hazard`. In its mixed-boundary derivative
  derivation, replace `h`, `gh` and `hh` (which actually describe H) with
  `cumulative_hazard`, `cumulative_hazard_gradient` and
  `cumulative_hazard_hessian`. Use `log_hazard_gradient` and
  `log_hazard_hessian` for the instantaneous log-hazard derivative arrays so
  the two quantities remain distinguishable throughout the same formula.
- Public API/exports are unchanged. No changes to spec, independent tests,
  marker, docs pages, dependencies, deviations, numerical assertions or
  tolerances. Covered spec: 4.1–4.2, 4.5, 7.2–7.3, 10.2 and 12.1.

### Validation and stop

- Audit `src/` and own tests for remaining ambiguous cumulative-hazard names;
  inspect the meanings rather than renaming actual instantaneous hazards.
- Reuse the existing tests; add no tests for a pure relocation/rename. Run
  the three changed test modules in separate x64/x32 processes, plus generic
  direct/density/infinity and grid modules to check the relocated helper's
  imports and callers. Run Ruff format/lint, mypy and git diff --check.
- Keep the step below about 300 changed lines. Overwrite REVIEW.md with reading
  order, unchanged numerical invariants, spec/test mapping, exact results and
  historical full-gate evidence. No full-gate rerun is planned for this
  relocation/naming-only fix-up unless a failure warrants it. Commit on A5.5
  and stop for review; do not start docs work.

Docs to-do: distinguish cumulative hazard H, log cumulative hazard log H,
instantaneous hazard h and log hazard log h in the later numerical-methods
topic. Internal helper identifiers are not public documentation API.

### A5 docs

Docs to-do:

- Audit all `docs/` against merged A5 code: mark stale, missing or fine.
  A5.1 commit `3f50962` changed `docs/distributions.md` (12 insertions,
  1 deletion); treat it as a draft. Include the earlier tutorial rewrite
  `60bee93` and its distributions, SurvivalData, index and config changes.
- Custom families: guarded-boundary extension recipe, proper-family zero
  limits and parameter-dependent cure limits; primitive safe operands/NaN
  propagation; mixed-entry derivative isolation and its limits. Correct
  PiecewiseHazard's reference to exactly `batch_shape` in `rates.dtype`.
- Built-in families: Exponential units/domain, rate=1/mean, log-rate/log-mean
  conversions and from_log overflow limits; Weibull's complete 4.6 table and
  conversions for R/flexsurv, NumPyro, survreg mu=log(b), sigma=1/k,
  WeibullPH m=b**(-k), inverse scale and log inputs; common-shape PH gamma=-k*beta.
  A5.4 includes these mappings in its class docstring as required by 4.6;
  the docs step expands them into executable conversion examples and explains
  natural estimates versus logged flexsurv coefficients. Cover zero/infinity
  hazard limits and the exclusion of discontinuous shape derivatives at k=1.
- Direct methods and grids: executable synthetic examples, paired-vector
  pitfalls and posterior batch `shape=k[:, None]`, `scale=jnp.exp(beta @ X.T)`.
- Update `docs/index.md` navigation/current coverage and `docs/playground.ipynb`
  for the implemented APIs. Preserve the tutorial; claim no unverified R
  agreement or full-Slice-A completion. Inspect rendered pages and execute
  examples/notebooks in both dtype modes.
- Code steps append any additional documentation needs here.

Docs steps:

- A5.d1: audit `docs/` against merged A5 code (A5.1 drafts included); list
  stale, missing, fine; split the rest into steps.
- A5.d2+: one per page or topic, e.g. SurvivalData interface, built-in families,
  custom families, grids, navigation or playground, as determined by the audit.
  Each is at most 300 changed lines; `scripts/build_docs.sh` builds clean and
  examples/notebooks execute. Review against the spec and merged code, covering
  every Docs to-do. A5 is done after its last docs step merges, before A6 starts.

### A5.d1 — audit result and docs-step split (2026-10-15)

Audited on branch `docs-A5` at `055e8b8` (equal to main) against the merged
A5 code, spec revision 25 and the docs to-do above. The maintainer asked for
narrative, tutorial-style pages with links to the mathematics and, where it
helps, to R's `survival`/`flexsurv`, rather than assertion-style examples,
and for a review of the public docstrings in the same style.

Audit of `docs/`:

| Page / part | Verdict | Finding |
| --- | --- | --- |
| `index.md` | stale | Says built-in distributions are planned; nav lacks built-ins and grids. |
| `distributions.md` top note, "Many distributions" last paragraph | stale | Calls `Exponential`, `Weibull` and time-first grids planned. |
| `distributions.md` constant hazard, Gompertz, batches, piecewise, JAX values | fine | Executes; tutorial voice the maintainer wants. `PiecewiseHazard.reference_time` already has exactly `batch_shape` in `rates.dtype` (to-do item satisfied). |
| `distributions.md` `ProportionalHazards.reference_time` | fine | Corrected on checking: the nested baseline is broadcast at construction, so it already returns `(3,)`. |
| `distributions.md` "edges of the time axis" (A5.1 draft) | stale/missing | Correct values, but assertion blocks instead of narrative; no mixed-batch derivative isolation (A5.2b), no cumulative-hazard/log-hazard guards. |
| `distributions.md` "why the log scale" | missing | No H / log H / h / log h distinction; no upper-tail `log_cdf` behaviour (rev 25, D-022). |
| `distributions.md` checklist/details | fine | Checklist already says "exactly `batch_shape`". |
| built-in families page | missing | Exponential and Weibull: formulas, domains, units, `from_log`, boundaries, k=1 derivative exclusion, 4.6 conversions. |
| grids page | missing | Seven `_grid` twins, paired-vector pitfall, posterior batch example, zero-time k=1 grid. |
| `survival-data.md` | fine | A1–A3 only; no A5 content needed. |
| `playground.ipynb` | stale | SurvivalData only; to-do asks for the implemented distribution APIs. |
| `simulation-guide.md` | fine | Excluded from the build; describes later slices. |

Audit of public docstrings (rendered only through `help()`; no autodoc):

- Wrong: `SurvivalDistribution.reference_time` says "broadcastable to
  `batch_shape`"; spec 4.2 (rev 22) requires exactly `batch_shape` in the
  parameter dtype.
- Terse and test-like: `Exponential`, `Weibull` and their methods compress
  formulas, limits and spec citations into fragments ("pairing equal-length
  vectors", "The infinity rate derivative is 1", "Uses no tail subtraction").
  Several use one-line `Args: t: ...` forms that do not read as sections.
- Implementation detail in user docs: base `log_survival`, `survival`,
  `log_cdf`, `cumulative_hazard`, `log_prob`, `log_hazard` describe the policy
  wrapper's substitution mechanics. Users need the observable rule (values at
  0/+inf/negative/NaN; which derivatives are guaranteed), not the mechanism.
- Grid docstrings are correct but identical boilerplate; they lack the
  `(K, *batch_shape)` worked shape example and the reason to prefer them.
- Fine: `param`, `static`, `distribution` (doctest), `batch_shape`,
  `__getitem__`, `register_family`.

Proposed steps (one commit, REVIEW.md and stop after each; each at most 300
changed lines; `scripts/build_docs.sh` clean in both dtype modes):

- **A5.d2 — built-in families page** (`docs/builtin-families.md`, toctree
  entry). Story: time to progression in a synthetic cohort. Exponential:
  constant hazard, `rate=1/mean`, memorylessness, `from_log(log_rate=...)`
  and log-mean conversion, overflow giving an infinite rate. Weibull: role of
  k (falling, constant, rising hazard), b as the time where S=e^{-1}, k=1
  equals the exponential, values at t=0 and +inf for k<1, =1, >1, and why the
  shape derivative of log f(0) is excluded at k=1. Math links: Wikipedia
  exponential and Weibull distributions, survival/hazard function articles.
  Executed examples print values; expected numbers are derived in the prose.
- **A5.d3 — coming from R and NumPyro** (section of the same page, or
  `docs/parameterizations.md` if d2 is near budget). Spec 4.6 table as a
  narrative: `dweibull`/`pweibull`, `flexsurvreg(dist="weibull")` natural
  estimates vs `coef(fit)` logs, `WeibullPH` m=b^-k, `survreg` mu=log b,
  sigma=1/k with a worked conversion, common-shape AFT→PH gamma=-k*beta,
  NumPyro `concentration`. R code appears as non-executed `r` blocks labelled
  as input conventions; no numerical agreement claimed (10.5; fixtures are A8).
  Links to the CRAN reference pages of `survival::survreg` and `flexsurv`.
- **A5.d4 — survival curves and time-first grids** (`docs/grids.md`).
  Pointwise pairing vs `_grid` twins; the pitfall when K equals N; posterior
  draws `Weibull(shape=k[:, None], scale=jnp.exp(beta @ X.T))` with synthetic
  draws giving `(K, D, N)`; the section 11.1 fixed vector (k=1, scales
  `10*exp([0,-0.5,-1])`) on a grid starting at t=0 with finite gradients;
  grids under `jit`/`vmap`.
- **A5.d5 — custom-family page refresh** (`docs/distributions.md`). Remove
  stale "planned" text and link to the built-ins; fix
  `ProportionalHazards.reference_time` to broadcast to `batch_shape`; rewrite
  the time-edges section as narrative with printed outputs instead of asserts,
  covering mixed-batch derivative isolation and finite cure limits; add H,
  log H, h, log h and both `log_cdf` tails to the log-scale section.
- **A5.d6 — built-in docstrings** (`exponential.py`, `weibull.py`). Narrative
  summary with formula, domains and units, Google-style Args/Returns/Raises,
  4.6 mapping kept in the Weibull class docstring, a short doctest each
  (run by `check.sh`), limits stated as observable behaviour. No code change.
- **A5.d7 — base and grid docstrings** (`base.py`, `_survival_methods.py`,
  `_grid_methods.py`). Fix `reference_time`; replace mechanism with the
  observable value/derivative rules; give grid twins a worked shape example.
  No code change; split if over budget.
- **A5.d8 — navigation, playground, slice evaluation.** `index.md` coverage
  and toctree; add a distributions section to `playground.ipynb`
  (Exponential/Weibull, a grid, `from_log`, things to try); the 10.5
  end-of-sub-slice rendered-page evaluation in REVIEW.md.

Maintainer decision (2026-10-15): do d2–d8 as one docs step, without the
300-line limit; plots and `matplotlib` (docs group only) are allowed;
document what is implemented (spec rev 25), `.spec-seen` follows separately.
Status: implemented on `docs-A5` as one commit; see REVIEW.md.

Open questions (resolved by the decision above):

- Plots would help the hazard-shape story but need `matplotlib` in the docs
  group (a new dependency). Default: no plots; small printed tables instead.
- The spec is at revision 25 (`1e9011c`) while `.spec-seen` is `b917ebe`
  (rev 24). Docs will describe rev 25 behaviour (upper-tail `log_cdf`), which
  the merged code implements; marking `.spec-seen` is a separate action.

## Verification and acceptance mapping

Independent A5 modules: construction/reference arrays -> A5.3/A5.4;
from_log -> A5.3/A5.4; closed forms/tails -> A5.3/A5.4;
boundaries -> A5.1/A5.2/A5.3/A5.4; derivatives -> A5.3/A5.4;
time-first grids (all seven methods) -> A5.5. The three revision-22 A4 additions
are A5.1 regressions. Tests purposely do not define finite-invalid-parameter
values, overflowing-from_log method values, discontinuous shape gradients,
arbitrary custom boundary derivatives or a particular reference-time formula.

- Own value tolerances remain in `tests/impl_tolerances.py`. For tiny H use
  relative-only comparisons at the existing rtol, with zero atol; do not accept
  an absolute tolerance that makes zero pass. New derivative/tail scales need
  justification in that same file. Expected values have stated algebra/identity
  origins. Never loosen tolerances to make failures pass.
- Run interior order-two forward/reverse `check_grads` with justified tolerances;
  independent finite-difference cases are x64-only, while hand-derived first/
  second derivatives run in both modes. Own x32 finite differences need their
  own scale justification. Do not toggle JAX x64 inside a process.
- Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Each code step runs
  `scripts/check.sh --through A4` and
  `git diff --check`, plus read-only independent tests for completed behavior
  in separate x64/x32 processes. At A5.5 run the full
  `scripts/check.sh --through A5` against the synced revision-24 suite.
  Report focused selection and still-unimplemented cases explicitly; never
  present an unsynced local gate as revision-24 acceptance.
  Docs steps run `scripts/build_docs.sh` in both dtype modes and inspect the
  rendered pages and outputs against the merged A5 implementation.
- REVIEW.md records reading order, numerical/storage invariants, spec-to-test
  mapping, exact outputs/skips and open deviations; docs steps also record
  executable docs and rendered-page review. One commit,
  then stop. No merge, publishing, remote setup or next-step work without approval.

## Evidence gathered while updating this plan

Read-only execution used this checkout's `.venv/bin/python -B`, with
`-p no:cacheprovider`, against `../spectest` at `7f1d6df` (no sync or writes):

- x64: `JAX_ENABLE_X64=1 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  ../spectest/tests/spec --collect-only -q -k test_a5 -p no:cacheprovider`
  exited 0: **370/1074 tests collected (704 deselected)**, 0.44s.
- x32: `JAX_ENABLE_X64=0 SPEC_THROUGH=A5 .venv/bin/python -B -m pytest
  ../spectest/tests/spec --collect-only -qq -k test_a5 -p no:cacheprovider`
  exited 0: boundaries 186, closed forms 58, construction 29, derivatives 39,
  from_log 10, grids 48 (**370 total**).
- Ran `JAX_ENABLE_X64=1 SPEC_THROUGH=A4 .venv/bin/python -B -m pytest
  ../spectest/tests/spec/test_a4_nan_precedence.py -q -k infinity
  -p no:cacheprovider`, then the identical command with `JAX_ENABLE_X64=0`.
  Both exited 1:
  **3 failed, 7 passed, 24 deselected** (x64 0.66s; x32 0.46s).
  The three new NaN-parameter cases get -inf rather than NaN from the current
  density policy. This is an implementation gap against the changed contract,
  not a suspected spec-test error; no new deviation is needed.
- A5 tests were collected/read, not executed. The full gate, docs build,
  numerical A5 implementation checks and R work were not run for this plan.
- `git diff --check` exited 0 with no output after the planning edits.

## Remaining sub-slices: docs planning

These entries reserve the final docs steps; code-step plans fill in the page
and topic details. A1–A4 and the older plans below are completed history;
the A5 transition does not reopen them. For each remaining sub-slice, docs
steps start after its last code step merges, cover one page or topic and at
most 300 changed lines each, and pass `scripts/build_docs.sh` with examples
and notebooks executing. The next sub-slice waits for the last docs merge.

### A6 docs

Docs to-do:

- A6.3: parameter_issues/parameter_conditions reference and hook contract,
  component reductions, own/nested ordering and dotted keys, fixed structure,
  and custom-condition/nesting/compiled-checkpoint examples. Explain the
  opposite polarity of constraint results and diagnostic masks.
- A6.2: ParameterIssues fields, ordering and polarity, dotted names, host-only
  raising (including traced rejection), the container reference, and the
  distinction between no reported issue and guaranteed validity. Producer
  examples follow A6.3; A6.2's minimal exception autodoc entry is implemented.
- A6.4: exact/right untruncated likelihoods, function/method and raw/prepared
  forms, unsummed output and shapes/dtypes, issue propagation, zero mass,
  uninformative rows, and the approved unsupported-mode policy. Document the
  approved D-023 mode-support check in NaN triage. Add the method reference
  and executable examples after the public implementation merges. There is
  no likelihood issue container or _with_info variant.

Docs steps:

- A6.d1+: one page or topic per step against merged A6 code; cover all Docs to-do.

### A7 docs

Docs to-do:

- Verified tail/boundary behavior, numerical limits and supported diagnostics;
  fill in any documentation needs found by A7 code/test steps.

Docs steps:

- A7.d1+: one page or topic per step against merged A7 code; cover all Docs to-do.

### A8 docs

Docs to-do:

- Reference-fixture provenance and verified R comparison claims; fill in from
  committed fixtures and A8 check results.

Docs steps:

- A8.d1+: one page or topic per step against merged A8 code; cover all Docs to-do.

### A9 docs

Docs to-do:

- Tiny synthetic example, supported Slice A workflows and limitations, and
  verified installation/transformation behavior; refine during A9 code steps.

Docs steps:

- A9.d1+: one page or topic per step against merged A9 code; cover all Docs to-do
  and record the final Slice A rendered-documentation review.

# A4.5 reviewer fix-up — F-1 and N-1 through N-4

Status: implemented and verified; see REVIEW.md for the successful A4 gate and
D1 mutation evidence. Authorized by the maintainer's request to address
`../review/slices/A4.5.md`.
Continue on `A4.5` after `fe9aa8e`; implement the review checklist in one
fix-up commit, then stop. No public API, protected-file or A5 changes.

- `_survival_methods.py`: rename the guarded logarithm to `_safe_log(x)`;
  remove the default density's duplicate +inf guard and the unreachable
  integer cast, retaining either only if a test result changes; have hazard
  resolve the raw log-hazard method via its marker, honoring overrides.
- `tests/unit/test_family_density.py`: add the quadratic-survival zero-time
  hazard gradients (single time and mixed batch, hand-derived 0 and 2);
  make the override deliberately different from the primitive hazard; count
  reference-time calls for every direct method. Covers spec 4.2/4.5/7.2/7.3.
- Temporarily remove the safe logarithm operand and show that the new F-1
  tests fail with NaN in both dtype modes; restore the implementation before
  final validation. No tolerance or independent-test changes.
- Run the full A4 gate, whole-word stale-name search and git diff checks.
  Overwrite REVIEW.md with findings addressed, corrected evidence, mutation
  results, exact gate output and the unchanged A5/D-016 handoff.
- Commit the review checklist as one fix-up with an imperative subject, then stop.

# A4.5 review fixes — WA naming comments

Status: implemented and verified; see REVIEW.md. Authorized by the maintainer's
request to address the inline WA comments. Continue on `A4.5` after `e2c08af`;
one naming-only fix-up commit, then stop.

- `src/survjax/distributions/_survival_methods.py`: rename the JVP helper to
  `_log_survival_and_derivative`, matching its actual return values. Rename
  `_log_rate` to `_log_hazard`, name its argument and safe operands for hazard,
  and document that it takes the logarithm of a linear hazard value. Update
  both defaults' calls and remove the three addressed WA comments.
- Public API and numerical behavior: unchanged. No new tests; existing density,
  hazard, zero-hazard derivative and transformation cases cover spec 4.2/7.2.
- Run `scripts/check.sh --through A4` with the existing temporary uv cache,
  check for remaining WA/stale helper names and run `git diff --check`.
- Overwrite REVIEW.md with the fixes, invariants and exact results. Commit this
  review fix-up on the same branch under the working agreement, then stop.

# A4.5 — Autodiff density and hazard defaults

Status: implemented and verified on maintainer-selected branch `A4.5`,
starting at `9d1d08d`, against specification revision 21 (2026-10-08).
The maintainer approved implementation and keeping density/hazard together
despite the roughly 375-line source/test/doc diff. See REVIEW.md for exact
evidence. The maintainer authorized the completion commit on 2026-10-08;
commit this step, then stop for review.
This is the active plan; the A4.4 entries below are completed history.

## Outcome and scope

Complete the Slice A distribution base's pointwise methods by adding typed
`SurvivalDistribution.log_prob(t)`, `log_hazard(t)` and `hazard(t)` defaults,
with the section 4.5 policy enforced for defaults and family overrides.
No new top-level exports. Their output shape is the ordinary broadcast of
`t.shape` and `batch_shape`; equal-length vectors pair entries, and singleton
axes request all pairs. These are not shared time-first grid methods.

Covered spec: 3.3, 4.1–4.2, 4.5, 5, 6.2, 7.2–7.3, 8.1 and 12.1–12.2.
Built-in Exponential/Weibull families and survival grids remain A5; likelihoods
and `parameter_issues()` remain A6. The review's D-016 positive-infinity
survival/log-CDF gradient fix stays explicitly deferred to A5. No dependencies,
support API, custom derivative rules, parameter-domain validator or new cache.

## Files and numerical implementation

1. Extend `src/survjax/distributions/_survival_methods.py`; keep numerical
   additions here, as directed by the A4.4 review, rather than growing base.py.
   Add a small typed private helper returning raw log survival and its pointwise
   time derivative using forward-mode `jax.jvp` with an all-ones time tangent.
   Reuse `_raw_log_survival`; differentiate the family primitive on the times
   already substituted by the outer policy, not the policy constants.
   Do not build a full time Jacobian or flatten distribution batch/component
   axes. Cast integer times to the common promoted floating calculation dtype
   before JVP; preserve normal weak typing and time/parameter promotion.
2. `log_prob(t)` uses `log(-d log S/dt) + log S`; it is independent of any
   `log_hazard` override. `log_hazard(t)` uses the derivative directly, avoiding
   cancellation from `log_prob - log_survival` at large cumulative hazard.
   Its generic result is NaN where raw log survival is exactly `-inf`, including
   at finite support ends; finite log survival remains eligible even when
   exp(log survival) underflows to zero. `hazard(t)` is exp(log hazard) and
   honors the family's `log_hazard` override, including its analytic +inf limit.
3. Substitute safe operands before inactive logarithms/additions; use the
   double-where pattern, stopping only placeholders/substituted inputs.
   True zero hazard gives log hazard/log density `-inf` where the formulas
   apply. Invalid negative/NaN derivatives stay NaN; never clip, floor, add an
   epsilon or infer a custom family's parameter validity domain.
4. Extend the existing policy wrapper only as needed for density at +inf.
   Negative times give log density/log hazard `-inf`, hazard zero; NaN times
   give NaN. Preserve the revision-21 NaN precedence at negative/NaN times.
   Zero remains active and obtains boundary values from the primitive or
   override. Enforce `log_prob(+inf)=-inf` for defaults and overrides, including
   formulas that otherwise yield `+inf - inf`. Keep family evaluation at zero
   and +inf consistent with 4.5, and do not change existing survival/CDF policy.
5. `src/survjax/distributions/base.py`: bind the three defaults explicitly and
   add them to registration's policy list. Keep the raw-method marker behavior
   so decorated subclasses preserve overrides without stacking wrappers.
   No construction, PyTree, static metadata or public diagnostic changes.

## Own tests, with the implementation

Create `tests/unit/test_family_density.py` (aim below 200 lines), using small
custom frozen dataclass families and hand-derived expectations. Extend the
existing method-name cases in `test_family_direct.py` only where they give
useful dtype/policy coverage for the three added methods. Use shared value and
existing justified derivative tolerances from `tests/impl_tolerances.py`;
change that file only if a distinct numerical scale needs justification.

- Constant hazard: log density `log r-r*t`, log hazard `log r`, hazard `r`;
  scalar, paired vectors, explicit outer broadcasting, integer times and
  mixed time/parameter dtypes. Exercise jit/vmap, shape-mismatch TypeError,
  and repeat compiled calls with changed parameter values.
- Power survival `log S=-(t/b)**k`: hand-derived log density/hazard across
  decreasing, constant and increasing hazards, zero-time values for k below,
  equal to and above one, and large H with representable hazard. Pin density
  at +inf to -inf and distinguish generic hazards from analytic overrides.
- A bounded-support primitive: generic hazards become NaN at/beyond the
  support end; finite very negative log survival keeps a valid hazard despite
  linear survival underflow. Include a locally zero hazard without log(0)
  contaminating an otherwise active batch's finite derivatives.
- Explicit/inherited density and log-hazard overrides: verify dispatch,
  hazard's use of the log-hazard override, policy at negative/NaN/+inf times,
  no mutation, and no stacked wrappers after subclass registration.
- Unguarded primitive and parameter-dependent reference: negative/NaN time
  slots see safe inputs and retain zero parameter gradients with valid
  parameters. Mixed valid/NaN parameters keep NaNs isolated by entry.
- Smooth positive-time interior: hand-derived gradient and Hessian (constant
  density has `d/dr=1/r-t`, `d2/dr2=-1/r**2`); check first/second parameter
  derivatives forward/reverse, composed with jit/vmap. Run order-two
  `check_grads` with the established tolerances. Do not promise parameter
  derivatives at invalid inputs, support changes or infinite log densities.

## Documentation, verification and stopping point

- Update `docs/distributions.md` with an executable concrete density/hazard
  example, the complete negative-time policy, generic zero-survival behavior,
  override dispatch, and the representable-linear-hazard limitation of the
  autodiff defaults. Update `docs/index.md` to reflect the completed methods.
- Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Run the full maintainer gate
  `scripts/check.sh --through A4`: Ruff, mypy, own tests and all independent A4
  tests in separate x64/x32 processes. The former focused subset is no longer
  sufficient; this step completes the remaining A4 acceptance surface.
- Run combined own/independent sessions with `SPEC_THROUGH=A4` in both modes
  to catch collection/import conflicts, source doctests on base.py and
  _survival_methods.py in both modes, executable docs in both modes, and
  `git diff --check`. No network, R fixture generation or wheel work here.
- Keep spec, acceptance tests, gate scripts and alignment markers read-only.
  On a genuine contract ambiguity or apparent acceptance-test error, record
  an open deviation and ask rather than silently choosing or editing tests.
- Aim for roughly 300 changed source/test/doc lines. If it grows materially
  beyond that, propose a split before expanding the implementation.
- Overwrite REVIEW.md with reading order, numerical/dispatch invariants,
  spec-to-test mapping, exact results/skips and D-016's A5 handoff. Update this
  status, then stop for review. Do not begin A5 or commit until instructed.

# A4.4 completion — specification revision 21 (2026-10-08)

Status: implemented and verified; the maintainer authorized the completion
commit on 2026-10-08. See REVIEW.md for current results and the A5 handoff.

Approved by the maintainer's request to finish A4.4 after spec/acceptance updates.
This continues the existing approved step on `A4.4` at `d0b5538`; no A4.5 work.
The earlier implementation was committed by the maintainer as `8326b45`.
Two new untracked files under tests/spec are read-only inputs; leave them alone.
The maintainer's 2026-10-08 commit request supersedes the earlier hold.

- Rename `distributions/_direct.py` to `_survival_methods.py` and update the
  base import/bindings/wrapping calls, as directed by review A4.4 item 3.
- In that module, reject nonreal/non-numeric times with field-independent
  TypeError using dtype only, including under jit/vmap. Preserve JAX conversion
  and promotion without float64-time warnings (spec 4.5 revision 21).
- Keep numerical NaN precedence; replace its D-013 comment with the now-settled
  spec 4.5 citation. Consolidate repeated raw-primitive lookups into the
  review-requested typed `_raw_log_survival` helper. No public API additions.
- Add own dtype rejection/transformation and no-warning/promotion tests in
  `tests/unit/test_family_direct.py`, plus the exact hand-derived time derivative
  at zero (`-rate = -2`). Keep existing tolerances unchanged.
- Add the review's D-016/D-017/D-018 entries to DEVIATIONS.md. D-016's fix is
  explicitly deferred to A5. D-017/D-018 now have normative revision-21 text;
  annotate that evidence but leave all statuses open for maintainer closure.
- Update docs/distributions.md for direct-time dtypes, NaN precedence and
  finite-invalid-parameter behavior. Update the active plan/status and overwrite
  REVIEW.md with current invariants and exact evidence. No protected edits.

Coverage: spec 3.3, 4.2, 4.5, 5, 6.2, 7.2–7.3, 12.1. Validation: through-A3
maintainer gate; prior 112 A4 nodes plus new acceptance nodes using only the
four A4.4 methods, in both dtype modes; combined own/spec through-A3 sessions;
docs build and source doctests in both modes; git diff --check. Do not run or
claim the full A4 gate until A4.5. Keep this fix-up under roughly 300 lines.
Commit this completion as authorized, then stop before A4.5.

# A4.4 — Direct-method policy and survival/CDF defaults

Historical plan: approved by the maintainer ("go for it and implement") and
implemented on `A4.4`, starting at `6dc4747` against specification revision 19.
The maintainer committed that implementation as `8326b45`. The revision-21
completion above is the active plan; this and earlier plans are history.

## Outcome and public API

Registered custom families gain `survival(t)`, `log_cdf(t)` and
`cumulative_hazard(t)` defaults. Registration enforces the section 4.5 time
policy for these methods and `log_survival(t)`, including family overrides.
The primitive family implementation sees only nonnegative, non-NaN times;
zero and positive infinity still reach it. Pointwise shapes follow ordinary
trailing-axis broadcasting, with TypeError for static time/batch mismatches.

No new top-level exports. Density and hazard defaults/policy stay A4.5;
built-ins and time-first grids stay A5. D-013 remains an A6 handoff: do not
invent a parameter diagnostic contract or custom parameter validity domain.

## Files and implementation

- Create `src/survjax/distributions/_direct.py`: typed ordinary functions for
  the three generic defaults and a single registration-time policy wrapper.
  Keep this numerical responsibility separate from construction/PyTrees.
  Use TYPE_CHECKING imports for the base type to avoid import cycles.
- `src/survjax/distributions/base.py`: explicitly bind the three default
  functions as base methods and integrate policy wrapping at registration.
  Preserve the concrete nested-family docstring and all A4.3 behavior.
  Avoid a large mechanical registration refactor just to add these methods.
- Defaults are `exp(log_survival(t))`, `-log_survival(t)`, and
  `log(-expm1(log_survival(t)))`. Guard zero CDF with safe inputs before the
  logarithm, then return -inf for true zero; no floors, epsilons or clipping.
  NaNs remain NaNs. Document the generic CDF lower-tail underflow limit.
- The policy wrapper first broadcasts times against `batch_shape`. For
  negative/NaN times, substitute the family's reference time before executing
  family code and stop gradients only through the substituted input. Apply
  negative-time constants (0, 1, -inf, 0 respectively) and NaN-time outputs
  afterwards. Preserve NaN family outputs; do not hide invalid parameters
  behind a negative-time constant or add host-side parameter inspection.
- Wrap both inherited defaults and overrides of the four in-scope methods.
  Retain raw implementations through a transparent wrapping convention, so
  registration of explicitly decorated subclasses does not stack policy
  wrappers or introduce recursion. Cover inherited overrides in own tests.
  This is registration-time setup, not a mutable evaluation cache.
- `tests/unit/test_family_direct.py`: independent custom primitive and override
  families, one behavior per test. Extend `tests/impl_tolerances.py` with the
  mandated float32/float64 value tolerances and any justified tail-specific
  tolerance; retain existing derivative tolerances.
- Add the required reference hook to the existing batching-test Wrapper,
  whose numerical method now goes through the policy.
- `docs/distributions.md` and `docs/index.md`: describe the now-available
  methods, boundary policy and shape convention. Add an executable primitive
  family example with actual output values and positive-time/negative-time
  evaluation; mark density/hazard defaults as later A4 scope.
- Overwrite REVIEW.md after implementation: reading order, raw/policy method
  dispatch invariants, spec-to-test mapping, exact checks/results and limits.

Spec coverage: 4.2, 4.5, 5, 7.1–7.3, 10.2–10.5 and 12.1.

## Tests and evidence

- Hand-derived constant-hazard values: `log S=-r*t`, `S=exp(-r*t)`,
  `H=r*t`, `log F=log(-expm1(-r*t))`. Check scalar, paired vectors and
  explicit `(D,1)` by `(N,)` evaluation; incompatible shapes raise TypeError.
- All four methods: negative infinity/negative/zero/interior/positive infinity/
  NaN time values, eagerly and under jit/vmap, including mixed batches.
  Include invalid parameter NaNs beside valid entries and verify isolation.
- Independently override each derived method and verify dispatch plus policy
  enforcement; register a decorated subclass inheriting an override and
  verify it is not wrapped twice. Methods do not mutate PyTree leaves.
- A deliberately unguarded positive-time primitive verifies safe substituted
  times and finite zero parameter gradients for negative policy values.
  Test parameter-dependent reference times without stopping active parameter
  calculations; test second-order forward/reverse gradients on valid smooth
  interior cases using existing justified derivative tolerances.
- Generic log-CDF tails: tiny representable mass compared in log space against
  high-precision mpmath; zero mass remains -inf. No expected value is derived
  from the implementation under test. Include zero-boundary inputs where
  inactive operations must not produce spurious gradient NaNs.

## Validation and stopping point

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` throughout:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests in separate dtype processes.
2. Repeat the existing 89 focused A4 nodes in x64/x32. Additionally select
   from `test_a4_generic_defaults.py` every test except the seven tests using
   `log_hazard`/`hazard`; from `test_a4_policy_enforcement.py`, select
   `test_unguarded_log_survival_is_zero_for_negative_time`,
   `test_unguarded_survival_is_one_for_negative_time`,
   `test_unguarded_cumulative_hazard_is_zero_for_negative_time`,
   `test_unguarded_log_cdf_is_minus_infinity_for_negative_time`,
   `test_unguarded_infinite_time_log_survival_comes_from_the_family`,
   `test_negative_time_log_survival_has_zero_parameter_gradients` and
   `test_negative_time_survival_has_zero_parameter_gradients`.
   Inspect dependencies and record exact selectors/results in REVIEW.md.
   Do not edit, skip or xfail protected acceptance tests. The full A4 gate
   remains required at A4.5.
3. Combined own/independent sessions with `SPEC_THROUGH=A3`, in both dtype
   modes, preserving the unique implementer tolerance-module name.
4. `scripts/build_docs.sh` (approved kernel-port escalation if needed), plus
   the existing concrete nested-field doctest in both modes.
5. `git diff --check`.

Aim for about 300 changed code/test lines. If implementation requires a
larger refactor or split, propose it before exceeding the approved step.
Do not change dependencies, the specification, protected scripts or tests/spec.
Log newly uncovered spec ambiguities as open deviations and ask before choosing.
The earlier no-commit instruction remains in effect: implement and verify after
approval, leave the result uncommitted, then stop for review.

Planning evidence: inspected branch/status, current source, relevant spec,
existing plans/deviations, docs and read-only generic-default/policy acceptance
sources. No runtime, lint, type or documentation checks ran in this planning
turn. `git diff --check` passed with no output.

# A4.3 review fixes — approved, uncommitted

The maintainer requested implementation of `../review/slices/A4.3.md`.
Preserve the prior concrete docstring change; leave all fixes uncommitted
until further review, overriding the review file's suggested commit.

- `src/survjax/distributions/base.py`: F-1, mark each explicitly registered
  class and reject other nested values with field-named TypeError. Do not
  inherit registration from a parent. F-2, use the short tree-map indexing
  form and a spec 4.1 shared-leading-batch-rank comment; delegate index errors
  to JAX. Wrap the long declaration error (N-3). No new public API.
- `tests/unit/test_family_batching.py`: test array/None/unregistered nested
  values and a subclass that inherits a registered parent's marker; update
  the derivative-tolerance import (N-1). Existing indexing tests cover F-2.
- Rename `tests/tolerances.py` to `tests/impl_tolerances.py` as decided by the
  maintainer (N-1). Tighten float32 derivative tolerances to 1e-3 and justify
  the rounding scale (N-2). Do not change the independent tolerance module.
- `docs/index.md`: wrap the flagged line. `docs/distributions.md`: document
  JAX's out-of-bounds integer clamping, as decided by the maintainer.
- Record the review-directed nested-field TypeError policy as open D-015 in
  DEVIATIONS.md; no approval question is needed because the maintainer
  explicitly requested these fixes. Overwrite REVIEW.md with current reading
  order, invariants and exact check results. Spec coverage: 3.3, 4.1–4.2, 5, 6.2, 7.3, 10.5, 12.1.

Run `scripts/check.sh --through A3`, the previous 89 focused A4 nodes in both
modes, `scripts/build_docs.sh`, combined `SPEC_THROUGH=A3` unit/spec pytest
sessions in both modes, the concrete docstring doctest in both modes, and
`git diff --check`. No new dependencies, spec or protected-file edits.

# A4.3 review follow-up — concrete nested-field example

Approved by the maintainer's request to do this edit and leave it uncommitted.
Change only the `distribution()` docstring in
`src/survjax/distributions/base.py`, plus this plan and the review note.
Show complete custom constant-hazard and scaled-hazard families, then construct
and index an actual nested instance with numerical outputs. Use the existing
A4.3 API; no built-in families, numerical defaults or new public API.
Covers spec 4.2, 5 and 10.5. Execute the docstring example in both dtype modes,
run focused Ruff and mypy checks, and record results in REVIEW.md.
No new test file or commit; stop for further review.

# A4.3 — Nested distributions and batch indexing (2026-10-02)

Status: approved by the maintainer ("ok implement it") and implemented on
`A4.3`; see REVIEW.md for evidence. This is the active step plan; earlier plans below are history. Work on the maintainer-created
`A4.3` branch, starting at `138db08` (A4.2 review fix-up), against specification
revision 19. The working tree was clean before this plan update.

## Outcome and public API

Registered frozen families can declare a required nested distribution with
`survjax.distribution()` and select batch entries with `dist[idx]`. Own
parameters and all nested distributions share the outer batch shape after
construction. Nested component axes, dtypes, family types and static
configuration survive broadcasting, indexing and JAX transformations.

Export `distribution()` from both `survjax` and `survjax.distributions`.
Add typed `SurvivalDistribution.__getitem__`; extend the existing
`batch_shape` property and `register_family` behavior. No other public names.
Spec coverage: 3.3 (independent nested dtypes), 4.1–4.2 (representation,
registration and indexing), 5 (batch shapes), 7.2–7.3 (compilation reuse and
parameter transformations), 10.2–10.5 and 12.1 (engineering/docs/tests).

## Files and implementation

- `src/survjax/distributions/base.py`: add the no-argument, no-default
  `distribution()` dataclass field helper. Recognize nested fields alongside
  `param()` and `static()` in declaration checks and document all three roles.
  Include nested objects as dynamic children, so their leaves and static
  auxiliary data participate in the outer PyTree. Reconstruction continues
  to bypass public constructors, normalization and value inspection.
- Extend construction to broadcast own parameter batch shapes together with
  nested `batch_shape` values. Normalize only own parameters, using the existing
  helper and preserving warning attribution. Recursively broadcast nested
  parameter leaves to the common batch shape plus their declared component
  shapes, retaining their original dtypes and static values. Rebuild nested
  objects without public constructors; leave the supplied baseline immutable.
  Incompatible batch shapes raise TypeError. Support wrappers whose dynamic
  fields are exclusively nested distributions.
- Derive `batch_shape` from current own and nested leaves, excluding component
  axes. Store no cached batch shape or extra static shape metadata.
- Implement batch-only indexing for integers, slices, integer arrays and
  tuples, including ellipsis and inserted singleton axes. Resolve indexing
  against the batch rank before appending untouched component-axis slices;
  ellipsis must never consume component axes. Recursively index nested
  distributions and reconstruct without constructors. Follow ordinary JAX
  indexing constraints, including traced integer indices under jit and the
  restrictions on dynamic-size indexing. Do not add custom bounds semantics.
- Keep `base.py` under about 300 lines. Only if necessary, move registration
  internals to `src/survjax/distributions/_registration.py`, preserving public
  imports and caller-site warning behavior; no general helper framework.
- `src/survjax/distributions/__init__.py`, `src/survjax/__init__.py` and
  `tests/unit/test_import.py`: add and check the new export.
- Create `tests/unit/test_family_batching.py` with independently defined custom
  families and small behavior-specific tests. Use existing tolerances;
  derivative-specific additions belong only in `tests/tolerances.py` with a
  scale justification.
- `docs/distributions.md`: add executable nested-field and batch-selection
  examples, explain component axes and independent nested dtypes, and show
  paired `(N,)` versus explicit `(D,1)` evaluation using a simple custom
  `log_survival` implementation at valid positive times.
  `docs/index.md`: update the current-feature description.
- Overwrite `REVIEW.md` after implementation with reading order, invariants,
  spec-to-test mapping, exact checks/results/skips and remaining questions.

## Tests and evidence

- Spec 4.2: helper takes no arguments; nested fields are required; round trips
  retain nested types, static configuration and supplied sentinel leaf identity.
- Spec 4.1–4.2/5: scalar baseline/vector wrapper, vector baseline/scalar wrapper,
  `(2,1)` with `(3,)`, nested-only wrapper and at least two nesting levels;
  incompatible batches fail. Verify stored leaf shapes and explicit values,
  nested components (including two component axes), and unchanged input objects.
- Spec 3.3: x64-only mixed float32/float64 nesting preserves each family's dtype
  through broadcasting and indexing; x32 covers available dtypes without
  changing global configuration or claiming float64 preservation.
- Spec 4.1–4.2: integer, slice, array and multi-axis tuple selection preserves
  components/static fields; ellipsis and singleton insertion operate on batch
  axes only. Compare indexed leaves to independently indexed input arrays.
  Tree-map slicing agrees with batch selection and updates outer/nested shapes.
- Spec 7.2–7.3: eager/jit/vmap construction and selection agree; jit supports
  a traced integer index; vmap sees per-entry nested batch shapes and full
  component vectors. Compile-count tests vary outer and nested dynamic values
  at fixed shapes/dtypes/structure and require reuse; changed nested static
  configuration specializes separately. Use a smooth custom log-survival
  formula with hand-derived derivatives and second-order forward/reverse
  `check_grads` to verify gradients through nested construction and selection.

Run with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests, separately in both dtype modes.
2. Focused read-only A4 acceptance nodes with `JAX_ENABLE_X64=1` and `0`,
   `SPEC_THROUGH=A4`, and `uv run --locked --offline pytest -q`:
   - `test_a4_registration.py`: all except `test_methods_do_not_mutate_parameters`.
   - `test_a4_parameters.py`: all except
     `test_construction_with_traced_parameters_inside_jit`,
     `test_construction_inside_vmap` and `test_gradient_through_construction`.
   - `test_a4_indexing.py`: all except
     `test_indexed_distribution_matches_indexed_values`,
     `test_vmap_over_distribution_matches_batched_call` and
     `test_vmap_over_distribution_and_time_pairs_entries`.
   - `test_a4_nested.py`: all except
     `test_vmap_over_outer_distribution_with_broadcast_base` and
     `test_broadcast_nested_values_in_log_prob`.
   These excluded nodes require later generic numerical methods. Record the
   exact node list and results in REVIEW.md; do not edit, skip or xfail tests.
3. `scripts/build_docs.sh`: execute examples in separate x32/x64 processes and
   inspect the rendered guide. Report any environment/permission limitations.
4. `git diff --check`.

## Boundaries and stopping point

Direct-method policy/defaults remain A4.4–A4.5; built-ins, grids and actual
PH/AFT wrappers remain later steps. D-013 remains an A6 diagnostic-contract
handoff. Do not change dependencies, the spec, independent acceptance tests
or protected scripts. Log any newly uncovered spec ambiguity as an open
DEVIATIONS.md entry and ask before choosing its behavior.

Aim for roughly 300 changed lines including code and tests. If the necessary
implementation grows past that, propose a smaller split before expanding it.
After approval, implement one concept on `A4.3`, write REVIEW.md, commit with
an imperative subject and a body explaining why, then stop for review.
This plan-only update remains uncommitted.

Planning evidence: read the relevant spec, A4.2 implementation/review,
dtype helper, docs and read-only A4 acceptance sources. No runtime, lint,
type, derivative or documentation checks were run for this planning update.

# A4.2 review fix-up (2026-10-02)

Status: approved by the maintainer on 2026-10-02 after reading
`../review/slices/A4.2.md`, with public-docstring improvements added. One extra commit on `A4.2`, then stop.
No A4.3 implementation or documentation-page rewrite.

## Changes

- `src/survjax/distributions/base.py`, F-1: reject dataclass fields whose
  metadata does not identify an available field role (`param` or `static`),
  with a field-named TypeError at registration. Record the unspecified plain
  field policy in D-014; the maintainer approved this implementation policy.
- N-1: validate `component_ndim` in `param()`: a non-negative Python int,
  excluding bool. Use TypeError for wrong kinds and ValueError for negatives.
- N-2: require an initializer owned by the class before wrapping it. Reject
  inherited initializers with a TypeError explaining that the subclass must
  be decorated with `@dataclass` itself. This prevents double normalization
  and incorrect warning attribution; retain the documented decorator recipe.
- N-3: catch static hashability TypeError and re-raise naming the field and
  stating that static fields must be hashable.
- Rewrite the public field helpers, base, batch-shape property, extension hooks
  and registration docstrings for users: purpose, arguments, shapes, errors and
  one short declaration example. Keep spec/development bookkeeping out of them.
- `tests/unit/test_family_registration.py`: add separate tests for plain
  default/default-factory field rejection; invalid component counts; inherited
  initializer rejection and a re-decorated subclass's x32 warning location;
  field-named static hashability errors; insufficient component rank and
  non-frozen registration guards. Keep tests independent of tests/spec.
- `DEVIATIONS.md`: add open D-014. Do not change existing statuses or spec.
- `REVIEW.md`: overwrite with reading order, requirement coverage, exact
  check outcomes and remaining scope; correct the earlier claim that no
  registration ambiguity existed.

Public names stay unchanged. These are declaration/error checks for the
extension interface (spec 3.3, 4.1–4.2, 5); no parameter-value checks or
numerical-method work. Aim below 300 changed lines including tests.

## Verification

Run `scripts/check.sh --through A3`, the same 60 focused A4 acceptance nodes
listed in the prior REVIEW.md in both dtype processes, `scripts/build_docs.sh`
in both modes, and `git diff --check`. Use
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache`. Record dtype-specific skips and
any failures. Commit this fix-up only after checks pass, then stop.

Planning checks: read the external review and current implementation;
working tree was clean before this plan/deviation update. Runtime checks
were not rerun for this plan-only task.

# A4.2 — Registration and flat family construction (2026-10-02)

Status: approved by the maintainer ("go for it and build it") and implemented
on the existing `A4.2` branch; see REVIEW.md for verification. This is the active step plan; the broader A4 sequence below remains
background. Based on specification revision 19 and completed A4.1 at
`15cfc6f` (current HEAD also contains the D-013 handoff note).

## Outcome and scope

A custom frozen dataclass can be explicitly registered, constructed by keyword
with normalized and broadcast parameter arrays, and passed through JAX as an
immutable PyTree. This step covers spec 3.3, 4.1–4.2, construction shapes in
section 5, and engineering/documentation requirements in 10.2–10.5.

Public additions: `SurvivalDistribution`, `register_family`,
`param(component_ndim=0)` and `static(...)`, exported from both `survjax` and
`survjax.distributions`. The base exposes typed `batch_shape` and the typed
`log_survival(t)` / `reference_time()` extension hooks. Hooks without family
implementations fail explicitly; no numerical defaults are supplied yet.
No base `shape` attribute: a family's own `shape` parameter remains usable.

Nested `distribution()` fields and batch indexing stay in A4.3. Direct-method
policy/defaults stay in A4.4–A4.5; built-ins and grids stay in A5. D-013 stays
open for A6. The guide will state these intermediate limitations explicitly.
No dependency, spec, protected-script or independent-test changes.

## Files and implementation

- Create `src/survjax/distributions/base.py` and its package `__init__.py`.
  Use dataclass field metadata to distinguish parameters from static fields,
  ordinary functions and explicit `jax.tree_util` registration. Require the
  documented frozen-dataclass subclass recipe; no automatic registration,
  metaclasses or dynamic attribute lookup. Keep the module under about 300 lines.
- Wrap the dataclass-generated constructor to reject positional arguments,
  preserving keyword argument errors, required parameters and static defaults.
  Support `kw_only=True`, including a defaulted static field before parameters.
  Static configuration must be hashable. Do not add parameter-value validation.
- Reuse `_dtypes.normalize_parameters` for all own parameters, including
  components. Pass the stack level needed for warnings to point to the user's
  constructor call. Constructors accept no `dtype` keyword.
- Derive each parameter's batch axes by excluding its declared trailing
  component axes. Broadcast those batch shapes, then broadcast each array to
  the common batch shape plus its original component shape. Translate
  incompatible batch shapes into `TypeError`. Never broadcast component axes
  against batch axes or inspect parameter values on the host.
- Flatten arrays as dynamic children, with static configuration in auxiliary
  data. Unflatten using allocation and frozen-field assignment, bypassing the
  constructor, normalization and broadcasting; supplied leaves retain identity,
  including JAX placeholders. Compute `batch_shape` from current leaves rather
  than store cached shape metadata, so tree mapping and vmap update it naturally.
- Update `src/survjax/__init__.py` and `tests/unit/test_import.py` for the four
  new names; preserve existing exports.
- Add `tests/unit/test_family_registration.py` with small independent custom
  families. Do not import acceptance-suite recipes or helpers.
- Create `docs/distributions.md` with an executable construction/PyTree example,
  decorator order, keyword-only recipe, static defaults, component axes, dtype
  behavior and absent `shape` alias. Add it to `docs/index.md` and update that
  page's current-feature description. Use supported construction operations;
  label numerical defaults, nested fields and indexing as planned.

If registration edge cases require choices the specification does not settle,
record a new open deviation and ask before implementing that behavior. If code,
unit tests and docs exceed roughly 300 changed lines, propose a further split
before expanding the step; approval here does not authorize a larger step.

## Tests and evidence

- Spec 4.1–4.2: immutability, explicit registration only, required/keyword-only
  arguments, static defaults/overrides and the static-first kw-only recipe.
- Spec 3.3: public-constructor dtype integration, components sharing dtype,
  rejected nonreal inputs and None, unavailable-float64 warning count and caller
  location, and rejection of `dtype=`. Reuse A4.1's detailed promotion tests.
- Spec 4.1–4.2/5: scalar and multi-axis broadcasting, component preservation,
  incompatible batch shapes, stored negative/NaN/infinite natural parameters,
  and a family with its own `shape` field alongside a family without one.
- Spec 4.2/7.2: round trips, static tree-structure equality/difference, sentinel
  identity on unflatten, tree-map slicing, eager/jit construction, and vmap
  construction/per-entry `batch_shape`. Use exact shape/value assertions where
  applicable; no derivative or direct-method policy claims in this step.

Run with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`:

1. `scripts/check.sh --through A3`: formatting, lint, mypy, all own tests and
   prior independent tests, in separate dtype processes where applicable.
2. Focused A4 acceptance nodes in separate processes with
   `JAX_ENABLE_X64=1` and `0`, `SPEC_THROUGH=A4`, and
   `uv run --locked --offline pytest -q`. Select registration tests except
   the all-extension-names, method-purity and three nested-distribution tests;
   select parameter tests except `test_nested_distribution_keeps_its_own_dtype`,
   `test_construction_with_traced_parameters_inside_jit`,
   `test_construction_inside_vmap` and `test_gradient_through_construction`
   (these three use the later generic `log_prob`);
   select `test_a4_indexing.py::test_tree_map_slicing_updates_batch_shape` and
   `::test_batch_shape_inside_vmap_is_the_per_entry_shape`. Inspect dependencies
   before execution and record the exact node selection in REVIEW.md. Never
   skip/xfail or edit unfinished acceptance tests, or claim the full A4 gate.
3. `scripts/build_docs.sh`: execute examples in x32/x64 and inspect the rendered
   distribution guide. Report any permission or environment limitation honestly.
4. `git diff --check`.

Overwrite `REVIEW.md` with reading order, construction/PyTree invariants,
requirement-to-test mapping, exact commands/results/skips and open questions.
Commit this one implementation step on `A4.2` with an imperative subject and
why in the body, then stop. The plan-only update remains uncommitted.

Planning checks: read the spec, existing implementation/docs, A4.1 review and
read-only A4 acceptance sources. Working tree was clean on `A4.2`. No runtime,
lint, type or documentation checks were run for this planning update.
`git diff --check` passed with no output.

# A4.1 review fix-up (2026-10-02)

Approved by the maintainer's request to implement `../review/slices/A4.1.md`.
One fix-up commit on `A4.1`, then stop; no A4.2 implementation.

- F-1: reject explicit None own parameter leaves with a field-named TypeError
  in `_dtypes.py`; add the case to `test_parameter_dtype.py` (spec 3.3, 4.2).
- N-1: include a float32 leaf in the x32 warning test, so a single float64
  leaf must still warn (spec 3.3).
- N-2: pass the warning noun explicitly from `_data_validation.py` and
  `normalize_parameters` rather than derive it from input-kind strings.
- Public API unchanged. Preserve observation errors and warning attribution.
- Run `scripts/check.sh --through A3`, `scripts/build_docs.sh` in both dtype
  processes, and `git diff --check`; overwrite REVIEW.md with results.
- Remove the obsolete D-001 template as requested in review bookkeeping.
  Preserve the existing unstaged DEVIATIONS.md planning edit; no deviation
  statuses are changed by this fix-up.

# Step A4: distribution base and extension interface (2026-10-02)

Status: A4.1 approved by the maintainer on 2026-10-02; later parts await approval. Based on specification revision
19; re-evaluated against the synced A4 acceptance suite at source commit
`f62c81ff5b2317ecf692e546a2d3beeafd5c683f`. A4.1 implementation is on the maintainer-created `A4.1` branch. Earlier plans below
are history; `REVIEW.md` continues to describe the completed A3 work.

## Scope and decisions needed

Implement the Slice A distribution foundation: frozen registered families,
parameter/static/nested fields, construction and reconstruction, batch indexing,
and the seven pointwise direct methods. Cover sections 3.3, 4.1, 4.2, 4.5, 5,
7.2–7.3, 10.2–10.5 and the relevant parts of 12.1–12.2.

New top-level exports: `SurvivalDistribution`, `register_family`, `param`,
`static`, `distribution`. Existing exports remain. Methods added by completion:
`batch_shape`, `__getitem__`, `reference_time`,
`log_survival`, `survival`, `log_prob`, `log_cdf`, `cumulative_hazard`,
`log_hazard`, `hazard`. `log_survival` and `reference_time` are the required
family primitives. There is no base `shape` attribute and no constructor
`dtype` keyword. Signatures, public fields and extension interfaces are typed.

The nine A4 acceptance files are now present: 203 tests collect in each dtype
mode. Read-only `API_ASSUMPTIONS.md` Q-019–Q-025 and `TESTMAP.md` confirm the
scope below. In particular, Q-024 explicitly leaves `parameter_issues()` and
its unspecified return type for A6. Remove that API and generic nested-issue
aggregation from A4; D-013 remains open as an A6 handoff, not an A4 blocker.
Do not infer domain validity for arbitrary custom parameters. A4 preserves
NaNs produced by family formulas and generic defaults; its acceptance examples
exercise a negative rate through `log_prob` and NaN parameters through direct
methods. Passing them is not evidence for general parameter diagnostics.

Built-in Exponential/Weibull families, log-parameter constructors and survival
grids remain A5. Likelihoods/result containers remain A6. Interval methods,
wrappers, summaries, sampling, `HazardFamily` and `check_family` remain their
later slices. Use small analytical custom families in tests and documentation
to exercise A4 without introducing built-ins early.

## Review-sized implementation sequence

A4 needs multiple commits to respect the roughly 300-line review target,
including tests. Each numbered part below is a separate step: confirm its
scope in this plan, obtain approval, implement with tests, overwrite `REVIEW.md`,
commit on the maintainer's branch, then stop. Approval to begin A4.1 does not
authorize automatically continuing through the remaining parts. If a part
exceeds the target, propose a further split before expanding it.

### A4.1 — Shared floating-dtype normalization

- Files: new `src/survjax/_dtypes.py`; update `_data_validation.py`; add
  `tests/unit/test_parameter_dtype.py`. Reuse the existing dtype logic by a
  narrow extraction rather than coupling distributions to observation rules.
  Preserve observation behavior, error meaning and warning attribution.
- Normalize each family's own parameters to one dtype, including components;
  nested distributions retain their dtype. Preserve weak promotion, the
  configured default for integer/scalar-only inputs, and the float32 minimum.
  Reject bool, complex, string and object inputs. Warn once per user call site
  for float64 arrays under x32; do not warn for Python scalars/lists. Never
  inspect traced values on the host or change global JAX configuration.
- Tests: dtype combinations, float16 promotion, weak inputs, rejected input
  kinds, x32 warning count/location and traced normalization. Existing data
  dtype tests protect the extraction. Public API unchanged (spec 3.3).
- Docs: explain the shared policy in `docs/survival-data.md`; mark distribution
  examples as planned until registration exists.

### A4.2 — Registration and flat family construction

- Files: new `src/survjax/distributions/base.py` and package `__init__.py`;
  update top-level `__init__.py` and `tests/unit/test_import.py`; add
  `tests/unit/test_family_registration.py`.
- Add the base, `param(component_ndim=0)`, `static(...)` and explicit
  `register_family`. Use dataclass field metadata and small ordinary functions;
  no metaclasses, dynamic attribute lookup or automatic subclass registration.
- Enforce keyword-only construction, required parameter fields and optional
  static defaults. Normalize parameter dtypes; broadcast leading batch axes
  while retaining trailing component axes. Incompatible batch shapes raise
  `TypeError`. Construction does not validate parameter values.
- Register dynamic arrays as children and hashable static configuration as
  auxiliary data. Reconstruct using allocation/field assignment, bypassing
  construction and normalization, including for JAX placeholders.
  Derive `batch_shape` from current leaves, never cached shape metadata.
- Tests: immutability; required/keyword-only arguments; static defaults;
  component and batch shapes; dtype integration; eager/jit construction;
  flatten/unflatten and tree mapping; no implicit registration; absent base
  `shape`. Explicitly test the `kw_only=True` recipe with a defaulted static
  field before required parameters, unchanged object identity for unflattened
  sentinel leaves, and storage of negative/NaN/infinite parameter values.
  Test malformed static structure without checking parameter values.
- Docs: add `docs/distributions.md` to `docs/index.md`, showing the decorator
  order, field roles and construction; label methods not yet implemented.
  Spec: 3.3, 4.1–4.2, 5, 10.2–10.5.

### A4.3 — Nested distributions and batch indexing

- Files: extend `distributions/base.py`; add a focused private
  `distributions/_registration.py` if needed to keep responsibilities and
  modules small; export `distribution()`; update import tests; add
  `tests/unit/test_family_batching.py`.
- Include nested batch shapes in outer broadcasting. Broadcast nested leaves
  recursively, preserving their component axes, own dtypes and static fields.
  The no-argument `distribution()` field is a required nested distribution.
- Implement batch-only indexing, applying indices to leading axes while
  retaining component axes. Rebuild without public constructors. Support
  scalar/slice/array/tuple indexing within ordinary JAX indexing constraints.
- Tests: nested scalar/vector/multi-axis broadcasting; incompatible shapes;
  nested component axes and mixed dtypes; indexing, tree-map slicing and
  vmap agreement; per-entry `batch_shape` inside vmap. Confirm changed dynamic
  values reuse compilation with equal shapes/dtypes/structure/configuration,
  and changed static configuration may specialize separately.
- Docs: expand `docs/distributions.md` with nested fields and batching;
  distinguish paired `(N,)` evaluation from explicit `(D,1)` broadcasting.
  Spec: 3.3, 4.1–4.2, 5, 7.2, 12.1.

### A4.4 — Direct-method policy and survival/CDF defaults

- Files: `distributions/base.py`,
  registration integration, new private `distributions/_direct.py` for the
  shared value-policy machinery; `tests/unit/test_family_direct.py`; create
  `tests/tolerances.py` for numerical tolerances with the mandated scales.
- Implement `survival = exp(log_survival)`,
  `log_cdf = log(-expm1(log_survival))`, and
  `cumulative_hazard = -log_survival`.
  Keep the log-CDF calculation stable without probability floors.
- Apply section 4.5 at explicit registration to both inherited defaults and
  family overrides. Use one understandable policy wrapper, preserving raw
  family implementations for internal composition and avoiding recursion.
  Static time/batch mismatch raises `TypeError`; numerical time values follow
  the total-function policy. Do not confuse this with observation validation.
- For negative/NaN times, substitute a safe stopped-gradient reference before
  invoking family code; select policy outputs afterwards. Pass zero and positive
  infinity to the family. Preserve NaNs from family calculations without
  inventing generic domain checks for unknown custom parameters. Retain
  `reference_time()` as the required family hook; our choice to use it for
  safe negative/NaN-time substitution needs our own tests because the A4
  acceptance suite does not test that hook directly.
- Tests: analytical exponential-survival recipe; scalar/paired/draw broadcasting;
  all time-policy rows for these four methods, defaults and overrides; NaN
  parameter propagation;
  family code never receives negative/NaN times; finite parameter gradients
  through inactive branches and reference-input isolation. Test log-CDF tails
  against analytical/mpmath values in representable ranges.
- Docs: runnable primitive-family example in `docs/distributions.md`, direct
  method boundaries, dtype behavior and precision limits of the CDF default.
  Spec: 4.2, 4.5, 5, 7.2, 12.1.

### A4.5 — Autodiff density/hazard and A4 completion

- Files: `distributions/base.py`, `_direct.py`; new
  `tests/unit/test_family_autodiff.py`; extend `tests/tolerances.py` with
  justified derivative scales; finish `docs/distributions.md` and update
  `docs/playground.ipynb` with a short custom-family example.
- Compute the elementwise time derivative of log survival with forward-mode
  autodiff after broadcasting, holding parameters fixed. An all-ones time
  tangent yields the pointwise derivative; do not differentiate a batch sum
  or materialize a full Jacobian. Use it for generic log hazard and log density;
  do not recover hazard by subtracting two large log values. Keep the density
  calculation independent of the public `log_hazard` result: its zero-survival
  NaN policy must not contaminate a density that should be `-inf`.
- Supply `hazard = exp(log_hazard)`. Apply the same policy to density/hazard
  overrides. Detect zero survival from `log_survival == -inf`, not from
  `exp(log_survival) == 0`: finite log survival such as `-1000` still gives a
  valid generic hazard. Generic log hazard is NaN at true zero survival,
  including finite support endpoints; analytic overrides retain their values,
  and generic `hazard` uses the `log_hazard` override. Enforce
  `log_prob(+inf)=-inf` for valid
  parameters. Substitute safe inputs before logarithms and other potentially
  invalid inactive operations; use no clipping, epsilon or probability floor.
- Tests: hand-derived density/hazard for exponential and a smooth nonlinear
  cumulative-hazard recipe; bounded-support and infinite-time generic/override
  distinctions; zero hazard; eager/jit/vmap agreement; nested transformed
  construction. Cover zero-time density values for power-family shapes below,
  equal to and above one; negative-rate density NaNs and isolation of invalid
  batch entries; large cumulative hazards and tiny positive survival.
  Use `check_grads(order=2, modes=("fwd", "rev"))` on smooth parameter
  interiors in float64, with an explicit float32 skip for noisy second-order
  finite differences. Check analytical first derivatives and Hessians in both
  modes, including jit/vmap composition. Test finite gradients for the base's
  negative-time substitutions and use boundary-safe custom recipes for any
  additional boundary-gradient checks; do not promise that the base repairs
  arbitrary custom derivatives at zero or infinity. Built-in Weibull's `S(0)`
  shape-gradient requirement remains A5. Expected values come from algebra or mpmath,
  never from the implementation being tested. State the extra derivative order
  required of a primitive when density comes from autodiff.
- Finish the seven-method policy matrix and run the synced independent A4
  suite. Examples explain the extension contract and generic numerical limits;
  keep built-in grids, likelihood debug-NaN batches, R fixtures, wheel tests and
  performance baselines assigned to their later Slice A steps.
  Spec: 4.2, 4.5, 7.2–7.3, 10.5, 12.1–12.2.

## Acceptance coverage and checks at each stop

All paths below are under read-only `tests/spec/`. Keep implementer tests
independent: do not import `spec_families.py`, helpers or tolerances from there.

| Acceptance file | Planned coverage |
| --- | --- |
| `test_a4_registration.py` (25) | A4.2 registration, constructors and PyTrees; A4.3 nested fields; method-purity checks complete with A4.5 |
| `test_a4_parameters.py` (42) | A4.1 dtype helper, A4.2 public integration, A4.3 nested dtypes; tests using default density complete with A4.5 |
| `test_a4_indexing.py` (17) | A4.3 indexing and current-leaf batch shapes; density comparisons complete with A4.5 |
| `test_a4_nested.py` (14) | A4.3 broadcasting, components, dtypes and structure; density/vmap comparisons complete with A4.5 |
| `test_a4_generic_defaults.py` (23) | A4.4 survival/CDF/cumulative hazard; A4.5 hazard defaults |
| `test_a4_policy_enforcement.py` (20) | A4.4 primitive and derived policy; A4.5 density/hazard, overrides and inactive gradients |
| `test_a4_log_prob_default.py` (31) | A4.5 autodiff, shape/dtype, dispatch, gradients and Hessians |
| `test_a4_value_policy.py` (19) | A4.5 zero/negative/infinite/NaN times, invalid density and elementwise isolation |
| `test_a4_hazard_at_zero_survival.py` (12) | A4.5 true zero versus underflow, bounded support and analytic override dispatch |

These files cross substep boundaries. Use relevant explicit node selections
for intermediate feedback when their dependencies exist, in both dtype
processes, recording the selection; do not claim an entire file passes early,
change its step markers, or skip/xfail unfinished tests. The complete A4 gate
is required after A4.5.

Use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` for uv-based commands.
For A4.1–A4.4 run `scripts/check.sh --through A3` (all implementer tests
including the new ones, plus prior acceptance tests) and
`scripts/build_docs.sh`, both using separate x32/x64 processes. At A4.5 run
`scripts/check.sh --through A4` and the docs build. Run `git diff --check` at
every stop. The gate already includes Ruff formatting/lint and mypy.

Record exact commands, test counts, skips and failures. In particular, report
dtype-specific skips (float64 normalization cases and float64-only finite
differences) separately from failures.
Inspect the rendered distribution guide after its runnable examples are added.
No dependency additions, R execution or spec edits are planned.

Each `REVIEW.md` gives reading order, construction/PyTree/numerical invariants,
tests mapped to spec requirements, actual check output and open questions.
Each commit has an imperative subject and a body explaining why. Stop after
each commit; never commit to `main`. This planning change is left uncommitted.

Planning re-evaluation checks (2026-10-02): with the cache directory above,
`JAX_ENABLE_X64=1 SPEC_THROUGH=A4 uv run --locked --offline pytest --collect-only -qq tests/spec/test_a4*.py`
and the same command with `JAX_ENABLE_X64=0` both exited 0 and collected 203
A4 tests across nine files. Collection does not execute tests or establish
implementation correctness. No implementation tests, type/lint checks or docs
build were run for this plan-only update. `git diff --check` passed.

# A3 review fix-up: constructor consistency and metadata evidence (2026-10-01)

Status: approved and implemented on 2026-10-01; A3 gate passed. Source: `A3-review-handoff.md`,
reviewing `791517a`. One extra commit on the existing `A3` branch, targeting
under 300 changed lines including tests. Follow the handoff's explicit
plan/approval/implementation/review/stop sequence.

## Files and changes

- `src/survjax/_data_metadata.py`: replace recursive tracer detection
  with `jax.tree.leaves`; retain the separate module and existing kind
  precedence. Add the spec 3.3 explanation that `(0, +inf]` is right censored.
  Implement item 8: compute the six presence reductions with `jnp` on the
  device, stack the scalar results, and call `jax.device_get` once on that
  six-element array before converting to Python bools. Avoid transferring
  observation arrays to NumPy. Keep `may_contain` inside the compile-time
  evaluation block on the concrete path only; never pass it tracers.
- `src/survjax/data.py`: extend the bounds constructor's compile-time
  block through all issue calculations, making its six leaves concrete at
  trace time when supplied inputs are concrete. Add the requested JAX
  out-of-range clamping sentence to `__getitem__`'s docstring.
- `tests/unit/test_data_metadata.py`: add explicit scalar `(0, +inf]`
  precedence, NaN truncation and negative-entry flag/issue cases; check all
  six leaves inside jit for both concrete constructor forms; assert both
  forms retain the host-only `ValueError` (not `InvalidDataError`) when
  `raise_if_invalid()` is invoked inside jit; cover conservative metadata
  from vmap construction for both forms; extend concrete-inside-jit tests
  to compare every leaf with eager construction.
- `DEVIATIONS.md`: add D-012, step A3, spec 3.2, status open, documenting
  overlapping kind predicates at `(0, +inf]`, the existing right precedence
  following 3.3, and the proposed clarification for a future spec revision.
- `REVIEW.md`: overwrite with reading order, invariants, requirement/test
  mapping, actual check results and the A6 handoff below.

Public API: signatures and exports stay as implemented; `__getitem__`
documentation gains an explicit description of its existing behavior.
Spec coverage: 3.2, 3.3, 6.2, 12.1. No numerical likelihood implementation.

## Precise review wording and A6 handoff

Presence flags are not validity checks. Preserve the spec 3.2 invariant that
metadata includes normalized rows with issue bits. NaN bounds and negative
entry can leave truncation flags false despite invalid data; this does not
mean all invalid bounds necessarily leave those flags false. Invalidity is
determined from the stored issue bits.

Concrete construction requires one host synchronization to obtain static
Python flags (spec 3.2), transferring only six booleans. Slices inherit the
parent flags and perform no metadata synchronization. Existing metadata tests
and the new precedence/invalid-truncation tests cover item 8's unchanged
behavior; add no dedicated test or timing measurement. Large-N construction
timing belongs with the separate spec 12.3 CPU baseline.

For concrete supplied inputs, both constructors' six stored leaves are
evaluated at compile time inside jit. Subsequent `data.issues` decoding inside
jit is still traced, and its host checkpoint must fail explicitly there.
For traced supplied inputs, operations involving those inputs remain traced
and all six metadata flags are true; do not promise that constant-only
intermediate operations become tracers.

Record for A6:

- Reuse the normalized kind definitions from `_data_metadata.py` so row
  dispatch and static flags agree. `may_contain` returns static Python flags
  after a host transfer, not jit-safe row masks. A6 must plan any shared
  pure-JAX row classification needed for likelihood use; add no such helper
  in this fix-up.
- Issue-bearing rows must produce NaN and `invalid_data` from `_issue_bits`
  regardless of metadata pruning. Test NaN truncation with false flags.
- Resolve how Slice A detects unimplemented modes on traced data: all-true
  metadata alone cannot establish that a row uses an unsupported mode.
  The A6 plan must address row-value detection and explicit failure under
  transformations before implementing likelihoods.

## Verification and finish

Run `scripts/check.sh --through A3` in the existing two dtype processes,
with `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`, and `git diff --check`.
Report actual results and any skips. Keep spec/tests/spec read-only and
exclude the existing notebook edit and maintainer-authored handoff from
staging. Commit as an extra A3 fix-up, without amend/rebase, then stop.

# Step A3: indexing and static observation metadata (2026-10-01)

Status: approved by maintainer and implemented on 2026-10-01; A3 gate passed.
Branch: existing maintainer-created `A3`. One implementation step and commit,
targeting about 300 changed lines including tests. If the implementation needs
a larger change, propose a split before expanding it. Earlier plans below are
history. The existing local edit to `docs/playground.ipynb` belongs to the
maintainer and is excluded from this step.

## Starting point and scope

A2 already supplies both pure constructors, the five normalized fields, the
sixth issue-bitfield leaf, diagnostics, and reconstruction without normalization.
`SurvivalData` currently has no `__getitem__`; its six auxiliary flags always
default to true. A3 completes indexing and value-derived concrete metadata.

Public API touched: add typed `SurvivalData.__getitem__(idx) -> SurvivalData`
and document its observation-axis indexing behavior. Both existing constructors
gain the specified metadata behavior without changing signatures or exports.
`data.issues` continues to decode masks and reduce `ok` over the resulting
object. Metadata remains private. No likelihood evaluation or branch pruning
is introduced; those belong to later steps/slices.

Spec coverage: 3.2 (indexing, six aligned leaves, auxiliary flags and
reconstruction), 6.2 (input tracing classification and indexed diagnostics),
12.1 (metadata, stored issues and transformation verification), and the data
portion of the Slice A gate in 12.2. Preserve the dtype policy in 3.3.

## Implementation and files

1. `src/survjax/data.py`
   - Apply `idx` to every array leaf, including the issue bits, and rebuild
     with the parent's auxiliary flags through the existing PyTree machinery.
     Do not reconstruct through a public constructor or reclassify a subset.
   - Support ordinary JAX array indexing: scalar/negative integer, slices,
     integer-array minibatches, concrete boolean masks, and multidimensional
     tuples. Under transformations, retain JAX's indexing constraints (for
     example, dynamic boolean masks do not have a fixed output shape).
   - Classify the supplied observation inputs at constructor entry, before
     normalization turns concrete inputs into tracers. Inspect every argument
     listed in section 6.2, including nested lists/tuples; `dtype` is not an
     observation input. Any tracer forces all six flags true.
   - For entirely concrete inputs, normalize and calculate flags in a
     compile-time evaluation context so concrete construction inside `jit`
     behaves like eager construction. Inputs already traced stay traced;
     unrelated traced function arguments do not affect this choice.
   - Pass the calculated flags explicitly when constructing either form.
     Keep `tree_unflatten` free of classification and validation so it also
     accepts JAX's reconstruction placeholders.
2. `src/survjax/_data_metadata.py` (new, small private module)
   - Hold the input-tracer check and concrete normalized-field classification.
     This responsibility is separate from row validation; `data.py` already
     exceeds 300 lines after the approved repr work, so do not add a general
     refactor or a second normalization implementation.
   - Reduce presence of exact/right/left/interval kinds and, independently,
     `left_trunc > 0` and `right_trunc < +inf` across all normalized rows,
     including invalid rows. Derive kinds from normalized endpoints, not the
     original keyword choices. Store only six Python bools in auxiliary data.
   - Convert only concrete values to host values; never convert a tracer.
3. `tests/unit/test_data_indexing.py` (new)
   - Cover indexing, preserved raw issue evidence, reconstruction, and
     transformed indexing as described below.
4. `tests/unit/test_data_metadata.py` (new)
   - Cover constructor classification, tree structures and compilation reuse
     as described below. Use existing dtype fixtures and exact assertions.
5. `REVIEW.md`
   - Overwrite after implementation with reading order, invariants, tests
     mapped to spec requirements, exact check results and remaining questions.
     Update this plan with the final scope if needed. Record any new spec
     question in `DEVIATIONS.md` and ask before choosing its behavior.

## Tests to add

- **Aligned indexing (3.2):** scalar, negative, strided/empty slice,
  repeated integer indices, concrete boolean selection and multidimensional
  indexing preserve each field's values, shape and dtype. Compare with
  explicitly indexed parent arrays, the identity required by the spec.
  Check agreement with `jax.tree.map` slicing and parent immutability.
- **Stored diagnostics (3.2, 6.2):** select rows whose normalized fields hide
  a raw-input fault (such as an invalid event indicator). Assert hand-derived
  issue masks survive selection and tree reconstruction. A valid subset of
  an invalid parent has scalar `ok=True`; an invalid subset has `ok=False`.
  Empty subsets retain parent flags and have no invalid rows.
- **Transformations (12.1):** eager, `jit` and `vmap` indexing agree for
  supported fixed-shape indices; decoded masks and scalar validity work
  inside `jit`. No numerical derivative checks are needed for this step.
- **Concrete flags (3.2):** both constructors yield the same flags for the
  same normalized data; separate kind and truncation changes alter the tree
  structure. Explicit default truncation matches omitted truncation. Include
  an interval flag with an infinite upper bound and rows with issue bits.
  Equal flags with different observation counts still give equal treedefs.
- **Input classification (3.2, 6.2):** concrete NumPy/Python inputs and
  captured concrete JAX arrays inside `jit` retain eager flags, including
  when another function argument is traced. An observation tracer, including
  one nested in a list/tuple or created by `jnp.asarray` inside `jit`, forces
  all flags true. Cover both constructor forms; compare leaves and issue
  masks separately because eager/traced metadata can intentionally differ.
- **Inheritance and compile counts (3.2, 12.1):** same shapes/dtypes/flags
  with different values trace once; different flags trace separately;
  equal-shape subsets of a mixed parent retain its flags and reuse a
  compilation. Traced construction and concrete data containing all kinds
  and both truncations share a tree structure and compilation.

## Verification and handoff

After approval and implementation, run `scripts/check.sh --through A3`:
the protected-path guard, Ruff formatting/lint, mypy, implementer tests and
independent acceptance tests through A3, with tests in separate float64 and
float32 processes. Run `git diff --check` as well. Record actual output and
any failure; do not claim A3 complete until the gate passes.

D-010 remains formally open, but the current acceptance test now checks
`batched[i].issues.ok`, matching the scalar-reduction contract. Its original
test discrepancy appears addressed; confirm through the A3 gate and leave
formal closure to the maintainer. Do not alter the read-only spec tests.

Finish with one commit on `A3` (proposed subject: `Add observation indexing
and static metadata`; body explains aligned issue preservation and compilation
reuse), the updated review, and a stop. Implementation/check evidence is in
`REVIEW.md`. The bounds constructor evaluates the normalized fields needed for
metadata at compile time; its issue-bit calculations remain pure JAX operations.
The implementation and tests total about 300 substantive changed lines;
normalization-block indentation accounts for additional diff lines.

# DOC-4 review fix-ups (2026-10-01)

Maintainer review of `1a15813`: print right censoring as its set
`(time, inf]` instead of `time+` (all censored rows now print as sets), and
make `docs/playground.ipynb` more informative. Extra commit on
`docs-startup`. Playground: short explanations between cells, the guide's
cohort (every censoring kind, truncation), stored fields, bad data with
`data.issues` and the host checkpoint (printed, so "Run all" completes),
`jax.jit`, and suggestions to try. No asserts, no saved outputs. Keep the
maintainer's `data` cell. Update repr tests and guide prose accordingly.

# Step DOC-4: readable reprs for SurvivalData and DataIssues (2026-10-01)

Requested by the maintainer after DOC-3 ("make a nice repr for SurvivalData
and DataIssues as well"). The spec does not cover repr. This replaces DOC-3's
"no summary repr" choice and adds no public API. One step, one commit on
`docs-startup`. Spec sections touched: 3.2 (private fields stay private), 6.2
(issue names), 6.3 (containers remain pytrees, unchanged).

## Behaviour

Both reprs read concrete values on the host. If any leaf is a JAX tracer, they
fall back to a short form with no value conversion (CLAUDE.md: never convert
tracers to NumPy), e.g. `SurvivalData(shape=(6,), dtype=float32, traced)`.

`SurvivalData`, eager: one line per observation in R-like notation, plus the
eligibility window when it is not the default `(0, inf]`. Invalid rows are
printed as stored, with a footer pointing at `.issues`:

```text
SurvivalData: 6 observations, float64
  0  4
  1  12+
  2  7
  3  9+
  4  (6, 9]   window (2, inf]
  5  (0, 3]   window (2, inf]
```

- Notation: exact `t`, right censored `t+` (as in R's print), every other
  censored row as its stored set `(lower, upper]`, so left censoring prints
  `(0, 3]` (maintainer's choice). Numbers use `:g`.
- More than 20 rows: first 10, `...`, last 5. N-d data: header shows the
  shape and rows show their index tuple.
- Footer only when non-empty: `issues: 4 invalid rows, 1 uninformative
  (see .issues)`.

`DataIssues`, eager: the overall verdict, then only the rules that fire,
with the row count and first index (the same wording as
`raise_if_invalid`):

```text
DataIssues: 4 of 5 rows invalid
  invalid_time            1 row, first index (0,)
  missing_time2           1 row, first index (1,)
  invalid_interval        1 row, first index (2,)
  censored_before_entry   1 row, first index (3,)
  uninformative           1 row, first index (4,)  (not an error)
```

A valid result prints `DataIssues: all 6 rows valid`.

## Files

Approved with changes: each repr lives next to its class, not in a separate
module, accepting files somewhat over 300 lines.

- `data.py`: `@dataclass(..., repr=False)` plus `__repr__` and two small
  formatting functions. The DOC-3 `field(repr=False)` declarations become
  redundant and are removed.
- `diagnostics.py`: the same for `DataIssues`. `raise_if_invalid` and the
  repr share one per-rule count helper; the exception message is unchanged.
- `tests/unit/test_data_repr.py`: replace the DOC-3 tests with hand-written
  expected strings for: each observation kind; truncation window; invalid
  footer; long-data elision; 2-d index tuples; traced fallback for both
  classes (inside `jax.jit`); valid and invalid `DataIssues`. Keep the
  six-leaves check.
- `docs/survival-data.md`: no code changes needed (cells already end with
  `data`); reread the rendered page and update prose that describes the
  printed repr. `REVIEW.md` as usual.

Checks: `scripts/check.sh --through A2`, `scripts/build_docs.sh`, `git diff
--check`. Expected diff about 250 lines.

## Decisions

- Left censoring prints as the stored set `(0, 3]`.
- No precision information beyond the dtype in the header.

# Step DOC-3: tutorial-style SurvivalData docs and a clean repr (2026-10-01)

Approved by the maintainer after reviewing DOC-2 (`726e561`): the guide reads
like a unit test, not a tutorial. One step, one commit, on a
maintainer-created branch (the worktree is a detached HEAD at `docs-startup`).
Spec sections: 3.1-3.3, 6.2, 10.5.

## Repr

- `src/survjax/data.py`: declare `_issue_bits` and `_may_contain` with
  `dataclasses.field(repr=False)`, so the repr shows only the five public
  arrays (spec 3.2 keeps the others private). No summary repr. `DataIssues`
  is unchanged. Pytree flattening and `__init__` are unchanged.
- `tests/unit/test_data_repr.py`: the eager repr names the five public
  fields and omits the private ones; `repr` of data built inside `jax.jit`
  does not raise and omits them; `jax.tree_util.tree_leaves` still gives six
  leaves.

## Docs: rewrite `docs/survival-data.md` as a tutorial

Keep the facts of DOC-2; change the form.

- Open with a short note: what works now (construction, diagnostics) and what
  doesn't yet (likelihoods). No closing scope section.
- Use one running synthetic story: a small cohort (about six patients,
  follow-up in months) built up section by section. Each section answers a
  reader task: "time and status, as in R"; "the event happened between two
  visits"; "known only to have occurred before the first visit"; "patients
  entered late (truncation)"; "finding bad rows"; "inside `jax.jit`".
- Show results instead of asserting them: end cells with the displayed value
  (`data`, `data.upper`, `data.issues.invalid_rows`). Use asserts only where a
  silent check helps the reader, if at all. The build failing on execution
  errors is the executable check (spec 10.5).
- Show real errors: call `raise_if_invalid()` and the misuse examples in cells
  tagged `raises-exception`, so the rendered page shows the message. No
  `try/except/else: raise AssertionError` scaffolding.
- Explain why before how: a sentence or two on what each censoring kind means
  for the patient before the call that encodes it. Introduce the normalized
  fields when the first output shows them, not as an opening list of rules.
- Present the R comparison as a table (`Surv(...)` call next to the survjax
  call, plus a notes column). Re-check each claim against the official `Surv`
  help page. In particular, the "(start, end]" wording there may refer to
  counting-process data, not interval censoring; drop the claim unless the
  page supports it.
- Dtype behavior: one short section with a single example, not a test of both
  modes.

`docs/playground.ipynb`: remove the empty trailing cell and reset the
kernelspec and `language_info` metadata to the DOC-1 minimal form. Keep the
maintainer's `data` cell, which now shows the clean repr. Have it follow the
same cohort as the guide, without asserts. `docs/index.md`: update the guide's
one-line description if needed.

## Checks and review

`scripts/build_docs.sh` (both dtype modes), `scripts/check.sh --through A2`,
`git diff --check`. Read the rendered x64 guide end to end and report in
`REVIEW.md` whether every output shown matches its surrounding prose. The
guide rewrite replaces most of a 176-line page, so the diff may exceed about
300 lines. If it does, say so in `REVIEW.md` rather than splitting the step.

# Documentation startup on `docs-startup` (2026-09-30)

The maintainer requested two separate tasks following specification section
10.5. Each is one reviewable step and one commit. Stop after the first step's
`REVIEW.md` and await instruction before starting the second. This plan is for
approval before either implementation step; the earlier plans below are history.

## Step DOC-1: local documentation process

Spec sections: 10.2, 10.3, 10.5, and the documentation item in 12.2.

Files to create or change:

- `pyproject.toml`, `uv.lock`: add a locked optional documentation group with
  Sphinx, MyST-NB, JupyterLab and a Python kernel. Keep core dependencies
  unchanged.
- `docs/conf.py`, `docs/index.md`: minimal Sphinx site for Markdown and
  notebooks, navigation, and a clear statement of the currently implemented
  subset. Include the existing `docs/simulation-guide.md` only if it can be
  visibly labelled as illustrative and unverified without build warnings.
- `docs/playground.ipynb`: a small synthetic, editable notebook using the
  current public `SurvivalData` API; no saved outputs.
- `scripts/build_docs.sh`: one documented, non-interactive command that builds
  HTML from a clean checkout, executes runnable content afresh in separate
  float32 and float64 processes, and fails on warnings or execution errors.
- `.gitignore`: exclude generated site, notebook execution artifacts and
  caches. `REVIEW.md`: report exact commands, outcomes and rendered-page
  inspection. Update this plan if implementation reveals a necessary split.

Public API: none. Tests: the build command itself is the end-to-end check;
verify it from a clean working tree and inspect the rendered index and notebook.
Run the existing A2 gate and `git diff --check` if dependency or tooling edits
affect those checks. No network, R, real data or demo dependencies at doc build
time. Do not publish or host the site.

## Step DOC-2: A1–A2 documentation

Spec sections: 3.1–3.3, 6.2, 10.2, 10.5, and 12.2. Cover implemented behavior
only; label A3 indexing/static-metadata guarantees and later likelihood modes
as planned. Revise this step plan, if needed, before writing its code and seek
approval under the working agreement.

The maintainer's follow-up authorizes DOC-2 and also asks for the commands to
open the rendered HTML. Add those commands to `docs/index.md`. A locally edited
playground currently contains exploratory cells and saved outputs: preserve its
cell code, clear saved outputs for the committed source, and extend it with the
new guide's working API. This is still the one documentation step below.

Files to create or change:

- `docs/survival-data.md`: executable synthetic examples for event/status,
  explicit right/left/interval flags, mixed data, `time2`, bounds, and
  truncation. Show normalized fields and the exactness rule for bounds.
- In the same page, explain issue masks, `invalid_rows`, `ok`, valid but
  uninformative rows, structural `TypeError`/dtype `ValueError`, and explicit
  host `InvalidDataError`. Demonstrate ordinary construction and diagnostic
  access inside `jax.jit`, with the host checkpoint after compiled execution.
- In the same page, compare the flag and bounds conventions to R's
  `survival::Surv` forms, including event coding, interval endpoints and
  counting-process `time2`. Check comparison wording against official R
  documentation. Claim no numerical agreement without R-generated fixtures.
- `docs/index.md` and `docs/playground.ipynb`: link the guide and make the
  notebook a short interactive path through the same working API.
- `REVIEW.md`: reading order, invariants, exact example/build outcomes in both
  dtypes, source of R-convention claims, known gaps and rendered-page review.

Public API: none. Tests: executable examples through `scripts/build_docs.sh`
in both dtype modes, plus the existing A2 gate and `git diff --check`. Examples
must check their stated outputs rather than merely run. No new unit tests that
repeat documentation examples. Keep this step near 300 changed lines; split it
before implementation if the reviewed plan requires more.

# D-006 diagnostic compatibility decision (2026-09-30)

The maintainer approved retiring `TracedInputError`, keeping `ValueError` for a
traced `DataIssues.raise_if_invalid()` call, and exposing only the
`data.issues` property (no callable `data.issues()` alias). On the
maintainer-created `spec-deviations` branch, update the normative contract in
`spec/SPECIFICATION.md` sections 6.2, 10.2 and 13, its review/decision history,
and D-006 in `DEVIATIONS.md`. Leave D-010 and the other deviation statuses
untouched. This explicit maintainer request authorizes the spec edit.

Public API: clarify that only `InvalidDataError` is exported for data
diagnostics; no code or behavior changes. No tests are added for this
documentation step. Review the diff for internal consistency and run
`git diff --check`. Overwrite `REVIEW.md` with the reading order, invariants,
evidence and uncertainties. Commit this one step on `spec-deviations` and stop.

# A2 constructor API migration — specification revision 9

## Pre-merge bookkeeping step (2026-09-30)

The maintainer requests a clarification of the non-exact equal-endpoint case
and cleanup of deviation IDs/statuses. Change only the `from_bounds` docstring
in `src/survjax/data.py`, the branch D-006 references in
`src/survjax/_data_validation.py` and
`tests/unit/test_data_construction.py`, and `DEVIATIONS.md`. Rename the branch
D-006 to D-011 and mark D-011, D-007, D-008 and D-009 as superseded by
revision 9, leaving final closure to the maintainer. Update `REVIEW.md` and
commit this small step on `a2-constructors`.

Public API: unchanged. Spec: sections 3.1, 6.2 and 13. Tests: the existing
independent A2 equal-endpoint acceptance cases and the full A2 gate; no new
test is needed for a docstring and bookkeeping change. Do not edit the
read-only specification or independent tests.

Branch: `a2-constructors`. A2-R1 exception alignment was committed as
`c6ef7e3`. The maintainer's 2026-09-30 request authorizes the remaining
constructor API migration: remove issue-returning constructor methods, accept
eager and traced observation inputs in the two ordinary forms, and expose
diagnostics through `data.issues`.

## Current step

Spec sections: 3.1–3.3, 6.2–6.3, 10.2 and 12.1–12.2.

Files and public API:

- `src/survjax/data.py`: retain exactly `SurvivalData(...)` and
  `SurvivalData.from_bounds(...)` as construction methods; remove automatic
  host validation and the `with_issues` method; add a typed `issues` property
  decoding the stored issue bitfield. No `from_bounds_with_issues` method.
- `src/survjax/_data_validation.py`: remove obsolete host-only constructor
  validation and references to the removed method, preserving normalization,
  dtype policy, argument-misuse checks and issue attribution.
- Own tests in `tests/unit/`: migrate callers to ordinary construction and
  `data.issues`; check eager, `jit` and `vmap` behavior, invalid-value masks,
  explicit host checkpoints, traced inputs, tree reconstruction and the absent
  methods. Keep tests with code in the same commit.
- `REVIEW.md`: map checks to requirements and report exact gate results.

The revision 9 specification and its acceptance criteria already require this
API; `spec/GOALS.md`, `spec/SPECIFICATION.md` and `tests/spec/` are read-only.
There are no current examples or docs modules to migrate. Historical deviation
entries retain their original wording and status.

## Verification

Run `scripts/check.sh --through A2` to cover Ruff, mypy, own tests and the
independent A2 tests in both dtype modes. Use focused A3 acceptance tests for
the removed-method and tracing checks, without claiming A3 complete. Run
`git diff --check`. If an independent test conflicts with the specification,
record the issue in `DEVIATIONS.md` and leave the test unchanged.

## Later handoff

A3 owns indexing and slicing, value-derived concrete static metadata and
compile-count tests (spec 3.2). The current all-true metadata is only a
conservative interim value. A6 owns likelihood issue propagation and explicit
failure for unimplemented observation modes. Neither is part of this step.

The traced host-checkpoint exception subtype and legacy diagnostic compatibility
remain open in spec section 13 and `DEVIATIONS.md`. This step preserves the
existing traced-checkpoint failure and does not introduce a callable alias.

## Performance amendment on `spec-speed` (2026-09-30)

The maintainer requests adapting the earlier performance draft to the current
specification at `9a403a3`, including revision 10 diagnostic compatibility.
This authorizes edits to the normally read-only specification and goals.

- Update `spec/GOALS.md` with speed, memory efficiency and acceleration.
- Update `spec/SPECIFICATION.md` sections 1, 7.2, 9.2, 10.3, 12 and 14:
  retain pure ordinary construction, `data.issues`, stored-issue propagation,
  conservative traced metadata and simulation fit-skipping semantics. Measure
  prepared-data evaluation and traced construction separately. Keep GPU checks
  and scalable metric algorithms optional, with a small CPU baseline in Slice A.
- Leave the final amendment revision number for acceptance: the current header
  says revision 9 while the history includes revision 10. No API changes,
  implementation edits, new dependencies or changes to independent tests.
- Overwrite `REVIEW.md` with reasoning, reading order, preserved contracts,
  documentation checks and unmeasured performance requirements.
- Checks: whitespace, Markdown/section references, allowed diff scope and exact
  preservation of unrelated statistical/API sections and example code. Runtime
  tests and benchmarks are not run for this documentation-only step.
- Commit this one documentation step on `spec-speed`; do not merge or publish.

## Illustrative simulation guide on `spec-speed` (2026-09-30)

The maintainer approved moving the simulation workflow out of the normative
contract and removing recipe-specific acceptance requirements. This authorizes
the following documentation step, including edits to the specification.

- Move the detailed recipe into `docs/simulation-guide.md`, explicitly
  illustrative and unverified; retain `lax.map` as one example choice.
- Update specification sections 6.2, 11.4, 12.1, 12.3 and 14: retain library
  guarantees, check compiled composition directly, and leave study workflow,
  external solver and loop selection to callers and measurements.
- Public API, goals, implementation, dependencies and tests remain unchanged.
  Add no runtime tests; check whitespace, local links, moved example code and
  preservation of unrelated contracts. Record results in `REVIEW.md`.
- Commit this single documentation step on `spec-speed` and stop.
