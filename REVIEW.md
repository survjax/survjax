# Main integration audit — A8 follow-up required (2026-10-29)

Requested: resolve PLAN.md/REVIEW.md conflicts in the maintainer-started merge
of main be1a5b3 into a8-1-r-generator, inspect spec/tests relevant to A8 and
plan any additional work. No new numerical or compatibility implementation
was authorized by this integration audit.

## Read first

1. PLAN.md, new integration findings and proposed compatibility step.
2. Incoming src/survjax/__init__.py and likelihood.py: main removes the
   top-level likelihood function; the distribution method remains.
3. Incoming tests/spec/test_a8_bc_fixtures.py: method-only calls now match
   spec revision 35, with unchanged fixture schema and numerical targets.
4. Existing tests/reference/test_exponential_r.py and
   test_weibull_bc_like_r.py: stale function calls require the planned fix.

PLAN.md preserves both branches' distinct plan prefixes and their shared
historical suffix once. Incoming main's size-interface plan is retained as
integration history, not an instruction to start another step. This review
replaces both previous per-step reviews; their exact records remain in Git
(382f8dd for A8.d1 fixes, 030314b for main's size interface).
Only PLAN.md/REVIEW.md conflict resolutions are authored here. All other
staged incoming changes, including independent tests, protected scripts,
.spec-seen, source and dependencies are retained unchanged from the merge.

## Relevant contract changes

Incoming marker: be26e50, revision 37; prior marker d243a966, revision 34.
Read-only specification comparison confirms:
- Revision 35: spec 6.1/7.1/10.2/12.1 remove the top-level likelihood and
  require dist.log_likelihood as the sole public likelihood entry point.
- Revision 36: spec 3.2 adds data.shape, len(data), undefined truth value.
  Incoming main implements and tests them; A8 requires no data-size changes.
- Revision 37: PH closure, staged RMST and release/demo requirements are
  outside this A8 fixed-vector reference scope. No R payload regeneration.

The independent A8 test diff migrates prepared/raw/design-matrix/total calls
to the method and removes a redundant method test. It adds no schema or
parameter changes. Spec tests stay read-only. The protected fixture is still
absent; through-A8 remains pending durable tester-repo delivery and sync.

Main also adds locked pytest-xdist and six-worker gate defaults. Retained;
no changes to numerical tolerances. No contract ambiguity or new deviation.

## Actual checks and known failure

Focused float64 checks on the combined working tree, using
UV_CACHE_DIR=/private/tmp/survjax-uv-cache and uv run --group test pytest -q:

- tests/reference/test_exponential_r.py -k function_prepared:
  6 failed, 39 deselected in 1.14s, all AttributeError because survjax has no
  log_likelihood attribute. Expected integration incompatibility, not a
  numerical mismatch. Log: /private/tmp/survjax-a8-merge-probe.log.
- Same module -k 'method or survival or diagnostics':
  27 passed, 18 deselected in 2.60s.
  Log: /private/tmp/survjax-a8-merge-method.log.

Initial sandboxed attempts could not download main's locked pytest-xdist
package (DNS); the escalated retry installed it and ran these probes.
No float32, full reference suite, mutation checks, full gate, R regeneration
or docs build was run for this audit. The known stale calls must be migrated
before claiming a passing integrated gate. Existing docs also still claim
function/method comparison coverage; a separate docs fix is planned.

Final Ruff formatting/lint, mypy and whitespace results are recorded below.
Stop for approval of the proposed code step; do not restore the removed API,
edit independent tests or start later-slice work.

Final static checks: Ruff format reports 78 files already formatted; Ruff lint
reports All checks passed; mypy reports no issues in 17 source files;
git diff --check exits 0. No conflict markers remain in either record.

After staging, git diff --cached --check reports two inherited whitespace
warnings in incoming read-only spec tests: API_ASSUMPTIONS.md:644 and
test_a6_likelihood_shapes.py:121 (new blank line at EOF). Left unchanged;
these are not conflict-resolution edits. Unmerged-path listing is empty.
Merge remains in progress with resolved records staged; no merge commit was
created during this audit. Compatibility implementation awaits plan approval.
