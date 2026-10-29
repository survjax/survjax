<!-- Moved verbatim from PLAN.md lines 1722-1895 (dev commit a5cfea7) on 2026-10-29. -->

# Revision 35 — Remove the public likelihood function (2026-10-25)

Status: **LL.1 merged; LL.2 implemented and verified; ready for review** ("ok merged; now up to
next bit"). Reuse the maintainer-created `remove-ll` branch at `fab100f`,
which is also the merged main tip. The starting tree is clean. No further
plan approval is needed for the LL.2 scope below.
This entry supersedes earlier plans that require two public likelihood entry
points; their implementation/review records below remain historical evidence.
Two small steps, each with its own commit, REVIEW.md and approval stop.
Approval of the first step does not start the second.

## Contract and current evidence

- Spec revision 35, decision 0048: `dist.log_likelihood(data=None, *,
  time=None, event=None)` is the only public likelihood entry point. Remove
  `survjax.log_likelihood`, retaining the method's signature, numerical
  behavior, derivatives, diagnostics and prohibition on family overrides.
  Sections covered: 4.2, 6.1, 7.1–7.3, 10.2, 12.1 and 12.2.
- `.spec-seen` already records `9696894`, the revision-35 spec commit;
  `../scripts/spec-since.sh dev` reports `spec unchanged since 9696894`.
  No marker update or design-repo change is needed.
- Updated independent tests arrived in `744fb55`, with source commit
  `8041b326a194e89f56aade79b112e7f3c5bbc601`. They now require absence of
  the top-level name and agreement of the method's two input forms in values,
  gradients and Hessians. Later-step tests also migrated to the method;
  their presence does not authorize implementing those steps.
- Current code still re-exports the function. The base method delegates to
  that function, and its docstring describes equivalence with it. Implementer
  tests still exercise the function/method matrix. The likelihood tutorial,
  API reference and simulation guide also retain function calls.
- Planning checks were read-only inspection of source, tests, docs, spec diff,
  branch/history/status and spec alignment. No tests or docs builds have run
  for this change yet. No unresolved contract question was found.

## LL.1 — Public API and implementer tests (first approval stop)

Public API change: remove the top-level function; retain the existing method
and both of its input forms. Target less than 300 changed source/test lines.

Files and implementation:

- `src/survjax/__init__.py`: remove the function import and `__all__` entry.
- `src/survjax/likelihood.py`: delete the public `log_likelihood` wrapper.
  Keep `_resolve_data` and the shared `_log_likelihood` evaluator here, as
  section 10.2 specifies. Their numerical bodies remain unchanged.
- `src/survjax/distributions/base.py`: delegate directly to those existing
  private helpers, retaining the deferred import to avoid a cycle. Make the
  method docstring self-contained: input forms, shapes, common dtype,
  unsummed output, NaN/zero/overflow behavior, diagnostics and static misuse.
  Do not add a compatibility alias or another public function.
- `tests/unit/test_import.py`: update the exact export set and explicitly
  assert absence of `survjax.log_likelihood`, including after importing the
  evaluator module and invoking the method.
- `tests/unit/test_likelihood_entry_points.py`: remove function/method
  parametrization and function-only invalid-distribution tests. Retain all
  method input-validation and transformation coverage. Add direct comparison
  of raw/prepared values, first parameter derivatives and Hessians for the
  two built-ins, eagerly and under jit; retain independent analytical anchors.
- `tests/unit/test_likelihood_transformations.py`: remove the obsolete
  entry-point axis while keeping both forms, both built-ins and second-order
  forward/reverse finite-difference checks, eager and compiled.
- `tests/unit/test_likelihood_tails.py`,
  `tests/unit/test_likelihood_boundaries.py` and
  `tests/unit/test_likelihood_debug_nans.py`: migrate function calls to the
  method and remove only duplicate function/method cases. Preserve tail
  expectations, tolerances, zero-censoring regression, debug mode and
  eager/jit/vmap coverage. Existing tests importing `_log_likelihood` in
  `test_likelihood_core.py` and `test_likelihood_parameters.py` need no change.
- Update this plan's status and overwrite `REVIEW.md` with reading order,
  invariants, spec-to-test mapping, exact results and remaining limitations.

Verification after approval:

1. Run the changed own modules plus likelihood core/parameter modules in
   separate `JAX_ENABLE_X64=1` and `=0` processes using `uv run pytest`.
2. Run the updated independent A6 entry-point/method modules in both dtypes
   with `SPEC_THROUGH=A7`. Never edit, skip or xfail independent tests.
3. Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
   on the final review snapshot. This includes Ruff, mypy, all own tests,
   source doctests and independent tests through the implemented A7 gate.
   A8–A10 acceptance tests are deferred to their own implementation work.
4. Search tracked source/tests/docs/notebook content for obsolete public
   calls; at this step remaining calls must be confined to the LL.2 docs
   to-do or explicit absence assertions. Run `git diff --check`.

Report actual counts, removed duplicate cases, skips and failures; never infer
numerical preservation from a rename alone. If a numerical change is needed
or the source/test diff exceeds the budget, split/replan before expanding.
Commit on `remove-ll` with an imperative subject and rationale, then stop.

Implementation evidence: source/test diff is 177 changed lines; all five
private likelihood helper ASTs match the starting commit. Focused own suites:
205 passed per dtype (114.52s float64 / 114.19s float32). Updated independent
A6 entry-point/method suites: 35 passed per dtype (7.01s / 7.03s). No skips
or failures. Removed 42 function-only/duplicate cases and added 17 cases
(16 input-form comparisons and one API-absence test), a net decrease of
25 cases per dtype. Ruff reports 71 files already formatted / All checks
passed; mypy reports no issues in 17 source files.

The final `scripts/check.sh --through A7` ran after saving REVIEW.md and
exited 0: `check: all gates passed through A7`. Ruff: 71 files already
formatted / All checks passed; mypy: no issues in 17 source files.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 752 passed, 7 skipped, 160.92s | 758 passed, 1 skipped, 158.62s |
| Source doctests | 4 passed, 1.09s | 4 passed, 1.03s |
| Independent through A7 | 2082 passed, 28 skipped, 218 deselected, 112.08s | 2086 passed, 24 skipped, 218 deselected, 109.65s |

Complete output: `/private/tmp/survjax-ll1-final-gate.log`. REVIEW.md remains
unchanged after the gate; only this plan's result record is updated afterward.
No new skips/exclusions. Final Ruff formatting/lint and diff checks follow
this result update, then commit LL.1 and stop. LL.2 has not started.

### Docs to-do

The narrative docs intentionally wait for the separate LL.2 docs step below.
The API reference and executable tutorial still reference the removed function
after LL.1, so the full docs build is expected to be broken until LL.2 lands.
LL.1 is not completion of the whole migration or of Slice A.

## LL.2 — Likelihood API documentation (separate approval stop)

Status: **implemented and verified; ready for review**. Followed the existing
docs plan below; write REVIEW.md, commit on `remove-ll` and stop. The spec is still
revision 35; `.spec-seen` already matches. No source/test changes are needed.

One topic: migrate documentation to the distribution method, below 300 changed
lines. Files: `docs/likelihood.md`, `docs/api.md`,
`docs/simulation-guide.md`, PLAN.md and REVIEW.md. Public API touched: none.
Spec: 6.1, 7.1, 10.2, 10.5 and 12.1.

- Rewrite “Two entry points, two input forms” around one method accepting
  `SurvivalData` or raw `time`/`event`. Compare those two forms in the
  executable example and remove function-choice advice.
- Remove the obsolete `autofunction` directive from the API reference;
  retain the method documentation and adjust nearby wording if needed.
- Change the simulation objective to construct a distribution and call its
  method. Audit the tracked notebook and other current guides for stale
  function calls; the distribution guide's local objective function is not
  the removed package API and need not be renamed.
- Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`
  with fresh execution, warnings fatal, in separate float32/float64 processes.
  Inspect rendered method documentation, the two-form example's output,
  links and layout in both modes. The simulation sketch uses optional
  optimistix outside the core; do not add a dependency or claim solver
  execution from a static code block. Record any visual inspection limit.
- Repeat the obsolete-call audit, Ruff formatting/lint and `git diff --check`.
  Update PLAN.md and overwrite REVIEW.md with exact checks and reading order;
  commit this docs topic on `remove-ll`, then stop for review.

LL.2 evidence: 36 changed lines across the three docs pages. The docs build
exits 0 with `build succeeded` in float32 and float64; likelihood-page execution
takes 4.01s / 3.35s. Log: `/private/tmp/survjax-ll2-docs-build.log`.
The initial sandbox run could not open Jupyter kernel sockets; the permitted
rerun passed. Existing kernel TCP transport notices appear in the log; there
are no Sphinx documentation warnings. Chrome screenshot inspection passes for
the tutorial's two-form section and API method reference in both dtype builds.
Both forms show -23.124392 / -23.12439364 respectively. The HTML audit checks
273 local page/anchor links per build and the method's reference anchor.
The simulation guide is excluded from Sphinx; a temporary check extracts only
its revised objective and verifies value/gradient/Hessian = 6/4/6 at log-rate
zero on synthetic data, in both dtypes. Its optional solver was not run.
Tracked docs/notebook audit finds no obsolete package-function references.
REVIEW.md records commands, evidence and scope. Final Ruff formatting/lint
passes after the review edit: 71 files already formatted / All checks passed.
`git diff --check` exits 0. The full library gate and R were not run for this
docs-only step. Generated HTML and temporary checks remain untracked.

---

---


