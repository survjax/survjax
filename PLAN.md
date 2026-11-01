# CI.1 — Three-platform CPU GitHub Actions (2026-11-01)

## CI.1 round 1 follow-up plan

Review 2166045 returned fix-first: F-1 Windows CRLF checkout; F-2 silently
skipped guard. S-1 asks which PR branch policy to apply; record D-030 and
ask the maintainer before dependent changes. S-2 is already settled by the
explicit request for CPU macOS/Linux/Windows. Fix F-1 by setting autocrlf
false and eol lf before checkout on the ephemeral runners. No protected
script change. Revalidate YAML and whitespace; full numerical rerun is
unnecessary for checkout configuration after the passing initial gate.
N-1/N-2/N-3 are nonblocking nits; retain the existing gate/triggers/action
choices for now. Record review limits and remaining work in REVIEW.md.

Start on maintainer-created branch ci at
84d2943e788440db60504afb49a13360d43ad95e (worktree starting base).
Previous implementation endpoint: aae68920769855aa08227f4b7c4818ce7a820923
(RMST.integration); subsequent commits are inherited maintainer metadata.
Earlier unfinished RMST/A9 records below remain
owned by their original workstreams; preserve them rather than archiving
unfinished work. This CI step adds no numerical feature or public API.

Files and behavior, planned before code:
- .github/workflows/ci.yml: push, pull_request and manual CPU checks on
  ubuntu-24.04, macos-15 (Apple silicon), windows-2022 (x64), Python 3.12,
  uv 0.11.19 and the unchanged lock. Read-only token, independent job failures,
  bounded runtime and cancellation of superseded runs. Force JAX_PLATFORMS=cpu.
- Provision locked dev/test dependencies and build a wheel while online;
  export its native absolute path with Python so Windows Git Bash does not
  leak an MSYS path. Run the unchanged scripts/check.sh --through A10 offline
  with two workers. Its own suite, source doctests and independent tests each
  run in separate float64/float32 processes. Keep tests/spec read-only.
- tests/install/test_wheel.py: select Scripts/python.exe on Windows and
  bin/python elsewhere. Existing out-of-tree wheel tests verify this path
  through actual installation on each matrix platform; no mock-only tests.
- REVIEW.md: overwrite for CI.1 with invariant/check map and honest platform
  limits. No docs/, protected paths, spec, dependencies or lock changes.

Spec covered: 10.1 (existing static checks), 10.3 (locked CPU baseline,
separate dtype processes, installed package, optional accelerator evidence),
12.1/12.2 (existing through-A10 acceptance gate, no new slice).
Validate YAML, Ruff/mypy, lock consistency, full local wheel gate, whitespace
and scope. Commit then use independent-review, at most three rounds; stop.
Hosted Linux/Windows execution requires pushing, which is outside this step.

GPU assessment / subsequent optional step (not enabled in this commit):
- Prefer manually dispatched Linux/NVIDIA verification. GitHub larger runners
  offer a T4, or use an ephemeral self-hosted/cloud GPU. Runner availability,
  account eligibility and billing must be chosen before adding a working job.
- Pin a separate accelerator environment to the CPU JAX version plus compatible
  CUDA plugins/libraries and record the NVIDIA driver. Do not mutate uv.lock
  or add GPU dependencies to core. Current JAX guidance offers CUDA 12/13;
  choose against actual GPU and driver (Linux minima 525/580 respectively).
- Set JAX_PLATFORMS=cuda, assert GPU devices, log versions/driver/devices and
  run applicable numerical/transformation/spec suites per dtype, initially
  serially to avoid GPU memory contention. Installation smoke tests retain
  their separate CPU-only environment. Record backend failures explicitly.
- Require float32 and float64 evidence before NVIDIA support claims; measure
  performance separately. Metal remains experimental/float32 only, outside
  required CPU CI. Do not execute untrusted PRs on a persistent GPU runner.
Sources checked 2026-11-01:
https://docs.github.com/en/actions/reference/runners/larger-runners
https://docs.jax.dev/en/latest/installation.html

Local validation completed: unchanged fresh-wheel A10 gate exits 0; own
suites 1097/7 x64 and 1103/1 x32 passed/skipped; source doctests 4 each;
independent suites 2332/28 x64 and 2322/38 x32. YAML/Ruff/mypy/lock and
whitespace checks pass. REVIEW.md records commands and platform limits.

Remaining sub-steps: CI.1 review and hosted CPU runs after maintainer push;
GPU provisioning, separate lock and optional workflow require a later step.

---

# RMST.integration — Resolve the maintainer-started main merge (2026-10-31)

Branch: release-rmst. Merge already in progress when requested; finish that
merge only, without rebasing, publishing, or merging this branch into main.
Parents: ddd7f1f808eea0c802848bf0489135b547fed67f (RMST) and
4f170654fdc5bb2840f89232e439e33d7e5fc3ff (incoming main).

Conflict-resolution plan, before edits:
- tests/impl_tolerances.py: preserve both independently added blocks verbatim:
  measured RMST factors and SIMULATION_SIGMAS=6. No tolerance value changes.
- PLAN.md: keep this active integration/RMST handoff first, then preserve both
  pre-merge plan versions verbatim in clearly labelled records below. Do not
  discard or archive unfinished history. Main's existing plans/A10.md remains.
- REVIEW.md: write this integration's reading order, preservation invariants,
  checks and limitations. The prior RMST and README reviews remain available
  verbatim at the two parent commits; no extra review/archive module needed.
- All other staged paths came from main. Preserve them exactly, including
  CLAUDE.md and .workmux.yaml; do not edit their protected contents. No new
  numerical code, public API, dependency or spec change in this integration.

Initial static check found an inherited main issue: README.md's fenced
Python example needs two blank lines before its top-level function. Apply
Ruff's formatting-only correction, preserving the example's behavior; this
is the only inherited-file exception, with no docs/ or protected-path edit.

Validate source equality against the RMST parent; imported-file equality
against main; both plan records as exact substrings; both tolerance blocks;
absence of conflict markers/unmerged paths; Ruff/mypy, locked environment,
focused RMST+generator tests in separate dtype processes, and unchanged
fresh-wheel through-A10 gate. Commit the existing merge, invoke the required
independent-review as RMST.integration (round 1/3), then stop/report.

Remaining work owned by this worktree after integration:
1. Core RMST.1/.2/.3 and math appendix are complete and independently reviewed
   (RMST.3 round 3 merge). Spec revisions 38/39 are now on design main; no
   new spec decision is needed. Deviation status remains maintainer-owned.
2. Handoff base.py/exponential.py/weibull.py closure edits to release-ph once
   this RMST branch is integrated into code main. This local main snapshot
   does not contain wrappers.py or ProportionalHazards yet. PH owns wrappers,
   exports and its own closure/RMST identity/derivative tests; do not duplicate.
3. Then one separate executable docs/rmst.md topic: staged surface; explicit
   cutoff axes rather than rmst_grid; units; invalid values; first-order-only
   zero boundary; unspecified NaN-own tangents; native higher-shape restriction
   and supported nonshape derivatives. Coordinate navigation/API ownership.
4. Recheck the combined PH+RMST documentation/examples and both-dtype release
   evidence after separately reviewed PH integration. No new core RMST feature
   is planned; generic quadrature/sampling/options stay outside this workstream.

Validation complete: 49 focused RMST/generator tests pass per dtype;
Ruff/mypy/lock/preservation/whitespace checks pass. The fresh-wheel A10 gate
exits 0: own 1097/7 x64 and 1103/1 x32, doctests four per mode, independent
2332/28 x64 and 2322/38 x32 passed/skipped. Finish the maintainer-started
merge and obtain the independent RMST.integration review before reporting.

---

# Preserved RMST-side plan (verbatim pre-merge version)

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Other sub-slices are archived verbatim in `plans/` (index: `plans/README.md`).
Still open there: A9 (`plans/A9.md`, "A9 Docs to-do" and "Final evidence
audit and planning checks") and A8 (`plans/A8.md`, "Closure still required").

---

# RMST.3 — Weibull derivative and isolation evidence (2026-10-30)

Start release-rmst at b2150e1fe10df97d7ac2d4e9926a2b8a803f90c4;
original launch base remains 8757c0fda78e34e6e818249adedd2a144a33c33d.
The maintainer explicitly requested the next implementation step. Appendix
review is running in its separate checkout against the committed snapshot;
its findings will be handled before this step is committed. Preserve history.

Status: implementation resumed from appendix fix-up
dc48b24cd89a571a082099f4055cd1669c8ff3cd. The maintainer explicitly
approved unspecified own derivatives for a NaN RMST result (2026-10-30),
settling S-1b. Appendix round 2 returned VERDICT: merge, all findings fixed.

One bounded numerical-verification concept, about 250 changed test lines:

- New tests/unit/test_weibull_rmst_derivatives.py: independent 60-digit
  mpmath quadrature of survival and its shape/scale derivative integrands;
  integrate the scale kernel directly to avoid cancellation in m-r*S,
  and derive the cutoff derivative from survival. Use actual dtype-rounded
  inputs for references. Cross
  k={0.2,0.5,1,2,10,50}, tau/b={0.01,0.1,0.5,1,10,100,1e5},
  b={1e-4,1,1e4}; straddle both branch thresholds at four shapes too.
  Compare eager/jit/vmap forward and reverse natural Jacobians; express
  parameter errors in log-coordinate units k*dM/dk and b*dM/db, divided
  by |M|, so the section 12.1 value-scaled exception respects time units.
  The cutoff derivative uses absolute scale 1 and the S(tau) identity.
- tests/impl_tolerances.py: add only RMST-specific absolute error factors,
  zero relative tolerance. Pre-test full-range probes (142 cases, both
  dtypes) found maxima [shape,scale,cutoff] of
  [1.72e-14,2.46e-15,1.56e-15] x64 and
  [1.29e-5,1.55e-6,6.45e-7] x32 under the above normalization.
  Fix parameter factors 2e-13/5e-5 and cutoff factors 2e-13/5e-6,
  allowing rounding margin without changing any existing tolerance.
- Verify finite supported (b,tau) Hessians across the grid and independent
  analytical Hessian entries on lower-series and gamma interior cases.
  Preserve explicit shape-Hessian NotImplementedError coverage; extend it
  across forward/reverse combinations in the gamma middle branch.
- Shared shape/scale parameters: valid entries beside invalid parameters
  or cutoffs retain the isolated call's first derivatives in both modes;
  supported scale Hessians stay isolated too. Do not assert a guarantee
  about an invalid result's own tangent. Extend zero first-derivative checks
  over shapes/scales, with eager/jit/vmap and debug_nans for valid entries.

Spec 4.5, 7.2–7.3, 8.7 revision 38, 12.1–12.2 and the release handoff were
read in full. No numerical algorithm/public API, wrappers, exports, modules
in src/, dependencies, quadrature tangent, grids or ordinary docs changes.
No new higher-order boundary or shape guarantee. Docs to-do: executable
RMST topic after PH integration, with first-order-only zero boundary,
unspecified NaN-own tangents, and the native higher-shape restriction. Explain
keeping shape outside differentiated arguments for supported Hessians;
packing it with scale can materialize a zero shape tangent in native JAX.

Run focused RMST tests in separate x64/float32 processes, Ruff/mypy,
whitespace checks, and unchanged scripts/check_wheel.sh --through A10
(fresh wheel supplies the standard gate's required install artifact).
Write REVIEW.md with exact outputs, commit, invoke independent-review
exactly as RMST.3, handle up to three rounds and stop/report. PH owns
wrappers.py/exports and must wait for maintainer core integration and an
explicit ownership handoff before built-in closure edits. No merge/push.

## RMST.3 approved tolerance clarification

Maintainer explicitly answered “Confirm log-parameter scaling” to S-1/D-029.
Approved clarification: change spec 12.1's exception to name k*dM/dk and b*dM/db, with cutoff error
still at absolute scale 1. Bump specification revision 38 to 39, commit alone
on design release-rmst, then mark this worktree via spec-since.sh
dev-release-rmst --mark and commit .spec-seen plus decision/review records.
Committed separately in design as
1389af4da8cd23f7c0d38da047ba14b5ac0801f2; .spec-seen now records it.
No numerical code, tests or factor changes are needed. Leave D-029's status
maintainer-owned. Finish round 2's source-fix review before invoking round 3
on the approved contract/records; do not exceed three rounds for RMST.3.

## RMST.3 round 2 preparation

Round 1 at ab7afb1 returned VERDICT: merge. Record S-1 as D-029 and ask
whether to clarify 12.1 in log-parameter units; do not edit the spec until
answered. N-1: narrow the old Weibull invalid-cutoff tangent assertion to
valid entries only, matching the confirmed unclaimed NaN-own-tangent rule.
Apply the same correction to the analogous Exponential test. The independent
precision-failure fix is already committed at 106078d. Run those focused
isolation tests per dtype, record round-1 outcome and final gate evidence,
commit records/test corrections and invoke round 2 on the fix and N-1.

## RMST.3 precision-failure isolation fix-up

A post-commit probe found a spec 8.7 isolation violation outside the accuracy
range: float32 k=[2,0.01], b=1.25 shared, tau=[0.75,0.01] gives a valid
first value and a NaN failed second value, but the first reverse scale derivative
is NaN (forward is finite). The gamma probability underflows; the inactive
exp(log_value) then overflows because its substituted probability was one.
Protect the exponentiation operand before it runs in src/survjax/_numerics.py;
keep NaN as the failed value. Add a dtype-specific regression for failed-entry
shared-scale isolation (k=0.005 in x64). No floors or tangent rule changes.
The extra source/test lines keep this step below about 300 numerical lines.
Validation: failed-entry first/second derivatives pass in both dtypes; the
fresh-wheel A10 gate passes with own 1080/7 and 1086/1, doctests four
per dtype, acceptance 2332/28 and 2322/38 passed/skipped. Record results,
commit a fix-up, then request
round 2 only after the current round 1 report is written.

Initial implementation evidence at ab7afb1: 198 new derivative-test lines, 10 RMST tolerance
lines and nine changed lines in the existing zero test (217 numerical-test/
tolerance lines total). Focused tests pass 31 per dtype. The fresh-wheel A10
gate exits 0: own 1079/7 x64 and 1085/1 x32 passed/skipped, doctests four
per mode, independent 2332/28 x64 and 2322/38 x32. No algorithm change.
Commit and independent review remain required before core handoff.

# RMST.appendix — Permanent mathematical appendix and conservative guarantees (2026-10-30)

## Round 2 fix-up plan

Maintainer explicitly confirmed S-1b: a NaN RMST result's own derivatives may
be unspecified. Record the confirmation verbatim in D-028 and REVIEW.md;
keep valid-entry isolation mandatory and leave deviation status maintainer-owned.
F-1: extend both existing RMST Docs to-do records with the zero-boundary and
NaN-own-tangent limits. F-2: add the missing gamma-middle section pointer.
N-1: restore the mathematical reason the 24b example needs a wider dtype;
rebuild and visually inspect the adjacent PDF. N-2: label the old report
reference historical. No numerical/spec/API changes; current revision 38
already says what the maintainer approved. S-1a's relation is symmetric;
no further spec wording change is needed to preserve the approved meaning.
Run static/whitespace checks and docs builds, commit the fix-up and invoke
round 2/3 exactly before continuing with the authorized RMST.3 step.

Maintainer approved the code/report and requested conservative guarantees,
a permanent math-only appendix outside docs/, tied to code, then the next
implementation step. Start clean release-rmst at
1a007fceb52ef1303cd739f14b58ce9d67f1cce9. Preserve unfinished PLAN history.

One documentation/contract concept; no numerical behavior change:

- Clarify spec 8.7/12.1: Weibull/PH-Weibull zero-cutoff guarantees cover
  value and first derivatives only; higher guarantees require valid smooth
  interior inputs. A NaN result's own tangents have no staged guarantee;
  valid-entry derivative isolation remains mandatory (also settles S-1).
  Bump revision 37 to 38 and commit this alone in the design repo. Its
  tracked files are clean; leave unrelated untracked .claude/ alone. The
  launch authorizes branch creation: use release-rmst there too, avoiding
  main. Mark this dedicated worktree via spec-since.sh dev-release-rmst
  --mark; literal dev would mark a different checkout. Commit .spec-seen here.
- Move/refine reports/weibull-rmst-math.md to appendices/weibull-rmst.md:
  retain the mathematical derivations and bounds; remove commit/step/review
  context, API proposals, empirical evidence and approval questions. Add
  appendices/README.md as the list of mathematical appendices with code
  pointers. Keep an adjacent rebuilt PDF; retire the old report/PDF paths.
- Add stable appendix-section references beside the three code branches
  and zero boundary; correct Weibull.rmst's boundary docstring. Do not
  alter its numerical expression, exports, wrappers or public docs pages.
- Record the maintainer decision in D-028 while leaving its status field
  maintainer-owned. Overwrite REVIEW.md with the contract and appendix map.

Aim at about 300 changed text lines, using a rename/extraction rather than
copying the historical report. Pandoc/XeLaTeX compilation, all-page Poppler
visual inspection, Ruff/mypy, focused RMST tests in separate dtype processes
and scripts/build_docs.sh (approved escalation for kernel cleanup) validate
this step. Commit and invoke independent-review exactly as RMST.appendix;
fix factual findings, then continue to the explicitly authorized next step.
No merge, rebase, push, tags or publication.

Next implementation: RMST.3 full-range derivative verification. Write its
concrete bounded plan after this appendix step; measure value-scaled error
factors before choosing any new RMST derivative tolerances. Keep the native
shape-Hessian restriction and code ownership; PH closure still waits for
reviewed core integration and handoff. Split evidence if about 300 lines
would be exceeded. No generic quadrature tangent or summary options/API.

---

# RMST.2-report — Mathematical approval report (2026-10-30)

Requested before maintainer approval of _weibull_rmst; start clean at
b3340749469e1513938a0cd5140f45305ab0d4bb on release-rmst. Preserve all
unfinished PLAN history. This is a maintainer report, not the later user
RMST guide or RMST.3 implementation. No production/spec/export changes.

- reports/weibull-rmst-math.md: one PDF-ready mathematical report, about
  300 source lines maximum. Derive the gamma identity, degree-12 series,
  remainder bounds, the mean-tail Chernoff bound and threshold choices;
  explain log arithmetic, safe operands, promotion, derivative limitations,
  measured evidence versus proof, and the staged decision against rmst_grid.
- output/pdf/weibull-rmst-math.pdf: compile with Pandoc/XeLaTeX; keep the
  editable source and exact build command in the report. Render pages with
  Poppler and inspect every page, extract text/check compile warnings.
  Temporary render/LaTeX products go under tmp/pdfs, not into the commit.
- DEVIATIONS.md: record D-028, discovered while explaining the zero branch:
  at k=1,b=2,tau=0 the right second cutoff derivative is -1/2 but autodiff
  returns 0. Ask whether 8.7's higher nonshape derivative promise includes
  existing higher right derivatives at the zero boundary. Leave status open,
  with no implementation/spec decision. Finishing the requested explanatory
  artifact does not depend on that decision; stop/report it afterward.
- REVIEW.md: overwrite for this report step, round 1/3; reading order,
  mathematical assumptions, factual checks, build/render evidence and limits.

References: spec 8.7/12.1; implementation at b334074; existing RMST tests and
independent-review report; primary NIST DLMF 8.2/8.7 definitions. No new
library/build dependencies. Existing Pandoc, XeLaTeX and Poppler are present.
Run scripts/build_docs.sh as the documentation regression, static checks and
whitespace checks; numerical report probes are separate from production tests.
Report completed: 269 Markdown lines plus D-028, 5 compiled/visually checked
PDF pages. Bound constants and zero-boundary probe verified at 60 digits /
x64 respectively. Final XeLaTeX log has no layout/font warnings; docs builds
pass in both modes after approved escalation removed sandbox cleanup errors.
Production/spec/tests/docs sources and project metadata remain unchanged.
REVIEW.md records exact checks, PDF identity, warning history and limits.
Commit report/source/PDF and records; invoke independent-review exactly for
RMST.2-report; act on factual/layout findings only, route spec questions to
maintainer, then stop.

Review round 1 on 6448c46: maintainer; proofs/constants/PDF verified.
S-2 expands D-028 (k=2 third derivative; k<1 divergent second derivative).
The user's loop question arrived during that review: add the explicit JAX
staging/count/option discussion with official documentation and a local
Jaxpr check; correct factual nits N-1/N-2. Final report is 289 lines / 5 PDF
pages, recompiled and visually checked. Follow-up additions are not separately
reviewed because the verdict requires a maintainer decision, not another
round seeking approval. N-3 artifact repository placement remains a maintainer
integration choice. Commit this bounded report update; stop and ask D-028.
No production, spec, test or website source change. No RMST.3 work, merge, push or publication.

---

# RMST.2 — Weibull RMST values and safe tails (2026-10-29)

Authorized by “ok for now; on to the next bit”. Continue clean release-rmst
at 01b29dc9e0ea5830580770c0183865cb159fc376; original workstream base stays
8757c0fda78e34e6e818249adedd2a144a33c33d. Same unfinished RMST sub-slice;
keep RMST.1 and unfinished A9/A10 history below, with no archive commit.
RMST.1 round 2 returned merge. Carry S-1 without a behavior change.

One concept: staged Weibull RMST values, with safe lower/upper tails. Target
at most about 300 changed source/test lines; split evidence before growing.
Spec: 4.5, 5, 7.2–7.3, 8.7, 10.2, 12.1–12.2. No dependency, export,
wrapper, docs, protected-path or generic quadrature changes.

- Create src/survjax/_numerics.py, explicitly named by spec 10.2 (not an
  unnamed module). Only shared RMST cutoff dtype/shape validation and the
  private Weibull value expression belong here; no configuration/backend API.
  Reuse validation in Exponential without changing its formula or policies
  (review N-2). Keep invalid substitution in the public family methods.
- Weibull.rmst(tau): thin typed method, parameter-issue/invalid-cutoff NaNs,
  safe inputs before arithmetic. Keep weibull.py around 300 lines (275 now).
  For log x=k*(log tau-log b), use the integrated exponential series for
  x<0.1 (12 terms; next term <=0.1**13/13!); incomplete gamma in the middle;
  the mean for x>64+4/k, safely classified in log space. The last threshold
  is well beyond the gamma tail over the required k range. Evaluate every
  inactive branch on finite safe inputs. Zero returns the active cutoff,
  preserving d/dtau=1 and zero parameter derivatives. No custom JVP.
- New tests/unit/test_weibull_rmst.py: independent mpmath quadrature values
  across k=[0.2,50], tau/b=[0.01,1e5], scale changes, explicit power underflow
  and overflow; k=1 identity, scalar and cutoff-grid broadcasting, dtype/shape
  misuse, eager/jit/vmap, zero/invalid/parameter-issue policies, first
  derivative and supported Hessian smoke checks, shape-Hessian failure.
  Existing central tolerances only; no unmeasured tolerance additions.
  Final audit adds common-dtype promotion plus its mixed-float32/float64
  regression (9 lines): total 309 source/test lines, one numerical concept.
- Overwrite REVIEW.md with round 1/3, reading order, invariants, evidence map,
  actual commands/results and remaining RMST.3 derivative-range evidence.

Checks: focused RMST/Weibull/Exponential suites in separate x64 and float32
processes; Ruff format/check, mypy, unchanged fresh-wheel through-A10 gate.
Inspect diff/size/scope; commit; use independent-review exactly for RMST.2,
follow up to three rounds and stop. No further step or PH handoff yet.

Implementation: 309 source/test changed lines after the mixed-dtype fix.
Weibull 300 lines, private numerical module 70 lines; no new tolerances.
Final focused four-module suites: 57 passed/1 existing skip (x64), 58 passed
(x32). Ruff: 114 files formatted, all lint checks pass; mypy: 21 files clean.
Final fresh-wheel through-A10 gate: exit 0; own tests 1069/7 skipped and
1075/1 skipped, doctests 4 per mode, independent 2332/28 skipped and
2322/38 skipped. REVIEW.md records exact commands/timings, the initial
pre-correction gate, corrected artifact and mixed-dtype mutation proof.
Initial implementation commit b552201. Round 1 returned fix-first for F-1:
missing tail-gradient assertions. Add 10 test lines (no runtime change),
checking finite first derivatives in every direction and [1,0] cutoff
slopes at the two power tails under jit-vmap, jacfwd and jacrev. Focused
suites 21 passed per dtype; both unsafe-input mutants now fail in both
modes; statics clean. N-1's promised value-range/float32 limitation is in
Docs to-do below. Commit the fix-up, request round 2/3 and follow its verdict.

Docs to-do: RMST topic after core/PH integration, stating the Weibull shape
higher-derivative limit, first-order-only zero boundary, unspecified own
derivatives for NaN results, and absolute-scale cutoff-derivative accuracy.
Accuracy is promised on k in [0.2,50], tau/b in [0.01,1e5]; outside that
range, float32 can return NaN for valid inputs when the gamma probability
underflows (review RMST.2 N-1).
RMST.3 still owns full-range measured derivative tolerances/evidence (N-1).
PH owns wrappers.py/exports and waits for the reviewed Weibull core merge
and explicit ownership handoff before private built-in closure edits.

---

# RMST.1 — Staged base surface and Exponential RMST (2026-10-29)

Authorized release workstream on `release-rmst`; actual clean base:
`8757c0fda78e34e6e818249adedd2a144a33c33d` (8757c0f).
Preserve unfinished A9/A10 history below; no archive of unfinished work.
One concept, at most about 300 source/test lines. No dependencies, exports,
protected paths, docs, wrappers or Weibull edits in this step.

- `src/survjax/distributions/base.py`: add spec-named `rmst(tau)`, `mean()`,
  `median()` and `quantile(p)` methods raising NotImplementedError. The RMST
  default covers unsupported families; Weibull remains staged until RMST.2.
- `src/survjax/distributions/exponential.py`: override only `rmst(tau)` with
  `-expm1(-rate*tau)/rate`. Validate dtype/shape statically, substitute invalid
  cutoff/parameter entries before arithmetic, return NaN for those entries.
  Keep tau=0 active so value=0, parameter derivatives=0, tau derivative=1.
- `tests/unit/test_rmst.py`: public surface/absence, unsupported custom family,
  misuse, zero and invalid values, scale changes, explicit cutoff axes,
  eager/jit/vmap, independent mpmath integration/derivatives, forward/reverse
  Hessians and higher derivatives, second-order check_grads and invalid-entry
  derivative isolation (shared and per-row parameters). Use existing central
  value/tail/derivative tolerances; add no tolerance without demonstrated need.
- `REVIEW.md`: reading order, invariants, spec-to-test map, exact checks/results,
  known staged limits and independent-review round count; overwrite now.

Spec read in full: 4.1–4.2, 4.4–4.5, 4.7, 5, 7.2–7.3, 8.1–8.7,
10.2, 12.1–12.2. Apply 8.7's explicit staged exception to 8.4.
Checks: focused RMST/distribution regression in separate x64/float32 processes;
Ruff format/check, mypy; unchanged fresh-wheel cumulative gate through A10.
Inspect diff and scope, commit, invoke exact independent-review skill for
RMST.1 in `../review/code-release-rmst`, act on at most three rounds, stop.

Docs to-do: separate executable RMST topic after core and PH integration;
state staged surface, Weibull shape-derivative limitation, first-order-only
zero boundary and unspecified own derivatives for NaN results. A9/A10 docs
remain with their assigned implementers. Ownership handoff: after reviewed
Exponential AND Weibull core merges, release-ph may edit built-in private
closure hooks and integrate wrapper RMST in its own reviewed step. No core
handoff yet after RMST.1; wrappers.py and exports remain PH-owned.

Implementation/evidence: exactly 300 source/test changed lines; no new
central tolerances. Final focused RMST: 13 passed per dtype. Distribution
regression: 140 passed/3 existing skips (x64), 143 passed (x32). Ruff: 112
files formatted, all lint checks pass; mypy: no issues in 20 source files.
Fresh-wheel unchanged through-A10 gate: exit 0; own tests 1061/7 skipped
and 1067/1 skipped, doctests 4 per mode, independent tests 2332/28 skipped
and 2322/38 skipped. Exact commands, times, environment, artifact identity
and final-source limits are in REVIEW.md. Initial commit 944582f.

Round 1: fix-first for F-1, missing cutoff-derivative mutation coverage.
Add an 18-line test-only fix-up; no runtime changes. Both dtype suites now
14 passed, and the safe-cutoff-removal mutant fails the new test in both
modes. Final statics clean. N-1/full-range evidence remains RMST.3;
N-2/validation reuse remains RMST.2. S-1 invalid-entry tangent semantics
is reserved for the maintainer; no implementation change for it. REVIEW.md
records the reviewer gate's missing-wheel environment and artifact cleanup
limits. Commit this bounded follow-up and request round 2/3, then stop.

---

# A10.4 — Synthetic R concordance references (2026-10-29)

Authorized by “plan and implement A10.4”. Work on maintainer-created `A10.4`
from clean `e8dbbed`. One concept: committed synthetic Harrell-C reference
data and comparisons, with no public API or production behavior change.
Spec: 6.1, 9.1–9.2, 10.4, 12.1. Target about 300 changed lines including
generated payloads, excluding plan/review records; split if it grows beyond
that. Existing A10 planning remains below because this is the same sub-slice.

Files to create/change:

- `scripts/fixtures/generate_concordance.R`: require R 4.4.2, survival 3.7-0
  and flexsurv 2.3.2 before output; fixed seed/RNG, synthetic cohorts only.
  Base-R export (no dependency added), optional output directory/check-only.
  Evaluate `survival::concordance` with `reverse=TRUE`, `timewt="n"`,
  `timefix=FALSE`. Include 100 heavily tied rows covering all time-tie kinds,
  plus untied positive times with >50% censoring and perfect/reversed scores.
- `tests/fixtures/concordance/{data.csv,concordance_synthetic.json,SHA256SUMS,README.md}`:
  CSV observations/scores and the already documented independent-test JSON
  schema; record versions, seed, generator hash and exact R call. Regenerate
  twice into temporary directories and compare every payload byte. Explain
  R's count mapping and exact ties, with no convention difference assumed.
- `tests/conftest.py`: extend the existing shared manifest-checked loader
  for this set, preserving CSV/JSON row/score alignment and provenance.
- `tests/reference/test_concordance_r.py`: compare R values and all four
  count pairs/masks for prepared/raw, eager/jit/vmap calls in both dtypes;
  existing central tolerances only, exact counts. Missing fixtures fail.
- Overwrite `REVIEW.md` with reading order, invariants, requirement mapping,
  actual commands/results and limits; update this section with evidence.

The independent loader requires its copy under read-only `tests/spec/fixtures/`.
Produce the matching export in our fixture directory for maintainer delivery,
as with A8. Do not edit protected paths or change its loader/selection. A full
through-A10 gate remains unavailable until the maintainer supplies that copy.
No spec/deviation/dependency/docs changes are needed. Docs to-do: the existing
concordance topic/playground and held-out example remain separate steps.

Checks: verify pinned R/package versions and installed concordance help; run
generator check-only and two independent generations, compare with committed
payloads and check hashes. Run focused own references/concordance in separate
dtype processes, Ruff, mypy and `scripts/check_wheel.sh --through A9`.
Probe all independent A10 modules in both modes and report the missing-file
failures explicitly. Inspect final diff/whitespace/scope, write REVIEW.md
with “Independent review: round 1/3”, commit on `A10.4`, then invoke the exact
independent-review skill command from `../review/code-A10.4`. Follow at most
three rounds, with separate fixes if needed, then stop and report.

Planning evidence: clean branch/status, current A10 outline and earlier
review read; normative sections above, shared fixture loader, both existing
reference sets and independent fixture schema read. R reports 4.4.2,
survival 3.7.0, flexsurv 2.3.2. The official concordance manual confirms
orientation, weighting, timefix and five count fields; pinned installed help
will govern generation. No implementation or tests before this plan.

Implementation completed: exactly 300 changed implementation/test/fixture
lines, including 105 CSV lines; public API and production code unchanged.
Pinned R generated 104 synthetic observations in two cohorts, three score
draws each. Both independent regenerations and committed payloads are
byte-identical; SHA-256 checks pass. All six R outputs also match a separate
pair enumeration of the spec rules, including all three time-tie kinds.
Focused own concordance/reference suites: 77 passed in 27.85s (float64),
77 passed in 27.86s (float32), no skips. Ruff: 104 files formatted / All
checks passed; mypy: no issues in 20 source files. Independent A10 probe:
8 failed, 113 passed (57.50s) / 8 failed, 112 passed, 1 existing skip
(59.90s); all failures are the absent protected fixture, before assertions.
REVIEW.md records commands, the corrected working-directory error and limits.
The matching JSON is ready for maintainer delivery; no full A10 pass claimed.
Fresh-wheel `scripts/check_wheel.sh --through A9`: exit 0, all gates passed.
Own suites: 1048 passed, 7 skipped / 1054 passed, 1 skipped; source doctests:
4 passed per mode; independent through-A9: 2211 passed, 28 skipped / 2202
passed, 37 skipped. Existing skips only. REVIEW.md records timings and wheel
hash. Final scope/whitespace/static audit is clean; commit and round 1 follow.

---

# A10.3 — Public concordance transformation and permutation evidence (2026-10-29)

Status: **F-2 fix-up verified; maintainer-authorized round 4 pending**, following
"implement and review (automatic via claude, use the skill) the next part of
A10" and the subsequent "approved". Work on the maintainer-created `a10b`
branch from clean `a5cfea7`.
The reviewed A10.2 API and main/A9 integration are present. The next outline
item is A10.3, before the separate A10.4 R-fixture step below.

Completed implementation: two new own-test modules, 231 lines, seven tests.
Focused concordance suites: 72 passed per dtype (21.99s float64 / 22.07s
float32), no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited
0, ending `check: all gates passed through A9`: own suites 1043 passed,
7 skipped / 1049 passed, 1 skipped; source doctests 4 passed per mode;
independent through-A9 suites 2211 passed, 28 skipped / 2202 passed,
37 skipped. Five non-fixture independent A10 modules: 113 passed (22.55s)
float64 / 112 passed, 1 skipped (30.07s) float32. Ruff: 92 files formatted,
All checks passed; mypy: no issues in 20 source files. Existing skips only.
Exact commands, timings, wheel identity, initial cache/lint failures and
limits are recorded in REVIEW.md. No production code changed; no full A10
claim. Final staged audit, commit and automatic Claude review follow.

Round 1 on fc7f82c returned fix-first for should-fix F-1 (no blockers/nits).
Add prepared-data static-metadata compilation evidence: equal flags reuse
one trace despite changed data/scores; changing flags produces another.
Retain raw tracing coverage with an accurate description. Final source/test
total: 260 lines, eight tests. Focused own suites now 73 passed per mode
(12.16s float64 / 12.22s float32), no skips; Ruff formatting/lint clean.
The earlier full gate/A10 selection are initial-commit evidence, not reruns
after this test-only follow-up. REVIEW.md records the distinction and the
reviewer's inability to execute checks under its headless permissions.
Commit the fix-up and request round 2 using the same skill; then stop.

The maintainer requested round 3 after correcting reviewer permissions.
Round 3 on bd1bd84 could execute checks/mutants and returned fix-first for
F-2: the third prepared-data call changed weak_type as well as metadata,
so a constant-metadata mutant survived. Use an explicit all-True event
array on that call, matching the other constructor paths. Final focused
suites: 73 passed per mode (13.31s float64 / 13.33s float32), Ruff clean.
The corrected test catches the constant-metadata mutant in both modes in
isolated temporary source copies. Source/test total: 263 lines, eight tests.
REVIEW.md records the reviewer's gate results (including missing-fixture and
unsupplied-wheel failures), mutant results, deferred N-1 and reviewer-owned
temporary worktree cleanup. Commit the F-2 fix-up and stop at three rounds;
no merge approval or full A10 claim, and no automatic A10.4 start.

The subsequent "do another round of review" explicitly authorizes one
additional independent review beyond the three-round limit. Review the
unchanged F-2 fix-up from 491f7cc; record round 4 and follow its verdict.
This request does not authorize starting A10.4 or changing protected paths.

One concept: complete own public-API evidence for transformations and
observation permutations. Aim for at most 300 changed source/test lines.
Public API and implementation behavior: unchanged. Spec sections: 6.1–6.3,
7.2 (compilation reuse), 9.1–9.2, 12.1 (transformations and metric shapes).

Files to create or change:

- Create `tests/unit/test_concordance_transformations.py`. Use distinct,
  hand-enumerated tied-time/tied-score cohorts, with counts and fractions
  derived in comments. Check `(B,1,N)` data against `(D,N)` scores, using
  different B and D lengths; compare eager, jit, nested vmap and jit-of-vmap
  to those expectations for values, all eight count words and all masks.
  Cover prepared inputs and raw construction inside the compiled function.
- In that module, test transformed invalid and unsupported cohorts beside
  valid and evaluated zero-pair cohorts. Check cohort isolation, preserved
  overlapping masks, zero failed counts and result shapes against explicit
  expectations. Add a trace counter showing equal-shape/dtype changes to
  scores, times and event values reuse compilation. No score derivatives.
- Create `tests/property/test_concordance.py`. Reproducible Hypothesis cases
  (`derandomize=True`) use small positive integer times, boolean events and
  bounded integer scores so ties and exact arithmetic are frequent. Verify
  joint row-permutation invariance of the complete result, count partition
  (`comparable = concordant + discordant + tied_score`), and score negation
  swapping concordant/discordant while preserving ties, masks and C's
  complement when defined. Cover no-pair NaNs explicitly. Identity-based
  expectations must not use production helpers or independent test helpers.
- Update this plan with actual results and overwrite `REVIEW.md` with the
  reading order, invariants, spec/test mapping, checks and limits. Retain
  historical plan records below. No changes to spec, `.spec-seen`, docs,
  protected tests/scripts, fixtures, dependencies or likelihood code.

Verification after approval:

- Run all own concordance modules, including both new modules, in separate
  `JAX_ENABLE_X64=1` and `=0` processes. Use existing central tolerances for
  non-exact floating identities; compare count words and masks exactly.
- Run Ruff format/check and mypy, then the unchanged
  `scripts/check_wheel.sh --through A9` for the combined-tree regression and
  installed exports. Run the five non-fixture independent A10 modules in
  both dtypes with `SPEC_THROUGH=A10`, without changing their selection logic
  or contents. The missing concordance R fixture precludes a full A10 claim.
- Inspect whitespace and the final diff; commit this step on `a10b` with
  `REVIEW.md` recording `Independent review: round 1/3`. From a clean tree,
  invoke the exact independent-review skill command in
  `../review/code-a10b`, step `A10.3`, and read `../review/slices/A10.3.md`.
  Follow its verdict, with at most three rounds and separate fix-up commits
  for blocker/should-fix findings; stop after this step and report.

Docs to-do: existing concordance topic/playground and held-out synthetic
example remain later steps. A10.4 owns pinned-R generation/reference tests.
Planning checks: read branch/status/history, the A10 outline, current public
metric and own/independent shape/property coverage, shared test fixtures and
spec sections above. No implementation, tests or review run during planning.

---

# A10 integration with merged main/A9 (2026-10-29)

Authorized by "main had moved and i've merge; check merge conflicts and
bring up to date; A9 has been merged now". The maintainer-initiated merge
is in progress: A10-cindex HEAD f71140a, incoming main ff9786f.
Five files conflict. This is merge resolution/integration of reviewed work,
not a new concordance feature or spec decision. Snapshot the initial index
and preserve every incoming non-conflict change except the two explicitly
listed integration edits below. Leave the merge commit to the maintainer.

Files to resolve: PLAN.md, REVIEW.md, src/survjax/__init__.py,
src/survjax/likelihood.py and tests/unit/test_import.py.
Necessary integration edits: src/survjax/distributions/base.py and
tests/install/test_wheel.py. No edits to spec, .spec-seen, independent
tests, protected scripts, incoming dependencies, fixtures or docs.

- Preserve main's method-only likelihood API and the two public A10 exports.
  Remove the obsolete standalone likelihood wrapper. Keep input-form
  normalization in _observation_inputs.py and import that same helper from
  the distribution method and metrics; do not duplicate the old resolver.
- Keep all incoming numerical likelihood behavior, A8 references, A9
  transformation/wheel tests, fixture handoff and spec alignment. Keep the
  withdrawn standalone A9 example absent. Extend the installed-wheel test's
  exact expected export set by the two reviewed C-index functions.
- Retain both branches' planning history below this current integration
  record. Overwrite REVIEW.md with current reading order, invariants,
  parent records, validation results and remaining A10 work.
- Sync the environment from the incoming lockfile without changing project
  metadata. Run focused integration checks, then scripts/check_wheel.sh
  --through A9, including a freshly built wheel, full own suites and unchanged
  independent through-A9 suites in both dtypes. Run the five non-fixture A10
  acceptance modules in both dtypes. A10 R fixtures remain a separate step;
  do not claim a full through-A10 gate or regenerate R data.
- Verify conflict markers/unmerged entries are gone, all seven intended
  resolution paths are staged, every other index entry matches the snapshot,
  and protected incoming files equal main. Record results and stop with the
  merge resolved and staged; no new merge, rebase or merge commit.


Validation completed 2026-10-29. All five conflicts are resolved/staged;
main's method-only API regression, the two C-index exports and the one
shared resolver are preserved. The incoming likelihood evaluator AST is
unchanged apart from moving its identical input resolver; only the method's
import changes. All other 173 original stage-0 index entries are unchanged.
Environment synced with uv sync --locked --offline --dev; no metadata edit.
Focused shared-input/likelihood/concordance/API suites: 120 passed per mode
(12.49s float64 / 11.93s float32). Fresh-wheel full gate:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check_wheel.sh --through A9`
exited 0, ending `check: all gates passed through A9`.

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests, including installed wheel | 1036 passed, 7 skipped, 56.62s | 1042 passed, 1 skipped, 59.24s |
| Source doctests | 4 passed, 1.09s | 4 passed, 1.10s |
| Independent through A9 | 2211 passed, 28 skipped, 60.17s | 2202 passed, 37 skipped, 53.97s |
| Five non-fixture independent A10 modules | 113 passed, 28.26s | 112 passed, 1 skipped, 21.81s |

Ruff: 89 files already formatted / All checks passed! Mypy: no issues in
20 source files. No source/test changes followed these runs. Complete
commands, wheel identity, selection and remaining limits are in REVIEW.md.
Net-to-main and resolution-only whitespace checks are clean. The full staged
check reports only two pre-existing incoming-main EOF blank lines in
read-only tests/spec/API_ASSUMPTIONS.md and test_a6_likelihood_shapes.py;
confirmed identical in main and left unchanged. No new whitespace issue.
The incoming .spec-seen remains be26e50. Subsequent design changes only add
use-case companion links; concordance/likelihood requirements are unchanged.
Final metadata restaging/index audit follows; leave the merge commit to
the maintainer under the repository agreement. No full A10 gate claim.

Spec: 6.1, 7.1-7.3, 9.1-9.2, 10.2-10.3, 12.1. Public API is the current
method-only likelihood plus the already reviewed concordance functions.
No new tests beyond updating existing exact export expectations are needed
for this merge; use both branches' existing transformation/acceptance suites.
Docs to-do: concordance topic/playground/held-out example, as recorded below.
Earlier missing A8/A9 integration evidence is historical; update current
results here after the combined-tree checks have actually run.

---

# A10.2b2 review follow-up — N-5 and N-8 (2026-10-28)

Review follow-up to the approved public integration step, requested by
"check the nits in ../slices/A10.md" and the standing "check the review and
address" instruction. Read the actual report at ../review/slices/A10.md,
round 4 for 8109470: merge verdict, no blockers or should-fix findings.
Work on A10-cindex from clean 8109470; one test-only fix-up commit, then stop.
No new public API, numerical behavior or spec decision.


Completed 2026-10-29: N-8 adds three eager/jit unsupported-mode cases; N-5
adds distinct ClosedJaxpr descent after the Jaxpr case. Only 38 test lines
were added; production code is unchanged. Focused five-file selection:
67 passed in 6.56s (float64), 67 passed in 5.83s (float32), no skips.
Ruff: 78 files already formatted / All checks passed. Mypy: no issues in
20 source files. Both reported unsupported-mode mutants are caught in both
dtypes: E1 fails right-trunc only (1 failed, 2 passed); E2 fails interval
and left-censored (2 failed, 1 passed). Each mutation run deselects 23 tests.
REVIEW.md records exact selections, timings, scope and the untested distinct
ClosedJaxpr representation. Final whitespace/staged/scope checks follow.
Ready for the one test-only fix-up commit and review stop.

Files: tests/unit/test_concordance_entry_points.py,
tests/unit/test_concordance_counts.py, PLAN.md and REVIEW.md.
Spec: 9.1 (unsupported modes), 9.2 (whole cohorts and workspace), 12.1.

- N-8: add three parametrized from_bounds cases: finite right truncation,
  interval observation and left censoring. Each unsupported cohort has a
  valid neighbor with hand-counted C=3/8 and counts (1,2,1,4). Check eager/jit
  agreement, unsupported-only masks, zero failed counts, and no evaluated
  zero-pair flag. Inputs must be valid observations; unsupported is distinct
  from invalid. Preserve the existing left-truncation coverage.
- N-5: descend through distinct ClosedJaxpr wrappers in the recursive own
  workspace guard, after the Jaxpr case. Locked JAX 0.11.2 aliases the two
  classes; handle that without recursing on Jaxpr.jaxpr repeatedly. Do not
  claim other JAX versions were executed.
- Reconcile the review's focused-count difference: its four files total 63;
  the previously reported 64 also included test_api_reference.py. Keep the
  original evidence intact and record the selection explicitly in REVIEW.md.
- Run the five focused own modules in separate dtype processes, Ruff, mypy
  and whitespace/protected-scope checks. In temporary source copies only,
  remove each unsupported-mode term in turn and require the new cases to
  catch both mutants in both modes. Do not mutate the working source.
- This is test-only: production runtime is unchanged from the passing public
  integration gate. No full regression/independent A10 rerun is necessary;
  record that limit honestly. No fixture, docs, dependency, protected test,
  .spec-seen, sync or likelihood changes. No merge or next subslice.

---

# A10.2b2 — Public concordance entry points (2026-10-26)

Status: **implemented and verified; ready for review**, authorized by
"review approved ... get on with next subslice". Work in dev-a10 on A10-cindex from clean `715a7e3`; no merges.
The maintainer also explicitly allows the two API-reference entries needed by
the export-coverage test, as a narrow exception to the code-step docs rule.
Tutorials and broader docs remain separate. The shared spec is now revision
37; its concordance contract is unchanged. Likelihood alignment/spec sync
remain owned elsewhere; do not change .spec-seen or that API here.


Final evidence (completed 2026-10-27): both public entry points and their
exports are implemented. Own focused suites pass 64 tests per dtype
(6.87s float64 / 6.27s float32), no skips. The five selected independent A10
modules pass 113 tests in float64 (45.13s), and 112 with 1 existing skip in
float32 (47.97s). This is focused A10 evidence, not a full A10 gate.

The initial through-A7 attempt failed only the old exact export smoke list
(1 failed, 848 passed, 7 skipped); adding the two intended names fixed it.
REVIEW.md records that failure and correction. The final unmodified
through-A7 gate exited 0, ending `check: all gates passed through A7`:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 849 passed, 7 skipped, 171.12s | 855 passed, 1 skipped, 172.98s |
| Source doctests | 4 passed, 1.03s | 4 passed, 0.97s |
| Independent through A7 | 2123 passed, 28 skipped, 223 deselected, 121.84s | 2127 passed, 24 skipped, 223 deselected, 116.32s |

Ruff: 78 files already formatted / All checks passed. Mypy: no issues in
20 source files. No new skips or exclusions. Optional complete output:
`/private/tmp/survjax-a102b2-final-gate.log`; this table is the durable summary.
REVIEW.md was finalized before the final gate and is unchanged afterward.
Source/test changes total 291 lines, plus seven approved API-reference lines;
the metrics module is 220 lines. Final static/whitespace/scope checks follow
this plan-only result edit. No merge, spec/sync marker, independent-test,
likelihood API or dependency changes; no full A10 completion claim.
Stop after this step's commit for review.

One concept: connect checked observations and caller scores to the existing
counter/result layer, with public exports. Target at most 300 source/test
changed lines; split if that budget is exceeded.

- Update metrics.py with concordance_index and concordance_index_with_info
  using the shared _resolve_data helper and one private cohort evaluator.
  Accept real JAX-supported score arrays without casting their precision to
  the data dtype. Reject static type/shape misuse; invalid runtime values give
  NaN plus masks. Recheck current data fields and retain construction issues.
- Require matching non-scalar last observation axes; broadcast only leading
  axes. Reduce invalid/support evidence per cohort and explicitly broadcast
  every mask to result shape (review N-7). Replace entire failed cohorts with
  safe finite arrays/boolean masks before counting; never drop observations.
  Flatten leading batches for vmap and restore result shapes. Empty cohorts
  remain evaluated zero-pair cases. Preserve data-float ratio dtype.
- Export both functions in __init__.py. Add only their autodoc entries and
  minimal section wrapper in docs/api.md, as approved. Rename result field
  docstring headings to Attributes (review N-6). Do not change other docs.
- Add tests/unit/test_concordance_entry_points.py: hand-counted raw/prepared
  values, all five shape-table rows with every info leaf checked, failed-
  cohort isolation/mask overlap, edited-field validation, empty/scalar/mismatch
  cases, preserved score ordering/precision and jit/vmap agreement.
- Update tests/unit/test_import.py's exact export set with the two intended
  additions; retain every existing name pending separate likelihood alignment.
- Update D-027's implementation note, this plan, and overwrite REVIEW.md with
  reading order, invariants, spec mapping, actual check outputs and limits.

Spec: 3.2–3.3, 6.1–6.3, 7.2, 9.1–9.2, 10.2, 12.1; accepted D-027. No new
helpers/dependencies outside this concrete evaluator, no score gradients.
Run new/existing concordance own suites plus the five non-fixture independent
A10 modules in separate dtype processes; static checks and the unmodified
through-A7 regression gate. A10 R fixtures/A8/A9 work remain unfinished;
selecting the five modules is focused evidence, not a complete through-A10 gate.
Verify only approved API-reference edits, protected paths and sync marker,
record results, commit this one step on A10-cindex, then stop for review.
Docs to-do: executable concordance tutorial, playground and held-out example;
no full docs-build claim for this code step's two reference entries.

---

# A10.2b1 — Concordance result containers and ratio (2026-10-25)

Status: **implemented and verified; ready for review**, authorized by
"review approved ... get on with next subslice". Work only in dev-a10 on A10-cindex, starting clean at
`10b8877`. Do not merge, rebase or touch the separately owned spec alignment,
test sync, .spec-seen, likelihood API or protected paths. The checkout retains
revision-34 API alignment while accepted spec revision 35 is visible through
the shared read-only link. This step is independent of the likelihood change.


Implementation evidence: a 100-line result module and 172-line test module
(272 source/test lines). Focused result/count suites pass, 39 per dtype
(2.71s x64 / 2.15s x32), including all 14 new result cases. No failures/skips.
The unmodified through-A7 gate exited 0, ending
`check: all gates passed through A7`:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 826 passed, 7 skipped, 145.51s | 832 passed, 1 skipped, 145.95s |
| Source doctests | 4 passed, 0.87s | 4 passed, 0.81s |
| Independent through A7 | 2123 passed, 28 skipped, 223 deselected, 96.59s | 2127 passed, 24 skipped, 223 deselected, 90.68s |

Ruff: 77 files already formatted / All checks passed. Mypy: no issues in
20 source files. No new skips or exclusions. Optional raw output:
`/private/tmp/survjax-a102b1-final-gate.log`; this table is the durable summary.
REVIEW.md was finalized before the gate and is unchanged afterward. Final
format/lint/mypy and working/staged whitespace/protected-path checks follow
this plan-only result edit. Both package/module imports resolve to dev-a10.
No spec, .spec-seen, independent tests, likelihood API, docs, dependencies,
lockfile, top-level exports or other branches changed. No merges or full
through-A10 completion claim. Stop after this step's commit for review.

One concept: turn already-counted pairs into transformable diagnostics and a
floating C-index. The full public integration exceeds one review-sized step;
implement this result layer first, with source/tests under 280 changed lines.

- Create `src/survjax/metrics.py`: frozen registered `ConcordanceCount` (hi/lo
  uint32 array fields) and `ConcordanceInfo` (the four accepted count fields,
  invalid, unsupported and no_comparable_pairs arrays). Result types stay in
  their module (spec 6.3/10.2); no top-level export or metric entry point yet.
- Add private `_concordance_result(counts, invalid, unsupported, dtype)` in
  that module. Inputs are aligned count words and per-result masks. Preserve
  exact uint32 words; zero all counts on failed cohorts, retaining both masks.
  Cast words into the data float dtype before reconstructing totals and
  dividing. Award half credit to tied scores. Substitute a safe denominator
  for zero-pair cohorts before division; return NaN with no_comparable_pairs
  only on evaluated zero-pair cohorts. No floors, epsilons or score gradients.
- Create `tests/unit/test_concordance_results.py`: hand-counted fractions,
  high-word totals, undefined/failed/overlapping-mask cases, per-result array
  shapes, preserved data dtype, frozen fields and pytree reconstruction;
  eager/jit/vmap agreement in both dtype processes. Expected counts/fractions
  come from explicit integer arithmetic, not code under test. Use existing
  central tolerances; do not add numeric tolerance constants.
- Update this plan, the implementation note in D-027 (retain formal open
  status), and overwrite REVIEW.md with reading order, invariants, requirement
  mapping, exact commands/results and limits. No docs or dependency changes.

Spec: 3.3 (floating division), 6.3 (result pytrees), 7.2 (safe operands),
9.1–9.2 (counts/ratio), 10.2 (result placement), 12.1; accepted D-027 contract.
Validate focused result/count suites in separate float64/float32 processes,
Ruff format/lint, mypy, and the unmodified through-A7 regression gate. Verify
imports use dev-a10. No through-A10 completion claim or independent-test edits.
Record actual results, inspect staged/protected diffs, commit and stop.

Docs to-do: retain the later concordance topic/API reference, adding named
hi/lo counts and result-mask interpretation once public integration is reviewed.
Next step: public observation/score validation, cohort-safe inputs, leading-axis
broadcasting and entry points, after coordinating pending alignment and the
code-step/API-reference documentation constraint. No automatic next step.

---

# A10 review follow-up — N-1 through N-4 (2026-10-25)

Status: **implemented and verified; ready for review**, authorized by
"check the review and address". Read all of `../review/slices/A10.md` for
`04bc1a7`: merge verdict, no blockers or should-fix findings. One follow-up
commit on `A10-cindex`, then stop; starting tree was clean at `04bc1a7`.

Files: `src/survjax/_concordance.py`, its own test module, PLAN.md,
DEVIATIONS.md and REVIEW.md. Public API: none. Source/test budget: under 60
changed lines. Spec: 9.1–9.2 workspace/preconditions and 12.1 verification.
No numerical, dependency or docs changes; the shared resolver is unchanged.

- N-1: document the boolean (N,) exact-mask precondition.
- N-2: inspect the traced JAX program recursively, including loop bodies, for
  an N-by-N intermediate at N=7. Verify sensitivity in a temporary source
  copy using an equivalent N-by-N comparison inside the loop; require the
  shape test to fail and the hand-counted value test to pass in both dtypes.
- N-3: require finite safe substitutes for entire invalid/unsupported cohorts
  before public counting. Retain cohort length/masks and zero failed counts;
  never drop rows. Record this in the later public integration outline.
- N-4: remove the redundant high-word assertion/import, make stale merge text
  historical, and commit complete check commands/results in REVIEW.md.
  Temporary logs supplement that durable evidence; no logging infrastructure.
- Record the maintainer's accepted D-027 decision, retaining its formal open
  status for maintainer disposition. No public result type implemented here.
- S-1: revision 35 alignment, spec-test sync and `.spec-seen` belong to another
  agent/branch. Do not edit, mark, merge or sync them here. S-2's early A10
  authorization is recorded; no delivery/role edits. Reconcile dependencies
  after those changes land, before starting public integration.

Verification: focused counter/shared-input/likelihood entry-point suites pass,
93 tests per dtype (5.90s x64 / 5.40s x32), no skips. The N-by-N mutant is
rejected while its hand-counted value test passes: 1 expected failure / 1 pass
in 0.21s per dtype. REVIEW.md records commands and sensitivity details.
Final Ruff: 75 files already formatted / All checks passed. Mypy: no issues
in 19 source files. Working/staged whitespace checks pass. An AST comparison
confirms production code is unchanged except docstrings; protected paths,
`.spec-seen`, docs and exports are unchanged. No full numerical gate rerun for
this docstring/test follow-up; the reviewer's passing through-A7 gate is
historical evidence only. No through-A10 completion claim. Docs to-do: retain
the separate public concordance documentation step. Stop after this commit.

---

# A10.2a — Share observation input normalization (2026-10-25)

Status: **implemented and verified; ready for review**, authorized by
"ok no issues here on to the next bit, which is approved by me". A10.1 is accepted. Split the next integration
step: first relocate its existing data-input resolver, then add public metrics
once D-027 is settled. This preparatory part introduces no public contract.

At planning start, `A10-cindex` was at `67bac4d` with the maintainer's merge
in progress and 21 resolved paths staged. This step preserved that index and
kept its own work unstaged until the maintainer completed the merge. Incoming
protected spec tests and the sync script were not edited.

The maintainer completed the merge as `34a7657` during this step. Verified
no merge remains and its index is empty; all new work stayed unstaged. The
shared resolver is independent of D-027, which the maintainer wants to discuss
before public integration; no proposed public choice has been adopted.


Implementation evidence: the resolver AST is identical to the original function
in the merged HEAD. The helper is 21 lines; new shared-input tests are 78 lines.
Together with its import/relocation, 116 source/test lines changed. Focused
shared-input/public-likelihood tests: 68 passed per dtype (4.35s / 3.79s),
without skips. The full through-A7 regression gate exited 0:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 811 passed, 7 skipped, 153.02s | 817 passed, 1 skipped, 169.74s |
| Source doctests | 4 passed, 1.13s | 4 passed, 1.05s |
| Independent through A7 | 2123 passed, 28 skipped, 223 deselected, 115.04s | 2127 passed, 24 skipped, 223 deselected, 107.53s |

Ruff: 75 files already formatted / All checks passed. Mypy: no issues in
19 source files. Full output: `/private/tmp/survjax-a102a-final-gate.log`;
last line `check: all gates passed through A7`. No new skips, test exclusions,
public API or numerical changes. The larger deselection count reflects incoming
later-step independent tests. REVIEW.md was finalized before the gate and is
unchanged afterward. Final static/diff checks follow this plan-only result edit.
D-027 records discussion only. The maintainer's merged docs/protected files
remain unchanged. No R, docs build, installation or performance check was run.

Files and public API:

- Create `src/survjax/_observation_inputs.py` and move `_resolve_data` there
  verbatim from `likelihood.py`; import it back at its existing call site.
  Preserve all checks, error messages and pure SurvivalData normalization.
  No public API, dtype, numerical, dependency or docs change.
- Create `tests/unit/test_observation_inputs.py` for shared-normalizer behavior
  needed by both callers: raw invalid-indicator evidence through eager/jit/vmap,
  missing/mixed input rejection, and prepared rich observations preserving
  every pytree leaf (including issue bits). Reuse the existing public likelihood
  suite for the unchanged values/first/second derivatives and input contract.
- Update this plan and REVIEW.md with this step's actual checks and limitations.
  Preserve the staged merge versions of those files until merge completion.
  Target under 150 changed source/test lines; one concept and one commit.
- Record the newly synced count-reader assumption and the maintainer's request
  for discussion in D-027, retaining its open status; adopt no public contract.

Spec: 6.1–6.2 (single shared checked normalizer), 3.2–3.3 (prepared leaves and
raw issues), 7.1/12.1 (likelihood regression evidence). No score contract or
ConcordanceInfo field/representation choice is made here. D-027 remains open;
new independent Q-042–Q-045 explicitly label their public choices assumptions.

After editing, run focused new/likelihood entry-point tests in separate x64/x32
processes, Ruff format/lint and mypy. Run the unmodified through-A7 gate for
regression evidence; A8 fixtures/A9 completion and the public A10 API remain
unimplemented, so do not claim a through-A10 pass. Record failures honestly.
Check whitespace, source diff and preservation of the incoming staged index.
Commit only this step after the maintainer finishes the merge, then stop.
If the merge is still pending, leave a checked, reviewable diff and report
that the step cannot yet be committed under the no-merge rule.

Docs to-do: the separately approved concordance docs topic/API reference after
public integration. No documentation is needed for moving a private resolver.

---

# Merge-conflict resolution (2026-10-24)

The maintainer requests resolving the in-progress merge on `A10-cindex`.
Resolve only `PLAN.md` and `REVIEW.md`, preserving the A10.1 records and the
incoming A7 docs and spec-test sync records. Keep prior reviews as historical
evidence; their statements describe their original checkouts. No public API,
source, spec, independent test or requirement changes; no new tests needed.
Check conflict markers, whitespace, preservation of both sides and the Git
unmerged-file list. Stage the two resolutions without creating a merge commit.

---

# A10 — Harrell's concordance (2026-10-23)

Status: **A10.1 implemented and verified; ready for review**, authorized by
"Approved go for it". The maintainer requested checking the updated roadmap
and starting A10. The existing branch is `A10-cindex`, starting at `9a034f4`.
The request authorizes bringing concordance forward now, ahead of the draft's proposed after-A9 sequence; it does not
authorize spec edits or publication. Statistical requirements remain accepted
spec revision 34, sections 3.3, 6.1–6.2, 9.1–9.2, 10.2 and 12.1.

Implementation evidence: a 61-line private counting module and 169-line test
module (230 source/test lines). Exact comparisons, event-first equal-time
pairs and two-word totals; no public API change. Focused checks pass, 24
per dtype (1.37s x64 / 1.34s x32), without skips. Final through-A7 regression
gate exited 0 and ended `check: all gates passed through A7`:

| Gate stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests | 801 passed, 7 skipped, 141.70s | 807 passed, 1 skipped, 143.20s |
| Source doctests | 4 passed, 0.82s | 4 passed, 0.81s |
| Independent through A7 | 2123 passed, 28 skipped, 44 deselected, 94.06s | 2127 passed, 24 skipped, 44 deselected, 90.73s |

Ruff: 73 files already formatted / All checks passed. Mypy: no issues in
18 source files. Full output: `/private/tmp/survjax-a101-final-gate.log`.
REVIEW.md records the focused commands, initial cache/lint failures and limits;
its metadata was finalized during the gate, with final static checks afterward.
No new skips or exclusions. No A10 independent suite, R fixture, docs build,
installation or performance evidence claimed. D-027 remains open. Stop after
this step's commit; A10.2 has not started.

Read `../design/ROADMAP.md` and its linked
`../design/proposals/concordance-at-end-of-slice-a.md`. Both describe a draft
delivery amendment: accepted section 12.2 still assigns concordance to E.
Do not modify the design repo or treat its proposed completion gate as adopted.
Do not claim A8, A9, the A7 docs or Slice A completion through this work.

Starting working-tree changes belong to the unfinished A7 docs step:
`PLAN.md`, `docs/index.md`, `docs/likelihood.md` and untracked
`docs/numerical-behaviour.md`. Preserve them. Prepend this plan without
rewriting its existing content; stage only A10 changes when committing.
No docs edits in A10 code steps. No dependencies, fitting helpers, KM, IPCW,
Brier, IBS, AUC, approximate metric API or score-gradient guarantee.

## A10.1 — Private comparable-pair counting (first approval and review stop)

One concept: exact counts for one valid, supported cohort, without public
entry points. Target approximately 230–280 changed source/test lines, plus
the plan and review. Split and replan if implementation exceeds 300 lines.

Files to create:

- `src/survjax/_concordance.py`: a private typed counting function for
  one-dimensional finite scores, finite times and exact/right status.
  Return concordant, discordant and tied-score totals as private `(hi, lo)`
  tuples of `uint32` scalar arrays. Each total denotes `hi * 2**32 + lo`.
  Keep this representation private pending D-027. Comparable totals are
  accumulated during the same loop. No public API or export changes.
- `tests/unit/test_concordance_counts.py`: hand-enumerated pair counts and
  eager/jit/vmap checks; private accumulator carry checks using synthetic
  word values. Test expectations come from Python integer arithmetic and
  explicit pair lists, never from the implementation being tested.

Update this plan's status and overwrite `REVIEW.md` only at the implementation
review stop, recording reading order, invariants, requirement-to-test mapping,
actual check output and remaining limitations.

Implementation invariants (spec 9.1–9.2):

- A JAX loop compares one row with the cohort and counts only partners with
  larger indices. Work is O(N squared), pair workspace O(N); never form an
  N-by-N array. Later leading-axis batching calls this core with `vmap`.
- For unequal times the earlier row must be an event. At equal times only
  event/censor pairs are comparable, with the event earlier. Compare scores
  exactly; higher score on the earlier event wins. No near-tie tolerance.
- Sum each loop's boolean counts as `int32`, then add the nonnegative count
  into two unsigned words with explicit carry. No int64 requirement and no
  total-count cap imposed to avoid int32 overflow. The cohort length must fit
  the per-row int32 counting contract; no quadratic large-cohort test needed.
- This core accepts already validated cohorts. Invalid scores, unsupported
  modes, stored data issues and floating ratios belong to the integration
  step; no public function returns partial or placeholder results meanwhile.

Own tests accompanying this step:

1. A small cohort with explicitly listed concordant, discordant and score-tied
   pairs; all counts sum to the hand-counted comparable total.
2. Separate equal-time event/event, event/censor and censor/censor cases,
   plus censoring earlier than an event. Include reversed array order to
   ensure array position never substitutes for event-time order.
3. Perfect, reversed and all-tied scores; singleton, empty and all-censored
   inputs produce zero counts in the private core. Public acceptance of empty
   inputs remains a D-027 question; do not infer it from core behavior.
4. Row permutations preserve all totals; eager, jit and cohort-wise vmap
   agree. Counts remain exact in separate x64/x32 processes.
5. Carry across `2**32`, repeated carries and high-word totals, checked
   against arbitrary-precision Python integers. Exercise totals beyond signed
   int32 capacity directly without allocating an enormous cohort.

Verification after approval:

- Run `tests/unit/test_concordance_counts.py` in separate
  `JAX_ENABLE_X64=1` and `=0` processes via `uv run --group test pytest`.
- Run `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`
  for the established implementation baseline. The existing A8 acceptance
  tests require unfinished fixture work; do not label this an A10 full gate.
  No independent A10 suite is currently synced; request maintainer coordination
  before declaring A10 complete. Never edit `tests/spec/` or the gate script.
- Run `git diff --check`; record pre-existing failures separately without
  repairing unrelated docs. Check staged diff excludes the A7 docs changes.
- Commit only this approved step on `A10-cindex`, with imperative subject and
  a body explaining why, then stop for review. No automatic A10.2 start.

Docs to-do: after public integration and review, a separate concordance topic
with score direction, pair/tie conventions, support, batching, masks,
undefined results, complexity and held-out synthetic evaluation. Update the
API reference and playground in the separately approved docs integration.

## Later A10 steps — outline only; each needs its own approved plan

- A10.2b: after the separately owned spec alignment and test sync land, refine
  the public integration plan around the maintainer's accepted D-027 decision.
  Introduce `metrics.py`, frozen registered count and `ConcordanceInfo` types,
  and both public entry points. `_resolve_data` is now shared by A10.2a; use
  that same helper. Validate last-axis lengths before broadcasting leading
  axes; reduce invalid/unsupported evidence per cohort. Before calling
  `_pair_counts`, substitute finite safe scores/times and a boolean exact mask
  for each entire invalid/unsupported cohort, preserving observation count and
  failure masks. Never drop rows or count a partially valid cohort. Force all
  returned count words to zero for those failed cohorts, and set
  `no_comparable_pairs` only for evaluated cohorts. Honor stored raw-input
  issues; compute the ratio in the data float dtype,
  casting words before floating arithmetic and using a safe denominator for
  undefined cohorts. Add input-form, masks, dtype and shape tests with code;
  split integration if its source/test budget exceeds 300 lines. Resolve the
  current export-reference test's docs requirement before adding top-level
  exports: code steps cannot silently edit docs or weaken that test.
- A10.3: finish public transformation/broadcast and permutation evidence in
  both dtypes; independent acceptance tests arrive only from the maintainer.
  No score derivatives are promised. Maintain one concept per small commit.
- A10.4: synthetic R generator, fixture README and reference tests, following
  section 10.4's pinned R/package requirements. Use `reverse=TRUE`,
  `timewt="n"`, `timefix=FALSE`. If the pinned environment is unavailable,
  hand off the generator and stop; never invent reference output.
- A10 docs and final audit: executable topic/playground, held-out synthetic
  example, installed wheel/source-archive exports and evidence map. Coordinate
  with unfinished A8/A9 work; a concordance step alone cannot close Slice A.

Planning checks actually run: branch/status/history inspection; accepted
sections above, roadmap and draft amendment read; existing input resolver,
dtype policy, data/diagnostic containers, export-reference test and gate read.
`../scripts/spec-since.sh dev` prints
`dev: spec unchanged since d243a96`. No implementation, tests, build, R
environment check, spec edit, spec-seen update or commit during planning.

---


---

# Preserved incoming main plan (verbatim pre-merge version)

# README.1 — Draft the alpha README (2026-10-30)

Rewrite README.md at the maintainer's request on add-readme. Preserve the
unfinished existing plan history below. One documentation topic; no API,
dependencies, protected paths, or spec changes. Spec 4.1–4.2, 6.1, 7.3,
9.2, 10.3, 12.2. Exponential/Weibull and concordance are the first built-ins;
more families and metrics are planned, not advertised as available.

Files: README.md, PLAN.md, REVIEW.md. Present current supported likelihood
scope, extension requirements, a core-only runnable gradient example, local
installation, principles and documentation. No new tests: execute the README
snippet against the source in both dtype modes with an analytical check, run
scripts/build_docs.sh directly, and check whitespace. Overwrite REVIEW.md,
commit, then run independent-review for README.1 and stop after its verdict.
Target under 300 changed content lines, excluding existing review records.

---

# SIM.3 — Complete the approved examples diagnostics environment (2026-10-30)

Start clean on release-simulation at
`ce68f76ffc4ee0fdcda1757232f88ee57ff13c3d`; SIM.2 round 1 returned merge.
Preserve the unfinished SIM/A10 plan history below. The maintainer approved
ArviZ 0.23.0 in the existing dependency proposal and now requests the next step.
One concept: finish the examples dependency group for Bayesian tail ESS.

Files and scope:
- `pyproject.toml` / `uv.lock`: add exact `arviz==0.23.0` to examples;
  resolve without upgrading existing packages or changing runtime dependencies.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark the six-package setup
  complete, record the actual installed diagnostics probe and its command.
- Overwrite `REVIEW.md`: reading order, dependency isolation, actual evidence
  and limits; preserve the short temporary probe source for reproducibility.

Spec 10.3; shared use-case 00 dependency-group rule and Bayesian details
Sampling/Diagnostics: NumPyro split R-hat/ESS, ArviZ tail ESS. No source API,
shared generator, ordinary tests, docs/navigation/config or protected scripts.
Docs to-do: the future Bayesian page should use this group for tail ESS.
Target below 300 changed implementation/config/lock lines, records excluded.

Checks: `uv lock --check`, `uv sync --locked --group examples`, compare old/new
lock name/version mappings. Explicit optional-environment probe uses seeded
4x1000 iid Normal draws (known equal-chain law) and deliberately shifted
chains to verify the chain/draw axes, NumPyro summary and ArviZ bulk/tail ESS
and R-hat; this is compatibility evidence, not posterior convergence evidence.
The pinned ArviZ import writes a daily stamp under Path.home()/arviz_data, so
run that probe with required sandbox escalation rather than modifying HOME.
No new ordinary tests for the dependency-only change. Run the existing focused
generator suite in both dtype processes, Ruff, mypy, and unchanged fresh-wheel
through-A10 gate. Record versions/results, write REVIEW.md, commit, invoke the
exact independent-review skill for SIM.3, follow at most 3 rounds and stop.
PH fitting still waits for maintainer integration of reviewed prerequisites.


Implementation: added the approved ArviZ pin and six lock packages. All 165
prior package versions unchanged; runtime dependencies remain jax/numpy.
Actual installed-group CPU probe passes named chain/draw axes, iid gates and
shifted-chain detection: NumPyro R-hat 1.0000270083 / ESS 3976.58; ArviZ
R-hat 1.0001031616 / bulk ESS 3979.33 / tail ESS 3931.29. ArviZ's expected
import warning was emitted; sandbox escalation allowed its home-cache stamp.
Existing generator suites: 17 passed per mode (3.60s/3.18s), Ruff/mypy/lock
checks pass. Fresh-wheel through-A10 gate exited 0: own suites 1065/1071
passed (7/1 skipped), doctests 4 per mode, independent 2333/2324 passed
(27/36 skipped). ArviZ's pandas dependency enabled the existing x32 pandas
conversion test. Wheel hash is identical to SIM.2 and runtime METADATA still
requires only jax/numpy. REVIEW.md records exact commands, outputs and limits.
Commit and exact independent review follow; stop after the verdict.

---

# SIM.2 — Address shared foundation questions and enable examples (2026-10-29)

Start on `release-simulation` at `3556d12c1abca553f77607c1e1739df2bd1911f0`.
The only initial dirty file is study.py with three maintainer question comments.
SIM.1 round 2 returned merge. Preserve all unfinished history below.
Maintainer approves the proposed dependencies in this turn; no renewed
approval is needed for those exact pins. Keep PH fitting deferred.

Files and scope:
- `pyproject.toml` / `uv.lock`: examples group with exact tested Optimistix
  0.1.0, Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0.
  Keep existing runtime/dev/docs versions unchanged. The five-package lock
  preview is 160 changed lines; full six-package setup is 295 lock lines
  alone. Approved ArviZ 0.23.0/tail ESS is a separate later step so this step
  including generator checks remains about 300 lines. No protected scripts.
- `examples/weibull_simulation/study.py`: answer/consume the maintainer notes,
  define quadrature rules before their closure, and explain calibration as
  once-per-target host setup outside the compiled replicate loop. PH input
  preference is asked asynchronously (baseline/log-HR versus PH rate m).
  Preserve the current call contract; any PH convenience will be optional,
  demo-only, and checked against the existing scale formulation.
- `tests/unit/test_simulation_generator.py`: if adding PH input, analytical
  PH/scale and calibration identities, row broadcasts and transformed values/
  derivatives. Reuse existing tolerances. Existing numerical checks also
  exercise the quadrature block reorder.
- `examples/weibull_simulation/DEPENDENCIES.md`: mark approval, staged pins,
  remaining approved ArviZ stage and exact optional-environment invocation.
- Overwrite `REVIEW.md` with reading order, input-contract updates, evidence
  and limits, preserving SIM.1's outcome in this plan and Git history.

Spec 4.4/4.6/7.3/10.3; use-cases/00-common and 01-details DGP/calibration.
No library API, shared docs/config or other workstream files. Docs to-do:
future simulation page should show PH inputs and setup versus compiled work.
Run generator checks in separate x64/float32 processes, approved-environment
compatibility probe, Ruff, mypy, lock consistency and fresh-wheel through-A10
gate. Inspect size before committing; split if needed. Commit, invoke exact
independent-review skill for SIM.2 (max 3 rounds), then stop/report.


Implementation: optional log_hazard_ratio=0.0 added to both generator and
calibration, with unchanged no-keyword calls and U/C key split. Rules now
precede the fraction closure; calibration explicitly remains host-only setup.
No response to the optional PH-form preference arrived during environment
work; used the stated baseline/log-HR convention from 01-details and spec 4.4.
Approved five-package group resolves/installs; all 155 existing locked
versions unchanged, 10 additions. Optional compiled BFGS/NN-update/NUTS/SVG
probe passes in the actual group environment. ArviZ remains approved for its
separate lockfile step. Total implementation/test/config/proposal diff is
303 changed lines, including 160 lockfile lines.

Focused final suites: 17 passed per dtype (10.17s/10.56s), Ruff/mypy/lock
checks pass. A joint three-parameter check_grads initially failed a float32
second-order projection near zero; no tolerance was changed. Retained the
original k/b finite-difference checks and verified all PH gradient/Hessian
entries against inverse-CDF algebra, including cross-partials. The full
fresh-wheel through-A10 gate exited 0: own suites 1064/1071 passed (7/1
skips), doctests 4 per mode, independent 2333/2323 passed (27/37 skips).
NumPyro enables one acceptance check previously skipped. REVIEW.md records
exact commands, times, initial numerical failure and final focused evidence.
Commit and exact independent review follow, then stop/report.

---

# SIM.1 — Shared Weibull generator and censoring calibration (2026-10-29)

Work on `release-simulation` from full base
`8757c0fda78e34e6e818249adedd2a144a33c33d` (clean launch snapshot).
The matching reviewer checkout already exists. Preserve the unfinished A10
history below; its docs and A9/PH/RMST work belong to other implementers.

One concept: reusable demo-only inverse-CDF generation and host-side numerical
calibration, without PH, fitting, core changes or optional dependencies.
Spec revision 37: 3.3, 4.4 (generating PH-equivalent scales only), 7.3,
10.3, 11.4, 12.1–12.3; use-cases/00-common.md and both 01 files.

Files and contract to implement:
- `examples/weibull_simulation/study.py`: `generate_weibull(key, shape, scale,
  censor_limit, *, sample_shape=())`, returning a named tuple with `time`,
  `event`, `event_time`, `censor_time`. Parameters broadcast to row/batch axes;
  sample_shape prefixes those axes. Exactly split the supplied key into U/C
  keys; callers derive scenario/replicate keys with the specified fold/split
  scheme. JAX generation supports jit/vmap and selected process precision.
- Same module: `calibrate_censoring(shape, scale, target) -> float` uses NumPy
  float64 quadrature and bracketed bisection of the mean integrated survival.
  Concrete positive finite broadcast parameters only; calibration is host-only,
  not an RMST implementation or differentiable library API. Check quadrature
  convergence; fail explicitly rather than return an unverified calibration.
- `tests/unit/test_simulation_generator.py`: analytical exponential calibration,
  independent mpmath Weibull quadrature, heterogeneous broadcasting, key mapping,
  transformation agreement and fixed-seed large-sample survival/HR/censoring
  recovery with justified statistical bounds. Centralize any new test bounds
  in `tests/impl_tolerances.py`. Ordinary checks remain offline.
- `examples/weibull_simulation/DEPENDENCIES.md` and a temporary isolated environment
  probe (record the exact invocation in REVIEW.md): concrete exact-version proposal for the later examples group, test
  Optimistix/Matplotlib/Equinox/Optax/NumPyro against locked Python/JAX/NumPy.
  No pyproject/lock edits or added diagnostics/loader packages in this step.
- `REVIEW.md`: reading order, invariants, evidence and limitations; freeze the
  reviewed consumer import contract and calibrated simulation c values.

Target at most about 300 implementation/test/proposal lines (plan/review
records separate); split fitting, scenario runner, results and docs into later
steps. Docs to-do: executable simulation page and recorded full results after
reviewed PH integration; propose fresh docs command with examples selected.

Checks: focused tests in separate JAX_ENABLE_X64=1/0 processes, Ruff formatting
and lint, mypy src, unchanged `scripts/check.sh --through A10` cumulative gate.
Record exact versions, commands and outputs. Commit the step, invoke the exact
independent-review skill command for SIM.1, follow up to three rounds, then stop.
Dependency installation into the project still awaits maintainer approval;
only an isolated proposal probe is authorized here. Never merge/push/publish.


Implementation: 300 added source/test/proposal lines, excluding plan/review
records; no library API, dependencies, lockfile or protected-file edits.
Reviewed consumer contract is recorded in REVIEW.md pending round 1.
Both simulation targets independently match 40-digit mpmath quadrature;
focused final suites: 12 passed in each dtype, including second derivatives.
Ruff formatting/lint and mypy pass. Calibrated c values: 47.793149053990135
and 19.17253543619806 years; pilot empirical censoring .249355/.550485.
Dependency versions proposed and tested in isolation: Optimistix 0.1.0,
Matplotlib 3.11.2, Equinox 0.13.8, Optax 0.2.8, NumPyro 0.22.0 against the
existing locked JAX/jaxlib 0.11.2 and NumPy 2.5.3; group/lock approval pending.
Cumulative fresh-wheel through-A10 gate: exit 0, all gates passed.
Own suites 1059/1066 passed (7/1 existing skips); independent 2332/2322
passed (28/38 existing skips); source doctests 4 per mode. REVIEW.md records
exact timings, commands and the focused derivative-test evidence. Commit and
independent round 1 now follow; stop after the review verdict.

Round 1 on 1e3f5fc returned fix-first for F-1: add explicit quadrature-failure
coverage. Add the k=20/b=5/target=.5 RuntimeError regression, and clarify N-1
underflow and N-2 time units in docstrings; N-3 import-helper suggestion stays
conditional. Fix-up is 7 net source/test lines (307 total), no behavior or
contract change. Focused suites: 13 passed per mode (4.32s/4.05s), Ruff and
mypy clean. Commit separately and request round 2 after round 1 completes.

---

# PLAN.md: current sub-slice only

This file holds the plan of the sub-slice in progress, newest section first.
Earlier A9 work is in `plans/A9.md`; all incoming A10 records are preserved verbatim in
`plans/A10.md` while this branch returns to A9 at the maintainer's request.
A10 is unfinished; archiving its records does not claim completion.

---

# A9.d1 — Using and verifying the installed Slice A subset (2026-10-29)

Authorized by "the current work is approved; if more work needed, get to it".
Work on the existing a9b branch from clean bd32ff9. The maintainer completed
the main merge, manually reran `scripts/check_wheel.sh --through A9` without
errors, and cleaned the reviewer-owned throwaway worktree. Record those as
maintainer-reported checks/cleanup, not as runs performed by this agent.
A9.4 is accepted; optional review nits N-1/N-2 do not block this docs step.

One topic: explain the installed subset's verification and limitations.
Spec: 6.1–6.2, 7.2–7.3, 10.3, 10.5, 12.1–12.3. Public API unchanged.
At most 300 changed docs lines. No dependency, source, test, fixture,
benchmark, protected-path or spec edits. PLAN.md keeps only current A9.

Files to create/change:

- Create docs/slice-a-verification.md as executable MyST. Give a compact
  synthetic exact/right example with construction and diagnostic access inside
  jit, value/parameter gradient/Hessian checked against hand algebra, eager
  and prepared/raw agreement, and vmap over rates. Document the smooth
  interior parameter-only guarantee and explicit host checkpoints.
- Explain the current subset and unsupported likelihood modes; link existing
  construction, likelihood, numerical, distribution/grid and concordance
  guides. Describe the fresh-wheel runner, installed out-of-tree custom-family
  checks, pinned synthetic R evidence and the exact CPU reproduction commands.
  Explain first-call versus warmed costs, prepared/traced flags, returned
  diagnostics, logical bytes versus unmeasured peak memory, and optional
  benchmark nits without silently changing accepted measurements.
- Add the new page to docs/index.md's prose and toctree. Audit the remaining
  executable pages and playground. Correct only stale capability statements
  within this verification topic; leave unrelated teaching work for separate
  steps. Document the locked JupyterLab command.
- The rendered audit found docs/survival-data.md's opening note still claims
  likelihoods, distributions and evaluation are unimplemented. Correct that
  one scope note and link current likelihood/concordance guides. This narrow
  factual correction belongs to the installed-subset verification topic;
  it adds no new tutorial or runtime behavior. Total docs remain below 300
  changed lines. Record the discovery before making the edit.
- Overwrite REVIEW.md with reading order, invariants, evidence mapping and
  remaining gate status. Record every Slice A gate item with concrete tests
  or artifacts and honest pass/pending status; this is not an A10/release
  completion claim. The agreed README/use-case example replacement is still
  separate from this small compiled-verification recipe.

Validation ownership: implementer owns fresh `scripts/build_docs.sh` in both
modes, Ruff and rendered-page/browser inspection. The maintainer's single-
runner preference means the independent reviewer should inspect source and
recorded outputs instead of repeating the same build/gate. No full wheel-gate
repeat for docs-only edits: the maintainer just reran it successfully on the
merged baseline, and no runtime/packaging change follows. Inspect links,
outputs, formulas, navigation and tables across the site in both builds;
inspect the new page carefully and record any existing warning/limitation.
Verify committed notebooks still have no outputs/counts. No new unit tests:
the executable page's assertions check its example, and existing suites
already cover the numerical and packaging behavior.

After successful build/source/visual checks, record exact commands/results,
commit this docs step on a9b with "Independent review: round 1/3" in REVIEW.md,
invoke the exact independent-review skill command for step A9.d1 in
../review/code-a9b, address blocker/should-fix findings in separate fix-ups
within three rounds, then stop and report. Do not start another step.

Docs/closure to-do: agreed replacement README/use-case teaching material and
any gaps from this rendered-site/evidence audit remain explicit; don't claim
Slice A or A10 complete until their separate gate items are accepted.

## A9.d1 evidence and audit conclusion

Implemented the verification page, index navigation and one stale data-guide
scope note; no code or notebook changes. Both final docs builds pass, nine
executed documents and 137 code cells per mode. The three new example cells
check exact analytical values, gradients/Hessians, input forms and jit/vmap.
All 1,048 local file/fragment links per rendered site pass. The committed
playground remains output-free. Ruff passes for the repository and the
extracted three-cell notebook; JupyterLab CLI reports 4.6.4. Native Brave
inspection covers all 11 content pages in float64, all pages' accessible
content in float32 and detailed views of the new page, corrected note,
playground and representative grids/formulas/tables. REVIEW.md records final
execution times, initial sandbox/interruption errors, checks and limits.

The final evidence map is included in this step's REVIEW.md rather than
requiring a separate metadata-only audit step. The new three-observation
verification recipe provides tiny synthetic smoke evidence; the larger
agreed README/use-case teaching replacement remains separate work and is
not claimed complete. The maintainer's post-merge fresh-wheel pass supplies
current regression evidence; no runtime/packaging changes follow it.
A9's verification evidence is now ready for independent review and maintainer
acceptance. No claim that A10, Slice A as a whole or the release is complete.
Stop after this docs step's review; no additional implementation starts.

## A9.d1 round-1 follow-up

Round 1 on 2c8cd60 returned fix-first for should-fix F-1: cite the explicit
unsupported-mode NaN/preserved-neighbor requirement and executed triage
recipe in the gate map, and map each narrative item in spec 12.2's paragraph
to its docs page. Existing code/tests already cover this behavior; no new
implementation or test is required. Also fix trivial N-1 (R paragraph line
wrap) and N-2 (clarify that tests/doctests, rather than static checks, run
in separate dtype processes). Code cells remain byte-identical.

The reviewer reran the build because its role requires it, despite the
initial ownership request. Its build passed in both modes (new page 3.12s /
3.07s); Ruff passed. No wheel gate or mutations were rerun. For this prose/
metadata fix-up, the reviewer alone owns round-2 builds/lint. The implementer
checks only the diff and unchanged cells, records F-1/N-1/N-2, commits a
separate fix-up and requests round 2. No duplicate build/full-gate run by
implementer; maximum three rounds, then stop.

---

# A9.4/main merge resolution (2026-10-29)

Authorized by "merged main into the branch; fix conflicts". The maintainer's
merge is in progress on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflict; resolve these metadata files. No public
API, code, docs, dependency, spec or test behavior changes are needed.

- Preserve A9.4's complete plan and evidence below. Independent review round
  1 on 51319c8 returned merge, with optional N-1/N-2; those nits remain open.
- Preserve all main's A10 plan sections verbatim in plans/A10.md, including
  A10.d2, A10.d1 and A10.4. Its older A10 history remains the unchanged suffix
  of that archive. PLAN.md keeps only the current A9 sub-slice. A10's incoming
  review record is preserved in REVIEW.md alongside the A9.4 record. These
  are historical records; this merge starts no new implementation step.
  This corrects the first resolution, which incorrectly kept incoming A10
  sections in PLAN.md; the maintainer explicitly checked the current-slice
  format before completing the merge.
- Preserve every original stage-0 index entry except the intentional
  plans/A10.md archive extension and plans/README.md index description.
  Incoming docs, concordance fixtures/tests, release-agent plans and
  .workmux.yaml remain untouched. The last is an incoming protected-path
  change, not an agent edit.
- Verify both parents' records are retained, no conflict markers/unmerged
  entries remain, whitespace is clean and every other staged object is
  unchanged from the captured merge index, apart from the two archive/index
  changes above. No runtime/build reruns: neither
  conflicted file affects execution, and no manual executable edit is needed.
  Preserve incoming A10.d2's single-runner validation ownership.
- Stage the two resolutions and archive/index corrections; leave the merge commit to the
  maintainer under the working agreement. Do not create a merge commit,
  start the next implementation step or rerun independent review for this
  metadata-only resolution.

---

# A9.4 — Reproducible CPU baseline (2026-10-29)

Authorized by "check the roadmap ... plan the needed work and go on to
implement it" on the maintainer-created `a9b` branch, initially clean at
`1c0fd06`. The separate archive commit preserves A10 records verbatim;
A10 is unfinished and outside this step. No approval exception is needed:
no new public API, dependency, protected path or spec interpretation.

## Roadmap audit and remaining sequence

The workspace `../ROADMAP.md` still describes A10.2a and missing A8/A9
integration. Current merged plans/evidence show A9.1 derivatives and A9.2
wheel/extension checks accepted, A8 fixtures integrated, and through-A9
combined-tree gates already passed during A10 integration/A10.3. A9.3's
standalone example was explicitly withdrawn; do not restore it. The roadmap
is maintained from the workspace root; report its needed status correction
in this branch's handoff instead of editing outside this repository.

Remaining A9 steps, each separately committed/reviewed with a stop:

1. **A9.4, now:** reproducible CPU likelihood/gradient measurements below.
2. **A9.d1, later docs step:** installed-subset verification page and
   navigation, benchmark reproduction, audit executable pages/playground,
   fresh two-dtype docs builds and rendered-page review (spec 10.5).
3. **A9 closure, later:** map every section 12.2 Slice A gate to concrete
   evidence and run the final combined-tree fresh-wheel gate. Keep the tiny
   synthetic example pending until the agreed replacement README/use-case
   material provides evidence. Use-case 1 needs the separately authorized
   PH work; it is not part of A9.4. Do not claim Slice A complete.

## Implementation plan

One concept: measure existing behavior without optimizing library code.
Spec: 7.2–7.3, 12.1–12.3. Public API touched: none.

Files: create `scripts/benchmark_slice_a.py` (at most 300 changed code lines)
and `benchmarks/slice_a_cpu.json` (machine-readable evidence); update this
plan and overwrite REVIEW.md. No source, docs, spec, protected files,
fixtures, lockfile or dependency changes.

- Fixed seed, CPU selected explicitly, separate x64/x32 processes, both
  Exponential/Weibull, N=1000/10000/100000, shared and row parameters.
- Measure a jitted summed log likelihood and parameter gradient together,
  returning data and parameter diagnostics as auxiliary outputs. Compare
  prepared SurvivalData with ordinary construction from traced time/event
  arrays. Record prepared versus conservative traced static metadata.
- Time host generation, initial placement and prepared construction
  separately. Synchronize all output leaves. Record compilation/first call,
  warmups and repeated warmed samples (minimum/median/maximum), never
  introduce timing thresholds in normal tests.
- Before recording a row, compare its objective/gradient to independently
  differentiated exact/right formulas evaluated with NumPy. Import existing
  ATOL/RTOL from tests/impl_tolerances.py; do not add/loosen tolerances.
  Assert diagnostics valid. Check changed dynamic values and equal-shaped
  minibatches from a parent reuse the compiled trace, with numerical checks;
  retain A9.1 tests as additional regression evidence.
- Record exact commands, seed, parameter/input shapes, dtype, locked versions,
  hardware/OS and actual CPU device, repetition count and logical array
  input/output bytes. Explicitly state peak memory unmeasured; do not infer
  peak memory from bytes. Summarize scaling/overhead and limitations in review.

Verification: execute the full matrix in separate dtype processes; inspect
all 48 rows/correctness/reuse checks and JSON metadata. Run the existing
likelihood composition tests (both modes), Ruff format/check, mypy src and
`scripts/check_wheel.sh --through A9` using a fresh wheel, then whitespace
and scope checks. No new regular tests: the benchmark's analytical checks
validate its full numerical workload and existing composition tests cover
recompilation behavior; no timing assertions. Record exact outputs/failures.
Commit on a9b with REVIEW.md saying "Independent review: round 1/3", invoke
the independent-review skill in ../review/code-a9b for step A9.4 and the
literal full commit hash. Address blocker/should-fix findings in separate
fix-ups within three rounds, then stop and report.

Docs to-do: A9.d1 described above. Example replacement and final gate map
remain pending; no A10 or later-slice work starts in this step.


## Implementation evidence

The 230-line harness and committed JSON cover all 48 cases on Apple M4 Max,
CPU only, Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3. Each case passes
analytical value/full-gradient checks for its initial input and two changed
parent minibatches, and one-trace reuse. Ten samples follow three warmups.
Independent report audits verify case completeness, medians, shapes, flags,
finite samples and summary fields. Peak memory remains explicitly unmeasured.
Focused existing derivative/composition suites: 15 passed in 28.03s in each
dtype, no skips. Fresh-wheel `scripts/check_wheel.sh --through A9` exited 0, ending
`check: all gates passed through A9`: own suites 1044 passed/7 skipped
(float64), 1050 passed/1 skipped (float32); source doctests 4 passed per mode;
independent suites 2211 passed/28 skipped and 2202 passed/37 skipped.
Ruff reports 104 files already formatted / All checks passed; mypy has no
issues in 20 source files. Exact commands, timings, artifact identity, initial
harness corrections and limits are in REVIEW.md. Final whitespace/scope
checks, commit and independent review follow. No library,
docs, dependency, fixture or protected-path change. Remaining A9 steps above
are still pending; this evidence does not close Slice A.
