Independent review: round 1/3

# BAYES.integration — Maintainer-started main merge resolution

Read PLAN.md integration section, DEVIATIONS.md reconciliation note, and the
preserved Bayesian/RMST reviews below. Scope is conflict resolution of three
record files and verification of the incoming staged implementation. No
implementation changes beyond the maintainer's merge. The maintainer explicitly
authorizes completing the in-progress merge (exception to the default no-merge
rule for agents). Pre-merge heads: 7a2ca65 and 95f4532; original base b1291a8.

Preserve both complete plan histories verbatim, including overlapping historical
sections, consistent with the incoming integration records. Preserve all prior
review content verbatim under labelled sections. Preserve deviation bodies
and statuses; Bayesian prior entry D-028 becomes D-033 because PH already uses
D-028, with a historical-reference map. Current spec is revision 41; Bayesian
priors still differ under the maintainer's committed choice, not silently fixed.

Verification so far (2026-11-01):
- Read current AGENTS.md, spec revision 41 (4.4/8.7/11.3/12.1–12.2), both
  Bayesian use-case files. .spec-seen equals current design HEAD
  00e7225b6b10b0d4e3137f7cf85c0b76e377d821; no marker/spec edits by this step.
- Preservation probe compared each full pre-merge PLAN/REVIEW stage with the
  resolved file; both present verbatim. Every deviation entry is present,
  allowing only Bayesian heading D-028 -> D-033. No status/body changes.
- No conflict markers or unmerged entries remain; staged whitespace clean.
  `git diff --cached MERGE_HEAD -- CLAUDE.md .spec-seen src docs tests` empty:
  incoming protected/core/docs/test content is unchanged.
- `JAX_ENABLE_X64=1 uv run --group examples python -m
  examples.bayesian_rmst.check_model`: exit 0. Trial/grid/aggregation checks
  pass; factor/plate/sites density=-502.786467405927. Analytical posterior
  route check passes with maximum absolute error=0.0.
- `uv run --group examples examples/bayesian_rmst/run.py --config smoke`:
  exit 0, 4 parallel CPU chains, 100 warmup+100 retained draws per chain;
  zero divergences per chain; route maximum absolute error
  2.4868995751603507e-14. Sampling including compile 1.2890000832 seconds;
  postprocessing 2.9797413331 seconds. tau_max=13.0 years. This verifies the
  runner but is not full posterior convergence/truth evidence. Priors remain
  Normal(0,5), JAX/jaxlib 0.11.2, NumPy 2.5.3, NumPyro 0.22.0, x64 true.
- `uv run ruff format --check . && uv run ruff check . && uv run mypy src`:
  142 files already formatted, All checks passed, mypy clean for 22 files.
- `scripts/check.sh --through A10`: regression suites pass, x64 1282 passed/
  8 skipped in 125.20s, x32 1287 passed/3 skipped in 172.98s; source doctests
  4 passed in 1.30s/1.19s. Initial x64 acceptance: 2328 passed/27 skipped/
  5 setup errors in 102.11s because SURVJAX_WHEEL was unset. Script exits 1;
  do not claim a clean one-shot gate. No independent test edited/skipped.
- Built a fresh wheel via `uv build --wheel --out-dir
  /private/tmp/survjax-bayes-integration-wheel-20261101`; wheel-backed
  acceptance reruns both exit 0. Exact commands used SURVJAX_WHEEL pointing
  at that wheel, JAX_ENABLE_X64=1/0, SPEC_THROUGH=A10, then
  `uv run --group test pytest tests/spec -q -n 6`: x64 2333 passed/27 skipped
  in 127.02s; x32 2324 passed/36 skipped in 117.33s. Together with the script's
  passing regression/doctest/static stages this supplies all gate evidence,
  without claiming the initial invocation exited 0.
 F-003's pairing
fixture weakness is carried forward, not hidden or fixed as merge resolution.
Incoming source now provides PH and its delegated staged RMST. Full convergence,
truth coverage, plots/predictive KM and docs remain later substeps.

---

# Preserved Bayesian review before merge (verbatim)

Independent review: round 1/3

# BAYES.2 — Batched posterior RMST and per-draw summaries

Read first:
1. PLAN.md BAYES.2; spec 11.3 and 02-details Route A, Sampling, Grids.
2. model.py PosteriorDraws, time_grids, posterior_draws, check_rmst_routes,
   summarize_draws (the existing model/priors are unchanged).
3. check_model.py check_aggregation_and_grids and check_posterior_routes.
4. run.py postprocessing and output metadata.

Scope: 234 changed example/check lines (plan/review records excluded), no new
module, public library API, dependencies, core, protected paths or docs.
Maintainer approves the earlier BAYES.1 work and explicitly requests this next
step. Same Bayesian sub-slice: its PLAN records remain in place. Previous step
head 12bc641; worktree original base b1291a8. Prior independent review is in
review/slices/BAYES.1.md, round 2. Its integration findings F-001/S-001 remain;
D-028 remains open for the maintainer, and the committed Normal(0,5) priors are
preserved. No claim that they match the delegated design.

Invariants:
- Flatten chains only through sampler.get_samples(); no hand flatten order.
- Route A: base (S,1), PH log-HR (S,2), RMST(5) (S,2), RMST grid (m,S,2),
  survival grid (K,S,2). Arms are always control then treatment.
- Route B uses conditioned parameters and deterministic sites outside the
  observation plate, enabled only after NUTS. Its (S,m,2) curve is swapped
  to (m,S,2). Require identical shapes and finite equality at absolute-only
  1e-12 (equal_nan=False). Return the maximum absolute discrepancy.
- Differences are paired per draw before mean/SD/quantiles/P(delta>0).
  SD uses ddof=1; intervals are equal-tail 2.5/97.5 percentiles. Curves stay
  draw-level for later plotting rather than evaluating at mean parameters.
- Host grid setup uses observed lower (valid for this exact/right-only demo):
  p90 rounded down to half-years; reject tau_max<5 or nonfinite. Grids are
  201 points on [0,20] and 101 points on [0,tau_max], including zero.
- JAX distribution evaluation is followed by host NumPy aggregation. These
  host helpers are demo-only; no jit/derivative promise is made for them.
- Sampling timing still measures sampling/compile only. New postprocessing
  timing includes both summary routes/aggregation and synchronizes output.

Evidence (2026-10-31):
- Explicit `JAX_ENABLE_X64=1 uv run --group examples python` process called
  check_trial(), check_aggregation_and_grids(), time_grids(make_trial().data):
  exit 0. Trial c=19.1725354361981; arithmetic and grid checks passed;
  actual trial grids (201,)/(101,), tau_max=13.0 years.
- Hand arithmetic check: deltas [1,-1,0], mean 0, sample SD 1,
  95% interval [-.95,.95], P(delta>0)=1/3; k draws [1,3,5] have sample SD 2.
  This checks pairing, strict positivity and aggregation independently of PH.
- Grids: p90([1,6])=5.5 and p90([1,6.4])=5.86 both give tau_max=5.5;
  short follow-up [1,4] raises. Exact expected linspaces check all points.
- `JAX_ENABLE_X64=1 uv run --group examples python -c 'from
  examples.bayesian_rmst.check_model import check_posterior_routes;
  check_posterior_routes()'`: exit 1, AttributeError: survjax has no
  ProportionalHazards, before RMST/survival evaluation. These analytical
  exponential and route-agreement assertions are implemented, NOT verified.
- `uv run ruff format --check examples/bayesian_rmst`: 3 files formatted.
- `uv run ruff check .`: All checks passed! (initial import-order lint fixed).
- `uv run mypy src`: Success: no issues found in 20 source files.
- `git diff --check`: exit 0.
- Direct-file and module runner --help both exit 0 with smoke/full choices.
- `scripts/check.sh --through A10`: exit 1 at unchanged README.md:67/72
  embedded-code format failure after guard ok. Later checks did not execute.
- Ordinary core suites are not repeated: no source/core/test changes; the
  previous step's suites were 1065/1071 passing in x64/x32. This is historical
  evidence, not a run for this commit. Float32 demo checks are not applicable:
  the design requires float64.

Requirements mapped to checks:
- 02-details Grids: full expected linspaces, exact/rounded endpoint and
  insufficient follow-up failure.
- Spec 11.3 per-draw effects: paired differences and independently derived
  summary statistics. RMST/survival batch axes: analytical shape-one cases
  with distinct scales/HRs (pending PH execution).
- Spec 11.3 route agreement: Predictive alignment and strict absolute 1e-12,
  with zero in the grid (pending PH/RMST execution).

Unknown / later steps:
- All PH-dependent execution, the approved smoke/full posterior, route errors
  and timings remain unverified until integration. No substitute core is added.
- Full convergence/truth audit, per-summary diagnostics, prior/posterior plots,
  posterior predictive KM, external KM validation and executable docs/results
  are later work. This step records means/intervals/probability, not those gates.
- Full smoke/module execution is not repeated: the PH API is still absent and
  the explicit posterior check confirms that boundary. No NUTS result claimed.

---

# Preserved incoming main RMST review (verbatim)

Independent review: round 1/3

# RMST.final — Combined evidence and maintainer handoff

Read first, in order:
1. Active PLAN.md RMST.final section; spec 8.7,10.3,10.5,12.1–12.2.
2. plans/release-agents/rmst-evidence.md: complete requirement/test map,
   combined checks, installation artifacts, decisions and remaining owners.
3. docs/rmst.md's two-line clarification of native calculation vs PH delegation;
   plans/release-agents/rmst.md's evidence link.
4. Check evidence/limits below and the unchanged D-028–D-032 decisions.

Step start /previous final: 4f8f2dbfa3a39f95e11b1efc34a8675d7509994a.
Original base: 8757c0fda78e34e6e818249adedd2a144a33c33d. Branch release-rmst.
PH.8 was integrated at 393c4e9; this is sub-step 3, the last planned RMST
implementer step. No numerical implementation, test, tolerance, API/export,
dependency, appendix, spec or protected-file edit. Both workstreams' complete
plan history remains verbatim. No branch merge, rebase, push or publication.

## Changes and invariants

RMST.d3 N-1 is addressed: native Exponential/Weibull evaluate their own RMST
formulas directly, whereas PH constructs the equivalent built-in and delegates.
Only that prose sentence changes; all 16 executable RMST cells are unchanged.

The permanent handoff is beside ph-evidence.md, linked from the original RMST
launch plan. It covers every staged 8.7/12.1 row and links PH's remaining
wrapper requirements. Full test names were checked against Python AST
function definitions: 33 explicit references resolve. Whole-module/API/install
references also exist. No expected numeric value is regenerated or edited.

Key mapped evidence: test_rmst.py::test_rmst_derivatives_against_independent_integral
covers native Exponential first/Hessian/third/fourth derivatives;
test_weibull_rmst_derivatives.py::test_weibull_rmst_first_derivatives_full_range
covers native first directions at 142 full-range/switch points;
test_proportional_hazards_rmst_derivatives.py::test_full_range_values_and_first_parameter_derivatives
uses physical PH survival integrals; supported Hessian and shape-rejection
cases are listed separately. The map identifies boundaries, failure/isolation,
broadcasting, staged surface, 16-cell guide and installed-package evidence.
These are implementer tests using independent math, not independent acceptance.

D-028 positional baseline, D-029 log-parameter errors, D-030 first-only Weibull
zero boundary, D-031 direct/likelihood derivative deferral and D-032 accepted
unrepresentable PH closure failure are all included. Formal statuses stay open
and maintainer-owned; DEVIATIONS.md is byte-identical to step start. No new
exception: D-031 does not relax RMST. NaN entries' own tangents are unclaimed;
valid-entry isolation and the mandatory accuracy domain remain unchanged.

## Fresh checks

scripts/check_wheel.sh --through A10: exit 0, all available gates passed.
Implementer suites: float64 1282 passed /8 skipped, 144.19s; float32
1287 passed /3 skipped, 226.02s. Source doctests: 4 passed per mode,
1.36s /1.23s. Existing independent acceptance through A10: float64
2332 passed /28 skipped, 110.78s; float32 2322 passed /38 skipped, 69.32s.
Guard passed. Skip counts match the prior combined gate; no skip introduced.
Full log: /private/tmp/rmst-final-gate.log. This gate does not select the
still-undelivered independent PH/staged RMST acceptance tests.


scripts/build_docs.sh invoked directly using its approved escalation: exit 0,
13 source pages rebuilt in each separate dtype process, warnings fatal.
All 16 RMST cells execute: float32 11.59s /float64 14.03s. PH's five cells
execute: 4.66s /5.62s. No Sphinx warning or notebook error; the usual kernel
TCP notices remain. HTML host audit passes in both modes: 16 RMST output
containers, numerical/shape/error messages, all three RMST API entries,
both guide links/navigation, local anchors, figure assets and exact appendix
PDF download. N-1 correction appears in both renders. Source playground
remains output/count-free. RMST.d3 records the unchanged figure's visual
inspection and unavailable browser host; this step makes no new browser
layout or MathJax-execution claim. Temporary host checks are outside the repo;
committed executable cells reproduce the examples via the direct docs build.

Ruff format: 139 files already formatted (gate's earlier run: 138);
Ruff check: all checks passed; mypy: no issues in 22 source files.
Whitespace clean. Host assertions: full prior PLAN.md preserved; no source,
test, decision, appendix, config, spec-marker or protected-path change.
Spec remains revision 41, design 00e7225b6b10b0d4e3137f7cf85c0b76e377d821.
No new test is appropriate for a prose/evidence-only step; the unchanged full
gate and executable docs assertions are the verification.

## Installation evidence

Fresh source archive and archive-built wheel are under
/private/tmp/survjax-rmst-final.c0dut0pb (exact names/hashes in the handoff).
The gate wheel is under /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.8foYDm.
All 23 source files match the checkout; py.typed, PH and _rmst.py present,
_numerics.py absent. Both wheels have 31 equal members and identical complete
bytes. Both SHA256s are e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d;
archive SHA256 ecce72f145b11658820b863cef1f9473403b432c48f0796d0666fdfb5f121e21.
Prior integration artifacts remain present, rehashed and equal. New manifest
is retained alongside the archive, and no artifact was removed.

uv build --offline --no-python-downloads --sdist succeeded; safely unpacked
with tarfile's data filter and built its wheel from the unpacked directory
using uv's built-in backend. No dependency or committed helper added. The
known uncached-backend limitation of direct isolated PEP-517 archive builds
is retained; that route was not retried or claimed to pass offline.

Existing tests/install/test_wheel.py with SURVJAX_WHEEL naming the archive-built
wheel, separate JAX_ENABLE_X64=1/0 processes: 3 passed in 17.80s /26.58s.
Fresh external runtime-only environments verify public exports, type marker,
core independence, PH RMST eager/jit and strict installed custom-family typing.
Logs: /private/tmp/rmst-final-archive-install-x64.log and -x32.log.

Environment checked: Darwin 25.6.0 arm64, CpuDevice(id=0), Python 3.12.9,
JAX/jaxlib 0.11.2, NumPy 2.5.3, pytest 9.1.1, mpmath 1.4.1,
Ruff 0.16.9, mypy 2.3.1, uv 0.11.19. No new platform/version claim.

## Remaining ownership and limits

No further RMST implementation/docs sub-step is planned after independent
review. Maintainer owns release-rmst -> main integration, formal D-028–D-032
disposition and independent PH/staged RMST acceptance delivery/sync plus the
subsequent cumulative run. Pending acceptance is confirmed in TESTMAP.md's
Revision 37 not-yet-tested section and API_ASSUMPTIONS.md Q-049. An existing
A10 pass is not evidence of those missing acceptance tests. No tests/spec
edit, skip/xfail or tolerance change was introduced.

The handoff identifies separate A9/A10/PH and release-demo owners; it does not
certify all of Slice A or the full public release. No R regeneration, new CPU
benchmark, accelerator/demo run or Linux CI claim. Generic summaries/full
shape Hessians/quadrature tangents remain later scope. No open numerical
question was introduced by this record. Commit, independently review (max
three rounds), stop/report. No merge/push/publication by this agent.
