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
