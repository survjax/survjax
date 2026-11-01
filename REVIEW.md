Independent review: round 1/3

# RMST.d3 — PH RMST guide, availability and API reference

Read first, in order:
1. Active PLAN.md RMST.d3 section; spec 4.4,8.7,10.5,12.1–12.2 and the
   accepted decisions in D-030/D-032. D-031 leaves RMST unchanged.
2. docs/rmst.md (16 executable cells), then README.md and docs/api.md.
3. Example/check evidence below. No numerical source or test changes.

This is integration sub-step 2, starting at previous step final
393c4e93aba42501c3eb622f525095b1eed4291d. Original worktree base:
8757c0fda78e34e6e818249adedd2a144a33c33d. Work only on release-rmst.
The complete previous PLAN.md is preserved verbatim below the new section;
both workstreams' unfinished histories remain. No new sub-slice or archive.

## Changes and contract map

- Integration3 N-1: replace the obsolete PH-planned sentence with current
  availability, both built-in closure formulas and executable values.
  N-2 /RMST.d2 N-3: add rmst to both native API member lists. Link PH/RMST
  from the API reference and README. README's existing example is unchanged.
- Native Exponential/Weibull implementations are independent. PH RMST uses
  equivalent built-in distributions; only the wrapper delegates (4.4,8.7).
- Cutoff section: (3,1) baseline x (3,2) eta, scalar (3,2), explicit cutoff
  (3,3,2), eager-equivalent/jit/vmap identities. Existing native pairing/error
  example remains (5,8.7). Mathematical family identities are the comparison
  origin, not an independently estimated reference or a new accuracy claim.
- First derivatives: PH-Weibull all k,b,eta,tau directions, forward/reverse
  agreement, eta chain rule -b_PH*M_b/k and cutoff survival identity;
  fixed-shape b/eta/tau Hessian finite. PH-Exponential all r/eta/tau directions
  agree in both modes and with the derivative of its closed expression.
  Higher shape Hessians raise for both Weibull and PH-Weibull (8.7,12.1).
- Ordinary PH zero checks: value 0, all parameter derivatives 0, cutoff
  derivative 1 for both built-in closures. Weibull/PH-Weibull zero guarantees
  stop at first order; supported higher interior derivatives exclude shape.
- Invalid eta beside valid eta: shared baseline k,b and valid eta derivatives
  of the selected valid result equal the scalar call. A NaN entry's own
  derivatives remain unspecified; valid-entry isolation remains required.
- D-032: k=.2,b=1,eta=-200 has clean original diagnostics but an unrepresentable
  equivalent scale exp(1000), producing NaN at tau=1 in both dtypes while
  tau=0 stays 0. This is outside the equivalent-scale accuracy domain, with
  no relaxation inside it. No new issue flag or repair is proposed. D-031's
  extreme PH direct/likelihood derivative exception does not relax RMST.
- A registered constant-hazard custom baseline and its PH wrapper both raise
  during jit tracing. The PH closure stays private; no generic quadrature,
  summary types/options, grids, exports, dependency or helper is introduced.

## Checks actually run

scripts/build_docs.sh invoked directly with the existing approved escalation:
exit 0, both separate dtype processes, all 13 source pages rebuilt with forced
execution and warnings fatal. All 16 RMST cells execute, with 16 output
containers per mode. RMST execution: float32 23.51s /float64 13.83s. Existing
PH guide also executes (9.72s /5.33s). No Sphinx warning or notebook error;
the usual kernel TCP notice remains. The first sandboxed build was stopped
after kernel-cleanup PermissionError from psutil/sysctl; the completed direct
unsandboxed run is the build evidence, not that interrupted attempt.

HTML host audit in both modes: all three RMST autodoc entries present;
home/sidebar/PH/RMST links and local anchors resolve; expected closure,
broadcast, derivative, zero, numerical-failure and custom-failure outputs
present; no error containers. Exact appendix PDF download matches appendices/
source. Source playground remains output/count-free. Retained RMST figure
visually inspected: readable time/survival labels and legend, cutoff 12 and
shading restricted to it. Full-page browser inspection unavailable: CUA reports
browser access host unavailable and native pipe startup failed. No browser
layout/MathJax execution claim is inferred from the host HTML audit.

Focused regressions, separate processes JAX_ENABLE_X64=1 and =0:
uv run --group test pytest -q with these seven unchanged modules:
- tests/unit/test_api_reference.py
- tests/unit/test_rmst.py
- tests/unit/test_weibull_rmst.py
- tests/unit/test_weibull_rmst_derivatives.py
- tests/unit/test_weibull_rmst_near_scale.py
- tests/unit/test_proportional_hazards_rmst.py
- tests/unit/test_proportional_hazards_rmst_derivatives.py
Results: float64 72 passed in 222.73s; float32 72 passed in 222.76s.
Logs: /private/tmp/rmst-d3-tests-x64.log and rmst-d3-tests-x32.log.
These cover native/PH staged values, full range, first derivatives, supported
Hessians, zero/isolation/unsupported calls and public API reference coverage.
No test skip/xfail/tolerance change or new redundant docs unit test.

README Python block freshly executed in both modes, checked against unchanged
impl_tolerances.py ATOL/RTOL: two events and exposure 9.5 imply
loss=-2*log(.2)+.2*9.5, log-rate gradient=.2*9.5-2, survival=exp(-.2*t).
Float64 loss/gradient 5.1188758248682005 /-0.1; float32 5.1188755 /-0.10000019.
README local links resolve. Temporary host audits are outside the repo;
committed notebook assertions are the reproducible docs checks.

Ruff format: 138 files already formatted. Ruff lint: all checks passed.
Mypy: no issues in 22 source files. Whitespace clean. Host assertions confirm
all prior plans preserved and no source/test/appendix/config/protected changes.
.spec-seen advanced to 00e7225b6b10b0d4e3137f7cf85c0b76e377d821 after reading
the full revision 40->41 diff: MNIST precision/censoring only, RMST unchanged.
No spec edit. Environment remains Python 3.12.9, JAX/jaxlib 0.11.2, NumPy 2.5.3,
pytest 9.1.1, mpmath 1.4.1, Ruff 0.16.9, mypy 2.3.1, uv 0.11.19, CPU arm64.

## Retained evidence and remaining sub-step

The full combined wheel-backed A10 gate and archive installation evidence
are in 393c4e9:REVIEW.md; not rerun for this unchanged numerical implementation.
All three artifacts were rehashed and still match:
- Direct wheel: /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.4VcAb8/survjax-0.0.1.dev0-py3-none-any.whl
- Source archive: /private/tmp/survjax-rmst-ph-integration.e_hbnrwo/survjax-0.0.1.dev0.tar.gz
- Archive-built wheel: /private/tmp/survjax-rmst-ph-integration.e_hbnrwo/from-sdist/survjax-0.0.1.dev0-py3-none-any.whl
Both wheel SHA256s: e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d.
Archive SHA256: ecce72f145b11658820b863cef1f9473403b432c48f0796d0666fdfb5f121e21.
Prior source/archive identity and fresh external runtime-only installations
remain valid evidence for unchanged source; they are not a new docs-step run.
The documented archive route used uv's built-in backend from an unpacked
directory; a direct isolated PEP-517 archive build is not claimed to pass.
Plain check.sh requires SURVJAX_WHEEL for acceptance smoke tests; the retained
combined gate was check_wheel.sh, which supplies it.

Sub-step 3 remains: final combined evidence and all five maintainer decisions
D-028/D-029/D-030/D-031/D-032, with formal statuses unchanged. Independent
PH/staged RMST acceptance remains pending maintainer delivery/sync; no A10
pass is evidence for those missing tests. No R regeneration, new performance
measurement or release-demo claim. After reviewed completion the maintainer
merges release-rmst into main. No push, publication or other branch mutation.
Commit and invoke independent-review exactly, then stop/report this docs step.
