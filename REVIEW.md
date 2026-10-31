Independent review: round 1/3

# RMST.d1 — Executable RMST guide and README scope correction

Read PLAN.md RMST.d1, spec 8.7/10.5/12.1–12.2 revision 39, then
docs/rmst.md, README.md and the four added docs/index.md lines.
Start/base: aae68920769855aa08227f4b7c4818ce7a820923, merged on main.
No library code, docstrings, tests, tolerances, exports, dependencies, spec,
protected files or existing mathematical appendix were changed. This is one
RMST documentation topic, about 225 changed content lines; records separate.

Scope and invariants:
- Guide covers the merged Exponential and Weibull APIs. PH is now free to
  implement closure after core integration, but neither wrapper nor export
  is in this checkout. Its examples and combined evidence are explicitly
  labelled pending, without importing unmerged code. docs/api.md stays with PH.
- Define M(tau)=E[min(T,tau)]=integral S and time units. Exponential example
  is independently hand-derived: (1-exp(-0.2*12))/0.2 ~= 4.54641023 months.
  Weibull k=1 agrees. No fit or comparison with external data is claimed.
- Explicit (cutoffs,draws,arms) axes preserve (3,3,2), with per-draw contrasts;
  jit and vmap agree. No rmst_grid, quadrature fallback or new summary API.
- Positive finite parameters; invalid numeric cutoffs/parameter issues return
  NaN even at zero. Invalid argument types/shapes raise TypeError. A NaN's
  own tangent is unspecified; valid shared-parameter isolation stays required.
- First k/b/tau derivatives in both modes; dM/dtau=S(tau) identity and the
  absolute cutoff-error convention; log-parameter error coordinates; finite
  supported Hessian with shape held separately; native shape Hessian raises;
  Weibull zero asserts only value/first derivatives. No higher boundary promise.
- Precision statements quote the staged domain/targets of 8.7, not a new
  empirical study or global promise. The existing math appendix stays outside
  docs and is linked via a copied PDF download; no derivations duplicated.
- README corrects RMST.integration N-1 and keeps other summaries planned.
  The requested guide is discoverable through one narrow index/toctree entry;
  A9 navigation changes are already merged. No global docs config change.

Executable/documentation checks:
- scripts/build_docs.sh was invoked directly with approved escalation, no
  redirection, env prefix or shell wrapper. Both strict (-E -n -W) dtype
  builds passed and execute the new page's eight code cells. Minor notation/
  parameter-domain clarifications prompted rebuilds; the final direct call
  exited 0 with both builds successful. Only the usual notebook-kernel TCP
  startup warnings appeared; no Sphinx warnings or execution errors.
- Observed RMST(12): 4.54641056060791 x32, 4.546410233552938 x64. Shapes
  (3,3,2); finite first derivatives in both modes; zero derivatives exactly
  (0,0,1); supported Hessian finite and shape Hessian restriction caught.
  Code-cell allclose checks illustrate analytical identities with JAX defaults;
  they do not alter the implementation test tolerances or claim new acceptance.
- Parsed generated HTML from both modes: all five section anchors, navigation,
  local links/images and printed outputs present; downloaded appendix PDF is
  byte-identical to appendices/weibull-rmst.pdf. No notebook error outputs.
- Viewed the rendered area figure for legible axes/legend and cutoff shading;
  both images are 590x290 with the expected curve/area; a decoded comparison
  found only one differing pixel out of 171100 between dtype modes. Full browser viewport
  inspection was attempted, but the CUA browser access host is unavailable.
  Thus full-page browser layout is not claimed. HTML structure/content/assets
  and the plot were inspected instead; no new dependency was installed.
  The temporary loopback server was stopped after the inspection attempt.
- uv run ruff format --check .: 126 files already formatted.
  uv run ruff check .: all checks passed. uv run mypy src: success, 21 files.
  uv run --group test pytest tests/unit/test_api_reference.py -q: 1 passed.
  git diff --check clean. No numerical suite rerun for prose-only changes;
  the unchanged core passed the full wheel-backed A10 gate in RMST.integration.

Remaining: the separate PH closure implementation/integration, its executable
examples, and item 3's combined release checks. This guide does not certify
PH or full release completion. Publication/hosting remains a separate action.
Commit, run independent-review for RMST.d1, handle at most 3 rounds, then stop.
