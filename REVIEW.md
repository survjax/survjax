# PH.3c — RMST near-unit log-ratio precision

Independent review: round 1/3

Step start and previous PH.3b final: e630d1c721600b6554b406cd8b78309b446a4847.
Original worktree launch: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-ph, current AGENTS.md/CLAUDE.md read. The unfinished PH PLAN
history remains here. No dependencies, public API, modules, protected paths,
tolerances, direct methods, likelihood, docs or PDFs changed.

## Read first

1. PLAN.md PH.3c and spec 7.2–7.3, 8.7 and 12.1–12.2, revision 40.
2. src/survjax/_numerics.py, only the log-ratio evaluation change.
3. tests/unit/test_weibull_rmst_near_scale.py: points and independent oracle,
   cutoff regression, parameter derivatives, supported Hessians, inactive-route
   safety. Existing native/PH RMST suites cover the original domain, boundary,
   broadcasting, invalid isolation and unsupported shape-Hessian behavior.

## Trigger and correction

After PH.3b was committed and under review, an extra probe found an ordinary
float32 failure at k=50,b=10000,eta=0,tau=10000.005859375 (six ULPs above b).
Both JIT forward/reverse cutoff derivatives were 0.3678792715072632;
60-digit physical survival is 0.3678686634565983. Absolute error
1.0608050664884239e-5 exceeds unchanged RMST_CUTOFF_ATOL=5e-6.
This is an in-domain native RMST error, inherited by PH, independent of the
PH.3b closure correction. D-031/D-032 do not waive this contract.

The new test was run BEFORE changing production code. Its float32 native
cutoff case failed: four of 30 points above tolerance, max error 1.06080507e-5.
Then replace log(tau)-log(b) with log1p((tau-b)/b) when the logs differ by
less than 1/2. Logs of nearby values lose low-order ratio bits; large shape
amplifies that loss. Within this branch tau/b lies between exp(-1/2) and
exp(1/2), inside [1/2,2], so subtraction is well-conditioned (Sterbenz for
normal floats), relative difference is bounded away from -1, and division is
bounded. This is equivalent algebra, not an approximation or probability clip.
The threshold selects a stable expression, not a parameter restriction.

Retain the log-difference for separated scales: dividing them could overflow,
and (tau-b)/b could round to -1. Before subtraction/division/log1p, replace BOTH
operands of that inactive route with 1. Thus its log1p argument is zero and
all local derivatives are safe under eager, jit and vmap. The fallback logs
already see positive finite operands after existing zero/invalid substitutions.
No active tangent is stopped. No floor, epsilon, global dtype toggle, tracer
to NumPy conversion, custom derivative or new option.

After correction the reported point's forward/reverse errors are
6.118074991e-8 /9.098307230e-8, about 100 times smaller than before and below
the ORIGINAL tolerance. Value policy and all branch approximations remain
unchanged. The shared private numerical helper must be fixed so native and
PH-delegated RMST agree; editing only the wrapper would leave the native bug.
The RMST core is already integrated; no parallel unfinished work is copied.

## Independent references and requirement/test map

The fixture has 30 native /48 PH points: k={0.2,2,50}, b={1e-4,1e4}, eta=0
(and 0.3 for PH), six ULPs below /at /above the equivalent scale. Twelve
points per family straddle BOTH expression boundaries log(tau/b)=+/-1/2
with relative offsets 0.9999 /1.0001, at eta=0. References consume the actual
dtype-rounded input arrays; host conversions occur only outside transforms.

At 60 mpmath digits, let a=exp(eta), r=tau/b, H(u)=a*u**k and S(u)=exp(-H(u)).
Integrate the physical law, not a library/gamma/closure output:
M=b*integral_0^r(S), I_H=integral_0^r(H*S).
First derivatives: M_k=-b*integral(H*log(u)*S), M_b=k*I_H,
M_eta=-b*I_H, M_tau=S(r). For native the unused eta direction is identically
zero. No production output supplies any expected value.

Supported Hessian oracle differentiates these expressions. Integration by
parts gives dI_H/deta=(r*H(r)*S(r)-I_H)/k. Thus:
- M_bb=-k*r*H*S/b; M_btau=k*H*S/b; M_tautau=-k*H*S/tau;
- M_beta=r*H*S-I_H; M_etaeta=-b*M_beta/k; M_etatau=-H*S.
Native eta rows/columns are zero. Shape stays OUTSIDE Hessian arguments;
no unsupported shape-Hessian promise or catch/skip is introduced.

- cutoff regression: spec 8.7 fundamental theorem against independent S(tau),
  both first-AD modes, eager vmap and jit(vmap), plus unbatched eager/jit at
  the reported tuple. Unchanged absolute cutoff tolerance, rtol=0.
- values/parameter derivatives: spec 8.7/12.1, eager/jit values, forward/reverse
  first derivatives in k,b,eta. Existing ordinary value RTOL with zero atol;
  parameter error scaled by (k,b,1)/|M| to existing RMST_PARAMETER_ATOL,
  rtol=0. Every assertion uses equal_nan=False.
- supported Hessians: spec 8.7, all four forward/reverse compositions under
  eager vmap and jit(vmap), compared to the independent calculus above.
  Units are (b,1,tau); compare H_ij*unit_i*unit_j/|M|, dimensionless natural
  scale, with unchanged ordinary ATOL/RTOL (no new tolerance). Typical entries
  are O(1) to O(k); these comparisons cover rounding error on either route.
- inactive-route safety: spec 7.2/8.7, native/PH zero and positive cutoffs,
  b/tau=1e+/-60, jit(vmap) values and forward/reverse cutoff derivatives under
  jax_debug_nans. The lower correction is relatively <1e-120; the upper tail
  is exponentially smaller. Expected values are 0, tau and b*Gamma(3/2),
  derived from the physical survival limits; survival rounds to [1,1,0].
  These extra values lie outside the mandatory ratio range, but remain
  representable; only values/cutoff derivatives, not extreme-scale Hessians,
  are asserted there.

## Actual validation

Environment: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1.

- Before production edit: `JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_weibull_rmst_near_scale.py -x`:
  **1 failed in 3.01s**, four cutoff-gradient mismatches, max error above.
- After production edit, initial six new cases per dtype before adding the
  two inactive-route cases: **6 passed in 25.45s float32 /24.66s x64**.
- Final focused regression command (separate dtype processes):
  `JAX_ENABLE_X64=0 uv run pytest -q tests/unit/test_weibull_rmst_near_scale.py tests/unit/test_weibull_rmst_derivatives.py tests/unit/test_weibull_rmst.py tests/unit/test_proportional_hazards_rmst.py tests/unit/test_proportional_hazards_rmst_derivatives.py tests/unit/test_rmst.py -x`:
  **69 passed in 184.45s**. Identical with JAX_ENABLE_X64=1:
  **69 passed in 182.42s**. Logs /private/tmp/ph3c-focused32.log and
  /private/tmp/ph3c-focused64.log. Includes all eight final new cases.
- `uv run ruff format --check .`: 135 files formatted;
  `uv run ruff check .`: all checks passed;
  `uv run mypy src`: no issues in 22 files; `git diff --check`: clean.
- `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed**.
  Final source and final test file, no edits during the gate:
  - own x64: **1279 passed, 8 skipped in 169.10s**;
  - own float32: **1284 passed, 3 skipped in 138.28s**;
  - doctests: **4 passed** per dtype, 1.25s /1.33s;
  - independent acceptance x64: **2332 passed, 28 skipped in 87.57s**;
  - independent acceptance float32: **2322 passed, 38 skipped in 73.90s**.
  Gate log /private/tmp/ph3c-wheel-gate.log. Wheel retained at
  /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.Ug9hO7/survjax-0.0.1.dev0-py3-none-any.whl;
  SHA256 b5a1a5437ca4a3631788206572c19fa763179859c174bbbdc851ac00e2791b52.
  No R fixture regeneration or docs build: neither artifact is changed.

The implementation diff is 206 lines: 196 new test lines plus 9 additions /
1 removal in _numerics.py. Plan/review records are excluded under the current
agreement. No test deletion or tolerance/acceptance/export-test change.

## Scope and remaining work

No new spec ambiguity or DEVIATIONS entry. PH.3b N-4/N-5 are optional shared-eta
isolation /fallback derivative coverage; not addressed by this single-concept
correction. S-2's D-032 wording question remains maintainer-owned; its policy
and formal open status are unchanged. Existing extreme PH direct/likelihood
limitations remain separate from mandatory RMST accuracy.

After this step: public exports/install/API-reference integration (still needs
the narrow docs/api.md code-step exception), executable PH documentation and
final PH acceptance/evidence audit. Possible appendix refinement can explain
the stable evaluation; its current log-ratio identity is unchanged. Do not
archive the unfinished PH sub-slice, merge, push, publish or start another step
after the independent-review verdict.
