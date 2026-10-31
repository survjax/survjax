# PH.3b — RMST range, broadcast derivatives and supported Hessians

Independent review: round 1/3

Step base: 61327fd4cb3a7b8d4f67c4a95b8c09d910c28bb8, clean release-ph after
the maintainer's agreement update. Previous PH step final:
b49638e2aec294bcaa0bab0c07443e2bf249aedd. Original worktree launch base:
8757c0fda78e34e6e818249adedd2a144a33c33d. Updated AGENTS.md and CLAUDE.md
(the same file via symlink) were read and are unchanged here. This step has
291 changed implementation lines: 266 new test lines and 22 additions/3
removals in the private Weibull closure. No public API or module added.

## Read first

1. PLAN.md PH.3b, DEVIATIONS.md D-032's explicit maintainer acceptance.
2. Spec 4.4, 7.2–7.3, 8.7 and 12.1–12.2, revision 40; staged RMST accuracy
   and derivative contracts remain unchanged by the extreme direct-method
   derivative exception. .spec-seen remains 765c65b (already aligned).
3. tests/unit/test_proportional_hazards_rmst_derivatives.py: point generator,
   independent reference, range checks, broadcast Jacobians, supported
   Hessians, then guarded-fallback regressions.
4. distributions/weibull.py _ph_equivalent: why ordinary scale arithmetic
   needs the product form, and why both routes substitute operands first.
5. Existing test_proportional_hazards_rmst.py and native RMST derivative suites
   as complementary boundary/invalid-row/shape-restriction evidence.

## Maintainer decision and scope

PH.3a round 2 found no remaining code/test fix. Its only decision was D-032:
NaN from valid PH inputs whose natural closure is unrepresentable, outside
the promised accuracy range. The maintainer answered “ok agreed and on to the
next bit” to the recommendation to accept/document a detectable NaN there for
the initial release and defer log-scale support. That decision is recorded in
D-032 without changing its formal open status (maintainer-owned). No further
interpretation, numerical change or spec edit answers that old question here.
The future PH topic must explain that clean original parameter diagnostics do
not guarantee a representable summary closure or a finite result.

This step does not loosen the mandatory RMST accuracy range or tolerances.
The discovered cutoff-derivative failure below is INSIDE that range and is
fixed, not relabelled as D-031/D-032. Source changes only the existing private
Weibull closure. Native RMST/gamma code, Exponential, wrapper direct methods,
likelihood, exports, dependency files and protected tests/scripts are unchanged.
No generic summaries, new result types, sampling, AFT or public options.

## Independent oracle and numerical scales

Reference inputs are the actual dtype-rounded JAX arrays, converted to host
only outside transformations. mpmath uses 60 decimal digits. For baseline
scale b, shape k and a=exp(eta), set u=t/b and limit=tau/b. The physical PH
survival integrand is S=exp(-a*u**k). Integrate across decade cuts measured
in equivalent units exp(-eta/k), so small/large limits and steep k=50 decay
are resolved. Expected M=b*integral(S), not incomplete gamma or another
library call.

Differentiate the physical law under the integral. Let I_H=integral(a*u**k*S):
- dM/deta=-b*I_H, dM/db=k*I_H;
- dM/dk=-b*integral(a*u**k*log(u)*S);
- dM/dtau=S(limit).
For Exponential, k=1,b=1/r and dM/dr=(dM/deta)/r. These compute every expected
first derivative independently. No wrapper/closure/native RMST output supplies
expected values. The independent Hessian oracle differentiates a separate
physical survival integral with mp.diff at dtype-rounded interior points.

Compare parameter errors in existing section-12.1 units: multiply r/k/b errors
by their natural parameter and eta errors by 1 (already a log coordinate),
then divide by |M|. Use unchanged RMST_PARAMETER_ATOL (2e-13 /5e-5), rtol=0;
cutoff errors use the unchanged absolute RMST_CUTOFF_ATOL (2e-13 /5e-6).
No new/looser tolerance. Values use zero atol and existing ordinary RTOL;
allclose always uses equal_nan=False. Hessian entries at the independent
points are O(1), so ordinary ATOL/RTOL cover rounding-only AD/quadrature
comparisons. check_grads uses the existing separately justified finite-
difference eps and derivative tolerances. Weibull k is OUTSIDE differentiated
arguments for supported Hessians, also when k is mapped over a range grid.

## Coverage (10 new cases, plus 19 existing PH RMST cases)

- test_full_range_values_and_first_parameter_derivatives (2): Exponential
  42 Cartesian rows; Weibull 252 Cartesian rows plus 16 switch neighbors.
  Cross k={0.2,0.5,1,2,10,50}, equivalent tau/b_PH={0.01,0.1,0.5,1,10,100,1e5},
  original b={1e-4,1,1e4}, eta={-0.4,0.3}. Every eta sign is crossed at every
  tuple. Straddle x=0.1 and x=64+4/k on both sides. Eager-map/jit-map values,
  fwd/rev first AD, mapped and jit-mapped; unmapped eager gradient cross-checks
  span gamma/lower/mean branches. No NaN is allowed in this mandatory domain.
- test_broadcast_closure_identity_in_every_first_derivative (2): baseline
  (S,1), eta (S,2), scalar or explicit (m,1,1) cutoffs, m=S=3. Jacobian leaves
  cover rate or shape/scale and eta, with all coupling axes; eager/jit and
  jit(vmap) across two parameter draws. Analytical Exponential/Weibull closure
  is the identity reference; independent range integrals validate its physics.
- test_supported_hessians_are_finite_across_full_range (2): full Exponential
  r/eta/tau and Weibull b/eta/tau, every point above, all four fwd/rev second-
  order combinations under jit(vmap). No higher shape derivative is asserted.
- test_supported_hessians_match_independent_integrals (2): independent full
  3x3 Hessians, all four eager/jit AD compositions, plus order-2 forward and
  reverse check_grads. Closes PH.3a N-1's supported-Hessian evidence gap.
- test_log_space_closure_fallback_keeps_representable_rmst (2): eta=±800 with
  b=1e±300 in x64, eta=±100 with b=1e±30 in float32, k=1 and equivalent
  cutoff ratio about 1/2. The multiplier over/underflows alone, yet scale and
  RMST are normal/representable. Independent exponential-law mpmath formula,
  eager/jit. These are not D-032 failures: the closure IS representable and
  must remain accurate. They guard the alternate route introduced below.

Existing PH.3a tests retain original-invalid-row isolation, exact zero values
and first derivatives (including eta=±2000), mixed-dtype grids, custom-family
rejection before diagnostics, and explicit shape-Hessian NotImplementedError.
The optional new shared-eta invalid-neighbor test (PH.3a N-4) passed both modes
on this source but was exchanged for the more important positive-cutoff
fallback regressions to fit the budget; permanent extra N-4 coverage remains
future defence-in-depth. No existing regression was deleted or weakened.

## Actual in-domain precision failure and correction

The initial smaller grid alternated eta signs and passed. A targeted float32
probe with k=50,b=1e4,eta=0.3 and tau=rounded(b*exp(-eta/k)) found cutoff
error 1.777321151e-5 against S(tau), above the existing 5e-6 bound. The old
closure computed exp(log(b)-eta/k); the log/exp round trip perturbed the
natural scale, and shape 50 amplified that perturbation near tau/b_PH=1.
Expand to BOTH eta signs at every Cartesian tuple so this row is a regression.

For normal finite multiplier AND final scale, compute b*exp(-eta/k). For all
other entries keep the guarded log-space route. This is the same PH law, not
an approximation or clipping. The log(finfo.tiny/max) checks choose a safe
algebraic expression; they never floor/cap a model parameter or probability.
Original issues are still neutralized before division/log/exp. The discarded
product substitutes scale=1 and multiplier-log=0 before operations. The
fallback separately substitutes scale=1 and shift=0 when discarded BEFORE
its log/exp, and restores infinity on real overflow exactly as before. Both
branches therefore remain safe when vmap evaluates them. No active tangent
is stopped, no NumPy/tracer conversion, global x64 toggle, epsilon or time
adjustment.

After correction, the targeted point has absolute forward/reverse errors
1.87749527e-8 /4.85772751e-8. The full Cartesian suite passes the ORIGINAL
bounds in both dtypes. No Exponential or native RMST numerical edit is needed.
Weibull's file grows from 311 to 330 lines (301 inherited before PH closure);
one family/one responsibility remains, with no unrelated module extraction.

## Actual validation

Environment unchanged: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1.

- Initial alternating-sign draft, old closure: 10 cases passed per dtype
  (46.16s /46.10s). Ruff initially caught loop captures/lambda assignments;
  bound current loop values explicitly and used partial. No numeric change.
- Targeted old-source cutoff probe: above-bound error 1.777321151e-5. Current
  source probe: fwd/rev errors 1.87749527e-8 /4.85772751e-8.
- Expanded both-sign suite plus PH.3a, with optional shared-eta cases:
  29 passed float32 (67.00s), 29 passed x64 (66.34s).
- Final suite (positive-cutoff fallback cases replace optional shared eta):
  `JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_proportional_hazards_rmst_derivatives.py tests/unit/test_proportional_hazards_rmst.py -x`:
  **29 passed in 70.89s**. Separate JAX_ENABLE_X64=0: **29 passed in 71.64s**.
- `uv run ruff format --check .`: 134 files formatted. `uv run ruff check .`:
  all checks passed. `uv run mypy src`: no issues in 22 files. Whitespace clean.
- `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed** on the
  final production source. Own 1271 passed/8 skipped x64 (118.03s), 1276/3
  float32 (126.80s); doctests 4 each (0.95s/0.88s); independent acceptance
  2332/28 x64 (72.49s), 2322/38 float32 (60.30s).
  The gate began before the two optional shared-eta cases were exchanged for
  the two fallback cases. Production source stayed unchanged; the final
  focused processes above verify the final test-file version in both modes.
- Wheel `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.WRAhO5/survjax-0.0.1.dev0-py3-none-any.whl`;
  SHA256 `523f364acb9c8b29f86ba5a61c3ba4d855815b3c506652e8005d7bc13887f113`.
  Gate log `/private/tmp/ph3b-wheel-gate.log`.

## Remaining work and limits

No universal safe eta/scale boundary is inferred. D-031 direct/likelihood
extreme-derivative limitation and D-032 accepted numerical-failure policy
remain distinct; neither excuses inaccurate ordinary-domain RMST results.
Formal D-032 spec disposition stays with the maintainer/designer; no status
is marked resolved. No protected file, tolerance, dependency or public export
change, no docs/PDF edit, no R fixture regeneration, no merge/push/publish.
The PH sub-slice remains in progress; prior PLAN history is preserved.

After review, remaining sub-steps: public-export/install/API-reference
integration (narrow docs/api.md code-step exception still required), executable
PH topic including both limitations, and final PH evidence/acceptance audit.
Additional defensive shared-eta/zero-Hessian coverage may join integration;
not a claim of initial-release completion. Commit, invoke independent-review
exactly for PH.3b, address findings within three rounds and stop/report.
