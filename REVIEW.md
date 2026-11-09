# AFT.2 — Numerical and integration verification

Independent review: round 1/3

Start and previous-step commit: 09148c5. Original worktree base: cc0e30f.
Read PLAN.md AFT.2, spec 4.4–4.5/7.1–7.3, wrappers.py (unchanged), then
new tests/unit/test_aft_integration.py. No public API, numerical library code,
docs, dependencies or protected files changed. No spec exception introduced.

Invariants and test map:
- test_extreme_predictor_preserves_representable_time uses eta=+/-720 (x64)
  or +/-90 (x32), times 1e+/-300 or 1e+/-30. exp(-eta) alone cannot be used
  across these regimes; combined log times remain representable. Unit-rate
  exponential log S, log f, log h, eta derivatives and joint rate/eta Hessians
  have explicit host float64 analytical references. TAIL_RTOL, atol=0 prevents
  tiny tails or derivatives being accepted as zero (4.4/7.1/7.3).
- Custom CureBaseline has log S=c*expm1(-t) and generic autodiff log density
  and hazard. Its finite-time formulas use hand differentiation. Infinity
  survival exp(-c), its amount/eta Jacobians and full Hessian are checked
  analytically, preserving the improper limit independently of eta (4.2–4.5).
- Nested AFT wrappers broadcast (2,1) with (2,2); equivalent exponential
  rate r*exp(-inner_eta-outer_eta) supplies the time-first grid reference,
  checked eager/jit/vmap, including zero and infinity (4.4/5).
- Built-in exact/right likelihoods match equivalent Weibull/Exponential and
  separate explicit formulas. Both first derivative modes, fwd/fwd, rev/rev
  and fwd/rev second derivatives agree eager/jit/jitted-vmap (6.3/7.3).
  Each valid mixed likelihood is run eagerly with jax_debug_nans enabled.
- Mixed invalid rate and NaN predictor rows yield NaN; valid-row Jacobians
  and Hessian match a hand-derived reference including mixed rate/eta entries
  and zero dependence on invalid leaves (7.1–7.3).
- Existing tolerances reused without changes; no probability floors, model
  clipping or PH-specific derivative exemption. This does not claim values
  can be recovered when the rescaled time itself is unrepresentable: AFT.1
  detects those failures as NaN. Smooth tests do not expand time-derivative
  guarantees or arbitrary custom-family differentiability guarantees.

Validation:
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_aft_integration.py --durations=15: 10 passed (8.82s), before the strengthened two-distinct-sets vmap check.
- Same command JAX_ENABLE_X64=0: 10 passed (8.82s), before the strengthened two-distinct-sets vmap check.
- scripts/check.sh --fast --through A10: all gates passed. Formatting 195 files
  already formatted; Ruff clean; mypy 22 source files clean. Implementation
  x64 955 passed/8 skipped (13.85s), x32 960 passed/3 skipped (13.86s).
  Source doctests 4 passed per dtype. Spec x64 1951 passed/27 skipped (11.43s),
  x32 1965 passed/13 skipped (11.60s). The gate included all new AFT.2 cases
  before timing-list regeneration, in addition to the full touched-file runs.
- scripts/update_slow_tests.sh --through A10 completed: 415 slow tests written,
  including the two new Hessian tests. Its script masks pytest exit statuses,
  so regeneration alone is not claimed as a full gate pass. The timing refresh
  changes 450 generated lines (mostly existing entries removed after measuring
  below threshold), beyond the 283 lines of implementation-owned tests. The
  maintainer previously directed keeping generated timing churn in the same
  commit; we follow that preference here. No hand edits to the generated list.
- Final distinct-parameter vmap check: full touched file passed x64 10/10
  (10.66s), x32 10/10 (10.58s), using the same Tier 0 command without durations.
- Final fast A10 gate after the strengthened vmap assertion: all gates passed.
  Implementation x64 955 passed/8 skipped (9.69s), x32 960 passed/3 skipped
  (10.07s); doctests 4 passed per mode; spec x64 1951 passed/27 skipped
  (12.26s), x32 1965 passed/13 skipped (10.62s). No test assertion edits after
  this run. Formatting 195 files, Ruff and mypy 22 source files passed.
- Docs build and Tier 2 gate not run: no docs edits; full gate is maintainer/CI.

Reading questions: are the focused identity cases adequate for AFT.2's stated
scope? No known failing numerical cases or unresolved spec choices observed
in these tests. Remaining after AFT.2: AFT.3 explanatory documentation.

- git diff --check passed. No library/fixture/protected-path changes.
