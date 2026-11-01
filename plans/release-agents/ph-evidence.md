# PH implementer handoff — PH.8

Audited 2026-11-01 on `release-ph`, starting at
`eeeb47a74e734acbfd40e240496b02c04c974c53` (PH.7 final).
Original worktree base: `8757c0fda78e34e6e818249adedd2a144a33c33d`.
Spec alignment: `.spec-seen` = `765c65bba9efb3af8ef4a13c1ecb8e302a47f346`
(revision 40). This record covers the brought-forward PH item in spec 12.2;
it does not certify the whole public release.

## Requirement-to-evidence map

Paths below are relative to the repository root. All listed tests run in the
unchanged full gate in separate float64/float32 processes. These are implementer
tests unless explicitly identified as independent acceptance.

| Requirement | Executable evidence |
| --- | --- |
| Frozen registered wrapper, nested leaves, broadcasting, indexing and dtype retention (4.1–4.4) | `tests/unit/test_proportional_hazards.py::test_broadcast_pytree_indexing_and_dtypes` |
| Positional or keyword baseline; keyword-only eta (D-028) | Same file: `test_positional_baseline_matches_keyword_under_tracing`, `test_static_misuse_raises` |
| Finite-real eta, nested `baseline.` keys, eager/jit/vmap masks and indexed issues (4.7) | Same file: `test_traced_diagnostics_include_predictor_and_prefixed_baseline`; isolation file below covers all violation kinds and named conditions |
| All seven direct methods and grids, zero/infinity limits and time-first shape (4.4–4.5,5) | Same file: `test_direct_and_grid_exponential_formulas`; `tests/unit/test_proportional_hazards_tails.py::test_direct_and_grid_methods_follow_complete_time_value_policy` |
| PH identities for both built-ins, values and every first parameter derivative (4.4,12.1) | `tests/unit/test_proportional_hazards_identities.py::test_direct_identity_values_and_first_derivatives`, `test_grid_identity_broadcast_limits_and_field_derivatives` |
| `(S,1)` baseline × `(S,2)` eta; pointwise versus grid shapes, including equal axis lengths (5) | Same identity/grid tests; executable PH guide uses `(3,1)` × `(3,2)` with `K=3` |
| Exact/right likelihood, both input forms and broadcast Jacobians (6–7) | Same identity file: `test_likelihood_identity_both_forms_and_broadcast_jacobians` |
| Ordinary baseline/eta/mixed Hessians and order-2 fwd/rev checks (7.3) | Same identity file: `test_likelihood_hessian_matches_hand_derivation`; construction file: `test_likelihood_and_second_order_interior_derivatives` |
| Invalid neighbors preserve valid likelihood values/gradients/Hessians in all AD orders (7.1–7.3) | `tests/unit/test_proportional_hazards_isolation.py::test_invalid_rows_have_nan_values_and_prefixed_diagnostics`, `test_valid_contribution_derivatives_survive_invalid_neighbors` |
| Custom baseline, named conditions, generic defaults and compilation reuse (4.2,7.2) | Same isolation file: `test_custom_baseline_grid_values_and_derivatives_match_scaled_rate`, `test_runtime_parameter_issues_and_raw_observations_reuse_one_trace` |
| Improper infinity limits, reference dtype and zero shape tangent (4.5) | Construction file: `test_improper_infinity_value_and_derivatives`, `test_reference_retains_wider_baseline_dtype`; identity file: `test_weibull_zero_grid_survival_has_zero_shape_derivative` |
| Independent high-precision tail values, lower-tail recovery and overflow guards (7.2,12.1) | `tests/unit/test_proportional_hazards_tails.py` uses 100-digit mpmath PH formulas; `tests/unit/test_proportional_hazards_guards.py` covers overflow, warnings and generic-baseline tangents |
| PH RMST closure, physical-law quadrature and all first derivatives (4.4,8.7) | `tests/unit/test_proportional_hazards_rmst.py::test_rmst_values_and_first_derivatives_match_independent_integrals`; `tests/unit/test_proportional_hazards_rmst_derivatives.py::test_full_range_values_and_first_parameter_derivatives` |
| RMST broadcast identity including eta/baseline derivatives and cutoff grids | RMST derivative file: `test_broadcast_closure_identity_in_every_first_derivative` |
| RMST parameter/cutoff boundaries, invalid-entry isolation and custom failure | RMST file: `test_rmst_zero_value_and_first_derivatives`, `test_invalid_rows_preserve_shared_parameter_derivatives`, `test_custom_baseline_rmst_has_no_generic_fallback` |
| Supported RMST Hessians; explicit Weibull shape restriction | RMST derivative file: `test_supported_hessians_are_finite_across_full_range`, `test_supported_hessians_match_independent_integrals`; RMST file: `test_weibull_shape_hessian_remains_explicitly_unsupported` |
| Public exports, exact export sets, installed PH and runtime-only core (10.2–10.3) | `tests/unit/test_import.py::test_ph_public_imports_are_the_same_registered_family`; `tests/unit/test_api_reference.py`; `tests/install/test_wheel.py` |
| Executable PH teaching, eta direction, compiled ordinary derivatives and diagnostics (10.5) | `docs/proportional-hazards.md`, five retained cells, freshly executed in both modes; home prose/toctree and API links verified |
| Permanent scaling derivation | `appendices/ph-log-survival.md` and companion PDF; linked from `_scale_log_survival` docstring; kept outside standard docs |

The seven dedicated PH test modules contain 41 test functions before
parametrization. No acceptance test is substituted with an implementer test.

## Fresh checks

`scripts/check_wheel.sh --through A10`: exit 0, all available gates passed.

| Check | float64 | float32 |
| --- | --- | --- |
| Implementer tests | 1282 passed /8 skipped, 137.26s | 1287 passed /3 skipped, 131.65s |
| Source doctests | 4 passed, 1.07s | 4 passed, 0.96s |
| Independent acceptance through A10 | 2332 passed /28 skipped, 70.03s | 2322 passed /38 skipped, 65.12s |

Guard passed; Ruff format reported 136 files already formatted, lint passed,
and mypy found no issues in 22 source files.
This builds a fresh wheel and runs Ruff/mypy, both implementation suites,
source doctests and existing independent acceptance through A10. It does not
select independent PH/RMST acceptance, because those tests have not arrived.

`scripts/build_docs.sh` invoked directly: exit 0 in both dtype modes, 12 source
pages with forced notebook execution and warnings fatal. PH execution took
4.77s float32 /4.69s float64. Each rendered guide has five output containers,
correct dtype, `(3,3,2)` time-first grid, likelihood/gradient/curvature `(3,2,3)`
and one finite/one NaN diagnostic example. Home prose, toctree, sidebar and
PH/API links remain; `rmst` autodoc remains available. Source playground has
no committed outputs/counts. The survival figure was visually inspected and
its bytes match across modes and the previous step.

The README minimal Python example was freshly executed in both modes and
compared using unchanged `tests/impl_tolerances.py` bounds to the hand-derived
loss `-2*log(0.2)+0.2*9.5`, gradient `0.2*9.5-2` and `exp(-0.2*t)`.
Float64 loss/gradient: `5.1188758248682005`, `-0.1`;
float32: `5.118875503540039`, `-0.10000018775463104`.

## Artifacts and archive route

- Direct wheel: `/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.0jF4mE/survjax-0.0.1.dev0-py3-none-any.whl`.
- Source archive: `/private/tmp/survjax-ph8.O46E9K/survjax-0.0.1.dev0.tar.gz`.
- Archive-built wheel: `/private/tmp/survjax-ph8.O46E9K/from-sdist/survjax-0.0.1.dev0-py3-none-any.whl`.
- Both wheel SHA256s: `e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d`.
- Source-archive SHA256: `ecce72f145b11658820b863cef1f9473403b432c48f0796d0666fdfb5f121e21`.

`uv build --offline --no-python-downloads --sdist --out-dir /private/tmp/survjax-ph8.O46E9K`
succeeded. All 23 packaged source files match the checkout, including the
wrapper, `_rmst.py`, both export modules and `py.typed`; `_numerics.py` is absent.

Direct `uv build --offline --no-python-downloads --wheel ARCHIVE --out-dir OUT`
failed resolving uncached `uv_build>=0.11.19,<0.12`. No dependency was added.
Safely unpacked that archive to a temporary directory, then ran the same wheel
build against its unpacked root. uv's built-in backend built it offline. All
31 wheel members and the full wheel hash equal the direct wheel. Ran
`tests/install/test_wheel.py` against this archive-built wheel with
`SURVJAX_WHEEL` set, `JAX_ENABLE_X64=1` and `=0`, separately:
**3 passed in 26.77s each**. Both checks installed only locked runtime packages
into fresh external environments, evaluated PH RMST eagerly/under jit, and
strictly type-checked the installed external custom family. The isolated
PEP-517 archive command itself is not claimed to pass offline.

Environment: macOS 26.6.2 arm64, CPU (`cpu:0`), Python 3.12.9, JAX/jaxlib
0.11.2, NumPy 2.5.3, pytest 9.1.1, mpmath 1.4.1, uv 0.11.19. No installed
`uv_build` distribution was found; directory builds used uv's built-in backend.
No Python/JAX matrix or accelerator support is inferred.

## Limits and remaining integration

- Extreme PH direct/likelihood parameter derivative stabilization is deferred
  under revision 40 /D-031. Accurate stress values and eta-only derivatives
  do not prove unrestricted baseline/mixed derivatives. Ordinary supported
  derivatives and value/isolation requirements remain unchanged.
- Weibull/PH-Weibull higher derivatives involving shape remain unsupported;
  zero-boundary guarantees cover values and first derivatives. RMST accuracy
  is measured on the equivalent-family scale domain, not arbitrary eta.
- D-032's accepted unrepresentable PH closure can return NaN outside that
  domain despite clean original diagnostics. Zero cutoff stays exact.
- Independent PH and staged RMST acceptance are **pending maintainer-owned
  test delivery/sync**, explicitly recorded in `tests/spec/TESTMAP.md` under
  "Revision 37 ... not yet tested" and `API_ASSUMPTIONS.md` Q-049. No protected
  test/spec edit, skip, xfail or tolerance change was made here.
- The RMST implementer owns separate narrative coverage, including PH closure,
  custom-baseline failure, derivative restrictions and the numerical-failure
  caveat removed from PH by request. `README.md` still calls RMST planned;
  correct that capability sentence with the RMST documentation integration.
- Release-wide merged-state tests, release demos, final RMST docs and formal
  deviation disposition remain with their assigned owners/maintainer. No A9
  worktree needs reopening. No further PH feature step is identified here.
- No R fixture regeneration or CPU benchmark rerun; committed reference tests
  run in the gate. Existing `benchmarks/slice_a_cpu.json` records the earlier
  Slice A baseline and its revision; it is not a new PH performance result.
- Publication, remote setup, merging and pushing remain separate actions.
