# RMST implementer handoff — RMST.final

Audited 2026-11-01 on `release-rmst`, starting at previous step final
`4f8f2dbfa3a39f95e11b1efc34a8675d7509994a` (RMST.d3).
Original worktree base: `8757c0fda78e34e6e818249adedd2a144a33c33d`.
Reviewed PH.8 through `8acfb9f866e6e530b762724ed4ec4c65d3cf1230` was integrated
at `393c4e93aba42501c3eb622f525095b1eed4291d` (RMST.integration3).
Both integration3 and d3 independent reviews returned merge in round 1.
Spec alignment: `.spec-seen` = `00e7225b6b10b0d4e3137f7cf85c0b76e377d821`
(revision 41). This record covers staged RMST, spec 12.2 release item 3,
and combined PH closure evidence; independent PH/RMST acceptance is pending.
It does not certify the whole public release.

## Delivered surface and mathematical record

Native Exponential and Weibull evaluate their own RMST formulas directly.
PH delegates to equivalent Exponential(r*exp(eta)) or Weibull(k,b*exp(-eta/k));
its direct methods/likelihood keep the PH formulas. The helper is `_rmst.py`;
`_numerics.py` is absent. No generic quadrature tangent, `rmst_grid`, sampling,
summary options/info types or additional exports were added.

The [RMST guide](../../docs/rmst.md) covers closures, cutoff broadcasting,
zero/invalid values, diagnostics, derivatives, isolation and custom-baseline
failure. [README](../../README.md) advertises PH/RMST with links; the
[API reference](../../docs/api.md) renders all three RMST methods. RMST.d3 N-1
is clarified: native evaluation is direct; delegation belongs to PH.
Permanent mathematics and code pointers remain outside the docs site in
[appendices](../../appendices/README.md), including the Weibull RMST note/PDF
and PH log-survival note/PDF. Their sources and PDFs are unchanged here.

## Requirement-to-evidence map

Test paths below are relative to `tests/unit/`. All run in the unchanged
full gate, in separate dtype processes. They are implementer tests with
independent analytical/mpmath references, not independent acceptance tests.
The [PH handoff](ph-evidence.md) maps the wrapper's other release requirements.

| Requirement (spec 8.7 /12.1 unless noted) | Executable evidence |
| --- | --- |
| Staged surface, absent APIs/options, custom-family failure | `test_rmst.py::test_staged_summary_surface`; `test_proportional_hazards_rmst.py::test_custom_baseline_rmst_has_no_generic_fallback` |
| Static cutoff dtype/shape errors (4.5,5) | `test_rmst.py::test_rmst_rejects_nonreal_cutoffs`; `test_rmst.py::test_rmst_rejects_incompatible_shapes` |
| Exponential values, scales, eager/jit/vmap | `test_rmst.py::test_rmst_mpmath_values_and_scale_changes` |
| Exponential first/Hessian/third/fourth derivatives and order-2 fwd/rev checks | `test_rmst.py::test_rmst_derivatives_against_independent_integral` |
| Native Weibull values across the required range and both power tails | `test_weibull_rmst.py::test_weibull_rmst_values_against_quadrature`; `test_weibull_rmst.py::test_weibull_rmst_power_tails` |
| k=1 identity; scalar and explicit cutoff-grid shapes | `test_weibull_rmst.py::test_weibull_shape_one_rmst_and_cutoff_grid` |
| All native Weibull first directions, transformation compositions and cutoff survival identity | `test_weibull_rmst_derivatives.py::test_weibull_rmst_first_derivatives_full_range`; `test_weibull_rmst_derivatives.py::test_weibull_rmst_cutoff_derivative_equals_survival` |
| Supported native Hessians, independent calculus and unsupported shape Hessians | `test_weibull_rmst_derivatives.py::test_weibull_rmst_supported_hessians_are_finite`; `test_weibull_rmst_derivatives.py::test_weibull_rmst_supported_hessian_matches_calculus`; `test_weibull_rmst_derivatives.py::test_weibull_rmst_shape_hessian_raises_in_both_modes` |
| Zero values/first derivatives, including extreme eta for PH | `test_rmst.py::test_rmst_zero_value_and_derivatives`; `test_weibull_rmst.py::test_weibull_rmst_zero_derivatives`; `test_proportional_hazards_rmst.py::test_rmst_zero_value_and_first_derivatives` |
| Invalid cutoff/parameter NaNs, including zero cutoff | `test_rmst.py::test_rmst_invalid_entries_are_nan`; `test_weibull_rmst.py::test_weibull_rmst_invalid_values_and_shapes`; `test_proportional_hazards_rmst.py::test_invalid_rows_preserve_shared_parameter_derivatives` |
| Valid shared-parameter derivatives isolated from invalid/failed entries | `test_rmst.py::test_rmst_invalid_entries_preserve_valid_derivatives`; `test_weibull_rmst_derivatives.py::test_weibull_rmst_shared_parameters_isolate_valid_entries`; `test_weibull_rmst_derivatives.py::test_weibull_rmst_failed_gamma_entry_isolates_shared_scale`; PH invalid-row test above |
| Valid cutoff tangents isolated from invalid cutoffs | `test_rmst.py::test_rmst_cutoff_derivatives_isolate_invalid_entries`; `test_weibull_rmst.py::test_weibull_rmst_invalid_cutoff_tangent_isolation` |
| PH values/all first derivatives against the physical survival integral over the full range (4.4) | `test_proportional_hazards_rmst.py::test_rmst_values_and_first_derivatives_match_independent_integrals`; `test_proportional_hazards_rmst_derivatives.py::test_full_range_values_and_first_parameter_derivatives` |
| PH broadcast closure identity, all first directions, explicit cutoff axes (4.4,5) | `test_proportional_hazards_rmst.py::test_rmst_broadcast_closure_identity_and_explicit_cutoff_grid`; `test_proportional_hazards_rmst_derivatives.py::test_broadcast_closure_identity_in_every_first_derivative` |
| Supported PH Hessians, independent integrals and shape restriction | `test_proportional_hazards_rmst_derivatives.py::test_supported_hessians_are_finite_across_full_range`; `test_proportional_hazards_rmst_derivatives.py::test_supported_hessians_match_independent_integrals`; `test_proportional_hazards_rmst.py::test_weibull_shape_hessian_remains_explicitly_unsupported` |
| Near-scale cancellation, switch/ULP accuracy, extreme separated/nearby scales | `test_weibull_rmst_near_scale.py` (five tests, native and PH); `test_proportional_hazards_rmst_derivatives.py::test_log_space_closure_fallback_keeps_representable_rmst` |
| Executable teaching and explicit derivative/failure limits (10.5) | `docs/rmst.md`: 16 cells, closure/derivative identities, fixed-shape Hessians, zero/isolation, D-032 failure and custom-baseline rejection; all three API entries rendered |
| Public surface and installed runtime-only package (10.2–10.3) | `test_api_reference.py`; `test_import.py`; `tests/install/test_wheel.py` (path relative to root for this last file) |

Native quadrature uses 60 digits, k in {0.2,0.5,1,2,10,50}, tau/b from
0.01 to 1e5, and scales {1e-4,1,1e4}. Native derivative checks add both sides
of the series/mean switches (142 points). PH checks use the equivalent-scale
domain, both eta signs, switch neighbors and independent integrals of physical
PH survival. Near-scale tests add adjacent floating-point values and expression
switches. Quadrature reference numbers come from independent survival integrals;
closure and cutoff-survival comparisons are identity checks, separate from
those accuracy references.

Value targets: rtol 1e-8 /1e-4 for float64/float32, zero absolute tolerance
for tail comparisons. D-029's log-parameter error units scale by |RMST|;
PH eta is already a log coordinate. Fixed measured parameter factors are
2e-13 /5e-5; cutoff factors are 2e-13 /5e-6 at absolute scale 1, rtol=0.
Origins and rounding justification stay in `tests/impl_tolerances.py`, unchanged.
The gate checks these fixed bounds; it does not remeasure or loosen them.

Exponential/PH-Exponential support all derivative orders. Weibull/PH-Weibull
support all first directions; higher derivatives excluding shape are supported
on valid smooth interior inputs. Higher shape derivatives are unsupported
(native incomplete-gamma rule); zero guarantees stop at first order for
Weibull/PH-Weibull. Finite test orders above are evidence for their cases,
not an exhaustive test of every order. A NaN result's own tangents are unclaimed.

## Fresh combined checks

`scripts/check_wheel.sh --through A10`: exit 0, all available gates passed.
Guard passed; Ruff format: 138 files already formatted; lint passed; mypy:
no issues in 22 source files. Full log: /private/tmp/rmst-final-gate.log.

| Check | float64 | float32 |
| --- | --- | --- |
| Implementer suite | 1282 passed /8 skipped, 144.19s | 1287 passed /3 skipped, 226.02s |
| Source doctests | 4 passed, 1.36s | 4 passed, 1.23s |
| Existing independent acceptance through A10 | 2332 passed /28 skipped, 110.78s | 2322 passed /38 skipped, 69.32s |

Skip counts match the prior combined gate; no skip was introduced here.
This does not select the still-undelivered PH/staged RMST acceptance tests.
A subsequent standalone Ruff format/lint run passed (139 formatted files),
as did mypy and whitespace checks.


`scripts/build_docs.sh` invoked directly: exit 0 in both dtype modes, 13 source
pages each, forced execution with warnings fatal. RMST's 16 cells execute
(float32 11.59s /float64 14.03s); PH's five cells also execute (4.66s /5.62s).
HTML audit: expected outputs, all three RMST API entries, guide/sidebar links,
local anchors, figure assets and exact appendix PDF download pass in both modes.
N-1 wording is present in both renders. Source playground remains output/count-free.
Figure inspection and browser-access limitation are recorded in RMST.d3;
this step adds no full-browser layout or MathJax execution claim.
All 33 named test references above resolve to actual test definitions;
plan history and unchanged source/tests/decisions/protected paths were checked.


The focused seven-module regression in RMST.d3 also passed 72 tests per dtype.
Its reviewer independently rebuilt docs in both modes and ran 39 selected
RMST tests per dtype; verdict merge. No mutation test is needed for this
records/prose step, which changes no numerical logic.

## Retained installation artifacts

Fresh direct gate wheel:
`/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.8foYDm/survjax-0.0.1.dev0-py3-none-any.whl`.
Fresh source archive:
`/private/tmp/survjax-rmst-final.c0dut0pb/survjax-0.0.1.dev0.tar.gz`.
Archive-built wheel:
`/private/tmp/survjax-rmst-final.c0dut0pb/from-sdist/survjax-0.0.1.dev0-py3-none-any.whl`.
Both wheel SHA256s:
`e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d`.
Archive SHA256:
`ecce72f145b11658820b863cef1f9473403b432c48f0796d0666fdfb5f121e21`.

`uv build --offline --no-python-downloads --sdist --out-dir DIR` succeeded.
Safely unpacked using tarfile's data filter, then built the wheel from its
unpacked root with `uv build --offline --no-python-downloads --wheel --out-dir DIR`.
All 23 packaged source files equal the checkout, including `py.typed`, PH and
`_rmst.py`; obsolete helper absent. All 31 wheel members and complete bytes
match the direct gate wheel. Manifest is alongside the archive. Prior
integration artifacts listed in `393c4e9:REVIEW.md` remain present and were
rehashed; fresh hashes also equal those artifacts and PH.8's artifacts.

`tests/install/test_wheel.py` with SURVJAX_WHEEL set to the archive-built wheel,
in separate JAX_ENABLE_X64=1 and =0 processes: **3 passed in 17.80s /26.58s**.
Fresh external environments install only locked runtime dependencies, verify
exports/type marker/core independence, evaluate PH RMST eager/jit and strictly
type-check an installed external custom family. Logs are retained at
/private/tmp/rmst-final-archive-install-x64.log and -x32.log.


PH.8's direct isolated archive-to-wheel command could not resolve uncached
`uv_build>=0.11.19,<0.12` offline. That route was not retried or claimed to
pass; this run uses uv's documented built-in directory-backend route. No
dependency or committed build helper was added.

Environment: Python 3.12.9, JAX/jaxlib 0.11.2, NumPy 2.5.3, pytest 9.1.1,
mpmath 1.4.1, Ruff 0.16.9, mypy 2.3.1, uv 0.11.19; Darwin arm64, CPU.
No untested Python/JAX matrix or accelerator claim is inferred.

## Decision handoff

These restate accepted maintainer decisions in [DEVIATIONS.md](../../DEVIATIONS.md).
All five formal statuses remain open and maintainer-owned; no entry is edited.

| Entry | Current implementation/guarantee and remaining disposition |
| --- | --- |
| D-028 | PH accepts positional or keyword baseline; eta remains keyword-only. The maintainer owns formal disposition. |
| D-029 | Parameter-error coordinates are log k/log b (or log r), scaled by the absolute RMST value; cutoff scale is 1. Revision 39 records approval. Keep measured factors unchanged; formal disposition remains with the maintainer. |
| D-030 | Weibull/PH-Weibull tau=0 guarantees cover value and first derivatives only; higher supported nonshape derivatives require a valid smooth interior. NaN entries' own tangents are unspecified; valid-entry isolation remains required. Revision 38 records approval. |
| D-031 | Extreme PH direct/likelihood parameter-derivative stabilization is deferred under revision 40. It does not relax staged RMST accuracy, derivative or isolation requirements. Post-release stabilization and formal disposition remain with their owners. |
| D-032 | An unrepresentable equivalent PH rate/scale may yield NaN despite clean original diagnostics outside the promised accuracy domain. Zero remains exact; the accuracy domain and valid-entry isolation requirements are unchanged. Log-scale support is deferred; formal disposition remains with the maintainer. |

## Remaining ownership and release limits

No further RMST implementation/docs sub-step is planned after this handoff's
independent review. Maintainer actions: integrate reviewed `release-rmst` into
main; formally dispose of D-028–D-032; arrange independent PH/staged RMST test
delivery/sync and the subsequent cumulative run. Acceptance remains pending,
as recorded in `tests/spec/TESTMAP.md` (Revision 37 not yet tested) and
`tests/spec/API_ASSUMPTIONS.md` Q-049. The A10 pass is existing regression
coverage, not evidence that the missing PH/RMST acceptance tests passed.
No test/spec file was edited; no skip, xfail or tolerance change was introduced.

A9/A10's existing core/concordance evidence and the linked PH handoff cover
their assigned scopes; this handoff requests no additional A9/PH feature step.
Simulation, MNIST and Bayesian agents retain their full demo evidence
(spec 12.2 items 1,2,4 have separate owners). Release-wide audit, any required
Linux CI and publication remain separate. No fresh R regeneration, CPU
benchmark or accelerator/demo run is claimed here. Generic summaries, full
Weibull shape Hessians/quadrature tangents and later slices remain future work.
Both workstreams' plan history is preserved; this agent performs no merge,
rebase, push or publication.
