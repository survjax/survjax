# PH.8 — Final PH evidence and acceptance audit

Independent review: round 1/3

Step start /previous PH.7 final: eeeb47a74e734acbfd40e240496b02c04c974c53.
Original launch: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Branch release-ph; spec alignment 765c65bba9efb3af8ef4a13c1ecb8e302a47f346,
revision 40. Current AGENTS.md/CLAUDE.md and PH packet read before work.

## Read first

1. PLAN.md PH.8; plans/release-agents/ph.md and its new evidence link.
2. plans/release-agents/ph-evidence.md: durable requirement-to-test map,
   current commands/results, artifacts, accepted limits and remaining owners.
3. Spec 4.1–4.7,5–7,8.7,10.2–10.5,12.1–12.2 and the mapped PH tests.
4. tests/spec/TESTMAP.md "Revision 37 ... not yet tested" and
   API_ASSUMPTIONS.md Q-049: independent PH/RMST acceptance has not arrived.

## Scope, invariants and audit conclusion

Evidence/handoff-only step, no library/API/test/docs-page/README, dependencies,
tolerances, protected files, spec/deviation statuses or appendices changed.
Existing PH implementation and its five-cell user guide stay unchanged.
The new evidence file plus two handoff-link lines are below 300 lines.
All unfinished/merged PLAN history is preserved verbatim below the new step.

The audit identifies no additional PH implementation or coverage step. The
seven PH modules have 41 test functions before parametrization; every one is
included by the unchanged implementation runner. Ordinary baseline/eta/mixed
Hessians, all first-derivative identities, transformations, tail values,
boundaries and valid-entry isolation remain required/tested; passing extreme
eta-only tests is not presented as unrestricted baseline derivative evidence.
The revision-40 exception does not relax RMST or value requirements. See the
specific named tests and independent oracle origins in the evidence file.

This closes the PH implementer's evidence handoff, not the whole public
release. Independent PH/RMST acceptance must still be authored/synced by the
maintainer; no test/selection/export assertion was weakened to hide that gap.
RMST narrative and README's stale "RMST planned" sentence are handed to the
RMST docs owner as requested. They do not justify another PH feature step.
The maintainer coordinates integrated demos and release-wide final evidence.

User ph2.md was untracked at the start. A later read found it absent; no
command in this audit edited, staged or removed it. Only explicitly named
owned files will be committed. No merge/rebase/push/publish.

## Fresh gate results

`scripts/check_wheel.sh --through A10` run unchanged and offline via its own
settings, building a fresh retained wheel:

| Check | float64 | float32 |
| --- | --- | --- |
| Implementer suite, including PH, RMST, reference and isolated install | 1282 passed /8 skipped, 137.26s | 1287 passed /3 skipped, 131.65s |
| Source doctests | 4 passed, 1.07s | 4 passed, 0.96s |
| Independent acceptance through A10 | 2332 passed /28 skipped, 70.03s | 2322 passed /38 skipped, 65.12s |

Guard: ok (release-ph). Ruff format: 136 files already formatted; Ruff lint:
All checks passed! Mypy: no issues in 22 source files. Final exit 0: check: all gates passed through A10.

All existing skips are emitted by unchanged tests; no new skip/xfail added.
Available acceptance through A10 is regression evidence, not independent
acceptance of the brought-forward PH/RMST items. Independent authors explicitly
record that missing scope in their own TESTMAP/Q-049; these files are read-only.

## Slice A completion-gate evidence map (spec 12.2)

This maps existing core coverage run in the current gate; it is not a claim
that the remaining public-release items/demos are complete.

| Gate item | Existing evidence run or artifact inspected |
| --- | --- |
| Construction and dtype policy | tests/unit/test_data_construction.py, test_data_validation.py, test_data_boundaries.py, test_data_dtype.py, test_data_tracing.py; independent A1–A3 suites |
| Data diagnostics, metadata and stored input issues | test_data_issues.py, test_data_metadata.py, test_data_indexing.py, test_data_size.py; test_likelihood_entry_points.py::test_invalid_raw_indicators_keep_evidence; independent A2/A3/A6 |
| Parameter constraints/diagnostics and nested isolation | test_constraints.py, test_parameter_constraints.py, test_parameter_issues.py, test_parameter_issue_container.py, test_likelihood_parameters_depth.py; PH isolation map in evidence file |
| Likelihood evaluation and supported/unsupported modes | test_likelihood_core.py, test_likelihood_entry_points.py, test_likelihood_boundaries.py, test_likelihood_composition.py; independent A6/A7 |
| Shapes and time-first grids | test_family_batching.py, test_survival_grid.py, PH identity tests; independent A4/A5/A9 |
| First/second parameter derivatives and compiled composition | test_likelihood_transformations.py, test_likelihood_custom_derivatives.py, test_likelihood_composition.py; PH hand-derived Hessians/check_grads and supported RMST Hessians |
| Stable tails and boundaries | test_likelihood_tails.py, test_log_cdf_tails.py, test_family_boundaries.py, test_family_infinity.py; PH tails/guards/improper-limit tests |
| Independent mathematical/committed synthetic R fixtures | tests/reference/test_exponential_r.py, test_weibull_bc_like_r.py; tests/unit/test_reference_fixtures.py and test_bc_reference_fixtures.py; independent A8; PH/RMST mpmath physical-law oracles |
| Eager jax_debug_nans on valid mixed batches | test_likelihood_debug_nans.py; PH test_likelihood_and_second_order_interior_derivatives and RMST zero-boundary checks |
| Installed wheel, exact exports, type marker, runtime-only core and typed external family | tests/install/test_wheel.py, tests/unit/test_import.py, test_api_reference.py; independent A9 wheel smoke; archive route below |
| Tiny synthetic example | README minimal example freshly executed in both modes against analytical loss/gradient/survival, values recorded in evidence file |
| Recorded CPU baseline | benchmarks/slice_a_cpu.json inspected; both dtype runs at earlier revision 0a200085b0ab7a61f154da3ebc6cdcceaaf893d9, not rerun and not claimed as a new PH benchmark |
| Executable docs and playground, subset/limitations/NaN triage | direct strict build below freshly executes all 12 pages including playground and PH guide in both modes; existing numerical/likelihood pages cover subset/triage |

## Packaging, docs and supplemental checks

Full commands/artifact paths and SHA256s are in ph-evidence.md. Fresh source
archive contains 23 package files byte-identical to current source. Fresh
wheel built from its safely unpacked source root has all 31 members identical
to the gate's direct wheel; both full hashes are
e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d.
Archive-built wheel isolated install tests: 3 passed in 26.77s in EACH mode,
including strict out-of-tree mypy and PH eager/jit RMST. Runtime only JAX/NumPy;
no optional demo/test/docs libraries in those installed environments.

Honest limit: initial direct ARCHIVE-to-wheel offline command exited 2 because
uv_build is uncached. Rebuilding from the unpacked archive root succeeded via
uv 0.11.19's built-in backend without downloads or dependency changes. An
exploratory importlib.metadata query likewise found no installed uv_build
package; environment reporting was corrected to distinguish the uv backend.
Neither the isolated offline PEP-517 command nor an installed uv_build version
is claimed verified. Source/wheel parity and installation ARE verified.

`scripts/build_docs.sh` run DIRECTLY: exit 0, strict fresh builds both modes,
12 pages, PH execution 4.77s x32 /4.69s x64. Parsed both PH main articles:
five output containers, correct dtype, grid shape (3,3,2), compiled
loss/gradient/curvature (3,2,3), and baseline.scale diagnostics. Home prose /
toctree /sidebar PH links, API link and all six selected PH method anchors
(including rmst) retained. Correct previous/next guide order. No RMST narrative
was reintroduced. Plot visually inspected: labeled month/probability axes,
legend, baseline curve above PH hazard-ratio-2 curve; bytes identical across
modes, SHA256
90a1e0a54cebfb8edac31f4086b1df2980160d95789160e8d40efbbeb4c6a12e.
No full browser-page visual inspection this step; PH guide body/layout is
unchanged since PH.7. Source playground has no outputs/counts. No docs source modified.

README example uses existing impl_tolerances and hand-derived expectations:
L=-2*log(.2)+.2*9.5, gradient=.2*9.5-2, S=exp(-.2*t). Both dtype runs pass.
PH mathematical source/PDF exist (7964/40772 bytes), PDF signature present;
unchanged source means no new PDF compile or PDF-content review needed.

Environment: Python 3.12.9, macOS 26.6.2 arm64, cpu:0, JAX/jaxlib 0.11.2,
NumPy 2.5.3, pytest 9.1.1, mpmath 1.4.1, uv 0.11.19. No accelerator/matrix
claim. No R regeneration: committed fixtures only. Generated artifacts/cache
remain uncommitted.

Final checks after writing evidence: git diff --check exits 0; git diff
--exit-code -- src tests docs README.md pyproject.toml uv.lock .spec-seen
DEVIATIONS.md scripts appendices AGENTS.md CLAUDE.md .agents .workmux.yaml
exits 0. Final Ruff format rerun: 137 files already formatted; Ruff lint all
passed; mypy no issues in 22 source files. New evidence file is 126 lines plus
two handoff-link lines, total 128 implementation-documentation lines.

## Remaining work /uncertainty

No unresolved PH code question found. Accepted D-028/D-030/D-031/D-032 limits
are retained; formal statuses remain maintainer-owned. Final PH/RMST acceptance
sync, RMST docs/README integration and release-wide demo/evidence work are
explicitly open. No A9 implementer needs reopening. Stop after independent
review, with no additional PH implementation step proposed.
