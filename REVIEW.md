Independent review: round 1/3

# A9.d1 — Using and verifying the installed Slice A subset

Authorized after A9.4 approval and the maintainer's completed main merge.
Branch a9b, clean base bd32ff9. One verification topic: the new page and
index navigation, plus a stale capability note discovered during the audit.
No source, tests, notebook, benchmark, fixture, dependency or protected-path
changes. Earlier A9.4/integration reviews remain in Git at bd32ff9:REVIEW.md.

## Reading order

1. docs/slice-a-verification.md: supported subset, three exact analytical
   smoke cells, installed-wheel runner, CPU reproduction and evidence limits.
2. docs/index.md's prose/toctree entry and docs/survival-data.md's corrected
   opening note. The old note incorrectly said distributions/likelihoods/
   evaluation were unimplemented; the replacement links current guides.
3. PLAN.md A9.d1: authorization, validation ownership and audit conclusion.
4. The gate map below, then existing tests/benchmark/fixture files only as
   evidence references. These executable inputs are unchanged by this step.

## Validation ownership

The maintainer requested one runner per step, not duplicate implementer and
reviewer runs. **The implementer owns the docs build, lint and browser audit
for A9.d1. Review should inspect the source and recorded evidence rather than
rerun the same build or full gate.** The maintainer also reported rerunning
`scripts/check_wheel.sh --through A9` without errors after completing the
main merge, and manually cleaning the A9.4 throwaway worktree. Those are
maintainer-provided outcomes, not commands rerun or counts inferred here.
There are no later runtime/packaging changes requiring another wheel gate.

## Invariants and specification mapping

- Spec 6.1/6.2/7.2/7.3: ordinary construction and both diagnostics are inside
  jit; log likelihood is summed by the caller. For two events and total
  follow-up 3, l(r)=2 log(r)-3r, l'(r)=2/r-3, l''(r)=-2/r**2. At r=1 the
  eager/compiled values are exactly -3/-1/-2. Prepared/raw contributions
  are exactly [-0.5,-1,-1.5]. Doubling follow-up gives mapped values [-3,-6],
  gradients [-1,-4], Hessians [-2,-2]. Expected values come from this hand
  algebra, not another evaluation of the library. Exact binary cases need
  no new tolerance. Explicit host checkpoints are kept outside jit.
- Spec 7.3: parameter derivatives are promised only through second order
  for finite valid smooth interior cases; arbitrary unsafe custom code,
  time derivatives and invalid/infinite/support-boundary cases excluded.
- Spec 10.3/12.1: explain why check_wheel supplies SURVJAX_WHEEL and plain
  check.sh alone does not. Installed core/custom-family typing checks,
  offline cache requirements and separate source-archive release check are
  accurately distinguished. No publishing or dependency change.
- Spec 10.4/12.1: R claims are only committed synthetic fixed-parameter
  Exponential/bc-like Weibull references, with the pinned versions. No real
  patient-data, fitted-model or unsupported-mode agreement claim.
- Spec 12.3: benchmark descriptions match the accepted harness and recorded
  artifact. Keep prepared/traced flags, returned diagnostics, warmups,
  synchronization, compilation reuse, logical bytes and unmeasured peak
  memory explicit. Explain optional A9.4 N-1/N-2 caveats without changing
  accepted timing data or pretending the benchmark's pre-harness revision
  contains its script. No universal speedup or accelerator claim.
- Spec 10.5/12.2: add executable verification/navigation, audit all existing
  content pages and editable playground, retain the represented/evaluated
  observation distinction and failure behavior. The new data-guide note
  corrects an objectively stale capability statement within this topic.

## Checks actually run

Final command: `scripts/build_docs.sh`, permitted outside the sandbox.
Exited **0**; both builds report **build succeeded**, with no Sphinx
warnings, execution failures or process-inspection errors. Existing
IPKernelApp TCP-transport warnings remain in the kernel logs; these are not
Sphinx warnings. No configuration/timeout/selection changes were made.
Final log: /private/tmp/survjax-a9d1-docs-final4.log.

| Executed document | float32 seconds | float64 seconds |
| --- | --- | --- |
| builtin-families | 2.81 | 2.56 |
| concordance | 3.05 | 3.11 |
| distributions | 4.77 | 4.75 |
| grids | 3.49 | 3.50 |
| likelihood | 3.79 | 3.74 |
| numerical-behaviour | 3.34 | 3.30 |
| playground | 4.44 | 4.49 |
| slice-a-verification | 3.07 | 3.09 |
| survival-data | 2.76 | 2.76 |

All nine runnable documents execute afresh: **137 code cells per mode**,
including three new assertion cells and 16 playground cells. The final
executed notebooks were inspected for execution counts and unexpected
error outputs; intentional raises-exception cells remain instructional
examples. The committed playground has no outputs/counts and is unchanged.
The HTML/parser audit checked **14 generated HTML files, 1,048 local
file/fragment links per mode**, with no missing targets. External links were
not fetched. The simulation guide remains deliberately excluded because
it describes future likelihood workflows; it is not counted as executed.

`uv run --group docs ruff format --check .`: **115 files already formatted**.
`uv run --group docs ruff check .`: **All checks passed!** The new MyST code
cells were also extracted to a temporary notebook and checked with the same
project configuration: **All checks passed!**, **1 file already formatted**.
`uv run --locked --group docs jupyter lab --version`: **4.6.4**. The documented
launch command is `uv run --locked --group docs jupyter lab docs/playground.ipynb`.
No interactive JupyterLab server/browser session was launched; notebook
execution in both build kernels is the reproducible playground check.
`git diff --check` passes. Final scope/staging audit follows before commit.
No mypy, pytest, R generator, benchmark, wheel, sdist or accelerator rerun
in this docs-only step; see the maintainer's current wheel result above.

Initial attempts/corrections: sandboxed docs build logged sysctl process-
inspection permission errors during kernel shutdown; interrupted that run
to rerun with permission (exit 2, DeadKernelError/cleanup AssertionError).
Subsequent builds passed. A visual pass prompted wrapping the benchmark
commands; the site audit found the stale data-guide note; a code-cell
format check prompted collapsing one mapped call. Rebuilt after each
source change; the table reports the final source only. An auxiliary .py
extraction incorrectly treated notebook display expressions as B018;
using actual code-cell notebook semantics resolved that tooling artifact.
A temporary notebook generator also added final newlines Ruff removes;
strip those from the temporary cells. Final extracted-cell checks pass.
No assertion, tolerance, behavior, selection or required check was weakened.

## Rendered-page audit

Native Brave through CUA inspected local html-x64 pages: index, survival-data,
builtin-families, likelihood, numerical-behaviour, concordance, grids,
distributions, api, playground and the new verification page. All 11 pages'
accessible content was also inspected in html-x32; screenshots covered the
new page, corrected data note and playground in both modes. Detailed views
covered the new derivative equation, exact outputs, compiled/map examples,
wrapped benchmark commands, prepared/traced table, peak-memory caveat,
index/toctree navigation, concordance pair table, API signatures, custom-
family formula table and grid survival/hazard figure. No desktop layout
issue observed after wrapping the new long shell commands. Existing long
code blocks use horizontal scrolling. This is desktop inspection at the
current browser width, not a mobile/responsive or external-link audit.

Observed values agree with the hand calculations in both modes: new page
-3/-1/-2, then mapped [-3,-6]/[-1,-4]/[-2,-2]; existing likelihood fit shows
6/103 and Weibull about (1.2904,16.006); tails distinguish zero survival from
finite log contribution near -10000; concordance C=0.7 and the five pairs;
playground prints the correct dtype mode and retains supported/unsupported/
zero-pair examples. The data note now links the available likelihood and
concordance guides while keeping interval/left/truncated likelihoods future.
No broad rewrite or new notebook topic was needed.

## Slice A completion-gate evidence map (spec 12.2)

Numerical/install rows below are **covered by existing tests and the
maintainer-reported post-merge through-A9 wheel pass**, not newly executed
by the implementer in this docs step. Docs rows have fresh results above.
The gate is restricted to Slice A's supported modes and methods; Slice B
interval/truncation mathematics and later-slice requirements are excluded.

| Gate item | Concrete evidence | Status/owner |
| --- | --- | --- |
| Construction, both forms, dtype and traced arguments | tests/unit/test_data_construction.py, test_data_validation.py, test_data_dtype.py, test_data_tracing.py | Gate pass reported by maintainer |
| Data diagnostics, indexing, size and metadata | test_data_issues.py, test_data_indexing.py, test_data_size.py, test_data_metadata.py (including test_static_flags_control_compilation_reuse) | Gate pass reported by maintainer |
| Parameter diagnostics and stored raw issues | test_constraints.py, test_parameter_issues.py, test_parameter_issue_container.py, test_likelihood_parameters_depth.py; test_likelihood_core.py::test_stored_input_issues_survive_indexing_and_reconstruction; test_observation_inputs.py | Gate pass reported by maintainer |
| Likelihood evaluation, both public input forms and shapes | test_likelihood_core.py, test_likelihood_entry_points.py; direct/grid shapes in test_family_batching.py and test_survival_grid.py | Gate pass reported by maintainer |
| First/second derivatives and compiled composition | test_likelihood_transformations.py, test_likelihood_composition.py, test_likelihood_custom_derivatives.py; test_survival_grid.py::test_grid_analytic_rate_gradient_hessian_and_order_two | Gate pass reported by maintainer; exact new docs assertions pass both modes |
| Stable tails and independent math | test_likelihood_tails.py, test_log_cdf_tails.py (mpmath values and differentiated identities), test_likelihood_boundaries.py | Gate pass reported by maintainer |
| Boundaries and inactive-input safety | test_family_boundaries.py, test_family_infinity.py, test_family_mixed_infinity.py; test_survival_grid.py::test_weibull_shape_one_zero_grid_and_nan_isolation | Gate pass reported by maintainer |
| Committed synthetic bc-like per-row R references | tests/fixtures/bc_like/; tests/reference/test_weibull_bc_like_r.py::test_likelihood_rows_match_r and test_time_first_survival_grid_matches_r; tests/reference/test_exponential_r.py | Gate pass reported by maintainer; no R regeneration |
| Eager jax_debug_nans mixed batches | tests/unit/test_likelihood_debug_nans.py::test_exponential_mixed_values_and_gradients_under_debug_nans and test_weibull_mixed_values_and_gradients_under_debug_nans | Gate pass reported by maintainer |
| Installed wheel, exports, type marker and external family | tests/install/test_wheel.py (three tests), tests/install/custom_family.py; independent A9 wheel acceptance tests | Fresh-wheel gate pass reported by maintainer |
| Tiny synthetic example | Three-observation exact checked recipe in docs/slice-a-verification.md; existing synthetic likelihood fit in docs/likelihood.md | Fresh docs smoke passes both modes; larger README/use-case teaching replacement remains separate |
| Recorded CPU baseline | scripts/benchmark_slice_a.py, benchmarks/slice_a_cpu.json; A9.4 review round 1 merge on 51319c8; 48 cases with analytical checks/reuse | Accepted by maintainer; no rerun, optional N-1/N-2 remain |
| Executable docs and editable playground | All nine runnable documents/137 cells per mode, clean source notebook, CLI 4.6.4, build command above | Fresh two-mode builds pass; interactive server not opened |
| Rendered pages and accurate subset/failure descriptions | 11-page audit above, both builds, corrected data note and current diagnostic/tail/unsupported guides | Implementer desktop audit complete; no mobile/external-link claim |

## Closure and remaining work

This supplies A9's verification page and completion-evidence audit, pending
this independent review and maintainer acceptance. The tiny recipe is a
verification smoke example; it does not restore the withdrawn standalone
A9.3 demo or complete the agreed README/use-case teaching replacement.
That teaching material and the concordance-specific held-out example are
still separate work. A10, the full Slice A milestone and first release are
not claimed complete here; their remaining steps/gates belong to their
respective plans. The workspace maintainer owns any ROADMAP.md update.

The maintainer reports the previous throwaway worktree cleaned up. Do not
create or clean reviewer-owned worktrees from the implementer. Stop after
this step's review; do not begin another step. No merge or publication.
