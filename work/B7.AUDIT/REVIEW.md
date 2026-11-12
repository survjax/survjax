# B7.AUDIT — Slice B completion evidence

Independent review: round 1/3

## Scope and reading order

1. PLAN.md, spec rev 62 sections 12.2 (Slice B), 12.1, 3.2 and 7.2.
2. The gate map below, then the named test bodies and current evaluator.
3. [B3.3](../B3.3/REVIEW.md) and its gap closures below.
4. Benchmark provenance, deviations and integration checklist at the end.

This audit concerns left/interval censoring, all truncation combinations,
interval/conditional hooks, static pruning and traced argument-form metadata.
It does not certify all of Slice D, v0, a release, or accelerator performance.
The only source edit corrects likelihood.py's obsolete comment that traced
flags are all true. Runtime code, tests, features and tolerances are unchanged.
`git diff 200c1f7 HEAD -- src tests scripts/spec-features.txt` was empty at
step start: B4.3's runtime/test snapshot remains the basis of this audit.

## Completion gate map (spec 12.2)

Paths abbreviated `spec/` below mean tests/spec/, not the design symlink.

| Slice B item | Current evidence |
| --- | --- |
| Left/interval censoring and every truncation mode | likelihood.py intersects real sets with windows, uses the conditional hook for censored rows and log density minus log Z for exact rows. The matrix below maps independent values; unit/test_likelihood_intervals.py and test_likelihood_truncation.py add safe-hook and transform checks. |
| log_interval_prob and log_conditional_interval_prob | distributions/base.py exposes generic defaults; Exponential and Weibull supply specialized methods. spec/test_observation_censoring_interval_prob.py and test_observation_truncation_hook.py check definitions, endpoints, cure mass, broadcasting, override dispatch and gradients; unit/test_interval_defaults.py checks rounded-equal finite logs as precision loss rather than zero mass. |
| Full likelihood matrix and diagnostics | Matrix, diagnostic and numerical evidence below. Stored/current issues and parameter masks survive pruning; no unchecked constructor is added. |
| Cross-kind properties | spec/test_observation_censoring_properties.py: interval-to-right, zero-to-left, adjacent union, partition and whole coverage. test_observation_truncation_properties.py: conditioning, partition, entry/full-window zeros and memorylessness. Reproducible Hypothesis, 40 cases per identity, primarily Weibull plus Exponential memorylessness. |
| Comparable R cells, excluded cells by mathematics | Committed Slice B fixture: 50 rows, ten cells, six vectors/three families, 300 values. spec/test_observation_censoring_flexsurv_fixture.py checks cells, constraints and mpmath; reference/test_slice_b_r.py checks all values eager/jit/vmap plus hashes/provenance/CSV alignment and identical acceptance copy. R comparisons are float64-only. Excluded cells have the independent matrix/diagnostic evidence below. |
| Transformed parameter derivatives | spec/test_observation_{censoring,truncation}_derivatives.py covers Exponential/Weibull and generic-family cases, analytical/mpmath Jacobians/Hessians, check_grads, jit/vmap, inactive-reference and invalid-entry isolation. B3.4/B3.5 add the brought-forward families and wrappers described below. |
| Static-metadata pruning | unit/test_likelihood_pruning.py forbids unused hooks for exact/censored/empty data; test_likelihood_untruncated_pruning.py forbids both interval hooks for untruncated exact/right data and independently checks each disabling flag. Both compare conservative/prepared routes, derivatives, issues and inherited slice behavior. |
| Traced argument-form metadata | spec/test_observation_traced_metadata.py's 51 tests cover both constructors, None versus supplied defaults, invalid inputs, nested tracers, vmap, slicing, compilation counts and raw likelihood derivatives. unit/test_data_metadata.py additionally covers concrete normalization, including invalid rows and concrete inputs inside jit. |

### Four kinds by four windows

This matrix is for Exponential and Weibull (including distinct Weibull shapes).
`C` = spec/test_observation_censoring_values.py;
`T` = spec/test_observation_truncation_values.py.

| Kind | No truncation | Left only | Right only | Two-sided |
| --- | --- | --- | --- | --- |
| Exact | C: test_exponential_mixed_batch_with_exact_and_right_rows; Weibull counterpart | T: test_truncated_exact_and_right_rows_are_hand_derived | same test, right window | same test, two-sided window |
| Right | same mixed-batch tests | same truncated exact/right test | same test, right window | same test, two-sided window |
| Left | C: test_exponential_left_censored_row_is_hand_derived; Weibull counterpart | T: test_truncated_left_and_interval_rows_are_hand_derived | same test, right window | same test, two-sided window |
| Interval | C: test_exponential_interval_censored_row_is_hand_derived; Weibull counterpart | same truncated left/interval test | same test, right window | same test, two-sided window |

Both files check their hand values against mpmath. T also exercises all kinds
in one batch, row-varying windows, partial overlap, generic defaults, improper
families and PH/Weibull equivalence. No fixed R convention substitutes for
the mathematical matrix.

LogNormal/LogLogistic have untruncated coverage in their family likelihood
suites (activated in B5.FAMILIES). Their excluded R cells are checked against
mpmath by spec/test_observation_truncation_lognormal_loglogistic.py: left and
interval with entry, finite-U clipping, covering sets, entry boundaries and
invalid observations. unit/test_likelihood_truncated_families.py supplies
all-kind values/Jacobians/Hessians under each of three truncated windows at
two vectors, using 80-digit independent survival/density algebra and mp.diff.

PH and AFT have a 21-row matrix over Exponential, Weibull, LogNormal and
LogLogistic in unit/test_wrapper_censoring_derivatives.py: all 16 cells plus
overlaps and covering sets. Both Jacobian modes and all four Hessian
compositions include joint baseline/eta cross terms, eager and jit(vmap),
two parameter vectors, plus order-two finite differences. This is the ordinary
smooth parameter regime of spec 7.3, not a stronger PH extreme-scaling promise.

### Diagnostics and numerical behavior

| Requirement | Concrete evidence |
| --- | --- |
| Invalid intersections, before entry, exact outside window | spec/test_observation_truncation_diagnostics.py: test_invalid_truncated_exact_or_right_row_is_nan and test_invalid_truncated_censored_row_is_nan; invalid masks under jit/traced construction and LogNormal/LogLogistic excluded-cell tests. |
| Structurally uninformative sets | Same file: test_right_censored_at_entry_is_exactly_zero_and_uninformative and test_censored_set_covering_the_window_is_exactly_zero_and_uninformative. Unit family/wrapper tests require exact zero values and parameter derivatives. |
| Zero numerator vs zero denominator | Same file: test_bounded_support_separates_zero_numerator_from_zero_denominator and test_zero_denominator_is_nan_and_triaged_as_other. Generic interval defaults also check the mass taxonomy. |
| Tiny mass, tails, narrow intervals | spec/test_observation_censoring_numerics.py and test_observation_truncation_numerics.py; unit/test_exponential_{intervals,conditional}.py and test_weibull_{intervals,conditional}.py. Cases include underflowed cumulative hazard with representable log mass and narrow deep-tail windows. |
| Safe inactive computations | unit/test_likelihood_truncation.py::test_conditional_hook_receives_intersection_and_safe_exact_window and test_likelihood_intervals.py::test_interval_override_receives_safe_inputs_before_evaluation. Mixed debug_nans cases occur in both spec numerical suites, generic defaults and the brought-forward-family/wrapper tests. |
| Stored/current input issues, parameter isolation | unit/test_likelihood_pruning.py::test_pruning_preserves_invalid_data_parameters_and_shapes; test_likelihood_untruncated_pruning.py::test_fast_path_checks_invalid_current_fields_and_stored_issues; both spec derivative suites' test_rows_without_parameter_issues_keep_derivatives. Earlier data/parameter suites remain in A12. |
| Shapes, construction, indexing and compilation | Both spec transformation suites cover draw/dataset vmap, bounds/flag forms and indexing. test_new_truncation_values_reuse_compilation, test_traced_construction_with_truncation_reuses_compilation, the traced-metadata suite and pruning slice-count tests cover reuse with fixed static configuration. |

Runtime inspection retains stopped safe substitutes only on inactive inputs,
log-space probability calculations, dtype promotion and combined stored/current
data diagnostics. Traced metadata inspects only argument presence, never
observation values. CPU jit/vmap evidence does not certify accelerator placement
or every possible custom family; no new device-performance claim is made.

## Closure of the earlier audit and documentation

B3.3's five gaps are addressed: B3.4 supplies truncated-family derivatives;
B3.5 supplies wrapper derivatives; B4.1/B4.2 implement pruning and B4.3 adds
rev-62 metadata; B5.FAMILIES activates both brought-forward family gates;
B6.1–B6.4 document the likelihood, diagnostics, metadata and R conventions.
The default feature list now includes observation.censoring,
observation.truncation, observation.traced_metadata, family.lognormal and
family.loglogistic. Summary/sampling feature gates are outside this audit.

Documentation evidence: B6.4's final direct build succeeded in both dtypes
at 2026-11-12T18:06:58Z; its handoff records executed cells and HTML checks
of 13 nested headings, preserved anchors, links and four fixture downloads.
The docs source is unchanged here, so no repeat build is claimed. That review
ended merge in round 2 at d44367e; the reviewer's local-binding restriction
prevented its own docs build. The implementer's checks did not include browser
layout/MathJax visual inspection.

## Performance evidence stays in experiments

The B4.3 archive moved after the cleanup. Current project:
../../../experiments/projects/r-speed-comparison, inspected at commit
36a8d4f3c21b9ac4ed1271b8be79abc661ec5c61. The preserved
[import record](../../../experiments/projects/r-speed-comparison/history/pruning-experiment/runs/20261112-library-b43-import.md)
and adjacent JSON manifest describe library commit
200c1f7f4b64d147d5d40cbcba44f97e44d3c1f5. Current local artifact prefix is
artifacts/pruning-consolidation-20261112-001/legacy-checkout/ followed by each
manifest artifact_path. Earlier step links retain their historical location.

Read-only verification matched all five archived files to the manifest's
byte lengths/SHA-256 and to original library Git blobs. Both dtype outputs
contain 18 cases plus environment metadata, with compilation_reused true in
every case. The preserved handoff records agreement/diagnostics checks,
synchronized interleaved warm timing, first-call/preparation costs and byte
counts for concrete, all-true control and ordinary traced-construction routes.

This is historical CPU evidence, not a rerun or isolated construction cost.
Shared parameters only in the B4.3 extension; the original Slice A baseline
covers shared/per-row parameters at its older revision. There is no guaranteed
speedup, timing gate, peak-memory measurement or accelerator result. Do not pool
these records with the newer experiment's different pin/JAX/output contract.
The library's historical benchmark harness still asserts obsolete all-true
metadata; B6.2/B6.4 explicitly scope its commands to its historical checkout.
No experiment file, benchmark artifact or timing table is changed or added here.

## Deviations and maintainer integration

- D-B2.1-001: the tiny-tail reference defect was corrected by independent sync
  0cc6d18 and integrated before B3. The affected test is in the focused run.
- D-B4.3-001: independent sync 3d9704a (in merge b4b6700) made the compile-count
  inputs the same shape. The affected test is also in the focused run.

Both entries remain formally open in their original records. Their technical
causes have fixes; only the maintainer may disposition the entries. This audit
does not rewrite them or assert that their status changed.

Integration requires maintainer acceptance of this audit/review, formal
deviation disposition and the usual full non-fast A12 merge gate. No merge,
rebase, push, release or shared roadmap edit is performed. The roadmap handoff
is Slice B implementation/evidence ready for integration once the checks below
and independent review pass, not a statement that integration has happened.
Reviewer-workflow maintenance remains separately deferred.

## Validation and commits

Focused command, run once with JAX_ENABLE_X64=1 and then with =0 in separate
processes (no slow-marker filter):

```sh
SPEC_THROUGH=A12 \
SPEC_FEATURES=observation.censoring,observation.truncation,observation.traced_metadata,family.lognormal,family.loglogistic \
uv run --group test pytest -n 6 -q \
  tests/spec/test_observation_censoring_*.py \
  tests/spec/test_observation_truncation_*.py \
  tests/spec/test_observation_traced_metadata.py \
  tests/unit/test_interval_defaults.py \
  tests/unit/test_exponential_intervals.py tests/unit/test_weibull_intervals.py \
  tests/unit/test_exponential_conditional.py tests/unit/test_weibull_conditional.py \
  tests/unit/test_likelihood_intervals.py tests/unit/test_likelihood_truncation.py \
  tests/unit/test_likelihood_pruning.py tests/unit/test_likelihood_untruncated_pruning.py \
  tests/unit/test_likelihood_truncated_families.py \
  tests/unit/test_wrapper_censoring_derivatives.py \
  tests/unit/test_data_metadata.py tests/reference/test_slice_b_r.py
```

- Float64: 646 passed in 195.64s. Float32: 584 passed / 58 skipped in
  250.10s, with the suites' explicit dtype-specific skips. Both exited 0;
  the slow family/wrapper derivative suites and both formerly failing
  deviation tests ran. No assertion, marker or tolerance was changed.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Format:
  286 files already formatted; Ruff clean; mypy clean on 25 source files.
  Implementation float64: 1212 passed / 8 skipped (31.72s); float32:
  1214 passed / 6 skipped (28.63s). Source doctests: 4 passed per mode.
  Spec float64: 4654 passed / 46 skipped (72.71s); float32:
  4597 passed / 93 skipped (65.29s). Two existing NumPy warnings per mode
  at test_a11_ph_direct_methods.py:78.
- spec-since checked before and after inspection: unchanged since a5c1dea.
  likelihood.py AST is identical to step start; git diff --check clean.
- Five archive/Git/manifest byte-and-hash matches; the relocated import note
  and manifest themselves match experiment commit 36a8d4f. Both archived dtype
  outputs have compilation_reused true in all 18 cases. These are preservation
  checks, not new timing runs.
- No new docs build, R regeneration, performance run, mutation test or full
  non-fast cumulative gate. Existing B6.4 build evidence and the fresh focused
  slow-test run are identified separately above.

No current implementation or test failure, or uncovered Slice B gate category,
was identified in this audit. Recommend maintainer integration subject to the
independent review and integration checks listed above.

Step starting / previous-step final: d44367e33a855a84b53ece31861fdfd8c5cf9025.
Original worktree base: cc0e30f. No later implementation sub-step is identified
by this audit; maintainer integration and its checks remain.
