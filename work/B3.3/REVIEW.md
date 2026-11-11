# B3.3 — Slice B evidence audit

Independent review: round 1/3

Base / previous step: ccb885d33231809ae757d75f878cb5ec3181d1ee (B3.2).
Original worktree base: cc0e30f. Cumulative gate: A12 with both observation
features. This is an evidence audit; it changes no implementation or tests.

## Read first

1. [Plan](PLAN.md), spec 2, 7.1–7.3, 10.4 and 12.1–12.2.
2. [B3.2 evidence](../B3.2/REVIEW.md) and its independent round-3 merge verdict
   at ../../../review/slices/B3.2.md.
3. The value and diagnostic test files named below, then property and derivative
   tests. All tests/spec paths are read-only independent evidence.
4. src/survjax/likelihood.py and src/survjax/data.py for the B4 boundary.
5. The gaps and validation sections below; a passing run is not evidence for
   combinations absent from the selected tests.

## Matrix and reference evidence

The table maps all sixteen observation-kind/window combinations for the Slice B
Exponential/Weibull evaluator. Paths below are relative to tests/spec/.
Untruncated evidence is in test_observation_censoring_values.py; truncated
evidence is in test_observation_truncation_values.py.

| Kind | No truncation | Left truncation | Right truncation | Two-sided |
| --- | --- | --- | --- | --- |
| Exact | mixed-batch tests | test_truncated_exact_and_right_rows_are_hand_derived | same, right window | same, two-sided window |
| Right | mixed-batch tests | same exact/right test | same, right window | same, two-sided window |
| Left | hand-derived left-row tests | test_truncated_left_and_interval_rows_are_hand_derived | same, right window | same, two-sided window |
| Interval | hand-derived interval-row tests | same left/interval test | same, right window | same, two-sided window |

Truncated tests parameterize Exponential, Weibull shape 1.5 and Weibull shape
0.6. Their independent NumPy algebra is cross-checked against mpmath by
test_hand_values_agree_with_mpmath. test_mixed_kinds_and_mixed_windows_in_one_batch
adds row-varying kinds/windows. Implementer coverage includes
tests/unit/test_likelihood_intervals.py and test_likelihood_truncation.py;
these smaller hand-derived batches do not duplicate every matrix cell.

For LogNormal and LogLogistic, test_family_<family>_likelihood.py covers
untruncated exact/right/left/interval and truncated exact/right values.
test_observation_truncation_lognormal_loglogistic.py supplies mpmath evidence for
left/interval rows with left truncation, partial overlaps, covering sets, entry
boundaries and invalid observations. These are concrete independent values,
not a full derivative matrix for these families.

test_observation_censoring_flexsurv_fixture.py checks B3.2's ten comparable
cells, fifty shared observations and six vectors across Weibull, LogNormal and
LogLogistic. Its fixture constraints and values are checked independently with
mpmath. tests/reference/test_slice_b_r.py checks all 300 R log likelihoods
through eager/jit/vmap and the manifest/provenance/CSV alignment. R comparisons
are float64-only; host integrity checks remain meaningful in float32.

## Diagnostics, numerics and transformations

Paths in this section are under tests/spec/ unless otherwise stated.

| Requirement | Concrete evidence |
| --- | --- |
| Partial overlap at L/U; mixed windows (2) | test_observation_truncation_values.py: test_truncated_left_and_interval_rows_are_hand_derived; test_mixed_kinds_and_mixed_windows_in_one_batch |
| Right at entry and whole-window sets are exactly zero (2/7.1) | test_observation_truncation_diagnostics.py: test_right_censored_at_entry_is_exactly_zero_and_uninformative; test_censored_set_covering_the_window_is_exactly_zero_and_uninformative |
| Empty intersections, before entry, exact outside window (2/3.3/7.1) | same file: test_invalid_truncated_exact_or_right_row_is_nan; test_invalid_truncated_censored_row_is_nan; brought-forward families' truncation file adds their cases |
| Zero numerator versus zero denominator (7.1) | same file: test_bounded_support_separates_zero_numerator_from_zero_denominator; test_zero_denominator_is_nan_and_triaged_as_other |
| Tiny positive mass, deep tails, narrow windows (7.1/7.2) | test_observation_censoring_numerics.py and test_observation_truncation_numerics.py; implementer test_exponential_conditional.py / test_weibull_conditional.py |
| Valid mixed batches under debug_nans (7.2) | both numerical files; test_full_matrix_under_debug_nans includes truncated mixed kinds |
| Safe inactive inputs and hook dispatch (4.2/7.2) | tests/unit/test_likelihood_intervals.py: test_interval_override_receives_safe_inputs_before_evaluation; test_likelihood_truncation.py: test_conditional_hook_receives_intersection_and_safe_exact_window |
| Interval-to-right, zero-to-left, adjacent union, full coverage (12.1) | test_observation_censoring_properties.py, reproducible Hypothesis with 40 examples per identity |
| Conditional partition, memorylessness, whole-window identities (2/12.1) | test_observation_truncation_properties.py, reproducible Hypothesis with 40 examples per identity |
| Eager/jit/vmap, traced construction, indexing (3.2/7.3/12.1) | test_observation_censoring_transformations.py and test_observation_truncation_transformations.py |
| Compilation reuse and conservative metadata (3.2/7.2) | truncation transformations: test_new_truncation_values_reuse_compilation; test_traced_construction_with_truncation_reuses_compilation; test_untruncated_rows_agree_with_and_without_truncation_metadata |
| Exponential/Weibull first/second derivatives (7.3) | censoring/truncation derivative files: hand/multiprecision oracles, all four AD Hessian compositions, check_grads, jit/vmap and valid-row isolation |
| Generic/custom and improper family behavior (4.2/7) | censoring values/interval/derivative files; truncation values/diagnostics/hook files; tests/unit/test_interval_defaults.py |

Property-based identities primarily exercise Weibull (plus Exponential
memorylessness). Fixed LogNormal/LogLogistic additivity tests exist in their
likelihood files; they are not randomized family-wide property evidence.
The implementer property directory currently contains only concordance tests;
the Slice B property evidence above belongs to the independent suite.

## Gaps and next bounded work

1. B3 follow-up derivative evidence: both brought-forward family derivative
   files test untruncated left/interval gradients and Hessians, and truncated
   exact/right first derivatives. No test located for truncated left/interval
   parameter Hessians, including jit/vmap and forward/reverse compositions.
   Add an implementer test step with independent mathematical references before
   claiming full transformed-derivative coverage for these combinations.
2. Wrapper follow-up: truncated PH/Weibull values have equivalent-Weibull
   checks in test_observation_truncation_values.py. Existing implementer PH/AFT
   likelihood derivative suites exercise exact/right data. Extend independent
   value/derivative evidence to the new left/interval/truncated combinations,
   subject to spec 7.3's PH extreme-scaling limitation. Do not infer their
   derivative correctness from baseline value agreement.
3. B4 pruning and measurement remains: data stores _may_contain, but
   likelihood._evaluate always evaluates the conditional interval, density
   and normalization pieces. It does not consume these flags. Compilation
   reuse/equivalence tests are present; they do not establish branch pruning or
   a measured performance benefit. Keep current-field validation and stored
   issue propagation when implementing the optimization.
4. Family feature activation is separate: scripts/spec-features.txt contains
   only observation.censoring and observation.truncation. The focused audit
   enables family.lognormal and family.loglogistic explicitly; the ordinary
   fast gate does not certify all of either family's acceptance suite.
5. The likelihood/docs follow-up remains as linked in PLAN.md. No docs build
   is needed for these work records; no user-facing docs changed.

These are evidence and implementation tasks, not proposed spec changes. No new
spec ambiguity is identified. This audit does not declare Slice B complete or
change the shared roadmap. After review, the next implementation step should
address item 1; keep the wrapper work and B4 separate to stay reviewable.

## Validation

Focused acceptance command (run separately with JAX_ENABLE_X64=1 and 0):

```sh
SPEC_THROUGH=A12 \
SPEC_FEATURES=observation.censoring,observation.truncation,family.lognormal,family.loglogistic \
uv run --group test pytest -n 6 -q \
  tests/spec/test_observation_censoring_*.py \
  tests/spec/test_observation_truncation_*.py \
  tests/spec/test_family_lognormal_likelihood.py \
  tests/spec/test_family_lognormal_derivatives.py \
  tests/spec/test_family_loglogistic_likelihood.py \
  tests/spec/test_family_loglogistic_derivatives.py
```

- Float64: exit 0, 646 passed (63.74s).
- Float32: exit 0, 574 passed, 68 skipped (37.06s).
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Guard clean;
  Ruff formatting (255 files), lint and mypy (25 source files) clean.
  Implementation float64: 1211 passed, 8 skipped; float32: 1212 passed,
  7 skipped. Source doctests: 4 passed per mode. Spec float64: 3444 passed,
  32 skipped; float32: 3426 passed, 46 skipped. Each spec mode reports the
  two existing PH reference-arithmetic warnings in test_a11_ph_direct_methods.py.
- git diff --check: clean.

No R regeneration, timing-list refresh, full nonfast cumulative gate or new
numerical implementation in this step. The focused acceptance run includes slow
tests and overrides only feature selection, not assertions or tolerances.

## Review setup

The maintainer restored the reviewer checkout and requested review. Read-only
checks confirm it is clean, detached at ccb885d and registered in this repository.
No implementer repair or reviewer-configuration change was made. Investigation
of the recurring checkout problems is deferred, as requested by the maintainer.
