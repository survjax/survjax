# B5.FAMILIES — Built-in family acceptance activation

Independent review: round 1/3

## Read first

1. [PLAN.md](PLAN.md), scope and dependencies.
2. scripts/spec-features.txt, the only implementation change (two added flags).
3. spec 12.2, "Families brought forward after the first release", and the
   feature table in ../roles/TESTER.md.
4. tests/spec/conftest.py feature selection and the existing
   test_family_lognormal_* / test_family_loglogistic_* acceptance suites.
5. Linked D.*.2 documentation handoffs and B3.3's feature-activation gap.

## Invariants and coverage

The default A12 gate now includes family.lognormal and family.loglogistic
alongside observation.censoring and observation.truncation. scripts/check.sh
reads this list itself. Its gate target remains A12, not the step ID.
Tests needing multiple feature flags require all of them. Neither summary
nor sampling flags are enabled, consistent with spec 12.2's staged family
and analytical-summary delivery. This is gate activation for existing code,
not a claim that all of Slice D or Slice B is complete.

Existing acceptance files demonstrate these requirements in both dtypes:

- *_construction.py and *_constraints.py: natural parameter construction,
  from_log, pytree/broadcast behavior and parameter diagnostics (4.1, 4.2, 7.1).
- *_closed_forms.py and *_boundaries.py: independent analytical values,
  stable tails and direct-method boundary policy (4.1, 4.5, 7, 12.1).
- *_grids.py: time-first grids and transformations (5).
- *_likelihood.py and *_derivatives.py: supported observation modes,
  diagnostics, transformed values and parameter derivatives (4.2, 7, 12.1).
- test_observation_censoring_flexsurv_fixture.py and
  test_observation_truncation_lognormal_loglogistic.py: committed R fixture
  comparisons and independent truncated-family mathematics (10.4, 12.1–12.2).

No source, tests, tolerances, docs pages, dependencies or protected files
changed. No unresolved question or deviation was identified.

## Validation

Full focused acceptance command (no slow-test filter):

```bash
JAX_ENABLE_X64=1 SPEC_THROUGH=A12 SPEC_FEATURES=observation.censoring,observation.truncation,family.lognormal,family.loglogistic uv run --group test pytest -n 6 -q tests/spec/test_family_lognormal_*.py tests/spec/test_family_loglogistic_*.py tests/spec/test_observation_censoring_flexsurv_fixture.py tests/spec/test_observation_truncation_lognormal_loglogistic.py
```

- Float64: 999 passed, 14 skipped in 109.88s, exit 0.
- Same command with JAX_ENABLE_X64=0: 953 passed, 54 skipped in 77.64s,
  exit 0. Summary tests in the glob remain feature-deselected.
- scripts/check.sh --fast --through A12: exit 0, "all gates passed through
  A12 (fast: slow tests skipped)". Formatting: 271 files already formatted;
  Ruff: all checks passed; mypy: no issues in 25 source files.
- Fast implementation tests: float64 1206 passed / 8 skipped (33.64s);
  float32 1208 passed / 6 skipped (18.61s).
- Source doctests: 4 passed in each dtype.
- Expanded fast acceptance tests: float64 4607 passed / 46 skipped (36.30s);
  float32 4550 passed / 93 skipped (33.97s). Each emitted the same two
  pre-existing NumPy warnings from test_a11_ph_direct_methods.py:78
  (log of zero and invalid multiply in the expected-value expression).
- git diff --check: clean.
- No full non-fast cumulative gate, docs build or fixture regeneration was
  run. Existing family docs were completed by the linked D.*.2 steps; the
  selected family acceptance suites above ran in full, including slow tests.

## Commits and remaining work

Base / previous-step final commit:
f568b9b61e5af45f43be13a67fdf80ee0e674468 (B4.2).
Original worktree base: cc0e30f. This step closes B3.3's family-activation gap;
no shared roadmap/progress file is edited.

Likelihood documentation, final Slice B evidence audit and maintainer
integration remain. The docs must carry forward B4.2's trusted-metadata
contract and pruning performance notes. Reviewer-workflow investigation
remains deferred. The full cumulative gate before integration belongs to the
maintainer's merge workflow.
