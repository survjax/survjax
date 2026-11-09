# Spec-test follow-up: tiny Weibull left-tail reference

Maintainer request: leave this instruction for the independent spec-test author
to pick up when available. B2.1 is blocked by D-B2.1-001 (DEVIATIONS.md here).

Please fix the reference used by:
`tests/spec/test_observation_censoring_numerics.py::test_weibull_left_row_where_the_cumulative_hazard_underflows`.

The test correctly requires a finite log CDF for Weibull shape=2.5, scale=2,
left censoring at 1e-30. However, `observation_reference._log_mass` computes
`log(S(a)-S(b))` at 50 decimal digits. Here H(b) is about 1.77e-76, so
`1-exp(-H(b))` rounds to zero, yielding an incorrect expected value of -inf.
The implementation returns approximately -174.4267499259533.

Independent reproduction (no survjax calls):

```python
import mpmath as mp
with mp.workdps(50):
    h = (mp.mpf('1e-30') / 2) ** mp.mpf('2.5')
    print(mp.log(1 - mp.exp(-h)))    # -inf: reference cancellation
    print(mp.log(-mp.expm1(-h)))     # -174.426749925953289574892439405...
```

Please use a stable reference expression, or sufficient precision for this
case. Preserve rounding of inputs to the active dtype. Keep the finite-value
assertion, existing tolerances and feature staging; do not weaken the test or
change the library to return -inf. Consider an oracle-only regression check
that the reference remains finite in this regime. You own the exact test fix.

Validate the affected case and the observation.censoring suite in separate
float64 and float32 processes with SPEC_FEATURES=observation.censoring. Leave
observation.truncation disabled for B2.2. Commit through the normal independent
spec-test workflow and give the maintainer the commit for integration into
li-censoring; do not edit the implementation worktree.

Implementer status: main 4239d61 merged as c9fee45. B2.1 code/flag/tests are
uncommitted pending this correction. Fast A12 gate: implementation suites and
doctests pass both dtypes; x64 acceptance has 3520 passed, 32 skipped and this
one failure. The gate stops before x32 acceptance. See REVIEW.md here for
additional checks. After integration, rerun the B2.1 gate, commit the completed
step, then request independent review. No B2.2 work has started.
