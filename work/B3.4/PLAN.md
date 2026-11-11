# B3.4 — Truncated LogNormal/LogLogistic derivatives

Feature: observation.truncation with brought-forward families.
Base / previous step: a28318fe8094c22fe7cdb68e547b7d969c5aa3bc
([B3.3 audit](../B3.3/REVIEW.md), independent verdict: merge).
Original worktree base: cc0e30f. Dependencies: [B2.2](../B2.2/PLAN.md),
[B3.2](../B3.2/PLAN.md), merged LogNormal/LogLogistic implementations.
Cumulative gate: A12 with existing observation feature flags unchanged.

Address B3.3 gap 1 in a test-only step. Create
tests/unit/test_likelihood_truncated_families.py. Independently derive per-row
values, parameter Jacobians and Hessians with 80-digit mpmath from the section-2
conditional-density/probability formulas. Exercise both families over left,
right and two-sided windows, all four observation kinds, and partial overlaps.
Compare eager/jit/vmap values and all four forward/reverse Hessian compositions;
use two distinct parameter vectors for mapped comparisons. Include eager
debug_nans on valid mixed data, order-two check_grads on interior cases, and
exact-zero whole-window values/parameter derivatives.

Reuse ordinary tolerances for multiprecision comparisons and existing finite-
difference tolerances/step sizes for check_grads; justify their application in
tests/impl_tolerances.py without changing the values. No public API, src/,
dependency, docs/, protected-path or feature-selection changes. Test code and
tolerance explanation target less than 250 changed lines. If measurements mark
new tests slow, run scripts/update_slow_tests.sh; keep a mechanical timing
refresh separate from the tests, following the B3.2 commit arrangement.

Spec: 2, 3.3, 7.1–7.3, 12.1–12.2. Run the new file in full with six workers in
each dtype mode, then the fast A12 gate. Record commands/results and invariants
in REVIEW.md, commit, run independent review (at most three rounds), then stop.
If any comparison fails, investigate without loosening a tolerance or changing
protected tests; a library fix needing wider scope is a separate decision.

## Docs to-do

Later likelihood docs may cite this derivative coverage after validation.
No new user-facing API to document. Remaining: wrapper left/interval/truncated
derivative coverage (B3.3 gap 2), B4 metadata pruning/measurements, feature
activation decisions and likelihood docs. Reviewer-workflow investigation stays
deferred at the maintainer's request. No Slice B completion claim.
