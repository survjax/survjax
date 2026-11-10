## Review resumption — 2026-11-10

Current base: 157a240 (main d8c85c7 merged, including the reference fix).
B2.1 implementation checkpoint: 5c271b4; last B2.1 checkpoint: 47aa112.
B2.2 subsequently integrated truncation and received a merge verdict in round 1.
The reference failure no longer reproduces; the fast A12 gate passes with both
observation features enabled. D-B2.1-001's formal status remains maintainer-owned.
Update this plan and REVIEW.md only, then request the deferred B2.1 review on
the integrated tree. No numerical changes or new implementation step here.
Review scope is the B2.1 changes and their current integration; the historical
plan below describes the originally isolated, untruncated implementation.

# B2.1 — Untruncated censoring likelihood

Feature: observation.censoring. Base: c9fee45 (merge of main 4239d61).
Previous step: B1.d2, 5fd6320; original worktree base: cc0e30f.
Dependencies: B1 family interval hooks; main's D-049 feature staging.
D-049 resolved by the maintainer on 2026-11-09; synced in 4239d61.
The separate reference issue D-B2.1-001 remains open.
Preceding plan: ../../PLAN.md (legacy B2.1 proposal and B1 docs history).
Cumulative gate: A12, plus observation.censoring only.

Enable all untruncated censored sets through log_interval_prob, preserving
exact contributions, diagnostics, common dtype and trailing-axis broadcasting.
Inactive intervals use (0,inf]; inactive densities use reference_time.
Truncation stays explicitly unsupported until B2.2. No API/dependency additions.
Files: src/survjax/likelihood.py, src/survjax/distributions/base.py,
scripts/spec-features.txt, tests/unit/test_likelihood_core.py,
tests/unit/test_likelihood_intervals.py, tests/impl_tolerances.py (finite-
difference step only); tests/slow_tests.txt if needed.
Spec: 2, 3.3, 4.2, 7.1–7.3. Tests: hand-derived mixed observations,
custom hook dispatch, safe invalid rows, jit/vmap, analytical gradients and
Hessians, check_grads, eager debug_nans. Run touched unit files fully in both
dtypes and scripts/check.sh --fast --through A12 before independent review.
The new censoring acceptance suites run; truncation remains staged out.

Regression discovered by full tests: generic interval fallback created an inactive
NaN under eager debug_nans. Update src/survjax/distributions/_log_math.py to
delay failure selection until endpoint cases are resolved; cover it in
tests/unit/test_interval_defaults.py. No change to failure semantics.

## Docs to-do

Update likelihood examples and supported-mode notes after B2 integration.
Remaining: B2.2 truncation, B3 numerical/reference matrix, B4 pruning/docs.

Blocked on D-B2.1-001. Maintainer requested a deferred spec-test handoff;
see SPECTEST-HANDOFF.md in this directory. Maintainer subsequently authorized starting B2.2 while the spec-test author
fixes the reference. Checkpoint B2.1 separately, still awaiting its gate and
independent review; this is not a completion claim.

Timing refresh generated 416 changed lines as new A12/censoring tests were
measured. Proposed split on resumption: isolate generated tests/slow_tests.txt
in a timing-record commit; keep the bounded implementation/test diff separate.
