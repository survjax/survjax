# B4.2 — Untruncated exact/right likelihood path

Feature: remaining Slice B static-metadata pruning. Base / previous step:
7ddec637227982785347880906ee15e1bcfe8036
([B4.1.COMMENTS](../B4.1.COMMENTS/PLAN.md)); implementation dependency
[B4.1](../B4.1/PLAN.md), f6a9bf3. Original worktree base: cc0e30f.
Cumulative gate: A12, existing observation feature flags unchanged.

Change only src/survjax/likelihood.py in production. When metadata rules out
left/interval censoring AND both truncation modes, use log_survival(c) for
right censoring and log_prob(t) for exact observations. Omit both interval
hooks and log Z. Keep B4.1's absent-kind pruning, safe substituted inputs,
current-field and stored-issue validation, parameter masking and output dtype.
All-true metadata and each disqualifying flag retain the general hook path.
Explain the flag indices and mathematical reason beside the branch.

Spec 2 explicitly states Z=1 without truncation and that sets closed at infinity
include the cure mass, so P(c<T<=inf)=S(c) even for improper families. This
settles the concern noted in B4.1's plan about custom-family boundary mass;
no family-type dispatch or proper-family assumption is needed. Do not rewrite
the earlier step's records. Correct custom overrides agree with these identities.

Add tests/unit/test_likelihood_untruncated_pruning.py (target <=230 lines):
prove omission of both interval hooks; independently exercise each disabling
flag; compare concrete/conservative metadata and inherited slice flags; verify
analytical values, Jacobians and Hessians for a cure model; preserve invalid-row
isolation, zero-censoring gradients and dtype/shape behavior via the new and
existing likelihood suites. Run eager debug_nans, jit/vmap and both AD modes.
No new public API, library module, dependency, tolerance or protected-path edit.

Reuse scripts/benchmark_likelihood_pruning.py unchanged in both dtypes. Record
the mixed-batch improvement versus all-true metadata and controls/limitations
in REVIEW.md. No new benchmark framework or timing threshold.

Run touched/focused tests fully in both dtypes, required slow-list refresh for
new tests exceeding one second (separate mechanical commit if needed), and
scripts/check.sh --fast --through A12. Commit the handoff and run independent
review, then stop. Spec: 2, 3.2–3.3, 4.2, 7.1–7.3, 12.1–12.3.

## Docs to-do and remaining work

Explain concrete versus conservative metadata performance in the later
likelihood docs. Family-feature activation, likelihood docs and final Slice B
evidence audit remain. Reviewer-workflow investigation stays deferred.
