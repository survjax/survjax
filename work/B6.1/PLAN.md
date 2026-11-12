# B6.1 — Document censored and truncated likelihoods

Step starting commit: 44b33b0731ebb6521dd36b402e61167ca8e9affd
Previous step: [B4.3.CLEANUP](../B4.3.CLEANUP/PLAN.md), same commit.
Original worktree base: cc0e30f. Cumulative gate: A12, existing features.

One docs topic: current likelihood semantics and diagnostics. Cover the docs
to-do items in [B2.1](../B2.1/PLAN.md) and [B2.2](../B2.2/PLAN.md), informed
by the [B3.3 audit](../B3.3/REVIEW.md) and wrapper evidence in
[B3.4](../B3.4/PLAN.md) / [B3.5](../B3.5/PLAN.md).

Update docs/likelihood.md to describe all four observation kinds with absent,
left, right or two-sided truncation. Explain conditioning on (L,U], event
density versus interval mass, partial intersections, uninformative sets and
empty intersections versus zero model mass. Add a small executable unit-rate
Exponential example with independent closed-form expected values; keep the
existing untruncated fitting example scoped to its cohort.

Update docs/technical-notes/likelihood-diagnostics.md so its valid interval row
is finite and its NaN triage separates data issues, parameter issues and
remaining model/numerical failures without an obsolete unsupported-mode mask.
Include executable assertions for the demonstrated masks.

Make brief scope/link corrections in docs/index.md, docs/survival-data.md,
docs/working-with-distributions.md, docs/numerical-behaviour.md and
docs/accelerated-failure-time.md so those introductions do not contradict the
likelihood guide. Leave metric-specific censoring restrictions unchanged.
Target fewer than 300 changed documentation lines across this one topic.
No source, tests, public API, dependency, tolerance or protected-path edits.

Spec: 2, 3.3, 4.2, 6.1–6.2, 7.1–7.3 and 10.5. Assertions belong in the
executable docs examples; no new standalone tests for these prose changes.
Run scripts/build_docs.sh directly (both dtype modes), inspect rendered changed
pages and executed values, then run scripts/check.sh --fast --through A12.
Commit the handoff, use the independent-review skill and the diff-display hook,
then stop.

## Docs coverage and remaining work

This step covers B2 likelihood/diagnostics documentation and the advertised
observation-mode scope, including wrapper availability from B3.5. Later docs
must cover trusted static metadata and rev-62 argument-form specialization
([B4.1](../B4.1/PLAN.md), [B4.2](../B4.2/PLAN.md),
[B4.3](../B4.3/PLAN.md)), update the historical CPU-baseline explanation, and
describe the ten comparable flexsurv cells/exclusions from
[B3.1](../B3.1/PLAN.md) / [B3.2](../B3.2/PLAN.md).
Pruning benchmark results belong to the experiments project. The final Slice B
evidence audit and integration remain after those docs updates.
