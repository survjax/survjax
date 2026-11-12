# B6.2 — Document observation metadata and compilation

Step starting commit: 30c378e376667ebfe4b1619b5af43e1cde39b61d
Previous step: [B6.1](../B6.1/PLAN.md), same commit.
Original worktree base: cc0e30f. Cumulative gate: A12, existing features.

One docs topic: static observation metadata and its effect on compilation and
likelihood pruning. Dependencies and docs to-do sources:
[B4.1](../B4.1/PLAN.md), [B4.2](../B4.2/PLAN.md), and
[B4.3](../B4.3/PLAN.md). Rev 62 supersedes the earlier all-true traced rule.

Extend docs/survival-data.md beside the jit discussion with concrete versus
traced classification, the full None-versus-supplied rules for both forms,
inherited slice metadata, safe reconstruction via constructors, and the
conditions for compilation reuse. Explain that metadata only prunes work;
diagnostics and numerical semantics remain unchanged. Add a small executable
example comparing public PyTree structures and demonstrating reuse across
changed event values, including an invalid indicator.

Update docs/technical-notes/verification.md to label the committed CPU baseline
as historical, distinguish its all-true traced metadata from current
argument-form metadata, and link the observation guide. Preserve measured
numbers and their provenance; add no new benchmark results or artifacts.
Inspection found that the legacy harness still asserts all-true traced flags.
Scope its reproduction commands to historical checkout 51319c8 and state why
they fail on the current branch; do not change benchmark code in this docs step.

Spec: 3.2, 6.2, 7.2, 10.5, 12.1–12.3. No public API, source, test,
dependency or protected-path edits. Under 300 changed documentation lines.
Executable assertions accompany the prose; no new standalone tests needed.
Run scripts/build_docs.sh directly for both dtypes, inspect generated HTML and
executed output, and run scripts/check.sh --fast --through A12. Commit plan and
handoff, run independent review, open the step diff, then stop.

## Docs to-do and remaining work

This step covers metadata/pruning guidance from B4.1–B4.3 and the historical
baseline distinction deferred by B6.1. The ten comparable flexsurv cells and
excluded conventions from B3.1/B3.2 remain a separate docs step, followed by
the final Slice B evidence audit and maintainer integration. Benchmarks remain
in the experiments project; no performance rerun is part of this docs step.
