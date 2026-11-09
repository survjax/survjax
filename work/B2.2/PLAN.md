# B2.2 — Truncated likelihoods

Feature: observation.truncation. Implementation base: 5c271b4.
Previous step last checkpoint: 47aa112 (B2.1 handoff formatting).
Original worktree base: cc0e30f. Dependency: ../B2.1/PLAN.md and B1 hooks.
Maintainer explicitly authorized this step while the independent author fixes
D-B2.1-001. B2.1 is checkpointed, not yet accepted or independently reviewed.
Cumulative gate: A12 with observation.censoring and observation.truncation.

Spec 2, 3.3, 4.2, 7.1–7.3: exact rows use log f - log Z; censored rows
use the conditional interval hook after intersection. Inactive set uses its
own window, inactive density uses reference_time; invalid data retains NaN.
Zero window mass is invalid, positive tiny mass is never floored; full-window
censoring is structurally zero through the hook. No family-type dispatch.
Files: src/survjax/likelihood.py, distributions/base.py (correct docstring),
scripts/spec-features.txt, tests/unit/test_likelihood_core.py,
tests/unit/test_likelihood_truncation.py; timing list if required.
No new public API/modules/dependencies. Implementation and tests under ~300 lines.
Tests: analytical mixed values, partial/full intersections, delayed entry,
invalid rows, hook dispatch, jit/vmap and first/second parameter derivatives.
Run touched unit tests in both dtypes, staged truncation tests, fast A12 gate.
Known censoring oracle failure remains a blocker to final acceptance/review;
do not modify protected tests or remove either feature to hide it.

## Docs to-do

Update likelihood and diagnostics examples for truncation after integration.
Remaining after B2.2: B2.1/B2.2 acceptance and review, B3 references, B4 pruning/docs.

Run the independent review while the already-started x32 timing refresh
finishes. Isolate generated timing changes in a follow-up checkpoint.
