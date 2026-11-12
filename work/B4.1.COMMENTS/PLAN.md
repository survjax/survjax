# B4.1.COMMENTS — Explain likelihood pruning flags

Feature: readability of B4.1's static-metadata pruning. Base / previous step:
f6a9bf33122451720465814224e3d0a7fbc7c7ee ([B4.1](../B4.1/PLAN.md)).
Original worktree base: cc0e30f. Cumulative gate: A12 with existing features.

At the maintainer's request, change comments in src/survjax/likelihood.py:
spell out the six metadata indices, distinguish dataset-wide possibilities
from per-row masks, and explain why each branch evaluates conditional set mass
or exact density/normalization. Explain the zero placeholders used when a
piece is absent. Spec coverage: 3.2 and 7.2; verify index order against
src/survjax/_data_metadata.py.

No executable code, public API, dependencies, protected paths or docs changes.
No new tests: verify Python AST equality with the base, run the required fast
A12 gate, write the handoff, commit, and independently review, then stop.

## Docs to-do and remaining work

No new documentation requirement. B4.2 mixed exact/right optimization,
family activation and likelihood docs remain as in B4.1. Reviewer-workflow
investigation remains deferred.
