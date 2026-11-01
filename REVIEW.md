# MNIST.13 — Showcase specification

Independent review: round 1/3

Read INSTRUCTION.md section 1 and Process, PLAN.md MNIST.13, then the
.spec-seen design commit 00e7225 diff in SPECIFICATION.md and use-case 03.
This is process step 1 of 4, specification only. Implementation remains at
its prior behavior until steps 2–3; docs follow in step 4.

Invariants: fixed seeds, cohorts, architecture, acceptance checks and all
library contracts remain unchanged. Default float32 now includes oracle and
intercept fitting; censoring is fixed at 7. Revision advances 40 to 41.
The requested float64-default exception reference is retained verbatim in
meaning; other float64/calibration requirements targeted by section 1 are removed.

Evidence: git -C spec diff --check passed before commit; standalone design
commit 00e7225 changes only the three requested spec files (9 insertions,
13 deletions). Required ../scripts/spec-since.sh dev --mark succeeded, and
its marker was copied to this worktree. The script also updates dev/.spec-seen
by design. No runtime tests/builds run: no executable files changed.

Starting and previous-step commit c1a0915; original worktree base ee72c83.
No unresolved implementation choice in this step. Remaining steps: data/tests;
model/runner/tests/full experiment; README/docs/rendered verification.
