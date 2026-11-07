# meta-parallel-records handoff

Read CLAUDE.md’s “Step records”, then the independent-review skill. The root
workflow and reviewer role are coordinated changes in the workspace repo.

Each unique step owns its plan, handoff and deviations; no shared index or
archive edit. Detached reviewers derive paths from the step ID in their request.
Existing legacy steps keep their records and round counts until completion.
No implementation, tests, public API or specification changes.

Independent review: not requested; this is a maintainer workflow change.
Validation: git diff --check passed in both repos. In a disposable repository,
Git merge-tree merged two branches adding PLAN.md, REVIEW.md and DEVIATIONS.md
under work/AFT.1 and work/LI.1 without conflicts; all six files and unchanged
legacy PLAN.md were verified in the merged tree. Library gates were not run
because only workflow instructions changed.
Remaining work: activate the meta branch and restart/redirect existing agents
before their next step. No active feature records were migrated.
