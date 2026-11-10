# meta-parallel-records

Feature: parallel feature bookkeeping. Base: e461c1f.
Dependencies: none; coordinate activation with root reviewer/workflow changes.

Change CLAUDE.md and the independent-review skill so each new step owns
work/<STEP>/PLAN.md, REVIEW.md and optional DEVIATIONS.md. Retain legacy records
for existing steps, without moving them or changing active worktrees. Public
API, dependencies, numerical code and accepted specification: unchanged.

Validation: inspect instruction consistency and whitespace; demonstrate that
separate record paths merge cleanly in a disposable Git repository. No library
tests are needed for instruction-only changes.

Docs to-do: root docs/workflow.md and roles/REVIEWER.md change with activation.
Activation: maintainer merges this meta branch and refreshes active agents;
existing legacy steps can finish, then use new unique step IDs. Prior root
records and their open deviations remain available as history.
