# TEST.STAGING: post-release spec-test staging

Base commit: 3abefc5 (main; A12 acceptance merge).
Previous step tip: ff07cf9 (approved A12 acceptance checkpoint).
Dependencies: A12 baseline. Preceding records: ../../REVIEW.md (legacy A12).
Cumulative gate target: A12. Enabled post-release features: none.

Change scripts/check.sh to read scripts/spec-features.txt, remove comments and
blank lines, join names with commas, and export SPEC_FEATURES for both spec-test
processes. Print the selected features. Add a guard exception for this authorized
test-staging branch limited to scripts/check.sh. Create the feature file with
only a header pointing to roles/TESTER.md's Post-release features table.
Verify CI and release already invoke the gate; no workflow edits are needed.

Public API: unchanged. Spec coverage: staged verification in sections 8.8 and
12.1–12.2; task instructions define the staging mechanism. No dependencies.
Validation: shell syntax, parser cases (empty, comments, whitespace, multiple
features), and scripts/check.sh --fast --through A12. No new library tests:
this step changes gate orchestration only.

Future feature steps add their fully implemented feature in the same diff and
name it alongside their cumulative gate target in their own PLAN.md.
summary.containers requires top-level SummaryOptions and SummaryInfo exports;
the other available names are summary.rmst, summary.mean, summary.quantile,
and sampling. None is enabled here.

Docs to-do: none; the workspace workflow already describes staging.
Integration: maintainer merges this step to main, then merges main into rmst
(no rebase). D-049 and the rmst plan belong to that later branch-specific work.
The pre-existing .spec-seen edit is not part of this step.
