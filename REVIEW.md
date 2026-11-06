# SIM.docs-rewrite — Maintainer's editorial revision

Read docs/use-cases/weibull-simulation.md, then the newest PLAN.md section.
This is a writing task; the maintainer explicitly waived tests and independent
review. No reviewer was invoked. No numerical/library files were changed.

The page incorporates the edited draft and resolves all WA comments: one trial
first, named patient count and fixed censoring limit, a visible fitting function
reused in vmap, explicit (20, 300) batch shape, and calibration introduced only
for the full study. The overlapping study-design example was removed. The
untracked draft was moved into the actual page, with links and root discovery
adjusted for docs. The saved curve figure is explicitly a different trial.

Invariants: the visible fit uses the summed likelihood, logs for positive
parameters, full Hessian covariance and exponentiated log-HR Wald endpoints.
The teaching fitter returns estimates/Hessian/acceptance; the full-run helper
also records individual failure reasons and skips no-event-arm optimization.
Full-study results, accepted-fit denominators and failure counts are unchanged.

Validation: scripts/build_docs.sh runs executable pages in both dtype modes.
The first sandboxed attempt was interrupted after kernel-shutdown permission
errors; the direct approved escalated retry is the validation run. The x32
build executed the revised page: HR 0.682, interval [0.472, 0.985], 20/20
accepted batch fits, and 80 smoke fits. Final build status recorded below.
No test suites or independent review ran. Browser visual inspection is not
claimed (browser host unavailable in the preceding draft task).

Starting commit and previous step final: b7a1a65.
Original worktree base: 8757c0f. This completes the requested editorial step;
no editorial substeps remain once the build passes.

Final result: scripts/build_docs.sh exited 0; both x32 and x64 builds succeeded
with warnings treated as errors. Weibull execution took 14.32 s and 14.11 s,
respectively. The study explicitly selects float64 in both builds.
