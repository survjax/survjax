# B5.FAMILIES — Activate built-in family acceptance gates

Feature: LogNormal and LogLogistic acceptance activation. Base / previous step:
f568b9b61e5af45f43be13a67fdf80ee0e674468
([B4.2](../B4.2/PLAN.md)). Original worktree base: cc0e30f.
Cumulative gate: A12 with both existing observation flags and both family flags.

Dependencies: family implementation and documentation in
[D.LOGNORMAL.1](../D.LOGNORMAL.1/PLAN.md),
[D.LOGNORMAL.2](../D.LOGNORMAL.2/PLAN.md),
[D.LOGLOGISTIC.1](../D.LOGLOGISTIC.1/PLAN.md), and
[D.LOGLOGISTIC.2](../D.LOGLOGISTIC.2/PLAN.md); subsequent likelihood evidence in
[B3.4](../B3.4/PLAN.md), [B3.5](../B3.5/PLAN.md), and B4.2.
This closes the feature-activation gap recorded in [B3.3](../B3.3/REVIEW.md).

Change scripts/spec-features.txt to add family.lognormal and family.loglogistic
after checking the full existing family acceptance suites in both dtypes.
Include the censoring flexsurv and truncated-family acceptance files so the
combined observation/family flags are exercised. Summary and sampling flags
remain disabled: spec 12.2 brings the families forward independently, while
analytical summaries are enabled with their matching summary features.

No public API, source module, dependency, documentation page or test is added
or changed. No protected paths or tolerances change. Existing tests already
cover construction, constraints, closed forms, boundaries and tails, grids,
likelihood diagnostics, transformations and derivatives. Spec coverage:
4.1, 4.2, 4.5, 5, 7, 10.4, 12.1, and 12.2 (families brought forward).

Validation: run the full selected acceptance files with explicit flags and
SPEC_THROUGH=A12, once per dtype; then run scripts/check.sh --fast --through A12
using the updated default feature list. No new tests require slow-list timing
calibration. Record exact results in REVIEW.md, commit, run independent review,
and stop. Any newly exposed failure is investigated before activation is
claimed; acceptance tests remain read-only.

## Docs to-do and remaining work

Family documentation is already complete in the linked D.*.2 steps. This step
adds no docs to-do. Likelihood documentation (including the trusted metadata
contract and pruning performance), final Slice B evidence audit and maintainer
integration remain. Reviewer-workflow investigation stays deferred.
