# B1a docstring clarification — Survival-function ratio

Start / previous step: bd9e60c; original worktree base: cc0e30f.
Replace the dense probability expression with S(t), separately defined
intersection endpoints, and a numerator/denominator explanation. Same files
and prose-only validation as the preceding follow-up; review waiver continues.

# B1a docstring follow-up — Conditional probability formula

Start / previous step: 84bd9bb; original worktree base: cc0e30f.
Add the spec 4.2 conditional interval formula to the existing docstring in
src/survjax/distributions/_log_math.py. No behavior or API change; no new tests.
Check formatting and whitespace. Maintainer explicitly waived independent review.

# B1a — Generic interval probability hooks

Start / previous step / original worktree base: cc0e30f.
Plan archive commit: 2813892. Maintainer approved Slice B and bounded splits.
Files: distributions/base.py, distributions/_log_math.py,
tests/unit/test_interval_defaults.py, tests/slow_tests.txt (duration refresh),
PLAN.md, REVIEW.md.
Public API: spec 4.2 log_interval_prob and log_conditional_interval_prob methods.
No new modules, dependencies, protected paths or docs changes.
Implement tail-selected log differences and safe inactive operands; conditional
intersection, structural full coverage, empty mass and zero denominator policy.
Tests: analytical exponential generic defaults; endpoint/invalid cases;
eager/jit/vmap agreement and second-order parameter gradients (4.2, 7.1–7.3).
Run touched tests in both dtypes and fast A10 regression gate. Commit and run
independent review, then stop.

Remaining: B1b built-in stable overrides (split further as needed); B2 likelihood
integration; B3 full numerical/reference matrix; B4 measured pruning and docs.
Docs to-do: document interval hooks and, after B2, expanded observation support.
