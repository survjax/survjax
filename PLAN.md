# B1.d1 — Document interval probabilities

Previous implementation step: 535b854; original worktree base: cc0e30f.
Maintainer requested useful work while the independent spec-test author handles
D-049. One docs topic in docs/working-with-distributions.md: interval mass,
conditioning on a truncation window, readable S/F formulas, executable examples,
boundary policy and the current likelihood-integration limitation.
No code/API/dependency/protected-path changes. Spec 4.2, 7.1–7.3 and 12.2.
Use already-implemented Exponential/Weibull hooks; do not imply likelihood
support is enabled. Run scripts/build_docs.sh directly in both dtype modes,
check rendered topic, write REVIEW.md, commit, independent review, then stop.
B2 remains pending below; docs do not resolve D-049.

# B2.1 — Untruncated left/interval likelihood integration (blocked on D-049)

Start / previous step: 535b854; original worktree base: cc0e30f.
Maintainer authorized likelihood integration. First bounded step enables left
and interval censoring without truncation; B2.2 adds truncation combinations.
Files planned: src/survjax/likelihood.py, distributions/base.py (correct existing
likelihood docstring), tests/unit/test_likelihood_core.py,
tests/unit/test_likelihood_intervals.py, PLAN.md, REVIEW.md; duration list if needed.
Spec 2, 4.2, 7.1–7.3: dispatch censored intervals through family hooks, preserve
exact/right contributions and gradients, current/stored diagnostics, dtype and
batching. Safe inactive set inputs use (0,inf]; density uses reference_time.
Tests: mixed kinds with hand-derived expected values, hooks dispatch, invalid
rows, transformed construction, jit/vmap, first/second derivatives and debug_nans.
No public API, module or dependency changes. Docs to-do: expanded likelihood
modes and diagnostics after integration. Truncation remains explicitly unsupported
in B2.1 and is implemented in B2.2.

STOP: tests/spec/test_a6_unsupported_modes.py unconditionally requires the new
modes to produce NaN. No SPEC_FEATURES switch exists in the current gate or
acceptance suite. Independent-author update/sync is needed; do not change, skip
or xfail protected tests. See D-049. No implementation changes yet.
After resolving the acceptance transition: touched tests both dtypes, fast A10
(or maintainer-designated Slice B) gate, commit, independent review, then stop.
Remaining: B2.2 truncation, B3 full numerical/reference matrix, B4 pruning/docs.
