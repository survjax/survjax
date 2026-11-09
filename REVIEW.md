## Survival-function notation clarification

Rewrote the requested formula using S(t), defined the intersected endpoints,
and explained numerator and denominator. No behavior change. Formatting and
whitespace checks only; no numerical tests or independent review rerun.

## B1a docstring follow-up

Added the conditional interval formula from spec 4.2; no behavior change.
Read the log_conditional_interval_prob docstring in distributions/_log_math.py.
Independent review waived explicitly by the maintainer for this edit.
Validation: Ruff formatting check and git diff --check; numerical tests not
rerun for a prose-only edit. Prior implementation review follows unchanged.

# B1a — Generic interval probability hooks

Independent review: round 2/3

Read PLAN.md, spec 4.2 and 7.1–7.3, distributions/_log_math.py,
distributions/base.py, then tests/unit/test_interval_defaults.py.

Start / previous step / original worktree base: cc0e30f.
Archive commit: 2813892.

Invariants: pointwise family methods own support boundaries; interval arithmetic
uses safe operands before log differences. Equal endpoints give zero mass;
unresolved equal finite logs give NaN rather than inventing zero mass.
Conditional inputs intersect before evaluation; inactive numerators use the
window. Structural coverage gives exact zero only with finite log window mass.
No likelihood modes have been enabled yet. Built-in cancellation-resistant
formulas are the next B1 substep; generic subtraction is not tail evidence for
those formulas. No new module, dependency, protected path or docs change.

Tests map:
- interval values and transformations: spec 4.2 tail selection, 7.3 jit/vmap;
- endpoint policy: spec 7.1 equality and invalid order;
- conditional intersections: spec 7.1 partial/empty/full sets;
- parameter derivatives: spec 7.3 second-order forward/reverse interiors.

Checks: touched suite 5 passed in each dtype. Fast A10 gate passed:
implementer 1077 passed/8 skipped (x64), 1082/3 (x32); doctests 4 passed
per mode; spec 2148/28 (x64), 2154/22 (x32). Ruff and mypy passed.
Initial gate stopped at E741; renamed window endpoint locals and reran cleanly.
Duration probe: 5 passed; four tests exceeded one second. Regenerate the slow
list with scripts/update_slow_tests.sh --through A10: completed, 449 entries.
Full float64 duration suites: implementer 1317 passed/8 skipped; spec
2332 passed/28 skipped. Float32 logs were removed by the script on success;
its exit status alone is not evidence of test success (script tolerates failures).
No R fixtures regenerated and no docs build (code-only step).

Remaining: B1 built-in overrides; B2 integration; B3 full reference/derivative
matrix; B4 pruning and documentation. Generic defaults alone do not complete B1.

## Round 1 findings and fix-up

Reviewed commit: 45768c48ebd0dbf18c4014986d5fbedb39650f95. Verdict fix-first.
F-1: added lower-tail, narrow and finite far-tail analytical cases in both dtypes.
F-2: added NaN/negative parameter and NaN-endpoint cases; bounded uniform
support distinguishes true zero probability from invalid results.
F-3: added exact window boundaries, infinity endpoints, zero numerator with
positive window mass, and zero denominator. Strict-boundary mutants may still
produce the same exact zero by subtracting identical finite logs; the tests
assert public values rather than forcing internal branch selection.
N-1: mixed-row gradient coverage remains B3 work.
N-2: narrow generic derivative cancellation remains a B1b override motivation;
no tolerance was loosened.
Touched tests: 9 passed in each dtype. Fast A10 fix-up gate: implementer
1092 passed/8 skipped (x64), 1097/3 (x32); doctests 4 per mode; spec
2112/28 (x64), 2119/21 (x32). All gates passed, Ruff/mypy clean.
Regenerated the slow list because a new test exceeded one second: 687 entries.
The timing-only refresh is committed separately because its machine-load-sensitive
diff exceeds the implementation step size. Full duration runs observed: x64
implementer 1321/8, spec 2332/28; x32 implementer 1326/3. The final x32 spec
log was removed by the script; rely on the passed fast gate for that mode.
Reviewer-owned temporary worktree cleanup is left to the maintainer; no changes
made outside the allowed implementation checkout.
