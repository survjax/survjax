# B1a — Generic interval probability hooks

Independent review: round 1/3

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
