# A5.2 review follow-up — nits and proposed A5.2b

Starting point: `a58c57c` on the maintainer-selected branch `A5`.
Reviewer verdict in `../review/slices/A5.2.md`: merge, with F-1 assigned to a
separately planned follow-up. The maintainer requested review fixes.

## Reading order and changes

1. `docs/distributions.md`: F-3 uses `jnp.isclose` for the cure survival
   derivative. Its analytical value is 1, but exp(log(p))/p need not round to
   exactly 1. The adjacent log-CDF derivative already used this comparison.
2. `tests/unit/test_family_infinity.py`: F-2 renames the last test to
   `test_finite_underflow_values_and_lower_tail_derivative`. The assertions
   check exactly that; their numerical content and tolerances are unchanged.
3. `PLAN.md`: proposed A5.2b addresses F-1 for mixed [1,+inf] batches, preserving
   finite cure limits and analytic overrides. `DEVIATIONS.md` records the
   remaining leak under D-016, which stays open for maintainer closure.

## Invariants and requirement evidence

No numerical source or public API changed in this fix-up. The renamed test
checks finite survival/log-CDF underflow values and the analytical lower-tail
log-CDF/rate derivative (spec 4.2/7.2). It does not prove that the infinity
substitution cannot activate at a finite time. That property currently rests
on the explicit `isposinf(time)` condition in the wrapper. No assertion treats
NaN gradients as guaranteed behavior.

The executable cure assertion checks the spec 4.2/7.3 analytical derivative
of S(inf)=p using a floating-point comparison (spec 10.5 recipe). F-4's extra
primitive evaluation remains accepted and unmeasured. F-1 is not repaired in
this fix-up: it requires the separately proposed numerical step. Derivatives
of infinite or NaN boundary outputs are outside that proposed guarantee;
finite interior components sharing a batch with them are the target.

## Checks actually run

`JAX_ENABLE_X64=1 .venv/bin/pytest -q tests/unit/test_family_infinity.py`:
**23 passed in 2.72s**. The separate `JAX_ENABLE_X64=0` process:
**23 passed in 2.35s**.

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh` exited 0:
**build succeeded** for x32 and x64, executing all three guides afresh,
including the revised cure assertion. Local kernel startup used the permission
required outside the sandbox. Existing kernel TCP transport warnings appeared;
Sphinx reported no errors. `ruff format --check` and `ruff check` on the renamed
test passed. `git diff --check` passed with no output.

The full through-A4 gate, mypy, mutations, R fixtures, wheel installation,
rendered-page visual checks and performance measurements were not rerun for
these test-name/doc-only fixes. The prior full-gate results remain recorded in
commit `a58c57c`; they are not new verification of this commit. Independent
tests, spec, marker, numerical source, dependencies and tolerances are untouched.

Commit the review nits and planning record, then stop for A5.2b plan approval
under AGENTS.md rule 1 before making its numerical changes. No A5.3 work.
