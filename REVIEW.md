# A6.3 — Base-owned parameter diagnostics

Status: implemented on maintainer-selected `A6`, based on `a715d3e`.
Spec revision 30: .spec-seen fd150db; independent snapshot f527f75.
Ready for review; focused tests and through-A5 gate pass. Stop after commit.
The production/test diff is 299 changed lines; plan/review bookkeeping is extra.

## Read first

1. `src/survjax/distributions/_parameter_issues.py`: the composition path.
2. `src/survjax/distributions/base.py`: base-owned producer and condition hook.
3. `tests/unit/test_parameter_issues.py`: 13 cases and hand-enumerated masks.
4. PLAN.md: approved A6.3 scope and A6 docs to-do.

## Invariants and test/spec mapping

- Spec 4.7: constraint True means valid; diagnostic True means violation.
  Reduce exactly component_ndim minus event_dim trailing axes with any.
  `test_component_reductions_and_ordered_nested_conditions` covers remaining
  component axes, scalar constraints, union, boolean dtype and own-first order
  eagerly/jit/vmap. Scalar vector-constraint coverage is in
  `test_undeclared_and_conditions_only_families`.
- Own constrained fields precede nested fields regardless of declaration
  interleaving. Own conditions precede nested conditions. False masks stay
  present; nested names retain dotted prefixes. Smaller reconstructed nested
  batches broadcast to the current outer batch shape. Rebuilt-broadcast and
  nested-only tests cover this independently of constructor broadcasting.
- Only declared fields/conditions contribute. An undeclared NaN is not
  diagnosed. Empty declarations give all-False masks and scalar True ok;
  conditions-only families still report violations. Indexing recomputes ok;
  vmap stacks per-example ok. Tests compare exact masks and analytical indexing
  identities; no expected numbers come from the aggregation implementation.
- Spec 4.7: condition results must be mappings with nonempty undotted string
  keys and boolean arrays of exactly batch_shape. Errors use structural
  checks during tracing. Own key/dtype/shape tests complement the independent
  eight-case eager/jit matrix; MappingProxyType is accepted without coercion.
- Spec 4.1/4.5: from_log checks stored natural values (overflow inf and
  underflow zero). Constructors/direct methods never invoke diagnostics;
  a forbidden condition hook and explicit log S=2 at rate=-1 demonstrate this.
- Spec 4.1/6.3/7.2: traced construction reuses one compiled function for new
  same-shaped values. The compiled runtime passes transfer_guard("disallow")
  after input placement/warmup; assertions and host checkpoints stay outside.
  Producer results work with A6.2's existing host checkpoint.
- Constraints and hooks must obey their pure, traceable protocol and fixed-key
  contract. No registry, inferred domains, parameter repair or new cache.

## Checks actually run

UV_CACHE_DIR=/private/tmp/survjax-uv-cache for every uv/gate command.

- Focused command, separately with JAX_ENABLE_X64=1 and 0, SPEC_THROUGH=A6:
  `uv run --locked --group test pytest tests/unit/test_parameter_issues.py tests/unit/test_parameter_issue_container.py tests/unit/test_parameter_constraints.py tests/spec/test_a6_constraints.py tests/spec/test_a6_parameter_issues.py -q --tb=short`.
  Each exit 0: 173 passed, 1 skipped in 4.33s. The existing optional NumPyro
  interoperability case is skipped because NumPyro is not installed.
- `scripts/check.sh --through A5`: exit 0; guard ok (A6); Ruff format:
  59 files already formatted; lint: All checks passed!; mypy:
  Success: no issues found in 15 source files.
  Own x64: 531 passed, 7 skipped (42.94s); x32: 537 passed, 1 skipped
  (44.83s). Source doctests: 4 passed per mode (1.07s/1.01s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 308 deselected (28.46s).
  Independent x32: 1261 passed, 21 skipped, 308 deselected (23.85s).
  Final output: "check: all gates passed through A5".
- `git diff --check`: exit 0, no output before delivery.

## Limits and uncertainties

No new spec ambiguity identified. Spec, independent tests, diagnostics.py,
constructor/registration behavior, dependencies, exports and docs/ unchanged.
Narrative/API-method documentation is recorded under A6 docs to-do.
No docs build, boolean-mask gradient suite, complete A6 likelihood gate,
R fixture, wheel/install or benchmark checks. D-023 remains open for A6.4.
