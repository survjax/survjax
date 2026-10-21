# A6.4 plan update after revision-32 spec-test sync

Status: planning only on maintainer-selected branch `A6`, based on `f07c560`.
The maintainer synced independent snapshot `08d97a6`; `.spec-seen` remains
`cf4fef9` (revision 32). This change updates PLAN.md and REVIEW.md only.
The replacement implementation still awaits plan approval.

## Read first

1. PLAN.md, "Synced revision-31/32 acceptance coverage": what the new suites
   require, which own checks remain, and when the public suites must run.
2. `tests/spec/test_a6_likelihood_gradient_isolation.py`: 41 cases comparing
   valid rows with evaluation alone, six derivative transforms, selected jit
   and method checks, and analytical anchors.
3. `tests/spec/test_a6_unsupported_modes.py`: supported-row values and first/
   second derivatives beside unsupported modes, diagnostics and triage.
4. PLAN.md's A6.4b1/b2/c: explicit four-way second-derivative coverage and
   the focused public acceptance command before the complete A6 gate.

## Invariants and test/spec mapping

- Spec 7.1–7.3/12.1 rev 32: the accepted mechanism is unchanged. Broadcast
  every own/nested parameter leaf to the current outer batch shape, expand
  the declared invalid-entry union over components, and stop gradients at
  invalid entries before family code runs. Valid entries remain differentiable;
  invalid-row numerical values are unchanged until final NaN selection.
- The isolation suite covers zero, negative, NaN and both infinities in
  built-ins and unguarded custom/nested families, component and shared leaves,
  named conditions at base and outer levels, and all four second-order
  forward/reverse compositions. Method checks accompany function checks.
- Spec 7.1/12.2 rev 31: the unsupported-mode suite now requires NaN eagerly,
  jit/traced construction and vmap, preserving supported values, gradients
  and Hessians with per-row/shared parameters. No data/parameter issue is
  invented; the recipe attributes the NaN to mode support.
- Own tests still cover smaller rebuilt leaves/current shapes, immutability,
  unconstrained siblings, draws and per-leaf vmap. The synced isolation suite
  does not replace these implementation responsibilities.
- A6.4a rework and private A6.4b1/b2 keep the through-A5 gate. Public A6.4c
  runs the new modules explicitly in both dtypes, then the full A6 gate.
  Their sync removes the pending-test prerequisite, not the implementation
  approval requirement. Invalid-row derivatives remain unspecified.

## Checks actually run

- Read all seven files changed by the maintainer's sync, including TESTMAP.md,
  API_ASSUMPTIONS.md, helpers.py and the parameter-issue test wording changes.
- `../scripts/spec-since.sh dev`: exit 0,
  `dev: spec unchanged since cf4fef9`.
- In separate processes with UV_CACHE_DIR=/private/tmp/survjax-uv-cache,
  SPEC_THROUGH=A6 and JAX_ENABLE_X64=1 / 0:
  `uv run --locked --group test pytest
  tests/spec/test_a6_likelihood_gradient_isolation.py
  tests/spec/test_a6_unsupported_modes.py --collect-only -q`.
  Both exited 0: `489 tests collected in 0.24s` per mode.
- `git diff --check`: exit 0, no output.

Collection verifies imports and parametrized test availability only. No test
bodies, runtime gate, Ruff, mypy or docs build ran for this planning change.
The likelihood API is still unimplemented; no passing acceptance claim.

## Limits and uncertainties

No new spec ambiguity or deviation. No spec-test, source, dependency, docs,
marker or protected-file edit. Step order remains A6.4a rework, A6.4b1,
A6.4b2, A6.4c, with separate review stops and the existing diff budgets.
