# D.LOGNORMAL.2 handoff

Independent review: round 1/3

## Read first

1. [Plan](PLAN.md) and the preceding [code-step plan](../D.LOGNORMAL.1/PLAN.md),
   especially its deferred docs to-do.
2. `docs/builtin-families.md`: revised introduction and new LogNormal section.
3. `docs/api.md`: new autodoc entry and guide link.

## Invariants and coverage

The accepted specification is revision 51. This step documents existing
behavior only; public API, implementation, docstrings, dependencies, tests and
protected paths are unchanged.

- Sections 4.1/4.6: meanlog is a finite log-time location and sdlog a positive
  finite standard deviation of log time. R/flexsurv and NumPyro mappings are
  explicit. Mathematical median exp(mu) explains S(1)=1/2 at mu=0 without
  promising a summary method. The first new executable cell asserts that
  independent normal-symmetry identity.
- Sections 3.3/4.1/5: from_log preserves meanlog, exponentiates log_sdlog and
  stores natural sdlog. Its example uses a two-entry batch and a four-time
  grid, asserting shape (4,2) and exact survival boundary rows.
- Sections 4.5/7.2: density/hazard boundary limits and strict observation times
  are distinguished. Direct log-tail methods are recommended to avoid
  logging rounded probabilities. The tail cell uses standardized magnitude
  seven and asserts negative log survival/CDF in both dtype modes.
- Sections 10.2/10.5: the existing all-exports API reference test now passes.
  Sphinx executes the guide and generates the class/member entries in both
  modes; no new unit test or numerical tolerance is needed for this docs step.

## Validation

- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6
  tests/unit/test_api_reference.py -q`: 1 passed (1.21s).
- Same command with `JAX_ENABLE_X64=0`: 1 passed (0.92s).
- `scripts/build_docs.sh`, directly, without wrappers, redirects or environment
  assignments: exit 0; both Sphinx HTML builds succeeded and all notebook
  execution completed, including the new cells. Build identity ended with
  `state: success`. Builds were labelled dirty because the docs were not yet
  committed. Kernel TCP notices were emitted by the existing Jupyter setup;
  there were no failing Sphinx warnings under `-n -W`.
- Read-only HTML checks found the LogNormal class, from_log, log_survival and
  log_hazard anchors in each API page, and the new guide section/examples in
  each built-in-families page. This is structural inspection, not visual QA.
- `scripts/check.sh --fast --through A12`: all gates passed. Guard, Ruff
  formatting/lint and mypy passed (23 source files). Implementer tests:
  x64 1078 passed/8 skipped; x32 1083 passed/3 skipped. Source doctests:
  4 passed in each mode. Spec tests: x64 3397 passed/32 skipped; x32
  3391 passed/38 skipped. Each spec mode emitted the same two existing NumPy
  divide/invalid warnings in the PH reference calculation. Slow tests were
  deselected by the fast gate; the full merge gate was not run in this step.
- `git diff --check`: no output.

No unresolved documentation questions or spec deviations were found.

## Integration and remaining work

Step base and preceding step's last commit:
c3d097ca0bb8f581a795de1e84717298720aa363 (D.LOGNORMAL.1).
Original worktree base: 3abefc5683a3aee8169eb2deffe57602897a3af3.

This completes the preceding code step's deferred API and parameterization/tail
documentation. No remaining substeps are planned for the assigned LogNormal
family implementation and documentation. Other Slice D features remain outside
this assignment. The maintainer can update roadmap status at integration;
no shared progress record was edited.

The independent LogLogistic branch may also edit `docs/api.md` and
`docs/builtin-families.md`. Retain both autodoc entries, both family sections
and an introduction describing all families available after integration.
No other agent records or implementation files changed. No merge, rebase,
push or publication was performed.
