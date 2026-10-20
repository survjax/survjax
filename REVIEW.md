# Revision 31 alignment — Approved unsupported-mode policy

Status: the maintainer approved the policy and spec clarification during
A6.4 planning. The spec change is committed separately in design; dev records
it as read on existing branch A6. No likelihood implementation is authorized
by this decision alone; A6.4a–c remain proposed.

## Read first

1. `spec/SPECIFICATION.md`, sections 7.1 and 12.2: row-wise failure and recipe.
2. `../design/decisions/0044-unsupported-likelihood-modes.md`: agreement and why.
3. PLAN.md's current A6.4 plan and DEVIATIONS.md D-023's decision note.

## Agreement and invariants

- Unsupported modes return NaN only on those rows, eagerly and under jit/vmap.
  Supported rows retain contributions and the derivatives guaranteed by 7.3.
- Unsupported valid observations acquire no invalid-data/parameter issue.
  Check both diagnostics, then mode support, before calling a remaining NaN
  zero normalization or numerical failure. No new public container or flag.
- Slice A's support predicate uses current normalized fields: untruncated
  exact rows or non-exact rows with upper=+inf. The recipe explicitly
  broadcasts support and both diagnostics to the contribution shape.
- This is accepted behavior, not evidence an evaluator exists. D-023's formal
  status remains the maintainer's. Implementation plan/code approvals are
  separate.
- The maintainer also approved the minimal generated API-reference exception
  in general. Code steps may add entries required by new exports without
  asking again. PLAN.md records this standing authorization; narrative docs,
  worked examples and broader reference changes remain separate docs steps.

## Commits and checks actually run

- Design `be7b253`: spec(7.1,12.2), revision bump to 31, policy and triage.
- Design `9cc2ab2`: decision 0044, register entry and cross-reference in 0043.
  No code or unrelated maintainer files are included in either design commit.
- `../scripts/spec-since.sh dev --mark`: exit 0, marker -> be7b253.
  Follow-up `../scripts/spec-since.sh dev`: exit 0,
  "dev: spec unchanged since be7b253".
- With UV_CACHE_DIR=/private/tmp/survjax-uv-cache and separate
  JAX_ENABLE_X64=1/0 processes, `uv run --locked python` executed the recipe
  extracted verbatim after its unimplemented likelihood call. Supplied a
  synthetic (2,6) output with supported finite, left-censored, truncated,
  unexplained NaN, invalid-data and invalid-parameter/unsupported cases.
  Exact hand-enumerated masks and every output shape (2,6) passed in both
  processes (exit 0). This tests classification of supplied values only.
- `git diff --check` in dev and design: exit 0, no output.
- No pytest suite, Ruff/mypy gate, docs build or likelihood runtime checks:
  source/tests/docs are unchanged and the evaluator is still unimplemented.

## Remaining work

Independent tests/spec remains read-only at snapshot f527f75, revision 30.
A maintainer-managed sync must supply acceptance coverage for revision 31's
stronger eager/compiled parity, preserved neighbors and triage requirements.
The A6.4 implementation steps still await approval; A6.4c's minimal function
reference entry is covered by the standing docs exception.
