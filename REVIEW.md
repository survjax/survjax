Independent review: round 1/3

# RELEASE.5d — Prepare organization identity

Reviewer Astra/low, authorized. Maintainer permits continuing separately
reviewed local steps until actual transfer; 5c verdict merge is recorded in
/tmp/survjax-RELEASE.5c-review-round1.md and its committed records.

## Read first

1. PLAN.md RELEASE.5d, current spec revision 47 10.1–10.3/13.
2. pyproject.toml / README.md / CITATION.cff diff: namespace links only.
3. .github/workflows/release.yml diff: owner equality only.
4. tests/install/test_wheel.py diff: built-artifact URL expectations.

## Invariants

- Intended target is survjax/survjax, explicitly authorized for local preparation.
  Transfer, availability and organization permissions remain unverified.
- Package repository/issues URLs, citation repository and README badge links
  use the new target. Author credit and Apache-2.0 unchanged. Version dev0,
  lockfile, runtime requirements, public API and numerical code unchanged.
- Release guard allows only intended new namespace, rejects old namespace and
  forks. This intentionally blocks old-namespace publication while transfer is
  pending. Trigger, artifact transfer, workflow filename/environment, pinned
  actions, OIDC permission scope and all other workflow behavior unchanged.
- No docs/ or shared CI edits in this code/metadata step. Current explanatory
  docs still describe old bindings; explicitly authorized next docs step 5e
  synchronizes those statements before final report/transfer. This intermediate
  commit is not a publishable release or assertion of completed migration.
- No git remote, account, transfer, hosting, visibility, push, tag/release or
  upload actions; no protected path, dependency, module or spec changes.

## Checks

- uv lock --check: exit 0, resolved 166 packages, lock unchanged.
- uv build --out-dir /tmp/survjax-release5d-dist: exit 0; sdist then wheel from
  sdist. Both still version 0.0.1.dev0. Metadata archive audit: URL fields,
  version/deps/content type/classifiers/license agree; descriptions match exact
  current README. Runtime ['jax>=0.11.2', 'numpy>=2.5.3'] unchanged.
- Official CFF 1.2.0 schema + format validation using existing docs environment:
  passed, URL equals new project repository; author unchanged.
- Parsed release YAML with existing BaseLoader: release published only,
  read-only workflow permissions and job-only OIDC unchanged. Exact guard
  evaluated for intended target (true), old owner (false), fork (false).
  Embedded run blocks bash -n passed; no old-owner strings in active changed
  metadata/test/workflow/README/CFF files. No new test dependency added.
- Full tests/install/test_wheel.py with the exact built SURVJAX_WHEEL:
  JAX_ENABLE_X64=1: 4 passed in 41.75s; mode 0: 4 passed in 41.82s.
  Includes runtime isolation, actual installed metadata, derivatives and typing.
- SURVJAX_WHEEL=/tmp/survjax-release5d-dist/survjax-0.0.1.dev0-py3-none-any.whl
  scripts/check.sh --fast --through A10: exit 0, all gates passed. Ruff 182
  files formatted/all checks passed; mypy 22 files clean. Implementer x64
  1066 passed/8 skipped; x32 1071/3. Source doctests 4 each. Spec x64 2149/27,
  x32 2156/20. Fast gate excludes slow cases; touched install tests above full.
- git diff --check: passed.

## Limits / next step

No hosted GitHub/RTD/Linux, OIDC, account/publisher, real destination, public
package or badge access verified. Guard check is local expression validation,
not remote workflow execution. No docs build in this code/metadata step; 5e
updates prose and builds both modes immediately after independent review.
Separately owned demo/docs/acceptance gates and final version selection remain.
A10 is not A11/A12 certification. No full numerical tier-2 rerun for link edits.

## Commit trace

Previous step / clean start: 9bffb535df46cff93511afc9578200ced49788e1.
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
After commit/review continue authorized 5e, then stop before actual transfer.
