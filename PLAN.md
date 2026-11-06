# RELEASE.1 — Prepare Read the Docs builds

Clean starting SHA / original worktree base / previous completed step:
3cb4cd7fbba3313f00dfabed0028ee6f404384df (DOCS.6).
Plan-only archive: 2c991d0; integrated docs records moved verbatim to
plans/pre-release-prepare.md without closing other workstreams.

## Workstream sequence (one reviewed step at a time)

1. RELEASE.1: RTD configuration; execute and audit existing docs pipeline.
2. RELEASE.2: package description, URLs and alpha metadata; propose 0.0.1a1,
   subject to maintainer choice, with consistent version/tag instructions.
3. RELEASE.3: explicit-release-only Trusted Publishing workflow; validate
   wheel and sdist locally and publish the same tested artifacts later.
4. RELEASE.4: one documentation topic for the operator handoff, including RTD
   connection/public-version settings, PyPI pending publisher and release checks.

## This step

Files: .readthedocs.yaml, PLAN.md, REVIEW.md. No public API, dependency,
lockfile, docs page/navigation, protected path or specification changes.
Spec coverage: 10.3 (locked Python 3.12 CPU environment), 10.4 (MNIST hash
cache), 10.5 (executable documentation), 12.2 (publication separate).

Use supported Ubuntu 24.04/Python 3.12, pin uv to the local 0.11.19 baseline,
sync docs/examples with --locked in this checkout's own .venv, then invoke
scripts/build_docs.sh directly. Both dtypes execute; stage the contents of
html-x64 into READTHEDOCS_OUTPUT/html only after both succeed. Keep downloads,
figures and relative navigation intact. Do not require the spec symlink for
hosting. Use current official RTD custom-job/output documentation.

Checks: YAML structure and command execution/staging with existing tools;
existing build-script unit tests; direct both-dtype executable docs build with
a cold MNIST cache; rendered local link/fragment/download/image audit;
scripts/check.sh --fast --through A10; git diff --check. No new implementation
or tests needed for this reversible configuration-only step. Record timings
and compare with RTD's documented budget; do not equate macOS execution with
Linux/hosted verification. Check available Linux tooling and record any limit.
Independent review required (RELEASE.1), then stop. No publication/account work.

## Docs to-do

RELEASE.4 owns user-facing instructions; leave existing docs-agent pages alone.
Before public launch, run a fresh Linux/RTD build (including cold MNIST fetch),
verify project-specific time/memory limits and public access/downloads, and
record evidence. RTD project/account ownership is unverified.

Round 1 follow-up: make install explicit and reshim the pip-installed uv;
validate command lifecycle and build an archived checkout with dangling spec.
No new scope, dependencies or protected paths.
