# DOCS.5 — Portal-compatible repository documentation

Read docs-worktree-instruction.md, scripts/build_docs.sh, DOCS-BUILD.md and the
URL map in DOCS-MIGRATION.md. No portal/discovery/selector implementation or
shared-spec changes. Existing Furo/Sphinx extension points remain available.
Previous/start: 80aee9e; original worktree base: 2093ca2.
Validation complete. Independent review remains waived for the docs overhaul.
User-authored inputs and callout edit remain unstaged.

## Build interface and compatibility

`--output-dir DIR` selects a dedicated build root; default is the script's
worktree-local docs/_build. Relative DIR is relative to the caller. Output stays
html-x32/html-x64; docs-root/source-discovery collisions are rejected. The script
selects its own worktree's .venv/source even if invoked elsewhere, assigns build
caches to the root, and generates an ignoring .gitignore. DOCS-BUILD.md documents
arguments, dependencies, exits and portal duties. DOCS-MIGRATION.md maps old URLs.
No portal, discovery or selector implementation; no spec symlink change.

## Verification

- `bash -n scripts/build_docs.sh`: passes. `--help` exits 0. Unknown flag,
  missing/empty output value and option-as-value each exit 2. Real calls with
  `--output-dir docs` and `--output-dir docs/custom-output` exit 2 as documented.
- `scripts/build_docs.sh` directly, escalated: exit 0, both dtype builds succeed.
- `/Users/wamster3/dev/survjax/dev-docs-review/scripts/build_docs.sh --output-dir
  'survjax portal build'` directly from /private/tmp, escalated: exit 0; both
  outputs in `/private/tmp/survjax portal build/html-x{32,64}`. This verifies a
  relative output path with spaces and invoking outside the selected worktree.
- Both build roots contain separate JAX/Matplotlib/IPython cache directories,
  Sphinx doctrees and matching source copies. MyST-NB's execution artifacts are
  rooted beside the HTML; its configured .jupyter_cache is created only if used.
  Default outputs pass git check-ignore; custom output is outside the repository
  and also contains the generated ignoring .gitignore.
- `uv run --locked --group docs --group examples python /tmp/audit_docs_kernel.py`:
  a live notebook kernel reports this worktree's .venv/bin/python3 and
  src/survjax/__init__.py. Local kernel argv uses Python from the running env.
- `uv run python /tmp/audit_prefix_docs.py <html-x64>` for each output root:
  real temporary HTTP serving at `/worktrees/circle/`, 22 HTML pages, 1,740 relative
  references, 75 unique HTTP URLs, CSS assets and search resources pass. No local
  root-relative URLs, localhost constants or escapes from the prefix. Internal
  fragments validated. Servers shut down afterwards. Search resources were
  retrieved; no browser-interactive search or portal selector was tested.
- `python /tmp/audit_docs.py`: both default dtype outputs, 22 pages and 1,740
  local links/assets each, zero failures; 17 executable pages / 145 cells.
- `JAX_ENABLE_X64=1 uv run --group test pytest -q tests/unit/test_docs_build_script.py`:
  7 passed in 2.12s. Same command with mode 0: 7 passed in 1.83s. Tests use a stub
  uv to verify worktree/env/output routing and failure behavior without notebooks.
- `uv run ruff format --check .`: 169 files already formatted. `uv run ruff check .`:
  All checks passed. Focused repeat after correcting a test assertion also passes.
  `git diff --check`: clean. No library/numerical regression gate: library unchanged.

Earlier audit tooling issues: HTTP status-only requests closed large responses
early, producing server BrokenPipe noise despite successful checks. Updated the
temporary checker to consume bodies; corrected an indentation error in that
checker; final two HTTP audits exit 0 without that noise. The first unit-test run
had 2 failures because the test expected uv arguments without its `run` subcommand;
corrected the assertion, with no production-code change; both final runs pass.

Portal limitations: allocate a unique root per worktree and serialize builds to
one root; publish only after success; preserve trailing slash; choose one dtype;
use URL mappings/fallbacks across branches. MathJax currently loads from its
external CDN. Immutable input/download caches may be shared; mutable build caches
are isolated. No selector/status widget or browser-interactive portal test claimed.
The user callout edit remains untouched and unstaged. All requested work complete.
