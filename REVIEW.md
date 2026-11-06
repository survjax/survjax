Independent review: round 1/3 requested; pending (no round written)

# DOCS-WORKFLOW.1 — Local build provenance and inspection

Read PLAN.md, scripts/build_docs.sh, tests/tooling/test_build_docs.py, then
Local commands in docs/index.md. Spec 10.1/10.5: locked optional tools, fresh
execution in separate dtype processes, warnings/errors fatal, rendered review.
No public API, dependency, numerical source, protected path or spec changes.

Starting worktree and previous completed work: e6b3f9e. Separate archive-only
commit: 8517177; incoming PLAN.md preserved verbatim in plans/pre-docs-workflow.md.

Invariants: the last state in the whole-run identity is authoritative. Starting
an attempt truncates any old success; only two successful Sphinx invocations
and unchanged tracked sources/commit/status append success. TERM/INT produce
failure; uncatchable death leaves building. Existing HTML may remain but is
uncertified. No concurrent builds/edits; untracked content and post-build edits
are explicitly outside detection. Dirty status is provenance, not a snapshot.
Build-local supported environment variables replace runtime/cache locations;
HOME and the shared MNIST default/explicit override are preserved. Installed
jupyter_core.paths, IPython.paths and matplotlib.__init__ confirm these variables.

Checks performed in this checkout on 2026-11-06:

- Direct `scripts/build_docs.sh`: exit 0, no escalation, both Sphinx builds
  succeeded with -E -n -W --keep-going and MyST force execution unchanged.
  Identity commit 85171775b6596757302168a3f3b0c2b3bedb6c81, dirty with planned
  changes; started 11:32:13Z, completed 11:34:48Z; dtypes x32/x64. Fifteen source
  pages per mode; runnable notebooks executed afresh. Kernel TCP warnings
  were emitted by IPKernelApp; no Sphinx build warning/error or cleanup failure.
  This used the warm default MNIST cache; cold acquisition was not tested.
- `uv run --locked --group test pytest -q tests/tooling/test_build_docs.py`:
  4 passed in 2.59s. Final rerun after formatting: 4 passed in 1.88s; slowest call 0.39s.
  Tests cover paths containing spaces, actual mkdir/export environment seen
  by the child process, preservation of explicit MNIST path, dirty provenance,
  absolute output paths, second-dtype failure/TERM and tracked edits removing
  old certification. Stub uv avoids numerical reruns; it does not test Sphinx.
- `bash -n scripts/build_docs.sh`: exit 0. Shellcheck not run.
- Initial `uv run ruff format --check .` caught one test-fixture quote format;
  `uv run ruff format tests/tooling/test_build_docs.py` fixed it (1 reformatted).
  Final `uv run ruff format --check .`: 165 already formatted;
  `uv run ruff check .`: All checks passed; `uv run mypy src`: no issues in
  22 source files. `git diff --check`: exit 0.
- No unrelated numerical gate/test rerun, per the task's explicit instruction.
  Git staging/commit needed escalation after sandbox index.lock denial;
  documentation execution itself did not need escalation.

Visual evaluation: cua.getState returned no browsers and exactly
"Browsers: Error: The browser access host is unavailable". Native
cua.getApp('com.google.Chrome') worked. Opened only one new tab, navigated to
absolute file URLs in this checkout, read AX content and captured/examined
screenshots of x32 index.html#local-commands (build/cache/identity paragraphs)
and x64 same section scrolled to fallback/escalation/JupyterLab paragraphs.
Text, inline code, margins and navigation are readable without overlap or
horizontal clipping at 1920x1003. Closed only our new tab and observed the
original tab remaining. This is a changed-topic visual check, not a full-site
math/table audit; no offline MathJax rendering claim. Both index sources are
identical across dtypes. Existing examples/results remain unchanged.

Uncertainties/limits: no cold MNIST download, no process-restriction failure
reproduced, no guarantee of native browser/escalation on other hosts, no
concurrency support. The post-build test formatting/this review are metadata
changes after the recorded build; rendered docs and runner remain identical.

Remaining candidates (not started): single-dtype authoring preview, offline
MathJax, docs CI/artifacts, browser-host repair. No further required sub-step
in this first workflow topic once independent review is accepted. Stop then.

## Independent-review invocation outcome

Implementation commit df167c62599b7beb9b6acab5cdafad6bfec96064 was clean before
invocation. Ran the exact skill command in ../review/code-docs-workflow;
exit 1: "You've hit your 5-hour limit · resets 3pm (Europe/Amsterdam)".
The expected ../review/slices/DOCS-WORKFLOW.1.md does not exist. The skill's
single permitted no-round retry returned the same exit/error; a second file
read again reported no such file. No reviewer verdict or findings exist.
Stop here under the skill; do not claim this step reviewed or accepted.
No reviewer configuration or global permission/settings changes were made.
The implementation is validated and committed; independent review remains
required when the maintainer resumes it after reviewer capacity is available.
