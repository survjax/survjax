# Documentation inspection for contributors

Run `scripts/build_docs.sh` directly from this checkout without redirects,
environment assignments or shell wrappers. The strict build executes notebooks
afresh in both dtype modes. Use the printed absolute HTML paths and read
`DIR/build-identity.txt` (`docs/_build/build-identity.txt` by default)
before inspection. DIR is the selected `--output-dir` build root. Its last `state:` must be
`success`; check the checkout, commit, dirty status and UTC completion time.
Compare with `git rev-parse HEAD` and `git status --short`. Edits after the build
require a fresh build; an already open browser tab can be stale. Reload the
intended checkout's local HTML. Do not run concurrent builds or edit sources
during verification. See [the build interface](DOCS-BUILD.md) for cache/runtime
details.

For agent inspection, use the currently documented computer-use APIs. Discover
available browser surfaces first. If discovery reports the browser access host
unavailable, native Chrome may still work: select `com.google.Chrome` with
`cua.getApp`, read the returned documentation, open a new tab and navigate to
the absolute local HTML URL using those APIs. Do not assume that either browser
host access or native Chrome is available on every host. Close only the tab
you created and record the actual capability failure if neither works.

Inspect navigation, tables, code outputs and equations in both dtypes. Reading
HTML, checking links or reading the accessibility tree verifies structure and
content; capture and examine screenshots to assess layout, clipping and math
rendering. Record pages inspected, checkout/build identity, screenshot findings
and limits in the review. MathJax currently loads from a CDN, so offline HTML
does not guarantee rendered equations.

If a build reproduces a sandbox Jupyter/psutil process-inspection or child
cleanup failure, retain its exact error and request the narrowly approved
`scripts/build_docs.sh` escalation, invoking the script directly. Writable
runtime directories do not repair process-inspection restrictions. Escalation
availability depends on the host. Historical cleanup errors are not evidence
that the current build failed; record the current command and outcome.

