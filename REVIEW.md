# Review: A7.d1 — Likelihoods in the tails

## What to read, in order

1. `docs/numerical-behaviour.md`, with executed output in
   `docs/_build/html-x32/numerical-behaviour.html` and the x64 equivalent.
2. `docs/likelihood.md`: the corrected -inf paragraph links to the new page;
   its construction-only NaN-triage qualification is retained.
3. `docs/index.md`: navigation and the guide description, then the current
   A7.d1 plan at the top of PLAN.md.

The new page is 212 lines. Together with navigation and the seven-line
likelihood paragraph diff, narrative/code-cell changes total 222 lines.
No source, docstring, test, dependency, spec or protected-path changes.

## Branch isolation

Commit target: existing A7, based on merged `9a034f4`, in
`/private/tmp/survjax-a7-docs`. During the work the shared dev checkout moved
onto A10-cindex. Before committing, saved the docs patch and original files,
transferred only this step's docs and A7 plan entry, and removed that exact
patch from the shared checkout. Its A10 plan/review/code were preserved;
the shared checkout was left clean on A10-cindex. No A10 commits or plan
sections are included here. Final builds/checks use this isolated A7 tree.

## Mathematics, examples and invariants

The tutorial gives the mathematics first, then code and interpretation:

- Three synthetic exact/right observations at 0, 1 and 100 months. For
  Weibull k=2,b=1, H(100)=10000: survival underflows but the right log
  contribution stays about -10000. The float32 -10000.001 rounding is explained.
- At c=1e-10, H=1e-20: survival rounds to one while log survival remains a
  representable small negative value. Both tail values agree with the defining
  H=(t/b)**k algebra; merged A7 tests provide independent mpmath evidence.
- k=50,b=1,c=1e10 gives H=1e500: the log contribution overflows to -inf with
  clear data/parameter diagnostics. True zero density/mass is distinguished
  from this overflow and from invalid/unsupported NaNs, without an R claim.
- Right censoring at zero yields exactly zero value and shape derivative,
  including k=.5 with infinite density at zero. Safe inactive evaluation is
  explained, with a link to the existing custom-family guide.
- Exponential per-row rates [1,-1] on an event at 2 and censoring at 1 give
  NaN loss but finite gradient [1,0]. The event's negative log derivative is
  2-1/r=1; the declared invalid row is stopped at its parameter leaf. The host
  finite-loss check rejects the proposal and identifies the second rate.
- Derivative guarantees are restricted to valid smooth finite parameter cases,
  through order two. Shared reverse derivatives beside an infinite row and
  overflow upstream of parameter leaves are explicitly outside the guarantee.

Examples display results rather than assertions, use existing public APIs and
synthetic data, and execute separately in the configured dtype modes. They do
not introduce an optimizer, global x64 setting, custom public family or fitting
API. Static output inspection is supporting evidence, not a numerical proof.

## Checks and actual outcomes

- Initial restricted `scripts/build_docs.sh` attempt exited 2 because Jupyter
  could not bind a local kernel port (PermissionError). Retried outside the
  sandbox with permission; shared-checkout builds passed, then were superseded
  by final verification on the isolated A7 tree.
- Isolated environment setup: offline sync lacked a cached JAX wheel; the
  first online sync hit missing imagesize metadata in the shared cache. Using
  a separate cache resolved it. `UV_CACHE_DIR=/private/tmp/survjax-a7-docs-uv-cache
  uv sync --locked --group docs` succeeded without changing pyproject/uv.lock.
  Verified survjax imports from this worktree's src/; Python 3.12.9,
  JAX 0.11.2, NumPy 2.5.3 and Sphinx 9.1.0.
- Final command, from this worktree:
  `UV_CACHE_DIR=/private/tmp/survjax-a7-docs-uv-cache scripts/build_docs.sh`.
  **Exit 0; build succeeded in both dtype modes**, warnings fatal, fresh
  notebook execution. New-page execution: **2.95s float32, 2.77s float64**.
  Complete log: `/private/tmp/survjax-a7d1-isolated-build.log`.
  IPKernelApp TCP notices are not Sphinx warnings; no execution error or
  Sphinx warning was reported by the successful final build.
- `.venv/bin/python /private/tmp/survjax-a7d1-inspect.py`: **passed in each
  dtype build**, verifying 7 output blocks, 6 headings, 2 display math blocks,
  5 table rows and 30 valid local links/anchors. Read every executed output
  against the algebra above, including NaN/-inf and gradient/diagnostic masks.
  Results: `/private/tmp/survjax-a7d1-html-inspection.json`.
- Final Ruff formatting/lint and git diff checks are run after this review's
  last edit, with results recorded in PLAN.md and the completion message.
  Full code/test gate, mypy, R reference generation, install, GPU and performance
  checks were not rerun for this docs-only step.

## HTML inspection and visual limitation

Generated HTML has the expected navigation, sections, classification table,
math markup and all outputs in both modes. The corrected likelihood paragraph
links to the guide and retains its data-as-constructed limitation. Existing
site pages/playground execute successfully with the new page included.

Browser security policy rejected opening the local file:// preview because
only http/https protocols are allowed. No alternate browser route was used.
Local HTML/output inspection is complete; **browser visual/layout and MathJax
rendering inspection remain unverified**. A maintainer can open the generated
page manually during review. No visual-layout or complete A7/Slice A sign-off
is claimed. Commit this docs topic on A7, then stop for review.
