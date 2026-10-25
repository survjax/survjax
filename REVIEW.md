# LL.2 — Document the distribution likelihood method

Read docs/likelihood.md first, especially “Two input forms”, then docs/api.md
and the small objective change in docs/simulation-guide.md. PLAN.md records
approval and completion evidence. Work reuses remove-ll from merged fab100f.

The docs now show the distribution method throughout. The tutorial compares
prepared SurvivalData with raw time/event inputs; the API reference renders
only the method. Its introduction also covers diagnostic types obtained from
methods. The simulation objective constructs a model and calls its method.
No source, docstring, test, dependency, spec or notebook content changed.

Invariants and spec mapping:

- Sections 6.1/7.1/12.1: the single method retains both input forms. The
  tutorial's two displayed totals match; keyword-only raw arguments, static
  misuse and explicit diagnostic checkpoints remain explained.
- Sections 7.1/10.2: there is no removed top-level function in current docs
  or notebook source. The API reference contains the inherited method with
  its arguments, returns, failure behavior and diagnostics.
- Section 10.5: synthetic examples execute afresh in separate dtype processes.
  Slice A still supports exact/right observations without truncation; the
  tutorial retains its unsupported-mode NaNs and diagnostics explanation.
  No R agreement or Slice A completion is claimed.

Checks actually run:

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`: exit 0,
  `build succeeded` in both float32 and float64. Nine pages built per mode;
  seven notebook/Markdown sources executed per mode. The likelihood tutorial
  executed in 4.01s / 3.35s. Full log:
  [/private/tmp/survjax-ll2-docs-build.log](/private/tmp/survjax-ll2-docs-build.log).
  The initial sandbox attempt failed to bind a Jupyter kernel socket; the
  approved rerun succeeded. Existing kernel TCP transport notices are printed;
  there are no Sphinx documentation warnings and warnings remain fatal.
- Chrome screenshot inspection through a temporary localhost server: the
  likelihood section and API method reference in both dtype builds have clear
  headings, readable code/output and navigation, with no new layout problem.
  The tutorial shows two matching totals, -23.124392 (float32) and
  -23.12439364 (float64). The expected TypeError example still renders.
- `JAX_ENABLE_X64=0` and `=1`, each with
  `UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run --no-sync python /private/tmp/survjax_ll2_docs_check.py`:
  exit 0 in each process. The temporary audit checks both HTML builds: matching
  totals, method reference anchor, absence of the old function and 273 local
  page/anchor links per build. It also checks notebook source. Logs:
  `/private/tmp/survjax-ll2-docs-check-x32.log` and
  `/private/tmp/survjax-ll2-docs-check-x64.log`.
- The same temporary check extracts only the simulation objective from its
  code block, without importing optimistix. With times [1,2,3], events [1,0,1],
  nll(eta)=6*exp(eta)-2*eta: value, gradient and Hessian at eta=0 are exactly
  6, 4 and 6 in both dtypes. The guide is excluded from Sphinx by existing
  config; the solver, replicate loop and full simulation recipe were not run.
- Tracked-docs search for the removed function/import, old two-entry-point
  heading and from_function: no matches. No generated output is staged.

Final Ruff formatting/lint and `git diff --check` run after saving this review;
record their output in PLAN.md. The full library gate and R were not run for
this 36-line docs migration. Generated HTML/caches and the temporary audit
are not committed. No unresolved concern or remaining visual inspection gap.
Commit one docs step on remove-ll and stop for review.
