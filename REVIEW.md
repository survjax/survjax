# A5.d2 tutorial follow-up — optional hazard/density endpoint limits

Directly requested by the maintainer after `a708c35`, on `docs-A5.6`.
Read `docs/distributions.md`, "The hazard and density need more", first,
then the nearby rules of thumb and closing checklist.

## What changed and why

- Added a short introduction explaining that ordinary likelihoods use density
  at finite positive event times and survival/CDF for censoring. Mean/RMST
  integrals use survival, and an integral does not depend on the value at one
  endpoint. Numerical integration still needs to handle endpoint singularities.
- Framed the following overrides as optional, useful for initial/asymptotic
  hazard displays or grids including zero; a positive starting time often
  suffices for a hazard plot. Made subsequent imperative wording conditional
  on needing those endpoint limits, including in the checklist.
- Preserved the maintainer's pre-existing uncommitted change verbatim:
  "In many medical settings, progression risk is highest soon after diagnosis. A"
  in the piecewise-constant hazard introduction. It is included in the updated
  document and commit as requested.

The mathematical boundary contract and the distinction between base survival
limits and optional family hazard/density limits remain unchanged (spec
4.2/4.5/7.2/8.3). Existing examples, formulas and tutorial structure are retained.
No source, docstring, test, spec, configuration or dependency changes.

## Verification

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`: exit 0,
  "build succeeded" in both JAX_ENABLE_X64=0 and 1 processes. The existing
  script executes the examples afresh with Sphinx warnings as errors.
  Jupyter emitted its existing TCP transport warnings; both builds succeeded.
  Log: `/private/tmp/survjax-a56-docs-endpoint-nuance.log`.
- Parsed generated HTML for both modes: confirmed the optional-extension
  introduction and checklist wording, and the maintainer's sentence verbatim.
  No browser visual inspection was attempted; this checks generated content.
- `git diff --check`: exit 0, no output. The product diff is 36 changed lines
  in docs/distributions.md, including the retained maintainer edit.

No runtime/lint/type gate was rerun for this prose-only follow-up. The docs
build executed the unchanged examples in both dtype modes. No new ambiguity
or implementation concern was identified. Commit this follow-up and stop.
