# A5.d2 — Document the merged family boundary contract

Completed on `docs-A5.6` from `282dcca`, as directly requested by the maintainer
on 2026-10-19. The existing style, headings, tutorial sequence and analytic
hazard/density example are retained. The two documentation pages total 188
changed lines (83 additions, 105 deletions); PLAN.md/REVIEW.md are bookkeeping.

## Read first, in order

1. `docs/distributions.md`, "The edges of the time axis" and "Gradients at
   the boundaries": the base supplies survival/CDF/cumulative-hazard limits.
   CureHazard declares log(fraction) through log_survival_at_infinity(),
   replacing its obsolete infinity branch. Its limit derivatives stay active.
2. "When the formula breaks down at zero or infinity": the plain LogLogistic
   primitive now gives correct survival boundaries and zero S(0) parameter
   gradients. Removed the redundant manually guarded primitive. The separate
   guarded analytic hazard/density subclass remains; raw generic defaults at
   zero have no value/gradient guarantee, and generic hazards at infinity are
   NaN, including cure families. The checklist follows the same contract.
3. `docs/api.md`, "Limit at infinity": adds the generated optional-hook entry,
   with the proper-family default and cure override described.

## Invariants and evidence

- Spec 4.2/4.5/7.2: family survival formulas need finite positive times;
  reference substitutions are stopped, active infinity-hook values are not.
  The guide separates the base's four survival/CDF/H limits from density and
  hazard overrides. Hook NaNs govern base-supplied infinity values, including
  log density; analytic hazard limits remain family-supplied (review F-2).
- Executed guide outputs in both modes match the stated analytical values:
  cure fraction 0.25 gives S(inf)=0.25 and log F(inf)=log(0.75), with fraction
  derivatives 1 and -4/3. Plain log-logistic gives S([0,inf])=[1,0], with zero
  shape gradient for S(0); its generic hazards are NaN at both boundaries.
  The analytic extension gives h(0)=[inf,0.1,0] for shapes [0.5,1,2], h(inf)=0,
  the matching density limits, and NaN only in the invalid parameter row.
- `docs/builtin-families.md`, `docs/grids.md` and `docs/playground.ipynb` were
  audited. Their explanations/examples remain accurate; no changes were
  needed. The source playground has no custom boundary recipe to migrate.
- No library source/docstrings, tests, specification, exports, dependencies,
  theme/configuration or deviation statuses changed.

## Checks actually run

All docs builds used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache` and the
existing `scripts/build_docs.sh` (fresh execution, nitpicky mode and Sphinx
warnings as errors in separate JAX dtype processes).

- Initial sandboxed build failed: Jupyter could not bind a local kernel socket
  (PermissionError: Operation not permitted). Reran with execution permission;
  no build configuration or environment dependency was changed.
- First permitted build exited 0: "build succeeded" for x32 and x64. After
  small wording/wrapping corrections, the final build also exited 0 with
  "build succeeded" in both modes. Logs: `/private/tmp/survjax-a56-docs-build.log`
  and `/private/tmp/survjax-a56-docs-final.log`. Jupyter emitted its TCP
  transport warning; the Sphinx warning-as-error builds succeeded.
- Inspected generated guide HTML and captured outputs for both dtype modes,
  including cure limits/derivatives, plain log-logistic boundaries and S(0)
  gradient, generic hazards, analytic density/hazard limits and mixed NaN rows.
  Confirmed the new infinity-hook API anchor in both generated API pages and
  final wording in the guide. The updated guide's executed notebook contains
  38 code cells with no error outputs. Across all five executed notebooks,
  the only three error outputs are pre-existing survival-data examples tagged
  raises-exception, demonstrating expected data-validation failures.
- Browser security policy rejected a local-file preview. Visual browser layout
  inspection was unavailable; generated HTML content and numerical outputs
  were checked directly instead. No alternate browser route was attempted.
- `git diff --check`: exit 0, no output. The diff scope check found no source,
  test, spec or marker changes. No runtime/lint/type gate was rerun for this
  documentation-only change; all documented code cells execute in the builds.

No new spec question or implementation uncertainty was identified. Stop for
maintainer review after this docs commit; this does not claim Slice A's
remaining likelihood, fixture or completion gates are done.
