# A5 docs — built-in families, grids, custom-family refresh, docstrings

One docs step on `docs-A5` covering the whole A5.d1 split (d2–d8), as the
maintainer decided on 2026-10-15: no 300-line limit, plots allowed, document
the implemented behaviour (spec revision 25). No numerical code changed; the
`src/` diff is docstrings only.

## Independent review round 1 fix-up (2026-10-17)

Responds to `../review/slices/docs-A5.md` (reviewed `02f7763`).

- **F-1 (should-fix):** `docs/distributions.md`. `LogLogistic.log_survival`
  and `LogLogisticWithLimits.log_hazard` now pass on the NaN of the safe
  interior value instead of the boundary constant
  (`limit = jnp.where(jnp.isnan(interior), interior, limit)`). That interior
  value is evaluated at t = 1, so it is NaN when a parameter is NaN. New
  prose explains why a selected constant hides a NaN parameter. A new
  executed cell shows a batch with shapes `[2, nan]`: limits in the valid
  row and NaN for survival, `log_cdf`, hazard and `log_prob` in the NaN row.
  There is also a new rule of thumb, "Keep NaN parameters NaN". No eager
  validation and no inferred domain.
- **N-1 (nit):** the "full relative precision" claims are qualified to the
  spec 4.1 limits. The lower tail holds wherever log H is representable. The
  upper tail holds wherever S is a normal float; past that, S underflows and
  log F rounds to 0. The guide shows the example from the review,
  `Exponential(rate=1.0).log_cdf(1000.0)`, which gives -0.0. I checked that
  it is -0.0 in **both** modes, not only float32, since e^-1000 is about
  1e-434. Same change in the `Exponential.log_cdf` and `Weibull.log_cdf`
  docstrings and in the generic late-tail sentence in `distributions.md`.
- Checks: the rendered x32 and x64 outputs of the new cell match the prose.
  A scratch run of the guide's classes gave NaN at t = 0, 1 and inf, eagerly
  and under `jit`, for NaN shape and for NaN scale. Under
  `jax.debug_nans(True)` the valid-parameter gradients were finite: S(0) in
  k gives 0, log F(inf) in k gives 0, and log f(0) in b with k = 1 gives
  -0.1. `scripts/build_docs.sh` exit 0 with "build succeeded" in both modes.
  `scripts/check.sh --through A5` exit 0: own tests 453/7 and 459/1,
  doctests 4 and 4, spec tests 1116/26 and 1121/21.
- Maintainer follow-up: the cure and log-logistic recipes no longer
  substitute a hard-coded time of 1. They substitute
  `jax.lax.stop_gradient(self.reference_time())`, the family's declared safe
  time and the same substitute the base uses (`_survival_methods.py:256`).
  The checklist and prose say so. Rendered outputs are unchanged except that
  the S(0) gradient prints as `-0.`. NaN shape or scale still gives NaN at
  0, 1 and inf, eagerly and under `jit`. Under `debug_nans` the boundary
  gradients in shape and scale are 0, and log f(0) in b at k = 1 is -0.1.
  Docs build: "build succeeded" twice. `check.sh` was not rerun (no `src/`
  change).
- Reviewer note on the opening paragraph: "the `src/` diff is docstrings
  only" was superseded by `00abf26`, which changed `base.py` constructor
  introspection (see "API reference" below).

## Docs theme (2026-10-16)

- Maintainer chose Furo (`furo` 2025.12.19, docs group only) for the API
  reference's navigation: pages on the left, the current page's sections
  and API objects on the right (`toc_object_entries_show_parents = "hide"`
  drops the repeated class prefix). `scripts/build_docs.sh` "build
  succeeded" in both modes, no warnings; API and grids pages inspected in
  headless Chrome screenshots.

## API reference (2026-10-16)

- `00abf26`: `register_family` wraps the dataclass `__init__` and gives it a
  keyword-only `__signature__`, so `inspect.signature`, `help()` and autodoc
  show `Weibull(*, shape, scale)` instead of `(*args, **kwargs)`. Test
  `test_signature_shows_keyword_only_fields` (built-ins, custom `Recipe`
  with a static default); it failed before the fix
  (`['args', 'kwargs'] == ['shape', 'scale']`).
- `docs/api.md`: autodoc (`{eval-rst}` blocks) plus napoleon, both part of
  Sphinx. Exports grouped as observation data, built-ins, writing a family;
  base-class methods grouped (implement, shape/indexing, direct, grid).
  `DataIssues` documented from `survjax.diagnostics` as `.issues`' type.
- `conf.py`: `autodoc_typehints = "none"` (ArrayLike expands to a long
  union), `autoclass_content = "both"`, `autodoc_preserve_defaults = True`
  (shows `default=MISSING`), `nitpick_ignore` for `jax.Array` (offline).
- Docstrings: formulas as inline literals or `::` blocks (`*`/`**` were
  parsed as emphasis); Returns first lines without a leading colon
  (napoleon read it as a type); `DataIssues` gains an Attributes section
  from spec 6.2 and `_data_validation.py`.
- `tests/unit/test_api_reference.py`: every `survjax.__all__` name has an
  autodoc entry; it failed when one entry was removed.
- Checks: `scripts/check.sh --through A5` exit 0 (own x64 453 passed,
  7 skipped; x32 459 passed, 1 skipped; doctests 4+4; spec 1116/26 and
  1121/21); `scripts/build_docs.sh` "build succeeded" in both modes with no
  Sphinx warnings; rendered signatures and entries read from the HTML.
- Not changed: SurvivalData/DataIssues/InvalidDataError docstrings from
  A1–A3 still cite spec sections in user-facing text.

## Review fix-up (maintainer comments, 2026-10-15)

1. `builtin-families.md`: "Unconstrained parameters: `from_log`" now comes
   before "What happens at time zero and at infinity" and "Precision in both
   tails". Sections moved unchanged.
2. `distributions.md`, new section "When the formula breaks down at zero or
   infinity": a log-logistic custom family written naively (NaN hazard at
   0/inf, NaN gradient of S(0)); selecting the known limits with the
   double-`where` pattern; the trap that a selected constant makes the
   autodiff hazard at zero 0 for every shape (wrong for k <= 1); overriding
   `log_hazard` and `log_prob` with derived limits; rules of thumb (values
   first, leading-term limits, leave NaN if unknown, check boundaries;
   `check_family` named as planned only).
3. WA inline comment answered in the checklist: where log S is not
   differentiable or a boundary limit is selected, override `log_hazard` and
   `log_prob` (`hazard` follows `log_hazard`, `log_prob` does not, per the
   generic default's own derivative); `log_cdf` needs no override because it
   uses only the value of log S. Comment removed.

Checks: `scripts/build_docs.sh` exit 0, "build succeeded" in both modes; new
section outputs read in both x32 and x64 HTML and match the prose (hazard at
zero inf/0.1/0, at infinity 0; interior log density equal for override and
default at t=5). No `src/` change, so `check.sh` was not rerun.

## Reading order

1. `docs/builtin-families.md` (new): exponential and Weibull as a narrative
   (constant vs rising/falling hazard, hazard/survival plot), values at time
   zero and infinity, the k=1 log-density derivative exclusion, precision in
   both `log_cdf` tails, `from_log`, and "Coming from R and NumPyro" (spec 4.6
   table, `survreg` mu/sigma, AFT→PH gamma=-k*beta, flexsurv `fit$res` vs
   `coef(fit)`, `weibullPH` m=b^-k, NumPyro `concentration`). R appears only
   as non-executed code blocks; the page says agreement is checked later.
2. `docs/grids.md` (new): pointwise pairing vs `_grid` twins, the
   equal-length pitfall, curves for three prognostic groups (the 11.1 fixed
   vector `(1.5, 12, -0.6, -1.2)`), a k=1 grid starting at t=0 with a finite
   gradient (11.1 vector `(1, 10, -0.5, -1)`), posterior draws with the
   `k[:, None]` pitfall from spec 5 giving `(K, D, N)` and a 90% band plot,
   and grids under `jit`/`vmap`.
3. `docs/distributions.md`: stale "planned" notes replaced with links; grid
   example added; "edges of the time axis" rewritten as narrative with printed
   outputs instead of asserts, now covering NaN precedence, zero-limit
   gradients at +inf, mixed-batch derivative isolation (A5.2b) and cure
   limits with safe operands; log-scale section adds the H / log H / h / log h
   table, the upper `log_cdf` tail (rev 25) and the built-ins' analytic hazard
   at infinity; checklist item for finite cure limits.
4. `docs/index.md`, `docs/playground.ipynb`: coverage text and toctree; a
   Weibull/grid/`from_log` section and new "things to try" in the playground.
5. Docstrings: `exponential.py`, `weibull.py` (formula, behaviour, Google
   Args/Returns/Raises, the 4.6 conversions kept in the Weibull class
   docstring, one doctest each); `base.py` (class overview, `log_survival`,
   `reference_time`); `_survival_methods.py` (six generic methods);
   `_grid_methods.py` (seven grid twins, shape example, one doctest).
6. `docs/conf.py` (`myst_heading_anchors = 3` for cross-page section links),
   `pyproject.toml`/`uv.lock` (`matplotlib` 3.11.2 in the `docs` group only).

## Invariants and claims

- Every number stated in prose was compared with the rendered output in both
  dtype modes (x32 and x64 HTML outputs extracted and read). Where modes
  differ the prose says so (naive `log(1 - S)` is 0 in float32 and inexact in
  float64). Random draws (`grids.md`) are not quoted numerically.
- Docstrings now describe observable values and derivative guarantees rather
  than the policy wrapper's mechanism. Corrected a wrong statement:
  `reference_time` must return exactly `batch_shape` (spec 4.2), not
  "broadcastable".
- The audit line claiming `ProportionalHazards.reference_time` had the wrong
  shape was itself wrong (the nested baseline is broadcast at construction;
  it returns `(3,)`); PLAN.md corrected.
- Core imports are unaffected: `matplotlib` is only in the `docs` group.

## Rendered-page evaluation (spec 10.5)

- Pages present: index, playground, SurvivalData guide, Exponential and
  Weibull, Survival curves and time grids, Writing a survival distribution.
- All examples executed in both modes; three figures inspected visually
  (Weibull hazards/survival crossing at t=b; group curves and hazards;
  posterior bands).
- Supported: built-in and custom families, pointwise and grid evaluation,
  `from_log`, boundary behaviour, jit/vmap/grad. Unsupported and labelled
  planned: likelihoods, log-normal/log-logistic, wrappers, summaries.
- Comparison evidence: none with R; R content explains input conventions
  only. Numerical agreement awaits the A8 fixtures.

## Checks actually run

    UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5

Exit 0, "check: all gates passed through A5": ruff and mypy clean; own tests
x64 451 passed, 7 skipped and x32 457 passed, 1 skipped; source doctests 4
passed in each mode; spec tests through A5 x64 1116 passed, 26 skipped and
x32 1121 passed, 21 skipped.

    UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh

Exit 0, "build succeeded" for both html-x32 and html-x64 (`-n -W`). The log
shows `[IPKernelApp] WARNING | Kernel is running over TCP without
encryption` from the Jupyter kernel's stderr; this is not a Sphinx warning.

## Unsure / for the maintainer

- `.spec-seen` is still rev 24 (`b917ebe`); the docs describe rev 25.
- The flexsurv `coef(fit)` and `weibullPH` covariate descriptions come from
  the spec and package documentation, not from running R here.
- A few code lines in the Markdown pages exceed 88 characters (ruff does not
  check `.md` cells); the notebook is ruff-formatted.
