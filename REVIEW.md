# A5 docs — built-in families, grids, custom-family refresh, docstrings

One docs step on `docs-A5` covering the whole A5.d1 split (d2–d8), as the
maintainer decided on 2026-10-15: no 300-line limit, plots allowed, document
the implemented behaviour (spec revision 25). No numerical code changed; the
`src/` diff is docstrings only.

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
