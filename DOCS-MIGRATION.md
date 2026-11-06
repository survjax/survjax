# Pre-release documentation migration

Source baseline: `2093ca2`; approved structure: maintainer's `docsstructure.md`
and the ensuing review. This records content disposition, not only filenames.
No library API, examples, results, mathematical source or specification changes.

## Navigation and entry points

- `docs/index.md`: replaced the flat list and long page inventory with a short
  executable introduction and grouped navigation: Survival data, Survival
  distributions, Derived quantities and metrics, Use cases, Reference, Technical
  notes. The API reference stays directly accessible.
- `docs/working-with-distributions.md` (new): common method table, construction,
  batches, indexing, JAX composition, diagnostics and dtype guidance. Draws on
  the opening, batching and pytree sections formerly buried in `distributions.md`;
  examples now use built-ins and the likelihood API rather than a hand-written
  custom-family likelihood.
- `docs/playground.ipynb`: removed at the maintainer's request. Its data,
  distribution and concordance demonstrations already have guide equivalents;
  no exercises were promoted into new pages. The MNIST notebook remains the
  editable companion. No replacement playground was created.
- `docs/api.md`: remains the generated signature/docstring reference.

## Guide-by-guide disposition

| Original source / sections | Destination and treatment |
| --- | --- |
| `survival-data.md`: time/status and data representation | Retained; added early checking and slicing before the longer encoding story. |
| `survival-data.md`: interval/left censoring, endpoints, truncation | Retained as subsections of Other observation encodings, explicitly separated from supported likelihoods. |
| `survival-data.md`: diagnostics, JIT, dtype, R input conventions | Retained; detailed examples remain useful for interpreting data issues. Links updated. |
| `distributions.md`: One function describes the whole distribution; Many distributions at once | Adapted into `working-with-distributions.md`, using built-in models. |
| `distributions.md`: Distributions are JAX values | Condensed into Working with distributions; the separate hand-written likelihood/profile/score demonstration removed as duplication of the likelihood guide. |
| `distributions.md`: ConstantHazard, Gompertz, piecewise hazard | Retained in the shortened Defining your own distribution guide. Piecewise definition also appears in PH so that page executes independently. |
| `distributions.md`: local ProportionalHazards class taking hazard_ratio | Removed. PH page demonstrates the supported built-in wrapper taking log_hazard_ratio; no competing public-looking wrapper remains. Nesting declaration remains documented in API/advanced details. |
| `distributions.md`: edges, cure fractions, log-logistic endpoint overrides, Details | Moved to `technical-notes/custom-distributions.md`, with standalone setup and clearer wording about undeclared constraints. |
| `distributions.md`: Why work on the log scale | Removed repeated early/late-tail demonstrations. Common quantities are in Working with distributions; numerical guide owns tail explanations; custom checklist retains stable-log and override requirements. |
| `distributions.md`: Checklist | Retained in short extension guide, linking to advanced examples. |
| `builtin-families.md`: distributions, from_log, boundary values | Retained. |
| `builtin-families.md`: Precision in both tails | Moved into `numerical-behaviour.md`, renamed Numerical behaviour. |
| `builtin-families.md`: Coming from R and NumPyro, survreg/AFT/PH/flexsurv conversions | Moved to `technical-notes/parameter-conventions.md` with independently executable imports. |
| `builtin-families.md`: Comparisons with R; Integrity and regeneration | Moved to `technical-notes/verification.md`; retains fixture provenance and limitations. |
| `proportional-hazards.md` | Retained exponential introduction, then added piecewise-baseline progression using the built-in wrapper. Existing batching/likelihood/diagnostics and extreme-scaling limitation retained. |
| `likelihood.md`: censored contributions, both input forms, exponential fit | Retained. Explicitly labels the short Newton loop as illustrative rather than a general optimizer. |
| `likelihood.md`: Weibull fit and likelihood-ratio comparison | Removed as a second fitting demonstration; use cases own extended model-fitting workflows. No claim that the pending simulation tutorial is ready. |
| `likelihood.md`: When a contribution is NaN | Retained essential diagnostics and unsupported-mode distinction. |
| `likelihood.md`: Telling the causes apart | Moved to `technical-notes/likelihood-diagnostics.md` with independent model setup. |
| `rmst.md`: interpretation, area plot, family formulas, PH example, cutoff/batch rules | Retained. Removed repeated PH batching check; same shape rules stated concisely. |
| `rmst.md`: accuracy domain, invalid-entry isolation, closure overflow, derivative demonstrations, unsupported summaries example | Moved to `technical-notes/rmst.md`; standalone setup preserves executable checks. Main guide retains supported families, NaN policy, derivative support table and shape-Hessian limitation. |
| `concordance.md`: counts, scores, ties, unsupported cases, performance and held-out evaluation | Retained. Stale promise of future held-out/playground work replaced with MNIST link. |
| `concordance.md`: R reference evidence | Moved to verification note. |
| `grids.md` | Retained; interoperability link updated. Owns the detailed time-first/patient/draw axis tutorial. |
| `slice-a-verification.md` | Renamed/moved to `technical-notes/verification.md`. Scope explicitly historical/subset evidence; build/notebook instructions updated and stale workstream promises removed. |
| Repository `appendices/` | Unchanged sources/PDFs, now discoverable through `technical-notes/mathematics.md`. No duplicated derivations. |

## Use cases and deferred work

- MNIST page, notebook, code and committed results remain unchanged.
- Bayes: reviewed `../dev-release-bayes/docs/use-cases/bayesian-rmst-v2.md` as
  the working narrative. Neither version was imported into this branch. Add
  the accepted page under Use cases after its separate merge.
- Simulation: `docs/simulation-guide.md` remains byte-for-byte unchanged and
  excluded in Sphinx. It is tentative, not the accepted simulation tutorial.
  Maintainer scrutiny and the eventual simulation branch merge remain separate.
- Intended final use-case order: survival regression/simulation, MNIST, Bayesian
  survival analysis. Only available pages appear in published navigation; no
  empty or misleading placeholder pages were created.

## Verification approach

All extracted executable notes receive their own imports and required setup.
Internal links follow moved sections. The full docs build executes both dtype
modes, while the existing MNIST workflow remains float32. REVIEW.md records
actual build, rendered-output and link-check results. No independent reviewer
is invoked for this maintainer-approved restructure.
