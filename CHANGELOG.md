# Changelog

## Unreleased — initial alpha

Draft notes for the proposed first public release, `0.0.1a1`.
The preparation baseline is `0.0.1.dev0`; no release date or published tag
is recorded here. Version, tag and publication follow the
[operator handoff](docs/releasing.md).
Documentation links below follow the source checkout or built docs version.

### Available in the preparation checkout

- Immutable `SurvivalData` pytrees, flag/bounds constructors, slicing and
  stored data-issue diagnostics. Data can represent exact, right-, left- and
  interval-censored observations and left/right truncation.
- Per-observation `model.log_likelihood(...)` for exact and right-censored
  observations without truncation, with prepared data or raw time/event arrays.
- `Exponential` and `Weibull`, log-parameter constructors, survival/hazard and
  log-density/log-CDF calculations, cumulative hazard and time-first grid
  methods. Data and distribution objects compose with JAX transformations.
- `ProportionalHazards` with caller-supplied log hazard ratios, nested
  distribution parameters and parameter diagnostics.
- Harrell's `concordance_index` and `concordance_index_with_info` for exact
  and right-censored data without truncation; higher scores indicate higher predicted risk.
- Staged `rmst(tau)` for Exponential, Weibull and their PH wrappers.
- Custom-family registration, parameter/static declarations, constraints and
  parameter-issue inspection; generic direct methods derive from the family's
  log survival and reference time.
- Bayesian Weibull/PH survival and RMST tutorial for a simulated randomized
  trial, and neural survival regression on MNIST with simulated outcomes.
  Their runnable notebooks and example dependencies are separate from core.
- Executable Sphinx/MyST documentation in separate float32/float64 processes,
  with notebook/example downloads. Read the Docs configuration, alpha package
  metadata and a release-only PyPI Trusted Publishing workflow are prepared.

### Limits of this alpha subset

- Likelihoods for left/interval censoring and any truncation remain
  unimplemented: unsupported rows return NaN, preserving supported rows in
  the same batch. Construction records issues; host-side validity checks
  remain explicit. Invalid observations or parameters also yield NaN
  likelihood rows, so inspect diagnostics before interpreting results.
- PH derivatives at extreme scales can be inaccurate, zero or nonfinite
  despite finite values and mathematical derivatives. See the
  [PH guide](docs/proportional-hazards.md).
- Weibull and PH-Weibull RMST support first shape derivatives; higher
  derivatives involving shape are unsupported. Their `tau=0` guarantees stop
  at value and first derivatives. Custom-family RMST is unavailable. See the
  [RMST guide](docs/rmst.md).
- Unrestricted means, medians, quantiles, event-time sampling, generic RMST,
  additional built-in families, AFT wrappers, Kaplan–Meier and additional
  evaluation metrics remain future work. Fitting, priors and neural-network
  architectures belong to external libraries.
- The core depends only on JAX and NumPy. The locked CPU baseline is Python
  3.12, JAX 0.11.2 and NumPy 2.5.3; no broader tested version or accelerator
  matrix is asserted by these notes. APIs remain experimental.

### Before publishing

The simulation demonstration, documentation cleanup and independent A11/A12
acceptance integration retain separate ownership and completion gates.
Recorded local checks through A10 do not certify those steps or the whole
initial release. Citation and contribution guidance are prepared. Local package
URLs, citation metadata, badge links and the publishing guard target
`survjax/survjax`. Publisher checks and documentation hosting remain pending.

Fresh Linux/RTD hosting, public documentation/download access, account
connections, publisher/environment protection and public package installation
remain unverified by this preparation. Complete the operator handoff's checks
and reconcile these notes with the final merged release commit before adding
a version/date and publishing. No pending work is declared complete here.
