# Changelog

## 0.0.1a3 — Unreleased

Prepared changes since alpha 2; publication and release date are pending:

- Likelihoods now support left and interval censoring and left, right or
  two-sided truncation, alongside exact events and right censoring.
- Add `log_interval_prob` and `log_conditional_interval_prob` for interval
  masses and conditional probabilities, with stable Exponential/Weibull methods.
- Add LogNormal and LogLogistic distributions and the AcceleratedFailureTime
  wrapper. PH and AFT likelihoods support the expanded observation modes.
- Derive traced observation metadata from constructor argument form, allowing
  unused likelihood paths to be pruned while retaining diagnostics.
- Document the expanded likelihoods, metadata and committed flexsurv comparison
  coverage. CI and release checks select all baseline acceptance tests plus
  the completed features in `scripts/spec-features.txt`.

Concordance still requires exact/right-censored data without truncation.
RMST remains limited to Exponential, Weibull and their PH wrappers, with the
[documented derivative limits](docs/rmst.md). These changes do not imply a new
published release or support for all planned slices. Unrestricted means,
medians, quantiles, event-time sampling, Kaplan–Meier and additional evaluation
metrics remain future work. The core still depends only on JAX and NumPy;
no broader version or accelerator support is claimed by this preparation.

## 0.0.1a2 — 2026-11-06

- Fix installed-version checks to follow project metadata after version bumps.
- Prepare alpha 2 with matching package, lockfile and citation versions.
- Keep the published GitHub prerelease `v0.0.1a1` unchanged. Its publishing
  build failed before the PyPI upload job ran.

## 0.0.1a1 — 2026-11-06 (GitHub prerelease)

The first GitHub alpha prerelease was published, but its PyPI publishing
workflow failed checks before upload. The preparation baseline was
`0.0.1.dev0`. Version, tag and publication follow the
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
