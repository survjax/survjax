# Chronox: project goals

This is the living statement of project direction. Edit it directly as priorities
change. Detailed API decisions and their rationale live in the
[design notebook](docs/design/README.md).

## Purpose

Build a lightweight, extensible JAX library for survival likelihoods, predictions,
and evaluation. Support classical, Bayesian, and neural-network workflows by
composing with external inference and optimization libraries.

Prioritize research impact and academic visibility, while welcoming future
company investment and co-development.

## What Chronox owns

- Explicit survival observations, with censoring and truncation that can differ
  per individual and compose correctly.
- Survival distributions and numerically reliable likelihoods/losses, including
  proportional-hazards and accelerated-failure-time wrappers around any
  baseline distribution.
- Predictions and summaries: survival curves, cumulative hazards, RMST, means,
  medians, and quantiles.
- Random event-time sampling for every family conditional on supplied
  parameters, optionally conditioned on truncation. A general observation-process
  simulation framework is deferred beyond v0.
- Kaplan–Meier estimation and survival metrics: Harrell's concordance, Brier
  score, integrated Brier score, and cumulative/dynamic time-dependent AUC, with
  Kaplan–Meier censoring weights by default. Metrics that need covariate-dependent
  nuisance models accept caller-supplied weights; Chronox does not fit those
  models.

The caller supplies distribution parameters, including linear predictors for the
wrappers. Fitting, priors, covariate mappings, optimizers, and neural-network
architectures belong to other libraries.

## Initial scope

- Parametric, continuous-time, single-event survival analysis.
- Exponential, Weibull, log-normal and log-logistic families, with a usable
  custom-distribution interface in which log survival is the one required
  primitive, and a hazard-based family base.
- JAX-compatible data and distribution objects.
- Cox, time-varying analyses, and multi-state models are outside initial scope.
- Conditional prediction APIs are deferred beyond v0, except
  truncation-conditioned sampling; truncation normalization remains part of
  likelihoods.
- Spline models are on the roadmap: a flexsurv-compatible Royston–Parmar family
  first, possibly an M-spline hazard family for Bayesian use.

## Quality goals

- Careful numerical behavior in tails, narrow intervals, and boundary cases.
- Differentiable likelihoods, predictions and summaries that work with JAX
  transformations, including derivatives of numerical summaries and
  reparameterized sampling.
- Check results against trusted implementations, particularly R survival,
  flexsurv, timeROC and riskRegression, plus independent mathematical cases.
- Keep dependencies and extension requirements small.
- Make likelihoods practical to use with NumPyro without requiring it for core
  use; the optional event-time distribution adapter is future work after v0.
- Provide reproducible v0 examples with survival curves for classical, Bayesian,
  and neural-network use; refine demo details during implementation.
- Use JAX-based external optimizers in Chronox fitting examples.

## Current priority

The independent review of the specification is complete and its findings are
settled (see the review dispositions). Next, on instruction, build Slice A: a
small working prototype checked against committed synthetic reference fixtures.
Keep the first working slice distinct from the complete v0 release. Prepare
broader release infrastructure as needed, without letting it displace the
prototype.

The [roadmap](docs/design/ROADMAP.md) records the next conversation and remaining
decisions. This document describes goals, not claims of implemented features.
