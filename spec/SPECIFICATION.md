# Chronox v0: specification

- **Revision:** 5, 2026-09-28. Amends revision 4 to clarify right censoring
  at entry and endpoint diagnostics.
- **Status:** Accepted design. Not implemented. Slice A may begin when the user
  instructs implementation.
- **Companions:** [GOALS.md](GOALS.md) (direction and priorities),
  [REVIEW-DISPOSITIONS.md](REVIEW-DISPOSITIONS.md) (how every review finding was
  settled and where it lands in this document), [claude-review.md](claude-review.md)
  (the review itself).

This document is the normative contract for v0. It states *what* must hold; the
disposition log records *why* and *when* it was decided. Process history and
decision numbers from revisions 1–3 are kept only in section 15.

**Implementation verification** means code and measurements must establish a
stated property later. Acceptance of this design does not imply those checks
have passed. Measurements quoted from review prototypes are evidence for
feasibility, not release claims.

Here **v0** means the complete first release. **Slice A** is its first
development milestone, a deliberately smaller subset (section 12). The public
name is **Chronox**, with tagline **Survival analysis in JAX**. The distribution
name is `chronox-jax`, the import package is `chronox_jax`, and the suggested
alias is `cx`.

## Contents

1. [Purpose, scope, and ownership](#1-purpose-scope-and-ownership)
2. [Statistical definitions](#2-statistical-definitions)
3. [Observation data](#3-observation-data)
4. [Distributions and extension](#4-distributions-and-extension)
5. [Shapes and broadcasting](#5-shapes-and-broadcasting)
6. [Entry points, validation, and diagnostics](#6-entry-points-validation-and-diagnostics)
7. [Likelihood evaluation and numerical requirements](#7-likelihood-evaluation-and-numerical-requirements)
8. [Predictions, summaries, and sampling](#8-predictions-summaries-and-sampling)
9. [Nonparametric estimation and metrics](#9-nonparametric-estimation-and-metrics)
10. [Engineering and distribution](#10-engineering-and-distribution)
11. [Reference analysis and demonstrations](#11-reference-analysis-and-demonstrations)
12. [Verification and implementation milestones](#12-verification-and-implementation-milestones)
13. [Open items and implementation work](#13-open-items-and-implementation-work)
14. [Review status](#14-review-status)
15. [Decision traceability](#15-decision-traceability)

## 1. Purpose, scope, and ownership

Chronox is a lightweight JAX library for continuous, nonnegative, single-event
survival calculations. Users supply event-time distribution parameters, possibly
one set per individual or posterior draw. Chronox separates the event-time law
from the observation process encoded in data. It supports classical
optimization, Bayesian inference, and neural training by composing with external
libraries.

| Chronox supplies in v0 | Caller or external library supplies |
| --- | --- |
| Exact, right-, left- and interval-censored observations; per-individual left/right truncation | Data acquisition, preprocessing, assumptions about observation and sampling mechanisms |
| Exponential, Weibull, log-normal and log-logistic families; a custom-family interface; a hazard-based family base; proportional-hazards (PH) and accelerated-failure-time (AFT) wrappers | Covariate-to-parameter mapping (including the linear predictor passed to PH/AFT), regression coefficients, neural architectures |
| Per-observation log likelihood and diagnostics | Reduction, weights for a chosen objective, optimization, priors, posterior inference |
| Survival/hazard predictions, RMST, mean, median, quantiles, inverse cumulative hazard | Choice of prediction target and covariate profiles; aggregation over parameter draws |
| Event-time sampling for every family, optionally conditioned on truncation | Parameter sampling and censoring-process simulation |
| Kaplan–Meier (KM) estimation, including a censoring-distribution KM | Covariate-dependent censoring models (callers may supply their own weights) |
| Harrell's concordance, Brier score, integrated Brier score (IBS), cumulative/dynamic time-dependent AUC | Model selection; separation of training/validation/test data |

All four observation types may be mixed within a dataset, and absent, left,
right or two-sided truncation may vary by individual. The custom-family
interface and the generic numerical summaries are release requirements even
though every built-in family has analytical summaries where they exist. Three
runnable demos are required: classical, neural and Bayesian (section 11).

**Outside v0** (deferrals, not additional commitments): outside-interval
censoring; exact events recorded at zero; general missing-observation, padding,
mask and weight helpers (KM's `where=` mask in section 9.3 is the only
exception); conditional prediction interfaces other than truncation-conditioned
sampling; an observation-simulation framework; Cox, time-varying-covariate and
multi-state models; Royston–Parmar and M-spline families and the
proportional-odds wrapper (roadmap, section 4.4); Turnbull/NPMLE estimation for
left- or interval-censored data; Uno's IPCW concordance (candidate); fit,
uncertainty and bootstrap APIs; core plotting; and a NumPyro event-time
distribution adapter.

A future NumPyro adapter would use explicit integration imports and optional
dependencies while reusing Chronox calculations. The v0 Bayesian demo uses a
likelihood factor and shows posterior predictive sampling without an adapter
(section 11.3).

## 2. Statistical definitions

Let `T` be a continuous nonnegative event time with density `f(t)`, survival
`S(t)=P(T>t)`, CDF `F(t)=P(T<=t)` and cumulative hazard `H(t)=-log S(t)`. Where
defined, `h(t)=f(t)/S(t)`. All v0 built-in families are proper (`S(+inf)=0`),
but the evaluator never hard-codes that limit: it obtains limits from the
family's primitives (section 4.2), so improper (cure) families can be added
without changing it.

Censoring specifies an observed event-time set. Truncation specifies
eligibility for inclusion, `(L,U]`, with `0 <= L < U <= +inf` and `L` finite.
The likelihood conditions on inclusion. Define `Z=P(L<T<=U)`; valid conditioning
requires `Z>0`. These are likelihoods under ignorable censoring and the stated
truncation mechanism, not a joint model for informative censoring or arbitrary
selection.

Every observation is either **exact** at `t`, or **censored** to a set
`lower < T <= upper`:

| Observation | Event-time set | Likelihood contribution |
| --- | --- | --- |
| Exact at t | `T=t`, requiring `L<t<=U` | `f(t)/Z` |
| Right censored at c | `(c, +inf]` | `P(max(c,L)<T<=U)/Z` |
| Left censored at c | `(0, c]` | `P(L<T<=min(c,U))/Z` |
| Interval censored | `(a, b]` | `P(max(a,L)<T<=min(b,U))/Z` |

The table defines semantics, not an instruction to compute in probability
space. Implement in log space. Partial overlap with the truncation window is
allowed.

Structural cases:

- **Empty intersection** (for example, right-censored at `c>=U`, or an interval
  ending at or before `L`): invalid data.
- **Right-censored before entry** (`c<L`): invalid data
  (`censored_before_entry`). This data-quality rule applies to any censored
  row with upper bound `+inf`, however it was constructed. R's counting-process
  `Surv(start, stop, event)` also rejects equality, but Chronox's conditional
  likelihood permits a row censored exactly at entry.
- **Right-censored at entry** (`c=L`): valid and `uninformative`. The observed
  set contains the entire inclusion window, so its numerator equals `Z` and
  its log-likelihood contribution is exactly zero, for any valid `L` and `U`.
- **Other censored sets covering the whole window** (left-censored at `c>=U`,
  a finite-upper-bound interval containing `(L,U]`): valid, contributing
  exactly zero log likelihood, and reported by the non-fatal `uninformative`
  issue mask.
- **Zero probability from the family's support** is a different case
  (section 7).

With no truncation, bounds normalize to `(0,+inf]`, so `Z=1`. Truncation used in
a likelihood never silently conditions ordinary predictions or draws; sampling
conditions on truncation only when asked explicitly (section 8.6).

## 3. Observation data

### 3.1 Construction

`SurvivalData` is the canonical observation object. Two constructors are
supported (`*` marks the keyword-only boundary; these are signatures, not
executable calls):

```text
SurvivalData(time, event=None, *, right_censored=None, left_censored=None,
             interval_censored=None, time2=None, left_trunc=None,
             right_trunc=None, dtype=None)

SurvivalData.from_bounds(lower, upper, *, exact=None, left_trunc=None,
                         right_trunc=None, dtype=None)
```

Flag constructor (names follow R's `Surv(time, time2, event)`):

- `SurvivalData(time, event)` uses true/1 for exact events and false/0 for right
  censoring. The second positional argument is the event indicator, not a
  censoring flag.
- Omitting `event` and all censoring flags means every row is exact.
- Alternatively, explicit right/left/interval flags select each row's kind;
  with all flags false a row is exact. At most one flag may be true per row.
- Supplying `event` together with any non-None censoring flag is rejected, even
  if the flag contains only false values.
- `time` is the exact time, the one-sided censoring time, or an interval's lower
  endpoint. `time2` is an interval's upper endpoint. Kind is never inferred from
  equal or nearly equal endpoints; interval rows require `time < time2`.
  An interval-flag row with `time2=+inf` normalizes to right censoring at
  `c=time`. It is valid and `uninformative` when `time=L`, and invalid with
  `censored_before_entry` when `time<L`, for any valid truncation window.
- Indicators accept booleans or numbers exactly equal to 0 or 1, validated
  before conversion. Missing values, NaN, infinity, other codes and strings are
  rejected. R-style 1/2 event coding is not inferred.
- Missing required values are invalid. There is no implicit dropping or
  imputation.

Bounds constructor (matching R's `interval2` convention, without inferring
exactness):

- Censored rows mean `lower < T <= upper`. `upper=+inf` expresses right
  censoring and `lower=0` left censoring.
- Exact rows are marked by the explicit `exact` mask; their time is `lower`, and
  their `upper` must equal `lower` or be NaN. A NaN upper on an exact row
  normalizes to `lower`. Equal endpoints on a non-exact row are invalid.

`outside_interval_censored` is recognized only to raise an explicit
unsupported-feature error; it is not in the normal signature.

### 3.2 Normalized representation, shapes, and storage

The first time argument (`time` or `lower`) determines the observation shape.
Every other field must broadcast to that shape without enlarging it. Scalars
stay scalar, and multidimensional observations are not flattened. Example:
`time` of shape `(N,)` with flags of shape `(N,1)` is invalid for `N>1`.

Both constructors normalize to five public fields, all with the observation
shape:

| Field | Meaning |
| --- | --- |
| `exact` | bool; true for exact rows |
| `lower` | exact time for exact rows; otherwise the set's lower endpoint (`0` for left censoring) |
| `upper` | equal to `lower` for exact rows; otherwise the set's upper endpoint (`+inf` for right censoring) |
| `left_trunc` | `L`; `0` when absent |
| `right_trunc` | `U`; `+inf` when absent |

There are no placeholder values: every field is meaningful on every row.
Right/left/interval "kinds" are derived views (`upper==inf`, `lower==0`), used
for diagnostics, static metadata and metric support checks. NaN truncation
bounds are invalid.

`SurvivalData` is an immutable registered PyTree. Its array leaves are the five
fields plus one `int32` **issue bitfield** leaf with the observation shape,
which preserves raw-input evidence (conflicting flags, invalid indicators, and
so on) through normalization. `DataIssues` decodes the bitfield. New validation
rules add bits, not leaves, so the PyTree structure stays stable. Reconstruction
bypasses public normalization.

**Indexing.** Because every leaf has the observation shape, `data[idx]` and
`jax.tree.map` slicing over observation axes are supported and tested. This is
how train/validation/test splits and minibatches are made.

**Static metadata.** The PyTree's auxiliary data carries "may contain" flags:
which derived kinds are present and whether any left or right truncation is
present. They are computed when data are built eagerly from concrete values and
are all true when built under tracing. Slices inherit them unchanged (a subset
cannot contain kinds its parent lacks). The evaluator may use them to prune
branches at trace time (section 7.2); datasets with different flags have
different tree structures and compile separately.

### 3.3 Boundary and dtype rules

- Exact times must be finite and strictly positive. Zero, negative and infinite
  exact times are invalid; no epsilon is added. Truncation membership is
  `L<t<=U`.
- Censoring endpoints are nonnegative. `lower` is finite; `upper` may be
  `+inf`. In the flag constructor, right censoring at `+inf` and left censoring
  at `+inf` are both invalid (`invalid_time`). A right-censored set
  `(c, +inf]` with `c=L` is valid and flagged `uninformative`, including
  `c=L=0` and finite right truncation. If `c<L`, it is invalid. Thus
  `from_bounds(0, inf)` is uninformative when `L=0` and invalid when `L>0`.
- Lower truncation must be finite and nonnegative; upper truncation may be
  `+inf`; `L<U` is required.

Dtype policy:

- Time and bound fields normalize to a common floating dtype using JAX
  promotion, including JAX's weak typing: Python scalars adopt the dtype of
  the array inputs, and Python lists behave as `jnp.asarray` of the list. The
  result is at least float32, and integer-only or scalar-only input uses the
  configured default floating dtype. Omitting a truncation argument and
  passing its default value give the same dtype. Boolean times, complex
  values, strings and object or mixed-None arrays are rejected.
- Array input with dtype float64 (NumPy, pandas, JAX) when JAX's 64-bit mode
  is disabled is converted to float32, as JAX does, with one warning per call
  site. It is not an error; most data arrive as float64 from NumPy or pandas.
  Python scalars and lists carry no chosen precision and convert without a
  warning.
- An **explicit** request for an unavailable dtype (`dtype=jnp.float64` with
  64-bit mode disabled) fails clearly.
- Distribution parameters follow the same policy while keeping their structured
  dimensions. Likelihoods use the common promoted data/parameter dtype.
- Integer counts inside calculations (KM, metrics) are cast to the data's
  floating dtype before division. Dividing two `int32` arrays in JAX silently
  yields float32; a review prototype lost precision this way.
- float32 and float64 are both supported; float64 is recommended for reference,
  tail and narrow-interval work. Chronox never changes JAX's global 64-bit
  setting. Precision lost before values reach Chronox cannot be recovered.

## 4. Distributions and extension

### 4.1 Representation and built-in families

Distributions are lightweight immutable PyTrees. Parameter arrays are dynamic
leaves. Construction with traced parameters inside a model or objective must
work. Methods are pure: no mutation, optimizer state or mutable caches.

- **Constructors never check parameter values.** They are traced in almost every
  real call, and an eager-only check would make eager and `jit` behaviour differ.
  Invalid parameters surface through `parameter_issues()` and NaN results.
- **Natural parameters are stored as the leaves**, under the names the
  constructor uses.
- **Batch dimensions are broadcast at construction**, so every parameter leaf
  shares the leading batch shape (component axes such as a coefficient vector
  stay trailing). `dist[idx]` indexes batch dimensions, and `jax.tree.map`
  slicing and `vmap` over a distribution work.
- **`.shape` is never used for array shape on distributions.** Use
  `batch_shape`. The Weibull keyword stays `shape`, matching R and flexsurv.
- **Log-parameter constructors** avoid overflow in float32 neural use (for
  example, a network's log-scale output above about 88 overflows `exp`):
  `Weibull.from_log(log_shape=..., log_scale=...)`,
  `Exponential.from_log(log_rate=...)` and equivalents for the other families.
  They compute in log space internally and return the same class.

| Family | Constructor | Survival `S(t)` |
| --- | --- | --- |
| Exponential | `Exponential(rate=r)` | `exp(-r*t)` |
| Weibull | `Weibull(shape=k, scale=b)` | `exp(-(t/b)**k)` |
| Log-normal | `LogNormal(meanlog=mu, sdlog=sigma)` | `Phi(-(log t - mu)/sigma)` |
| Log-logistic | `LogLogistic(shape=beta, scale=alpha)` | `1/(1+(t/alpha)**beta)` |

All parameters are finite and positive except `meanlog`, which is any finite
real. External models transform unconstrained values themselves, for example
`scale=exp(X @ beta)`.

Family-specific numerical requirements:

- **Log-normal:** compute `log S` as `log1p(-ndtr(-x))` for `x>0` and
  `log_ndtr(x)` for `x<=0`, where `x=-(log t - mu)/sigma`, and `log_cdf`
  symmetrically. JAX's `log_ndtr` loses relative precision for moderately large
  positive arguments (up to 3.5e-2 relative error near `x=7.9` in float64, 4e-2
  near `x=5` in float32); see [notes/log_ndtr-precision.md](notes/log_ndtr-precision.md).
- **Log-logistic:** `log S = -softplus(beta*(log t - log alpha))` and
  `log F = -softplus(-beta*(log t - log alpha))`, both accurate in both tails.
- **Weibull:** use `xlogy`-style terms so that `(k-1)*log(t/b)` at `t=0`, `k=1`
  gives `log(1/b)` rather than NaN (section 4.5).

### 4.2 Extension interface

A custom family is a frozen dataclass subclass of `SurvivalDistribution`,
registered with an explicit decorator:

```python
@cx.register_family
@dataclasses.dataclass(frozen=True)
class MyFamily(cx.SurvivalDistribution):
    rate: jax.Array = cx.param()                      # batch parameter
    coef: jax.Array = cx.param(component_ndim=1)      # trailing component axis
    method: str = cx.static(default="a")              # hashable configuration

    def log_survival(self, t): ...
    def reference_time(self): ...
```

`register_family` generates PyTree flatten/unflatten (bypassing `__init__` on
reconstruction), `batch_shape` (from parameter fields, excluding component
axes), batch broadcasting and dtype normalization at construction, and batch
indexing. Field helpers are `cx.param(component_ndim=0)`, `cx.static(...)` and
`cx.distribution()` (a nested distribution, used by wrappers). There is no
automatic subclass registration.

**Required:**

| Member | Contract |
| --- | --- |
| `log_survival(t)` | `log P(T>t)`. Must keep relative precision when `|log S|` is small (compute it as `-H`,`log1p(-F)` with an accurate `F`, or a stable special function), because the default`log_cdf` depends on it. Must return correct limits at `t=0` and `t=+inf`. |
| `reference_time()` | A positive finite time per batch entry: the input substituted for inactive density evaluations (section 7.2) and the scale `s` for summary quadrature and the inverse-cumulative-hazard solver. |

**Optional overrides, with generic defaults:**

| Member | Default |
| --- | --- |
| `log_prob(t)` | `log(-∂t log S(t)) + log S(t)` by forward-mode autodiff. Accurate wherever the hazard is representable in linear space; review measurements: ≤5e-16 in float64 for Weibull and log-normal over `t` in [1e-6, 1e8]. |
| `log_cdf(t)` | `log(-expm1(log S(t)))`. Accurate until `|log S|`underflows (`F` below ~1e-38 in float32, ~1e-308 in float64). Override for CDF-natural families or when the far lower tail matters in float32. |
| `log_hazard(t)`, `cumulative_hazard(t)` | `log(-∂t log S)`, `-log S`. Built-ins override analytically; the generic `exp(log_prob - log_survival)` form cancels badly when `H` is large. |
| `log_interval_prob(lower, upper)` | Stable log difference with tail selection: `log_survival` when `upper=+inf`, `log_cdf` when `lower=0`, otherwise the better-conditioned of the two differences via `log1mexp`. |
| `log_conditional_interval_prob(lower, upper, left_trunc, right_trunc)` | Generic `log P(max(lower,L)<T<=min(upper,U)) - log Z`. Families override it to simplify terms that would otherwise cancel (for example, Weibull cumulative-hazard increments). The likelihood never dispatches on family type. |
| `inverse_cumulative_hazard(e)` | The generic solver of section 8.2. Built-ins override with closed forms. |
| `support()` | `(0, +inf)`. Bounded-support families override; used for quantile endpoints and quadrature breakpoints. |
| `parameter_issues()` | Reports no known violation. This is not a guarantee of validity. |
| summaries (`rmst`, `mean`, `median`, `quantile`) | Generic numerical versions (section 8). Families may override, including to report a known infinite mean. |
| `sample` | Inverse cumulative hazard applied to exponential draws (section 8.6). |

`cx.testing.check_family(dist)` compares every override with its generic
default and checks identities: `logaddexp(log S, log F)=0`, interval
additivity, `log_prob` consistency with the derivative of `log_survival`, and
finite parameter gradients at reference inputs. Built-ins and the documentation
recipe use it.

No NumPyro subclass, neural framework, flattened parameter vector or public
derivative method is required.

### 4.3 Hazard-based families

`cx.HazardFamily` is a small base for families defined through the cumulative
hazard. The author supplies `log_cumulative_hazard(t)` (and optionally
`log_hazard(t)`); the base derives:

- `log_survival = -exp(log H)`;
- `log_cdf = log(-expm1(-exp(log H)))`, switching to `log H` when `H` is tiny,
  so both tails stay representable;
- `log_hazard = log H + log(∂t log H)` by autodiff, which is well scaled (for
  Weibull, `∂t log H = k/t`);
- `inverse_cumulative_hazard` by the generic solver on `log H`.

Gompertz, piecewise-exponential and (on the roadmap) Royston–Parmar families fit
this base. The v0 recipe includes a Gompertz example.

### 4.4 Proportional-hazards and accelerated-failure-time wrappers

Wrappers take any baseline distribution and a caller-computed linear predictor.
Covariate mapping stays with the caller.

```python
base = cx.Weibull(shape=k, scale=b0)                              # batch ()
ph = cx.ProportionalHazards(base, log_hazard_ratio=X @ beta)      # batch (N,)
aft = cx.AcceleratedFailureTime(base, log_time_ratio=X @ beta)    # batch (N,)
```

The wrapper's batch shape is the broadcast of the baseline batch shape and the
linear predictor's shape; both are broadcast at construction (section 4.1).

| Quantity | PH: `H(t)=exp(eta)*H0(t)` | AFT: `S(t)=S0(t*exp(-eta))` |
| --- | --- | --- |
| `log_survival` | `exp(eta) * log S0(t)` | `log S0(t*exp(-eta))` |
| `log_prob` | `log h0(t) + eta + exp(eta) * log S0(t)` | `log f0(t*exp(-eta)) - eta` |
| `inverse_cumulative_hazard(e)` | `H0⁻¹(e*exp(-eta))` | `exp(eta) * H0⁻¹(e)` |
| `mean`, `rmst(tau)` | generic fallback | `exp(eta)*mean0`, `exp(eta)*rmst0(tau*exp(-eta))` |
| median, quantile | via `inverse_cumulative_hazard` | `exp(eta)` times the baseline value |

`eta>0` means higher hazard under PH and longer event times under AFT;
docstrings state both conventions. Required identity tests: PH-Weibull equals
`Weibull(shape=k, scale=b0*exp(-eta/k))`; AFT-Weibull equals
`Weibull(shape=k, scale=b0*exp(eta))`; PH-Exponential equals
`Exponential(rate=r*exp(eta))`.

**Roadmap (not v0):** a flexsurv-compatible Royston–Parmar family (a restricted
cubic spline in log time on the log cumulative hazard; knots as array fields; a
knot helper matching flexsurv's default placement; exact monotonicity checking
in `parameter_issues()`, since the spline derivative is piecewise quadratic);
an M-spline hazard family with positive coefficients, better suited to NUTS;
and a proportional-odds wrapper.

### 4.5 Direct-method value policy

Distribution methods are total functions of `t` over `[-inf, +inf]`. They never
raise and carry no issue masks.

| Input | Result |
| --- | --- |
| `t<0` | `S=1`, `log_prob=-inf`, `log_cdf=-inf`, `H=0`, `h=0` |
| `t=0` | boundary values from the family (Weibull `k<1`: infinite density; `k=1`: `log(1/b)`; `k>1`: `-inf`) |
| `t=+inf` | `log S = log S(+inf)` from the family (`-inf` for proper families), `log_prob=-inf` |
| NaN `t` | NaN |
| invalid parameters | NaN |
| `S(t)=0` for `hazard` | analytic `log_hazard` where overridden, otherwise NaN |
| summaries: `tau<0`; `p` outside `[0,1]` | NaN |

This differs deliberately from observation validation, which rejects exact
times at zero. Boundary values must also avoid NaN parameter gradients where the
value is finite. Test cases: a survival grid starting at `t=0` with Weibull
`k=1` (the `bc` fixed vector `(1,10,-0.5,-1)` does this); the gradient of
`S(0)` with respect to `k`.

### 4.6 Parameterization documentation

Docstrings give defining formulas, domains, units, shapes, common
parameterization conversions, reference-library mappings and a short conversion
example. Explaining an alternative does not promise another constructor mode
(the `from_log` classmethods of section 4.1 are the only alternatives).

The Weibull docstring explains this table:

| Convention | Mapping |
| --- | --- |
| Ordinary flexsurv / base R Weibull | `shape=k`, `scale=b` |
| NumPyro Weibull | `concentration=k`, `scale=b` |
| `survival::survreg` Weibull | log-time location `mu=log(b)`, reported scale `sigma=1/k` |
| flexsurv WeibullPH | `shape=k`, `scale=m=b**(-k)`, so `S(t)=exp(-m*t**k)` |
| Log-parameter inputs | `k=exp(log_shape)`, `b=exp(log_scale)`; see `Weibull.from_log` |
| Inverse time scale `r` | `S(t)=exp(-(r*t)**k)`, so `b=1/r` |

For common shape `k`, AFT regression has `log(b)=X @ beta` and equivalent PH
coefficients `gamma=-k*beta`; varying shape removes that simple
correspondence. flexsurv's natural estimates differ from `coef(fit)`, where
baseline shape and scale are logged. For the exponential, document
`rate=1/mean` and conversions from log rate and log mean. Log-normal documents
R's `meanlog`/`sdlog` and NumPyro's `loc`/`scale`; log-logistic documents
flexsurv's `llogis(shape, scale)`. Time scales have time units; shapes are
dimensionless.

## 5. Shapes and broadcasting

Distribution methods and likelihoods use trailing-axis broadcasting. They never
infer semantic roles from lengths, add outer products, pool cohorts or average
parameter draws. Likelihoods keep every broadcast dimension, including the
individual dimension.

Documentation labels (not stored metadata): `N` individuals, `D` parameter
sets or draws, `B` datasets, `K` prediction or evaluation times, `R` sampled
event times.

| Operation | Shapes | Result |
| --- | --- | --- |
| Likelihood | scalar data, scalar distribution | `()` |
| Likelihood | data `(N,)`, scalar distribution | `(N,)` |
| Likelihood | data `(N,)`, distribution `(N,)` | `(N,)`, paired |
| Likelihood | data `(N,)`, distribution `(D,1)` | `(D,N)` |
| Likelihood | data `(N,)`, distribution `(D,N)` | `(D,N)` |
| Survival | time scalar, distribution `(N,)` | `(N,)` |
| Survival | time `(N,)`, distribution `(N,)` | `(N,)`, paired |
| Shared grid | time `(K,)`, distribution `(D,N)` | `(K,D,N)` |
| Sampling | sample shape `(R,)`, distribution `(D,N)` | `(R,D,N)` |
| PH/AFT | baseline `()`, linear predictor `(D,N)` | distribution `(D,N)` |

A bare distribution batch `(D,)` against data `(N,)` pairs elements if `D=N`
and fails if the lengths differ and neither is 1. To evaluate every draw against
every individual, use `(D,1)` explicitly. The documentation shows this pitfall
with a posterior regression example (`Weibull(shape=k[:, None],
scale=jnp.exp(beta @ X.T))`). Data fields cannot enlarge the first time
argument's shape (section 3.2).

`survival_grid`, `cumulative_hazard_grid` and `hazard_grid` take a
one-dimensional shared time vector and return time-first outputs, equivalent to
passing `grid[:, None, None]` for batch `(D,N)`. Component axes never appear in
outputs. Sampling follows `sample_shape + batch_shape` (event shape is scalar).

Metrics and nonparametric estimators reduce the **last** axis (observations)
and follow different rules; see section 9.1.

## 6. Entry points, validation, and diagnostics

### 6.1 One data-input convention

Every function that consumes observations accepts the same two forms:

```text
f(preds, data)                          # data: SurvivalData
f(preds, time=..., event=...)           # raw exact/right-censored, no truncation
```

This applies to `log_likelihood`, `concordance_index`, `kaplan_meier`,
`censoring_kaplan_meier`, `brier_score`, `integrated_brier_score` and
`cumulative_dynamic_auc`, and to their `_with_info` variants.

- Predictions come first and data second. KM has no predictions:
  `kaplan_meier(data=None, *, time=None, event=None, where=None)`.
- The raw form takes only `time` and `event`, both required. Anything richer
  (other censoring kinds, truncation) goes through `SurvivalData`. The
  constructor signature is not copied into calculation functions.
- Mixing the forms is rejected. Passing a raw array positionally as `data`
  raises a `TypeError` pointing to the keyword form.
- Time grids at which metrics are evaluated are always the keyword-only
  `eval_times`, never `times`, to avoid confusion with `time`.
- One internal helper resolves both forms through the pure checked normalizer
  (section 6.2). Every entry point therefore behaves identically eagerly and
  under `jit`: invalid runtime values give NaN plus issues; static misuse (bad
  argument combinations, types or shapes) raises. An eager raw-form call with,
  say, event code 2 returns NaN rather than raising; the documentation points to
  `SurvivalData` for eager validation.

### 6.2 Validation

Structural data validation is a pure array computation, usable under `jit` and
`vmap`, independent of distribution parameters. Raw inputs are checked before
normalization and normalized fields afterwards. The result is per-rule masks
with the data shape, an invalid-row union and a scalar validity reduction.

```text
SurvivalData(...)                        # eager; raises descriptive errors
SurvivalData.with_issues(...) -> (data, DataIssues)     # pure; works under jit
DataIssues.raise_if_invalid()            # explicit host-only checkpoint
```

- The ordinary constructor raises descriptive structural errors, reporting
  counts and the first offending indices. Called with traced observation values,
  it raises at trace time and points to `with_issues`. Distribution constructors
  accept traced parameters.
- `with_issues` returns normalized data plus issues without Python tests on
  traced values. A returned data object alone is not a validity certificate.
  Issue arrays do not halt execution; validate simulation output before
  expensive fitting with a host checkpoint or explicit compiled orchestration.
- Inputs that cannot form an object at all raise rather than manufacture data.

`DataIssues` fields: `invalid_indicator`, `conflicting_indicators`,
`missing_time2`, `invalid_time`, `invalid_time2`, `invalid_interval`,
`invalid_truncation`, `exact_outside_truncation`, `censored_before_entry`,
`empty_observation`. The `censored_before_entry` mask applies to censored
rows with `upper=+inf` and `lower<L`, not `lower=L`. Their union is
`invalid_rows`; `ok` is the scalar reduction. `uninformative` is reported
separately and is **not** part of `invalid_rows` (section 2).

Upper-endpoint issue attribution for interval `time2` and `from_bounds`
`upper` is:

- `missing_time2`: an active interval row has omitted or NaN `time2`, or a
  non-exact `from_bounds` row has NaN `upper`. An exact `from_bounds` row
  may have NaN `upper` and does not set this issue.
- `invalid_time2`: an active interval `time2` or a supplied `from_bounds`
  upper is negative or `-inf`; also, an exact `from_bounds` row supplies an
  upper other than its lower or NaN.
- `invalid_interval`: an explicitly interval-censored row or a non-exact
  `from_bounds` row has `lower >= upper` after both endpoints pass presence
  and domain checks. A `+inf` interval `time2` or non-exact bounds upper is
  valid.

The flag constructor's `time` uses `invalid_time` when its active value
violates section 3.3 on any row, including an interval row's lower endpoint.
A NaN `time` sets `invalid_time`, since this required first endpoint has no
separate missing-value issue mask. The `from_bounds` `lower` uses the same mask,
including when it is NaN: exact times must be finite and positive; censored
lower bounds must be finite and nonnegative. Evaluate
presence and endpoint domains before interval order, so a malformed lower or
upper is not also labelled a reversed interval. Independent problems, such as
invalid truncation, may still set multiple issue masks on one row. Inputs with
nonnumeric time/bound fields or incompatible shapes cannot form an object and
raise instead of setting a row mask.

Likelihood and metric diagnostics share calculations with the ordinary paths.
`LikelihoodIssues` fields: `invalid_data`, `invalid_parameters`,
`invalid_conditioning`, `numerical_failure`, each with the full contribution
shape. Issues may overlap. Custom code may hide the precise cause; no claim is
made to detect every lost bit. `checkify` is not the public interface.

Ordinary likelihood returns NaN for detected invalid active evaluations,
without value-dependent exceptions, including eagerly. Genuine zero likelihood
with positive normalization is `-inf`. No useful derivatives are promised at
invalid or infinite likelihoods.

### 6.3 Result containers

Every diagnostic container (`DataIssues`, `LikelihoodIssues`, `SummaryInfo`,
`ConcordanceInfo`, `KaplanMeierInfo`, `MetricInfo`) and every estimate object
(`KaplanMeierEstimate`, `CensoringWeights`) is a registered PyTree, so it can be
returned from `jit` and `vmap`. Status codes are a public `IntEnum`. Containers
are accessed by field name; `_with_info` functions return exactly
`(value, info)`.

## 7. Likelihood evaluation and numerical requirements

### 7.1 Contributions and failure categories

```text
log_likelihood(dist, data=None, *, time=None, event=None) -> Array
log_likelihood_with_info(dist, data=None, *, time=None, event=None)
    -> (Array, LikelihoodIssues)
```

Return per-observation log contributions with no implicit sum, average or
weights. Callers use `-values.sum()` or `-values.mean()`. Multiplying invalid
rows by zero does not make them valid.

| Situation | Result |
| --- | --- |
| Valid data and parameters, positive numerator and `Z` | finite log contribution when representable |
| Valid observation, true zero model mass or density, positive `Z` | `-inf` (zero likelihood, not automatically an error) |
| Structurally impossible observation, invalid input or parameters | NaN plus the relevant issue |
| `Z=0` | NaN, `invalid_conditioning` |
| Positive but tiny `Z` | evaluated accurately in log space; no floor |
| Detected numerical failure | NaN plus issue; loss of precision is never relabelled as mathematical zero |
| Censored set covering the whole window, valid `Z` | exactly zero, detected structurally rather than by arithmetic cancellation |

Limits come from the family's primitives (`log S(0)`, `log S(+inf)`), not from
constants in the evaluator. Direct interval probability with equal endpoints is
zero; reversed endpoints are invalid. Emptiness produced by intersecting valid
sets is detected before interval primitives are called. Input intervals of
collapsed width are invalid.

### 7.2 Stable calculation and inactive inputs

Evaluation is vectorized over observations, with safe inputs selected before
each operation, including inside helpers and unselected branches. Selecting the
final value with `where` does not undo invalid derivatives already computed, and
under batching all branches execute.

Each row needs at most three pieces: the density (exact rows), the set
probability (censored rows) and `log Z`:

1. Intersect the real censoring set with the real truncation window first.
2. For **exact** rows, the inactive set-probability piece uses the row's own
   truncation window `(L,U]`, which has positive mass whenever the row is valid.
3. For **censored** rows, the inactive density piece uses `reference_time()`.
4. Substituted inputs are never intersected with the truncation window again.
   `stop_gradient` is applied to substituted values, never to actual parameters
   or active calculations. References never repair invalid active data or `Z`.

When the static metadata (section 3.2) shows a piece is never needed (for
example, no truncation, or only exact and right-censored rows), the evaluator
may skip it at trace time. For the dominant case this avoids two to three times
redundant work. Pruning is an optimization with identical results.

Interval probabilities use stable log differences with tail selection and
`log1mexp`/`expm1` identities. Built-ins override: for a finite exponential
interval `(a,b]`, `-r*a + log(-expm1(-r*(b-a)))`; Weibull uses cumulative-hazard
increments computed without subtracting nearly equal powers. Never clip
probabilities upward, floor `Z`, add an epsilon to data, or turn a narrow
interval into a density. Verify family reference inputs across tested scales,
shapes and dtypes.

### 7.3 Transformations and derivatives

Built-in likelihoods require first and second parameter derivatives on finite,
valid, smooth interior cases, composed with `jit` and `vmap`, in forward and
reverse modes, including Hessians. The same holds for the PH and AFT wrappers
over built-in baselines. The custom-family recipe is verified where its
primitives are twice differentiable; arbitrary user code has no automatic
guarantee. A default `log_prob` obtained by autodiff adds one derivative order;
JAX handles it.

Custom JAX derivative rules are allowed and must be documented and tested; use
`custom_jvp` where both forward and reverse modes are needed. Gradients with
respect to data times or truncation endpoints, at support-change boundaries, at
invalid or infinite likelihoods, and above second order are outside the
guarantee.

## 8. Predictions, summaries, and sampling

### 8.1 Surface

Methods belong on distributions and refer to the population event-time law at
the supplied parameters.

| Method | Meaning |
| --- | --- |
| `survival(t)`, `log_survival(t)` | `P(T>t)` |
| `cumulative_hazard(t)` | `-log S(t)` |
| `hazard(t)`, `log_hazard(t)` | `f(t)/S(t)`, where defined |
| `inverse_cumulative_hazard(e)` | the `t` with `H(t)=e`, for `e>=0` |
| `rmst(tau)` | `∫_0^tau S(t) dt`, finite nonnegative `tau` |
| `mean()` | `∫_0^inf S(t) dt` |
| `quantile(p)` | `H⁻¹(-log1p(-p))` for `0<p<1`; endpoints from `support()` |
| `median()` | `quantile(0.5)` |
| `sample(key, sample_shape=(), *, left_trunc=None, right_trunc=None)` | section 8.6 |

`survival_grid`, `cumulative_hazard_grid` and `hazard_grid` are thin time-first
wrappers (section 5). `rmst_with_info`, `mean_with_info`, `quantile_with_info`
and `median_with_info` return `(values, SummaryInfo)` and share the ordinary
backend.

Analytical summaries of the built-ins:

| Family | Mean | Median / quantile | RMST |
| --- | --- | --- | --- |
| Exponential | `1/r` | closed form | `-expm1(-r*tau)/r` |
| Weibull | `b*Γ(1+1/k)` | `b*(-log1p(-p))**(1/k)` | `b*Γ(1+1/k)*P(1/k, (tau/b)**k)`: closed-form value, quadrature tangent (section 8.4) |
| Log-normal | `exp(mu+sigma²/2)` | `exp(mu+sigma*Φ⁻¹(p))` | `tau*S(tau) + exp(mu+sigma²/2)*Φ((log tau-mu-sigma²)/sigma)` |
| Log-logistic | `alpha*(π/beta)/sin(π/beta)` for `beta>1`; known infinite for `beta<=1` | `alpha*(p/(1-p))**(1/beta)` | generic quadrature |

A converged numerical mean does not prove a finite expectation. RMST at a large
cutoff is never reported as a mean. A known infinite mean (from a family
override) has its own status; unresolved convergence yields NaN.

### 8.2 Inverse cumulative hazard solver

Quantiles, generic sampling and PH wrappers all use one solver for `H(t)=e`:

- Work in `u=log t` on `g(u)=log H(e^u) - log e`, using safeguarded Newton
  steps with slope `t*h(t)/H(t)` from `log_hazard` (or autodiff), and bisection
  when a step leaves the bracket. Bracket comparisons are strict, so a converged
  iterate is never replaced by a bisection step.
- The bracket starts at `log reference_time()` and expands geometrically in both
  directions up to a static step limit. Tolerances are relative in time.
- Improper families: if `e >= H(+inf)`, the result is `+inf`. Bounded support
  limits the bracket through `support()`.
- Non-convergence gives NaN and a failure status.
- Built-in closed forms: exponential `e/r`; Weibull `b*e**(1/k)`; log-logistic
  `alpha*expm1(e)**(1/beta)`; log-normal `exp(mu+sigma*Φ⁻¹(F))` using
  `F=-expm1(-e)` for small `e` and `-Φ⁻¹(exp(-e))` for large `e`.

Review measurements (float64): the generic solver matched Weibull closed forms
to 1.6e-15 in one Newton step, and log-normal and log-logistic to ≤2e-11 within
8 iterations over 200k draws; log-logistic extreme tails (`e` from 1e-12 to 300)
to 9e-15.

### 8.3 Numerical integration

Generic RMST and mean use double-exponential quadrature:

- **RMST:** tanh-sinh on `[0, min(tau, s)]` and `[min(tau, s), tau]`, where `s`
  is `reference_time()`, with further splits at known breakpoints (a finite
  support end inside the interval, future spline knots). Cut the upper range
  where `log S` falls below the dtype's representable minimum.
- **Mean:** tanh-sinh on `[0, s]` plus exp-sinh on `[s, +inf)`.
- Levels double with nested nodes; the maximum level is static, so node arrays
  have fixed size. Node positions near endpoints use complement forms; the node
  range depends on dtype.
- A non-finite integrand at any node is a failure status, never dropped.
- Convergence requires two successive levels within `atol + rtol*|estimate|`.
  This is a heuristic, not a guaranteed bound; verification covers heavy tails
  (log-logistic `beta` near 1), bounded support and scale changes. A family's
  known-infinite mean status is still needed, since divergence detection is
  heuristic.

Splitting matters: in the review prototype, one tanh-sinh rule on `[0, tau]`
gave a 38% gradient error for Weibull `k=20`, `tau=1000*b`; splitting at `b` and
cutting the tail gave ~1e-15.

`quadax` is a JAX library with tanh-sinh integration but depends on Equinox; it
may serve as a test cross-check only.

### 8.4 Derivatives of summaries

Summaries are differentiable; nothing raises.

- **Inverse cumulative hazard and quantiles** (implicit function theorem):
  `dt = (∂θ log S(t)·dθ + de) / h(t)`, evaluated with `jax.jvp` of
  `log_survival` at `stop_gradient(t)`. `dq/dp = 1/f(q)`.
- **Generic RMST and mean:** tangent `∫ ∂θ S·dθ dt` (plus `S(tau)·dtau` for
  RMST), evaluated on a **static** node set. The level chosen by the primal loop
  enters only as a `stop_gradient` mask, because reverse mode must transpose the
  tangent computation and a `while_loop` cannot be transposed. Forward, reverse
  and higher-order derivatives all work.
- **Closed forms without usable special-function derivatives** use the
  closed-form value plus the quadrature tangent via `custom_jvp`. This applies
  to Weibull RMST: `lax.igamma_grad_a` has no second-order rule, so `gammainc`
  has only first derivatives in its first argument. JAX's `betainc` has no
  derivatives in `a` or `b` at all, which is why log-logistic RMST uses
  quadrature. Review measurement: Weibull RMST gradients and full Hessians
  matched mpmath to ≤1e-14 over `k` in [0.2, 50] and `tau/b` in [0.01, 1e5].
- Entries that failed or did not converge get NaN tangents, never zeros.
  Endpoint quantiles and known-infinite means get NaN or infinite tangents.
- Derivative accuracy is limited by the quadrature tolerance and is documented
  method by method.

### 8.5 Controls and diagnostics

Summary methods take `options=cx.SummaryOptions(...)`, an immutable PyTree:

| Field | Kind | Default |
| --- | --- | --- |
| `rtol` | dynamic | `1e-8` (float64), `1e-4` (float32) |
| `atol` | dynamic, in time units | `1e-10*s` (float64), `1e-6*s` (float32) |
| `max_levels` | static | 8 |
| `max_solver_steps` | static | 64 |
| `max_bracket_steps` | static | 64 |

Work limits are static under compilation; convergence and status are per
broadcast entry. `SummaryInfo` contains a status (`IntEnum`: converged,
known infinite, invalid input, non-finite arithmetic, failed bracket, work
exhausted), an error estimate where meaningful, and work counts. Defaults are
starting values to be confirmed by measurement.

### 8.6 Sampling

Every family can sample:

- Draw `E = jax.random.exponential(key, sample_shape + batch_shape, dtype)` and
  return `inverse_cumulative_hazard(E)`. Built-ins use their closed forms of the
  same map, so a given key yields the same draws (to solver tolerance) on the
  generic and closed-form paths; tests compare them.
- **Truncation-conditioned sampling** (`left_trunc`, `right_trunc`, each
  broadcastable to `batch_shape`): with `ΔH = H(U) - H(L)`, map the same `E` to
  `E' = -log1p(-(-expm1(-E)) * (-expm1(-ΔH)))` and return
  `H⁻¹(H(L) + E')`. With `U=+inf`, `E'=E`. `Z=0` gives NaN.
- Draws are differentiable with respect to parameters (reparameterized) through
  the rule in section 8.4.
- Sampling never samples parameters or creates censoring. Draws are
  reproducible for a fixed key within a Chronox version, not across versions.

## 9. Nonparametric estimation and metrics

### 9.1 Common rules

- Data-input forms follow section 6.1.
- The observation axis is the **last** axis, and it must have the same length in
  data and predictions. It is never broadcast (a single individual broadcast
  across a cohort gives all-tied times or scores and is never meaningful).
  Leading axes broadcast normally and are never pooled.
- Supported observations are exact and right-censored rows; KM additionally
  supports left truncation. Other rows give NaN plus an `unsupported` diagnostic;
  rows are never dropped or replaced with representative times.
- Tie convention everywhere: at equal times, events precede censorings (a
  censored time means survival strictly beyond it).
- No useful gradient is promised except where stated (Brier score).
- Integer counts use exact arithmetic without a capacity cap: per-row counts in
  `int32`, totals in a two-word (hi/lo) accumulator; ratios are computed in the
  data's floating dtype.

### 9.2 Harrell's concordance

```text
concordance_index(scores, data=None, *, time=None, event=None) -> Array
concordance_index_with_info(...) -> (Array, ConcordanceInfo)
```

Scores are finite, caller-supplied; higher means earlier predicted event.
Chronox never chooses a prediction summary or averages draws. Each unordered
pair is counted once. Unequal times are comparable when the earlier one is
exact. Equal-time event/event and censor/censor pairs are excluded; an
equal-time event/censor pair is comparable, with the event earlier. A larger
score on the earlier event is concordant, a smaller one discordant, tied scores
get half credit:

```text
C = (concordant + 0.5 * tied_score) / comparable
```

No comparable pairs gives NaN. `ConcordanceInfo` holds concordant, discordant,
tied-score and comparable counts plus invalid, unsupported and
no-comparable-pair masks. Algorithm: a JAX loop comparing one row with the
cohort, O(N²) work and O(N) workspace per batch entry. Cross-check against R
`survival::concordance(..., reverse = TRUE, timewt = "n")`, documenting any
convention difference.

| Data | Scores | Result |
| --- | --- | --- |
| `(N,)` | `(N,)` | `()` |
| `(N,)` | `(D,N)` | `(D,)` |
| `(B,N)` | `(B,N)` | `(B,)` |
| `(B,1,N)` | `(D,N)` | `(B,D)` |
| `(N,)` | `(K,D,N)` | `(K,D)` |

### 9.3 Kaplan–Meier

```text
kaplan_meier(data=None, *, time=None, event=None, where=None)
    -> KaplanMeierEstimate
censoring_kaplan_meier(data=None, *, time=None, event=None, where=None)
    -> KaplanMeierEstimate
kaplan_meier_with_info(...) -> (KaplanMeierEstimate, KaplanMeierInfo)
```

- Pure functions returning an estimate object; there is no stateful fitter.
- Fixed-shape algorithm: sort event times (padded to `N`), count events `d_j`
  and the risk set `n_j = #(L < t_j) - #(exit < t_j)` with `searchsorted`, then
  cumulative sums of `log1p(-d_j/n_j)` and the Greenwood terms. Left truncation
  (delayed entry) only changes the risk-set count. Review prototype: matched a
  textbook loop to ≤7e-16 with and without delayed entry, and N=34,300 ran in
  about 25 ms after compiling.
- `where=` is a boolean row mask so that groups of unequal size can be vmapped.
  It is the only mask argument in v0.
- `censoring_kaplan_meier` estimates the censoring distribution `G` (censorings
  as events), with the tie convention built in: at a time where both occur,
  censorings are counted after events.
- `KaplanMeierEstimate` (a PyTree, not a `SurvivalDistribution`) offers:
  - `survival(t)` (right-continuous) and `survival_left_limit(t)`;
  - `variance(t)` (Greenwood) and `confidence_interval(t, level=0.95)` on the
    log scale, as R `survfit` does by default;
  - `rmst(tau)`, exact as a sum of rectangles;
  - `quantile(p)` and `median()`: the smallest `t` with `S(t) <= 1-p`, using R's
    midpoint rule when `S` equals `1-p` (within the square root of the dtype's
    machine epsilon) over an interval.
- Beyond the largest observed time, `survival`, `variance` and intervals are
  NaN, and `rmst(tau)` is NaN for such `tau`.
- Reference: R `survival::survfit` on committed synthetic data.

### 9.4 Brier score and integrated Brier score

```text
brier_score(predictions, data=None, *, eval_times, time=None, event=None,
            censoring=None) -> Array                  # (K, *batch)
integrated_brier_score(predictions, data=None, *, eval_times, time=None,
                       event=None, censoring=None) -> Array   # (*batch)
```

- `predictions` is a `SurvivalDistribution` with batch `(..., N)`, evaluated
  internally at `eval_times`, or an array of survival probabilities with shape
  `(K, ..., N)`, aligned with `eval_times` by the caller. Passing a distribution
  makes misalignment impossible. The dispatch is on the Python type and is fine
  under `jit`.
- Definition (Graf et al., 1999), with censoring survival `G`:
  `BS(t) = (1/N) Σ_i [ S_i(t)² · 1{T_i<=t, exact_i} / G(T_i−) + (1−S_i(t))² · 1{T_i>t} / G(t) ]`.
  Rows censored at or before `t` contribute zero but stay in `N`.
- IBS integrates `BS` over `eval_times` with the trapezoid rule and divides by
  the grid's range.
- `censoring`: `None` uses `censoring_kaplan_meier` of the evaluation data; a
  `KaplanMeierEstimate` (for example, from training data) is evaluated as
  needed; a `CensoringWeights` container supplies per-row values `G_i(T_i−)` and
  `G_i(t_k)` computed by the caller, for covariate-dependent censoring models.
- An evaluation time where a needed `G` value is zero gives NaN.
- The Brier score is differentiable with respect to the predicted
  probabilities, with the censoring weights treated as constants, so it may
  serve as a training loss.
- Reference: R `riskRegression::Score` with a Kaplan–Meier censoring model, on
  committed synthetic data. The Chronox definition is normative; differences in
  another implementation's conventions are documented, not copied.

### 9.5 Cumulative/dynamic time-dependent AUC

```text
cumulative_dynamic_auc(markers, data=None, *, eval_times, time=None,
                       event=None, censoring=None) -> Array   # (K, *batch)
```

- `markers`: a distribution (the marker is the time-specific `1-S_i(t)`), an
  array `(K, ..., N)` of time-specific markers, or an array `(..., N)` of one
  risk score per individual. Higher means higher risk, as for concordance.
- Definition (Uno et al., 2007; Hung and Chiang, 2010): cases are
  `T_i <= t` with an exact event, weighted by `1/G(T_i−)`; controls are
  `T_j > t`. `AUC(t)` is the weighted proportion of case–control pairs with the
  case's marker larger, tied markers counting one half.
- Algorithm: a weighted Mann–Whitney comparison by sorting, O(K·N log N) with
  fixed shapes.
- `censoring` behaves as in section 9.4. No cases or no controls gives NaN.
- Reference: R `timeROC` with marginal (Kaplan–Meier) weighting, on committed
  synthetic data.

## 10. Engineering and distribution

### 10.1 Build, quality tools and license

Use **uv** for environments and dependencies and **uv_build** as the build
backend. Use Ruff for formatting and linting with rule families `E`, `W`, `F`,
`I`, `B`, `C4` and `UP`, ignoring `E501`. Pre-commit follows the TorchSurv
approach: Ruff safe fixes and formatting; end-of-file and trailing-whitespace
checks; YAML/TOML validation; case-conflict, large-file, debug-statement and
private-key checks; and a local mypy hook through uv against the core source.
Keep hook and CI settings consistent. No project-specific TorchSurv hooks or
automatic spelling edits.

Chronox's original code is Apache-2.0. Add the license text and matching
metadata during packaging, record actual copyright ownership, and preserve
notices for reused material. This does not license third-party code or data.
Citation and contribution guidance belong to release preparation.

### 10.2 Package layout and public exports

Distribution metadata `name = "chronox-jax"`, `src/chronox_jax` layout, explicit
public exports. Document installation as `uv add chronox-jax` or
`pip install chronox-jax` and import as `import chronox_jax as cx`. No top-level
`chronox` import is provided (another distribution owns that namespace).

```text
pyproject.toml, uv.lock, .python-version, .gitignore, LICENSE,
.pre-commit-config.yaml
src/chronox_jax/
    __init__.py
    data.py
    diagnostics.py
    likelihood.py
    metrics.py              # concordance, Brier, IBS, AUC
    nonparametric.py        # Kaplan–Meier
    testing.py              # check_family
    _numerics.py
    _quadrature.py
    _solvers.py
    distributions/
        __init__.py
        base.py             # SurvivalDistribution, register_family, param/static
        hazard.py           # HazardFamily
        wrappers.py         # ProportionalHazards, AcceleratedFailureTime
        exponential.py
        weibull.py
        lognormal.py
        loglogistic.py
    py.typed
tests/
examples/
docs/
```

Create modules when their slice needs them. Re-export the data object,
distributions, wrappers, likelihood, KM and metric entry points at top level;
result and configuration types stay discoverable in their modules; numerical
helpers stay private. Initial version `0.0.1.dev0`; no custom build hooks; no
compiled extensions.

**Compatibility policy.** 0.0.x releases make no stability promises. The first
things intended to stabilize are constructor keywords, the entry-point
signatures of section 6.1 and result shapes.

### 10.3 Environment, dependencies and verification tools

Python 3.12 (`requires-python >=3.12`), a locked CPU environment and local Git,
initialized at implementation start with the design history as the first
snapshot. Select mutually compatible versions during setup, record them in the
lockfile and justify lower bounds through checks. Do not advertise an untested
Python/JAX matrix.

| Dependency group | Contents |
| --- | --- |
| Runtime | `jax`, `numpy` (host inspection and fixed constants only; never convert tracers) |
| Tests | pytest, mpmath, hypothesis; optionally quadax as a quadrature cross-check |
| Development | Ruff, mypy, pre-commit, uv_build |
| Classical demo | Optimistix, Matplotlib |
| Neural demo | Equinox, Optax, Matplotlib, and only the loaders actually used |
| Bayesian demo | NumPyro, Matplotlib; diagnostics tooling only if needed |
| Reference generation | a separate pinned R environment: R 4.4.2, flexsurv 2.3.2, survival 3.7-0, plus timeROC and riskRegression at versions pinned when fixtures are generated |

Demo dependencies live in development groups; core code imports and runs
without NumPyro, Equinox, Optax, Optimistix, R or plotting libraries.

Annotate public APIs, PyTree fields, result containers and extension interfaces.
Run mypy on core source with checks on typed function bodies; keep typing
exceptions focused. No shape-typing library or runtime type decorator. Include
`py.typed`. Type-check an installed out-of-tree custom-family example.

Run float32 and float64 tests in separate processes with 64-bit mode set by the
runner. Smoke-test wheel installation and imports outside the source tree in
Slice A; check wheel and source archive before release. Add one Linux CPU CI job
when a remote exists. **Regular tests and CI never need network, R or locally
acquired data**: every fixture they use is committed. Coverage identifies gaps;
it is not a target.

### 10.4 Data and fixture handling

No patient rows or row-level outputs derived from patient data are committed.
Instead:

- **Synthetic fixtures (committed).** Synthetic datasets are simulated with a
  recorded seed, and R (flexsurv, survival, timeROC, riskRegression) evaluates
  per-row log likelihoods, survival values, KM curves and metric values at fixed
  parameters. The synthetic data and the R outputs are committed together with
  the generating scripts. R is needed only to regenerate them.
- **Real datasets (local only).** `bc` and GBSG2 are acquired locally with
  documented commands for the demos. A provenance note records source, version,
  acquisition command, transformations, hashes and distribution status. Preserve
  `bc` row order, event coding, units and group contrasts.
- Aggregate numbers already published in the flexsurv vignette may be recorded
  in the demo.

The R audit scripts and their aggregate results are reference evidence, not
proof of Chronox correctness or permission to redistribute data.

## 11. Reference analysis and demonstrations

### 11.1 Classical reference: breast-cancer Weibull AFT

Reproduce the `fs1` Weibull AFT analysis of `flexsurv::bc` from the flexsurv
paper and vignette, with an R `survreg` cross-check, fitting with Optimistix
BFGS in float64. This is the **classical demo**, run on demand with an explicit
command and never in CI (Slice C).

| Item | Definition |
| --- | --- |
| R environment | R 4.4.2, flexsurv 2.3.2, survival 3.7-0, macOS arm64 (audited) |
| Rows | all 686 (the German Breast Cancer Study Group cohort, 1984–89) |
| Time | `recyrs = rectime/365` years; observed `rectime` spans 8–2659 days |
| Event | `censrec==1` exact (299); `censrec==0` right-censored (387) |
| Groups | Good, Medium, Poor: 229, 229, 228 rows; 51, 103, 145 events |
| Design | intercept, Medium and Poor indicators; Good is baseline |
| Other | no weights, truncation, penalty, standardization or other covariates |

For row `i`: `k=exp(theta[0])`, `b_i=exp(X_i beta)`,
`theta=(log k, beta_intercept, beta_Medium, beta_Poor)`,
`S_i(t)=exp(-(t/b_i)^k)`. Minimize the negative summed log likelihood:

```text
event:    log(k) - log(b_i) + (k-1)*(log(t_i)-log(b_i)) - (t_i/b_i)^k
censored:                                                        - (t_i/b_i)^k
```

| Quantity | R audit value |
| --- | --- |
| Shape k | 1.3796517717622814 |
| Good-group scale | 11.422861823436245 |
| beta_Medium | -0.6135891674792526 |
| beta_Poor | -1.2122137266943085 |
| Total fitted log likelihood | -811.9419065292103 |
| AIC (four parameters) | 1631.8838130584206 |
| Fixed `(1, 10, -0.5, -1)` total log likelihood | -840.8238842715208 |
| Fixed `(1.5, 12, -0.6, -1.2)` total log likelihood | -815.9893852271490 |

Demo checks, in order:

1. Totals at the two fixed vectors and at the recorded R optimum.
2. Curves at fixed parameters on a grid from 0 to 7 years in steps of 0.05
   (141 times, three groups, shape `(141,3)`); `S(0)=1`; survival at 1, 3 and 5
   years; Chronox and R curves plotted together.
3. The fit, from at least two BFGS starts, `(0, log 10, 0, 0)` and
   `(log 1.5, log 12, -0.6, -1.2)`. Record solver settings, versions and
   termination diagnostics.
4. The exact JAX Hessian at the solution: positive definiteness, conditioning,
   gradient.
5. **Independent optimum check:** Chronox's gradient of the summed objective
   evaluated at the recorded R optimum. Compare coefficients against a polished
   `survreg` reference (Newton at `rel.tolerance=1e-12`, recorded), so the
   comparison measures Chronox rather than R's stopping rule.
6. In the completed demo: KM curves per group overlaid on the fitted curves,
   group medians and five-year RMST.

| Check | Target |
| --- | --- |
| Fixed-parameter totals, float64 | absolute difference ≤ 1e-8 |
| Fitted total log likelihood | absolute difference ≤ 1e-6 |
| Fitted unconstrained coefficients vs polished reference | absolute difference ≤ 1e-5 |
| Max absolute component of final gradient; of Chronox gradient at R optimum | < 1e-5 |

Investigate failures; do not relax thresholds until a run passes. flexsurv and
survreg agreeing is not fully independent evidence, because flexsurv uses
survreg on this route.

**Per-row correctness in CI** comes from the synthetic fixtures of section 10.4:
a dataset simulated from the fitted `bc` model with the same size, group sizes,
time units and a similar censoring pattern, with per-row log likelihoods and
survival values from flexsurv and survreg at the three parameter vectors above.
Targets: `atol=1e-10`, `rtol=1e-8` in float64.

### 11.2 Neural demo

GBSG2, an Equinox tabular network producing individual Weibull log shape and
log scale (built with `Weibull.from_log`), Optax training on mean negative log
likelihood, and a linear model on the same features and split for comparison.
GBSG2 is **the same 686-patient cohort as `bc`**, used here for a different
task with its own covariates. Fixed train/validation/test partitions,
preprocessing fitted on training data only, and training settings and
checkpoints chosen by validation likelihood.

Report on held-out data: log likelihood; Harrell's C with a documented scalar
risk score; integrated Brier score; and `AUC(t)` at stated `eval_times` using
the time-specific marker `1-S_i(t)`. Censoring weights come from
`censoring_kaplan_meier` of the training data. Show survival curves for selected
profiles, finite training gradients and a working functional training loop.
There is no requirement to beat the baseline or match another library's weights.
Record acquisition, preprocessing, seed, architecture, optimizer settings and
CPU environment.

### 11.3 Bayesian demo

The classical `bc` regression with explicit priors and NumPyro NUTS. The sum of
Chronox row contributions enters through `numpyro.factor` exactly once, verified
at fixed parameters. No event-time adapter.

Show prior predictive survival, posterior summaries and multi-chain diagnostics;
group survival curves with pointwise credible intervals; posterior distributions
of group medians and five-year RMST; event-time draws conditional on posterior
parameter draws; and a **posterior predictive check**: the observed KM per group
against KMs of replicated datasets, built with the simulation recipe's censoring
process. Evaluate every curve and summary per draw before aggregating.

Show posterior predictive sampling both ways: manually (posterior draws →
`cx.Weibull(...)` → `.sample(key)`), and inside `numpyro.infer.Predictive`
through `numpyro.deterministic("t_rep", dist.sample(numpyro.prng_key()))`,
behind a flag so it does not run during MCMC. Record priors, seeds, sampling
settings and diagnostic thresholds with the demo.

### 11.4 Extension and simulation recipes

- A custom family written outside the package with only `log_survival` and
  `reference_time`, exercising the autodiff density, generic summaries, generic
  sampling and `cx.testing.check_family`.
- A `HazardFamily` example (Gompertz).
- A PH-wrapper example on a non-Weibull baseline.
- A right-censoring simulation recipe: draw event times, draw or choose
  censoring times separately, form observed times and event flags; the same
  recipe builds replicated datasets for posterior predictive checks.

Plotting uses an existing library outside core code.

## 12. Verification and implementation milestones

### 12.1 Required verification

These are acceptance criteria, not claims of passing tests.

| Area | Required evidence |
| --- | --- |
| Construction | every indicator and endpoint rule; both constructors normalize identically; dtype policy (float64 under 32-bit mode warns and converts; explicit request fails); eager and traced construction; indexing and slicing; issue bitfield decoding; static metadata |
| Likelihood mathematics | hand-derived exact/right/left/interval cases under absent/left/right/two-sided truncation; mixed data; partial, full and empty intersections; censored-before-entry and uninformative rows; a bounded-support family separating zero numerator from zero denominator |
| Cross-kind identities (property-based) | interval `(c, +inf]` equals right-censored at `c`; `(0, c]` equals left-censored at `c`; `logaddexp` of adjacent intervals equals their union; full coverage gives exactly zero |
| Numerical behaviour | tiny nonzero masses, both tails, narrow intervals, cancellation, boundaries, inactive branches; no floors; valid `-inf` versus invalid NaN; the likelihood suite run eagerly under `jax_debug_nans` on valid mixed batches |
| Boundaries | survival grid from `t=0` with Weibull `k=1`; gradient of `S(0)` in `k`; log-normal tails against mpmath over `x` in [0,10] (float64) and [0,7] (float32) |
| Transformations | eager/`jit`/`vmap` agreement; `jax.test_util.check_grads(order=2, modes=("fwd","rev"))` on interior cases for families and wrappers; reference-input gradient isolation; compile-count tests for static metadata |
| Independent references | analytical cases, mpmath, committed R fixtures (flexsurv and survreg for all observation kinds and truncation; survfit; concordance; timeROC; riskRegression) |
| Shapes | scalar, observation, draw batches, multiple axes; time-first grids; sample-shape prefixes; metric observation-axis matching |
| Families and wrappers | PH/AFT identities with Weibull and exponential; `check_family` on every built-in |
| Summaries | analytical values; generic fallbacks against known answers; heavy tails, bounded support, known infinite means, work exhaustion, reported failure; derivative rules against finite differences and mpmath |
| Sampling | key reproducibility; generic versus closed-form agreement on the same key; truncated draws inside `(L,U]` with correct distribution (statistical tests with controlled seeds); reparameterized gradients |
| KM and metrics | hand-enumerated cases; all tie cases; delayed entry; `where` groups under `vmap`; beyond-follow-up NaN; unsupported rows; `G=0`; distribution versus array prediction inputs |
| Extension and install | out-of-tree custom family; installed wheel and source archive; public exports and type marker; core imports without demo dependencies |
| End-to-end | classical demo targets; neural training and held-out metrics; Bayesian wiring, diagnostics and posterior predictive check; recorded environments |

Ordinary finite-value targets: `atol=1e-10, rtol=1e-8` (float64) and
`atol=1e-5, rtol=1e-4` (float32) on representative domains. Derivative and tail
tests need separately justified scales; compare log probabilities when
probabilities underflow. Stochastic tests use controlled seeds and statistical
bounds.

### 12.2 Staged delivery

| Slice | Deliverable | Completion gate |
| --- | --- | --- |
| A | Installable package; `SurvivalData` (both constructors, all v0 kinds representable, dtype policy, issue bitfield, indexing, static metadata) and `DataIssues`; exponential and Weibull exact/right likelihoods without truncation; direct-method policy; survival grids; core extension interface (`register_family`, `log_survival` + `reference_time`, autodiff `log_prob` default); result containers | Construction, evaluation and shape checks; first and second derivatives; stable tails; boundary cases; independent math fixtures; committed synthetic `bc`-like per-row R fixtures; eager `jax_debug_nans` run; wheel smoke test; a tiny synthetic example |
| B | Left and interval censoring; all truncation combinations; `log_conditional_interval_prob` hook; static-metadata pruning | Full likelihood matrix and diagnostics; cross-kind property tests; synthetic flexsurv fixtures for every kind; transformed derivatives |
| C | Classical demo: external BFGS fit and `bc` reproduction, with analytical summaries as needed | The checks and targets of section 11.1, including the independent optimum check |
| D | Log-normal and log-logistic families; `HazardFamily`; PH and AFT wrappers; analytical and generic summaries (quadrature, solver, derivative rules); sampling including truncation; `check_family`; extension recipes | Summary convergence, failure and derivative checks; wrapper identities; sampling agreement and distribution tests |
| E | Concordance, Kaplan–Meier, censoring KM, Brier score, IBS, cumulative/dynamic AUC | Metric and KM verification rows of section 12.1 against committed R fixtures |
| F | Neural and Bayesian demos | Reproducibility, held-out metrics, sampling diagnostics, posterior predictive check, recorded environments |

Slice A is deliberately smaller than v0. Its documentation states the subset,
and unimplemented observation modes fail explicitly instead of producing
plausible values. The data representation accommodates every v0 kind from the
start. Later slices need their own evidence; passing A is not evidence of full
v0 support. Publication, remote setup and credentials are separate actions.

## 13. Open items and implementation work

**Separate thread:** JAX `log_ndtr` precision
([notes/log_ndtr-precision.md](notes/log_ndtr-precision.md)); the log-normal
requirement in section 4.1 already works around it.

**Settled in drafting revision 4, open to user override** (listed in the
disposition log): the treatment of full-coverage censored rows other than
censoring strictly before entry (valid, flagged `uninformative`); the
`_with_info` suffix and `SurvivalData.with_issues` names;
`LogNormal(meanlog, sdlog)` naming; the `CensoringWeights` container.

**Implementation work under the accepted contracts:** compatible version pins;
stable family formulas and reference times; the synthetic fixture generators and
their R environment (including timeROC and riskRegression versions); tested
numerical domains and derivative tolerances; summary option defaults confirmed
by measurement; solver, training and sampling settings recorded with results;
GBSG2 preparation; Bayesian priors and diagnostic thresholds.

**Before public release:** license and notices; fixture provenance;
package-name availability; supported-version matrix; metadata and changelog;
wheel and source archive contents; citation and contribution guidance.

## 14. Review status

Revision 3 received an independent review on 2026-09-24 ([claude-review.md](claude-review.md)).
Every finding, clarification and discussion point was settled on 2026-09-25;
[REVIEW-DISPOSITIONS.md](REVIEW-DISPOSITIONS.md) maps each to its decision and
to the sections above. Slice A may begin when the user instructs implementation.
Revision 5 resolves the right-censoring-at-entry conflict (`c=L` is valid
and uninformative; `c<L` is invalid) and assigns endpoint issue masks.
At entry, the right-censoring numerator equals the truncation denominator, so
the log contribution is zero. Endpoint issue masks distinguish missing or
out-of-domain values from incorrect interval order. R's counting-process
input rule for equal start and stop is separate from this likelihood identity.
Later revisions should record new decisions in the disposition log or in the
design notebook's decision records.

## 15. Decision traceability

Revision 4 changes are traced in [REVIEW-DISPOSITIONS.md](REVIEW-DISPOSITIONS.md).
The decision records behind revisions 1–3 (paths relative to `docs/design/`) remain
the historical basis for the parts of this document the review did not change:

| Area | Earlier decision records |
| --- | --- |
| Observation construction and storage | 0001, 0010, 0011, 0012, 0013 (representation superseded by revision 4, section 3.2) |
| Distributions, summaries, shapes | 0002, 0003, 0005, 0007, 0032 |
| Likelihood composition, computation and validation | 0006, 0008, 0014, 0015 (reference hook superseded), 0016, 0017, 0018, 0019, 0025 |
| Concordance and metric scope | 0020 (KM, AUC deferral superseded), 0021, 0022, 0023 |
| v0 boundary and fallbacks | 0027, 0028 (algorithms superseded), 0029 |
| Fitting, plotting and demonstrations | 0024, 0026, 0030, 0031 |
| Tooling, license, section 6 handoff, naming | 0033, 0034, 0035, 0036 |
| Accepted future designs excluded from v0 | NumPyro adapter 0004; outside-interval censoring 0009 |
