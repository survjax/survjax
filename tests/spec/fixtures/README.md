# Committed reference fixtures for the spec tests

The spec tester does not generate fixtures (spec 10.4: R evaluates them; the
maintainer commits them). Spec 11.1 (rev 34) and 10.4 (rev 59, Slice B
likelihood fixture): "The spec tester defines the file name and schema; the
generator writes that schema." This file is that definition: the **file
names and schema the spec tests read** (Q-039 and Q-073 in
`../API_ASSUMPTIONS.md`).

Until a file listed here exists, every test that reads it **fails** (it never
skips or passes without reference values).

## `bc_synthetic_weibull.json` (spec 10.4, 11.1, 12.2)

The Slice A "committed synthetic `bc`-like per-row R fixture": a dataset
simulated from the fitted `bc` Weibull model with the same size, group sizes,
time units and a similar censoring pattern (11.1), with per-row log
likelihoods and survival values from flexsurv and from survreg at the three
parameter vectors of section 11.1.

All numbers are JSON numbers written with full float64 precision (for
example R `format(x, digits = 17)` or `jsonlite::toJSON(digits = NA)`).

```json
{
  "meta": {
    "seed": 20261022,
    "generator": "path/to/generator.R",
    "R": "4.4.2",
    "flexsurv": "2.3.2",
    "survival": "3.7-0"
  },
  "data": {
    "time":  [0.83, 2.41, ...],
    "event": [1, 0, ...],
    "group": ["Good", "Medium", ...]
  },
  "parameters": {
    "fixed_a":   {"shape": 1.0, "good_scale": 10.0, "beta_medium": -0.5, "beta_poor": -1.0},
    "fixed_b":   {"shape": 1.5, "good_scale": 12.0, "beta_medium": -0.6, "beta_poor": -1.2},
    "r_optimum": {"shape": 1.3796517717622814, "good_scale": 11.422861823436245,
                  "beta_medium": -0.6135891674792526, "beta_poor": -1.2122137266943085}
  },
  "results": {
    "fixed_a": {
      "flexsurv": {"loglik": [...], "survival": [...]},
      "survreg":  {"loglik": [...], "survival": [...]}
    },
    "fixed_b":   {"...": "as fixed_a"},
    "r_optimum": {"...": "as fixed_a"}
  },
  "grid": {
    "time": [0.0, 0.05, ..., 7.0],
    "survival": {
      "fixed_a":   [[S_Good(t0), S_Medium(t0), S_Poor(t0)], ...],
      "fixed_b":   [[...], ...],
      "r_optimum": [[...], ...]
    }
  }
}
```

Field meanings:

- `data.time`: event or censoring time in **years** (`rectime/365` units),
  686 positive finite values. `data.event`: `1` exact, `0` right-censored.
  `data.group`: `"Good"`, `"Medium"` or `"Poor"`, with 229, 229 and 228 rows.
  No truncation, weights or other covariates.
- `parameters`: the three vectors of section 11.1 in the form
  `(k, Good-group scale, beta_Medium, beta_Poor)`. Row `i` has
  `b_i = good_scale * exp(beta_g(i))` with `beta_Good = 0`, and
  `S_i(t) = exp(-(t/b_i)^k)`.
- `results.<vector>.<source>.loglik`: per-row log-likelihood contribution on
  the **time scale** (years), 686 values in row order: `log f_i(t_i)` for
  `event=1` and `log S_i(t_i)` for `event=0`. `flexsurv` is from
  `flexsurvreg(..., dist = "weibull")` evaluated at fixed parameters
  (`fixed = ...` or `inits` with no iterations, or `dweibull`/`pweibull`
  via flexsurv's own functions); `survreg` is from the `survreg` Weibull
  parameterisation (`intercept = log good_scale`, `scale = 1/k`) with the
  time-scale Jacobian included, so both sources give the same quantity.
- `results.<vector>.<source>.survival`: `S_i(t_i)` at each row's own time
  (11.1, rev 34), 686 values.
- `grid.time`: 141 times `0, 0.05, ..., 7` (11.1 check 2).
  `grid.survival.<vector>`: shape `(141, 3)`, columns Good, Medium, Poor,
  from flexsurv (`summary(..., type = "survival", t = grid.time)` at fixed
  parameters).

## `slice_b_likelihood.json` (spec 2, 10.4, 12.1, 12.2; rev 59; Q-073)

The Slice B likelihood fixture of spec 10.4: one synthetic dataset with a
recorded seed covering the ten cells where flexsurv's likelihood conventions
match spec section 2, and per-row log likelihoods from
`flexsurvreg(..., fixedpars = TRUE)` for Weibull, log-normal and
log-logistic, each at two or more fixed parameter vectors, without
covariates. Used by `test_observation_censoring_flexsurv_fixture.py`.

All numbers are JSON numbers written with full float64 precision, as for
`bc_synthetic_weibull.json`. JSON has no infinity: `null` stands for `+inf`
in `right_trunc` and for "absent" in `time2`.

```json
{
  "meta": {
    "seed": 20261110,
    "generator": "path/to/generator.R",
    "R": "4.4.2",
    "flexsurv": "2.3.2",
    "survival": "3.7-0",
    "call": "flexsurvreg(<Surv(...)> ~ 1, data = d, dist = <dist>, inits = <vector>, fixedpars = TRUE, rtrunc = <U>)"
  },
  "data": {
    "cell":        ["untruncated_exact", "untruncated_left", "right_truncated_interval", ...],
    "kind":        ["exact", "left", "interval", ...],
    "time":        [0.83, 1.20, 0.40, ...],
    "time2":       [null, null, 0.95, ...],
    "left_trunc":  [0.0, 0.0, 0.0, ...],
    "right_trunc": [null, null, 2.5, ...]
  },
  "families": {
    "weibull": {
      "flexsurv_dist": "weibull",
      "vectors": [
        {"name": "shape_one", "parameters": {"shape": 1.0, "scale": 2.0}, "loglik": [...]},
        {"name": "shape_1_5", "parameters": {"shape": 1.5, "scale": 1.7}, "loglik": [...]}
      ]
    },
    "lognormal": {
      "flexsurv_dist": "lnorm",
      "vectors": [{"name": "...", "parameters": {"meanlog": 0.3, "sdlog": 0.8}, "loglik": [...]}, "..."]
    },
    "loglogistic": {
      "flexsurv_dist": "llogis",
      "vectors": [{"name": "...", "parameters": {"shape": 2.0, "scale": 1.5}, "loglik": [...]}, "..."]
    }
  }
}
```

Field meanings:

- `data`: `N` rows, all arrays of length `N`, in row order. Every family and
  vector is evaluated on the same rows.
  - `kind`: `"exact"`, `"right"`, `"left"` or `"interval"`.
  - `time`: the exact time `t` (exact), the censoring time `c` (right: set
    `(c, +inf]`; left: set `(0, c]`), or the interval's lower end `a`
    (interval: set `(a, b]`). Positive and finite.
  - `time2`: the interval's upper end `b` (finite, `b > a`); `null` on every
    other row.
  - `left_trunc`: `L`, finite, `0` when there is no left truncation.
  - `right_trunc`: `U`, a finite number, or `null` for `+inf` (no right
    truncation). Every row has `L < U`.
  - `cell`: the row's cell, one of the ten names below; the test checks that
    `kind` and the window agree with it.
- `families.<family>`: exactly `weibull`, `lognormal` and `loglogistic`.
  `flexsurv_dist` records the `dist` argument used (documentation only).
  `vectors`: two or more objects with a unique `name`, the natural
  parameters in survjax's names (spec 4.1: Weibull `shape`, `scale`;
  log-normal `meanlog`, `sdlog`; log-logistic `shape`, `scale`, matching
  flexsurv's `weibull`, `lnorm` and `llogis` parameters one to one), and
  `loglik`: `N` per-row log-likelihood contributions (for example
  `fit$logliki`) at those fixed parameters, on the time scale (densities
  include no Jacobian of any transformation). At least one Weibull vector
  has `shape` exactly `1`.

Cells (spec 10.4, rev 59); every cell has at least one row and no other row
is allowed:

| `cell` | window | rows |
| --- | --- | --- |
| `untruncated_exact` | `L=0`, `U=null` | exact |
| `untruncated_right` | `L=0`, `U=null` | right |
| `untruncated_left` | `L=0`, `U=null` | left |
| `untruncated_interval` | `L=0`, `U=null` | interval |
| `left_truncated_exact` | `L>0`, `U=null` | exact, `t>L` |
| `left_truncated_right` | `L>0`, `U=null` | right, `c>L` (not at entry) |
| `right_truncated_exact` | `L=0`, finite `U` | exact, `t<=U` |
| `right_truncated_left` | `L=0`, finite `U` | left, `c<U` |
| `right_truncated_interval` | `L=0`, finite `U` | interval, `b<=U` |
| `two_sided_exact` | `L>0`, finite `U` | exact, `L<t<=U` |

Generator constraints (10.4), checked by the integrity tests:

- Every finite `time`, `time2`, `left_trunc` and `right_trunc` has cumulative
  hazard `H = -log S` at most 15 under every vector of every family.
- `time2 - time >= 1e-4 * time2` on interval rows, and
  `right_trunc - left_trunc >= 1e-4 * right_trunc` on rows with finite `U`
  (Q-073 reads a two-sided window as a finite interval too).
- Values are finite; censored rows are at most `0`; and each value equals the
  spec 2 contribution (50-digit mpmath) within float64 `atol=1e-10`,
  `rtol=1e-8`.
- No derivatives, no survreg values, no covariates, no weights.

The cells 10.4 excludes (left or interval censoring with left truncation;
right-censored rows under a finite `U`; censored sets reaching beyond `U`;
rows censored at or before entry; observations outside the window; exact zero
times) must not appear: flexsurv defines them differently from section 2.

## `concordance_synthetic.json` (spec 9.2, 10.4, 12.1; step A10)

Synthetic cohorts with Harrell's C and pair counts from
`survival::concordance` (Q-046). Used by `test_a10_concordance_fixtures.py`.

```json
{
  "meta": {
    "seed": 20261023,
    "generator": "path/to/generator.R",
    "R": "4.4.2",
    "survival": "3.7-0",
    "call": "concordance(Surv(time, event) ~ score, reverse = TRUE, timewt = \"n\", timefix = FALSE)"
  },
  "cases": [
    {
      "name": "time_and_score_ties",
      "time":  [3.0, 1.0, 3.0, ...],
      "event": [1, 0, 1, ...],
      "scores": [[0.25, -1.5, ...], [1.0, 1.0, ...]],
      "r": [
        {"concordance": 0.6123, "concordant": 812, "discordant": 501,
         "tied_x": 17, "tied_y": 40, "tied_xy": 3},
        {"...": "one object per score vector, in order"}
      ]
    }
  ]
}
```

Field meanings:

- `meta.call`: the exact R call, with `reverse = TRUE` (higher score means
  earlier event, spec 9.2), `timewt = "n"` (Harrell's C) and
  `timefix = FALSE` (no merging of nearly equal times).
- `cases[].time`, `cases[].event`: one cohort of `N` rows, exact (`1`) or
  right-censored (`0`) only, finite positive times, no truncation, no
  weights, no strata. Write times with at most two decimals and scores
  with at most four significant digits, so float32 keeps every tie and
  every order.
- `cases[].scores`: `D` score vectors of length `N` for that cohort
  (`D >= 1`). `cases[].r[d]` is R's result for `scores[d]`: `concordance`
  is `fit$concordance`; the counts are `fit$count` columns `concordant`,
  `discordant`, `tied.x`, `tied.y`, `tied.xy` written as JSON integers.

Required coverage across the cases (checked by the integrity tests):
event/event, event/censor and censor/censor time ties; score ties;
including at least one cohort of 100–500 rows with heavy ties (for example
integer times from 1 to 20 and scores rounded to one decimal), one with
continuous times and no ties, one with more than 50% censoring, and one
with a score vector perfectly concordant and one perfectly reversed.

R's counts must agree with the spec 9.2 pair rules
(`test_fixture_counts_agree_with_the_spec_pair_rules`); if R treats a tie
differently, document the difference in the generator and in `meta`, and
keep that case out of the file.
