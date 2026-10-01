---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Describing survival observations

```{note}
Chronox can currently build and check survival observations with
`SurvivalData`. Likelihoods, distributions and evaluation are not implemented
yet, so this page stops once the data are described.
```

This page follows a small synthetic cohort of six patients. Time is months
since diagnosis and the event is disease progression. Each section adds
something we know (or don't know) about a patient and shows how to write it
down.

## Time and status, as in R

For the first four patients, follow-up went as usual. Patients 1 and 3
progressed at months 4 and 7. Patients 2 and 4 had not progressed when we
last saw them at months 12 and 9. Their event time is only known to be later
than that: they are *right censored*.

As in R's `Surv(time, status)`, pass the times and a status of 1 for an event
and 0 for censoring.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import chronox_jax as cx

data = cx.SurvivalData([4.0, 12.0, 7.0, 9.0], [1, 0, 1, 0])
data
```

Chronox stores every observation as a set of possible event times. An exact
event has `exact=True` and `lower == upper`. A censored row says the event
happened after `lower` and at or before `upper`; right censoring has
`upper=inf`. The `left_trunc` and `right_trunc` fields are covered in the
truncation section below. Here they are at their defaults of 0 and `inf`.
Without a status, every row is an exact event.

## The event happened between two visits

Patient 5 had a clear scan at month 6 and showed progression at month 9.
Progression happened sometime after month 6 and no later than month 9: the
observation is *interval censored*.

A single status column cannot say this, so use keyword flags instead. A row
with every flag false is an exact event. `time` holds the start of the
interval and `time2` its end. `time2` is read only on interval-censored rows,
so the other rows can hold anything, here `nan`.

```{code-cell} ipython3
nan = jnp.nan
data = cx.SurvivalData(
    [4.0, 12.0, 7.0, 9.0, 6.0],
    right_censored=[False, True, False, True, False],
    interval_censored=[False, False, False, False, True],
    time2=[nan, nan, nan, nan, 9.0],
)
data.lower, data.upper
```

The interval row stores `lower=6`, `upper=9`. Like R, Chronox treats the
interval as open on the left and closed on the right, `(6, 9]`.

## Known only to have occurred before the first visit

Patient 6 already showed progression at the first scan at month 3. All we
know is that progression happened at or before month 3: the observation is
*left censored*. Its stored lower bound is 0.

```{code-cell} ipython3
cohort = cx.SurvivalData(
    [4.0, 12.0, 7.0, 9.0, 6.0, 3.0],
    right_censored=[False, True, False, True, False, False],
    left_censored=[False, False, False, False, False, True],
    interval_censored=[False, False, False, False, True, False],
    time2=[nan, nan, nan, nan, 9.0, nan],
)
cohort.lower, cohort.upper
```

At most one flag may be true on a row. Combining `event` with flags in one
call is a mistake in the call itself. It raises at once:

```{code-cell} ipython3
:tags: [raises-exception]

cx.SurvivalData([4.0], [1], left_censored=[False])
```

## The same cohort as endpoints

If your data already have lower and upper endpoints for each patient,
`SurvivalData.from_bounds` takes them directly. A censored row means
`lower < T <= upper`. Use `inf` for no upper limit and 0 for no lower limit.

Exact events need the `exact` mask. Their upper endpoint may repeat the
lower one or be `nan`.

```{code-cell} ipython3
inf = jnp.inf
same_cohort = cx.SurvivalData.from_bounds(
    [4.0, 12.0, 7.0, 9.0, 6.0, 0.0],
    [nan, inf, nan, inf, 9.0, 3.0],
    exact=[True, False, True, False, False, False],
)
same_cohort.upper
```

Equal endpoints alone never mean an exact event. Without `exact=True` they
are flagged as an invalid interval, as shown in the section on finding bad
rows.

## Patients who entered late (truncation)

Patients 5 and 6 were diagnosed elsewhere and joined the study two months
after diagnosis. Had they progressed before month 2, we would never have met
them. Their data are *left truncated* at 2: we only see event times after 2.

Pass the entry times as `left_trunc`. If a study only admits patients who
progressed by some cutoff, pass that cutoff as `right_trunc`. Both may vary
by row. Each row is then only informative about times in
`(left_trunc, right_trunc]`.

```{code-cell} ipython3
cohort = cx.SurvivalData(
    [4.0, 12.0, 7.0, 9.0, 6.0, 3.0],
    right_censored=[False, True, False, True, False, False],
    left_censored=[False, False, False, False, False, True],
    interval_censored=[False, False, False, False, True, False],
    time2=[nan, nan, nan, nan, 9.0, nan],
    left_trunc=[0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
)
cohort.left_trunc
```

## Finding bad rows

Data entry goes wrong. Each of these five new records has a typical problem:

1. an exact event at month 0 (exact times must be after 0);
2. an interval with no end time;
3. an interval that ends at month 6 but starts at month 9;
4. a patient last seen at month 1 who entered the study at month 2;
5. an interval `(0, inf]` that covers every possible time.

Construction does not stop on bad values. It returns the data and records
each problem in `data.issues`, with one boolean mask per rule. Each mask has
one entry per row. `invalid_rows` combines them:

```{code-cell} ipython3
bad = cx.SurvivalData(
    [0.0, 6.0, 9.0, 1.0, 0.0],
    right_censored=[False, False, False, True, False],
    interval_censored=[False, True, True, False, True],
    time2=[nan, nan, 6.0, nan, inf],
    left_trunc=[0.0, 0.0, 0.0, 2.0, 0.0],
)
bad.issues.invalid_rows
```

Record 5 is not invalid. It rules nothing out, so it carries no information
about when progression happens. Chronox marks such rows as `uninformative`
and keeps them out of `invalid_rows`:

```{code-cell} ipython3
bad.issues.uninformative
```

`bad.issues.ok` is a single boolean for the whole data set. To stop on bad
data, call the explicit check. It names each rule that failed, how many rows
broke it and the first such row:

```{code-cell} ipython3
:tags: [raises-exception]

bad.issues.raise_if_invalid()
```

Two more mistakes are worth knowing. R also accepts a status coded 1/2, with
2 for an event. Chronox accepts only 0/1 or booleans and flags a `2` as an
invalid indicator rather than guessing:

```{code-cell} ipython3
cx.SurvivalData([5.0, 8.0], [2, 1]).issues.invalid_indicator
```

And equal endpoints given to `from_bounds` without the `exact` mask are an
invalid interval, not an event:

```{code-cell} ipython3
cx.SurvivalData.from_bounds([5.0], [5.0]).issues.invalid_interval
```

## Inside `jax.jit`

Construction and `data.issues` also work inside compiled code. The check
that raises needs concrete values, so return the issues from the compiled
function and check them afterwards:

```{code-cell} ipython3
@jax.jit
def diagnose(time, status):
    return cx.SurvivalData(time, status).issues

compiled_issues = diagnose(jnp.array([4.0, 12.0]), jnp.array([1, 2]))
compiled_issues.invalid_rows
```

```{code-cell} ipython3
:tags: [raises-exception]

compiled_issues.raise_if_invalid()
```

Calling `raise_if_invalid()` inside the compiled function raises a
`ValueError`, because the values are not known there yet.

## Floating-point precision

Times and bounds share one floating dtype. Chronox follows JAX's setting: the
default is float32, and float64 when JAX's 64-bit mode is on. The library
never changes that setting itself. This page was built with:

```{code-cell} ipython3
cohort.lower.dtype
```

You can pass `dtype=` to either constructor. Asking for float64 while 64-bit
mode is off raises a `ValueError`.

## Comparison with R's `survival::Surv`

The R calls below follow the official
[`Surv` help page](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html).
They compare how the input is written. No numerical agreement with R has
been checked.

| Situation | R | Chronox | Notes |
| --- | --- | --- | --- |
| Events and right censoring | `Surv(time, status)` | `SurvivalData(time, status)` | R also accepts 1/2 coding and guesses it from `max(status) == 2`. Chronox accepts only 0/1 or booleans. |
| Mixed censoring with codes | `Surv(time, time2, status, type = "interval")` | `SurvivalData(time, right_censored=..., left_censored=..., interval_censored=..., time2=...)` | R codes 0 right, 1 event, 2 left, 3 interval and ignores `time2` unless the code is 3. Chronox uses one boolean flag per kind and likewise reads `time2` only on interval rows. |
| Intervals as endpoints | `Surv(lower, upper, type = "interval2")` | `SurvivalData.from_bounds(lower, upper, exact=...)` | R treats equal endpoints as an exact event and allows `NA` or `Inf` for an unbounded end. Chronox needs the `exact` mask, uses 0 for no lower limit and `inf` for no upper limit. |
| Interval endpoints | `(start, end]` | `(lower, upper]` | Both are open on the left and closed on the right. |
| Delayed entry | `Surv(start, stop, status)` | `SurvivalData(stop, status, left_trunc=start)` | With `time2`, R defaults to counting-process data. Chronox's `time2` is never a stop time; entry times go in `left_trunc`. |
