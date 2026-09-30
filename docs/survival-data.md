---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Constructing survival observations

`chronox_jax.SurvivalData` currently constructs and diagnoses observations.
The normalized fields have the observation shape: `exact` (boolean), `lower`,
`upper`, `left_trunc` and `right_trunc`. Exact events store the time in both
`lower` and `upper`; right censoring stores `upper=+inf`; left censoring stores
`lower=0`. Missing truncation bounds become `0` and `+inf`. The first time
argument fixes the shape; other inputs may broadcast to it without enlarging
it. Times and bounds share a floating dtype, at least float32.

## Event status and explicit flags

The second positional argument is event status: `1` or `True` means an exact
event, and `0` or `False` means right censoring. Omitting status and all flags
marks every row exact. Explicit flags allow all four kinds in one array. A row
with no flag is exact; at most one flag may be true on a row. `time2` supplies
only an interval row's upper endpoint. Values on other rows are ignored.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import chronox_jax as cx

status = cx.SurvivalData([2.0, 3.0], [1, 0])
assert status.exact.tolist() == [True, False]
assert status.lower.tolist() == [2.0, 3.0]
assert status.upper[0] == 2.0 and jnp.isposinf(status.upper[1])
assert bool(status.issues.ok)

mixed = cx.SurvivalData(
    [2.0, 3.0, 4.0, 5.0],
    right_censored=[False, True, False, False],
    left_censored=[False, False, True, False],
    interval_censored=[False, False, False, True],
    time2=[jnp.nan, jnp.nan, jnp.nan, 7.0],
)
assert mixed.exact.tolist() == [True, False, False, False]
assert mixed.lower.tolist() == [2.0, 3.0, 0.0, 5.0]
assert mixed.upper[0] == 2.0 and jnp.isposinf(mixed.upper[1])
assert mixed.upper[2:].tolist() == [4.0, 7.0]
assert bool(mixed.issues.ok)
```

## Bounds and truncation

`from_bounds` describes a censored row as `lower < T <= upper`. Use `+inf` for
an unbounded upper endpoint and `0` for a left-censored lower endpoint. Mark
exact events explicitly with `exact=True`: their supplied upper endpoint must
equal lower or be NaN. A NaN exact upper normalizes to lower. Equal endpoints
*without* the exact flag are invalid, never reclassified as an event.

`left_trunc` and `right_trunc` encode the eligibility window `(L,U]` and may
vary by row. The normalized defaults are `L=0`, `U=+inf`. Exact times must be
finite and strictly positive; censored lower endpoints may be zero.

```{code-cell} ipython3
bounds = cx.SurvivalData.from_bounds(
    [2.0, 3.0, 0.0, 5.0],
    [jnp.nan, jnp.inf, 4.0, 7.0],
    exact=[True, False, False, False],
    left_trunc=[0.0, 1.0, 0.0, 1.0],
    right_trunc=[jnp.inf, jnp.inf, 6.0, 9.0],
)
assert bounds.exact.tolist() == [True, False, False, False]
assert bounds.lower.tolist() == [2.0, 3.0, 0.0, 5.0]
assert bounds.upper[0] == 2.0 and jnp.isposinf(bounds.upper[1])
assert bounds.upper[2:].tolist() == [4.0, 7.0]
assert bounds.left_trunc.tolist() == [0.0, 1.0, 0.0, 1.0]
assert bounds.right_trunc[2:].tolist() == [6.0, 9.0]
assert bool(bounds.issues.ok)
```

## Diagnosing data

Both constructors return data even when observation *values* are invalid.
`data.issues` exposes per-row masks for invalid indicators, conflicting flags,
missing or invalid time values, invalid intervals or truncation, an exact event
outside truncation, censoring before entry, and an empty observation window.
`invalid_rows` combines fatal masks; `ok` is a scalar reduction. `uninformative`
is separate and nonfatal: a censored set that covers the whole eligibility
window contributes no information about the event-time law.

```{code-cell} ipython3
problem = cx.SurvivalData.from_bounds([0.0, 2.0], [jnp.inf, 2.0])
assert problem.issues.uninformative.tolist() == [True, False]
assert problem.issues.invalid_interval.tolist() == [False, True]
assert problem.issues.invalid_rows.tolist() == [False, True]
assert not bool(problem.issues.ok)

try:
    problem.issues.raise_if_invalid()  # explicit host checkpoint
except cx.InvalidDataError as error:
    assert "invalid_interval" in str(error)
else:
    raise AssertionError("expected an invalid-data error")
```

Malformed *calls* raise immediately: event status cannot be combined with
censoring flags, unsupported field types and incompatible shapes raise
`TypeError`, and an explicit unavailable float64 dtype raises `ValueError`.
The library follows JAX's configured dtype mode; it does not toggle it.

```{code-cell} ipython3
try:
    cx.SurvivalData([1.0], [1], left_censored=[False])
except TypeError:
    pass
else:
    raise AssertionError("expected a call error")

expected_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
assert status.lower.dtype == expected_dtype
if not jax.config.jax_enable_x64:
    try:
        cx.SurvivalData([1.0], dtype=jnp.float64)
    except ValueError:
        pass
    else:
        raise AssertionError("expected an unavailable-dtype error")
```

Ordinary construction and diagnostic access also work inside `jax.jit`.
Return the diagnostic container from compiled code, then raise at a host
checkpoint if needed. Calling `raise_if_invalid()` on traced data is a
host-only misuse and raises `ValueError`.

```{code-cell} ipython3
@jax.jit
def diagnose(time, event):
    return cx.SurvivalData(time, event).issues

compiled_issues = diagnose(jnp.array([1.0, 2.0]), jnp.array([1, 2]))
assert compiled_issues.invalid_indicator.tolist() == [False, True]
assert compiled_issues.invalid_rows.tolist() == [False, True]
assert not bool(compiled_issues.ok)
try:
    compiled_issues.raise_if_invalid()
except cx.InvalidDataError:
    pass
else:
    raise AssertionError("expected an invalid-data error")
```

## Relation to R's `survival::Surv`

The conventions here are compared with the official
[`Surv` documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html).
R's `Surv(time, status)` usually uses 0 for censoring and 1 for an event, but
also accepts 1/2 coding; Chronox requires exactly 0/1 or booleans. R's
`type="interval"` uses status codes 0/right, 1/exact, 2/left and 3/interval;
Chronox uses separate boolean flags. In both systems, interval endpoints are
open on the left and closed on the right. With `time2`, R defaults to a
counting-process `(start, stop]` representation; Chronox reads `time2` only for
interval-censored rows. R's `type="interval2"` infers exactness from equal
endpoints and accepts `NA` or `Inf` for an unbounded endpoint. Chronox
requires the explicit `exact` mask and `+inf` for an unbounded censored upper.
These are input-convention comparisons; no numerical agreement with R has
been checked here.

## Current scope

A1–A2 construction and diagnostics are implemented. A3 indexing and concrete
static metadata are planned. The representation accepts left and interval
censoring and truncation, but likelihood evaluation is not yet implemented;
later Slice A work adds exact and right-censored likelihoods, and Slice B adds
the other likelihood modes. Do not use these constructed observations as
likelihood results yet.
