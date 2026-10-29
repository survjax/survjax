---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Harrell's concordance

Harrell's C measures how well a score orders event times when some
observations are right censored. A higher score must predict an earlier
event. survjax takes the scores you supply; it does not choose a prediction
summary, fit a model or average posterior draws. For example, a predictor
of log hazard has the right direction, while a predicted survival time
needs its sign reversed.

C describes ranking among comparable pairs. It does not assess calibration
or measure the size of a difference in survival time. No useful derivative
with respect to scores is promised, so it is an evaluation metric rather
than a differentiable training loss.

## A cohort with five comparable pairs

These four synthetic patients have follow-up times in months. Patients 1,
2 and 4 experienced the event; patient 3 was censored at month 2.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

import survjax

time = jnp.array([1.0, 2.0, 2.0, 3.0])
event = jnp.array([True, True, False, True])
scores = jnp.array([4.0, 3.0, 1.0, 4.0])
data = survjax.SurvivalData(time, event)
value, info = survjax.concordance_index_with_info(scores, data)
print(f"C = {float(value):.3f}")
```

Each unordered pair is considered once. When times differ, the pair is
comparable if the earlier observation is an event. When times are equal,
an event precedes a censoring: censoring at that time means the event
occurs strictly later. Equal-time event/event and censor/censor pairs are
excluded. A higher score on the earlier event is concordant; a lower one
is discordant. Equal scores on a comparable pair receive half credit.

| Patients | What is known | Score comparison | Count |
| --- | --- | --- | --- |
| 1, 2 | Event 1 is earlier | 4 > 3 | Concordant |
| 1, 3 | Event 1 precedes censoring 3 | 4 > 1 | Concordant |
| 1, 4 | Event 1 is earlier | 4 = 4 | Tied score |
| 2, 3 | Event 2 precedes censoring 3 at month 2 | 3 > 1 | Concordant |
| 2, 4 | Event 2 is earlier | 3 < 4 | Discordant |
| 3, 4 | Censoring 3 does not order the two events | — | Excluded |

There are three concordant, one discordant and one tied-score pair, giving

$$
C = \frac{\mathrm{concordant} + \tfrac12\mathrm{tied\_score}}
         {\mathrm{comparable}}
  = \frac{3 + 1/2}{5} = 0.7.
$$

`concordance_index` returns just C. Both entry points also accept raw time
and event arrays instead of prepared data; supply both keywords together.

```{code-cell} ipython3
print(survjax.concordance_index(scores, time=time, event=event))
```

## Counts and diagnostics

`info` contains four exact counts: `concordant`, `discordant`, `tied_score`
and `comparable`. Each uses two uint32 words in either dtype mode. On the
host, reconstruct a scalar count as `int(count.hi) * 2**32 + int(count.lo)`.
For arrays, retain the two words or convert individual entries to Python
integers; adding them in uint32 would lose the high word.

```{code-cell} ipython3
counts = tuple(
    int(count.hi) * 2**32 + int(count.lo)
    for count in (info.concordant, info.discordant, info.tied_score, info.comparable)
)
assert counts == (3, 1, 1, 5)  # The table above, independent of the implementation.
print("concordant, discordant, tied score, comparable:", counts)
```

The three boolean masks have one entry per result, rather than per
observation:

| Mask | Meaning |
| --- | --- |
| `invalid` | At least one score is nonfinite or an observation is invalid |
| `unsupported` | At least one observation uses an unsupported mode |
| `no_comparable_pairs` | The cohort was evaluated but has no comparable pairs |

An invalid or unsupported cohort gives NaN and zero counts. Its first two
masks may overlap; `no_comparable_pairs` remains false. A valid cohort with
no comparable pairs also gives NaN, with only `no_comparable_pairs` true.
Rows are never silently dropped to obtain a finite result.

## One result per score draw

The last axis is always the observation axis N. Its length must match in
data and scores, even if one side has length 1. Only leading axes broadcast;
each resulting cohort is evaluated separately, without pooling observations.
Scores do not have a prediction-time grid axis like `survival_grid` outputs.
Choose one score per patient before calling the metric.

| Data shape | Scores shape | Result shape |
| --- | --- | --- |
| `(N,)` | `(N,)` | `()` |
| `(N,)` | `(D,N)` | `(D,)` |
| `(B,N)` | `(B,N)` | `(B,)` |
| `(B,1,N)` | `(D,N)` | `(B,D)` |
| `(N,)` | `(K,D,N)` | `(K,D)` |

Here each row is a score draw for the same patients. Reversing the first
scores swaps concordant and discordant pairs; constant scores give half
credit to all five pairs. The returned values are 0.7, 0.3 and 0.5.

```{code-cell} ipython3
draws = jnp.stack([scores, -scores, jnp.ones_like(scores)])
values, draw_info = survjax.concordance_index_with_info(draws, data)
assert values.shape == draw_info.comparable.lo.shape == (3,)
assert not bool(jnp.any(draw_info.invalid | draw_info.unsupported))
print(values)
```

Choose how to summarize draws after evaluating them; the metric does not
average scores before counting. It also composes with ordinary construction
inside `jit`, and `vmap` can evaluate score rows explicitly:

```{code-cell} ipython3
def evaluate(s, t, e):
    return survjax.concordance_index(s, time=t, event=e)


compiled = jax.jit(evaluate)(draws, time, event)
mapped = jax.vmap(evaluate, in_axes=(0, None, None))(draws, time, event)
assert bool(jnp.array_equal(compiled, values))
assert bool(jnp.array_equal(mapped, values))
print("jit:", compiled, "vmap:", mapped)
```

Keep masks beside values when using batched results. A nonfinite score in
one draw invalidates that result, preserving the other draws:

```{code-cell} ipython3
bad_draws = draws.at[1, 0].set(jnp.nan)
bad_values, bad_info = survjax.concordance_index_with_info(bad_draws, data)
assert bool(jnp.array_equal(bad_info.invalid, jnp.array([False, True, False])))
assert int(bad_info.comparable.lo[1]) == int(bad_info.comparable.hi[1]) == 0
assert not bool(bad_info.no_comparable_pairs[1])
print(bad_values, bad_info.invalid)
```

## Supported observations and undefined results

Concordance currently supports exact events and right censoring without
truncation. `SurvivalData` can represent left/interval censoring and left/right
truncation too, but those modes make the whole metric cohort unsupported.
Check `data.issues` for observation validity; a valid observation may still
use an unsupported metric mode. Calling `data.issues.raise_if_invalid()` is
an optional host-side validation checkpoint, not a support check.

For example, these observations are valid but have delayed entry:

```{code-cell} ipython3
entered = survjax.SurvivalData(time, event, left_trunc=0.5)
unsupported_value, unsupported_info = survjax.concordance_index_with_info(scores, entered)
assert bool(entered.issues.ok)
assert bool(unsupported_info.unsupported) and not bool(unsupported_info.invalid)
assert bool(jnp.isnan(unsupported_value))
print(unsupported_value, unsupported_info.unsupported)
```

All-censored data have no comparable pairs. This is an evaluated, undefined
result rather than an invalid input or an unsupported mode:

```{code-cell} ipython3
undefined, empty_info = survjax.concordance_index_with_info(
    scores, time=time, event=jnp.zeros_like(event)
)
assert bool(jnp.isnan(undefined)) and bool(empty_info.no_comparable_pairs)
assert not bool(empty_info.invalid | empty_info.unsupported)
print(undefined, empty_info.no_comparable_pairs)
```

## Reference evidence and evaluation practice

`tests/reference/test_concordance_r.py` compares values, counts and masks
against committed R-generated synthetic fixtures, with prepared/raw inputs
and eager/jit/vmap execution in float32 and float64. The fixtures contain
100 heavily tied rows and a small untied cohort, three score vectors each.
Their provenance and regeneration commands are recorded in
`tests/fixtures/concordance/README.md`. Regular tests and this guide need no R.

The pinned reference uses R 4.4.2 and survival 3.7-0, with
`reverse=TRUE`, `timewt="n"` and `timefix=FALSE`. R's `concordant`,
`discordant` and `tied.x` map to our counts; their sum is comparable.
`tied.y` and `tied.xy` are excluded. These fixtures agree with the pair rules
above, including event/censor time ties. The comparisons establish evidence
for these cases, rather than every possible dataset.

The current implementation does O(N²) work with O(N) pair workspace per
result and computes exact concordance over every supplied observation.
For larger cohorts, callers may explicitly select random subsets: record
their sizes and seeds, use the same subsets for model comparisons, and
assess sensitivity to the selection. A subset result need not equal the
full-cohort value; variation across subsets is not itself a population
confidence interval.

For predictive evaluation, fit on training data, choose settings on validation
data, and evaluate fixed predictions on held-out observations. Select the
score definition before inspecting test outcomes. A worked held-out synthetic
workflow and playground integration remain separate documentation steps.
