---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Verifying the Slice A subset

Slice A supplies survival observations and diagnostics, Exponential and
Weibull distributions, direct methods and time grids, custom families,
and exact/right-censored likelihoods without truncation. Harrell's
concordance is also available; its [guide](concordance.md) describes its
separate cohort and score rules. Completing these checks does not establish
support for the full first release or later slices.

`SurvivalData` can represent left/interval censoring and truncation already.
Their likelihood contributions currently return NaN, including inside
`jit` and `vmap`; supported neighboring rows still evaluate. Invalid data
or declared parameter violations also produce NaN, with diagnostics.
See [likelihood diagnostics](likelihood.md#telling-the-causes-apart) to
separate these cases, and [numerical behavior](numerical-behaviour.md) for
true zero probability, overflow and underflow.

## Construction and derivatives inside JIT

Here is a synthetic cohort with two events, one right-censored observation,
and total follow-up 3. For an exponential rate $r$, its summed log
likelihood and parameter derivatives are

$$
\ell(r)=2\log r-3r,\qquad
\ell'(r)=2/r-3,\qquad
\ell''(r)=-2/r^2.
$$

At $r=1$ these are exactly $-3$, $-1$, and $-2$. Keeping these checks exact
avoids needing a floating-point tolerance for this small example.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np

import survjax


def evaluate(rate, time, event):
    data = survjax.SurvivalData(time, event)
    model = survjax.Exponential(rate=rate)

    def total(candidate_rate):
        return survjax.Exponential(rate=candidate_rate).log_likelihood(data).sum()

    return (
        data.issues.ok,
        model.parameter_issues().ok,
        total(rate),
        jax.grad(total)(rate),
        jax.hessian(total)(rate),
    )


time = jnp.array([0.5, 1.0, 1.5])
event = jnp.array([True, False, True])
compiled = jax.jit(evaluate)
for result in (evaluate(1.0, time, event), compiled(1.0, time, event)):
    for actual, expected in zip(result, (True, True, -3.0, -1.0, -2.0), strict=True):
        np.testing.assert_array_equal(actual, expected)
print("data ok, parameters ok, log likelihood, gradient, Hessian:")
compiled(1.0, time, event)
```

Both ordinary constructors are pure and work with traced observations.
Construction records issues; it does not raise for invalid numeric values.
The diagnostics above remain arrays suitable for compiled control flow.
Call `data.issues.raise_if_invalid()` or `model.parameter_issues().raise_if_invalid()`
only at an explicit host checkpoint. These host checks are outside a JIT
computation. See the [data guide](survival-data.md) for both constructors.

Prepared data and raw `time`/`event` input use the same likelihood semantics:

```{code-cell} ipython3
prepared = survjax.SurvivalData(time, event)
model = survjax.Exponential(rate=1.0)
prepared_values = model.log_likelihood(prepared)
raw_values = model.log_likelihood(time=time, event=event)
np.testing.assert_array_equal(prepared_values, [-0.5, -1.0, -1.5])
np.testing.assert_array_equal(raw_values, prepared_values)
prepared_values
```

Map over two independent cohorts. Doubling follow-up changes the total
from 3 to 6, leaving two events: at rate 1 the second cohort has value $-6$,
gradient $2-6=-4$ and Hessian $-2$.

```{code-cell} ipython3
mapped = jax.jit(jax.vmap(evaluate))
results = mapped(jnp.ones(2), jnp.stack((time, 2 * time)), jnp.stack((event, event)))
expected = ([True, True], [True, True], [-3, -6], [-1, -4], [-2, -2])
for actual, wanted in zip(results, expected, strict=True):
    np.testing.assert_array_equal(actual, wanted)
results[2:]
```

The parameter derivative guarantee covers valid, finite, smooth interior
cases through second order, including `jit` and `vmap`. It does not cover
time derivatives, invalid/infinite objectives, support-change boundaries,
or arbitrary unsafe user implementations. The regression suites also check
Weibull, forward/reverse derivatives, custom-family defaults and isolation
of invalid parameter rows. [Custom distributions](distributions.md) explains
the extension contract; [time grids](grids.md) explains time-first axes.

## Check the installed wheel

From the repository root in the locked development environment, run:

```bash
scripts/check_wheel.sh --through A9
```

The runner builds a fresh wheel, sets its absolute `SURVJAX_WHEEL` path,
and runs the unchanged gate in separate float32/float64 processes. The gate
includes lint/type checks, own tests, source doctests and independent tests
through A9. Plain `scripts/check.sh --through A9` needs that wheel variable
already set; it does not build an artifact itself.

Own installation checks create a fresh environment outside the source tree,
install the locked core dependencies and wheel offline, and check exports,
metadata, `py.typed`, a hand-derived likelihood and absence of demo packages.
An out-of-tree custom family is checked at runtime and with strict mypy.
Cached locked build/install artifacts must be available; missing artifacts
are failures, not skipped tests. A source archive remains a separate
pre-release check. No publication or remote setup is performed.

The R comparisons use committed synthetic data generated with R 4.4.2,
flexsurv 2.3.2 and survival 3.7-0. Exponential and bc-like Weibull per-row
likelihood/curve references are checked in both dtypes. These compare synthetic observations at fixed parameter values, not a fit
to real breast-cancer data or
evidence for unsupported observation modes. Ordinary tests need no R or
network access; see each fixture set's README under `tests/fixtures/`.

## Reproduce the CPU baseline

Run these commands sequentially from the repository root:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python \
  scripts/benchmark_slice_a.py --output /tmp/survjax-cpu.json
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run python \
  scripts/benchmark_slice_a.py --output /tmp/survjax-cpu.json
```

The output retains both dtype runs. The committed baseline is
`benchmarks/slice_a_cpu.json`. Each mode covers both families, 1,000, 10,000
and 100,000 observations, shared/per-observation parameters, and two data paths:

| Path | Construction | Static metadata |
| --- | --- | --- |
| Prepared | Concrete `SurvivalData` before the compiled call | Exact/right present; other modes absent |
| Traced | Ordinary constructor inside `jit` | All may-contain flags true |

Both paths return data and parameter diagnostics alongside the summed
likelihood and parameter gradient, so validation is part of the measured
work. Each case checks analytical values/gradients and compilation reuse
with changed dynamic values and same-shaped parent minibatches. Three
warmups precede ten synchronized samples; all output leaves finish before
timing stops. First call, host generation, initial placement and concrete
construction are separate from warmed evaluation.

On the recorded Apple M4 Max CPU, medians at 100,000 observations range
from 0.305 to 1.117 ms in float64 and 0.231 to 0.523 ms in float32 across
the eight cases per mode. Python 3.12.9, JAX/jaxlib 0.11.2 and NumPy 2.5.3
are recorded with seed, commands, shapes, source revision and raw samples.
These are descriptive results on one machine, not a speed guarantee or an
accelerator comparison. There is no timing threshold in ordinary tests.

Concrete construction is also recorded on traced-path rows as a separate
comparison measurement; that object is not used by the traced evaluator.
Initial eager compilation/cache setup can dominate that field. Fixed case
ordering, short sample sets and machine load affect comparisons; the paths
show no universal timing advantage. Apple CPU placement does not measure
physical bus-transfer cost. Logical input/output byte counts sum array
leaves and are **not peak memory**; peak memory was not measured.
The recorded revision identifies the library source before the harness was
committed; the harness and artifact were committed together in `51319c8`.

## Build and explore the documentation

```bash
scripts/build_docs.sh
uv run --locked --group docs jupyter lab docs/playground.ipynb
```

The build executes all runnable pages and the playground afresh in separate
float32/float64 processes and fails on execution errors or Sphinx warnings.
Open `docs/_build/html-x32/index.html` or `docs/_build/html-x64/index.html`.
The notebook is editable, uses synthetic data and the current public API,
and is committed without outputs. The exploratory simulation guide is
excluded from the build because it describes future likelihood modes.

This verification recipe and the existing likelihood examples are small
synthetic checks. The agreed README/use-case teaching material, held-out
concordance example and full release demos remain separate work; no
standalone demo removed from A9 is restored here.
