---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Verification and reproducibility

This note records checks of the exact/right-censored likelihood subset,
compiled derivatives, installed packages and a historical CPU baseline.
It is evidence for those specific checks, not a full-release certification.
The `A9` and `slice_a` names below identify the original verification scope.

`SurvivalData` can represent left/interval censoring and truncation already.
Their likelihood contributions currently return NaN, including inside
`jit` and `vmap`; supported neighboring rows still evaluate. Invalid data
or declared parameter violations also produce NaN, with diagnostics.
See [likelihood diagnostics](likelihood-diagnostics.md) to
separate these cases, and [numerical behavior](../numerical-behaviour.md) for
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
computation. See the [data guide](../survival-data.md) for both constructors.

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
of invalid parameter rows. [Custom distributions](../distributions.md) explains
the extension contract; [time grids](../grids.md) explains time-first axes.

## Check the installed wheel

From the repository root in the locked development environment, run:

```bash
scripts/check_wheel.sh --through A9
```

The runner builds a fresh wheel, sets its absolute `SURVJAX_WHEEL` path,
and runs the unchanged gate. Lint and type checks run once; own tests,
source doctests and independent tests through A9 each run in separate
float32/float64 processes. Plain `scripts/check.sh --through A9` needs that wheel variable
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
likelihood/curve references are checked in both dtypes. These compare
synthetic observations at fixed parameter values. They do not establish
agreement for a fit to real breast-cancer data or for unsupported observation
modes. Ordinary tests need no R or network access; see each fixture set's
README under `tests/fixtures/`.

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

From the repository root, run `scripts/build_docs.sh`. It executes runnable
pages afresh in separate float32/float64 processes and fails on notebook errors
or Sphinx warnings. Open `docs/_build/html-x32/index.html` or
`docs/_build/html-x64/index.html` afterwards. Use
`scripts/build_docs.sh --output-dir DIR` to choose another build root; its HTML
subdirectories are still `html-x32` and `html-x64`. See the
{download}`build interface and portal compatibility notes <../../DOCS-BUILD.md>`
for path handling, dependencies, caches, exit behavior and serving under a prefix.

The MNIST use case executes its smoke example in float32 and downloads
hash-checked images on first use. A warm cache at `SURVJAX_EXAMPLES_DATA`
(default `~/.cache/survjax-examples/mnist`) avoids later downloads.
The Bayesian use case computes its results from committed
posterior draws; its notebook also demonstrates sampling the posterior.

For interactive exploration with the locked environment:

```bash
uv run --locked --group docs --group examples jupyter lab examples/survival_mnist/survival-mnist.ipynb
```

The notebook has editable smoke/full settings; see the MNIST use case for
reproduction instructions and the distinction between live and recorded results.

## Comparisons with R

The committed synthetic fixtures compare exponential and Weibull likelihoods
and predictions with R's **flexsurv 2.3.2** and **survival 3.7-0**, generated
with **R 4.4.2**. Only simulated observations and their R outputs are stored.
The reference tests read those files without running R or accessing a network.
They evaluate fixed parameter vectors; they do not fit models in survjax.

| Fixture | Observations and parameter cases | R quantities compared |
| --- | --- | --- |
| Exponential | 16 rows, seed 8101; rates 0.1, 0.35 and 1.2 per year | Every row's log likelihood, survival and log survival; separate total log likelihood |
| Weibull AFT | 686 rows, seed 8201; three common-shape, group-scale vectors | Every row's log likelihood, survival and log survival; separate totals; a time-first survival grid |

Both fixtures contain exact events and right censoring without truncation.
The Weibull fixture uses Good/Medium/Poor groups of sizes 229/229/228,
with scales $b_g = b_{\mathrm{Good}}\exp(\beta_g)$. Its three vectors are
named `fitted`, `unit_shape` and `alternative`. `fitted` identifies the
recorded vector used to simulate the data, rather than a fit to these
synthetic rows. The rows are newly simulated; no patient observations from
`flexsurv::bc` are loaded or committed. Fixture times are in **years**,
while the built-in distribution guide uses months.

The Weibull grid has shape `(141, 3)`: times from 0 to 7 years in rows,
Good/Medium/Poor in columns. Tests use the stored R times; rebuilding R's
`seq` with JAX `arange` or `linspace` can shift a time by one ULP. The survreg
grid route evaluates `psurvreg` directly at supplied AFT locations and is
cross-checked against flexsurv. Singleton-fit and full-data likelihood checks
cover the observed-time contributions, not individual grid cells.

The comparisons exercise eager execution, `jax.jit` and `jax.vmap` in separate
float64 and float32 processes. Weibull checks include both natural and
`from_log` constructors, a common shape with batched scales, and raw and
prepared likelihood inputs through the distribution method.
Expected R values retain their stored float64 precision; only inputs are cast
to the active JAX dtype. The central test tolerances are:

| Active dtype | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
| float64 | `1e-10` | `1e-8` |
| float32 | `1e-5` | `1e-4` |

These finite reference cases complement the independent mathematical and
[numerical boundary tests](../numerical-behaviour.md). The two R package paths
are related: flexsurv uses survreg on this route, so their agreement is not
fully independent numerical evidence. The fixtures provide no evidence for
unimplemented censoring or truncation likelihoods, or for fitting accuracy.

### Integrity and regeneration

In a source checkout, `tests/reference/test_exponential_r.py` and
`tests/reference/test_weibull_bc_like_r.py` use one shared fixture loader.
It verifies all required payloads against each fixture's `SHA256SUMS` before
parsing them, then checks provenance, row alignment, dimensions and labels.
The Weibull loader also checks that the acceptance-schema export preserves
the original observations, parameters, source results and grid.

Regenerate only with the pinned R/package versions above. The generators in
`scripts/fixtures/` support `--check-only` without writing. A normal run
**overwrites payload files** in its destination; without a destination it
writes into the committed fixture directory. To inspect fresh outputs first,
pass a new scratch directory, for example from the repository root:

```sh
Rscript --vanilla scripts/fixtures/generate_bc_like.R --check-only
Rscript --vanilla scripts/fixtures/generate_bc_like.R /tmp/bc-like-review-run1
```

Follow the full two-run comparison and checksum recipes in the fixture
READMEs: {download}`exponential regeneration <../../tests/fixtures/exponential/README.md>`
and {download}`Weibull regeneration <../../tests/fixtures/bc_like/README.md>`.
Compare fresh outputs with each other and with committed payloads before
replacing anything. Exports use 17 significant digits and record seed,
versions, generating scripts and checksums. During fixture validation,
Python separately checked decimal read-back against original R hexadecimal
doubles; Python did not generate expected numerical values.

The directory also holds the same R outputs in the acceptance-test schema
(`bc_synthetic_weibull.json`); ordinary regeneration never writes to
the independent acceptance-test directory.

## Concordance reference evidence

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
in the [concordance guide](../concordance.md), including event/censor time ties. The comparisons establish evidence
for these cases, rather than every possible dataset.
