---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Verification and reproducibility

This note records checks of likelihoods, compiled derivatives, installed
packages and a historical CPU baseline. Its small executable examples use
exact/right observations; the R comparison section also covers left/interval
censoring and truncation where the conventions match.
It is evidence for those specific checks, not a full-release certification.
The `A9` and `slice_a` names below identify the original verification scope.

All four observation kinds support likelihood evaluation with optional
truncation. Invalid data or declared parameter violations produce NaN, with
diagnostics; valid zero-probability observations give `-inf`. See
[likelihood diagnostics](likelihood-diagnostics.md) for NaN triage, including
zero inclusion-window mass, and [numerical behavior](../numerical-behaviour.md)
for overflow and underflow.

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

## Comparisons with R

The committed fixtures contain synthetic observations and their R outputs,
using R 4.4.2, flexsurv 2.3.2 and survival 3.7-0 as applicable. Ordinary
reference tests read them without R or network access. Likelihood and curve
comparisons evaluate fixed parameter vectors; they do not fit models in
survjax or establish agreement for fits to real patient data. The subsections
below distinguish the observation modes, quantities and dtypes checked by
each fixture set.

### Exact/right likelihoods and predictions

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

These exact/right comparisons exercise eager execution, `jax.jit` and
`jax.vmap` in separate float64 and float32 processes. Weibull checks include
both natural and `from_log` constructors, a common shape with batched scales,
and raw and prepared likelihood inputs through the distribution method.
Expected R values retain their stored float64 precision; only inputs are cast
to the active JAX dtype. The central test tolerances are:

| Active dtype | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
| float64 | `1e-10` | `1e-8` |
| float32 | `1e-5` | `1e-4` |

These finite reference cases complement the independent mathematical and
[numerical boundary tests](../numerical-behaviour.md). The two R package paths
are related: flexsurv uses survreg on this route, so their agreement is not
fully independent numerical evidence. These two fixtures provide evidence only
for untruncated exact/right observations; the broader likelihood comparisons
follow below. They do not establish fitting accuracy.

### Censoring and truncation: comparable flexsurv cases

The committed Slice B fixture adds Weibull (including shape 1), LogNormal and
LogLogistic, each at two fixed natural parameter vectors without covariates.
All six vectors use the same 50 synthetic observations, generated with seed
20261110: five rows in each of the ten comparable combinations below.
The 300 reference values are per-row `flexsurvreg(..., fixedpars=TRUE)$logliki`
outputs. Parameters are held fixed; these are not comparisons of fitted estimates.

Write the inclusion window as `(L, U]`, an exact time as `t`, a censoring
time as `c`, and a finite censored interval as `(a, b]`.

| Truncation | Observation kinds compared | Restrictions |
| --- | --- | --- |
| None: `L=0`, `U=inf` | Exact, right, left, interval | Ordinary valid observations |
| Left only: `L>0`, `U=inf` | Exact, right | Exact `t>L`; right censoring `c>L` |
| Right only: `L=0`, finite `U` | Exact, left, interval | Exact `t<=U`; left censoring `c<U`; interval `b<=U` |
| Two-sided: `L>0`, finite `U` | Exact | `L<t<=U` |

The fixture includes exact events at `U` and intervals ending at `U` in the
applicable cells. Its numerical range is deliberately moderate: cumulative
hazard is at most 15 at every finite endpoint or truncation bound; finite
intervals and two-sided windows have width at least `1e-4` times their upper
bound; each censored set and nontrivial inclusion window has probability at
least `1e-6` under every parameter vector. These limits keep the probability
differences used by flexsurv suitable as reference values. They are fixture
restrictions, not restrictions on survjax inputs or evidence for deep tails.

The R values are compared **in float64 only**, with absolute tolerance `1e-10`
and relative tolerance `1e-8`. The implementation reference test checks all
300 values eagerly, under `jit`, and with row-wise `vmap`; independent
acceptance tests check per-cell, batched and summed values and audit the
fixture with high-precision mathematics. Manifest hashes, pinned provenance,
CSV/JSON alignment and the identical acceptance payload are also checked.
Integrity checks remain active in float32; numerical float32 and derivative
evidence comes from separate analytical and high-precision tests. This R
fixture contains neither derivatives nor `survreg` values.

The {download}`Slice B fixture README <../../tests/fixtures/slice_b/README.md>`
records generation and integrity checks. The generator is
`scripts/fixtures/generate_slice_b.R`; reference comparisons are in
`tests/reference/test_slice_b_r.py` and the independent
`tests/spec/test_observation_censoring_flexsurv_fixture.py`.

#### Where flexsurv is not the reference

The following cases are excluded from the Slice B R comparison because
flexsurv cannot express them or uses a different likelihood convention:

- Left or interval censoring with left truncation, including two-sided windows.
- Right censoring under a finite `U`: flexsurv uses `S(c)` for the numerator,
  whereas survjax uses `P(c<T<=U)`.
- Censored sets extending beyond `U`: survjax intersects the set with the
  inclusion window. flexsurv does not, so a set covering that window can
  contribute a positive log likelihood there.
- Censoring at or before entry: survjax treats right censoring at `c=L` as
  valid and uninformative (zero contribution), and `c<L` as invalid.
- Observations outside the inclusion window.
- Exact events at time zero.

These exclusions do not mean the modes are unimplemented. The
[likelihood guide](../likelihood.md#left-and-interval-censoring-with-truncation)
explains intersection, conditioning and invalid-data behavior. Independent
mathematical tests cover these cases, including boundary diagnostics, tails
and transformed derivatives; they do not require matching a different R
convention. In the source tree, see `tests/spec/test_observation_truncation_*.py`
and `tests/spec/test_observation_censoring_*.py`, with family-specific evidence
in the LogNormal and LogLogistic suites.

### Concordance reference evidence

`tests/reference/test_concordance_r.py` compares values, counts and masks
against committed R-generated synthetic fixtures, with prepared/raw inputs
and eager/jit/vmap execution in float32 and float64. The fixtures contain
100 heavily tied rows and a small untied cohort, three score vectors each.
Their provenance and regeneration commands are recorded in
`tests/fixtures/concordance/README.md`. Regular tests and this guide need no R.

The concordance reference uses survival with
`reverse=TRUE`, `timewt="n"` and `timefix=FALSE`. R's `concordant`,
`discordant` and `tied.x` map to our counts; their sum is comparable.
`tied.y` and `tied.xy` are excluded. These fixtures agree with the pair rules
in the [concordance guide](../concordance.md), including event/censor time ties.
The comparisons establish evidence for these cases, rather than every possible
dataset.

### Integrity and regeneration

The reference suites use the shared fixture loader in `tests/conftest.py`.
It verifies all required payloads against each fixture's `SHA256SUMS` before
parsing them, then checks provenance, row alignment, dimensions and labels.
The Weibull loader also checks that the acceptance-schema export preserves
the original observations, parameters, source results and grid.

Regenerate only with the pinned R/package versions above and the instructions
for the relevant fixture. The generators in `scripts/fixtures/` support
`--check-only` without writing. A normal run **overwrites payload files** in
its destination. The exponential, bc-like and concordance generators default
to their committed fixture directories; the Slice B generator requires a destination.
To inspect fresh outputs first, pass a new scratch directory, for example
from the repository root:

```sh
Rscript --vanilla scripts/fixtures/generate_bc_like.R --check-only
Rscript --vanilla scripts/fixtures/generate_bc_like.R /tmp/bc-like-review-run1
```

Follow the full two-run comparison and checksum recipes in the fixture
READMEs: {download}`exponential regeneration <../../tests/fixtures/exponential/README.md>`,
{download}`Weibull regeneration <../../tests/fixtures/bc_like/README.md>`,
{download}`Slice B regeneration <../../tests/fixtures/slice_b/README.md>` and
{download}`concordance regeneration <../../tests/fixtures/concordance/README.md>`.
Compare fresh outputs with each other and with committed payloads before
replacing anything. Exports use 17 significant digits and record seed,
versions, generating scripts and checksums. For the exponential and bc-like
fixtures, Python separately checked decimal read-back against original R
hexadecimal doubles; Python did not generate expected numerical values.

The Weibull directory also holds the same R outputs in the acceptance-test schema
(`bc_synthetic_weibull.json`); ordinary regeneration never writes to
the independent acceptance-test directory.

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

## Reproduce the CPU baseline

The following commands belong to the historical Slice A baseline. They require
the baseline checkout at `51319c8`, where the harness and artifact were
committed together. Run them sequentially from that checkout's repository root:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python \
  scripts/benchmark_slice_a.py --output /tmp/survjax-cpu.json
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run python \
  scripts/benchmark_slice_a.py --output /tmp/survjax-cpu.json
```

The output retains both dtype runs. The committed **historical Slice A** baseline
is `benchmarks/slice_a_cpu.json`, recorded at source revision `0a200085`.
Each mode covers both families, 1,000, 10,000
and 100,000 observations, shared/per-observation parameters, and two data paths:

| Path | Construction | Static metadata in the historical run |
| --- | --- | --- |
| Prepared | Concrete `SurvivalData` before the compiled call | Exact/right present; other modes absent |
| Traced | Ordinary constructor inside `jit` | All may-contain flags true |

Current traced construction uses argument-form metadata: this harness supplies
`time` and `event` without truncation arguments, so it now carries exact/right
flags without truncation, just like its prepared data. General traced bounds
construction can retain more possibilities. See
[Metadata, pruning and compilation reuse](../survival-data.md#metadata-pruning-and-compilation-reuse)
for the full rules, including supplied default values. The legacy harness still
asserts all-true metadata for its traced path, so the commands above fail that
assertion on the current checkout. Current pruning benchmarks are maintained
in the separate experiments project. The recorded timings below describe the
earlier implementation, not current performance or an isolated pruning effect.

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

The runner keeps Jupyter runtime files, IPython state and Matplotlib
configuration/cache under the selected build root's ignored `.cache/`, using
`JUPYTER_RUNTIME_DIR`, `IPYTHONDIR` and `MPLCONFIGDIR`. It sets these for the
build without changing `HOME` or global settings. It preserves the MNIST cache
choice to avoid downloading it in every worktree. For a cold cache, set
`SURVJAX_EXAMPLES_DATA` to a writable location (for example an absolute path
under `docs/_build/mnist`) before invoking the script directly. An explicit
value is never replaced. The first download needs network; warm-cache success
does not establish cold-cache portability.

### Verify and open the built pages

The script prints absolute HTML paths and `DIR/build-identity.txt`
(`docs/_build/build-identity.txt` by default).
Before inspection, read that record: its **last** `state:` must be `success`,
with the intended absolute checkout, commit, dtype pair and UTC completion
time. Compare with `git rev-parse HEAD` and `git status --short`. A dirty tree
is labelled with its status; this is provenance, not a reproducible snapshot
of uncommitted or untracked file contents. Do not edit sources or run builds
concurrently during verification. Tracked edits during the build are detected;
untracked content edits and changes after completion require a fresh build.
A new attempt invalidates the old success before setup. Errors/signals leave
a failed record; setup failure or abrupt termination may leave `building`. Neither certifies
old HTML that remains on disk. Only completion of both dtype builds certifies
the pair. The record is external to HTML; an already open tab can be stale.

To view the rendered site on macOS, run `open
docs/_build/html-x64/index.html` (or use `html-x32` for float32). On other
systems, open that `index.html` file in a browser. Use the absolute printed
path to avoid a different worktree's site, and reload after building. Local
`file:///absolute/path/.../index.html` is sufficient for routine checks; no
preview server or Quarto command is needed for this Sphinx/MyST-NB site.

MathJax currently loads from a CDN, so offline HTML does not guarantee
rendered equations.

### Explore use-case notebooks

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
