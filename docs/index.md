# survjax documentation

survjax is a JAX library for survival analysis. The current implementation
can describe survival observations (`SurvivalData`), provides the exponential
and Weibull distributions and a proportional-hazards wrapper, and lets you
write your own survival
distributions, from which survjax derives survival, cumulative hazard,
hazard, density and CDF. Every distribution can be evaluated pointwise or on
a shared time grid. Exact/right-censored likelihoods without truncation are
available, along with Harrell's concordance for caller-supplied risk scores.
Restricted mean survival time (`rmst`) is available for the exponential and
Weibull families and their PH wrappers. Additional likelihood modes, families,
summaries and evaluation metrics are planned.

[Verifying the Slice A subset](slice-a-verification.md) covers compiled
derivatives, installed-wheel checks and reproducing the CPU baseline.
[Log likelihoods](likelihood.md) explains how survjax computes likelihood
contributions and how to call it, illustrated by fitting exponential and
Weibull models with JAX.
[Likelihoods in the tails](numerical-behaviour.md) explains underflow, overflow,
boundary gradients and checking that a fitting loss is finite.
[Harrell's concordance](concordance.md) evaluates risk rankings with right
censoring, including tied observations, score draws and cohort diagnostics.
[Restricted mean survival time](rmst.md) summarizes expected time before a
finite cutoff, with explicit cutoff axes and staged derivative limits.
[Bayesian survival analysis of a randomized trial](use-cases/bayesian-rmst.md)
fits a synthetic trial with NumPyro and reports the posterior RMST difference.
[Survival MNIST](use-cases/survival-mnist.md) trains an image-based survival
model and compares a live smoke run with committed full results.
The [SurvivalData guide](survival-data.md) walks a synthetic cohort through
both constructors, diagnostics and R input conventions.
[Exponential and Weibull](builtin-families.md) introduces the built-in
families, their behaviour at time zero and infinity, and how their parameters
map to R's `survival` and `flexsurv` and to NumPyro.
[Proportional hazards](proportional-hazards.md) introduces the built-in wrapper,
log hazard ratios, batch and time-grid shapes, likelihoods and the
initial-release derivative limits.
[Survival curves and time grids](grids.md) evaluates batches of
distributions, including posterior draws, at many times at once. The
[custom-distribution guide](distributions.md) builds constant-hazard, Gompertz,
piecewise-constant and proportional-hazards models from their survival
functions. The [playground](playground.ipynb) uses only synthetic data and the
current public API. The [API reference](api.md) lists every public class and
function with its docstring. Runnable examples execute afresh when the site
is built.

```{toctree}
:maxdepth: 1

playground
survival-data
builtin-families
proportional-hazards
rmst
use-cases/bayesian-rmst
likelihood
numerical-behaviour
slice-a-verification
concordance
use-cases/survival-mnist
grids
distributions
api
```

## Local commands

From the repository root, run `scripts/build_docs.sh` to build the site and
execute its notebooks separately with JAX float32 and float64. The generated
pages are in `docs/_build/html-x32/` and `docs/_build/html-x64/`. The build
fails on warnings and notebook errors. Dependencies use the locked docs and
examples groups. Core examples use synthetic data and need no external data or
R. The MNIST page downloads hash-checked images on first use; a warm cache at
`SURVJAX_EXAMPLES_DATA` (default `~/.cache/survjax-examples/mnist`) avoids network
access on later builds.

The runner keeps Jupyter runtime files, IPython state and Matplotlib
configuration/cache under ignored `docs/_build/runtime/`, using
`JUPYTER_RUNTIME_DIR`, `IPYTHONDIR` and `MPLCONFIGDIR`. It sets these for the
build without changing `HOME` or global settings. It preserves the MNIST cache
choice to avoid downloading it in every worktree. For a cold cache, set
`SURVJAX_EXAMPLES_DATA` to a writable location (for example an absolute path
under `docs/_build/mnist`) before invoking the script directly. An explicit
value is never replaced. The first download needs network; warm-cache success
does not establish cold-cache portability.

The script prints absolute HTML paths and `docs/_build/build-identity.txt`.
Before inspection, read that record: its **last** `state:` must be `success`,
with the intended absolute checkout, commit, dtype pair and UTC completion
time. Compare with `git rev-parse HEAD` and `git status --short`. A dirty tree
is labelled with its status; this is provenance, not a reproducible snapshot
of uncommitted or untracked file contents. Do not edit sources or run builds
concurrently during verification. Tracked edits during the build are detected;
untracked content edits and changes after completion require a fresh build.
A new attempt invalidates the old success before setup. Errors/signals leave
a failed record; abrupt termination may leave `building`. Neither certifies
old HTML that remains on disk. Only completion of both dtype builds certifies
the pair. The record is external to HTML; an already open tab can be stale.

To view the rendered site on macOS, run `open
docs/_build/html-x64/index.html` (or use `html-x32` for float32). On other
systems, open that `index.html` file in a browser. Use the absolute printed
path to avoid a different worktree's site, and reload after building. Local
`file:///absolute/path/.../index.html` is sufficient for routine checks; no
preview server or Quarto command is needed for this Sphinx/MyST-NB site.

For agent inspection, use the currently documented computer-use APIs. Discover
available browser surfaces first. If discovery reports the browser access host
unavailable, native Chrome may still work: select `com.google.Chrome` with
`cua.getApp`, read the returned documentation, open a new tab and navigate to
the absolute local HTML URL using those APIs. Do not assume that either browser
host access or native Chrome is available on every host. Close only the tab
you created and record the actual capability failure if neither works.

Inspect navigation, tables, code outputs and equations in both dtypes. Reading
HTML, checking links or reading the accessibility tree verifies structure and
content; capture and examine screenshots to assess layout, clipping and math
rendering. Record pages inspected, checkout/build identity, screenshot findings
and limits in the review. MathJax currently loads from a CDN, so offline HTML
does not guarantee rendered equations.

If a build reproduces a sandbox Jupyter/psutil process-inspection or child
cleanup failure, retain its exact error and request the narrowly approved
`scripts/build_docs.sh` escalation, invoking the script directly. Writable
runtime directories do not repair process-inspection restrictions. Escalation
availability depends on the host. Historical cleanup errors are not evidence
that the current build failed; record the current command and outcome.

For an editable notebook, run `uv run --locked --group docs jupyter lab
docs/playground.ipynb`. JupyterLab uses the same locked environment as the
build. Notebook outputs and generated pages are local artifacts.
