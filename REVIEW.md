# README.1 — Alpha README

Independent review: round 1/3

Read README.md, then PLAN.md's README.1 section. Compare current claims with
src/survjax/__init__.py, distributions/base.py and metrics.py and spec 12.2.

One topic: an accurate introductory README. Exponential and Weibull are the
first built-in families, concordance the first metric; future additions are
explicitly planned. Representation does not imply likelihood support.
No public API, dependencies, source, tests, protected paths or spec changed.

The fenced Python example is executed unchanged with uv run Python spawning
separate JAX_ENABLE_X64=0 and 1 processes. Independent analytical checks use
negative log likelihood 2*log(5)+0.2*9.5, gradient 0.2*9.5-2, and survival
exp(-0.2*t). Both print:
README example: analytical value, gradient, survival PASS
Comparisons use rtol=1e-6 for value/survival, atol=1e-6 for gradient.
These are temporary documentation checks, not changes to test tolerances.
No new tests needed for a prose-only change; no full numerical suite run.

Installation is explicitly from a local checkout; no package-index release
or public documentation hosting is newly verified. The existing docs URL is
retained. No browser rendering check performed for the README.

Validation: git diff --check passes. scripts/build_docs.sh completed both dtype builds with “build succeeded.”
The notebook kernels emitted their TCP-without-encryption startup warning. No numerical behavior changes.
