# MNIST.20 — One compiled concordance; no shadowed module name

Independent review: round 1/3, verdict merge (2026-11-06), in
review/slices/MNIST.20.md. No findings.

Previous step MNIST.19 (bdce9b2, merge verdict); main merged in 9e742bd. The
merge's preserved review records are in that commit's REVIEW.md.

## Read first

1. PLAN.md MNIST.20: maintainer resolution of MNIST.19's nits. N-1 accepted
   as is, N-3 approved, N-2 fixed here.
2. examples/survival_mnist/model.py: the single compiled
   `concordance_index = jax.jit(survjax.concordance_index)`, used by `evaluate`.
3. examples/survival_mnist/run.py: `_concordance_seconds(log_scale,
   observations)` calls `model.concordance_index`; the parameter no longer
   shadows the imported `data` module; the unused `jax` import is gone.
4. tests/unit/check_mnist_metrics.py: the timing test patches
   `model.concordance_index`.

## Invariants

- One jit wrapper: `evaluate`'s C and the runner's timing call the same
  compiled function, so the runner's warm-up call is still made before the
  timed call (use-case 03: C run time at n=10k, compile excluded).
- Scores are unchanged: `-broadcast(log_scale)` in float32, synchronized
  before the warm-up call.

## Tests and evidence

- `test_concordance_timing_excludes_the_compile_call`: the fake clock shows the
  warm-up cost (11) excluded and the single timed call (2) reported.
- The remaining check_mnist_metrics cases pin `evaluate`'s C through
  `model.concordance_index`.

## Checks run

- `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src`: clean.
- Tier 0, optional MNIST suite with the examples and docs groups:
  31 passed (x64), 31 passed (float32).
- Tier 1, `scripts/check.sh --fast --through A10`: all gates passed (own
  1052/8 skipped and 1057/3 skipped; doctests 4+4; spec 2149/27 skipped and
  2156/20 skipped).

## Unsure about

- `concordance_index` is now a non-underscore name in an example module. The
  example's names are not library API; it is public there because run.py uses it.
