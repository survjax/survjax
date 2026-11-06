# BAYES clean-up and main merge (2026-11-06)

No independent review cycle for this step, by maintainer decision; the
maintainer does one final pass before merging into main. Previous REVIEW.md
contents: BAYES.d1 at 55565fd, MNIST.20 at main 2093ca2.

Last commit of the clean-up: add507a (user commit). Previous Bayesian step:
55565fd (BAYES.d1). Working tree: merge of main 2093ca2 into release-bayes.

## Read first

1. DEVIATIONS.md D-041 (formerly D-035): agreed scope of the lean example.
2. docs/use-cases/bayesian-rmst.md: the approved page; every number, table and
   figure is computed on the page from `results/posterior.npz`.
3. examples/bayesian_rmst/model.py, then run.py (acceptance asserts at the end
   of `run`).
4. examples/bayesian_rmst/bayesian-rmst.ipynb: the same tutorial, sampling
   itself; it imports only `make_trial`.
5. The merge resolutions listed in PLAN.md.

## Invariants

- `make_trial` uses the original key (root 20261025, fold 2, stream 0) and the
  shared generator, so the committed trial is unchanged; NUTS uses stream 1.
- `run.py` saves before it asserts, so a failing run leaves its evidence.
  Diagnostic and truth thresholds apply only to the 1000-draw configuration.
- The page reads `metadata.json` only for the truth and R-hat/ESS; means, SDs,
  intervals and figures come from the draws.

## Tests and evidence

- `tests/unit/check_bayesian_run.py`: a fresh float64 run passes the four
  acceptance checks (factor wiring, routes within 1e-12, R-hat/ESS/divergences,
  truth inside the 95% intervals) and reproduces the committed trial and draws
  bit for bit.
- `tests/unit/check_bayesian_notebook.py`: the notebook runs in a fresh kernel
  with 100 draws per chain.

## Checks run

- Merged tree: `uv run ruff format --check .`, `uv run ruff check .`,
  `uv run mypy src`: clean; `uv lock --check`: consistent.
- Tier 2 at merge commit 9a2d723, `scripts/check.sh --through A10`: all gates
  passed (own 1292 passed/8 skipped and 1297 passed/3 skipped; doctests 4+4;
  spec 2333 passed/27 skipped and 2324 passed/36 skipped).
- `scripts/build_docs.sh`: both builds succeeded.
- Optional suites with the docs and examples groups, `check_bayesian_run.py`,
  `check_bayesian_notebook.py` and `check_mnist_*.py`: 24 passed (x64),
  24 passed (float32).

## Unsure about

- The notebook and the page each define the 8-line Kaplan–Meier helper and
  the model; the duplication is deliberate (readable, editable notebook).
- The ArviZ removal drops tail ESS from the acceptance checks (D-041).
