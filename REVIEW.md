# MNIST.19 — Simplify the example support code

Independent review: round 1/3 (pending).

## Read first

1. PLAN.md MNIST.19: maintainer-requested cleanup after the tutorial notebook
   and docs page were accepted (22906cb). Example code only.
2. examples/survival_mnist/model.py: `evaluate` returns only mean log
   likelihood, shape and C; `fit_intercept(data)` returns `(log_k, log_b)`.
3. examples/survival_mnist/run.py: `_concordance_seconds` (moved from
   `evaluate`), rows add `n` and timing, CSVs via pandas, pandas version.
4. examples/survival_mnist/data.py: constants and `UC_KEY` at the top.
5. Notebook: three source lines for the new signatures. README: notebook
   default and docs-page wording. pyproject.toml: notebook-only ruff exception.
6. Tests: check_mnist_metrics/reference/training/run updates and the new
   optional check_mnist_notebook.py.

## Invariants

- Results are unchanged: `UC_KEY` is the same `fold_in(key(20261025), 3)`;
  `fit_intercept` gets the same float32 rows (`observations[train]`) and the
  same BFGS start (exponential MLE scale, log k = 0); `evaluate` computes the
  same likelihood and C. Only the timing moved to run.py, where it keeps the
  warm-up and the synchronized single call (use-case 03: C run time at n=10k).
- metrics.csv/history.csv keep their columns, order, LF endings and float
  repr, which the docs page and acceptance test read.
- The docs page source is untouched; the notebook's saved outputs are kept.

## Tests and evidence

- `check_mnist_metrics.py`: hand-derived `evaluate` cases without n/timing;
  timing test now pins `run._concordance_seconds` (compile call excluded).
- `check_mnist_reference.py`: tuple return, float32, profile-optimum match.
- `check_mnist_run.py`: smoke CSV headers equal the committed ones; no CR.
- `check_mnist_notebook.py`: the notebook runs every cell in smoke config
  (would have caught the HEADER NameError found earlier).

## Checks run

- `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src`: clean.
- Optional MNIST suite, float32, with the docs group:
  `pytest tests/unit/check_mnist_*.py tests/unit/test_mnist_*.py`: 31 passed.
- `scripts/run_tests.sh`: 1106 passed, 7 skipped (x64); 1112 passed,
  1 skipped (x32).
- Fresh `run --config full`: history.csv byte-identical; metrics.csv identical
  except `concordance_seconds`; metadata identical except timings and the new
  pandas version. Committed results were not regenerated.
- `scripts/build_docs.sh`: both modes succeed; MNIST tables equal the
  previous build.
- Notebook executed with nbclient in full config: every code cell's text/HTML
  output and all three PNGs are byte-identical to the saved outputs.

## Unsure about

- The ruff exception for the notebook versus formatting it. Formatting would
  rewrap seven lines of the accepted notebook, two with trailing comments.
- Whether `results/metadata.json` should be regenerated to record pandas;
  left as committed because no result changed.
