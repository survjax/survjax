# Merge main into release-minst (2026-11-05)

Both parent review records are preserved below, release-minst's first. Only
record files and docs/index.md were manually resolved; see PLAN.md.

---

# MNIST.19 — Simplify the example support code

Independent review: round 1/3, verdict merge (2026-11-05), in
review/slices/MNIST.19.md. Nits left open for the maintainer: N-1
(`fit_intercept` follows the data's dtype; float32 in practice), N-2 (two jit
wrappers of `concordance_index`; `data` parameter shadows the module in
`run._concordance_seconds`), N-3 (notebook-only ruff exception, see below).

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

---

# Merge ci into main (2026-11-04)

Both parent review records are preserved below. Only PLAN.md and REVIEW.md
were manually resolved; executable files retain the automatic merge result.
This metadata resolution starts no implementation step.

---

Independent review: round 2/3

# GPU.2 — Manual Modal GPU check and README badge

Read PLAN.md GPU.2, `.github/workflows/gpu.yml`, `README.md`,
`scripts/gpu_ci.py`, and this review. Base commit: `fe71bd1` (GPU.1 plan
archive). Previous implementation endpoint: `409d773`.

Invariants:
- The workflow has only `workflow_dispatch`; a tag push does not spend GPU
  budget. It checks out `github.sha`, the commit selected when dispatch begins.
- A successful or failed Modal invocation determines the GitHub Actions job
  conclusion. The result ZIP is uploaded when one was produced, even on a
  failing invocation.
- The README badge is explicitly scoped to the latest manual GPU run on
  `main`; it is not a per-version record. The Actions check and run are the
  commit-specific record.
- Modal credentials are read only from repository secrets. The workflow has
  read-only GitHub contents permission and is not exposed to pull requests.

Validation plan: parse the workflow YAML while preserving the `on` key, check
dispatch/SHA/secrets/artifact/summary configuration, and run `git diff --check`.
No numerical or Python code changes; no new tests are applicable.

Uncertainties: GitHub Actions secrets must be configured by the maintainer
before a live run. A real Actions/Modal workflow run is outside local
validation.

Implementation and validation:
- `.github/workflows/gpu.yml` has only the `workflow_dispatch` trigger,
  checks out `${{ github.sha }}`, limits the GitHub job to six hours and the
  Modal run to five hours/$10, uploads any returned result ZIP for 90 days,
  and writes the selected ref, SHA, and job result to the Actions summary.
- `README.md` contains a workflow badge filtered to manual runs on `main` and
  explains that it represents the latest such run.
- YAML parsed with PyYAML `BaseLoader`; assertions for trigger, selected SHA,
  step-scoped Modal secrets, artifact upload, and summary passed.
- `git diff --check`: clean.
- No project tests or live workflow were run. GitHub secrets are not present
  until configured by the maintainer; the workflow must reach the default
  branch before its Run workflow button is available.

Round 1 finding S-1 resolved by the maintainer's direction: GPU CI is a
pre-tag check only for releases claiming the tested T4/CUDA 12 configuration.
The README badge alt text and explanation now name that configuration, and
CPU-only releases are explicitly left to CPU CI.

---

# A9.close — Reconcile completed A9 records

Read PLAN.md's A9.close section, plans/A9.md's closure note and retained
A9.d1 gate map, then plans/README.md's A9 row. Previous current review is
preserved in Git at a351952:REVIEW.md (CI.integration).

Invariants:
- Original plans/A9.md text and the integrated PLAN.md history remain
  verbatim after new wrappers. No unfinished unrelated work is discarded.
- The gate/narrative map is copied verbatim from 14155ed:REVIEW.md. Its
  historical pending/attribution language is retained, not a new run claim.
- A9.4 round 1 and A9.d1 round 2 both have independent merge verdicts.
  A9.d1's F-1/N-1/N-2 are fixed. Maintainer requested this bookkeeping.
- A9.3 remains withdrawn. A9 closure does not accept all Slice A/release
  work, resolve deviations or change other workstream ownership.
- No source/docs pages/tests/spec/dependency/protected paths are changed.
  No new test is appropriate for these record-only edits.

Spec context: 7.3, 10.3, 12.1–12.3. The retained gate map names each test,
artifact and docs narrative item with its original validation provenance.
The newer combined through-A10 result is cited from a351952:REVIEW.md,
not rerun. The final A9.d1 verdict was read from the independent report.

Validation: preservation/reference/scope assertions pass: both historical
plans and the gate map are verbatim; all three artifact paths and four
commit references exist; exactly the four intended record files differ.
Whitespace check: git diff --check exits 0. No pytest, numerical gate, benchmark, R, docs build or accelerator
execution in this bookkeeping step. No uncertainty about implementation is
introduced; overall Slice A/release acceptance remains maintainer-owned.
