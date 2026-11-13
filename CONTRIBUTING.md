# Contributing to survjax

Bug reports, focused improvements and co-maintenance are welcome. The
repository is [vanamsterdam/survjax](https://github.com/vanamsterdam/survjax).

Open an issue before a substantial API change, new dependency or large feature.
Describe the problem and a small proposed scope. For bugs, include a minimal
synthetic example, shapes, dtype mode, Python/JAX/NumPy versions, expected
behavior and actual output. Keep patient records and patient-derived row-level
outputs out of issues, tests and commits.

## Set up the locked environment

Use Python 3.12 and the project's uv 0.11.19 baseline in your own checkout.
From its root, install development dependencies into its own `.venv`:

```sh
uv sync --locked --group dev
```

`make setup` runs the same command; `make help` lists the other entry points
(lint, format, typecheck, tests, the gate, docs). Each target wraps one command
or script named in this guide. Install the git hook from
`.pre-commit-config.yaml` with [prek](https://github.com/j178/prek), a
drop-in replacement for pre-commit, locked in the dev group:

```sh
make install-hooks   # uv run prek install
make hooks           # every hook on every file, as the CI hooks job runs it
```

Runtime dependencies are JAX and NumPy only. Example, inference and plotting
packages stay in optional development groups. Discuss dependency changes first;
keep `pyproject.toml` and `uv.lock` consistent. Never enable JAX float64 globally
in library code: runners choose the mode in separate processes.

## Check a change

While editing tests or code, run the touched test files in both dtype modes,
without a slow-test filter. For example, replace the path below with your files:

```sh
JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/unit/test_data_boundaries.py
JAX_ENABLE_X64=0 uv run --group test pytest -n 6 tests/unit/test_data_boundaries.py
```

The step check runs formatting, lint, typing, tests and source doctests:

```sh
scripts/check.sh --fast
```

Before a PR is ready, run the full gate or obtain its CI result for that commit:

```sh
scripts/check.sh
```

The committed gate selects independent acceptance through A12 and the completed
features listed in `scripts/spec-features.txt`, including censoring/truncation
and LogNormal/LogLogistic. Passing it does not certify all planned slices or
overall release readiness. The fast gate excludes listed
slow cases; CI runs the full gate. `scripts/run_tests.sh` runs the implementation
suite in both dtypes and excludes the independently authored `tests/spec/`.
Coordinate acceptance/contract changes with the maintainer; do not edit, skip,
xfail or weaken independent tests or protected gates to make a change pass.

Keep tests deterministic. Explain expected values using hand derivation,
analytical identities, high-precision calculations or committed reference
fixtures. Tolerances live in `tests/impl_tolerances.py`; report numerical failures
instead of loosening them. Numerical changes need the relevant boundary,
parameter-derivative and JAX transformation checks. Reference fixture READMEs
record provenance and pinned regeneration tools; ordinary tests need no R or
network. Use synthetic data, and do not fabricate reference outputs.

## Documentation and review

For documentation or examples, use the locked optional groups:

```sh
uv sync --locked --group dev --group docs --group examples
scripts/build_docs.sh
```

This executes runnable pages in separate dtype processes and fails on notebook
errors or Sphinx warnings. MNIST needs a hash-checked download on a cold cache;
see [verification and reproducibility](docs/technical-notes/verification.md).
Inspect rendered pages, figures and downloads; retain live execution.

Keep each PR focused and reviewable. Explain the problem, resulting behavior,
checks run with their outcomes, and material limits. Add tests with behavior
changes and type annotations/docstrings for public APIs. Keep documentation work
separate from code changes where practical and coordinate changes to navigation.
Preserve license notices and provenance for reused material; the project uses
Apache-2.0. Maintainers review changes before merging or publishing.

<!-- agent-docs-inspection -->

## Agent documentation inspection

For documentation layout checks and agent inspection, see
[the contributor inspection workflow](CONTRIBUTING_DOCS.md).
