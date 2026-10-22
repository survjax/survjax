# A6.4c — Public likelihood function and base method

Status: implemented, ready for review on `A6`, based on `6a35b7a`,
authorized by "do it".
Spec revision 32, marker `cf4fef9`; independent snapshot `08d97a6`.
Stop after this integration commit; no A6.5 work.

## Read first

1. `src/survjax/likelihood.py::_resolve_data` and `log_likelihood`: the shared
   argument convention, checked raw constructor and single core call.
2. `src/survjax/distributions/base.py::SurvivalDistribution.log_likelihood`:
   typed delegation, with a local evaluator import to avoid a cycle.
3. `tests/unit/test_likelihood_entry_points.py`: small input-contract cases,
   named transforms and hand-derived values/derivatives.
4. `src/survjax/__init__.py`, `tests/unit/test_import.py`, `docs/api.md`:
   exactly one new export and the authorized generated API entry.

## Invariants and test/spec mapping

- Spec 4.2/6.1/7.1: the base method forwards directly to the function. One
  resolver accepts prepared data unchanged or calls the existing pure checked
  SurvivalData constructor. It checks only static argument presence/types;
  runtime values never enter Python conditionals or a host checkpoint.
- Raw and prepared forms preserve stored raw-input issue evidence. Numerical
  preparation, dtype promotion, current-field checks, unsupported-mode masks
  and invalid-parameter gradient isolation are unchanged from A6.4b2.
- Spec 6.1: incomplete/mixed forms, positional raw arrays, extra constructor
  options and incompatible raw shapes raise TypeError. Own tests exercise
  each behavior through both entry points and a static error under jit.
- Spec 6.2/7.1: invalid raw event code 2 can normalize to finite exact fields,
  but its contribution remains NaN. Own evidence tests check eager/jit/vmap
  through both entry points/forms and preserve the valid neighbors. Existing
  core tests retain normalized-field reconstruction coverage.
- Spec 7.1/7.3/12.1: own analytical anchors use rate=2 and times [1,2,3],
  event=[1,0,1]. Contributions are [log(2)-2,-4,log(2)-6]; summed likelihood
  is 2*log(r)-6*r, with derivative -5 and second derivative -0.5 at r=2.
  Named jacfwd, jacrev and Hessian cases cover both entry points/forms,
  eager and jit. Existing tolerances are unchanged.
- Independent entry-point/method suites additionally cover inherited custom
  family methods, nested families, broadcasting, jit/vmap, function/method
  derivative equality, compile reuse and device-transfer restrictions.
  Independent isolation and unsupported-mode suites cover the revision-32
  parameter guarantee and revision-31 mode policy at the public API.
- Spec 10.2/10.5: import/export and API-coverage tests include log_likelihood.
  Public docstrings cover arguments, shapes, returns and errors. The sole
  docs edit is the generated function entry under standing authorization.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Focused own API tests, separate JAX_ENABLE_X64=1/0:
  `uv run --locked --group test pytest
  tests/unit/test_likelihood_entry_points.py -q --tb=short`.
  Both exit 0: 52 passed (4.53s each).
- Focused independent acceptance tests, separate JAX_ENABLE_X64=1/0 and
  SPEC_THROUGH=A6: `uv run --locked --group test pytest` with
  `tests/spec/test_a6_likelihood_entry_points.py`,
  `tests/spec/test_a6_likelihood_method.py`,
  `tests/spec/test_a6_likelihood_gradient_isolation.py` and
  `tests/spec/test_a6_unsupported_modes.py`, `-q --tb=short`.
  Both exit 0: 544 passed (68.58s/68.43s).
  Logs: `/private/tmp/survjax-a64c-focused-x64.log` and `...-x32.log`.
- Ruff formatting/lint and mypy src: exit 0; lint clean and no typing issues
  in 17 source files. The full gate checks the final formatted version too.
- `scripts/build_docs.sh`: initial sandbox run exited 2 because Jupyter could
  not bind a local kernel port (PermissionError). Retried with permission
  outside the sandbox: exit 0, both dtype builds report `build succeeded`.
  Log: `/private/tmp/survjax-a64c-docs.log`. Jupyter emitted its kernel TCP
  advisory; Sphinx reported no build warnings/errors. Inspected both generated
  api.html entries: correct keyword-only signature and complete argument,
  return and error text. An initial HTML extraction stopped at a nested
  argument-list tag; correcting the extraction to the section boundary passed.
- `scripts/check.sh --through A6`: exit 0, final output
  `check: all gates passed through A6`. Guard: `ok (A6)`; Ruff:
  `65 files already formatted`, `All checks passed!`; mypy:
  `Success: no issues found in 17 source files`.
  Own x64: 700 passed, 7 skipped (127.27s); x32: 706 passed, 1 skipped
  (122.99s). Source doctests: 4 passed per mode (0.89s/0.86s).
  Independent through-A6 x64: 2023 passed, 28 skipped (92.53s);
  x32: 2027 passed, 24 skipped (88.82s). No independent tests were modified,
  skipped or xfailed by this step; the suite retains its existing dtype and
  optional-dependency skips (including optional NumPyro interoperability).
  Full log: `/private/tmp/survjax-a64c-check.log`.
- `../scripts/spec-since.sh dev`: exit 0, unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits and uncertainties

D-026 remains open: edited normalized fields may cause NaN without stored
issue bits, so the spec recipe's remainder needs a separate wording decision.
No spec or independent-test changes were made. Narrative likelihood docs,
method reference and A6.5 composition checks remain later steps. No new
runtime/test dependencies, result containers or later-slice modes. R fixtures,
wheel installation and CPU baseline were not run here; the A6 gate alone is
not the Slice A completion gate.

Source/test/docstring/API-entry diff: 234 changed lines. The new own test file
has 141 lines; likelihood.py has 136. Existing base.py remains larger than the
preferred module size; no unrelated module split is included in this step.
