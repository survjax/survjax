# A6.1 — Constraint vocabulary and declarations

Status: A6.1 implemented and ready for review on maintainer-created `A6.1`.
Through-A5 gate and both dtype docs builds pass. Stop here; no A6.2 work has
started. Planning/alignment is committed separately as `62636da`.
Spec revision 28, marker `0683e35`. The planning/alignment files carried into
this branch were authorized by the preceding planning request.

## Read first

1. `src/survjax/constraints.py`: the nine domain checks and bound validation.
2. `src/survjax/distributions/base.py`: param metadata and decoration-time
   validation, followed by the read-only parameter_constraints class mapping.
3. Exponential/Weibull field declarations and top-level constraints export.
4. `tests/unit/test_constraints.py`, `test_parameter_constraints.py` and the
   one-name update to the explicit export set in `test_import.py`.
   `docs/api.md` has the maintainer-approved minimal constraints autodoc entry.
5. The current A6 plan and D-023 in DEVIATIONS.md. D-013 has a revision-28
   follow-up note, with its status left to the maintainer.

## Invariants and scope

- Constraint True means in-domain; no parameter transforms, constructor checks
  or extra distribution PyTree leaves. Constructors never invoke constraints.
- Built-in constraint objects are frozen; declarations stay metadata. event_dim defaults to
  zero and cannot exceed the declared component axes. Registration records
  constrained own fields in declaration order, excluding nested distributions.
- Simplex reduces the final axis with the specified 4*n*eps tolerance in the
  parameter dtype and accepts zeros. This tolerance is the explicit diagnostic
  contract, not a probability floor or a change to numerical evaluation.
- frexp mantissas preserve subnormal signs at zero bounds, where ordinary JAX
  comparisons flush the smallest negative subnormal to zero. Signed zero
  remains valid for inclusive bounds. No global dtype settings change.
- parameter_issues(), ParameterIssues, conditions and likelihoods remain later
  approved steps. Narrative documentation remains on the plan's Docs to-do.
  Independent tests and specification sources were not changed or synced.

## Requirement evidence

- Spec 4.7 constraint domains, boundary polarity, malformed bounds and
  eager/jit/vmap: test_constraints.py, plus the independently authored
  test_a6_constraints.py read/executed directly from ../spectest.
- Spec 4.7/12.1 vector rules: own ordered tie/reduction cases, simplex softmax
  lengths 2/10/100/1000, zero/negative/nonfinite weights and a 1.001 sum;
  independent module adds per-constraint boundary neighbours/nonfinite cases.
- Spec 4.1/4.2/4.7 metadata: test_parameter_constraints.py checks declaration
  order, exact object identity, mapping immutability, non-callables and
  excessive event axes, permitted vector constraints, excluded nested fields,
  unchanged leaves and no constraint execution in eager/jit construction.
- Spec 4.1 built-in positive declarations and spec 10.2 export: the own
  declaration and import tests. The existing direct-method suite is included
  in the through-A5 gate rather than changing its implementation.

## Checks actually run

All uv commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Ruff format --check .: exit 0, 56 files already formatted.
- Ruff check .: exit 0, All checks passed!
- mypy src: exit 0, Success: no issues found in 14 source files.
- Final focused command, once with JAX_ENABLE_X64=1 and once with 0:
  SPEC_THROUGH=A6 uv run --locked --group test pytest
  tests/unit/test_constraints.py tests/unit/test_parameter_constraints.py
  tests/unit/test_import.py ../spectest/tests/spec/test_a6_constraints.py
  -q --tb=short -o cache_dir=/private/tmp/survjax-a61-pytest-<mode>.
  Both modes: 87 passed (exit 0). The external independent module alone has
  57 tests; the remaining 30 are own tests.
- Initial scripts/check.sh --through A5: exit 1 in x64 own tests,
  1 failed, 493 passed, 7 skipped. The sole failure is
  test_api_reference_lists_every_export: missing docs/api.md constraints entry.
  The gate stopped there; x32 own tests, doctests and independent through-A5
  tests were not reached in that invocation. This preceded the subnormal fix.
- Initial independent constraint runs: 1 failed, 56 passed in each dtype.
  The negative-subnormal unit_interval failure motivated the fix and own
  signed-zero/nextafter regression; the final focused runs above pass.
  Those first external runs also warned about an unwritable parent pytest
  cache. Later runs explicitly use the writable /private/tmp cache directory.

## Final gate and minimal API entry

The maintainer explicitly approved the minimal docs/api.md exception. Added
only an automodule entry for the nine constraints; no test was relaxed.

- Final scripts/check.sh --through A5: exit 0, guard ok (A6.1), Ruff format
  and lint clean, mypy clean (14 source files). Own tests: x64 495 passed,
  7 skipped in 37.92s; x32 501 passed, 1 skipped in 37.88s. Source doctests:
  4 passed in each mode. Independent through-A5: x64 1256 passed, 26 skipped
  in 24.68s; x32 1261 passed, 21 skipped in 23.08s. Ends with
  "check: all gates passed through A5".
- scripts/build_docs.sh initially exited 2 because the sandbox denied a local
  Jupyter kernel socket bind. The authorized retry exited 0, "build succeeded"
  in both modes; its kernels emit the existing unencrypted-local-TCP notices.
  Initial HTML inspection found three callable instances omitted by autodoc.
  An incremental explicit-autodata attempt exited 1 with two unresolved type
  warnings (_GreaterThan and bound) from the generated dataclass docstring.
  Added three source attribute-doc comments describing those domains, keeping
  the final docs/api.md change to one five-line automodule entry. Final
  scripts/build_docs.sh: exit 0, "build succeeded" for x32 and x64, with no
  Sphinx warnings. Inspected the generated API descriptions and verified all
  nine constraint IDs in both HTML outputs with the standard HTMLParser.
- After the final API correction, reran Ruff format/lint and mypy: same clean
  outputs as above. git diff --check is clean. Only the API entry and three
  attribute-doc comments changed after the successful runtime gate; numerical
  code and tests were unchanged.
- Final API export check: JAX_ENABLE_X64=0 uv run --locked --group test
  pytest tests/unit/test_api_reference.py -q: exit 0, 1 passed in 0.28s.
- No complete A6 gate, R fixture check, install check or performance benchmark
  was run; A6.2 onward is not implemented.

D-023 concerns A6.4 unsupported-mode diagnostics and does not affect A6.1.
