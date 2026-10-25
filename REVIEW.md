# LL.1 — Remove the public likelihood function (spec revision 35)

Read `src/survjax/distributions/base.py` first for the remaining public method,
then `src/survjax/__init__.py` and `src/survjax/likelihood.py` for the removed
export/wrapper. Read `tests/unit/test_import.py` and
`tests/unit/test_likelihood_entry_points.py` next, followed by the small test
migrations. The current entry at the top of PLAN.md records scope and results.

The method now calls the existing private input resolver and evaluator
directly. Its signature is unchanged. The public function is deleted, including
from its old module; no compatibility alias was added. The method's docstring
now states the input, output, failure and diagnostic contract itself.

Invariants and requirement evidence:

- Spec 6.1/10.2/12.1: `test_public_package_imports` checks the exact export set;
  `test_likelihood_is_only_available_as_a_method` checks absence before and
  after importing the private evaluator module and calling the method.
- Spec 6.1/6.2/7.1: the entry-point tests retain mixed/missing-form, keyword,
  shape and invalid-indicator checks, including jit/vmap. `_resolve_data`
  still uses the checked SurvivalData constructor for raw inputs.
- Spec 7.3/12.1: 16 new cases compare per-row raw/prepared values exactly and
  their forward/reverse Jacobians and Hessians with existing tolerances,
  for both built-ins, eager and compiled. Existing analytical exponential
  anchors and both built-ins' order-two finite-difference checks remain.
- Spec 3.3/7.1–7.3: no numerical helper was changed. An AST comparison with
  HEAD verified all five private functions in likelihood.py are identical.
  The core/parameter suites still check dtype promotion, unsupported/invalid
  rows and parameter isolation; tails/boundaries retain mpmath/analytical
  references. Debug-NaN tests retain shared/per-row parameters and the singular
  custom density at zero, eager/jit/vmap, including ordinary and debug modes.
- The old public-function matrix is gone: 26 entry-point duplicates, eight
  finite-difference duplicates, two debug duplicates and six invalid-function-
  receiver cases removed. The 17 new cases give a net decrease of 25 per dtype.
  No numerical tolerances or independent tests were changed.

Checks actually completed before the final gate:

- `uv run ruff format --check .`: `71 files already formatted`.
- `uv run ruff check .`: `All checks passed!` after correcting two initial
  E731 findings in the new test's local functions.
- `uv run mypy src`: `Success: no issues found in 17 source files`.
- `uv run pytest -q` on test_import, test_likelihood_entry_points,
  test_likelihood_transformations, test_likelihood_tails,
  test_likelihood_boundaries, test_likelihood_debug_nans,
  test_likelihood_core and test_likelihood_parameters (all under tests/unit):
  205 passed in 114.52s with JAX_ENABLE_X64=1; 205 passed in 114.19s with
  JAX_ENABLE_X64=0. No skips. Logs:
  `/private/tmp/survjax-ll1-focused-x1.log` and `survjax-ll1-focused-x0.log`.
- `SPEC_THROUGH=A7 uv run pytest -q` on
  tests/spec/test_a6_likelihood_entry_points.py and
  tests/spec/test_a6_likelihood_method.py: 35 passed in 7.01s float64 /
  7.03s float32, separate processes, no skips. Logs:
  `/private/tmp/survjax-ll1-spec-focused-x1.log` and
  `survjax-ll1-spec-focused-x0.log`.
- Tracked-file audit: the only old API references in current source, own
  tests, docs and notebook content are four docs references: api.md:12,
  likelihood.md:109/129 and simulation-guide.md:48. The distribution guide's
  local function is unrelated. `git diff --check`: exit 0.

All uv commands above used `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.
After saving this review, run that same cache setting with
`scripts/check.sh --through A7`, capturing the complete output in
[/private/tmp/survjax-ll1-final-gate.log](/private/tmp/survjax-ll1-final-gate.log).
The final exit status/counts belong in PLAN.md after completion; this review
makes no advance pass claim and stays unchanged through that gate. Final Ruff
and diff checks follow any plan-only result update.

LL.2 remains pending separate approval. Docs are untouched and still contain
the obsolete references listed above; the executable tutorial/API reference
will fail to build until LL.2. No docs build or R run was attempted. A8–A10
acceptance and the remaining Slice A gate are outside this step. No new spec
ambiguity was found. Commit this step on remove-ll, then stop for review.
