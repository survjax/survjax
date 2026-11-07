# D.LOGLOGISTIC.1 — LogLogistic implementation handoff

Independent review: round 1/3

## Read first

1. PLAN.md for scope, base and docs follow-up.
2. Accepted spec rev 47 sections 4.1, 4.2, 4.5–4.7 and 5.
3. src/survjax/distributions/loglogistic.py.
4. tests/unit/test_log_logistic.py, then the export-inventory changes.

## Invariants and coverage

Registration owns dtype normalization, parameter broadcasting, pytree rebuilding,
indexing, diagnostics, direct-time policy and grid shape. Both parameters declare
positive finite domains. from_log exponentiates normalized inputs and stores
natural leaves; no persistent log representation or new dependency exists.

The survival and CDF primitives implement the specified opposite softplus tails.
The hazard uses lower/upper algebraic forms to avoid subtracting large
beta*log(t/alpha) terms. The branch at t=alpha is smooth through second order;
finite-difference checks include the switch itself. Zero/infinity density and
hazard operands are substituted before logs/products. Base policy owns all
other time boundaries, including NaN/negative times. Finite zero limits retain
scale derivatives; the shape discontinuity at beta=1 is outside the guarantee.

- test_values_grids_and_transforms: hand-derived S=1/5, h=4/15, f=4/75;
  all seven pointwise methods under eager/jit/vmap, grid values/shapes,
  multiaxis broadcasting,
  indexing and tree-map reconstruction (4.1/4.2/5).
- test_boundaries: shapes below/equal/above one, negative/zero/infinite/NaN
  times and finite boundary derivatives (4.5/7.2).
- test_tails_against_mpmath: 160-digit independent log-space references,
  normal tiny tails and underflowed probabilities, no absolute tail allowance
  (4.1/7.1/7.2/12.1).
- test_second_parameter_derivatives: second-order check_grads in forward/reverse
  modes for every method below/at/above t=alpha, unchanged tolerance constants
  (7.3). Float32 step 1e-3 uses the documented O(eps/h) scale in existing
  LIKELIHOOD_GRAD_EPS; float64 uses 1e-4.
- test_log_constructor_and_diagnostics: natural leaves/dtype, from_log Hessian
  equivalence, invalid natural parameters and exp overflow (3.3/4.1/4.7).
- test_mixed_likelihood_debug_nans: exact/right contributions including censoring
  at zero, with hand-derived parameter gradient under eager debug-nans (7.2).
- Existing import and installed-wheel inventories include the new export.

## Scope and integration

The maintainer approved a size exception for the complete 343-line draft,
excluding generated timing metadata. No spec ambiguities found; no deviations. No summaries, sampling, AFT or other
Slice D features implemented. No docs edits: the API-reference assertion has an
explicit pending-docs entry linked to this plan, to remove in the docs step.

Parallel LogNormal integration overlaps: both package __init__.py files,
tests/unit/test_import.py, tests/install/test_wheel.py,
tests/unit/test_api_reference.py and generated tests/slow_tests.txt. Preserve
both families' export and pending-doc entries; regenerate slow durations after
integration. Family code and tests are separate. No shared numerical changes.

Base/worktree starting point: 3abefc5683a3aee8169eb2deffe57602897a3af3.
Previous completed step checkpoint: ff07cf9f7942db0ce0e5d3a27525e0c1ca3d3431
(A12 acceptance). No dependent family code step is required; docs follow-up
remains. The maintainer owns roadmap status and integration. Full Slice D is
not claimed. Runtime used: Python 3.12.9, JAX 0.11.2, NumPy 2.5.3.

## Validation

- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/unit/test_log_logistic.py tests/unit/test_import.py tests/unit/test_api_reference.py -q`: 27 passed, 24.93s after the spec-aligned module rename.
- Same command with `JAX_ENABLE_X64=0`: 27 passed, 19.33s.
- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/install/test_wheel.py -q`: after module rename, 4 passed, 59.71s; float32: 4 passed, 61.39s. Earlier pre-rename runs also passed (51.96s/46.00s).
- `uv run ruff format --check .`: 201 files already formatted.
- `uv run ruff check .`: All checks passed.
- `uv run mypy src`: Success: no issues found in 23 source files.
- `git diff --check`: clean.
- `scripts/check.sh --fast --through A12`: exit 0. Implementer tests:
  float64 1095 passed / 8 skipped; float32 1100 passed / 3 skipped.
  Source doctests: 4 passed each mode. Spec tests: float64 3396 passed /
  33 skipped; float32 3389 passed / 40 skipped. Two existing NumPy warnings
  each mode in test_a11_ph_direct_methods.py:78 (log(0), 0*log(0) in reference).
  The gate used the preceding slow list, so it included the new family tests.
- `scripts/update_slow_tests.sh --through A12`: exit 0; generated 1023 slow
  test entries. Timing remeasurement changes existing entries as well as
  adding family tests; commit the generated file as required. The gate above
  used the earlier, smaller list and therefore exercised more tests.
  Its initial float64 measurement captured
  the old installed-wheel export inventory failure (1 failed, 1334 passed,
  8 skipped); the explicit wheel reruns above verify the corrected inventory.
  Timing collection is not claimed as a successful full Tier 2 gate.
- No R generator, docs build or full Tier 2 gate was run for this step.

## Independent review round 1 — maintainer (incomplete)

Reviewed commit: 2b70e8ca56259fd7562dd127ec52e96180f4372d.
Review file: /Users/wamster3/dev/survjax/review/slices/D.LOGLOGISTIC.1.md.
The reviewer could not check out the commit or execute tests because of sandbox
permissions. Its archive's spec symlink was unavailable, so it did not read the
spec. No independent gates or mutations ran. Final verdict: `maintainer`.
No second review was requested: a round was written, so the skill prohibits a
retry for a different verdict. Implementation and tests remain unchanged.

F-1 (should-fix, disputed): the report reasons about raw softplus primitives at
zero/infinity. Accepted spec 4.2/4.5 explicitly calls these primitives only at
finite positive times. `register_family` wraps every direct method with
`_survival_methods.apply_policy`: for log_survival/log_cdf, zero and infinity
are substituted with stop-gradient reference times BEFORE raw family code is
called. The wrapper then restores policy values, with zero parameter gradients
for this proper family. Spec 4.5 even uses a naive softplus log-logistic custom
family as the example of this guarantee. Repeating boundary policy in these
primitives would duplicate the registered base's responsibility. The reviewer's
proposed shape-gradient cases are useful additional coverage, but the asserted
raw-boundary execution is prevented by the existing wrapper. No speculative
family-code fix was made; the maintainer can review this disagreement.

F-2 (should-fix, open): the repository requires running the full duration script
when new tests exceed one second and committing its generated list. The script
measures every implementer/spec test in both modes, not just this family's
entries, and says not to edit its output by hand. This full measurement overlapped
our fast gate, touched tests and wheel runs; concurrent load may inflate timings.
The generated list grew from 424 to 1023 entries, so later fast gates deselect
more existing tests. Our successful A12 fast gate used the PREVIOUS list and did
not gain a weaker selection from this change. The maintainer should decide
whether to accept the generated calibration or regenerate it on an otherwise
idle machine / authorize a narrower metadata update. No handmade filtering was
applied to conceal the measured results.

N-1: size exception is explicitly approved above. N-2: blank lines in the
from_log docstring are cosmetic; static checks pass. No implementation changes
were made after the maintainer verdict. Remaining work is review-environment
repair, disposition of F-1/F-2, integration with LogNormal and the separate docs
follow-up; no next family substep has been started.
