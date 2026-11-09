# D.LOGLOGISTIC.2 — log-logistic documentation handoff

Independent review: round 1/3

## Read first

1. PLAN.md and linked D.LOGLOGISTIC.1 docs to-do.
2. docs/builtin-families.md, new LogLogistic section and four-family intro.
3. docs/api.md, LogLogistic autodoc entry.
4. tests/unit/test_api_reference.py, removal of the temporary docs exemption.

Current contract: accepted spec rev 52 sections 3.3, 4.1, 4.5–4.7, 5 and 10.5.
The quadrature changes in recent revisions are outside this docs topic.

## Claims and executable evidence

The narrative covers the implemented natural shape/scale API, positive finite
domains, time units, mathematical median and flexsurv llogis mapping. Log-time
logistic location log(alpha) and scale 1/beta follow from the defining survival
formula. For t>0, the hazard derivative has the sign of
(beta-1)-(t/alpha)**beta; this gives the stated decreasing/single-peaked behavior.
No R comparison was run or claimed: parameter names/mapping follow spec 4.6.

Examples assert independently derived identities:

- beta=2, alpha=10: S(10)=1/(1+1)=1/2 and S(20)=1/(1+4)=1/5.
- Three broadcast shapes share scale 10, with grid shape (4,3), S(0)=1,
  S(infinity)=0 and S(10)=1/2. from_log stores natural parameters.
- Opposite softplus log tails remain negative at t=1e-10 and 1e10, with
  magnitudes around 1e-22 and 1e-18, normal in both floating dtypes.

Zero density/hazard limits and the undefined shape derivative at beta=1, t=0
are described consistently with the existing implementation and tests. The
median is a mathematical property, not a call to a not-yet-implemented summary
method. No sampling, summaries or unrelated Slice D API is introduced.

The API-reference assertion now checks every export with no pending-docs
exceptions. All LogNormal docs and records are preserved. No executable source code,
dependencies, protected paths or numerical tolerances changed. One log_hazard
Returns description is corrected because Napoleon treated its prose colon
as a type declaration.

## Validation

- `uv run ruff format --check .`: 210 files already formatted.
- `uv run ruff check .`: All checks passed.
- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/unit/test_api_reference.py -q`: 1 passed, 1.19s.
- Same command with JAX_ENABLE_X64=0: 1 passed, 0.79s.
- First `scripts/build_docs.sh`: float32 notebook examples executed, then
  strict Sphinx failed with one warning, an unresolved class reference
  "Broadcast log hazard. At zero" from the old Returns prose. Reworded that
  docstring to remove the ambiguous colon. Float64 was not reached in that run.
- Second `scripts/build_docs.sh`: exit 0, strict Sphinx builds succeeded in float32 and float64, all notebook pages executed, ending with "Verified both dtypes" and build identity state success. This was a pre-commit dirty build at efd1b5a, labelled as such in docs/_build/build-identity.txt. Standard Jupyter transport notices appeared, with no remaining Sphinx warning failure.
- Inspected both generated builtin-families.html sections and api.html. Both modes show the expected dtype, constructor/grid/tail outputs, both family API entries and all six requested LogLogistic methods. The log_hazard Returns paragraph is rendered as prose, not as a type cross-reference.
- `git diff --check`: clean.
- `scripts/check.sh --fast --through A12`: exit 0. Formatting: 211 files already formatted; lint passed; mypy: no issues in 24 source files. Implementer tests: float64 1272 passed / 8 skipped (24.80s), float32 1277 passed / 3 skipped (26.65s). Source doctests: 4 passed each mode. Spec tests: float64 3563 passed / 32 skipped (46.66s), float32 3548 passed / 47 skipped (47.13s), two existing NumPy warnings each mode at test_a11_ph_direct_methods.py:78.
- No full non-fast gate or R comparison was run; this is a docs-only change apart from one docstring and the strengthened reference guard.

## Commits and remaining work

Base and preceding integration-step commit:
efd1b5a4e37a9822c580eeb36fdaaa898275c833.
Preceding family code-step final commit:
c31f73368c07786c95ed48371ba4f51e59485ad5; implementation 2b70e8c.
Original worktree base: 3abefc5683a3aee8169eb2deffe57602897a3af3.

This step covers all family docs to-do items in D.LOGLOGISTIC.1/PLAN.md. No
further family-only implementation or docs substeps are planned. The full
non-fast gate, merge to main and publishing remain maintainer actions.
No spec questions or deviations; no shared roadmap/progress record edited.
