# A8.4b — Exponential R reference comparisons (2026-10-26)

Approved by "review passed; implement next subslice" after a942479.
Branch: a8-1-r-generator. One test step, then stop for review.

## Read in this order

1. tests/reference/test_exponential_r.py: shared-loader fixtures, active-dtype
   inputs and unchanged float64 R expectations; explicit API/input-form tests.
2. Its direct survival/log-survival tests and separate scalar-total test.
3. tests/fixtures/exponential/README.md: updated comparison/loader evidence.
4. PLAN.md: A8.5 and the separate docs topic remain future work.

## Invariants and spec mapping

- Spec 3.3: read through r_fixture only. Cast times and rate to the process's
  JAX dtype; keep expected R doubles untouched. Assert output shape and active
  dtype for every numerical result. No JAX global configuration changes.
- Spec 4.6/12.1: compare all 16 observed-time survival and log-survival values
  for rates slow=0.1, generating=0.35, fast=1.2 against both R sources.
- Spec 6.1/12.1: compare every likelihood contribution for function/method
  entry points and raw/prepared inputs. Each numerical row test runs eagerly,
  under jax.jit, and row-wise jax.vmap against the same committed R expectation.
  The distribution is a dynamic argument to those transformed functions.
- Spec 4.7/6.2: valid fixture data and positive rates have clear data/parameter
  diagnostics in all three cases.
- Spec 12.1: scalar sums compare to separately recorded R package totals.
  They supplement, never replace, per-row assertions. Only existing
  ATOL/RTOL in tests/impl_tolerances.py are used; no thresholds change.

Every numerical expectation comes from the committed flexsurv/survreg fixture,
not from running survjax. The two R paths remain related numerical evidence;
this step covers finite interior exponential cases, not A7's delicate tails
or A9's broader derivative suite. No fixture parsing outside the shared loader.

No loader, payload, generator, library API/source, dependency, tolerance,
protected test or docs/ changes. The fixture README removes its obsolete
claim that comparisons/automatic loading remain future work. Docs to-do:
describe the scope and limits of this R evidence in A8.d1.

## Focused verification

With UV_CACHE_DIR=/private/tmp/survjax-uv-cache,
`uv run pytest -q tests/reference/test_exponential_r.py`:
JAX_ENABLE_X64=1: 45 passed in 3.61s;
JAX_ENABLE_X64=0: 45 passed in 3.37s.
Logs: /private/tmp/survjax-a84b-focused-x64.log and
/private/tmp/survjax-a84b-focused-x32.log.

Sensitivity control in a temporary test/fixture copy: monkeypatch both
likelihood entry points to roll their row output by one. No committed source
or fixture is edited. The scalar total is preserved, but the ordered per-row
R comparisons fail for every case/source/API/input combination:
24 assertion failures / 21 passed per dtype (1.68s x64 / 1.60s x32).
All failed IDs are row tests; all six scalar-total checks still pass.
Logs: /private/tmp/survjax-a84b-control-x64.log and
/private/tmp/survjax-a84b-control-x32.log. These expected child exit-1 results
are mutation evidence, not failures of the unchanged implementation.

## Final gate

`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`:
exit 0, check: all gates passed through A7.
Log: /private/tmp/survjax-a84b-gate.log.

- Guard OK; Ruff: 76 files formatted / All checks passed!
- mypy: no issues in 17 source files.
- Own float64: 880 passed, 7 skipped, 161.90s.
- Own float32: 886 passed, 1 skipped, 190.08s.
- Source doctests: 4 passed per mode, 1.10s / 1.01s.
- Independent float64: 2123 passed, 28 skipped, 223 deselected, 114.53s.
- Independent float32: 2127 passed, 24 skipped, 223 deselected, 111.26s.

Only result records and plan status are updated after the gate. Final Ruff
formatting/lint and git diff --check pass on the completed records.

## Limits and stop

The real independent bc acceptance fixture remains absent pending maintainer
handoff. No full through-A8 claim, R regeneration, docs build or new deviation.
Stop after this commit. Next: A8.5 Weibull row/grid comparisons, then A8.d1
reference-evidence docs after reviewed code merges.
