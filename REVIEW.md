# A8.2 — Synthetic bc-like Weibull generator

Implemented atop accepted A8.1 (`07c6d24`) on `a8-1-r-generator`.
Review this step with `git diff 07c6d24..HEAD`; A8.1 is intentionally unmerged.
One commit, then stop. No merge or A8.3 payloads; the maintainer defers merging
until at least A8.3. The code/fixture-README addition is 297 lines.

## Read in this order

1. `scripts/fixtures/generate_bc_like.R`: fixed parameter vectors and package
   helpers; simulation; per-row and grid checks; then export.
2. `tests/fixtures/bc_like/README.md`: schema, commands and generator limits.
3. `PLAN.md`: A8.2 implementation details and remaining integration steps.

## Invariants and requirement mapping

- Spec 10.3/10.4: check exact R 4.4.2 / flexsurv 2.3.2 / survival 3.7.0 before
  loading packages. Seed 8201 and explicit RNG settings generate synthetic
  rows only. No `bc`, patient rows or real-data log-likelihood totals are read.
- Spec 11.1: 686 rows, Good/Medium/Poor blocks of 229/229/228, fitted generating
  vector from the spec, independent Uniform(0.5, 7.5) year censoring. The row
  IDs/order, group sizes, finite positive times and both event kinds are checked.
  Realized events are 52/91/150 (293 total), with 393 censored. These are
  similar to the cohort's pattern; matching the exact real counts is not claimed.
- Spec 4.6/11.1: all three specified vectors are evaluated without fitting.
  Check `b_g=b_Good*exp(beta_g)`, `mu_g=log(b_g)`, `sigma=1/k` and treatment
  contrasts with Good baseline. Each case exports natural and AFT parameters;
  label/parameter checks prevent one case silently replacing another.
- Spec 10.4/12.1: flexsurv receives four fixed natural-scale initial values;
  the result must retain them and report zero free parameters. The survreg
  path uses fixed `offset(mu)`, zero intercept, fixed scale and zero iterations;
  assertions require unchanged locations, scale and intercept. Density/CDF
  composition agrees with every singleton total and the full-data total.
  Both packages agree per row and in total at the ordinary float64 targets.
- Spec 5/11.1: paired flexsurv predictions check returned time/group order.
  Grids have 141 times (0–7 by 0.05), Good/Medium/Poor columns, shape (141,3),
  and exactly S(0)=1 in both paths. Observed survival/log survival and both
  grids are compared. Matrix export checks that its row arrays reconstruct
  the original matrix exactly, preventing a serialization transpose.
- Spec 10.4: retain A8.1's base-R 17-digit numeric formatting, exact CSV text
  read-back, written-JSON check, case checks and generator/CSV MD5 hashes.
  Header includes versions/platform, RNG, units, group/event counts, generating
  parameters and evaluation paths. No Python runtime dependency or framework.

## Commands and results

- `../scripts/spec-since.sh dev-a8`: exit 0, reports changes through `fd150db`
  (spec revision 30). Read the current relevant contract; inherited `.spec-seen`
  stays unchanged pending A6 integration. No spec or protected-path changes.
- Installed `?flexsurvreg`, `?summary.flexsurvreg`, `?survreg` documentation:
  checked fixed parameters, paired predictions and fixed-scale evaluation.
- `Rscript --vanilla scripts/fixtures/generate_bc_like.R --check-only`: exit 0
  outside the sandbox for the known OpenMP issue; `Checks passed: 686 synthetic
  rows, 3 fixed vectors, 141-by-3 grids, both R paths`.
- `python3 /private/tmp/survjax-a82-check.py`: exit 0 outside the sandbox;
  actual generator in check-only/full-export modes plus scratch mutations:
  - Two full exports pass; CSV and JSON are byte-identical across runs. JSON
    parses, hashes and fixed-vector labels/values match the spec and source.
  - All per-row likelihoods/log survivals/survivals and grid values match the
    spec 11.1 closed forms. Max absolute error 7.96e-13; max relative error
    5.58e-14. Python checks the stated algebra; fixture values come from R.
  - All 15,770 JSON numbers and all 686 CSV times exactly match original R `%a`
    values on Python read-back. This avoids the pinned R decimal-parser issue
    handled by the maintainer's A8.1 decision; no precision tolerance is weakened.
  - Unknown `--chek-only` fails before writing. All temporary payloads/mutants
    are removed after success; no generated files remain in the checkout.

| Mutation | Check that rejects it |
| --- | --- |
| sigma=k instead of 1/k | flexsurv/survreg agreement |
| reversed groups in survreg grid | grid agreement |
| matrix exported by columns | exact matrix reconstruction |
| CSV event replaced by 1-event | CSV text read-back |
| JSON formatted with six digits | 17-digit text check |
| alternative case replaced by unit_shape | label/parameter identity |

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A5`:
  exit 0, `check: all gates passed through A5`. Ruff: 55 files already
  formatted, all checks passed; mypy: no issues in 13 source files.
  Own tests: x64 466 passed/7 skipped; x32 472 passed/1 skipped.
  Doctests: 4 passed per mode. Spec tests: x64 1256 passed/26 skipped;
  x32 1261 passed/21 skipped. Log: `/private/tmp/survjax-a82-gate.log`.
- `git diff --check`: exit 0, no output.

## Limits and next review stop

The two R paths are not independent oracles. CDF complements are used only
on these checked finite interior cases; delicate tails remain A7's scope.
Exact decimal read-back is not claimed for the pinned R parser itself.
No Python/library, dependency, independent-test or narrative-doc changes.
No survjax/R comparison, through-A8 gate or completion claim. A8.3 still owns
committed generation/acceptance, A8.4/A8.5 own integration, A8.d1 owns docs.
No unresolved ambiguity or new DEVIATIONS entry. Stop for review after commit.
