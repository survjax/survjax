# Review: A7.2 follow-up — F1 formatting and F2 tiny derivatives

## What to read, in order

1. `tests/unit/test_likelihood_debug_nans.py`: `_check_eager`, then the two
   per-row built-in callers identifying the tiny-right observation column.
2. Mutation evidence below, especially the float32 zero-derivative failures.
3. The follow-up entry at the top of `PLAN.md` and the final gate log.

Authorized by "immediately implement the required fixes" after the updated
review of `8f903d7`. This is a small fix-up on A7. No source, public API,
constraint, numerical tolerance, dependency, independent-test or docs change.

## Required fixes

F2: the helper still checks ordinary gradients with ATOL/RTOL. It additionally
asserts known structural zero components exactly and checks observation column
4 of each per-row built-in gradient with TAIL_ATOL=0 and the unchanged
TAIL_RTOL. These components are the normal nonzero derivatives of tiny right
censoring: exponential about -1e-10, Weibull about 1.19e-9 and 2.5e-11.
The column is selected explicitly by the callers; no threshold or tolerance
number was introduced. Values/references and the 19-case test matrix are
unchanged. Spec 12.1 requires this relative-only comparison, so complete loss
of these derivatives must fail in float32 as well as float64.

F1: the earlier full gate ran before the last Python-fence edit in REVIEW.md.
Its passing result did not validate the committed `8f903d7` formatting, and
that final-snapshot passing-gate claim was wrong. Reproduced the failure:
`ruff format --check .` reported `1 file would be reformatted, 69 files
already formatted`, at the pytest.main snippet. Formatted that snippet and
replaced the old review with this follow-up record, using plain command text.
This review is completed before the final full gate and is not edited after it.

## Invariants and evidence

- Expectations remain independent NumPy/mathematical algebra on concrete,
  dtype-rounded inputs. No expected value comes from code under test.
- All selected tiny derivative components are normal in both dtype modes;
  zero absolute tolerance prevents an erroneous zero from passing. Known
  zero components also require exact equality. Other gradient components
  retain the existing ordinary tolerances.
- RootHazard, its S1 checks, boundary tests and all independent tests remain
  unchanged. No data-time or infinite-likelihood derivative guarantee is added.
- NumPy conversion in the helper happens on completed gradients/reference
  arrays, outside transformations. Separate processes select the dtype mode.

## Focused checks and mutation results

- `.venv/bin/ruff format tests/unit/test_likelihood_debug_nans.py REVIEW.md`:
  `2 files reformatted`; `.venv/bin/ruff check
  tests/unit/test_likelihood_debug_nans.py`: `All checks passed!`.
- `JAX_ENABLE_X64=1 SPEC_THROUGH=A7 .venv/bin/python -m pytest -q -ra
  tests/unit/test_likelihood_boundaries.py tests/unit/test_likelihood_debug_nans.py
  tests/spec/test_a7_likelihood_boundaries.py tests/spec/test_a7_debug_nans.py`:
  **53 passed in 10.73s**. Same with `JAX_ENABLE_X64=0`:
  **53 passed in 10.81s**. No focused skips or runtime failures.
- Copied src/ to `/private/tmp/survjax-a72-f2-mut-b02w0jch`. Only the copy
  stops gradients on survival at `0 < survival_time < 1e-8`, exactly the
  reviewer's F2 mutation. Repository source remains unchanged. Each process
  asserted that survjax.__file__ points to the copied package before pytest.
- Mutation command used separate `JAX_ENABLE_X64=1/0` processes,
  `PYTHONPATH=/private/tmp/survjax-a72-f2-mut-b02w0jch/src`,
  `PYTHONDONTWRITEBYTECODE=1`, and `.venv/bin/python -`; the import-asserting
  stdin wrapper called pytest.main with `-q -p no:cacheprovider --tb=short`
  and the exact per-row exponential and Weibull nodes in the debug module.
  Both runs exited 1: **2 failed in 1.35s** each, as required.
- Float32 now rejects both zeroed gradients with `atol=0, rtol=1e-4`.
  Exponential actual 0 versus about -1e-10; Weibull actual [0,0] versus about
  [1.19e-9,2.5e-11]. In float64 the exponential fails the new relative check;
  the Weibull fails its existing ordinary check first. This closes F2's
  specific float32 gap. These are selected mutation checks, not a general proof.

## Final-snapshot validation record

Run after this review and the test changes are finalized/formatted:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`.
The complete output, including exit status via the final gate message,
counts and timings, is saved in
[the final gate log](/private/tmp/survjax-a72-fixes-final-gate.log).
The shell uses pipefail while tee writes that log. The follow-up evidence in
PLAN.md and the final response record the observed result after completion;
this review is left unchanged. Final Ruff formatting and git diff checks
also run after the plan-only result update.

## Remaining work

No new spec ambiguity or deviation was found. This follow-up changes assertion
strength and formatting only. Numerical docs are still a separate pending
step; no docs build, R reference generation, install, GPU or performance check
was run here. Stop after this fix-up commit for review. A7/Slice A completion
is not claimed.
