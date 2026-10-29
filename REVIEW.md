Independent review: round 2/3

# RMST.2 — Weibull RMST values and safe tails

Clean starting commit 01b29dc9e0ea5830580770c0183865cb159fc376 on release-rmst;
original workstream base 8757c0fda78e34e6e818249adedd2a144a33c33d. Spec rev 37.
Read PLAN.md RMST.2, spec 8.7 and 12.1, src/survjax/_numerics.py,
Weibull.rmst, Exponential.rmst's validation refactor, then test_weibull_rmst.py.
No unfinished PLAN history archived. RMST.1 round 2 returned merge.

## Invariants and evidence

- _numerics.py is named explicitly in spec 10.2; this step needs it to keep
  Weibull's public method thin and its file at 300 lines. It contains only
  shared RMST cutoff validation and one private numerical expression. No
  public exports, dependencies, generic summary layer or custom JVP added.
- _rmst_cutoff does dtype/shape checks only, using JAX conversion/promotion.
  Exponential now shares it; formula, masking and zero behavior unchanged.
  RMST.1 tests remain independent regression evidence; N-2 validation reuse
  is addressed without growing the already-oversized base.py.
- Weibull parameter issues and invalid cutoffs are masked before numerical
  operations on every leaf. The numerical zero case substitutes before logs
  and returns the active cutoff: value/parameter derivatives=0, d/dtau=1.
  Tests cover zero, mixed-invalid parameters, cutoff tangents, shapes and
  preservation of explicit float32 parameters with integer cutoffs.
- Evaluate log x=k*(log tau-log b), not (tau/b)**k. At x<0.1 integrate the
  exponential series termwise; 12 terms leave remainder bounded by
  0.1**13/13! < 2e-23. All inactive series/gamma/mean branches have safe
  parameters and inputs before exp/log/special functions. At x>64+4/k use
  the mean, with a threshold classified in log space. For a=1/k<=5 the
  gamma tail above 64+4a is far below the ordinary value targets. The middle
  combines log scale, gammaln and log gammainc before exponentiation, avoiding
  overflow of the mean factor when the product is representable. A lost
  gamma probability returns NaN rather than a false zero (spec 7.1).
- test_weibull_rmst_values_against_quadrature: mpmath 60-digit quadrature of
  survival, split by decades and at the scale, at k={0.2,0.5,1,2,10,50} and
  tau/b={0.01,0.1,0.5,1,10,100,1e5}; scale factors 1e-4,1,1e4; eager,
  jit and jit-vmap. Zero-absolute existing tail tolerances, no new tolerances.
- test_weibull_rmst_power_tails: k=50 with x=1e-400 and x=1e400 (beyond
  either dtype, while RMST is finite), under debug_nans. Independent lower
  integral and analytical b*Gamma(1+1/k) upper expectation.
- Shape-one test: analytical exponential expression and explicit (m,S,2)
  cutoff axes, including equal m and S. Scalar cutoff keeps batch shape.
- Derivative smoke: mpmath differentiation of that independent integral for
  first derivatives in k,b,tau, and Hessian in b,tau, eager AD under jit;
  check_grads order 2, forward/reverse. O(1) point (2,1.25,0.75) uses the
  existing central derivative tolerances/finite-difference steps (O(h^2)
  truncation and eps/h rounding). Native shape Hessian raises
  NotImplementedError, as required; no higher-shape derivative promise.

## Checks run

Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1, pytest 9.1.1,
CPU. Unchanged dependency metadata/lockfile; separate local environment.

- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_weibull_rmst.py
  tests/unit/test_rmst.py tests/unit/test_weibull.py tests/unit/test_exponential.py
  -q`: 57 passed, 1 existing skip, 30.13s. With =0: 58 passed, 30.54s.
- Final Weibull-only `JAX_ENABLE_X64=1 uv run --group test pytest
  tests/unit/test_weibull_rmst.py -q`: 7 passed, 20.06s. With =0: 7 passed,
  19.80s. These final tests include the simplified equivalent Hessian
  reference expression; the larger focused run preceded that test-only edit.
- `uv run ruff format --check .`: 114 files already formatted;
  `uv run ruff check .`: All checks passed; `uv run mypy src`: no issues in
  21 source files. An initial lint pass caught unbound loop variables in a
  test closure (B023); bound them before final runs. `git diff --check`: clean.
- Final corrected focused run, same four-module commands above: 57 passed,
  1 skipped (35.50s), and 58 passed (36.26s). Statics still clean.
- Mixed-dtype regression proof: `JAX_ENABLE_X64=1 uv run --group test
  python -`, in-memory _numerics copy with only shape/scale promotion removed,
  then pytest on test_weibull_rmst_invalid_values_and_shapes: 1 failed in
  0.97s, actual inf versus expected 24*b=2.4e39. No on-disk mutation.
- Initial `scripts/check_wheel.sh --through A10`: exit 0; own tests 1069
  passed/7 skipped (89.58s), 1075 passed/1 skipped (82.45s); doctests 4 per
  mode (1.29s/1.20s); independent x64 2332 passed/28 skipped (85.09s),
  x32 2322 passed/38 skipped (70.26s).
  This wheel predates the mixed-dtype correction. Its SHA-256 is
  df30106734c75c405325297b3baf10bbf028d62ce4867a5ebfd0b88caa367cc6.
- Corrected `scripts/check_wheel.sh --through A10`: own tests 1069 passed,
  7 skipped (167.04s), and 1075 passed, 1 skipped (163.58s); source doctests
  4 passed per mode (2.55s/2.09s). Independent through-A10: 2332 passed,
  28 skipped (128.89s), and 2322 passed, 38 skipped (97.29s). Existing skips
  only. Exit 0: check: all gates passed through A10.
  Corrected wheel SHA-256:
  728bfb03ad0cc949d023920ce328094530e214400c1d0740cbdee3a0c2295a2c,
  artifact /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.nbZOwa/survjax-0.0.1.dev0-py3-none-any.whl.
  No production/test edits after this final wheel build.

## Limits and handoff

309 changed source/test lines, including the validation extraction and a
9-line mixed-dtype correction/regression after the initial 300-line version; no tests
removed from RMST.1. New private helper 70 lines, Weibull file 300 lines.
No base.py, wrapper/export, docs, spec, acceptance-test or protected-path edit.
This step completes value evidence with derivative smoke only. RMST.3 still
owns full-range first derivatives, value-scaled measured derivative tolerance
factors, branch-adjacent checks and additional shared-parameter/Hessian
isolation evidence. Existing tolerances unchanged. It must retain N-1's
absolute-scale rule for the cutoff derivative, without relative-tail claims.

RMST.1 S-1 stays a nonblocking maintainer question: invalid entries' own
zero tangents versus the 8.4 NaN-tangent bullet. No behavior change or
resolution claimed. No new implementation deviation. Keep PH wrappers and
exports with release-ph; hand built-in core ownership over only after the
reviewed Weibull core/evidence merge, before private PH closure edits.
_numerics.py remains RMST-owned through RMST.3; include it in the core
handoff to release-ph.
The executable RMST docs topic follows core/PH integration and must state
Weibull's higher shape-derivative limitation. A9/A10 docs remain separately
owned. Reviewer-owned mutation worktree cleanup is not ours.

## Round 1 response and round 2 request

Round 1 on b552201: fix-first for F-1, untested safe-input substitutions in
inactive tail branches. The implementation is unchanged. Add 10 test lines
inside test_weibull_rmst_power_tails: jit-vmap of jacfwd and jacrev in
(k,b,tau), at (50,2,2e-8) and (50,2,2e8); every derivative must be finite.
The cutoff derivatives also equal [1,0], since survival rounds to those
values in these respective tails. No new tolerance or stronger promise.

- `JAX_ENABLE_X64=1 uv run --group test pytest tests/unit/test_weibull_rmst.py
  tests/unit/test_rmst.py -q`: 21 passed, 18.19s. With =0: 21 passed, 18.05s.
- In-memory mutation proofs, separate processes for both dtypes, using
  `JAX_ENABLE_X64=1` / `=0` `uv run --group test python -`, then pytest on
  test_weibull_rmst_power_tails: M1 (series_x=exp(log_x)) fails in both
  modes (1 failed in 1.23s/1.26s); M2 (gamma_x=exp(log_x)) also fails in
  both (1 failed in 1.19s/1.17s). Each fails the finite-gradient assertion
  with the predicted NaNs. No source file or reviewer worktree mutated.
- Ruff format: 114 files formatted; Ruff check: All checks passed; mypy:
  no issues in 21 source files; git diff --check clean. Earlier complete
  fresh-wheel gate covers identical production code; not repeated for this
  test-only follow-up. Initial source/test diff 309 lines; follow-up 10 lines.

N-1 addressed in PLAN.md's docs to-do: accuracy promised for k in [0.2,50]
and tau/b in [0.01,1e5]; valid parameters outside this range can give NaN
in float32 when the gamma probability underflows. No numerical scope change.
RMST.1 N-2 is confirmed fixed by shared cutoff validation; RMST.1 N-1 remains
RMST.3 full-range derivative work. RMST.1 S-1 stays for the maintainer with
no change to invalid entries' own tangents.

Reviewer independently ran check.sh through A10: all statics, own tests and
doctests passed; x64 acceptance 2327 passed/28 skipped with 5 setup errors
because SURVJAX_WHEEL was unset. Direct x32 acceptance without wheel-smoke
module: 2317 passed/38 skipped. It did not repeat the fresh-wheel gate,
which its permissions refused in RMST.1. Our final fresh-wheel pass above
includes those five tests in both modes. Its independent value/branch probes
passed the required domain; they supplement our suite, not our own runs.
Reviewer mutation worktree /tmp/survjax-mut-RMST.2 remains with source restored,
after cleanup was refused. Leave that reviewer-owned artifact to maintainer
cleanup. Commit this bounded fix-up and request round 2 from a clean tree.
