# NP.ADAPTER.1 handoff

Independent review: round 2/3.
The maintainer approved the explicitly requested prototype-only ndim shim.
The adapter remains experimental example code, not a package API addition.

## Round 1 follow-up

Previous reviewed commit: dca1bdb318b122d5bf340329d7944d380e0df1e1.
The maintainer approved this minimal follow-up after discussing the findings
as a proposal for a co-contributor, not a production integration claim.

- F-1: added test_excluded_invalid_records_do_not_warn with eager validation
  explicitly enabled and the support warning treated as an error. This tests
  the user-visible reason for replacement: excluded invalid observations do
  not produce support-validation warnings. The README now explicitly says
  that core numerical safeguards already protect the tested likelihoods and
  that the tests do not establish replacement's necessity for gradient safety.
  This takes the finding's alternative of explaining the need and narrowing
  the claim, rather than constructing an unsafe family to force a NaN gradient.
- N-2: documented stop_gradient as the existing spec 7.2 substitution convention;
  the outer zero mask can already cancel dummy-time derivatives in these cases.
  No artificial mutation-sensitive derivative test was added.
- N-1: retained the explicitly documented incremental timing-list provenance
  instead of introducing unrelated timing churn. The maintainer accepted the
  minimal scope after this bookkeeping tradeoff was discussed.
- S-1: retained the already authorized, scoped prototype shim. No core/spec
  contract changes are made or claimed. Deviation status remains maintainer-owned.

Follow-up validation:
- Focused file, both dtypes, no marker filter: 8 passed in each (8.94s/9.32s).
- In a fresh x64 Python process, temporarily replaced _select_records with
  a function returning the original value, then ran only the new regression:
  1 failed/7 deselected with the expected out-of-support UserWarning. The
  harness asserted TESTS_FAILED and exited 0. No source file was mutated.
- Isolated unmodified regression: 1 passed/7 deselected per dtype. Initial
  concurrent-run call timings were 1.58s/1.31s, triggering the scoped timing
  refresh required for a new slow test.
- Fast A12 gate passed: Ruff 236 files formatted/lint clean, mypy 24 source
  files clean; implementation x64 1091 passed/8 skipped, x32 1096 passed/3
  skipped; doctests 4 passed each; spec x64 3271 passed/32 skipped, x32
  3273 passed/30 skipped (two existing NumPy warnings per spec run).
- Ran scripts/update_slow_tests.sh --through A12 with PYTEST_ADDOPTS scoped
  to the new regression. It classified that one case as slow; preserved all
  previous classifications and appended its generated scoped timing summary
  verbatim. No unrelated timing changes. The gate above included the new test
  before this classification; focused runs always included every adapter case.
- No numerical implementation changes; NUTS demo, full non-fast gate and docs
  build were not repeated for this follow-up.

Read PLAN.md, examples/numpyro_adapter/README.md, examples/numpyro_adapter/adapter.py,
tests/unit/test_numpyro_adapter.py, then examples/numpyro_adapter/run.py.

Invariants: the adapter delegates all scoring to survjax; array values denote
event times, structured values denote observation evidence. Masking replaces
all six observation leaves before scoring. Dummy records use an already
allowed exact/right kind, so selection preserves original may-contain flags.
Both private NumPyro wrappers preserve custom mask/expand dispatch. Sampling
raises explicitly. The example is not a package export or release commitment.

Validation (Python 3.12.9, NumPyro 0.22.0, JAX 0.11.2):
- Ruff format/check passed on the new files; repository Ruff and mypy passed.
- Direct float32 probe: structured contributions [-1.1931472, -1, -2.1931472];
  raw-time density contributions [-1.1931472, -1.6931472, -2.1931472].
  Metadata preserved for exact-only, right-only and mixed invalid-row
  replacements; replacement issues cleared. Invalid rate raises ValueError.
- Initial direct probe ended with TypeError from a positional Exponential
  constructor; corrected the same mistake in run.py and reran the JIT probe
  with rate= explicitly. JIT structured values agree; summed rate derivative
  -2 agrees with 2/rate - 6 at rate=0.5.
- scripts/check.sh --fast --through A12 passed. Implementation x64:
  1085 passed/9 skipped; x32: 1090 passed/4 skipped. Source doctests: 4 passed
  each. Spec x64: 3270 passed/33 skipped; x32: 3271 passed/32 skipped.
  Two existing NumPy warnings per spec run at test_a11_ph_direct_methods.py:78.
  This initial gate used the ordinary environment without NumPyro; the new optional
  adapter test module was skipped, so the gate is baseline evidence only.

- Initial focused pytest runs failed collection because the tolerance import
  used tests.impl_tolerances. Corrected to the repository's impl_tolerances
  convention and added the repository-root path for the optional example.
- Final focused tests, no marker filter:
  JAX_ENABLE_X64=1 (then 0) uv run --with numpyro==0.22.0 --group test pytest
  -n 6 tests/unit/test_numpyro_adapter.py: 7 passed in each dtype (8.19s/6.91s).
  The context fixture also verifies that the ndim property is restored.
- Seeded NUTS demo in both dtypes, 300 warmup/1000 retained samples:
  analytic Gamma(4, rate=7) mean 0.5714285714; x64 sample mean 0.5501001859,
  ESS 304.15, reported R-hat 1.00, zero divergences; x32 mean 0.5477800965,
  ESS 376.54, reported R-hat 1.00, zero divergences. These are illustrative
  single-chain diagnostics, not a statistical convergence acceptance gate.
- scripts/update_slow_tests.sh --through A12 ran the full suites after
  uv sync --group examples (existing dependency group). Implementation x64:
  1421 passed/8 skipped; x32: 1426 passed/3 skipped. Spec x64:
  3581 passed/32 skipped; x32: 3559 passed/54 skipped, two existing warnings
  per mode. Retained all existing slow classifications and mechanically added
  only the two newly measured adapter entries; omitted unrelated timing churn.
  The script's complete logs were retained temporarily to verify exit summaries
  because the timing script itself tolerates pytest failures.
- Final scripts/check.sh --fast --through A12 with NumPyro installed: exit 0,
  "check: all gates passed through A12 (fast: slow tests skipped)". Ruff:
  236 formatted files, lint clean; mypy: 24 source files clean. Implementation
  x64: 1090 passed/8 skipped; x32: 1095 passed/3 skipped; source doctests:
  4 passed each. Both spec dtype processes passed, with the same two existing
  NumPy warnings each. Five new adapter cases ran in each fast implementation
  process; the two slow cases were already covered by the focused/full runs.

Test map: analytical derivatives and metadata test covers 3.2/7.2–7.3;
handler-order test covers nested masks, broadcast expansion and invalid raw
indicator replacement; factor/pointwise test covers unsummed 6.3 contributions
and explicit indexing; arrays/validation/sampling test covers direct scoring,
eager parameter validation, issue preservation, false masks and missing sampling.

Uncertainties/boundaries: no SVI, ArviZ conversion, Predictive, generic wrapper
composition or structured obs_mask demonstration. Unsupported likelihood modes
retain core behavior; the mask replacement covers only exact/right observations.
No production spec change is claimed. The temporary class property is confined
to a context manager and is unsuitable for concurrent library use. No docs build
was run (no docs/ changes). A production integration needs coordinated ndim
spec/acceptance-test changes and core sampling work.

Worktree starting base and preceding commit: 6e779b6a6c05a53f46fec749b36d9b07d9d4562c.
Remaining later work: co-developer feedback, issue reply, and potentially a
draft PR; the maintainer has not authorized posting or publishing.
