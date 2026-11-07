# A12.GATE — PH/RMST acceptance integration

Independent review: round 1/3

Read PLAN.md, .github/workflows/ci.yml, .github/workflows/release.yml,
then tests/spec/TESTMAP.md A11/A12 and the existing independent suites.
Start / previous step / original worktree base: cc0e30f.
Plan archive: e944567. No API, dependency, numerical or protected-path edits.

Both hosted gates previously stopped at A10 despite integrated PH/RMST code
and independent acceptance suites. Advance the cumulative selection to A12.
CI retains its OS/dtype matrix, offline setup and exact built-wheel input;
release retains both dtypes and its validated wheel. Neither workflow uses
--fast: full tests remain required there. The generic runner already supports
A12. No test assertions, markers or tolerance changes are needed.

Evidence: scripts/check.sh --fast --through A12 exited 0:
- guard ok; Ruff 192 formatted, lint clean; mypy 22 source files clean;
- implementation x64: 1072 passed, 8 skipped; x32: 1077 passed, 3 skipped;
- source doctests: 4 passed per dtype;
- independent acceptance x64: 3396 passed, 33 skipped;
  x32: 3389 passed, 40 skipped.
- Each acceptance run emitted two RuntimeWarnings from NumPy expected-value
  arithmetic at t=0 in test_a11_ph_direct_methods.py:78; no failures.
- git diff --check passed.

A11 tests cover PH construction, diagnostics, methods/grids, identities,
likelihood, ordinary supported derivatives and invalid-entry isolation.
A12 tests cover staged RMST policy, values, independent references and
supported derivatives. See their TESTMAP entries for detailed requirements.
The fast gate omits the registered slow tests; the full merge gate and hosted
workflow execution were not run here. Docs/demos and the full first-release
completion audit are not certified by these tests. No docs changed.

Roadmap handoff: ../ROADMAP.md is stale (A10 underway, PH/RMST not started).
The maintainer should record integrated implementations and this local A12
fast-gate evidence, retaining separate docs/demo/full-gate closure requirements.
No ambiguity or new deviation identified. Remaining substeps: independent
review, then maintainer full merge gate/hosted execution. Stop after review.

---

## Preserved incoming demo audit record

# DEMO.E1 — Release-demo evidence audit

Independent review: round 1/3

Read first: PLAN.md; examples/RELEASE-EVIDENCE.md; spec 12.2 item 4,
11.2–11.3 and all three use-case main/details; DEVIATIONS.md D-039/D-041,
D-045–D-047; plans/mnist-showcase-instruction.md sections 3–5; then cited
result metadata/CSV/NPZ and historical git objects. Starting/previous integrated
step/original base cc0e30f; plan archive 86d4a56.

Only a demo-specific evidence index is added. No code, numerical outputs,
protected paths, docs, shared release records or dependencies changed.
Invariants: historical results retain their original versions/settings;
maintainer-approved simplifications are not silently reversed; a smoke run is
not full acceptance; no claim of A11/A12, hosting or independent review of
historical steps. Unamended normative text and formal open deviations remain
maintainer-owned. No new ambiguous implementation choice or deviation made.

Validation performed:
- scripts/check.sh --fast --through A10: exit 0, all gates passed.
  Ruff 193 formatted/lint clean; mypy 22 source files clean.
  Implementation x64 1072 passed/8 skipped; x32 1077 passed/3 skipped.
  Doctests 4 passed per mode. Spec x64 2149 passed/27 skipped;
  x32 2156 passed/20 skipped. Slow tests excluded by this prescribed tier.
- uv run --group examples --group test pytest -q
  tests/unit/check_mnist_run.py::test_full_results_meet_acceptance:
  1 passed in 0.44s (03 cohort sizes, CNN vs intercept, shape tolerance).
- Read-only Python artifact audit via uv run --group examples python:
  five simulation output hashes match; 12 rows/12000 attempted/11863 accepted/
  137 failures, per-group counts match; n1000 coverage distances .9860885 and
  1.7853224 MCSE, both <3 (01 acceptance).
  Bayesian NPZ loads without pickle: 300-row trial, 101 horizons, three
  (4,1000) chains; all seven retained arrays exactly match 55565fd NPZ.
  Saved R-hat/ESS/divergence/route/truth gates pass; tau_max >=5 (02).
  git merge-base --is-ancestor verifies ac0fa01,8349e7f,36c290e,55565fd,
  add507a,7a5670b,b757e88 are all in dispatch main.
- Historical REVIEW.md/metadata inspected via git show, not rerun outcomes.

Not run: full simulation, NUTS, MNIST training/download, docs build/browser,
full slow gate, independent A11/A12, historical R/KM comparison. No need to
repeat those for an index of already committed evidence. No new test code.

Uncertainty/limits: lean metadata does not preserve all original runtime
identity fields; records are not cryptographic attestation of execution.
NumPyro ESS is not renamed bulk ESS; current Bayesian predictive/tail workflows
were intentionally removed. Historical evidence is identified separately.
Simulation regeneration 8349e7f explicitly lacked independent review.

Remaining: independent review of this index; docs owner validates live/full
rendering and hosting; maintainer reconciles formal spec/deviation status;
independent acceptance remains separately owned. Do not start those steps here.
