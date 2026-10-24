# Review: A10.2a — Share observation input normalization

## Reading order

1. `src/survjax/_observation_inputs.py`: the relocated `_resolve_data`.
2. `src/survjax/likelihood.py`: import the shared resolver at the same call site.
3. `tests/unit/test_observation_inputs.py`, then the unchanged public likelihood
   entry-point suite for value, derivative and input-form regression coverage.
4. The A10.2a entry in PLAN.md and the gate log linked below.

Authorized by "ok no issues here on to the next bit, which is approved by me".
The next integration step is split so shared normalization can be reviewed
before public concordance choices. No public API, numerical, dtype, dependency,
docs, spec or independent-test changes. D-027 remains open for discussion.

## Invariants and requirement mapping

Spec 6.1 requires one resolver for prepared data or raw time/event inputs.
The function was moved verbatim; an AST comparison against the original HEAD
function passed before the maintainer completed the merge. No exception type,
message, argument rule or normalization behavior changed. Prepared data are
returned with their existing leaf dtypes, metadata and raw issue evidence.
Raw observations still use the ordinary pure checked SurvivalData constructor.
Static misuse raises TypeError; invalid runtime codes return data with issues.

New own tests map to these requirements:

- `test_shared_raw_form_preserves_invalid_indicator_evidence`: spec 6.1–6.2;
  event code 2 retains its invalid-indicator bit and invalid-row union, even
  though normalized lower times are finite. Eager, jit and cohort-wise vmap
  agree with explicit expected masks.
- `test_prepared_rich_observations_keep_every_leaf`: spec 3.2/6.1; left/interval
  modes, truncation, dtypes and stored invalid-indicator evidence survive the
  prepared form eagerly and under jit. This is an identity assertion, not a
  new numerical reference.
- `test_shared_normalizer_rejects_static_input_misuse`: spec 6.1; six missing,
  mixed and positional-raw combinations raise TypeError.

The unchanged `test_likelihood_entry_points.py` exercises both public function
and distribution method, prepared/raw forms, analytical values, eager/jit/vmap
issue propagation, structural misuse and first/second parameter derivatives.
No tolerance changes or additional gradient guarantee.

## Checks before the regression gate

Commands use `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.
Ruff format changed only the new test's layout; Ruff lint passed. Mypy:
`Success: no issues found in 19 source files`.

`JAX_ENABLE_X64=1 UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run --group test pytest -q tests/unit/test_observation_inputs.py tests/unit/test_likelihood_entry_points.py`:
**68 passed in 4.35s**. Same command with `JAX_ENABLE_X64=0`:
**68 passed in 3.79s**. No failures or skips.

The resolver AST comparison and exact staged-index snapshot comparison passed.
The maintainer then completed the merge as `34a7657`; git status showed no merge,
no staged changes, and only this step's new/modified files. Incoming docs,
protected spec tests and sync-script changes were preserved in that merge.

## Final regression gate and remaining work

Run the unchanged gate against finalized source/tests and this review:
`UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A7`.
Full output is saved in
[the A10.2a regression gate log](/private/tmp/survjax-a102a-final-gate.log).
PLAN.md records its observed exit status, counts and timings after completion;
this review is left unchanged afterward. Final Ruff and diff checks validate
any plan-only result edit before committing.

This is through-A7 regression evidence for a resolver relocation, not a complete
A10 gate. A10 now has independent tests, but its public API remains unimplemented.
No score representation, ConcordanceInfo contract or ratio is introduced here.
D-027's original tuple proposal differs from independent Q-044's current guessed
count-reader representation; report and discuss that before implementing either.
No acceptance test is adapted, skipped or xfailed.

No R fixture generation, docs build, installed-artifact check, benchmark or
accelerator test was run. Public concordance integration is a later step after
the maintainer decides the open contract. Commit A10.2a on A10-cindex, then stop.
