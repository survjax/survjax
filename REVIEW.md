# A6.4b2 review follow-up — Make the evaluator readable

Status: R1–R6 implemented on `A6`, based on `401ce87`, in response to
`../review/slices/A6.4b2.md`. Spec revision 32, marker `cf4fef9`; independent
snapshot `08d97a6`. Stop after this follow-up commit; no A6.4c work.
SQ1 is recorded as D-026 and awaits a separate maintainer decision.

## Read first

1. `src/survjax/likelihood.py::_log_likelihood`: orchestration of shape
   checking, row masks, dtype promotion, parameter preparation and evaluation.
2. `_unusable_data_rows`, `_supported_rows` and `_evaluate` in the same file:
   named stages separate the rules and make the numerical path easy to inspect.
3. `src/survjax/_data_validation.py::stored_field_issues`: why reconstructed
   fields need rechecking, with rule citations and normalized/raw distinctions.
4. `tests/unit/test_likelihood_core.py::NarrowExponential` and its precision
   test: the reason for retaining the final output conversion.
5. DEVIATIONS.md D-026: SQ1's wording gap and proposed clarification.

## Review findings and invariants

- R1: the entry point reads as orchestration; named helpers own data-row
  rejection, support and safe evaluation. No new public API or helper module.
- R2: move the existing expressions unchanged to stored_field_issues. Tree
  reconstruction bypasses constructors. Raw indicators/flags are gone after
  normalization, so those fault inputs are false and their earlier evidence
  comes from stored diagnostics. Exact normalized upper must equal lower;
  unlike raw bounds input, it is not optional. Window/interval checks still
  reuse row_issues. Constructors were not rewritten in this follow-up.
- R3: build bad once and use its complement for both substitutions and the
  same mask for final NaN selection. R4: exact_rows names the broadcast mask;
  the original data.exact field is never rebound.
- R5: cite spec 3.3 and explain each conversion boundary: stored data times,
  family references with fixed precision, and custom outputs that stay narrow
  despite promoted inputs. The output cast is observable in the new test;
  individual redundant guards/casts are not claimed to be distinguishable.
  Name no_fault in the field helper and explain the unavailable raw evidence.
- R6: read data.issues.invalid_rows instead of data._issue_bits. This is the
  same stored union; uninformative is separate and remains non-invalidating.
- Spec 7.1–7.3: parameter masking still precedes all numerical family methods,
  including reference_time. Preserve safe likelihood-level substitution even
  though the base's direct methods also guard inputs. Keep true zero mass as
  -inf and invalid/unsupported contributions as NaN. No new parameter domains,
  floors, clipping, metadata assumptions or stopped valid parameter entries.
- The precision probe is a unit-rate exponential with fixed float32 survival,
  density, reference and infinity limit. On integer times its hand-derived
  exact/right contributions are -t, exactly representable. Common output dtype
  must follow the data even when every family output remains float32.
- SQ1 does not block this behavior-preserving refactor. data.issues.invalid_rows still
  reports stored bits; the core still checks current fields. The existing
  rebuilt-field tests deliberately expose that distinction. No spec change
  has been inferred from the review or made without explicit approval.

## Checks actually run

All uv/gate commands use UV_CACHE_DIR=/private/tmp/survjax-uv-cache.

- Focused core and core-matrix branches, separate JAX_ENABLE_X64=1/0:
  `uv run --locked --group test pytest tests/unit/test_likelihood_core.py
  tests/unit/test_likelihood_parameters.py -k core -q --tb=short`.
  Both exit 0: 62 passed, 38 deselected (56.01s/55.94s).
- After pinning the precision probe's infinity limit to float32 too, reran
  `uv run --locked --group test pytest tests/unit/test_likelihood_core.py
  -k fixed_precision -q --tb=short` in both modes: exit 0, 1 passed,
  26 deselected (0.99s/0.96s). Full gate covers the final version.
- In-memory output-cast mutant, x64 only:
  `uv run --locked --group test python
  /private/tmp/survjax-a64b2-output-cast-mutation.py`.
  Removing only the final output conversion produces float32 rather than
  float64: pytest exit 1, 1 failed, 26 deselected (0.73s). The harness exits
  0 for this expected detection; repository source is unchanged by the mutant.
- Focused Ruff format/lint and mypy src: exit 0; files formatted, lint clean,
  no issues found in 17 source files.
- `scripts/check.sh --through A5`: exit 0, final output
  `check: all gates passed through A5`. Guard: `ok (A6)`; Ruff:
  `64 files already formatted`, `All checks passed!`; mypy:
  `Success: no issues found in 17 source files`.
  Own x64: 648 passed, 7 skipped (112.22s); x32: 654 passed, 1 skipped
  (114.64s). Source doctests: 4 passed per mode (0.93s/0.82s).
  Independent through-A5 x64: 1256 passed, 26 skipped, 769 deselected
  (23.48s); x32: 1261 passed, 21 skipped, 769 deselected (19.54s).
  Full log: `/private/tmp/survjax-a64b2-review-check.log`.
- `../scripts/spec-since.sh dev`: exit 0, spec unchanged since cf4fef9.
- `git diff --check`: exit 0, no output.

## Limits and uncertainties

D-026/SQ1 awaits maintainer policy on recipe wording; no behavior change is
proposed. Public API integration, the complete A6 gate and documentation stay
with A6.4c/later docs work. The reviewer's throwaway public-wrapper probes
are external review evidence, not a gate run in this follow-up. No protected
file, docs, spec-test, dependency or spec change; no docs build, R fixtures,
install test or performance measurements.
Source/test diff: 172 changed lines; likelihood.py 86 lines, validation module
271 lines and core tests 246 lines. Existing tolerances are unchanged.
