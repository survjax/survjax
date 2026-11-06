Independent review: round 1/3

# SIM.9b — CLI figure delivery and committed SVGs

Read PLAN.md SIM.9b, run.py's post-evidence call, the extended real smoke
assertions, full-panel check_performance.py check, then the three SVGs and
results/README.md. Previous final
aec234d255c65c3ad3e18b442a02b4c48bcb55de (SIM.9a round 2 merge, F-1 fixed).
Original base 8757c0fda78e34e6e818249adedd2a144a33c33d; integration base
d2469e0a7e719dbcb276277e6fd80a447626ad4e. User explicitly approved compact
evidence then figure implementation/delivery; this completes that scope.

## Invariants and scope

run.py --config {smoke,full} --out DIR now writes raw local diagnostics,
compact evidence and all three SVGs, in that order. No refit is needed to
regenerate committed figures: README gives the direct write_figures call.
The input bundle is the reviewed clean-source a3792a7 run, not a new full run.
Data payloads/manifest, all numerical fitting/keys/calibration/acceptance and
the shared generator import contract are unchanged. No core/public API,
dependency, protected path, spec or docs/ edits. Full raw CSV remains ignored
and is not staged/tracked. Temporary PNG previews stay outside Git.

SIM.9a N-1 fixed: Figure 1 arm colours are gray/green, distinct from the
blue/orange model colours in Figures 2/3. Legends disambiguate all line styles.
N-2 (optional helper extraction) is left as explicit sequential figure code.
All Python loops added here only inspect host artists/files, never fit.

SVGs retain vector geometry/text; no raster or per-fit diagnostic payload.
Assets total 127,300 bytes: survival 22,250, sampling 85,833, coverage 19,217.
Committed numeric evidence is 23,633 bytes plus its explanatory README.
There are 64 stored histogram edges and 63 bins; every accepted fit contributes.

## Validation

```bash
JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q examples/weibull_simulation/check_*.py
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

72 passed in 48.40 s; Ruff/mypy pass (150 files / 22 source files).
The real direct-script smoke now requires all three SVG paths; no extra
smoke subprocess. Its existing metadata/precision/count checks still pass.
The new full-bundle display check verifies all six histogram titles and each
model's density against the recorded bin counts/accepted denominator, then
all six coverage points by scenario/model. This addresses the earlier
single-scenario fixture limitation. Analytical curve/marker, MCSE, empty-set,
SVG XML and tamper-refusal checks from SIM.9a remain in the complete suite.

Generated figures from committed inputs via write_figures; parsed all three
SVG XML roots and confirmed no <image raster payload. A second export was
byte-identical to every asset. Sampling and coverage also match the earlier
reviewed bytes exactly; only survival arm colours differ. Current hashes:
- fig1: 0e5c4ae9ffb0ed52e81e98a0f7ba166d701016ee8e8b6bd16c7d80ac2851b82c
- fig2: 1cd91aae5704bc70ea59c038daf97abb1e5aa8b94d94a245e04db576d8066bca
- fig3: 83a94a85e6bf4847a18ea6c1e184d319f7d6851abd7c98279fa225e474cd2a72

Inspected all full-study Figure-object PNG previews; rechecked the final
survival preview after the colour change. Legends, all six panels and error
bars are legible. Separate manual XML audit compared the serializer input
SVG with output: every geometry attribute/path token and non-whitespace text
was preserved (169/769/217 nodes). This is not a browser SVG inspection.

Fresh scripts/check_wheel.sh --through A10 is running for final regression
and wheel-only core-import evidence; its final result will be recorded in a
verification-only follow-up. The earlier wheel-supplying gate passed both
modes. Reviewer raw gates lack SURVJAX_WHEEL and get five wheel-setup errors;
distinguish that environment limitation from the wheel-supplying command.
No full numerical simulation rerun: numerical implementation is unchanged.

Remaining after this reviewed delivery: executable MyST docs page with live
smoke and labelled committed full results/figures. No docs/nav/config or
protected build script changes in this step. No merge/push/publication.
