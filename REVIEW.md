Independent review: round 2/3

# MNIST.9 — Three SVG figures

Read PLAN.md MNIST.9, use-case 03 details "Figures", run.py _figures and its
selected-checkpoint call, then the figure and runner checks.

Invariants:
- Figures use the selected validation checkpoint and all test predictions in
  official index order. Labels illustrate/group predictions only after selection.
- Figure 1 selects the first test image per digit, with image and curves beside
  each other in 2 x 5 groups. Public survival_grid uses float32 parameters and
  linspace(0, 20, 201). Truth is dashed; survival limits are 0 to 1.
- Figure 2 includes every test log scale, grouped by digit, with the generating
  log scales shown as diamonds. It does not estimate digit-specific references.
- Figure 3 uses the recorded training and validation NLL histories and marks
  the one-based selected epoch. Smoke has one point per series.
- Save three standalone SVGs and close all figures. The CLI sets MPLBACKEND=Agg;
  the parent still imports only the standard library. Record Matplotlib version.
- No model, training, selection, metric or timing-policy change. No dependency,
  core API, module, protected file or spec change. D-031 remains a future full-run
  timing decision. This step does not choose a warm timing policy.

Test evidence:
- Two rows per digit have different images and predictions. First occurrence
  checks pin the first image in official order rather than the last.
- With learned shape k=2 and b=11-digit, independent exp(-(t/b)**2) pins every
  predicted curve. Independent exp(-(t/(10*exp(-.2*digit)))**1.38) pins truth.
  Both use the fixed 201-point grid and existing float32 tolerances.
- Actual image and line artists pin pixels, digit titles, grid, curves, dashed
  truth and limits. A forwarding boxplot wrapper pins all grouped predictions;
  actual scatter artists pin truth marker positions and tick labels.
- History artists pin both series and selected epoch 2, distinct from last
  epoch 3. Check SVG output, figure closure and explicit failure on missing digits.
- Runner checks pin selected shape/scales, test pixels/labels, histories, selected
  epoch, output directory, headless backend and Matplotlib version metadata.

Actual smoke and visual QA:
- Run the fixed CLI against the verified warm cache, writing only temporary
  outputs at /private/tmp/survjax-mnist-figures9. Render all three SVGs with the
  installed rsvg-convert and inspect their PNGs with view_image.
- All ten image/curve pairs are visible without clipping or overlap. Truth is
  dashed. Boxplot labels and reference diamonds are legible. The one-epoch
  history shows both points and the selected vertical line. No layout fix needed.
- Matplotlib could not write its default home cache and successfully used a
  temporary cache instead; command exited 0. No installation or permission change.
- Metrics reproduce MNIST.8: CNN mean LL -1.1935251951, C 0.6301579475,
  k 1.1871362925; intercept mean LL -1.2371411324, C 0.5; oracle mean LL
  -1.1055958271, C 0.6948196888. One-epoch k is still about 14% below 1.38.
  No tuning or full-run acceptance claim; no committed MNIST data or results.

Completed checks before review:
- Eight-file optional suites: x64 82 passed, 1 skipped (38.57 s); float32
  60 passed, 23 skipped (31.19 s). Initial focused checks: 9 passed (x64).
- scripts/check_wheel.sh --through A10: all gates passed. Ruff format: 137
  files already formatted; Ruff check passed; mypy: no issues in 21 source files.
- Own suites: x64 1119 passed, 7 skipped (76.50 s); float32 1125 passed,
  1 skipped (75.44 s). Doctests: 4 passed in each mode. Acceptance: x64
  2333 passed, 27 skipped (93.92 s); float32 2324 passed, 36 skipped (92.71 s).
- No docs build, R generation or full run. Final diff check clean.

Round 1 findings:
- F-1: replace the unit-shape fixture with k=2 and its independent analytic
  survival curve. A constant k=1 implementation can no longer satisfy the test.
  Production code and tolerance constants are unchanged.
- N-1: run.py remains 301 lines, within the about-300 guideline. No action now.
- N-2: rewrite this record with normal spacing and readable metric labels.
- Reviewer ran the direct gate; only its five missing-SURVJAX_WHEEL setup errors
  prevented completion. Our passing fresh-wheel gate includes those tests.

Remaining sub-steps: decide D-031 before the fixed full run and committed
artifacts, then executable docs. Stop after this figure step's independent review.
Previous step final: 97203e74143aa1a0cf81a6dbcdf11ba8c5ca7843.
Original worktree base: ee72c837549d14cb26e9983bf6d63ac54563f189.

Fix-up checks: figure/runner checks pass in both modes, 9 passed with x64
(10.40 s) and 9 passed with float32 (10.46 s). Ruff lint and format pass;
git diff --check is clean. No production changes, so no repeated smoke/full
run or cumulative gate after the existing complete check.
