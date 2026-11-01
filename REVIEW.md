Independent review: round 1/3

# MNIST.9 — Three SVG figures

Read PLAN.md MNIST.9; use-case03 details Figures; run.py _figures and its
call with selected checkpoint predictions; check_mnist_figures.py and the
runner's figure-input checks. No core, dependency or protected-path changes.

Invariants:
- Figure inputs come from the selected validation checkpoint and full test
  predictions in official index order. Labels group/illustrate predictions
  after training/selection only. No seed/checkpoint/tuning change.
- First test image per digit is the first matching row in that official order.
  Missing digits fail explicitly. Headline layout is2x5 groups, image and
  curves side by side. Public survival_grid uses float32 t=linspace(0,20,201)
  and float32 selected/true parameters; truth is dashed, survival axes0..1.
- Boxplots include all test predictions grouped by true digit; reference
  diamonds use the host float64 generating log scales. No fitted digit means.
- History uses recorded epoch minibatch means/full-validation NLLs and the
  one-based selected epoch. One-epoch smoke intentionally has one point per
  history, not an invented initial value.
- Save standalone fig1/2/3.svg and close all three figures. MPLBACKEND=Agg in
  child environments keeps plotting headless. Matplotlib version joins the
  runtime metadata; the stdlib-only parent still never imports JAX/Matplotlib.
- No training, metrics, timing-policy or metadata-selection change. D-031 stays
  open for later full-run timing; no warm pass added and no spec interpretation.
- run.py is301 lines, about the300-line guideline; new source/test changes
  fit within the step's about300-line implementation budget.

Tests:
- Analytic exponential k=1,b=11-digit pins predicted curves over201 time points;
  independent exp(-(t/b_d)^1.38), b_d=10*exp(-.2d), pins true curves.
  Two rows per digit with different images/predictions catch first-row errors.
- Inspect actual Matplotlib image/line artists for first image pixels, time
  grid, curves, true dashed style, axis ranges and digit titles.
- Capture real boxplot groups with a forwarding wrapper and check every row;
  inspect true-marker positions and digit tick labels.
- Actual history artists pin both recorded series and selected epoch2 (best
  validation), distinct from last epoch3. Files parse as SVG; no figures remain
  open. Missing digit gets explicit ValueError.
- Runner collaborator checks selected shape/scales, official-order images and
  labels, histories/selected epoch and output directory at the figure boundary.
  Existing tests continue pinning all metrics/metadata and child precision.

Validation: actual unchanged fixed smoke CLI using verified warm cache, into
/private/tmp/survjax-mnist-figures9. All three SVGs emitted, then rendered with
installed rsvg-convert into temporary PNGs and visually inspected with view_image.
QA: all ten digits and paired curves visible without clipping/overlap; true
curves distinct and dashed; all box labels/reference diamonds legible; smoke
history has its two epoch1 points and selected vertical line. No layout fix
needed after inspection. The sandbox denied Matplotlib's default home cache;
it automatically used a temporary cache, built fonts and completed with exit0.
No dependency installation or cache permission change made.

Smoke metrics match previous step: CNN meanLL-1.1935251951, C.6301579475,
k1.1871362925; intercept meanLL-1.2371411324,C.5; oracle meanLL-1.1055958271,
C.6948196888. One-epoch k remains~14% below1.38; no tuning/full-run acceptance
claim. No full run, committed MNIST data/results, docs build or R generation.

Checks: eight-file optional suites x64 82 passed/1 skipped (38.57s), float32
60 passed/23 skipped (31.19s). Initial focused figures/runner x64 9 passed
(12.95s). Ruff lint clean; cumulative fresh-wheel gate results appended below.
Remaining: D-031 timing decision before fixed full run/committed artifacts,
then executable docs. No new deviation. Stop after independent review MNIST.9.
Previous step final: 97203e74143aa1a0cf81a6dbcdf11ba8c5ca7843.
Original worktree base: ee72c837549d14cb26e9983bf6d63ac54563f189.

Cumulative check: scripts/check_wheel.sh --through A10 exit0, all gates passed.
Ruff format137 files already formatted; Ruff all checks passed; mypy no issues
in21 source files. Own suites x64 1119 passed/7 skipped (76.50s), float32
1125 passed/1 skipped (75.44s). Doctests4 passed each mode. Acceptance x64
2333 passed/27 skipped (93.92s), float32 2324 passed/36 skipped (92.71s).
Final format and git diff --check clean.
