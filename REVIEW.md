Independent review: round 1/3

# MNIST.12 — Executable documentation page

Read first: PLAN.md MNIST.12, spec/use-cases/03-survival-mnist.md and its
-details.md companion; docs/use-cases/survival-mnist.md; scripts/build_docs.sh;
docs/index.md; examples/survival_mnist/README.md. Previous accepted numerical
implementation and committed full results are unchanged.

Invariants: no test-driven tuning, parameter or result-artifact changes. The
page invokes the existing CLI for one live smoke epoch in temporary output;
its float64 preparation and float32 neural subprocesses retain the existing
precision boundary under either docs parent dtype. Source hashes protect the
warm MNIST cache. Committed full results are explicitly labeled separately
from live smoke. Their figures are unchanged; an epoch-summary plot reads the
same committed history. Literalincludes show the actual CNN and likelihood.
C risk direction, constant-score ties, shared compiled-callable caveat and
both first/warm timings match the existing experiment.

The maintainer already chose BFGS initialization for both CNN bias and shared
shape (D-030), both concordance timings (D-031), and accepted the experiment
and dense figures (D-032). Those formal entries remain maintainer-owned.
No additional spec decision is needed for this docs topic. No API or dependency
is added: build_docs activates the existing locked examples group.

Acceptance mapping: executed generating curves use public survival_grid;
literalincluded loss uses public likelihood on minibatches; live CLI enforces
finite initial/post-first-update gradients. Full metrics show CNN beating the
intercept on LL/C, the learned-shape 5% target, and both oracle gaps. Cohort
sizes, validation-only checkpoint selection, precision, raw-data cache/network
requirements, attribution, reproducibility and timing caveats appear on page.
Three committed figures plus epoch-summary plot expose predictions and losses.

Validation:
- scripts/build_docs.sh, run directly: exit 0; both x32/x64 builds report
  "build succeeded." Final MNIST notebook execution: 23.25s / 21.70s, below
  the existing 120s timeout. The prior first build also passed; direct HTML
  inspection then caught raw Markdown tables, fixed with page-local MyST
  output rendering before the final complete rebuild.
- uv run ruff format --check .: 139 files already formatted; exit 0.
- uv run ruff check .: All checks passed!; exit 0.
- uv run mypy src: Success: no issues found in 21 source files; exit 0.
- Inline stdlib HTMLParser checks, both output pages: exactly three real HTML
  tables, five image targets and three downloadable targets. All targets exist;
  full LL/timing values match committed data. All three tables, including live
  smoke metrics, match across both parent dtype builds. Exit 0.
- Inspected the emitted true-curve and epoch-summary PNGs with view_image;
  rasterized and inspected all three copied SVGs with rsvg-convert/view_image.
  Labels, legends, selected epoch and predicted/true curves are legible in
  those assets. No committed artifact changed.
- Full-page browser layout inspection could not run: cua.createBrowserTab
  reports "browser access host is unavailable"; inventory also reports native
  pipe startup failed, with no apps/browsers. This is a remaining visual-QA
  limitation; direct generated-HTML checks and plot inspection are not a
  substitute for a browser screenshot. The temporary preview server was stopped.
- git diff --check: exit 0. No numerical tests or R fixture generation repeated
  for this docs-only step after MNIST.11's passing full wheel-backed gate.
README resolves prior review N-5 (heading) and N-6 (format provenance).

Remaining: browser layout inspection when that surface is available. No code,
experiment or docs implementation substep remains; no publishing or merging.
