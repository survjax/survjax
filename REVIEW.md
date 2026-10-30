Independent review: round 1/3

# RMST.2-report — Mathematical approval report

Start: b3340749469e1513938a0cd5140f45305ab0d4bb, clean release-rmst.
Read PLAN.md RMST.2-report, reports/weibull-rmst-math.md and the compiled
output/pdf/weibull-rmst-math.pdf, then _numerics.py at that starting commit,
spec 8.7 and D-028. This is a maintainer explanation, not the public RMST
guide, RMST.3, or an implementation change. No production/spec/test changes.

## Mathematical assumptions and approval points

- k,b,tau positive in the integral derivations; zero and invalid values are
  handled separately. a=1/k and x=(tau/b)^k. The gamma identity follows by
  substitution, with P/Q defined as in NIST DLMF 8.2 (opened/verified).
  DLMF 8.7 corroborates the series; the report derives its own expansion.
- Degree 12 means 13 terms including n=0, not 12 total terms. Uniform
  absolute convergence justifies integration; decreasing alternating terms
  bound the first omitted term. Dividing by M>=tau*exp(-x) yields the stated
  relative bound. The cutoff derivative is the truncated exponential.
- The upper-tail proof uses Gamma(a,1)'s mgf (1-s)^(-a), Markov with s=3/4,
  and x>64+4a, giving Q<exp(-48+a*(log(4)-3))<exp(-48). Relative value
  error is Q/(1-Q). This bound is for exact arithmetic, not all FP errors
  or all parameter derivatives. Thresholds are conservative, not optimized.
- Distinguish domain guarantees, sampled accuracy, truncation bounds and
  special-function precision loss. Full derivative-range evidence remains
  RMST.3. The staged shape-Hessian limitation and older S-1 are explicit.
- D-028: the zero branch's first derivatives are correct, but at k=1,b=2,
  tau=0 autodiff gives second cutoff derivative 0 while the analytic right
  derivative is -0.5. No interpretation of the higher-boundary contract,
  code change or spec change was adopted. The report prominently routes
  this approval question to the maintainer.
- rmst_grid: the report recommends retaining spec 8.7's staged exclusion,
  explains the explicit-axis equivalent and the benefit of a future thin
  adapter, and does not add API or claim the future decision is settled.

## Checks run and outputs

- Existing tools: Pandoc 3.9.0.2; XeTeX/TeX Live 2025; system Poppler.
  No dependency or lockfile edits. The PDF skill's artifact-start marker
  ran once successfully before first authoring, with expected PDF count 1.
- `JAX_ENABLE_X64=1 uv run --group test python -` using mpmath at 60 digits:
  0.1**13/13! = 1.60590438368216146e-23;
  exp(0.1)*0.1**13/13! = 1.77479882205572145e-23;
  exp(-48) = 1.42516408274093511e-21;
  exp(-64) = 1.60381089054863785e-28.
  Threshold-ratio pairs at k=.2,1,50 match the report table. For
  f(t)=Weibull(shape=1.,scale=2.).rmst(t), f(0)=0, grad f(0)=1,
  grad(grad f)(0)=0 versus analytic right derivative -0.5 (D-028).
- `pandoc reports/weibull-rmst-math.md --standalone --number-sections
  --pdf-engine=xelatex -o output/pdf/weibull-rmst-math.pdf`: exit 0.
  pdfinfo: 5 letter pages, PDF 1.5, unencrypted, 66004 bytes.
  SHA-256: 8068389abc3b9de6773a8f3d5e5b39cb0c858f38b8057ae2cc44314bb604534f.
- Also generated temporary TeX and ran xelatex with -halt-on-error twice.
  The first pass requested label rerun; the final log has only Output
  written (5 pages), with no Overfull, Underfull, missing-character or
  LaTeX-warning matches. No generated TeX edits.
- `pdftoppm -r 120 -png` and `pdftotext -layout`: exit 0; extracted 5 pages.
  Viewed every final page PNG. An initial long inline SHA overflow was
  corrected in Markdown and rebuilt; final title, equations, tables,
  code, margins and page numbers are legible with no clipping/overlap.
  Render/TeX intermediates are removed before commit; final PDF/source stay.
- Pandoc JSON AST: 83 inline math nodes, 16 display math nodes, zero raw
  LaTeX nodes. Extracted text contains Chernoff, D-028, rmst_grid and both
  stated approximation bounds. The report source is 269 lines.
- `scripts/build_docs.sh`: initial sandbox run exited 0 and built both modes,
  but notebook shutdown logged psutil sysctl PermissionError. Reran the
  unchanged command with approved escalation: exit 0, build succeeded for
  x32 and x64, no ERROR/Traceback/PermissionError. Stock local-kernel TCP
  warnings remain; no Sphinx warning/error. No docs source was edited.
- `uv run ruff format --check .` / `uv run ruff check .`: pass (115 files already formatted; All checks passed); `git diff --check`: clean. src/tests/spec/docs,
  project metadata and lockfile have no changes. No numerical test-suite
  rerun: this step changes explanation/artifacts only; the boundary probe
  is a documented observation, not an acceptance test of disputed behavior.

## Remaining limits

The report's proof bounds do not certify uniform floating-point or derivative
accuracy. Boundary higher derivatives await D-028; invalid-own-tangent S-1
also remains maintainer-owned. This report does not authorize implementation
approval, a grid API, PH handoff, merge, push or publication. Preserve prior
PLAN history and prior implementation evidence by commit. Independent
review is requested only after report/source/PDF and these records commit.
