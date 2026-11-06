Independent review: round 1/3

# SIM.10 — Executable Weibull simulation page

Read newest PLAN section, docs/use-cases/weibull-simulation.md, then the
committed results/README and model/runner source referenced by the page.
Previous final / start: 04ec2d8436a75125adc89eaf5361f968c0bf476b (SIM.9b
round 2 merge). Original base 8757c0fda78e34e6e818249adedd2a144a33c33d;
integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e. Branch
release-simulation. Maintainer "ok do it" authorizes this remaining docs step.
All unfinished PLAN history is retained. No core/example runtime, dependency,
lock, spec, protected script, shared navigation/config or other-agent edits.

## Content and invariants

One 280-line MyST notebook page. True/fitted curves are labelled committed
n=300/.55; live smoke is n=100,R=20,two levels, and a separate live Weibull fit
uses scenario 1, replicate 0. Actual model/solver source is literal-included,
and the live helper call executes it. Study kernel explicitly selects float64
in both site build modes, before model construction. This does not modify
library precision policy or other pages' kernels. Smoke output is temporary;
full data and all figures are always loaded from the reviewed committed run.

Full payload hashes, source commit/clean status, failure counts and n=1000
coverage gates are checked in executable cells. Five tables show smoke,
calibrated/realized censoring, bias/coverage/counts, precision/power with
supplied MCSEs, and shared 2x1000-fit timings. RMS model SE is explicit.
All 127 gradient-only failures remain rejected; every other reason is zero.
No events in an arm is unestimable. Counts/reason overlap and accepted-only
performance denominators are explained; full per-fit diagnostics stay local.
First-call/warm timing scope follows approved D-034; compact storage follows
D-035. Times are years. All data synthetic; exact/right without truncation.
Python loops only format/check computed host records; the reviewed runner
continues JAX batching. No new fitter or temporary PH/RMST implementation.

The initial table helper used Markdown, which this site's commonmark output
renderer treated as text. Replaced it with escaped HTML tables and confirmed
real table elements/values. Image/download paths reuse committed small assets.
Page uses orphan:true because shared navigation is maintainer/A9-owned.

## Checks run

```bash
uv sync --locked --group docs --group examples
scripts/build_docs.sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

The direct docs script (no wrapper, env assignment or redirect) exited 0,
with build succeeded in BOTH x32/x64 site modes under -E -n -W --keep-going.
No Sphinx warnings, notebook errors or page stderr outputs. New notebook
execution: 8.55 s (x32 site), 9.12 s (x64 site), well below 120 s timeout.
Live smoke portion: 5.63 / 6.01 s; both explicitly report float64. Live fit
accepted in both, HR .5794, Wald interval [.3206,1.0473]. Full simulation was
not rerun. Existing pages also executed successfully in each mode.

The first sandboxed script exited 0 but printed psutil/sysctl permission
errors during kernel shutdown and live CPU probing. Reran the exact script
with approved escalation; those errors disappeared. Stock ipykernel TCP
startup banners remain in build logs, not Sphinx warnings or page stderr.
The two-group sync uses already approved pins and changed no lock/deps.
The unchanged script's uv run --group docs retains the prepared examples group;
this two-command setup is documented on the page and tested here.

Ruff format/lint and mypy pass (151 files / 22 source files); git diff --check
passes. No repeated core/optional test suite for this docs-only step; the
previous step's 72 optional checks and complete wheel-supplying A10 gate
passed, both core dtypes included.

## Rendered HTML evaluation and limitation

Programmatically inspected BOTH generated HTML pages using BeautifulSoup:
5 real tables with 4/6/12/12/6 data rows; every bias, coverage, count, empirical
SE, model SE and power cell matches the committed CSV formatting. All 3 SVG
image paths and 4 downloadable files resolve and match original bytes. No
problematic/system-message/traceback elements or notebook stderr. Live stdout
and accepted fit are present, literal-included PH/solver code is present.
No generated HTML/notebook outputs or caches are committed.

BROWSER VISUAL QA NOT COMPLETED: attempted cua.createBrowserTab for the local
page; result "The browser access host is unavailable". cua.getState then
returned no apps or browsers, with native pipe startup failure. No alternative
UI automation was used. The temporary localhost server was stopped. Thus
pixel layout, table overflow and browser math rendering were not visually
verified. This is an external inspection limitation, not a claimed pass.
Committed SVG/PNG figure previews were visually checked in SIM.9a/9b; that is
not a substitute claim for this HTML page's layout. Rendered HTML files are
under docs/_build/html-{x32,x64}/use-cases/weibull-simulation.html for follow-up.

## Maintainer integration handoff

Page is intentionally standalone pending serialized navigation integration.
Concrete proposed nav entry: use-cases/weibull-simulation in docs/index.md's
toctree. For one-command fresh docs builds without a preparatory sync, the
maintainer can append --group examples to scripts/build_docs.sh's existing
uv run --locked --group docs invocation. This branch does neither shared/
protected edit. Existing documented preparation works and has been tested.
Browser visual inspection remains outstanding while UI access is unavailable.
No publication, hosting, merge or push. Shared generator contract unchanged.
