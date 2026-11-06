Independent review: round 1/3

# SIM.11 — Navigation integration, N-3 and rendered inspection

Read the newest PLAN section, docs/index.md toctree diff, then the
weibull-simulation.md frontmatter and committed-results introduction diff.
Start / previous final e7660a728a9fd00c2612fb28c6055d7a8d21e4ac (SIM.10
round 2 merge). Original base 8757c0fda78e34e6e818249adedd2a144a33c33d;
integration base d2469e0a7e719dbcb276277e6fd80a447626ad4e. Branch
release-simulation. All unfinished PLAN history is preserved.

## Authorization and scope

The maintainer explicitly requested: "finish navigation integration and
inspection, and do nit N-3; i'll be going through prose and formatting
separately, and dependencies will change also because of other commits".
This authorizes the previously shared docs/index.md integration. One added
toctree entry and removal of orphan:true make the existing notebook part of
the site. N-3 is resolved by accurately limiting the hash-check statement to
data files. No new SVG-hash scheme or numerical code, assets, dependencies,
lockfile, API, shared config or script changes. N-1/N-2/N-4 are intentionally
left to the maintainer's separate prose/formatting/dependency work.

## Invariants and requirements

Spec 10.5 requires executable docs and evaluation of rendered pages;
use-cases/00-common and both 01 designs require the live float64 smoke and
clearly labelled committed full evidence. Navigation must resolve from the
home page and sidebar in both generated site modes. The manifest continues
to check its data payloads, not the SVGs, exactly as the revised text says.
Live smoke still uses temporary output; committed full evidence is not rerun
or modified. The page continues to select float64 in its own kernel.

## Checks and inspection

scripts/build_docs.sh ran directly with approved escalation, no wrapper,
redirect or environment assignment. Exit 0; both modes ended in build
succeeded without Sphinx warnings or notebook errors. All existing executable
pages ran. Simulation-page execution was 9.29 s / 9.58 s (x32/x64 site),
below the 120 s timeout; live float64 smoke took 6.17 s / 6.27 s. The single
fit was accepted with HR .5794 in both. Stock ipykernel TCP startup banners
remain in build logs; they are not Sphinx warnings or page stderr.

One-off BeautifulSoup checks passed in both modes: the home toctree entry
has the correct title/target; every top-level site's sidebar and the study's
sidebar resolve to the study; its active-page marker is present. All local
internal page links and fragment anchors resolve. Five real tables have
4/6/12/12/6 data rows; math markup is present. The revised N-3 statement is
present, and three SVGs plus four downloads resolve and match committed
source bytes. Zero stderr/system-message/problematic elements. These are
structural checks, not browser layout or MathJax-rendering claims.

No implementation-mirroring test is added for these small source-doc changes.
uv run --locked --no-sync ruff format --check . and ruff check . passed
(151 files already formatted / All checks passed!); git diff --check passed.
No numerical suites or mypy rerun because no Python/library code changed.
No generated HTML, notebook output or caches are committed.

Browser access probe: cua.getState returned apps=[], browsers=[], with
"Sky Computer Use native pipe startup failed" and "The browser access host
is unavailable". The CUA part was closed with its required completion
summary. Asked the maintainer to restore browser/computer access while the
build and structural inspection proceed. Pixel layout/math/table overflow
must not be reported as visually checked until that access is restored.

No merge, push or publication. Shared generator import contract unchanged.
