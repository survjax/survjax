# Building docs for a worktree or a future portal

`scripts/build_docs.sh` is the stable entry point. Sources, `docs/conf.py`, the
Furo theme and dependency lockfile belong to this repository. A workspace portal
can call this script; it does not need to own or rewrite the documentation.

## Interface

```sh
scripts/build_docs.sh
scripts/build_docs.sh --output-dir /tmp/survjax-docs-review
scripts/build_docs.sh --help
```

- No arguments: use `docs/_build` in the worktree containing the script.
- `--output-dir DIR`: use a dedicated, disposable build root. Relative paths
  resolve from the caller's current directory; paths containing spaces work.
  Within `docs/`, only `docs/_build` and its descendants are allowed, to avoid
  treating generated files as documentation sources. The repo/docs roots are
  rejected. Do not point this option at an existing source or data directory:
  the build owns its contents, including a generated `.gitignore`.
- Both float32 and float64 always build. Successful output is
  `DIR/html-x32/index.html` and `DIR/html-x64/index.html`.
- Exit 0 means both builds succeeded; help also exits 0. Invalid arguments or
  rejected output paths exit 2. Dependency, notebook and Sphinx failures return
  nonzero; the script stops at the first failed command. A failed build may
  leave partial output, so a portal should keep serving its last successful
  build until a replacement succeeds.

Run the script from any directory using its path in the selected worktree.
It derives the repository location from that script, not from the shell's
current Git checkout. It uses that worktree's `.venv`, locked docs/examples
optional groups and source tree for Sphinx, autodoc and notebook imports.
Inherited `UV_PROJECT_ENVIRONMENT` and `PYTHONPATH` are replaced deliberately.
The worktree's environment-provided Python kernel takes precedence over user
kernels. The build does not copy or relocate the shared `spec/` symlink.

Dependencies: Bash, `uv`, and the Python version declared in `.python-version`.
`uv run --locked` synchronizes the environment using `uv.lock`; first setup can
need network access. Sphinx, MyST-NB, Furo, Jupyter and example dependencies
come from the locked optional groups. No portal dependency is added.
The MNIST example uses a hash-checked dataset cache, described below.

## Output and cache isolation

Each worktree must have its own output root. The default satisfies this because
`docs/_build` is worktree-local. A portal supplying a custom root must allocate
one unique directory per worktree; do not build multiple worktrees into the
same root, or run concurrent builds into one root. Branch switches in one
worktree can reuse its root, but publication should wait for a successful build.

Sphinx doctrees live inside each HTML output. MyST-NB writes `jupyter_execute`
and `.jupyter_cache` beside the HTML directories, under the chosen root. JAX,
Matplotlib and IPython caches are also assigned paths under that root. The root
contains a generated `.gitignore` ignoring all its contents. Default build
outputs and the worktree's `.venv` are also ignored by the repository.

The uv download cache and immutable, hash-checked MNIST input cache may be
shared; neither contains a rendered worktree build. `SURVJAX_EXAMPLES_DATA`
selects the MNIST cache when needed. `JUPYTER_RUNTIME_DIR` is set to
`DIR/.cache/jupyter-runtime`, with per-kernel identifiers. Outputs and execution caches should not
be committed or used as shared mutable state between worktrees.

## Serving under a prefix

Serve the *contents of* `html-x64` (or `html-x32`) at a route such as
`/worktrees/circle/`. Include the trailing slash or redirect to it so relative
links resolve correctly. HTML pages, navigation, search index, styles, images
and downloads use relative URLs. No build-time hostname or port is required.
Sphinx's MathJax renderer uses its normal external CDN URL; mathematical
rendering therefore needs that CDN available, or a future locally supplied
MathJax asset configured in this repository.

Keep page paths after the worktree prefix when switching worktrees. Consult
plans/docs-migration.md for the historical old-to-new page map. If the target branch lacks a
page, the portal should fall back to its index rather than assume every branch
has the same documentation. Fragment IDs may differ after editorial changes.

Furo remains configured in `docs/conf.py`. A later selector/status display can
use Sphinx's `templates_path`, an inherited Furo template and `html_static_path`
for local CSS/JavaScript; no theme fork or navigation rewrite is required.
The portal owns worktree discovery, route mapping and build-status collection.
It must also handle spaces in filesystem paths, quote subprocess arguments,
choose one dtype output, and avoid exposing partial builds or build caches.
Do not infer successful status from a directory's existence.

## Successful-build identity and inspection

Read `DIR/build-identity.txt` (default `docs/_build/build-identity.txt`). The
last `state:` must be `success`, with the intended absolute checkout, commit,
dirty status, dtype pair and UTC completion time. A new attempt invalidates
old success before dependency setup when the record exists; failures leave
`building` or `failed`. Old HTML may remain but is uncertified. Do not edit
sources or run concurrent builds during verification. Dirty status is not a
snapshot; untracked content edits and edits after completion require a rebuild.
The record covers both dtype outputs, regardless of which one a portal serves.

For agent browser/screenshots and sandbox troubleshooting, see
[contributor inspection guidance](CONTRIBUTING_DOCS.md). Structural HTML checks
and actual screenshot/layout inspection supply different evidence.
