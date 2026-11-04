Independent review: round 1/3

# GPU.2 — Manual Modal GPU check and README badge

Read PLAN.md GPU.2, `.github/workflows/gpu.yml`, `README.md`,
`scripts/gpu_ci.py`, and this review. Base commit: `fe71bd1` (GPU.1 plan
archive). Previous implementation endpoint: `409d773`.

Invariants:
- The workflow has only `workflow_dispatch`; a tag push does not spend GPU
  budget. It checks out `github.sha`, the commit selected when dispatch begins.
- A successful or failed Modal invocation determines the GitHub Actions job
  conclusion. The result ZIP is uploaded when one was produced, even on a
  failing invocation.
- The README badge is explicitly scoped to the latest manual GPU run on
  `main`; it is not a per-version record. The Actions check and run are the
  commit-specific record.
- Modal credentials are read only from repository secrets. The workflow has
  read-only GitHub contents permission and is not exposed to pull requests.

Validation plan: parse the workflow YAML while preserving the `on` key, check
dispatch/SHA/secrets/artifact/summary configuration, and run `git diff --check`.
No numerical or Python code changes; no new tests are applicable.

Uncertainties: GitHub Actions secrets must be configured by the maintainer
before a live run. A real Actions/Modal workflow run is outside local
validation.

Implementation and validation:
- `.github/workflows/gpu.yml` has only the `workflow_dispatch` trigger,
  checks out `${{ github.sha }}`, limits the GitHub job to six hours and the
  Modal run to five hours/$10, uploads any returned result ZIP for 90 days,
  and writes the selected ref, SHA, and job result to the Actions summary.
- `README.md` contains a workflow badge filtered to manual runs on `main` and
  explains that it represents the latest such run.
- YAML parsed with PyYAML `BaseLoader`; assertions for trigger, selected SHA,
  step-scoped Modal secrets, artifact upload, and summary passed.
- `git diff --check`: clean.
- No project tests or live workflow were run. GitHub secrets are not present
  until configured by the maintainer; the workflow must reach the default
  branch before its Run workflow button is available.
