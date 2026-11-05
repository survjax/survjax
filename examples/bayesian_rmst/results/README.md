# Recorded synthetic full run

`posterior.npz` contains the synthetic trial, four-chain parameter draws,
divergence flags and grids. `metadata.json` records the full audit and original
measured timings. Five PNG previews retain the reviewed plots. Derived curves,
prior draws and predictive trials are regenerated, not committed archives.
These outputs are not independent reference-fixture expectations.

`delivery.json` records code commits and SHA-256 hashes. To materialize the
saved draws without rerunning NUTS, run this from the repository root using
`JAX_ENABLE_X64=1 uv run --group examples python`:

```python
import hashlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from examples.bayesian_rmst.model import posterior_draws

if not jax.config.x64_enabled:
    raise ValueError("Run in a float64 process")
source = Path("examples/bayesian_rmst/results")
manifest = json.loads((source / "delivery.json").read_text())
assert (
    hashlib.sha256((source / "posterior.npz").read_bytes()).hexdigest()
    == manifest["sha256"]["posterior.npz"]
)
with np.load(source / "posterior.npz", allow_pickle=False) as saved:
    arrays = {name: saved[name] for name in saved.files}
samples = {
    name: jnp.asarray(arrays[f"sample_{name}"]).reshape(-1)
    for name in ("log_k", "log_b0", "gamma")
}
derived = posterior_draws(samples, arrays["t_grid"], arrays["tau_grid"])
output = Path("/tmp/bayes-recorded")
output.mkdir(exist_ok=False)
np.savez_compressed(output / "arrays.npz", **arrays, **derived._asdict())
shutil.copyfile(source / "metadata.json", output / "metadata.json")
```

The new directory is accepted by the existing figure renderer. It regenerates
all five PNG/SVG pairs and prior/predictive arrays from the saved parameter
draws and recorded keys. Use the parent README's renderer command with
`/tmp/bayes-recorded` as its source and a new figure destination.
