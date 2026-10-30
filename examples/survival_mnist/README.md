# Survival MNIST (implementation in progress)

MNIST by Y. LeCun, C. Cortes and C. J. C. Burges, under CC BY-SA 3.0.
Dataset: http://yann.lecun.com/exdb/mnist/.
Mirror and first-fetch MD5s: [torchvision source](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
SHA-256 values in `data.py` were measured from all four MD5-verified compressed
IDX downloads on 30 October 2026. Every cached file is checked against SHA-256;
a mismatch raises instead of silently downloading a replacement.

From the repository root:

```python
from examples.survival_mnist.data import load_mnist, split_indices

images, labels = load_mnist()
indices = split_indices("smoke")
train_images = images[indices["train"]]
```

Downloads need network on first use. Set `SURVJAX_EXAMPLES_DATA` to the cache
directory, or use the default `~/.cache/survjax-examples/mnist`. Data stays
outside Git; do not point the cache into the repository. Images are float32
NCHW, scaled by 255 with no fitted preprocessing. Labels are reserved for
synthetic outcomes and oracle evaluation; the network will see pixels alone.
The fixed validation permutation and smoke prefixes follow use-case 03.

This step provides acquisition and splits only. Shared-generator outcomes,
float64 censoring calibration, CNN training, references, metrics, full results
and the executable documentation page remain to be implemented.
