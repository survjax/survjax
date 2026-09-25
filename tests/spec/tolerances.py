"""Ordinary finite-value tolerances of spec 12.1 (rev 4).

Derivative and tail tests need separately justified scales, stated in the test.
"""

import numpy as np

FLOAT64_ATOL = 1e-10
FLOAT64_RTOL = 1e-8
FLOAT32_ATOL = 1e-5
FLOAT32_RTOL = 1e-4

TOLERANCES = {
    np.dtype(np.float64): {"atol": FLOAT64_ATOL, "rtol": FLOAT64_RTOL},
    np.dtype(np.float32): {"atol": FLOAT32_ATOL, "rtol": FLOAT32_RTOL},
}
