"""Implementer test tolerances (spec 12.1)."""

# check_grads uses finite differences with step 1e-4. Second-order differences
# in float32 amplify rounding for O(1) polynomial values/derivatives.
DERIVATIVE_ATOL = {True: 1e-7, False: 2e-2}
DERIVATIVE_RTOL = {True: 1e-7, False: 2e-2}
