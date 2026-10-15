"""Implementer test tolerances (spec 12.1)."""

# check_grads uses finite differences with step 1e-4. Second-order differences
# in float32 amplify rounding for O(1) polynomial values/derivatives.
# At |f| ~= 6, eps32 * |f| / 1e-4 ~= 7e-3 is a conservative rounding
# scale; this exactly bilinear test needs only 1e-3 in both derivative modes.
# Grid checks use r=1 and t in {0.5, 1}: smooth values/derivatives are O(1).
# With step 1e-4, first differences of AD derivatives have rounding scale
# eps32/1e-4 ~= 1e-3 (eps64/1e-4 ~= 2e-12), within these existing bounds.
DERIVATIVE_ATOL = {True: 1e-7, False: 1e-3}
DERIVATIVE_RTOL = {True: 1e-7, False: 1e-3}

ATOL = {True: 1e-10, False: 1e-5}
RTOL = {True: 1e-8, False: 1e-4}
