"""mpmath references for Slice B likelihood rows (spec 2, 7.1; rev 51). No survjax calls.

Every reference evaluates the spec 2 table in mpmath at 50 digits, at inputs
first rounded to the active dtype (so a comparison measures only survjax's
arithmetic):

- exact at ``t``: ``log f(t) - log Z``;
- censored to ``(lower, upper]``:
  ``log P(max(lower, L) < T <= min(upper, U)) - log Z``, with
  ``Z = P(L < T <= U)``; an empty intersection is ``-inf`` here (the caller
  decides whether it is invalid data).

``P(a < T <= b) = S(a) - S(b)`` with ``S(0) = 1`` and ``S(+inf) = 0`` (proper
built-in laws, spec 2), evaluated in cumulative-hazard form
(``S = exp(-H)``):

``log(S(a) - S(b)) = -H(a) + log(-expm1(-(H(b) - H(a))))``,

with ``H(0) = 0`` and, for ``b = +inf``, ``-H(a)``. Subtracting survival
values instead cancels even at 50 digits for tiny masses: Weibull ``k=2.5``,
``b=2``, left at ``1e-30`` has ``H ~ 1.77e-76``, so ``1 - exp(-H)`` rounds to
``0`` and the log to ``-inf``, while the true value is
``log(-expm1(-H)) ~ -174.42675``. Derivatives are ``mpmath.diff`` partials of
the same expression in the natural parameters, also at 50 digits.

Families (spec 4.1): ``"exponential"`` with parameters ``(r,)``,
``H(t) = r t``, ``log f = log r - r t``; ``"weibull"`` with
``(k, b)``, ``H(t) = (t/b)**k``,
``log f = log k - log b + (k-1) log(t/b) - (t/b)**k``; ``"lognormal"``
(rev 55) with ``(meanlog, sdlog) = (mu, sigma)``, ``z = (log t - mu)/sigma``,
``H(t) = -log Phi(-z)`` and ``log f = -log t - log sigma - log(2 pi)/2 -
z**2/2`` (``family_reference``; ``log Phi(y)`` as ``log1p(-Phi(-y))`` for
``y > 0``, so a tiny ``H`` keeps its digits); ``"loglogistic"`` (rev 55) with
``(shape, scale) = (beta, alpha)``, ``H(t) = log1p((t/alpha)**beta)`` and
``log f = log beta - log alpha + (beta-1) log(t/alpha) - 2 log1p((t/alpha)**beta)``.
"""

import math

import numpy as np
from helpers import FLOAT

DPS = 50


def _mp():
    import mpmath

    return mpmath


def rounded(*values):
    """Values rounded to the active dtype, as Python floats (``inf`` stays ``inf``)."""
    return [float(np.asarray(v, dtype=FLOAT)) for v in values]


def _cumulative_hazard(family, params, t):
    """``H(t)``, with ``H(0) = 0`` and ``H(+inf) = +inf``."""
    mp = _mp()
    if t == 0:
        return mp.mpf(0)
    if mp.isinf(t):
        return mp.inf
    if family == "exponential":
        (r,) = params
        return r * t
    if family == "lognormal":
        mu, sigma = params
        return -_log_ndtr(-(mp.log(t) - mu) / sigma)
    if family == "loglogistic":
        beta, alpha = params
        return mp.log1p((t / alpha) ** beta)
    k, b = params
    return (t / b) ** k


def _log_ndtr(y):
    """``log Phi(y)``; ``log1p(-Phi(-y))`` for ``y > 0`` keeps the digits of a tiny ``1 - Phi(y)``."""
    mp = _mp()
    if y > 0:
        return mp.log1p(-mp.ncdf(-y))
    return mp.log(mp.ncdf(y))


def _log_density(family, params, t):
    mp = _mp()
    if family == "exponential":
        (r,) = params
        return mp.log(r) - r * t
    if family == "lognormal":
        mu, sigma = params
        z = (mp.log(t) - mu) / sigma
        return -mp.log(t) - mp.log(sigma) - mp.log(2 * mp.pi) / 2 - z**2 / 2
    if family == "loglogistic":
        beta, alpha = params
        z = t / alpha
        return mp.log(beta) - mp.log(alpha) + (beta - 1) * mp.log(z) - 2 * mp.log1p(z**beta)
    k, b = params
    z = t / b
    return mp.log(k) - mp.log(b) + (k - 1) * mp.log(z) - z**k


def _log_mass(family, params, a, b):
    """``log P(a < T <= b) = -H(a) + log(-expm1(-(H(b) - H(a))))`` for ``a < b`` (module docstring)."""
    mp = _mp()
    h_a = _cumulative_hazard(family, params, a)
    h_b = _cumulative_hazard(family, params, b)
    if mp.isinf(h_b):
        return -h_a
    return -h_a + mp.log(-mp.expm1(-(h_b - h_a)))


def _contribution(family, params, lower, upper, exact, left, right):
    mp = _mp()
    log_z = _log_mass(family, params, left, right)
    if exact:
        return _log_density(family, params, lower) - log_z
    lo, hi = max(lower, left), min(upper, right)
    if not lo < hi:
        return mp.mpf("-inf")
    return _log_mass(family, params, lo, hi) - log_z


def _to_mp(values):
    mp = _mp()
    return [mp.inf if math.isinf(v) else mp.mpf(v) for v in values]


def _row(lower, upper, left, right):
    upper = lower if upper is None else upper
    return rounded(lower, upper, left, right)


def contribution(family, params, lower, upper=None, *, exact=False, left=0.0, right=math.inf):
    """mpmath value of one row's log contribution, as a float64.

    ``params`` and the row's numbers are rounded to the active dtype first.
    Exact rows pass ``exact=True`` and use ``lower`` as the event time.
    """
    mp = _mp()
    params = rounded(*params)
    lower, upper, left, right = _row(lower, upper, left, right)
    with mp.workdps(DPS):
        lo, hi, lt, rt = _to_mp([lower, upper, left, right])
        return float(_contribution(family, _to_mp(params), lo, hi, exact, lt, rt))


def _partial(family, params, row, orders):
    mp = _mp()
    lower, upper, exact, left, right = row

    def fn(*p):
        return _contribution(family, list(p), lower, upper, exact, left, right)

    return float(mp.diff(fn, tuple(params), tuple(orders)))


def gradient(family, params, lower, upper=None, *, exact=False, left=0.0, right=math.inf):
    """mpmath gradient of one row's log contribution in the natural parameters (float64 array)."""
    mp = _mp()
    params = rounded(*params)
    lower, upper, left, right = _row(lower, upper, left, right)
    n = len(params)
    with mp.workdps(DPS):
        lo, hi, lt, rt = _to_mp([lower, upper, left, right])
        row = (lo, hi, exact, lt, rt)
        p = _to_mp(params)
        return np.array(
            [_partial(family, p, row, [1 if j == i else 0 for j in range(n)]) for i in range(n)]
        )


def hessian(family, params, lower, upper=None, *, exact=False, left=0.0, right=math.inf):
    """mpmath Hessian of one row's log contribution in the natural parameters (float64 array)."""
    mp = _mp()
    params = rounded(*params)
    lower, upper, left, right = _row(lower, upper, left, right)
    n = len(params)
    out = np.zeros((n, n))
    with mp.workdps(DPS):
        lo, hi, lt, rt = _to_mp([lower, upper, left, right])
        row = (lo, hi, exact, lt, rt)
        p = _to_mp(params)
        for i in range(n):
            for j in range(i, n):
                orders = [0] * n
                orders[i] += 1
                orders[j] += 1
                out[i, j] = out[j, i] = _partial(family, p, row, orders)
    return out


def rows(family, params, table, **window):
    """``contribution`` for each ``(lower, upper, exact)`` in ``table``, as a float64 array."""
    return np.array(
        [contribution(family, params, lo, hi, exact=ex, **window) for lo, hi, ex in table]
    )
