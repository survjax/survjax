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
built-in laws, spec 2). Derivatives are ``mpmath.diff`` partials of the same
expression in the natural parameters, also at 50 digits.

Families (spec 4.1): ``"exponential"`` with parameters ``(r,)``,
``S(t) = exp(-r t)``, ``log f = log r - r t``; ``"weibull"`` with
``(k, b)``, ``S(t) = exp(-(t/b)**k)``,
``log f = log k - log b + (k-1) log(t/b) - (t/b)**k``.
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


def _survival(family, params, t):
    mp = _mp()
    if t == 0:
        return mp.mpf(1)
    if mp.isinf(t):
        return mp.mpf(0)
    if family == "exponential":
        (r,) = params
        return mp.exp(-r * t)
    k, b = params
    return mp.exp(-((t / b) ** k))


def _log_density(family, params, t):
    mp = _mp()
    if family == "exponential":
        (r,) = params
        return mp.log(r) - r * t
    k, b = params
    z = t / b
    return mp.log(k) - mp.log(b) + (k - 1) * mp.log(z) - z**k


def _log_mass(family, params, a, b):
    mp = _mp()
    return mp.log(_survival(family, params, a) - _survival(family, params, b))


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
