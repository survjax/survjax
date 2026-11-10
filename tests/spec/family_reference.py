"""mpmath references for the built-in families brought forward after the release (spec 4.1, 8.1, 12.1, 12.2; rev 55). No survjax calls.

Every value is evaluated at 50 digits, at inputs first rounded to the active
dtype, so a comparison measures only survjax's arithmetic.

**Log-normal** ``LogNormal(meanlog=mu, sdlog=sigma)`` (4.1: ``S(t) =
Phi(-(log t - mu)/sigma)``). With ``z = (log t - mu)/sigma`` (so 4.1's
``x = -z``):

- ``log S = log Phi(-z)``, ``log F = log Phi(z)``, ``H = -log S``;
- ``log f = -log t - log sigma - log(2 pi)/2 - z**2/2`` (``f = -dS/dt``);
- ``log h = log f - log S``.

``Phi`` of a large positive argument is ``1 - Phi(-z)``, which 50 digits cannot
hold once ``Phi(-z) < 1e-50``, so ``log Phi(y)`` is evaluated as
``log1p(-Phi(-y))`` for ``y > 0`` (the same split spec 4.1 asks survjax for,
here at 50 digits).

**Log-logistic** ``LogLogistic(shape=beta, scale=alpha)`` (4.1: ``S(t) =
1/(1+(t/alpha)**beta)``). With ``u = beta (log t - log alpha)`` and
``softplus(u) = log1p(exp(u))``:

- ``log S = -softplus(u)``, ``log F = -softplus(-u)`` (4.1), ``H = softplus(u)``;
- ``f = -dS/dt = (beta/t) w/(1+w)**2`` with ``w = exp(u)``, so ``log f = log
  beta - log alpha + (beta-1)(log t - log alpha) - 2 softplus(u)``;
- ``log h = log f - log S = log beta - log alpha + (beta-1)(log t - log
  alpha) - softplus(u)``.

Its location is ``loc = log alpha``. Parameters are passed in the
constructor's table order ``(shape, scale) = (beta, alpha)``.

**Tolerances.** A method value ``g`` (on the log scale) computed in floating
point carries an absolute error of a few ulps times

``A = M + sum_theta |theta dg/dtheta| + |t dg/dt| (1 + 2 (|log t| + |loc|))``,

where ``M`` is the sum of the magnitudes of the additive terms of ``g``, each
parameter enters with its own rounding, and the standardized argument (``z``,
or ``u/beta`` for the log-logistic) is formed from ``log t`` and the location
``loc`` (``mu``, or ``log alpha``), whose absolute rounding is ``eps (|log t| +
|loc|)`` (plus ``eps |z|`` for the scaling; ``|z| sigma <= |log t| + |loc|``,
hence the factor 2). The returned ``rtol`` is the 12.1
``rtol`` or ``16 eps A / |g|``, whichever is larger; linear quantities
(``exp(g)``) get ``16 eps (A + 1)``, the relative error of an exponential with
an argument error ``eps A``. Comparisons use zero absolute tolerance, so an
erroneous zero cannot pass (12.1). Derivatives use ``mpmath.diff`` at 50
digits.
"""

import math

import numpy as np
from helpers import FLOAT
from tolerances import TOLERANCES

DPS = 50

LOG_METHODS = ("log_survival", "log_cdf", "log_prob", "log_hazard")
DIRECT_METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)
# The log-scale quantity each direct method is derived from.
_BASE = {
    "log_survival": "log_survival",
    "survival": "log_survival",
    "cumulative_hazard": "log_survival",
    "log_cdf": "log_cdf",
    "log_prob": "log_prob",
    "log_hazard": "log_hazard",
    "hazard": "log_hazard",
}


def _mp():
    import mpmath

    return mpmath


def rounded(*values):
    """Values rounded to the active dtype, as Python floats (``inf`` stays ``inf``)."""
    return [float(np.asarray(v, dtype=FLOAT)) for v in values]


def _eps():
    return float(np.finfo(FLOAT).eps)


def _rtol(scale):
    return max(TOLERANCES[FLOAT]["rtol"], 16.0 * _eps() * float(scale))


def _log_ndtr(y):
    """``log Phi(y)`` at the working precision, without forming ``1 - tiny`` (module docstring)."""
    mp = _mp()
    if y > 0:
        return mp.log1p(-mp.ncdf(-y))
    return mp.log(mp.ncdf(y))


# --------------------------------------------------------------- log-normal


def _lognormal(method, mu, sigma, t):
    """``(g, M)`` for a log-scale method of ``LogNormal(meanlog=mu, sdlog=sigma)`` at ``t`` (module docstring)."""
    mp = _mp()
    log_t = mp.log(t)
    z = (log_t - mu) / sigma
    log_s = _log_ndtr(-z)
    log_f = -log_t - mp.log(sigma) - mp.log(2 * mp.pi) / 2 - z**2 / 2
    m_f = abs(log_t) + abs(mp.log(sigma)) + mp.log(2 * mp.pi) / 2 + z**2 / 2
    if method == "log_survival":
        return log_s, abs(log_s)
    if method == "log_cdf":
        log_c = _log_ndtr(z)
        return log_c, abs(log_c)
    if method == "log_prob":
        return log_f, m_f
    if method == "log_hazard":
        return log_f - log_s, m_f + abs(log_s)
    raise ValueError(method)


# --------------------------------------------------------------- log-logistic


def _softplus(u):
    mp = _mp()
    return mp.log1p(mp.exp(u))


def _loglogistic(method, beta, alpha, t):
    """``(g, M)`` for a log-scale method of ``LogLogistic(shape=beta, scale=alpha)`` at ``t`` (module docstring)."""
    mp = _mp()
    log_ratio = mp.log(t) - mp.log(alpha)
    u = beta * log_ratio
    sp = _softplus(u)
    head = mp.log(beta) - mp.log(alpha) + (beta - 1) * log_ratio
    m_head = abs(mp.log(beta)) + abs(mp.log(alpha)) + abs((beta - 1) * log_ratio)
    if method == "log_survival":
        return -sp, sp
    if method == "log_cdf":
        sm = _softplus(-u)
        return -sm, sm
    if method == "log_prob":
        return head - 2 * sp, m_head + 2 * sp
    if method == "log_hazard":
        return head - sp, m_head + sp
    raise ValueError(method)


# (log-method function, location subtracted from log t as a function of the parameters)
_FAMILIES = {
    "lognormal": (_lognormal, lambda p: p[0]),
    "loglogistic": (_loglogistic, lambda p: _mp().log(p[1])),
}


def _location(family, params):
    """``mu`` (log-normal) or ``log alpha`` (log-logistic): the location subtracted from ``log t``."""
    return _FAMILIES[family][1](params)


def _base_and_partials(family, params, t, base):
    mp = _mp()
    fn = _FAMILIES[family][0]
    args = [mp.mpf(x) for x in (*params, t)]
    g, magnitude = fn(base, *args)
    n = len(args)
    partials = [
        mp.diff(lambda *a: fn(base, *a)[0], tuple(args), tuple(1 if j == i else 0 for j in range(n)))
        for i in range(n)
    ]
    return args, g, magnitude, partials


def direct(family, params, t, method):
    """``(value, rtol)`` of a direct method at a finite ``t > 0``, from mpmath (module docstring).

    ``params`` are the constructor's natural parameters in the 4.1 table order
    (log-normal ``(meanlog, sdlog)``; log-logistic ``(shape, scale)``).
    """
    mp = _mp()
    params = rounded(*params)
    (t,) = rounded(t)
    base = _BASE[method]
    with mp.workdps(DPS):
        args, g, magnitude, partials = _base_and_partials(family, params, t, base)
        *theta, tt = args
        loc = _location(family, theta)
        scale = magnitude + sum(abs(x * d) for x, d in zip(theta, partials[:-1], strict=True))
        scale += abs(tt * partials[-1]) * (1 + 2 * (abs(mp.log(tt)) + abs(loc)))
        if g == 0 and method != "survival" and method != "hazard":
            raise ValueError(f"{method} is exactly 0 here; a relative tolerance needs another case")
        if method in LOG_METHODS:
            return float(g), _rtol(scale / abs(g))
        if method == "cumulative_hazard":
            return float(-g), _rtol(scale / abs(g))
        return float(mp.exp(g)), _rtol(scale + 1)


def direct_derivative(family, params, t, method, orders):
    """mpmath partial derivative of a log-scale direct method in the natural parameters (``orders`` per parameter)."""
    mp = _mp()
    params = rounded(*params)
    (t,) = rounded(t)
    fn = _FAMILIES[family][0]
    with mp.workdps(DPS):
        tt = mp.mpf(t)
        return float(
            mp.diff(lambda *p: fn(method, *p, tt)[0], tuple(mp.mpf(x) for x in params), tuple(orders))
        )


def lognormal_time(mu, sigma, x):
    """The time at which ``-(log t - mu)/sigma`` equals ``x`` (spec 4.1's argument): ``exp(mu - sigma x)``."""
    return math.exp(mu - sigma * x)


def loglogistic_time(beta, alpha, u):
    """The time at which ``beta (log t - log alpha)`` equals ``u``: ``alpha exp(u/beta)``."""
    return alpha * math.exp(u / beta)
