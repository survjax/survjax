"""Independent mpmath references for the post-release summaries (spec 8.1, 8.8, 12.1; rev 50).

No survjax code is used here. Closed forms are those of spec 8.1 (and their
derivatives by mpmath ``diff`` at 60 digits); log-logistic RMST is mpmath
quadrature of the physical law ``S(t) = 1/(1 + (t/alpha)**beta)`` split at
``alpha * 10**j``.

- Weibull mean ``b Gamma(1+1/k)``; quantile ``b (-log1p(-p))**(1/k)``.
- Log-logistic mean ``alpha (pi/beta) / sin(pi/beta)`` for ``beta > 1``.
- Uniform on ``(0, end]``: RMST ``tau - tau**2/(2 end)`` (``tau <= end``),
  mean ``end/2``, quantile ``p end``.
- Mixture cure ``S = c + (1-c) exp(-r t)``: RMST
  ``c tau + (1-c)(1 - exp(-r tau))/r``; for ``p < 1-c`` the quantile solves
  ``(1-c) exp(-r q) = 1-p-c``: ``q = log((1-c)/(1-p-c))/r``; for ``p >= 1-c``
  (``e = -log1p(-p) >= H(+inf) = -log c``) it is ``+inf`` (8.2).
"""

import mpmath

DPS = 40


def _mp(x):
    if isinstance(x, mpmath.mpf):
        return x
    return mpmath.mpf(float(x))


def weibull_mean(k, b):
    with mpmath.workdps(DPS):
        k, b = _mp(k), _mp(b)
        return float(b * mpmath.gamma(1 + 1 / k))


def weibull_mean_derivative(k, b, orders):
    """Partial derivative of ``b Gamma(1+1/k)`` in ``(k, b)``, ``orders = (n_k, n_b)``."""
    with mpmath.workdps(60):
        return float(mpmath.diff(lambda kk, bb: bb * mpmath.gamma(1 + 1 / kk), (_mp(k), _mp(b)), orders))


def weibull_quantile(k, b, p):
    with mpmath.workdps(DPS):
        k, b, p = _mp(k), _mp(b), _mp(p)
        return float(b * (-mpmath.log1p(-p)) ** (1 / k))


def weibull_quantile_derivative(k, b, p, orders):
    """Partial derivative of ``b (-log1p(-p))**(1/k)`` in ``(k, b, p)``."""
    with mpmath.workdps(60):
        return float(
            mpmath.diff(
                lambda kk, bb, pp: bb * (-mpmath.log1p(-pp)) ** (1 / kk), (_mp(k), _mp(b), _mp(p)), orders
            )
        )


def log_logistic_mean(alpha, beta):
    with mpmath.workdps(DPS):
        alpha, beta = _mp(alpha), _mp(beta)
        return float(alpha * (mpmath.pi / beta) / mpmath.sin(mpmath.pi / beta))


def log_logistic_rmst(alpha, beta, tau):
    with mpmath.workdps(DPS):
        alpha, beta, tau = _mp(alpha), _mp(beta), _mp(tau)
        pts = [mpmath.mpf(0)]
        for j in range(-6, 30):
            t = alpha * mpmath.mpf(10) ** j
            if 0 < t < tau:
                pts.append(t)
        pts.append(tau)
        return float(mpmath.quad(lambda t: 1 / (1 + (t / alpha) ** beta), pts))


def mixture_cure_rmst(c, r, tau):
    with mpmath.workdps(DPS):
        c, r, tau = _mp(c), _mp(r), _mp(tau)
        return float(c * tau - (1 - c) * mpmath.expm1(-r * tau) / r)


def mixture_cure_quantile(c, r, p):
    with mpmath.workdps(DPS):
        c, r, p = _mp(c), _mp(r), _mp(p)
        if p >= 1 - c:
            return float("inf")
        return float(mpmath.log((1 - c) / (1 - p - c)) / r)
