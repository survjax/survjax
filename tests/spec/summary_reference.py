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
- Log-normal (rev 55, 8.1 table): mean ``exp(mu + sigma**2/2)``; quantile
  ``exp(mu + sigma Phi⁻¹(p))``; ``H⁻¹(e) = exp(mu - sigma y)`` with ``Phi(y) =
  exp(-e)`` (``S = exp(-e)``, 8.2); RMST ``tau S(tau) + exp(mu + sigma**2/2)
  Phi((log tau - mu - sigma**2)/sigma)``, by parts: ``∫_0^tau S = tau S(tau) +
  ∫_0^tau t f(t) dt``, and the partial expectation of a log-normal is the last
  term. ``Phi⁻¹`` solves ``log Phi(y) = log p`` with ``mpmath.findroot``, using
  ``log Phi(y) = log1p(-Phi(-y))`` for ``y > 0``, so both tails keep their
  digits (``exp(-e)`` down to ``1e-261`` for ``e = 600``).
- Log-logistic quantile ``alpha (p/(1-p))**(1/beta)``; ``H⁻¹(e) = alpha
  expm1(e)**(1/beta)`` (8.2, from ``H = log1p((t/alpha)**beta)``). RMST
  parameter derivatives are mpmath quadratures of ``dS/dtheta`` (8.4: the
  generic tangent ``∫ ∂θ S dθ dt``): with ``w = (t/alpha)**beta``, ``dS/dalpha
  = beta w / (alpha (1+w)**2)`` and ``dS/dbeta = -w log(t/alpha) / (1+w)**2``.
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


def _log_logistic_points(alpha, tau):
    pts = [mpmath.mpf(0)]
    for j in range(-6, 30):
        t = alpha * mpmath.mpf(10) ** j
        if 0 < t < tau:
            pts.append(t)
    pts.append(tau)
    return pts


def log_logistic_rmst(alpha, beta, tau):
    with mpmath.workdps(DPS):
        alpha, beta, tau = _mp(alpha), _mp(beta), _mp(tau)
        return float(mpmath.quad(lambda t: 1 / (1 + (t / alpha) ** beta), _log_logistic_points(alpha, tau)))


def log_logistic_rmst_derivative(alpha, beta, tau, wrt):
    """``d rmst/d alpha`` or ``d rmst/d beta`` (``wrt``) by quadrature of ``dS/dtheta`` (module docstring)."""
    with mpmath.workdps(DPS):
        alpha, beta, tau = _mp(alpha), _mp(beta), _mp(tau)

        def ds(t):
            w = (t / alpha) ** beta
            if wrt == "alpha":
                return beta * w / (alpha * (1 + w) ** 2)
            return -w * mpmath.log(t / alpha) / (1 + w) ** 2

        return float(mpmath.quad(ds, _log_logistic_points(alpha, tau)))


def log_logistic_mean_mp(alpha, beta):
    return alpha * (mpmath.pi / beta) / mpmath.sin(mpmath.pi / beta)


def log_logistic_quantile(alpha, beta, p):
    with mpmath.workdps(DPS):
        alpha, beta, p = _mp(alpha), _mp(beta), _mp(p)
        return float(alpha * (p / (1 - p)) ** (1 / beta))


def log_logistic_inverse_cumulative_hazard(alpha, beta, e):
    with mpmath.workdps(DPS):
        alpha, beta, e = _mp(alpha), _mp(beta), _mp(e)
        return float(alpha * mpmath.expm1(e) ** (1 / beta))


def mixture_cure_rmst(c, r, tau):
    with mpmath.workdps(DPS):
        c, r, tau = _mp(c), _mp(r), _mp(tau)
        return float(c * tau - (1 - c) * mpmath.expm1(-r * tau) / r)


def derivative(fn, args, orders):
    """``mpmath.diff`` at 60 digits of an mpmath expression ``fn(*args)`` (``orders`` per argument)."""
    with mpmath.workdps(60):
        return float(mpmath.diff(fn, tuple(_mp(a) for a in args), tuple(orders)))


def _log_ndtr(y):
    if y > 0:
        return mpmath.log1p(-mpmath.ncdf(-y))
    return mpmath.log(mpmath.ncdf(y))


def ndtri_from_log(log_p):
    """``y`` with ``log Phi(y) = log_p`` for ``log_p < 0`` (``Phi⁻¹`` of ``exp(log_p)``), by ``findroot``.

    Solved as ``log(-log Phi(y)) = log(-log_p)``, which is well scaled in both
    tails (``log_p = -1e-30`` and ``-600`` alike).
    """
    log_p = _mp(log_p) if not isinstance(log_p, mpmath.mpf) else log_p
    if log_p < -1:
        start = -mpmath.sqrt(-2 * log_p)
    elif log_p > -1e-3:
        start = mpmath.sqrt(-2 * mpmath.log(-log_p))
    else:
        start = mpmath.mpf(0)
    target = mpmath.log(-log_p)
    return mpmath.findroot(lambda y: mpmath.log(-_log_ndtr(y)) - target, start)


def ndtri(p):
    """``Phi⁻¹(p)`` for ``0 < p < 1``, symmetric so that ``p`` near 1 keeps its digits."""
    p = _mp(p) if not isinstance(p, mpmath.mpf) else p
    if p <= 0.5:
        return ndtri_from_log(mpmath.log(p))
    return -ndtri_from_log(mpmath.log1p(-p))


def lognormal_mean_mp(mu, sigma):
    return mpmath.exp(mu + sigma**2 / 2)


def lognormal_quantile_mp(mu, sigma, p):
    return mpmath.exp(mu + sigma * ndtri(p))


def lognormal_inverse_cumulative_hazard_mp(mu, sigma, e):
    """``S = exp(-e)``: ``Phi(-(log t - mu)/sigma) = exp(-e)``, so ``log t = mu - sigma ndtri_from_log(-e)``."""
    return mpmath.exp(mu - sigma * ndtri_from_log(-e))


def lognormal_rmst_mp(mu, sigma, tau):
    z = (mpmath.log(tau) - mu) / sigma
    return tau * mpmath.ncdf(-z) + mpmath.exp(mu + sigma**2 / 2) * mpmath.ncdf(z - sigma)


def _value(fn, *args):
    with mpmath.workdps(DPS):
        return float(fn(*(_mp(a) for a in args)))


def lognormal_mean(mu, sigma):
    return _value(lognormal_mean_mp, mu, sigma)


def lognormal_quantile(mu, sigma, p):
    return _value(lognormal_quantile_mp, mu, sigma, p)


def lognormal_inverse_cumulative_hazard(mu, sigma, e):
    return _value(lognormal_inverse_cumulative_hazard_mp, mu, sigma, e)


def lognormal_rmst(mu, sigma, tau):
    return _value(lognormal_rmst_mp, mu, sigma, tau)


def mixture_cure_quantile(c, r, p):
    with mpmath.workdps(DPS):
        c, r, p = _mp(c), _mp(r), _mp(p)
        if p >= 1 - c:
            return float("inf")
        return float(mpmath.log((1 - c) / (1 - p - c)) / r)
