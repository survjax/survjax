"""Independent mpmath references for staged RMST (spec 8.1, 8.7, 12.1; rev 41).

No survjax code is used here. RMST is ``int_0^tau S(t) dt`` (8.1), and its
first parameter derivatives are ``int_0^tau dS/dtheta dt`` (differentiation
under the integral; ``d/dtau = S(tau)``). Both are computed by mpmath
tanh-sinh quadrature of the **physical survival law**, not of a closed form.

Quadrature on very long intervals can miss mass if the nodes do not resolve
where ``S`` falls. The interval is therefore split at the times where the
cumulative hazard ``H(t) = c (t/b0)**k`` crosses fixed levels from ``1e-12``
to ``512`` (each subinterval holds a smooth piece of the law), and cut where
``H`` reaches ``CUT = 2000``: beyond it ``S < exp(-2000)``, and every
integrand here is ``S`` times at most a polynomial in ``H`` and ``log t``, so
the omitted tail is below ``1e-800`` relative. ``test_a12_rmst_reference.py``
validates these integrals against the independent closed form
``b Gamma(1+1/k) P(1/k, (tau/b)**k)`` (mpmath ``gammainc``), its
high-precision numerical derivatives and a refined quadrature.
"""

import functools

import mpmath

DPS = 40
CUT = 2000
LEVELS = (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _mp(x):
    """An ``mpf``: Python/NumPy floats exactly; ``mpf`` values (``mpmath.diff`` steps) unchanged."""
    if isinstance(x, mpmath.mpf):
        return x
    return mpmath.mpf(float(x))


def breakpoints(k, b0, c, tau, levels=LEVELS):
    """Split points of ``[0, min(tau, t_cut)]`` at ``H(t) = c (t/b0)**k = level``."""
    k, b0, c, tau = _mp(k), _mp(b0), _mp(c), _mp(tau)
    t_cut = b0 * (CUT / c) ** (1 / k)
    upper = min(tau, t_cut)
    pts = {mpmath.mpf(0), upper}
    for level in levels:
        t = b0 * (mpmath.mpf(level) / c) ** (1 / k)
        if 0 < t < upper:
            pts.add(t)
    return sorted(pts)


def _law(k, b0, c):
    """``S(t) = exp(-c (t/b0)**k)`` and its cumulative hazard."""

    def hazard(t):
        return c * (t / b0) ** k

    def survival(t):
        return mpmath.exp(-hazard(t))

    return survival, hazard


@functools.cache
def ph_weibull(k, b0, eta, tau, levels=LEVELS):
    """RMST of ``S(t) = exp(-exp(eta) (t/b0)**k)`` and its first derivatives, by quadrature.

    Returns floats ``(rmst, d/dk, d/db0, d/deta, d/dtau)``. With
    ``u = exp(eta) (t/b0)**k``: ``dS/dk = -S u log(t/b0)``,
    ``dS/db0 = S u k / b0``, ``dS/deta = -S u``; ``d/dtau = S(tau)``.
    ``eta = 0`` is the Weibull law ``(k, b = b0)``.
    """
    with mpmath.workdps(DPS):
        k_, b0_, eta_, tau_ = _mp(k), _mp(b0), _mp(eta), _mp(tau)
        c = mpmath.exp(eta_)
        survival, hazard = _law(k_, b0_, c)
        pts = breakpoints(k_, b0_, c, tau_, levels)
        if tau_ == 0:
            return 0.0, 0.0, 0.0, 0.0, 1.0

        def quad(fn):
            return mpmath.quad(fn, pts)

        value = quad(survival)
        dk = quad(lambda t: -survival(t) * hazard(t) * mpmath.log(t / b0_))
        db0 = quad(lambda t: survival(t) * hazard(t) * k_ / b0_)
        deta = quad(lambda t: -survival(t) * hazard(t))
        return float(value), float(dk), float(db0), float(deta), float(survival(tau_))


def weibull(k, b, tau):
    """RMST of ``Weibull(k, b)`` and ``(d/dk, d/db, d/dtau)`` by quadrature of ``exp(-(t/b)**k)``."""
    value, dk, db, _, dtau = ph_weibull(float(k), float(b), 0.0, float(tau))
    return value, dk, db, dtau


def weibull_closed_form(k, b, tau):
    """``b Gamma(1+1/k) P(1/k, (tau/b)**k)`` (8.7), an mpmath ``mpf`` at the working precision."""
    k, b, tau = _mp(k), _mp(b), _mp(tau)
    return b * mpmath.gamma(1 + 1 / k) * mpmath.gammainc(1 / k, 0, (tau / b) ** k, regularized=True)


def weibull_closed_form_derivative(k, b, tau, orders):
    """Partial derivative of the closed form, ``orders = (n_k, n_b, n_tau)``, by mpmath ``diff`` at 60 digits."""
    with mpmath.workdps(60):
        return float(
            mpmath.diff(lambda kk, bb, tt: weibull_closed_form(kk, bb, tt), (_mp(k), _mp(b), _mp(tau)), orders)
        )


def ph_weibull_closed_form_derivative(k, b0, eta, tau, orders):
    """Partial derivative in ``(k, b0, eta, tau)`` of the PH-Weibull RMST.

    The PH-Weibull law ``exp(-exp(eta)(t/b0)**k)`` is ``Weibull(k, b0 exp(-eta/k))``
    (algebra: ``exp(eta) (t/b0)**k = (t / (b0 exp(-eta/k)))**k``), so the closed
    form at that scale is an independent analytic reference for higher orders.
    """
    with mpmath.workdps(60):

        def fn(kk, bb, ee, tt):
            return weibull_closed_form(kk, bb * mpmath.exp(-ee / kk), tt)

        return float(mpmath.diff(fn, (_mp(k), _mp(b0), _mp(eta), _mp(tau)), orders))


def exponential(r, tau):
    """``-expm1(-r tau)/r`` (8.7) and its derivatives ``(d/dr, d/dtau)``, in mpmath.

    ``R = (1 - e)/r`` with ``e = exp(-r tau)``: ``dR/dr = tau e/r - (1 - e)/r**2``,
    ``dR/dtau = e``.
    """
    with mpmath.workdps(DPS):
        r, tau = _mp(r), _mp(tau)
        e = mpmath.exp(-r * tau)
        one_minus = -mpmath.expm1(-r * tau)
        return float(one_minus / r), float(tau * e / r - one_minus / r**2), float(e)


def exponential_derivative(r, tau, orders):
    """Partial derivative of ``-expm1(-r tau)/r`` in ``(r, tau)`` by mpmath ``diff`` at 60 digits."""
    with mpmath.workdps(60):
        return float(mpmath.diff(lambda rr, tt: -mpmath.expm1(-rr * tt) / rr, (_mp(r), _mp(tau)), orders))
