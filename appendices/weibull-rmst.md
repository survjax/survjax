---
title: "Mathematical appendix: Weibull RMST"
fontsize: 11pt
geometry: margin=0.8in
colorlinks: true
---

Mathematical reference for [`_weibull_rmst`](../src/survjax/_numerics.py).
The symbols $k$, $b$ and $\tau$ denote shape, scale and cutoff; $k$ is
dimensionless and $b,\tau$ share time units. Error bounds below concern
real arithmetic; rounding and special-function error are additional effects.

# Integral and gamma identity

For positive shape $k$, positive scale $b$, and finite cutoff $\tau>0$, write
$$
S(t)=\exp\{-(t/b)^k\},\qquad M(k,b,\tau)=\int_0^\tau S(t)\,dt.
$$
Set $a=1/k$, $x=(\tau/b)^k$, and substitute $u=(t/b)^k$.
Then $dt=ba\,u^{a-1}du$ and
$$
M=ba\int_0^x u^{a-1}e^{-u}\,du
 =ba\,\gamma(a,x)=b\Gamma(1+a)P(a,x).
$$
Here $\gamma$ is the lower incomplete gamma function, $P(a,x)=\gamma(a,x)/\Gamma(a)$,
and $Q(a,x)=1-P(a,x)$. These definitions are given in
[NIST DLMF section 8.2](https://dlmf.nist.gov/8.2).
The unrestricted mathematical mean is $\mu=b\Gamma(1+a)$, so $M=\mu(1-Q)$.

An equivalent normalization, useful for the lower tail, is
$$
M=\tau\int_0^1\exp(-xv^k)\,dv.
$$
The target is always between 0 and $\tau$. A tiny $x$ therefore means
$M$ is close to $\tau$, not close to zero.

# Lower-tail series

Expanding the last integrand and integrating each term gives
$$
M=\tau\sum_{n=0}^{\infty}\frac{(-x)^n}{n!(kn+1)}.
$$
Termwise integration is valid by uniform absolute convergence on $v\in[0,1]$
for fixed finite $x$. This is also consistent with the incomplete-gamma series
in [NIST DLMF section 8.7](https://dlmf.nist.gov/8.7).
The implementation evaluates
$$
M_{12}=\tau\sum_{n=0}^{12}\frac{(-x)^n}{n!(kn+1)}.
$$
It starts `term = 1`, updates `term *= -x/n`, and adds `term/(k*n+1)`.
It forms neither factorials nor large powers explicitly.

For $0\le x<0.1$, the magnitudes decrease: the ratio of consecutive terms is
$$
\frac{x}{n+1}\frac{kn+1}{k(n+1)+1}<\frac{0.1}{n+1}.
$$
The alternating-series remainder therefore satisfies
$$
|M-M_{12}|\le\tau\frac{x^{13}}{13!(13k+1)}.
$$
Since $M\ge\tau e^{-x}$, a uniform relative bound is
$$
\frac{|M-M_{12}|}{M}
\le e^x\frac{x^{13}}{13!(13k+1)}
<e^{0.1}\frac{0.1^{13}}{13!}<1.78\times10^{-23}.
$$
Thus `0.1` makes convergence rapid and the fixed loop simple; degree 12 gives
ample margin below float64 rounding. These choices are conservative, not
claimed to be the smallest sufficient degree or an optimized crossover.

Differentiating the polynomial with respect to the cutoff also has a useful form:
$$
\frac{\partial M_{12}}{\partial\tau}
=\sum_{n=0}^{12}\frac{(-x)^n}{n!}.
$$
Its absolute error as an approximation to $S(\tau)=e^{-x}$ is at most
$x^{13}/13!<1.61\times10^{-23}$. This is a cutoff-derivative bound only;
it is not a uniform bound for every shape, scale, or higher derivative.
If computing $x$ underflows to zero, the polynomial returns $\tau$ naturally.
No epsilon, probability floor, or altered cutoff is introduced.

# Gamma middle region

The code first computes $\ell=k(\log\tau-\log b)=\log x$, without forming
$\tau/b$. For the middle branch it evaluates `gammainc(a, exp(ell))` and
$$
\log M=\log b+\log\Gamma(1+a)+\log P(a,x).
$$
Only the final result is exponentiated. This avoids an overflowing mean factor
multiplied by a small probability when the final product is representable.
It does not implement a new incomplete-gamma algorithm: special-function
accuracy still depends on JAX.

For $k\in[0.2,50]$, the middle region has $a\in[0.02,5]$ and
$x\in[0.1,84]$, avoiding extreme gamma probabilities. If a numerical gamma
probability underflows to zero outside this range, the mathematical integral
can still be positive. Returning NaN signals lost precision; replacing that
probability by a floor or reporting zero would change the mathematical meaning.

# Mean-tail approximation

Returning $\mu$ instead of $M$ has absolute error $\mu Q(a,x)$ and relative error
$$
\frac{\mu-M}{M}=\frac{Q(a,x)}{1-Q(a,x)}.
$$
A direct bound explains both constants in the threshold. Let $Y$ have a gamma
law with shape $a$ and unit scale. From its defining integral, for $0<s<1$,
$$
\operatorname{E}[e^{sY}]=(1-s)^{-a}.
$$
Markov's inequality with $s=3/4$ gives
$$
Q(a,x)=\Pr(Y\ge x)\le e^{-3x/4}4^a.
$$
If $x>64+4a$, this becomes
$$
Q(a,x)<\exp\{-48+a(\log4-3)\}<e^{-48}
 =1.4251641\times10^{-21},
$$
because $\log4-3<0$ and $a>0$. Consequently the relative error is below
$e^{-48}/(1-e^{-48})$. This real-arithmetic bound holds for every positive $k$;
it is distinct from a guarantee about floating-point evaluation.
The fixed margin 64 is deliberately generous; the $4a$ term controls dependence
on gamma shape. The constants give a conservative, shape-uniform tail bound.

The cutoff derivative of this branch is zero, whereas the exact derivative is
$S(\tau)=e^{-x}<e^{-64}<1.61\times10^{-28}$. That is negligible at the required
absolute scale 1 for cutoff derivatives, but it is **not relative accuracy**
for an extremely small nonzero derivative. Parameter derivatives of the mean
also approximate those of RMST; the value bound alone does not prove their errors.

The threshold is tested as $\ell>\log(64+4/k)$, using `logaddexp` to form the
right side. The large, possibly unrepresentable $x$ need never be constructed.

# Stable evaluation

The thresholds depend on $x$, not on $\tau/b$ alone. For illustration:

| $k$ | Series when $\tau/b$ is below | Mean when $\tau/b$ is above |
|---|---:|---:|
| 0.2 | $10^{-5}$ | $4.182119424\times10^9$ |
| 1 | 0.1 | 68 |
| 50 | 0.9549925860 | 1.0867620143 |

For $k\in[0.2,50]$ and $\tau/b\in[0.01,10^5]$, $k=50$ makes $x$ run from $10^{-100}$ to
$10^{250}$. Both endpoints are representable in float64, but not in float32.
The more extreme ratios $\tau/b=10^{-8}$ and $10^8$ give
$x=10^{-400}$ and $10^{400}$, beyond either dtype, while RMST is representable.

A final `where` is insufficient protection: autodiff can encounter an infinite
or NaN derivative in an unselected expression, and batching evaluates all
branches. Each branch therefore receives its own safe operands *before* the
logarithm, exponential, or gamma call. Inactive gamma inputs use $k=b=x=1$;
inactive series inputs use $k=x=1$; inactive mean inputs use $k=b=1$.
These are discarded finite calculations, not replacements for valid active inputs.
All operands must use their common floating dtype before evaluating these
expressions. A wider cutoff dtype can make the result representable when the
parameter-storage dtype alone cannot. For example, $k=1/4$ gives
$\mu=b\Gamma(5)=24b$.

```{=latex}
\clearpage
```

# Derivatives and zero cutoff

For positive $k,b,\tau$, differentiation gives
$$
\frac{\partial M}{\partial\tau}=S(\tau),\qquad
\frac{\partial M}{\partial b}=\frac{M-\tau S(\tau)}{b},
$$
$$
\frac{\partial M}{\partial k}
=-\int_0^\tau e^{-(t/b)^k}(t/b)^k\log(t/b)\,dt.
$$
The scale identity also follows from homogeneity: $M=b\,m(\tau/b)$.
These identities concern the exact integral. Derivatives of numerical
approximations have separate errors; a value bound alone does not bound all
parameter derivatives.

At the origin,
$$
M(k,b,0)=0,\qquad \partial_kM=\partial_bM=0,\qquad
\partial_\tau^+M=1.
$$
Returning the cutoff itself at exactly zero represents these boundary values
and first derivatives. It is not a representation of higher boundary calculus.
For $\tau>0$,
$$
\partial_\tau^2M=-\frac{k}{b^k}\tau^{k-1}e^{-(\tau/b)^k}.
$$
Its right limit is $-\infty$ for $0<k<1$, $-1/b$ for $k=1$, and 0 for
$k>1$. For integer $k=m$, the first nonzero derivative beyond the first is
of order $m+1$, with value $-m!/b^m$ at zero. Thus no finite, all-orders
boundary calculus covers all positive shapes. Higher-order calculations
require a smooth interior point; at zero the numerical branch represents
only the value and first derivatives.
