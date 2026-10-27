# The circle of survival: Functions for (parametric) survival analysis

## Intro

Survival analysis is the topic of describing expected failure times (e.g. time to death, or failure of a car-part), with the particular twist that observations may be **exact**: an event (e.g. death) is observed to have occurred at a certain time, or **right-censored** where an observation was still alive at some date, so all that is known is that the failure time is past this time.

The canonical end-goal can be seen as generating a Survival Curve $S(t)$ (a.k.a. *Survivor Function*) which for every time $t$ denotes the fraction of the population still alive. We have $S(0)=1$ and $S(\inf)=0$ (for proper survival distributions).

We'll now describe the *circle of survival*: a set of functions that are deterministically related and each describe the same survival process but from a different angle, but together they form as what we'll call the **circle of survival**.

We'll illustrate each using the survival curve of Dutch men aged 18 and more, taken from the CBS, 2025

## Survival Curve $S(t)$

At each time $t$, what fraction of the population is still alive?

left: constraints + figure

As can be seen, before the age of 60, not much is happening, but after age 60, survival starts decreasing clearly.

## Cumulative Incidence $F(t)$

An alternative representation is the cumulative Incidence
Relation: $F = 1-S$

constraints + figure

## Density $f(t)$

To better see where most events (deaths) occur, we can calculate the event density, namely the first derivative of the Cumulative Incidence:

relation: $f = \frac{d}{dt} F$

constraints + figure

## Hazard $h(t)$

One notable observation is that death density peaks around 80 but then rises. From a population standpoint this is understandable, as fewer and fewer man make it past 80. At the same time we have the intuition that *for men who make it to 80*, the risk of death should keep increasing. This notion is formalized with the **hazard function**, a.k.a. 'the force of death' which is the density ('risk of death at $t$') divided by the Survival Curve ('given that the person is still alive at $t$'):

relation: $h = f / S$

constraints + figure

## Cumulative Hazard $H(t)$

A natural next step is to ask how much *hazard* (force of death) someone has accrued over time, which is the **Cumulative Hazard**:

relation: $H = \int_0^t h$

constraints + figure

## Connecting the Cumulative Hazard and the Survival Curve

To finish the circle we need one final piece of information: how are $H$ and $S$ related? Though maybe less intuitive at first sight, the relationship comes from a mathematical identity:

$H = \int h = \int f/S = - \int \frac{S'}{S}$

Realizing that $\frac{d}{dx} \log f(x) = \frac{f'(x)}{f(x)}$ we see that

$H = - \log S$, or equivalently $S = e^{-H}$

This completes the circle!
