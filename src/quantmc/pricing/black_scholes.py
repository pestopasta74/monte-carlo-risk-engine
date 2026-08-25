from math import exp, log, sqrt

from scipy.stats import norm


def european_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Price a European call option using the Black-Scholes model."""

    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt(maturity)
    )

    d2 = d1 - volatility * sqrt(maturity)

    return spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2)


def european_put_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Price a European put option using the Black-Scholes model."""

    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt(maturity)
    )

    d2 = d1 - volatility * sqrt(maturity)

    return strike * exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
