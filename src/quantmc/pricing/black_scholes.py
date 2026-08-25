from math import exp, log, sqrt

from scipy.stats import norm


def _calculate_d1_d2(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> tuple[float, float]:
    """Calculate the Black-Scholes standardized terms."""

    volatility_time = volatility * sqrt(maturity)

    d1 = (
        log(spot / strike) + (rate + 0.5 * volatility**2) * maturity
    ) / volatility_time

    d2 = d1 - volatility_time

    return d1, d2


def european_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Price a European call option using the Black-Scholes model."""

    d1, d2 = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    price = spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2)

    return float(price)


def european_put_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Price a European put option using the Black-Scholes model."""

    d1, d2 = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    price = strike * exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return float(price)
