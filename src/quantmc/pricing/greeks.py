from math import exp, sqrt

from scipy.stats import norm

from quantmc.pricing.black_scholes import _calculate_d1_d2


def european_call_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the Black-Scholes Delta of a European call option."""

    d1, _ = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    return float(norm.cdf(d1))


def european_put_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the Black-Scholes Delta of a European put option."""

    d1, _ = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    return float(norm.cdf(d1) - 1.0)


def european_option_gamma(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the Black-Scholes Gamma of a European option."""

    d1, _ = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    gamma = norm.pdf(d1) / (spot * volatility * sqrt(maturity))

    return float(gamma)


def european_option_vega(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the Black-Scholes Vega of a European option."""

    d1, _ = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    vega = spot * norm.pdf(d1) * sqrt(maturity)

    return float(vega)


def european_call_theta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the annual Black-Scholes Theta of a European call."""

    d1, d2 = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    diffusion_term = -spot * norm.pdf(d1) * volatility / (2.0 * sqrt(maturity))
    discount_term = -rate * strike * exp(-rate * maturity) * norm.cdf(d2)

    return float(diffusion_term + discount_term)


def european_put_theta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the annual Black-Scholes Theta of a European put."""

    d1, d2 = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    diffusion_term = -spot * norm.pdf(d1) * volatility / (2.0 * sqrt(maturity))
    discount_term = rate * strike * exp(-rate * maturity) * norm.cdf(-d2)

    return float(diffusion_term + discount_term)


def european_call_rho(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the Black-Scholes Rho of a European call."""

    _, d2 = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    rho = strike * maturity * exp(-rate * maturity) * norm.cdf(d2)

    return float(rho)


def european_put_rho(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Calculate the Black-Scholes Rho of a European put."""

    _, d2 = _calculate_d1_d2(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    rho = -strike * maturity * exp(-rate * maturity) * norm.cdf(-d2)

    return float(rho)
