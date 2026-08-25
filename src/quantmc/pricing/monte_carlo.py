from dataclasses import dataclass

import numpy as np

from quantmc.models.geometric_brownian_motion import simulate_terminal_prices


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]


def european_call_price_mc(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> float:
    """Price a European call option using Monte Carlo simulation."""

    terminal_prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
    )

    payoffs = np.maximum(terminal_prices - strike, 0.0)
    discount_factor = np.exp(-rate * maturity)

    return float(discount_factor * np.mean(payoffs))


def european_call_price_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European call and return Monte Carlo uncertainty statistics."""

    terminal_prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
    )

    payoffs = np.maximum(terminal_prices - strike, 0.0)
    discount_factor = np.exp(-rate * maturity)

    price = discount_factor * np.mean(payoffs)
    standard_error = discount_factor * np.std(payoffs, ddof=1) / np.sqrt(simulations)

    margin = 1.96 * standard_error

    return MonteCarloResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval=(
            float(price - margin),
            float(price + margin),
        ),
    )


def european_put_price_mc(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> float:
    """Price a European put option using Monte Carlo simulation."""

    terminal_prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
    )

    payoffs = np.maximum(strike - terminal_prices, 0.0)
    discount_factor = np.exp(-rate * maturity)

    return float(discount_factor * np.mean(payoffs))


def european_put_price_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European put and return Monte Carlo uncertainty statistics."""

    terminal_prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
    )

    payoffs = np.maximum(strike - terminal_prices, 0.0)
    discount_factor = np.exp(-rate * maturity)

    price = discount_factor * np.mean(payoffs)
    standard_error = discount_factor * np.std(payoffs, ddof=1) / np.sqrt(simulations)

    margin = 1.96 * standard_error

    return MonteCarloResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval=(
            float(price - margin),
            float(price + margin),
        ),
    )
