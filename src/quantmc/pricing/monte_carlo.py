from dataclasses import dataclass

import numpy as np

from quantmc.models.geometric_brownian_motion import simulate_terminal_prices


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]


def european_call_price_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
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
