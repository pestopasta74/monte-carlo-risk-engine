import numpy as np
from numpy.typing import NDArray


def simulate_terminal_prices(
    spot: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate terminal asset prices under geometric Brownian motion."""

    rng = np.random.default_rng(seed)

    z = rng.standard_normal(simulations)

    drift = (rate - 0.5 * volatility**2) * maturity
    diffusion = volatility * np.sqrt(maturity) * z

    terminal_prices = spot * np.exp(drift + diffusion)

    return terminal_prices


def simulate_antithetic_terminal_price_pairs(
    spot: float,
    rate: float,
    volatility: float,
    maturity: float,
    pairs: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate antithetic pairs of terminal asset prices under GBM."""

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(pairs)

    drift = (rate - 0.5 * volatility**2) * maturity
    diffusion = volatility * np.sqrt(maturity) * z

    positive_prices = spot * np.exp(drift + diffusion)
    negative_prices = spot * np.exp(drift - diffusion)

    return positive_prices, negative_prices
