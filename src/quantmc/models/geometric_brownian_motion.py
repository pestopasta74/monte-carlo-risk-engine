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
