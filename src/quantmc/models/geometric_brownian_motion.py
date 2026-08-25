import numpy as np
from numpy.typing import NDArray


def _terminal_prices_from_normal_draws(
    spot: float,
    rate: float,
    volatility: float,
    maturity: float,
    normal_draws: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform standard-normal draws into GBM terminal prices."""

    drift = (rate - 0.5 * volatility**2) * maturity
    diffusion = volatility * np.sqrt(maturity) * normal_draws

    return spot * np.exp(drift + diffusion)


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
    normal_draws = rng.standard_normal(simulations)

    return _terminal_prices_from_normal_draws(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        normal_draws=normal_draws,
    )


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
    normal_draws = rng.standard_normal(pairs)

    positive_prices = _terminal_prices_from_normal_draws(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        normal_draws=normal_draws,
    )
    negative_prices = _terminal_prices_from_normal_draws(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        normal_draws=-normal_draws,
    )

    return positive_prices, negative_prices
