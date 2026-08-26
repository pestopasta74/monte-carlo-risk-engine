from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from quantmc.models.geometric_brownian_motion import simulate_terminal_prices

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloGreekResult:
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]


def _calculate_pathwise_delta_samples(
    terminal_prices: NDArray[np.float64],
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: OptionType,
) -> NDArray[np.float64]:
    """Calculate discounted pathwise Delta samples."""

    discount_factor = np.exp(-rate * maturity)
    terminal_sensitivity = terminal_prices / spot

    if option_type == "call":
        payoff_derivative = np.where(
            terminal_prices > strike,
            1.0,
            0.0,
        )
    else:
        payoff_derivative = np.where(
            terminal_prices < strike,
            -1.0,
            0.0,
        )

    return discount_factor * payoff_derivative * terminal_sensitivity


def _summarize_greek_samples(
    samples: NDArray[np.float64],
) -> MonteCarloGreekResult:
    """Summarize simulated Greek samples."""

    estimate = np.mean(samples)
    standard_error = np.std(samples, ddof=1) / np.sqrt(samples.size)
    margin = 1.96 * standard_error

    return MonteCarloGreekResult(
        estimate=float(estimate),
        standard_error=float(standard_error),
        confidence_interval=(
            float(estimate - margin),
            float(estimate + margin),
        ),
    )


def _european_delta_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int,
    seed: int | None,
    option_type: OptionType,
) -> MonteCarloGreekResult:
    """Run the shared pathwise Monte Carlo Delta workflow."""

    terminal_prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
    )

    samples = _calculate_pathwise_delta_samples(
        terminal_prices=terminal_prices,
        spot=spot,
        strike=strike,
        rate=rate,
        maturity=maturity,
        option_type=option_type,
    )

    return _summarize_greek_samples(samples)


def european_call_delta_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloGreekResult:
    """Estimate European call Delta using pathwise Monte Carlo."""

    return _european_delta_mc_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type="call",
    )


def european_put_delta_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloGreekResult:
    """Estimate European put Delta using pathwise Monte Carlo."""

    return _european_delta_mc_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type="put",
    )
