from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from quantmc.models.geometric_brownian_motion import (
    simulate_antithetic_terminal_price_pairs,
    simulate_terminal_prices,
)

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]


def _calculate_payoffs(
    terminal_prices: NDArray[np.float64],
    strike: float,
    option_type: OptionType,
) -> NDArray[np.float64]:
    """Calculate European option payoffs."""

    if option_type == "call":
        return np.maximum(terminal_prices - strike, 0.0)

    return np.maximum(strike - terminal_prices, 0.0)


def _summarize_discounted_payoffs(
    payoffs: NDArray[np.float64],
    rate: float,
    maturity: float,
) -> MonteCarloResult:
    """Calculate a discounted price and sampling statistics."""

    discount_factor = np.exp(-rate * maturity)

    price = discount_factor * np.mean(payoffs)
    standard_error = discount_factor * np.std(payoffs, ddof=1) / np.sqrt(payoffs.size)

    margin = 1.96 * standard_error

    return MonteCarloResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval=(
            float(price - margin),
            float(price + margin),
        ),
    )


def _simulate_standard_payoffs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int,
    seed: int | None,
    option_type: OptionType,
) -> NDArray[np.float64]:
    """Simulate European option payoffs using standard Monte Carlo."""

    terminal_prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
    )

    return _calculate_payoffs(
        terminal_prices=terminal_prices,
        strike=strike,
        option_type=option_type,
    )


def _european_price_mc(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int,
    seed: int | None,
    option_type: OptionType,
) -> float:
    """Run the shared point-price Monte Carlo workflow."""

    payoffs = _simulate_standard_payoffs(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type=option_type,
    )

    discount_factor = np.exp(-rate * maturity)

    return float(discount_factor * np.mean(payoffs))


def _european_price_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int,
    seed: int | None,
    option_type: OptionType,
) -> MonteCarloResult:
    """Run the shared standard Monte Carlo statistics workflow."""

    payoffs = _simulate_standard_payoffs(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type=option_type,
    )

    return _summarize_discounted_payoffs(
        payoffs=payoffs,
        rate=rate,
        maturity=maturity,
    )


def _european_price_mc_antithetic_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    pairs: int,
    seed: int | None,
    option_type: OptionType,
) -> MonteCarloResult:
    """Run the shared antithetic Monte Carlo statistics workflow."""

    positive_prices, negative_prices = simulate_antithetic_terminal_price_pairs(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        pairs=pairs,
        seed=seed,
    )

    positive_payoffs = _calculate_payoffs(
        terminal_prices=positive_prices,
        strike=strike,
        option_type=option_type,
    )
    negative_payoffs = _calculate_payoffs(
        terminal_prices=negative_prices,
        strike=strike,
        option_type=option_type,
    )

    paired_payoffs = (positive_payoffs + negative_payoffs) / 2.0

    return _summarize_discounted_payoffs(
        payoffs=paired_payoffs,
        rate=rate,
        maturity=maturity,
    )


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

    return _european_price_mc(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type="call",
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

    return _european_price_mc(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type="put",
    )


def european_call_price_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European call and return Monte Carlo statistics."""

    return _european_price_mc_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type="call",
    )


def european_put_price_mc_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European put and return Monte Carlo statistics."""

    return _european_price_mc_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=simulations,
        seed=seed,
        option_type="put",
    )


def european_call_price_mc_antithetic_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    pairs: int = 50_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European call using antithetic variates."""

    return _european_price_mc_antithetic_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        pairs=pairs,
        seed=seed,
        option_type="call",
    )


def european_put_price_mc_antithetic_stats(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    pairs: int = 50_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European put using antithetic variates."""

    return _european_price_mc_antithetic_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        pairs=pairs,
        seed=seed,
        option_type="put",
    )
