import numpy as np
import pytest

from quantmc.models.geometric_brownian_motion import simulate_terminal_prices


def test_number_of_simulations() -> None:
    prices = simulate_terminal_prices(
        spot=100,
        rate=0.05,
        volatility=0.2,
        maturity=1,
        simulations=10_000,
        seed=42,
    )

    assert prices.shape == (10_000,)


def test_terminal_prices_are_positive() -> None:
    prices = simulate_terminal_prices(
        spot=100,
        rate=0.05,
        volatility=0.2,
        maturity=1,
        simulations=10_000,
        seed=42,
    )

    assert np.all(prices > 0)


def test_expected_terminal_price() -> None:
    spot = 100
    rate = 0.05
    maturity = 1

    prices = simulate_terminal_prices(
        spot=spot,
        rate=rate,
        volatility=0.2,
        maturity=maturity,
        simulations=1_000_000,
        seed=42,
    )

    expected = spot * np.exp(rate * maturity)

    assert np.mean(prices) == pytest.approx(expected, rel=0.01)
