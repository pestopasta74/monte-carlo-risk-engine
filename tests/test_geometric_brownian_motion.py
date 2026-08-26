import numpy as np
import pytest

from quantmc.models.geometric_brownian_motion import (
    simulate_antithetic_terminal_price_pairs,
    simulate_terminal_prices,
    simulate_terminal_prices_with_normal_draws,
)


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


def test_number_of_antithetic_pairs() -> None:
    positive_prices, negative_prices = simulate_antithetic_terminal_price_pairs(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        pairs=5_000,
        seed=42,
    )

    assert positive_prices.shape == (5_000,)
    assert negative_prices.shape == (5_000,)


def test_antithetic_terminal_prices_are_positive() -> None:
    positive_prices, negative_prices = simulate_antithetic_terminal_price_pairs(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        pairs=5_000,
        seed=42,
    )

    assert np.all(positive_prices > 0.0)
    assert np.all(negative_prices > 0.0)


def test_antithetic_pair_product_is_deterministic() -> None:
    spot = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0

    positive_prices, negative_prices = simulate_antithetic_terminal_price_pairs(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        pairs=5_000,
        seed=42,
    )

    expected_product = spot**2 * np.exp(2.0 * (rate - 0.5 * volatility**2) * maturity)

    assert positive_prices * negative_prices == pytest.approx(
        expected_product,
        rel=1e-12,
    )


def test_terminal_prices_can_return_normal_draws() -> None:
    prices, normal_draws = simulate_terminal_prices_with_normal_draws(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=10_000,
        seed=42,
    )

    standard_prices = simulate_terminal_prices(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=10_000,
        seed=42,
    )

    assert prices.shape == (10_000,)
    assert normal_draws.shape == (10_000,)
    assert prices == pytest.approx(standard_prices)
