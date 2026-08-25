import numpy as np
import pytest

from quantmc.pricing.black_scholes import (
    european_call_price,
    european_put_price,
)
from quantmc.pricing.monte_carlo import (
    european_call_price_mc,
    european_call_price_mc_stats,
    european_put_price_mc,
    european_put_price_mc_stats,
)


def test_mc_call_matches_black_scholes() -> None:
    analytical = european_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    result = european_call_price_mc_stats(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=500_000,
        seed=42,
    )

    lower, upper = result.confidence_interval

    assert lower < analytical < upper


def test_mc_put_matches_black_scholes() -> None:
    analytical = european_put_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    result = european_put_price_mc_stats(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=500_000,
        seed=42,
    )

    lower, upper = result.confidence_interval

    assert lower < analytical < upper


def test_mc_put_call_parity() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0

    call = european_call_price_mc_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=500_000,
        seed=42,
    )

    put = european_put_price_mc_stats(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        simulations=500_000,
        seed=42,
    )

    theoretical_difference = spot - strike * np.exp(-rate * maturity)

    assert call.price - put.price == pytest.approx(
        theoretical_difference,
        abs=0.1,
    )


def test_point_call_pricer_returns_positive_price() -> None:
    price = european_call_price_mc(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=100_000,
        seed=42,
    )

    assert price > 0.0


def test_point_put_pricer_returns_positive_price() -> None:
    price = european_put_price_mc(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=100_000,
        seed=42,
    )

    assert price > 0.0
