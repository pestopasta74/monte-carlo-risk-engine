import numpy as np
import pytest

from quantmc.pricing.black_scholes import (
    european_call_price,
    european_put_price,
)


def test_at_the_money_call_price() -> None:
    price = european_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert price == pytest.approx(10.4506, abs=1e-4)


def test_call_price_is_positive() -> None:
    price = european_call_price(
        spot=100.0,
        strike=110.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert price > 0.0


def test_call_price_increases_with_spot() -> None:
    low_spot_price = european_call_price(
        90.0,
        100.0,
        0.05,
        0.2,
        1.0,
    )
    high_spot_price = european_call_price(
        110.0,
        100.0,
        0.05,
        0.2,
        1.0,
    )

    assert high_spot_price > low_spot_price


def test_at_the_money_put_price() -> None:
    price = european_put_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert price == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0

    call_price = european_call_price(
        spot,
        strike,
        rate,
        volatility,
        maturity,
    )
    put_price = european_put_price(
        spot,
        strike,
        rate,
        volatility,
        maturity,
    )

    discounted_strike = strike * np.exp(-rate * maturity)

    assert call_price - put_price == pytest.approx(
        spot - discounted_strike,
        abs=1e-10,
    )
