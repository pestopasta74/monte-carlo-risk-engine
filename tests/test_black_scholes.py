import pytest

from quantmc.pricing.black_scholes import european_call_price


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
    low_spot_price = european_call_price(90, 100, 0.05, 0.2, 1)
    high_spot_price = european_call_price(110, 100, 0.05, 0.2, 1)

    assert high_spot_price > low_spot_price
