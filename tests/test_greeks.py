import pytest

from quantmc.pricing.black_scholes import (
    european_call_price,
    european_put_price,
)
from quantmc.pricing.greeks import (
    european_call_delta,
    european_call_rho,
    european_call_theta,
    european_option_gamma,
    european_option_vega,
    european_put_delta,
    european_put_rho,
    european_put_theta,
)


def test_at_the_money_call_delta() -> None:
    delta = european_call_delta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert delta == pytest.approx(0.6368, abs=1e-4)


def test_at_the_money_put_delta() -> None:
    delta = european_put_delta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert delta == pytest.approx(-0.3632, abs=1e-4)


def test_call_and_put_delta_relationship() -> None:
    call_delta = european_call_delta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    put_delta = european_put_delta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert call_delta - put_delta == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_call_delta_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.01

    price_above = european_call_price(
        spot=spot + step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    price_below = european_call_price(
        spot=spot - step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    finite_difference_delta = (price_above - price_below) / (2.0 * step)

    analytical_delta = european_call_delta(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_delta == pytest.approx(
        finite_difference_delta,
        abs=1e-6,
    )


def test_put_delta_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.01

    price_above = european_put_price(
        spot=spot + step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    price_below = european_put_price(
        spot=spot - step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    finite_difference_delta = (price_above - price_below) / (2.0 * step)

    analytical_delta = european_put_delta(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_delta == pytest.approx(
        finite_difference_delta,
        abs=1e-6,
    )


def test_at_the_money_option_gamma() -> None:
    gamma = european_option_gamma(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert gamma == pytest.approx(0.018762, abs=1e-6)


def test_call_gamma_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.1

    price_above = european_call_price(
        spot=spot + step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    price_at_spot = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    price_below = european_call_price(
        spot=spot - step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    finite_difference_gamma = (
        price_above - 2.0 * price_at_spot + price_below
    ) / step**2

    analytical_gamma = european_option_gamma(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_gamma == pytest.approx(
        finite_difference_gamma,
        abs=1e-6,
    )


def test_put_gamma_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.1

    price_above = european_put_price(
        spot=spot + step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    price_at_spot = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    price_below = european_put_price(
        spot=spot - step,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    finite_difference_gamma = (
        price_above - 2.0 * price_at_spot + price_below
    ) / step**2

    analytical_gamma = european_option_gamma(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_gamma == pytest.approx(
        finite_difference_gamma,
        abs=1e-6,
    )


def test_at_the_money_option_vega() -> None:
    vega = european_option_vega(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert vega == pytest.approx(37.5240, abs=1e-4)


def test_call_vega_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.0001

    price_above = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility + step,
        maturity=maturity,
    )
    price_below = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility - step,
        maturity=maturity,
    )

    finite_difference_vega = (price_above - price_below) / (2.0 * step)

    analytical_vega = european_option_vega(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_vega == pytest.approx(
        finite_difference_vega,
        abs=1e-5,
    )


def test_put_vega_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.0001

    price_above = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility + step,
        maturity=maturity,
    )
    price_below = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility - step,
        maturity=maturity,
    )

    finite_difference_vega = (price_above - price_below) / (2.0 * step)

    analytical_vega = european_option_vega(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_vega == pytest.approx(
        finite_difference_vega,
        abs=1e-5,
    )


def test_at_the_money_call_theta() -> None:
    theta = european_call_theta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert theta == pytest.approx(-6.4140, abs=1e-4)


def test_at_the_money_put_theta() -> None:
    theta = european_put_theta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert theta == pytest.approx(-1.6579, abs=1e-4)


def test_call_theta_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.0001

    price_with_more_time = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity + step,
    )
    price_with_less_time = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity - step,
    )

    maturity_derivative = (price_with_more_time - price_with_less_time) / (2.0 * step)

    analytical_theta = european_call_theta(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_theta == pytest.approx(
        -maturity_derivative,
        abs=1e-5,
    )


def test_put_theta_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.0001

    price_with_more_time = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity + step,
    )
    price_with_less_time = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity - step,
    )

    maturity_derivative = (price_with_more_time - price_with_less_time) / (2.0 * step)

    analytical_theta = european_put_theta(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_theta == pytest.approx(
        -maturity_derivative,
        abs=1e-5,
    )


def test_at_the_money_call_rho() -> None:
    rho = european_call_rho(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert rho == pytest.approx(53.2325, abs=1e-4)


def test_at_the_money_put_rho() -> None:
    rho = european_put_rho(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert rho == pytest.approx(-41.8905, abs=1e-4)


def test_call_rho_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.0001

    price_above = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate + step,
        volatility=volatility,
        maturity=maturity,
    )
    price_below = european_call_price(
        spot=spot,
        strike=strike,
        rate=rate - step,
        volatility=volatility,
        maturity=maturity,
    )

    finite_difference_rho = (price_above - price_below) / (2.0 * step)

    analytical_rho = european_call_rho(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_rho == pytest.approx(
        finite_difference_rho,
        abs=1e-5,
    )


def test_put_rho_matches_finite_difference() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.2
    maturity = 1.0
    step = 0.0001

    price_above = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate + step,
        volatility=volatility,
        maturity=maturity,
    )
    price_below = european_put_price(
        spot=spot,
        strike=strike,
        rate=rate - step,
        volatility=volatility,
        maturity=maturity,
    )

    finite_difference_rho = (price_above - price_below) / (2.0 * step)

    analytical_rho = european_put_rho(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert analytical_rho == pytest.approx(
        finite_difference_rho,
        abs=1e-5,
    )
