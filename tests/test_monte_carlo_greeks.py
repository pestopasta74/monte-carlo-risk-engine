import pytest

from quantmc.pricing.greeks import (
    european_call_delta,
    european_put_delta,
)
from quantmc.pricing.monte_carlo_greeks import (
    european_call_delta_mc_stats,
    european_put_delta_mc_stats,
)


def test_mc_call_delta_matches_analytical_delta() -> None:
    analytical = european_call_delta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    result = european_call_delta_mc_stats(
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


def test_mc_put_delta_matches_analytical_delta() -> None:
    analytical = european_put_delta(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    result = european_put_delta_mc_stats(
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


def test_mc_delta_put_call_parity() -> None:
    call = european_call_delta_mc_stats(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=500_000,
        seed=42,
    )

    put = european_put_delta_mc_stats(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        simulations=500_000,
        seed=42,
    )

    assert call.estimate - put.estimate == pytest.approx(
        1.0,
        abs=0.01,
    )
