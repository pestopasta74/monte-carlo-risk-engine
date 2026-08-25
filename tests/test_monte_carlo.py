from quantmc.pricing.monte_carlo import european_call_price_mc_stats


def test_black_scholes_price_inside_mc_confidence_interval() -> None:
    result = european_call_price_mc_stats(
        spot=100,
        strike=100,
        rate=0.05,
        volatility=0.2,
        maturity=1,
        simulations=500_000,
        seed=42,
    )

    analytical_price = 10.4506
    lower, upper = result.confidence_interval

    assert lower < analytical_price < upper
