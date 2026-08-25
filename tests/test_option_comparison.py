from dataclasses import FrozenInstanceError

import pytest

from quantmc.analysis import (
    EuropeanOptionAnalysis,
    EuropeanOptionParameters,
    OptionGreeks,
    PricingComparison,
    analyze_european_option,
)
from quantmc.pricing.monte_carlo import MonteCarloResult


def test_european_option_parameters_store_inputs() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=105.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    assert parameters.spot == 100.0
    assert parameters.strike == 105.0
    assert parameters.rate == 0.05
    assert parameters.volatility == 0.2
    assert parameters.maturity == 1.0


def test_european_option_parameters_are_immutable() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        parameters.spot = 110.0


def test_pricing_comparison_calculates_errors() -> None:
    standard = MonteCarloResult(
        price=10.5,
        standard_error=0.2,
        confidence_interval=(10.1, 10.9),
    )
    antithetic = MonteCarloResult(
        price=10.4,
        standard_error=0.1,
        confidence_interval=(10.2, 10.6),
    )

    comparison = PricingComparison(
        analytical_price=10.45,
        standard_mc=standard,
        antithetic_mc=antithetic,
    )

    assert comparison.standard_absolute_error == pytest.approx(0.05)
    assert comparison.antithetic_absolute_error == pytest.approx(0.05)
    assert comparison.standard_error_reduction_factor == pytest.approx(2.0)
    assert comparison.variance_reduction_factor == pytest.approx(4.0)


def test_analyze_european_option_returns_complete_analysis() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    analysis = analyze_european_option(
        parameters=parameters,
        simulations=20_000,
        seed=42,
    )

    assert isinstance(analysis, EuropeanOptionAnalysis)
    assert analysis.parameters == parameters
    assert isinstance(analysis.call_pricing, PricingComparison)
    assert isinstance(analysis.put_pricing, PricingComparison)
    assert isinstance(analysis.call_greeks, OptionGreeks)
    assert isinstance(analysis.put_greeks, OptionGreeks)


def test_analysis_contains_expected_analytical_values() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    analysis = analyze_european_option(
        parameters=parameters,
        simulations=20_000,
        seed=42,
    )

    assert analysis.call_pricing.analytical_price == pytest.approx(
        10.4506,
        abs=1e-4,
    )
    assert analysis.put_pricing.analytical_price == pytest.approx(
        5.5735,
        abs=1e-4,
    )
    assert analysis.call_greeks.delta == pytest.approx(
        0.6368,
        abs=1e-4,
    )
    assert analysis.put_greeks.delta == pytest.approx(
        -0.3632,
        abs=1e-4,
    )


def test_analysis_call_and_put_share_gamma_and_vega() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    analysis = analyze_european_option(
        parameters=parameters,
        simulations=20_000,
        seed=42,
    )

    assert analysis.call_greeks.gamma == pytest.approx(
        analysis.put_greeks.gamma,
        abs=1e-12,
    )
    assert analysis.call_greeks.vega == pytest.approx(
        analysis.put_greeks.vega,
        abs=1e-12,
    )


@pytest.mark.parametrize("simulations", [0, 1])
def test_analysis_rejects_too_few_simulations(
    simulations: int,
) -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    with pytest.raises(
        ValueError,
        match="simulations must be at least 2",
    ):
        analyze_european_option(
            parameters=parameters,
            simulations=simulations,
            seed=42,
        )


def test_analysis_rejects_odd_simulation_count() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    with pytest.raises(
        ValueError,
        match="simulations must be even",
    ):
        analyze_european_option(
            parameters=parameters,
            simulations=9_999,
            seed=42,
        )
