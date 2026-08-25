from dataclasses import FrozenInstanceError

import pytest

from quantmc.analysis.option_comparison import (
    EuropeanOptionParameters,
)


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
