from dataclasses import dataclass


@dataclass(frozen=True)
class EuropeanOptionParameters:
    """Parameters shared by European option pricing methods."""

    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float
