from dataclasses import dataclass
from typing import Literal

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
from quantmc.pricing.monte_carlo import (
    MonteCarloResult,
    european_call_price_mc_antithetic_stats,
    european_call_price_mc_stats,
    european_put_price_mc_antithetic_stats,
    european_put_price_mc_stats,
)
from quantmc.pricing.monte_carlo_greeks import (
    MonteCarloGreekResult,
    european_call_delta_mc_stats,
    european_call_gamma_mc_stats,
    european_call_vega_mc_stats,
    european_put_delta_mc_stats,
    european_put_gamma_mc_stats,
    european_put_vega_mc_stats,
)

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class EuropeanOptionParameters:
    """Parameters shared by European option pricing methods."""

    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float


@dataclass(frozen=True)
class OptionGreeks:
    """Analytical Black-Scholes option sensitivities."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class GreekEstimateComparison:
    """Comparison of an analytical Greek and its Monte Carlo estimate."""

    analytical_value: float
    monte_carlo: MonteCarloGreekResult

    @property
    def absolute_error(self) -> float:
        """Return the absolute Monte Carlo estimation error."""

        return abs(self.monte_carlo.estimate - self.analytical_value)

    @property
    def contains_analytical_value(self) -> bool:
        """Return whether the interval contains the analytical value."""

        lower, upper = self.monte_carlo.confidence_interval

        return lower <= self.analytical_value <= upper


@dataclass(frozen=True)
class OptionGreekComparison:
    """Monte Carlo comparisons for Delta, Gamma, and Vega."""

    delta: GreekEstimateComparison
    gamma: GreekEstimateComparison
    vega: GreekEstimateComparison


@dataclass(frozen=True)
class PricingComparison:
    """Comparison of analytical and Monte Carlo option prices."""

    analytical_price: float
    standard_mc: MonteCarloResult
    antithetic_mc: MonteCarloResult

    @property
    def standard_absolute_error(self) -> float:
        """Return the standard Monte Carlo absolute pricing error."""

        return abs(self.standard_mc.price - self.analytical_price)

    @property
    def antithetic_absolute_error(self) -> float:
        """Return the antithetic Monte Carlo absolute pricing error."""

        return abs(self.antithetic_mc.price - self.analytical_price)

    @property
    def standard_error_reduction_factor(self) -> float:
        """Return the standard-to-antithetic error ratio."""

        return self.standard_mc.standard_error / self.antithetic_mc.standard_error

    @property
    def variance_reduction_factor(self) -> float:
        """Return the standard-to-antithetic variance ratio."""

        return self.standard_error_reduction_factor**2


@dataclass(frozen=True)
class EuropeanOptionAnalysis:
    """Combined pricing and sensitivity analysis."""

    parameters: EuropeanOptionParameters
    call_pricing: PricingComparison
    put_pricing: PricingComparison
    call_greeks: OptionGreeks
    put_greeks: OptionGreeks
    call_greek_comparison: OptionGreekComparison
    put_greek_comparison: OptionGreekComparison


def _compare_pricing_methods(
    parameters: EuropeanOptionParameters,
    simulations: int,
    seed: int | None,
    option_type: OptionType,
) -> PricingComparison:
    """Compare analytical, standard MC, and antithetic pricing."""

    antithetic_pairs = simulations // 2

    if option_type == "call":
        analytical_price = european_call_price(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        )
        standard_mc = european_call_price_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )
        antithetic_mc = european_call_price_mc_antithetic_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            pairs=antithetic_pairs,
            seed=seed,
        )
    else:
        analytical_price = european_put_price(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        )
        standard_mc = european_put_price_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )
        antithetic_mc = european_put_price_mc_antithetic_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            pairs=antithetic_pairs,
            seed=seed,
        )

    return PricingComparison(
        analytical_price=analytical_price,
        standard_mc=standard_mc,
        antithetic_mc=antithetic_mc,
    )


def _compare_greek_estimators(
    parameters: EuropeanOptionParameters,
    analytical_greeks: OptionGreeks,
    simulations: int,
    seed: int | None,
    gamma_bump_size: float,
    option_type: OptionType,
) -> OptionGreekComparison:
    """Compare analytical Greeks with Monte Carlo estimators."""

    if option_type == "call":
        delta_mc = european_call_delta_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )
        gamma_mc = european_call_gamma_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
            bump_size=gamma_bump_size,
        )
        vega_mc = european_call_vega_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )
    else:
        delta_mc = european_put_delta_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )
        gamma_mc = european_put_gamma_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )
        vega_mc = european_put_vega_mc_stats(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
            simulations=simulations,
            seed=seed,
        )

    return OptionGreekComparison(
        delta=GreekEstimateComparison(
            analytical_value=analytical_greeks.delta,
            monte_carlo=delta_mc,
        ),
        gamma=GreekEstimateComparison(
            analytical_value=analytical_greeks.gamma,
            monte_carlo=gamma_mc,
        ),
        vega=GreekEstimateComparison(
            analytical_value=analytical_greeks.vega,
            monte_carlo=vega_mc,
        ),
    )


def analyze_european_option(
    parameters: EuropeanOptionParameters,
    simulations: int = 100_000,
    seed: int | None = None,
    gamma_bump_size: float = 1.0,
) -> EuropeanOptionAnalysis:
    """Run a complete European call and put option analysis."""

    if simulations < 2:
        raise ValueError("simulations must be at least 2")

    if simulations % 2 != 0:
        raise ValueError(
            "simulations must be even for an equal-budget antithetic comparison"
        )

    call_pricing = _compare_pricing_methods(
        parameters=parameters,
        simulations=simulations,
        seed=seed,
        option_type="call",
    )
    put_pricing = _compare_pricing_methods(
        parameters=parameters,
        simulations=simulations,
        seed=seed,
        option_type="put",
    )

    gamma = european_option_gamma(
        spot=parameters.spot,
        strike=parameters.strike,
        rate=parameters.rate,
        volatility=parameters.volatility,
        maturity=parameters.maturity,
    )
    vega = european_option_vega(
        spot=parameters.spot,
        strike=parameters.strike,
        rate=parameters.rate,
        volatility=parameters.volatility,
        maturity=parameters.maturity,
    )

    call_greeks = OptionGreeks(
        delta=european_call_delta(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        ),
        gamma=gamma,
        vega=vega,
        theta=european_call_theta(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        ),
        rho=european_call_rho(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        ),
    )

    put_greeks = OptionGreeks(
        delta=european_put_delta(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        ),
        gamma=gamma,
        vega=vega,
        theta=european_put_theta(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        ),
        rho=european_put_rho(
            spot=parameters.spot,
            strike=parameters.strike,
            rate=parameters.rate,
            volatility=parameters.volatility,
            maturity=parameters.maturity,
        ),
    )

    call_greek_comparison = _compare_greek_estimators(
        parameters=parameters,
        analytical_greeks=call_greeks,
        simulations=simulations,
        seed=seed,
        gamma_bump_size=gamma_bump_size,
        option_type="call",
    )

    put_greek_comparison = _compare_greek_estimators(
        parameters=parameters,
        analytical_greeks=put_greeks,
        simulations=simulations,
        seed=seed,
        gamma_bump_size=gamma_bump_size,
        option_type="put",
    )

    return EuropeanOptionAnalysis(
        parameters=parameters,
        call_pricing=call_pricing,
        put_pricing=put_pricing,
        call_greeks=call_greeks,
        put_greeks=put_greeks,
        call_greek_comparison=call_greek_comparison,
        put_greek_comparison=put_greek_comparison,
    )
