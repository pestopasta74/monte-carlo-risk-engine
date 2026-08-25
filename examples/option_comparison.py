from quantmc.analysis import (
    EuropeanOptionParameters,
    OptionGreeks,
    PricingComparison,
    analyze_european_option,
)


def print_pricing_comparison(
    option_name: str,
    comparison: PricingComparison,
) -> None:
    """Print analytical and Monte Carlo pricing results."""

    print(f"\n{option_name} pricing")
    print("-" * 72)
    print(f"{'Method':<20}{'Price':>12}{'Abs. error':>14}{'Std. error':>14}")
    print("-" * 72)

    print(
        f"{'Black-Scholes':<20}{comparison.analytical_price:>12.6f}{'-':>14}{'-':>14}"
    )
    print(
        f"{'Standard MC':<20}"
        f"{comparison.standard_mc.price:>12.6f}"
        f"{comparison.standard_absolute_error:>14.6f}"
        f"{comparison.standard_mc.standard_error:>14.6f}"
    )
    print(
        f"{'Antithetic MC':<20}"
        f"{comparison.antithetic_mc.price:>12.6f}"
        f"{comparison.antithetic_absolute_error:>14.6f}"
        f"{comparison.antithetic_mc.standard_error:>14.6f}"
    )

    print(
        "\nStandard-error reduction factor: "
        f"{comparison.standard_error_reduction_factor:.3f}x"
    )


def print_greeks(
    option_name: str,
    greeks: OptionGreeks,
) -> None:
    """Print analytical Black-Scholes Greeks."""

    print(f"\n{option_name} Greeks")
    print("-" * 30)
    print(f"{'Delta':<12}{greeks.delta:>12.6f}")
    print(f"{'Gamma':<12}{greeks.gamma:>12.6f}")
    print(f"{'Vega':<12}{greeks.vega:>12.6f}")
    print(f"{'Theta':<12}{greeks.theta:>12.6f}")
    print(f"{'Rho':<12}{greeks.rho:>12.6f}")


def main() -> None:
    parameters = EuropeanOptionParameters(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    simulations = 100_000
    seed = 42

    analysis = analyze_european_option(
        parameters=parameters,
        simulations=simulations,
        seed=seed,
    )

    print("European Option Analysis")
    print("=" * 72)
    print(f"Spot:             {parameters.spot:.2f}")
    print(f"Strike:           {parameters.strike:.2f}")
    print(f"Risk-free rate:   {parameters.rate:.2%}")
    print(f"Volatility:       {parameters.volatility:.2%}")
    print(f"Maturity:         {parameters.maturity:.2f} years")
    print(f"Simulation budget:{simulations:>12,}")
    print(f"Random seed:      {seed:>12}")

    print_pricing_comparison(
        option_name="Call",
        comparison=analysis.call_pricing,
    )
    print_pricing_comparison(
        option_name="Put",
        comparison=analysis.put_pricing,
    )

    print_greeks(
        option_name="Call",
        greeks=analysis.call_greeks,
    )
    print_greeks(
        option_name="Put",
        greeks=analysis.put_greeks,
    )


if __name__ == "__main__":
    main()
