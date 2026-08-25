from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantmc.analysis import (
    EuropeanOptionParameters,
    analyze_european_option,
)

SIMULATION_COUNTS = np.array(
    [
        2_000,
        6_000,
        20_000,
        60_000,
        200_000,
        600_000,
    ]
)

PARAMETERS = EuropeanOptionParameters(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    volatility=0.2,
    maturity=1.0,
)

SEED = 42


def main() -> None:
    call_standard_prices = []
    call_antithetic_prices = []
    put_standard_prices = []
    put_antithetic_prices = []

    call_standard_errors = []
    call_antithetic_errors = []
    put_standard_errors = []
    put_antithetic_errors = []

    call_analytical_price = 0.0
    put_analytical_price = 0.0

    print("Variance Reduction Analysis")
    print("=" * 100)
    print(
        f"{'Budget':>10}"
        f"{'Option':>10}"
        f"{'Standard SE':>16}"
        f"{'Antithetic SE':>16}"
        f"{'SE ratio':>12}"
        f"{'Variance ratio':>18}"
    )
    print("-" * 100)

    for simulations in SIMULATION_COUNTS:
        analysis = analyze_european_option(
            parameters=PARAMETERS,
            simulations=int(simulations),
            seed=SEED,
        )

        call = analysis.call_pricing
        put = analysis.put_pricing

        call_analytical_price = call.analytical_price
        put_analytical_price = put.analytical_price

        call_standard_prices.append(call.standard_mc.price)
        call_antithetic_prices.append(call.antithetic_mc.price)
        put_standard_prices.append(put.standard_mc.price)
        put_antithetic_prices.append(put.antithetic_mc.price)

        call_standard_errors.append(call.standard_mc.standard_error)
        call_antithetic_errors.append(call.antithetic_mc.standard_error)
        put_standard_errors.append(put.standard_mc.standard_error)
        put_antithetic_errors.append(put.antithetic_mc.standard_error)

        print(
            f"{simulations:>10,}"
            f"{'Call':>10}"
            f"{call.standard_mc.standard_error:>16.6f}"
            f"{call.antithetic_mc.standard_error:>16.6f}"
            f"{call.standard_error_reduction_factor:>12.3f}"
            f"{call.variance_reduction_factor:>18.3f}"
        )
        print(
            f"{'':>10}"
            f"{'Put':>10}"
            f"{put.standard_mc.standard_error:>16.6f}"
            f"{put.antithetic_mc.standard_error:>16.6f}"
            f"{put.standard_error_reduction_factor:>12.3f}"
            f"{put.variance_reduction_factor:>18.3f}"
        )

    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    create_convergence_figure(
        call_analytical_price=call_analytical_price,
        put_analytical_price=put_analytical_price,
        call_standard_prices=call_standard_prices,
        call_antithetic_prices=call_antithetic_prices,
        put_standard_prices=put_standard_prices,
        put_antithetic_prices=put_antithetic_prices,
        output_dir=output_dir,
    )

    create_standard_error_figure(
        call_standard_errors=call_standard_errors,
        call_antithetic_errors=call_antithetic_errors,
        put_standard_errors=put_standard_errors,
        put_antithetic_errors=put_antithetic_errors,
        output_dir=output_dir,
    )


def create_convergence_figure(
    call_analytical_price: float,
    put_analytical_price: float,
    call_standard_prices: list[float],
    call_antithetic_prices: list[float],
    put_standard_prices: list[float],
    put_antithetic_prices: list[float],
    output_dir: Path,
) -> None:
    """Plot standard and antithetic price convergence."""

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        sharex=True,
    )

    plot_price_convergence(
        axis=axes[0],
        analytical_price=call_analytical_price,
        standard_prices=call_standard_prices,
        antithetic_prices=call_antithetic_prices,
        title="European Call",
    )
    plot_price_convergence(
        axis=axes[1],
        analytical_price=put_analytical_price,
        standard_prices=put_standard_prices,
        antithetic_prices=put_antithetic_prices,
        title="European Put",
    )

    figure.suptitle("Standard and Antithetic Monte Carlo Convergence")
    figure.tight_layout()

    figure.savefig(
        output_dir / "variance_reduction_convergence.png",
        dpi=200,
    )

    plt.close(figure)


def plot_price_convergence(
    axis: plt.Axes,
    analytical_price: float,
    standard_prices: list[float],
    antithetic_prices: list[float],
    title: str,
) -> None:
    """Plot convergence for one option type."""

    axis.plot(
        SIMULATION_COUNTS,
        standard_prices,
        marker="o",
        label="Standard MC",
    )
    axis.plot(
        SIMULATION_COUNTS,
        antithetic_prices,
        marker="s",
        label="Antithetic MC",
    )
    axis.axhline(
        analytical_price,
        color="black",
        linestyle="--",
        label="Black-Scholes",
    )

    axis.set_xscale("log")
    axis.set_xlabel("Terminal-price evaluations")
    axis.set_ylabel("Option price")
    axis.set_title(title)
    axis.legend()


def create_standard_error_figure(
    call_standard_errors: list[float],
    call_antithetic_errors: list[float],
    put_standard_errors: list[float],
    put_antithetic_errors: list[float],
    output_dir: Path,
) -> None:
    """Plot standard errors under equal simulation budgets."""

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.loglog(
        SIMULATION_COUNTS,
        call_standard_errors,
        marker="o",
        label="Call — standard",
    )
    axis.loglog(
        SIMULATION_COUNTS,
        call_antithetic_errors,
        marker="s",
        label="Call — antithetic",
    )
    axis.loglog(
        SIMULATION_COUNTS,
        put_standard_errors,
        marker="o",
        label="Put — standard",
    )
    axis.loglog(
        SIMULATION_COUNTS,
        put_antithetic_errors,
        marker="s",
        label="Put — antithetic",
    )

    axis.set_xlabel("Terminal-price evaluations")
    axis.set_ylabel("Estimator standard error")
    axis.set_title("Standard Error: Standard vs Antithetic Monte Carlo")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir / "variance_reduction_standard_error.png",
        dpi=200,
    )

    plt.close(figure)


if __name__ == "__main__":
    main()
