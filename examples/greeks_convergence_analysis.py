import csv
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

GAMMA_BUMP_SIZE = 1.0
SEED = 42

OPTION_NAMES = ("call", "put")
GREEK_NAMES = ("delta", "gamma", "vega")


def create_empty_results() -> dict[str, list[float]]:
    """Create empty result lists for every option and Greek."""

    return {
        f"{option_name}_{greek_name}": []
        for option_name in OPTION_NAMES
        for greek_name in GREEK_NAMES
    }


def main() -> None:
    estimates = create_empty_results()
    standard_errors = create_empty_results()
    analytical_values: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    print("Monte Carlo Greeks Convergence Analysis")
    print("=" * 92)
    print(
        f"{'Simulations':>12}"
        f"{'Option':>10}"
        f"{'Greek':>10}"
        f"{'Analytical':>16}"
        f"{'MC estimate':>16}"
        f"{'Std. error':>16}"
    )
    print("-" * 92)

    for simulations in SIMULATION_COUNTS:
        analysis = analyze_european_option(
            parameters=PARAMETERS,
            simulations=int(simulations),
            seed=SEED,
            gamma_bump_size=GAMMA_BUMP_SIZE,
        )

        option_comparisons = (
            ("call", analysis.call_greek_comparison),
            ("put", analysis.put_greek_comparison),
        )

        for option_name, comparison in option_comparisons:
            greek_comparisons = (
                ("delta", comparison.delta),
                ("gamma", comparison.gamma),
                ("vega", comparison.vega),
            )

            for greek_name, greek_comparison in greek_comparisons:
                key = f"{option_name}_{greek_name}"

                analytical_values[key] = greek_comparison.analytical_value
                estimates[key].append(greek_comparison.monte_carlo.estimate)
                standard_errors[key].append(greek_comparison.monte_carlo.standard_error)

                lower, upper = greek_comparison.monte_carlo.confidence_interval

                rows.append(
                    {
                        "simulations": int(simulations),
                        "option_type": option_name,
                        "greek": greek_name,
                        "analytical_value": (greek_comparison.analytical_value),
                        "monte_carlo_estimate": (greek_comparison.monte_carlo.estimate),
                        "absolute_error": (greek_comparison.absolute_error),
                        "standard_error": (greek_comparison.monte_carlo.standard_error),
                        "confidence_interval_lower": lower,
                        "confidence_interval_upper": upper,
                        "contains_analytical_value": (
                            greek_comparison.contains_analytical_value
                        ),
                    }
                )

                print(
                    f"{simulations:>12,}"
                    f"{option_name.capitalize():>10}"
                    f"{greek_name.capitalize():>10}"
                    f"{greek_comparison.analytical_value:>16.6f}"
                    f"{greek_comparison.monte_carlo.estimate:>16.6f}"
                    f"{greek_comparison.monte_carlo.standard_error:>16.6f}"
                )

    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    create_convergence_figure(
        estimates=estimates,
        standard_errors=standard_errors,
        analytical_values=analytical_values,
        output_dir=output_dir,
    )

    create_standard_error_figure(
        standard_errors=standard_errors,
        output_dir=output_dir,
    )

    data_dir = Path("docs/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    write_results_csv(
        rows=rows,
        output_dir=data_dir,
    )


def write_results_csv(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Write the convergence results to a reproducible CSV file."""

    output_path = output_dir / "monte_carlo_greeks_convergence.csv"

    fieldnames = [
        "simulations",
        "option_type",
        "greek",
        "analytical_value",
        "monte_carlo_estimate",
        "absolute_error",
        "standard_error",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "contains_analytical_value",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def create_convergence_figure(
    estimates: dict[str, list[float]],
    standard_errors: dict[str, list[float]],
    analytical_values: dict[str, float],
    output_dir: Path,
) -> None:
    """Plot Monte Carlo Greek convergence with confidence intervals."""

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
        sharex=True,
    )

    for row, option_name in enumerate(OPTION_NAMES):
        for column, greek_name in enumerate(GREEK_NAMES):
            key = f"{option_name}_{greek_name}"
            axis = axes[row, column]

            estimates_array = np.asarray(estimates[key])
            errors_array = np.asarray(standard_errors[key])

            axis.errorbar(
                SIMULATION_COUNTS,
                estimates_array,
                yerr=1.96 * errors_array,
                marker="o",
                capsize=3,
                label="Monte Carlo 95% CI",
            )
            axis.axhline(
                analytical_values[key],
                color="black",
                linestyle="--",
                label="Black-Scholes",
            )

            axis.set_xscale("log")
            axis.set_title(f"{option_name.capitalize()} {greek_name.capitalize()}")
            axis.grid(alpha=0.25)

            if row == 1:
                axis.set_xlabel("Simulations")

            if column == 0:
                axis.set_ylabel("Greek estimate")

            axis.legend()

    figure.suptitle("Monte Carlo Greek Estimates Against Black-Scholes")
    figure.tight_layout()

    figure.savefig(
        output_dir / "monte_carlo_greeks_convergence.png",
        dpi=200,
    )

    plt.close(figure)


def create_standard_error_figure(
    standard_errors: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Plot Greek estimator standard errors against simulation count."""

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        sharex=True,
    )

    for column, greek_name in enumerate(GREEK_NAMES):
        axis = axes[column]

        call_errors = np.asarray(standard_errors[f"call_{greek_name}"])
        put_errors = np.asarray(standard_errors[f"put_{greek_name}"])

        reference_errors = call_errors[0] * np.sqrt(
            SIMULATION_COUNTS[0] / SIMULATION_COUNTS
        )

        axis.loglog(
            SIMULATION_COUNTS,
            call_errors,
            marker="o",
            label="Call",
        )
        axis.loglog(
            SIMULATION_COUNTS,
            put_errors,
            marker="s",
            label="Put",
        )
        axis.loglog(
            SIMULATION_COUNTS,
            reference_errors,
            color="black",
            linestyle="--",
            label=r"$N^{-1/2}$ reference",
        )

        axis.set_xlabel("Simulations")
        axis.set_title(f"{greek_name.capitalize()} standard error")
        axis.grid(
            alpha=0.25,
            which="both",
        )
        axis.legend()

        if column == 0:
            axis.set_ylabel("Estimator standard error")

    figure.suptitle("Monte Carlo Greek Estimator Standard Errors")
    figure.tight_layout()

    figure.savefig(
        output_dir / "monte_carlo_greeks_standard_error.png",
        dpi=200,
    )

    plt.close(figure)


if __name__ == "__main__":
    main()
