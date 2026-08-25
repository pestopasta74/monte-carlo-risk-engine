import matplotlib.pyplot as plt
import numpy as np

from quantmc.pricing.monte_carlo import european_call_price_mc_stats

SIMULATION_COUNTS = np.array(
    [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
)

SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOLATILITY = 0.2
MATURITY = 1.0
SEED = 42


def main() -> None:
    standard_errors = []

    for simulations in SIMULATION_COUNTS:
        result = european_call_price_mc_stats(
            spot=SPOT,
            strike=STRIKE,
            rate=RATE,
            volatility=VOLATILITY,
            maturity=MATURITY,
            simulations=int(simulations),
            seed=SEED,
        )

        standard_errors.append(result.standard_error)

        print(
            f"N={simulations:>8,} | "
            f"MC={result.price:.4f} | "
            f"SE={result.standard_error:.4f}"
        )

    standard_errors = np.array(standard_errors)

    reference = standard_errors[0] * np.sqrt(SIMULATION_COUNTS[0] / SIMULATION_COUNTS)

    plt.figure(figsize=(9, 5))

    plt.loglog(
        SIMULATION_COUNTS,
        standard_errors,
        marker="o",
        label="Measured standard error",
    )

    plt.loglog(
        SIMULATION_COUNTS,
        reference,
        linestyle="--",
        label=r"$N^{-1/2}$ reference",
    )

    plt.xlabel("Number of simulations")
    plt.ylabel("Standard error")
    plt.title("Monte Carlo Standard Error Scaling")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "docs/figures/standard_error_scaling.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":
    main()
