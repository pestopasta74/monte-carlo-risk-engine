from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantmc.pricing.black_scholes import european_call_price
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
    analytical_price = european_call_price(
        spot=SPOT,
        strike=STRIKE,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
    )

    estimates = []
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

        estimates.append(result.price)
        standard_errors.append(result.standard_error)

        print(
            f"N={simulations:>8,} | "
            f"MC={result.price:.4f} | "
            f"SE={result.standard_error:.4f} | "
            f"error={abs(result.price - analytical_price):.4f}"
        )

    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        SIMULATION_COUNTS,
        estimates,
        yerr=np.array(standard_errors) * 1.96,
        marker="o",
        capsize=4,
        label="Monte Carlo estimate (95% CI)",
    )

    plt.axhline(
        analytical_price,
        linestyle="--",
        label=f"Black-Scholes ({analytical_price:.4f})",
    )

    plt.xscale("log")
    plt.xlabel("Number of simulations")
    plt.ylabel("European call price")
    plt.title("Monte Carlo Convergence to Black-Scholes Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "monte_carlo_convergence.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":
    main()
