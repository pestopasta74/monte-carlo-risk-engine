# Monte Carlo Option Pricing & Risk Engine

A numerical finance project exploring derivative pricing through Monte Carlo simulation, analytical benchmarking, and statistical convergence analysis.

Built as part of my exploration of **quantitative finance, numerical methods, and scientific computing** as a Theoretical Physics student at the University of Bath.

## Overview

This project implements a Monte Carlo framework for pricing European options under geometric Brownian motion.

Rather than treating Monte Carlo pricing as a black box, the implementation is validated against the analytical Black–Scholes solution and used to investigate the statistical behaviour of the estimator.

The current version includes:

- Black–Scholes analytical pricing for European call options
- Risk-neutral geometric Brownian motion simulation
- Monte Carlo European call pricing
- Standard-error estimation
- 95% confidence intervals
- Analytical validation against Black–Scholes
- Monte Carlo convergence analysis
- Empirical verification of the expected \(N^{-1/2}\) standard-error scaling
- Automated tests and code-quality checks

## Mathematical Model

Under the risk-neutral measure, the underlying asset follows geometric Brownian motion:

$$
dS_t = rS_t\,dt + \sigma S_t\,dW_t.
$$

The terminal asset price therefore has the exact solution

$$
S_T =
S_0
\exp\left[
\left(r-\frac{\sigma^2}{2}\right)T
+
\sigma\sqrt{T}Z
\right],
\qquad Z\sim\mathcal{N}(0,1).
$$

For a European call option with strike \(K\), the terminal payoff is

$$
\max(S_T-K,0).
$$

The Monte Carlo estimator is then

$$
\hat C =
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\max(S_T^{(i)}-K,0).
$$

## Validation Against Black–Scholes

For the benchmark parameters:

| Parameter | Value |
| --- | ---: |
| Spot price | 100 |
| Strike | 100 |
| Risk-free rate | 5% |
| Volatility | 20% |
| Maturity | 1 year |
| Simulations | 1,000,000 |

the implementation produces approximately:

```text
Black-Scholes:  10.4506
Monte Carlo:    10.4532
Absolute error:  0.0026
```

The Monte Carlo estimator produced a standard error of approximately `0.0147`, with the analytical Black–Scholes value lying within its 95% confidence interval.

## Convergence Analysis

### Convergence towards the analytical solution

![Monte Carlo convergence](docs/figures/monte_carlo_convergence.png)

As the number of simulations increases, the Monte Carlo estimate becomes increasingly concentrated around the analytical Black–Scholes price.

### Standard-error scaling

![Standard error scaling](docs/figures/standard_error_scaling.png)

For ordinary Monte Carlo simulation, statistical error is expected to scale as

$$
O(N^{-1/2}).
$$

The numerical experiment compares the measured standard error against an \(N^{-1/2}\) reference curve, demonstrating the expected convergence behaviour.

## Analytical Greeks

The project includes analytical Black–Scholes sensitivities for European call and put options:

- Delta
- Gamma
- Vega
- Theta
- Rho

Each Greek is tested against known benchmark values and independently validated using central finite-difference approximations of the Black–Scholes pricing functions.

Sensitivity conventions:

- Theta is reported per year using the calendar-time convention.
- Vega is reported per unit change in volatility.
- Rho is reported per unit change in the continuously compounded risk-free rate.

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── figures/
│       ├── monte_carlo_convergence.png
│       └── standard_error_scaling.png
├── examples/
│   ├── convergence_analysis.py
│   └── error_scaling.py
├── src/
│   └── quantmc/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── geometric_brownian_motion.py
│       ├── pricing/
│       │   ├── __init__.py
│       │   ├── black_scholes.py
│       │   ├── greeks.py
│       │   └── monte_carlo.py
│       └── risk/
│           └── __init__.py
├── tests/
│   ├── test_black_scholes.py
│   ├── test_geometric_brownian_motion.py
│   ├── test_greeks.py
│   └── test_monte_carlo.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone git@github.com:pestopasta74/monte-carlo-risk-engine.git
cd monte-carlo-risk-engine

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

## Running the Analysis

Run the test suite:

```bash
pytest -v
```

Check code quality:

```bash
ruff check .
```

Generate the convergence figures:

```bash
python examples/convergence_analysis.py
python examples/error_scaling.py
```

## Roadmap

Future development will explore:

- Variance-reduction techniques
- Value at Risk and Expected Shortfall
- Additional stochastic models
- Performance comparisons between numerical approaches

## Motivation

My background in theoretical physics has made numerical simulation, probability, and mathematical modelling particularly interesting to me.

This project is an opportunity to apply those ideas to quantitative finance while developing a better understanding of both the underlying mathematics and the engineering required to build reliable numerical software.

## Author

**Preston Whiteman**  
Theoretical Physics — University of Bath

[LinkedIn](https://www.linkedin.com/in/pestopasta74/) · [GitHub](https://github.com/pestopasta74)