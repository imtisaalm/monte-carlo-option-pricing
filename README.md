# Monte Carlo Option Pricing

Prices European call and put options with risk-neutral Monte Carlo simulation
under geometric Brownian motion, and validates the estimates against the
analytical Black-Scholes price.

## Features

- **Monte Carlo pricing** of European calls and puts with configurable spot,
  strike, maturity, risk-free rate, volatility, and continuous dividend yield
- **Antithetic variates** variance reduction (on by default)
- **Standard error and 95% confidence interval** for every estimate
- **Analytical Black-Scholes benchmark** 

## Usage

Run the built-in example (spot 100, strike 105, 1 year, 4% rate, 25% vol):

```bash
python monte_carlo_pricing.py
```

Example output:

```
Option type: Call
Simulations: 100,000
Monte Carlo price: $9.5798
Standard error: $0.0530
95% confidence interval: $9.4760 to $9.6836
Black-Scholes price: $9.5563
Absolute pricing error: $0.0235
```

Or use it as a library:

```python
from monte_carlo_pricing import (
    MonteCarloOptionPricer,
    OptionParameters,
    OptionType,
)

parameters = OptionParameters(
    spot_price=100.0,
    strike_price=105.0,
    time_to_maturity=1.0,
    risk_free_rate=0.04,
    volatility=0.25,
    dividend_yield=0.0,
)

pricer = MonteCarloOptionPricer(parameters, random_seed=42)
result = pricer.price(OptionType.CALL, number_of_simulations=100_000)

print(result.estimated_price)
print(result.confidence_interval_95)
print(result.black_scholes_price)
```

Under the risk-neutral measure, the terminal asset price follows

```
S_T = S_0 · exp((r − q − σ²/2)·T + σ·√T·Z),  Z ~ N(0, 1)
```

The option price is the discounted expected payoff, estimated as the sample
mean of `exp(−rT) · payoff(S_T)` across simulated paths. Antithetic variates
pair each draw `Z` with `−Z`, which reduces the variance of the estimator at
no extra simulation cost.

## Tests

```bash
python -m unittest discover tests
```

The test suite checks Monte Carlo convergence to Black-Scholes, put-call
parity, parameter validation, and the zero-volatility edge case.
