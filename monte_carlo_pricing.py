from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import erf, exp, log, sqrt
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class OptionParameters:
    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot_price <= 0:
            raise ValueError("spot_price must be greater than 0.")

        if self.strike_price <= 0:
            raise ValueError("strike_price must be greater than 0.")

        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be greater than 0.")

        if self.volatility < 0:
            raise ValueError("volatility cannot be negative.")


@dataclass(frozen=True)
class PricingResult:
    estimated_price: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    number_of_simulations: int
    option_type: OptionType
    black_scholes_price: Optional[float] = None

    @property
    def absolute_error_vs_black_scholes(self) -> Optional[float]:
        if self.black_scholes_price is None:
            return None

        return abs(self.estimated_price - self.black_scholes_price)


class MonteCarloOptionPricer:
    """
    Prices European call and put options using risk-neutral Monte Carlo
    simulation under geometric Brownian motion.
    """

    def __init__(
        self,
        parameters: OptionParameters,
        random_seed: Optional[int] = None,
    ) -> None:
        self.parameters = parameters
        self.random_seed = random_seed

    def simulate_terminal_prices(
        self,
        number_of_simulations: int = 100_000,
        use_antithetic_variates: bool = True,
    ) -> NDArray[np.float64]:
        if number_of_simulations <= 0:
            raise ValueError("number_of_simulations must be greater than 0.")

        rng = np.random.default_rng(self.random_seed)
        p = self.parameters

        if use_antithetic_variates:
            half_size = (number_of_simulations + 1) // 2
            random_draws = rng.standard_normal(half_size)

            random_draws = np.concatenate(
                [random_draws, -random_draws]
            )[:number_of_simulations]
        else:
            random_draws = rng.standard_normal(number_of_simulations)

        drift = (
            p.risk_free_rate
            - p.dividend_yield
            - 0.5 * p.volatility**2
        ) * p.time_to_maturity

        diffusion = (
            p.volatility
            * sqrt(p.time_to_maturity)
            * random_draws
        )

        terminal_prices = p.spot_price * np.exp(drift + diffusion)

        return terminal_prices.astype(np.float64)

    def price(
        self,
        option_type: OptionType,
        number_of_simulations: int = 100_000,
        use_antithetic_variates: bool = True,
    ) -> PricingResult:
        terminal_prices = self.simulate_terminal_prices(
            number_of_simulations=number_of_simulations,
            use_antithetic_variates=use_antithetic_variates,
        )

        p = self.parameters

        if option_type == OptionType.CALL:
            payoffs = np.maximum(
                terminal_prices - p.strike_price,
                0.0,
            )
        elif option_type == OptionType.PUT:
            payoffs = np.maximum(
                p.strike_price - terminal_prices,
                0.0,
            )
        else:
            raise ValueError(f"Unsupported option type: {option_type}")

        discount_factor = exp(
            -p.risk_free_rate * p.time_to_maturity
        )

        discounted_payoffs = discount_factor * payoffs
        estimated_price = float(np.mean(discounted_payoffs))

        if number_of_simulations > 1:
            payoff_standard_deviation = float(
                np.std(discounted_payoffs, ddof=1)
            )

            standard_error = (
                payoff_standard_deviation
                / sqrt(number_of_simulations)
            )
        else:
            standard_error = 0.0

        confidence_margin = 1.96 * standard_error

        confidence_interval = (
            max(0.0, estimated_price - confidence_margin),
            estimated_price + confidence_margin,
        )

        analytical_price = black_scholes_price(
            parameters=p,
            option_type=option_type,
        )

        return PricingResult(
            estimated_price=estimated_price,
            standard_error=standard_error,
            confidence_interval_95=confidence_interval,
            number_of_simulations=number_of_simulations,
            option_type=option_type,
            black_scholes_price=analytical_price,
        )


def standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def black_scholes_price(
    parameters: OptionParameters,
    option_type: OptionType,
) -> float:
    """
    Calculates the analytical Black-Scholes price for a European option.
    """

    p = parameters

    if p.volatility == 0:
        deterministic_terminal_price = (
            p.spot_price
            * exp(
                (
                    p.risk_free_rate
                    - p.dividend_yield
                )
                * p.time_to_maturity
            )
        )

        if option_type == OptionType.CALL:
            payoff = max(
                deterministic_terminal_price - p.strike_price,
                0.0,
            )
        else:
            payoff = max(
                p.strike_price - deterministic_terminal_price,
                0.0,
            )

        return exp(
            -p.risk_free_rate * p.time_to_maturity
        ) * payoff

    denominator = p.volatility * sqrt(p.time_to_maturity)

    d1 = (
        log(p.spot_price / p.strike_price)
        + (
            p.risk_free_rate
            - p.dividend_yield
            + 0.5 * p.volatility**2
        )
        * p.time_to_maturity
    ) / denominator

    d2 = d1 - denominator

    discounted_spot = (
        p.spot_price
        * exp(-p.dividend_yield * p.time_to_maturity)
    )

    discounted_strike = (
        p.strike_price
        * exp(-p.risk_free_rate * p.time_to_maturity)
    )

    if option_type == OptionType.CALL:
        return (
            discounted_spot * standard_normal_cdf(d1)
            - discounted_strike * standard_normal_cdf(d2)
        )

    if option_type == OptionType.PUT:
        return (
            discounted_strike * standard_normal_cdf(-d2)
            - discounted_spot * standard_normal_cdf(-d1)
        )

    raise ValueError(f"Unsupported option type: {option_type}")


def print_result(result: PricingResult) -> None:
    lower_bound, upper_bound = result.confidence_interval_95

    print(f"\nOption type: {result.option_type.value.title()}")
    print(f"Simulations: {result.number_of_simulations:,}")
    print(f"Monte Carlo price: ${result.estimated_price:.4f}")
    print(f"Standard error: ${result.standard_error:.4f}")

    print(
        "95% confidence interval: "
        f"${lower_bound:.4f} to ${upper_bound:.4f}"
    )

    if result.black_scholes_price is not None:
        print(
            "Black-Scholes price: "
            f"${result.black_scholes_price:.4f}"
        )

    if result.absolute_error_vs_black_scholes is not None:
        print(
            "Absolute pricing error: "
            f"${result.absolute_error_vs_black_scholes:.4f}"
        )


def main() -> None:
    parameters = OptionParameters(
        spot_price=100.0,
        strike_price=105.0,
        time_to_maturity=1.0,
        risk_free_rate=0.04,
        volatility=0.25,
        dividend_yield=0.0,
    )

    pricer = MonteCarloOptionPricer(
        parameters=parameters,
        random_seed=42,
    )

    call_result = pricer.price(
        option_type=OptionType.CALL,
        number_of_simulations=100_000,
        use_antithetic_variates=True,
    )

    put_result = pricer.price(
        option_type=OptionType.PUT,
        number_of_simulations=100_000,
        use_antithetic_variates=True,
    )

    print_result(call_result)
    print_result(put_result)


if __name__ == "__main__":
    main()
