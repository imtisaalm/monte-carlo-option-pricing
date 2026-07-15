import unittest
from math import exp

from monte_carlo_pricing import (
    MonteCarloOptionPricer,
    OptionParameters,
    OptionType,
    black_scholes_price,
)

BASE_PARAMETERS = OptionParameters(
    spot_price=100.0,
    strike_price=105.0,
    time_to_maturity=1.0,
    risk_free_rate=0.04,
    volatility=0.25,
    dividend_yield=0.0,
)


class TestOptionParameters(unittest.TestCase):
    def test_rejects_non_positive_spot(self):
        with self.assertRaises(ValueError):
            OptionParameters(0.0, 105.0, 1.0, 0.04, 0.25)

    def test_rejects_non_positive_strike(self):
        with self.assertRaises(ValueError):
            OptionParameters(100.0, -1.0, 1.0, 0.04, 0.25)

    def test_rejects_non_positive_maturity(self):
        with self.assertRaises(ValueError):
            OptionParameters(100.0, 105.0, 0.0, 0.04, 0.25)

    def test_rejects_negative_volatility(self):
        with self.assertRaises(ValueError):
            OptionParameters(100.0, 105.0, 1.0, 0.04, -0.25)


class TestBlackScholes(unittest.TestCase):
    def test_known_call_price(self):
        # Reference value computed independently for these parameters.
        price = black_scholes_price(BASE_PARAMETERS, OptionType.CALL)
        self.assertAlmostEqual(price, 9.5563, places=3)

    def test_put_call_parity(self):
        p = BASE_PARAMETERS
        call = black_scholes_price(p, OptionType.CALL)
        put = black_scholes_price(p, OptionType.PUT)

        forward_minus_strike = (
            p.spot_price * exp(-p.dividend_yield * p.time_to_maturity)
            - p.strike_price * exp(-p.risk_free_rate * p.time_to_maturity)
        )
        self.assertAlmostEqual(call - put, forward_minus_strike, places=10)

    def test_zero_volatility_call(self):
        p = OptionParameters(100.0, 90.0, 1.0, 0.04, 0.0)
        expected = 100.0 - 90.0 * exp(-0.04)
        price = black_scholes_price(p, OptionType.CALL)
        self.assertAlmostEqual(price, expected, places=10)

    def test_zero_volatility_out_of_the_money_put_is_worthless(self):
        p = OptionParameters(100.0, 90.0, 1.0, 0.04, 0.0)
        self.assertEqual(black_scholes_price(p, OptionType.PUT), 0.0)


class TestMonteCarloOptionPricer(unittest.TestCase):
    def test_call_converges_to_black_scholes(self):
        pricer = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=42)
        result = pricer.price(OptionType.CALL, number_of_simulations=200_000)
        self.assertLess(
            abs(result.estimated_price - result.black_scholes_price),
            4 * result.standard_error + 1e-9,
        )

    def test_put_converges_to_black_scholes(self):
        pricer = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=42)
        result = pricer.price(OptionType.PUT, number_of_simulations=200_000)
        self.assertLess(
            abs(result.estimated_price - result.black_scholes_price),
            4 * result.standard_error + 1e-9,
        )

    def test_seed_makes_results_reproducible(self):
        first = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=7).price(
            OptionType.CALL, number_of_simulations=10_000
        )
        second = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=7).price(
            OptionType.CALL, number_of_simulations=10_000
        )
        self.assertEqual(first.estimated_price, second.estimated_price)

    def test_antithetic_variates_reduce_standard_error(self):
        pricer = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=42)
        with_av = pricer.price(
            OptionType.CALL,
            number_of_simulations=100_000,
            use_antithetic_variates=True,
        )
        without_av = pricer.price(
            OptionType.CALL,
            number_of_simulations=100_000,
            use_antithetic_variates=False,
        )
        self.assertLess(with_av.standard_error, without_av.standard_error)

    def test_odd_simulation_count_returns_exact_size(self):
        pricer = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=1)
        prices = pricer.simulate_terminal_prices(
            number_of_simulations=99_999,
            use_antithetic_variates=True,
        )
        self.assertEqual(prices.shape, (99_999,))

    def test_rejects_non_positive_simulation_count(self):
        pricer = MonteCarloOptionPricer(BASE_PARAMETERS)
        with self.assertRaises(ValueError):
            pricer.simulate_terminal_prices(number_of_simulations=0)

    def test_confidence_interval_contains_estimate(self):
        pricer = MonteCarloOptionPricer(BASE_PARAMETERS, random_seed=3)
        result = pricer.price(OptionType.PUT, number_of_simulations=50_000)
        lower, upper = result.confidence_interval_95
        self.assertLessEqual(lower, result.estimated_price)
        self.assertGreaterEqual(result.estimated_price, lower)
        self.assertLessEqual(result.estimated_price, upper)


if __name__ == "__main__":
    unittest.main()
