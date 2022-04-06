import gc
import time
from random import randint, random
from unittest import TestCase

import numpy as np
import pytest
from scipy.stats import chisquare

from qmf.lambdas import power_base_2
from qmf.search_space import InputVariable, SearchSpace


class TestInputVariable(TestCase):
    # Distribution from the given implemented
    def test_not_implemented_distribution(self):
        with pytest.raises(Exception):
            InputVariable(name="test_distribution", distribution="fake_distribution")

    # __add__
    def test_sum_sampled_integers_random(self):
        first = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int + second_int,
            second=first.current_sample + second.current_sample,
        )

    def test_sum_sampled_integers_normal(self):
        first = InputVariable(
            name="first", distribution="integer_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second", distribution="integer_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int + second_int,
            second=first.current_sample + second.current_sample,
        )

    def test_sum_sampled_floats_random(self):
        first = InputVariable(
            name="first", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float + second_float,
            second=first.current_sample + second.current_sample,
        )

    def test_sum_sampled_floats_normal(self):
        first = InputVariable(
            name="first", distribution="float_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float + second_float,
            second=first.current_sample + second.current_sample,
        )

    # __sub__
    def test_sub_sampled_integers_random(self):
        first = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int - second_int,
            second=first.current_sample - second.current_sample,
        )

    def test_sub_sampled_integers_normal(self):
        first = InputVariable(
            name="first", distribution="integer_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second", distribution="integer_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int - second_int,
            second=first.current_sample - second.current_sample,
        )

    def test_sub_sampled_floats_random(self):
        first = InputVariable(
            name="first", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float - second_float,
            second=first.current_sample - second.current_sample,
        )

    def test_sub_sampled_floats_normal(self):
        first = InputVariable(
            name="first", distribution="float_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float - second_float,
            second=first.current_sample - second.current_sample,
        )

    # __mul__
    def test_mul_sampled_integers_random(self):
        first = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int * second_int,
            second=first.current_sample * second.current_sample,
        )

    def test_mul_sampled_integers_normal(self):
        first = InputVariable(
            name="first", distribution="integer_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second", distribution="integer_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int * second_int,
            second=first.current_sample * second.current_sample,
        )

    def test_mul_sampled_floats_random(self):
        first = InputVariable(
            name="first", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float * second_float,
            second=first.current_sample * second.current_sample,
        )

    def test_mul_sampled_floats_normal(self):
        first = InputVariable(
            name="first", distribution="float_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float * second_float,
            second=first.current_sample * second.current_sample,
        )

    # __truediv__
    def test_truediv_sampled_integers_random(self):
        first = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        second_int = second.random_sample()
        self.assertEqual(
            first=first_int / second_int,
            second=first.current_sample / second.current_sample,
        )

    def test_truediv_sampled_integers_normal(self):
        first = InputVariable(
            name="first", distribution="integer_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_int = first.random_sample()
        second = InputVariable(
            name="second", distribution="integer_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_int = 0
        while second_int == 0:
            second_int = second.random_sample()
        self.assertEqual(
            first=first_int / second_int,
            second=first.current_sample / second.current_sample,
        )

    def test_truediv_sampled_floats_random(self):
        first = InputVariable(
            name="first", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_random", kwargs={"lower": 1, "upper": 10}
        )
        second_float = second.random_sample()
        self.assertEqual(
            first=first_float / second_float,
            second=first.current_sample / second.current_sample,
        )

    def test_truediv_sampled_floats_normal(self):
        first = InputVariable(
            name="first", distribution="float_normal", kwargs={"mu": 0, "rho": 10}
        )
        first_float = first.random_sample()
        second = InputVariable(
            name="second", distribution="float_normal", kwargs={"mu": 0, "rho": 1.5}
        )
        second_float = 0
        while second_float == 0:
            second_float = second.random_sample()
        self.assertEqual(
            first=first_float / second_float,
            second=first.current_sample / second.current_sample,
        )

    # new samples are added to object history based on update argument
    def test_new_sample_added_to_history(self):
        instance = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        for i in range(100):
            instance.random_sample()
            history = instance.history_samples
            self.assertEqual(first=i + 1, second=len(history))

    def test_new_sample_added_to_history_random_update(self):
        instance = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        times_history_updated = 0
        for _ in range(100):
            update_flag = random()
            if update_flag < 0.5:
                times_history_updated += 1
            instance.random_sample(update_history=True if update_flag < 0.5 else False)
            history = instance.history_samples
            self.assertEqual(first=times_history_updated, second=len(history))

    # new distributions are added to object history based on update argument
    def test_new_distribution_added_to_history_random_update(self):
        instance = InputVariable(
            name="first",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        times_history_updated = 0
        for _ in range(100):
            update_flag = random()
            if update_flag < 0.5:
                times_history_updated += 1
                instance.conditioned_sample(
                    new_kwargs={"lower": randint(1, 3), "upper": randint(5, 8)}
                )
            history = instance.history_distributions
            self.assertEqual(first=times_history_updated, second=len(history))

    # kwargs errors handling
    def test_upper_bound_leq_lower_bound_integer_random(self):
        with pytest.raises(Exception):
            InputVariable(
                name="test_distribution",
                distribution="integer_random",
                kwargs={"lower": 10, "upper": 5},
            )

    def test_upper_bound_leq_lower_bound_neg_and_zero_integer_random(self):
        with pytest.raises(Exception):
            InputVariable(
                name="test_distribution",
                distribution="integer_random",
                kwargs={"lower": -10, "upper": 0},
            )

    def test_upper_bound_leq_lower_bound_float_random(self):
        with pytest.raises(Exception):
            InputVariable(
                name="test_distribution",
                distribution="float_random",
                kwargs={"lower": 10, "upper": 5},
            )

    def test_upper_bound_leq_lower_bound_neg_and_zero_float_random(self):
        with pytest.raises(Exception):
            InputVariable(
                name="test_distribution",
                distribution="float_random",
                kwargs={"lower": -10, "upper": 0},
            )

    def test_integer_normal_rho_negative(self):
        with pytest.raises(Exception):
            InputVariable(
                name="test_distribution",
                distribution="integer_normal",
                kwargs={"mu": -10, "rho": -5},
            )

    def test_float_normal_rho_negative(self):
        with pytest.raises(Exception):
            InputVariable(
                name="test_distribution",
                distribution="float_normal",
                kwargs={"mu": -10, "rho": -5},
            )

    # bounded (random) distributions
    def test_both_bound_inclusive_integer_random(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 2},
        )
        for _ in range(100):
            value = instance.random_sample()
            self.assertIn(member=value, container=[1, 2])

    def test_upper_bound_inclusive_integer_random(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 2, "lower_inclusive": False},
        )
        for _ in range(100):
            value = instance.random_sample()
            self.assertEqual(first=value, second=2)

    def test_lower_bound_inclusive_integer_random(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 2, "upper_inclusive": False},
        )
        for _ in range(100):
            value = instance.random_sample()
            self.assertEqual(first=value, second=1)

    def test_both_bound_exclusive_integer_random(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="integer_random",
            kwargs={
                "lower": 0,
                "upper": 2,
                "lower_inclusive": False,
                "upper_inclusive": False,
            },
        )
        for _ in range(100):
            value = instance.random_sample()
            self.assertEqual(first=value, second=1)

    # lambda functions
    def test_lambda_after_integer_random(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="integer_random",
            kwargs={
                "lower": 1,
                "upper": 3,
                "lower_inclusive": False,
                "upper_inclusive": False,
                "lambda": power_base_2,
            },
        )
        for i in range(10):
            value = instance.conditioned_sample(
                new_kwargs={
                    "lower": 1 + i,
                    "upper": 3 + i,
                    "lower_inclusive": False,
                    "upper_inclusive": False,
                    "lambda": power_base_2,
                }
            )
            self.assertEqual(first=value, second=2 ** (i + 2))

    # normal distributions are normal
    def test_integer_normal_samples_are_normal(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="integer_normal",
            kwargs={"mu": 0, "rho": 5},
        )
        drawn_samples = []
        pvalue = 0.0
        while pvalue == 0:
            for i in range(1000):
                drawn_samples.append(instance.random_sample())
            _, pvalue = chisquare(drawn_samples)
            if pvalue == 0:
                time.sleep(0.5)
                gc.collect()
        self.assertGreaterEqual(
            a=pvalue, b=0.05
        )  # Chi-Square test to reject Null hypothesis if p-value > 0.05
        self.assertLess(a=np.mean(drawn_samples), b=1e-1)
        self.assertLess(a=np.std(drawn_samples) - 5, b=1e-1)

    def test_float_normal_samples_are_normal(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="float_normal",
            kwargs={"mu": 0, "rho": 5},
        )
        drawn_samples = []
        pvalue = 0.0
        while pvalue == 0:
            for i in range(1000):
                drawn_samples.append(instance.random_sample())
            _, pvalue = chisquare(drawn_samples)
            if pvalue == 0:
                time.sleep(0.5)
                gc.collect()
        self.assertGreaterEqual(
            a=pvalue, b=0.05
        )  # Chi-Square test to reject Null hypothesis if p-value > 0.05
        self.assertLess(a=np.mean(drawn_samples), b=1e-1)
        self.assertLess(a=np.std(drawn_samples) - 5, b=1e-1)

    # choice in given options
    def test_choice_sample_in_options(self):
        instance = InputVariable(
            name="test_distribution",
            distribution="choice",
            kwargs={"options": [1, 2, 3, 4]},
        )
        self.assertIn(member=instance.random_sample(), container=[1, 2, 3, 4])

    # constant value remains the same
    def test_constant_not_changing(self):
        instance = InputVariable(
            name="test_distribution", distribution="constant", kwargs={"value": 1}
        )
        self.assertEqual(first=instance.random_sample(), second=1)
        instance = InputVariable(
            name="test_distribution", distribution="constant", kwargs={"value": 1.56}
        )
        self.assertEqual(first=instance.random_sample(), second=1.56)
        instance = InputVariable(
            name="test_distribution", distribution="constant", kwargs={"value": True}
        )
        self.assertEqual(first=instance.random_sample(), second=True)
        instance = InputVariable(
            name="test_distribution", distribution="constant", kwargs={"value": "text"}
        )
        self.assertEqual(first=instance.random_sample(), second="text")


class TestSearchSpace(TestCase):
    ...
