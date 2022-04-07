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
    def test_add_InputVariable(self):
        se = SearchSpace()
        for i in range(100):
            self.assertEqual(first=i, second=len(se.search_space))
            input_variable = InputVariable(
                name=f"distribution_{i}",
                distribution="float_random",
                kwargs={"lower": 1, "upper": 10},
            )
            se.add(input_variable=input_variable)

    def test_add_and_remove_randomly_InputVariable(self):
        se = SearchSpace()
        for i in range(100):
            input_variable = InputVariable(
                name=f"distribution_{i}",
                distribution="float_random",
                kwargs={"lower": 1, "upper": 10},
            )
            se.add(input_variable=input_variable)
        for i in range(100):
            se.remove(name=f"distribution_{i}")
            self.assertEqual(first=100 - i - 1, second=len(se.search_space))

    def test_history_search_space(self):
        input_variable_int_random = InputVariable(
            name="integer_random_dist",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        input_variable_int_normal = InputVariable(
            name="integer_normal_dist",
            distribution="integer_normal",
            kwargs={"mu": 1, "rho": 0.5},
        )
        input_variable_float_random = InputVariable(
            name="float_random_dist",
            distribution="float_random",
            kwargs={"lower": -1, "upper": 0},
        )
        input_variable_float_normal = InputVariable(
            name="float_normal_dist",
            distribution="float_normal",
            kwargs={"mu": -10, "rho": 1.25},
        )
        input_variable_choice = InputVariable(
            name="choice_dist",
            distribution="choice",
            kwargs={"options": ["A", "B", "C"]},
        )
        input_variable_constant = InputVariable(
            name="constant_dist", distribution="constant", kwargs={"value": True}
        )
        se = SearchSpace()
        se.add(input_variable=input_variable_int_random)
        se.add(input_variable=input_variable_int_normal)
        se.add(input_variable=input_variable_float_random)
        se.add(input_variable=input_variable_float_normal)
        se.add(input_variable=input_variable_choice)
        se.add(input_variable=input_variable_constant)
        for i in range(10):
            se.sample_all_random()
        self.assertEqual(first=10, second=len(se.history_search_space))

    def test_history_not_updated_search_space(self):
        input_variable_int_random = InputVariable(
            name="integer_random_dist",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        input_variable_int_normal = InputVariable(
            name="integer_normal_dist",
            distribution="integer_normal",
            kwargs={"mu": 1, "rho": 0.5},
        )
        input_variable_float_random = InputVariable(
            name="float_random_dist",
            distribution="float_random",
            kwargs={"lower": -1, "upper": 0},
        )
        input_variable_float_normal = InputVariable(
            name="float_normal_dist",
            distribution="float_normal",
            kwargs={"mu": -10, "rho": 1.25},
        )
        input_variable_choice = InputVariable(
            name="choice_dist",
            distribution="choice",
            kwargs={"options": ["A", "B", "C"]},
        )
        input_variable_constant = InputVariable(
            name="constant_dist", distribution="constant", kwargs={"value": True}
        )
        se = SearchSpace()
        se.add(input_variable=input_variable_int_random)
        se.add(input_variable=input_variable_int_normal)
        se.add(input_variable=input_variable_float_random)
        se.add(input_variable=input_variable_float_normal)
        se.add(input_variable=input_variable_choice)
        se.add(input_variable=input_variable_constant)
        for i in range(10):
            se.sample_all_random(update_history_samples=False)
        self.assertEqual(first=0, second=len(se.history_search_space))

    # TODO: We are not keeping track of the history kwargs, just the sampled values
    # BUG: Sampling a new value overwrites all history_search_space items with the new sampled value
    def test_sample_whole_space_with_one_distribution_changed(self):
        integer_1 = InputVariable(
            name="integer_1",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        integer_2 = InputVariable(
            name="integer_2",
            distribution="integer_random",
            kwargs={"lower": 1, "upper": 10},
        )
        se = SearchSpace()
        se.add(input_variable=integer_1)
        se.add(input_variable=integer_2)
        for i in range(10):
            se.sample_all_random()
        second_sample = se.sample_all_conditioned(
            new_kwargs_dict={"integer_1": {"lower": 100, "upper": 101}}
        )
        self.assertEqual(first=2, second=len(se.history_search_space))
        self.assertEqual(first=2, second=len(se.history_search_space))
        self.assertEqual(first=2, second=len(se.history_search_space))

    def test_sample_whole_space_with_all_InputVariable_changed(self):
        ...

    def test_return_sampled_search_space(self):
        ...

    def test_return_history_search_space(self):
        ...

    def test_pickle_sampled_search_space(self):
        ...

    def test_pickle_history_search_space(self):
        ...

    def test_pickle_search_space_class(self):
        ...

    def test_load_pickled_search_space_class(self):
        ...
