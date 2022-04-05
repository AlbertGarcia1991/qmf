from random import randint, random
from unittest import TestCase

from numpy import log10

from qmf.globals import TOL
from qmf.lambdas import log_10, power_base_2, power_base_10, power_exp_2, power_exp_10


class TestLambdaFunctions(TestCase):
    def test_power_base_2_integer(self):
        for _ in range(100):
            controlled_input = randint(-10, 10)
            self.assertEqual(
                first=2 ** controlled_input,
                second=power_base_2(input_value=controlled_input),
            )

    def test_power_base_2_float(self):
        for _ in range(100):
            controlled_input = random()
            self.assertEqual(
                first=2 ** controlled_input,
                second=power_base_2(input_value=controlled_input),
            )

    def test_power_base_10_integer(self):
        for _ in range(100):
            controlled_input = randint(-10, 10)
            self.assertEqual(
                first=10 ** controlled_input,
                second=power_base_10(input_value=controlled_input),
            )

    def test_power_base_10_float(self):
        for _ in range(100):
            controlled_input = random()
            self.assertEqual(
                first=10 ** controlled_input,
                second=power_base_10(input_value=controlled_input),
            )

    def test_power_exp_2_integer(self):
        for _ in range(100):
            controlled_input = randint(-10, 10)
            self.assertEqual(
                first=controlled_input ** 2,
                second=power_exp_2(input_value=controlled_input),
            )

    def test_power_exp_2_float(self):
        for _ in range(100):
            controlled_input = random()
            self.assertEqual(
                first=controlled_input ** 2,
                second=power_exp_2(input_value=controlled_input),
            )

    def test_power_exp_10_integer(self):
        for _ in range(100):
            controlled_input = randint(-10, 10)
            self.assertEqual(
                first=controlled_input ** 10,
                second=power_exp_10(input_value=controlled_input),
            )

    def test_power_exp_10_float(self):
        for _ in range(100):
            controlled_input = random()
            self.assertEqual(
                first=controlled_input ** 10,
                second=power_exp_10(input_value=controlled_input),
            )

    def test_log_10_integer(self):
        for _ in range(100):
            controlled_input = randint(1, 10)
            self.assertEqual(
                first=log10(controlled_input),
                second=log_10(input_value=controlled_input),
            )

    def test_log_10_float(self):
        for _ in range(100):
            controlled_input = random() + TOL
            self.assertEqual(
                first=log10(controlled_input),
                second=log_10(input_value=controlled_input),
            )
