from math import ceil, floor
from random import randint, random
from typing import Union

import numpy as np

from qmf.exceptions import DistributionKwargError, NotSampledYet, SampleOutOfBounds
from qmf.globals import IMPLEMENTED_DISTRIBUTIONS, TOL


class InputVariable:
    """ Contains the distribution of a single variable, its parameters in order to sample it, and all methods to create
    a new sample, sampled values history, among others.

    Examples of basic distributions:
        - integer_random: given the upper and lower bound, returns an integer in that range. It has as kwargs the value
            for both upper and lower bound, plus two Boolean flags indicating whether the bound value (upper and lower)
            is inclusive or exclusive. Upper bound value must be greater than lower bound.
                InputVariable("n_layers", "integer_random", {
                                            "upper": 1, "lower": 4, "lower_inclusive": True, "upper_inclusive": False})
        - integer_normal: given the mean and standard deviation, returns an integer from the Gaussian normal
            distribution made using those values. Its kwargs are the mean of the distribution (mu) and its standard
            distribution (rho).
                InputVariable("n_units", "integer_normal", {"mu": 1, "rho": 2.5})
        - float_random: given the upper and lower bound, returns an float in that range.
                InputVariable("l1_penalty", "integer_float", {"upper": 1, "lower": 4})
        - float_normal: given the mean and standard deviation, returns a float from the Gaussian normal distribution
            made using those values. Its kwargs are the mean of the distribution (mu) and its standard distribution
            (rho).
                InputVariable("n_units", "float_normal", {"mu": 1, "rho": 2.5})
        - choice: given a list of possible values, returns one of them. It only kwarg is "options" which represents a
            list of the options to be sampled.
                InputVariable("kernel", "choice", {"options": ["lineal", "poly", "rbf"]})
        - constant: placeholder to keep values that does not change over iterations. They can be of any type and its
            only kwarg is the value of the constant itself.
                InputVariable("tolerance", "constant", {"value": 1e-8})

    For integer_random, integer_normal, float_random, float_normal is is possible to also add a callable (function)
    object as kwarg under the key name "lambda". This function will be called on the sampled value and will be the value
    returned when sampling:
         InputVariable("l1_penalty", "integer_float", {
            "upper": 1, "lower": 4, "lambda": lambda_power_base_2})

    """

    def __init__(self, name: str, distribution: str, kwargs: dict):
        self.name = name
        self.distribution = distribution
        assert self.distribution in IMPLEMENTED_DISTRIBUTIONS, NotImplementedError(
            f"The given distribution '{self.distribution}' is not correct or has not been implemented yet"
        )

        self.init_distribution_kwargs = kwargs
        self.current_distribution_kwargs = kwargs
        self._check_kwargs()

        self.current_sample = NotSampledYet
        self.history_samples = list()
        self.history_distributions = list()

    def __repr__(self):
        rep = (
            f"InputValue type {self.distribution}\n"
            f"Current value: {self.current_sample}\n"
            f"Current distribution: {self.current_distribution_kwargs}\n"
            f"Number of samples stored: {len(self.history_samples)}"
        )
        return rep

    def __len__(self):
        return len(self.history_samples)

    def __add__(self, other):
        assert type(self) == type(other), ValueError(
            "To perform arithmetic operations between InputValues both distributions must return the same type"
        )
        return self.current_sample + other.current_sample

    def __sub__(self, other):
        assert type(self) == type(other), ValueError(
            "To perform arithmetic operations between InputValues both distributions must return the same type"
        )
        return self.current_sample - other.current_sample

    def __mul__(self, other):
        assert type(self) == type(other), ValueError(
            "To perform arithmetic operations between InputValues both distributions must return the same type"
        )
        return self.current_sample * other.current_sample

    def __truediv__(self, other):
        assert type(self) == type(other), ValueError(
            "To perform arithmetic operations between InputValues both distributions must return the same type"
        )
        return self.current_sample / other.current_sample

    def history_samples_setter(self, value):
        """ Setter method for history_samples attribute.
        """
        self.history_samples.append(value)

    def current_distribution_kwargs_setter(self, value):
        """ Setter method for current_distribution_kwargs attribute.
        """
        self.current_distribution_kwargs = value

    def history_distributions_setter(self, value):
        """ Setter method for history_distributions attribute.
        """
        self.history_distributions.append(value)

    def _check_kwargs(self):
        """ Checks the correctness of the given distribution parameter arguments based on the selected distribution
        type.
        """
        if self.distribution == "integer_random":
            for k in self.current_distribution_kwargs.keys():
                assert k in [
                    "upper",
                    "lower",
                    "upper_inclusive",
                    "lower_inclusive",
                    "lambda",
                ], KeyError
            assert (
                "lower" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert (
                "upper" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert isinstance(
                self.current_distribution_kwargs["lower"], (int, float)
            ), TypeError
            assert isinstance(
                self.current_distribution_kwargs["upper"], (int, float)
            ), TypeError
            if "lower_inclusive" not in self.current_distribution_kwargs.keys():
                self.current_distribution_kwargs["lower_inclusive"] = True
            if "upper_inclusive" not in self.current_distribution_kwargs.keys():
                self.current_distribution_kwargs["upper_inclusive"] = True
            assert (
                self.current_distribution_kwargs["lower"]
                < self.current_distribution_kwargs["upper"]
            ), ValueError
        elif self.distribution in ["integer_normal", "float_normal"]:
            for k in self.current_distribution_kwargs.keys():
                assert k in ["mu", "rho", "lambda"], KeyError
            assert (
                "mu" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert (
                "rho" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert isinstance(
                self.current_distribution_kwargs["mu"], (int, float)
            ), TypeError
            assert isinstance(
                self.current_distribution_kwargs["rho"], (int, float)
            ), TypeError
            assert self.current_distribution_kwargs["rho"] >= 0, ValueError
        elif self.distribution == "float_random":
            for k in self.current_distribution_kwargs.keys():
                assert k in [
                    "upper",
                    "lower",
                    "upper_inclusive",
                    "lower_inclusive",
                    "lambda",
                ], KeyError
            assert (
                "lower" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert (
                "upper" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert isinstance(
                self.current_distribution_kwargs["lower"], (int, float)
            ), TypeError
            assert isinstance(
                self.current_distribution_kwargs["upper"], (int, float)
            ), TypeError
            assert (
                self.current_distribution_kwargs["lower"]
                < self.current_distribution_kwargs["upper"]
            ), ValueError
        elif self.distribution == "choice":
            for k in self.current_distribution_kwargs.keys():
                assert k in ["options"], KeyError
            assert (
                "options" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert isinstance(
                self.current_distribution_kwargs["options"], (list, tuple, set)
            ), TypeError
        elif self.distribution == "constant":
            for k in self.current_distribution_kwargs.keys():
                assert k in ["value"], KeyError
            assert (
                "value" in self.current_distribution_kwargs.keys()
            ), DistributionKwargError
            assert isinstance(
                self.current_distribution_kwargs["value"], (int, float, str, bool)
            ), TypeError

    def _check_sampled_value(self):
        """ Checks if the current sampled value is inside the allowed values base don the value conditions.
        """
        if self.distribution == "integer_random":
            if (
                self.current_distribution_kwargs["lower_inclusive"]
                and self.current_distribution_kwargs["upper_inclusive"]
            ):
                if (
                    self.current_distribution_kwargs["lower"]
                    > self.current_sample
                    > self.current_distribution_kwargs["upper"]
                ):
                    raise SampleOutOfBounds
            elif (
                not self.current_distribution_kwargs["lower_inclusive"]
                and not self.current_distribution_kwargs["upper_inclusive"]
            ):
                if (
                    self.current_distribution_kwargs["lower"]
                    >= self.current_sample
                    >= self.current_distribution_kwargs["upper"]
                ):
                    raise SampleOutOfBounds
            elif (
                not self.current_distribution_kwargs["lower_inclusive"]
                and self.current_distribution_kwargs["upper_inclusive"]
            ):
                if (
                    self.current_distribution_kwargs["lower"]
                    >= self.current_sample
                    > self.current_distribution_kwargs["upper"]
                ):
                    raise SampleOutOfBounds
            elif (
                self.current_distribution_kwargs["lower_inclusive"]
                and not self.current_distribution_kwargs["upper_inclusive"]
            ):
                if (
                    self.current_distribution_kwargs["lower"]
                    > self.current_sample
                    >= self.current_distribution_kwargs["upper"]
                ):
                    raise SampleOutOfBounds
        elif self.distribution == "float_random":
            if (
                self.current_distribution_kwargs["lower"]
                > self.current_sample
                >= self.current_distribution_kwargs["upper"]
            ):
                raise SampleOutOfBounds
        elif self.distribution == "choice":
            if self.current_sample not in self.current_distribution_kwargs["options"]:
                raise SampleOutOfBounds
        elif self.distribution == "constant":
            if self.current_sample != self.current_distribution_kwargs["value"]:
                raise SampleOutOfBounds

    def random_sample(
        self, update_history: bool = True
    ) -> Union[int, float, str, bool]:
        """ Initialized the distribution parameters based on the given keyword arguments (kwargs).

        Args:
            update_history: if True, the sampled value will be appended to history_samples list after being check its
                correctness.

        Returns:
            current_sample: value obtained from the current InputVariable distribution.
        """
        self.current_sample = NotSampledYet()
        if self.distribution == "integer_random":
            lower_tol = (
                0 if self.current_distribution_kwargs["lower_inclusive"] else TOL
            )
            upper_tol = (
                0 if self.current_distribution_kwargs["upper_inclusive"] else TOL
            )
            self.current_sample = randint(
                a=ceil(self.current_distribution_kwargs["lower"] + lower_tol),
                b=floor(self.current_distribution_kwargs["upper"] - upper_tol),
            )
            print(self.current_sample)
        elif self.distribution == "integer_normal":
            self.current_sample = int(
                np.random.normal(
                    loc=self.current_distribution_kwargs["mu"],
                    scale=self.current_distribution_kwargs["rho"],
                )
            )
        elif self.distribution == "float_random":
            self.current_sample = random()
            self.current_sample = self.current_distribution_kwargs["lower"] + (
                self.current_sample
                * (
                    self.current_distribution_kwargs["upper"]
                    - self.current_distribution_kwargs["lower"]
                )
            )
        elif self.distribution == "float_normal":
            self.current_sample = np.random.normal(
                loc=self.current_distribution_kwargs["mu"],
                scale=self.current_distribution_kwargs["rho"],
            )
        elif self.distribution == "choice":
            self.current_sample = np.random.choice(
                a=self.current_distribution_kwargs["options"]
            )
        elif self.distribution == "constant":
            self.current_sample = self.current_distribution_kwargs["value"]
        self._check_sampled_value()

        if "lambda" in self.current_distribution_kwargs.keys():
            self.current_sample = self.current_distribution_kwargs["lambda"](
                self.current_sample
            )

        if update_history:
            self.history_samples_setter(value=self.current_sample)
            self.history_distributions_setter(value=self.current_distribution_kwargs)

        return self.current_sample

    def conditioned_sample(
        self,
        new_kwargs: dict,
        update_history: bool = True,
        return_to_init_distribution: bool = False,
    ):
        """ Gets a sample of the value based on the given conditions.

        Args:
            new_kwargs: new conditions to consider when sampling a new value.
            update_history: if True, the sampled value will be appended to history_samples list after being check its
                correctness.
            return_to_init_distribution: if True, it draws a sample from the given distribution but keeps the original
                distribution kwargs as the current distribution of the object.
        """
        self.current_distribution_kwargs_setter(value=new_kwargs)
        self._check_kwargs()
        self.random_sample(update_history=update_history)
        if return_to_init_distribution:
            self.current_distribution_kwargs_setter(value=self.init_distribution_kwargs)
        return self.current_sample

    def constant_sample(self, value, update_history: bool = True):
        """ Gets a sample of the value based on the given conditions.

        Args:
            value: New value to give (force) to the current_sample attribute.
            update_history: if True, the sampled value will be appended to history_samples list after being check its
                correctness.

        Returns:
            value: value passed as argument.
        """
        self.current_sample = value
        self._check_sampled_value()
        if update_history:
            self.history_samples_setter(value=self.current_sample)
            self.history_distributions_setter(value="constant")
        return value


class SearchSpace:
    """ Contains the search space as individual InputVariable objects set, and all methods and attributes to work with
    all of them (no specific for a single InputVariable). Conditional search spaces are handled inside this object,
    deciding when and how to sample an InputVariable object based on the values of other ones.
    """

    def __init__(self):
        self.search_space = dict()
        self.sample_current = dict()
        self.history_samples = list()

    def __repr__(self):
        rep = ""
        for k, v in self.sample_current.items():
            rep += f"Parameter {k}: {v}\n"
        return rep

    @property
    def search_space(self):
        """ Getter method for search_space attribute.
        """
        return self.__search_space

    @search_space.setter
    def search_space(self, value: dict):
        """ Setter method for search_space attribute.
        """
        self.__search_space[value["key"]] = value["value"]

    @property
    def sample_current(self):
        """ Getter method for sample_current attribute.
        """
        return self.__sample_current

    @sample_current.setter
    def sample_current(self, value):
        """ Setter method for sample_current attribute.
        """
        self.__sample_current[value["key"]] = value["value"]

    @property
    def history_samples(self):
        """ Getter method for history_samples attribute.
        """
        return self.__history_samples

    @history_samples.setter
    def history_samples(self, value):
        """ Setter method for history_samples attribute.
        """
        self.__history_samples.append(value)

    def add(self, input_variable: InputVariable):
        # TODO
        assert input_variable.name not in self.search_space.keys(), KeyError(
            f"The given Key to add '{input_variable.name}' already exists in the SearchSpace dictionary"
        )
        arg_dict = {input_variable.name: input_variable}
        self.search_space(value=arg_dict)

    def remove(self, name: str):
        # TODO
        assert name in self.search_space.keys(), KeyError(
            f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
        )
        del self.search_space[name]

    def _update_history_samples(self):
        # TODO
        self.history_samples(value=self.sample_current)

    def sample_all(
        self,
        update_history_samples: bool = True,
        update_input_variable_history: bool = True,
    ):
        # TODO
        for key, value in self.search_space.items():
            self.sample_current[key] = value.random_sample(
                update_history=update_input_variable_history
            )
        if update_history_samples:
            self._update_history_samples()

    def sample_all_conditioned(
        self,
        new_kwargs: dict,
        update_history_samples: bool = True,
        update_input_variable_history: bool = True,
    ):
        # TODO
        for key, value in self.search_space.items():
            if new_kwargs[key] is None:
                self.sample_current[key] = value.random_sample(
                    update_history=update_input_variable_history
                )
            else:
                self.sample_current[key] = value.conditioned_sample(
                    new_kwargs=new_kwargs[key],
                    update_history=update_input_variable_history,
                )
        if update_history_samples:
            self._update_history_samples()

    def sample_individuals(
        self,
        key: str,
        new_kwargs: dict,
        update_history_samples: bool = True,
        update_input_variable_history: bool = True,
    ):
        # TODO
        if new_kwargs is None:
            self.sample_current[key] = self.random_sample(
                update_history=update_input_variable_history
            )
        else:
            self.sample_current[key] = self.conditioned_sample(
                new_kwargs=new_kwargs, update_history=update_input_variable_history
            )
        if update_history_samples:
            self._update_history_samples()


if __name__ == "__main__":
    pass
