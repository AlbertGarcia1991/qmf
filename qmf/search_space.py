import os
from math import ceil, floor
from pathlib import Path
from pickle import dump, load
from random import randint, random
from typing import Union

import numpy as np

from qmf.exceptions import (
    MissingKwargError,
    NotSampledYet,
    SampleOutOfBounds,
    TypeKwargError,
)
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

    Raises:
        NotImplementedError: when the given distribution does not exist across the implemented ones.

    """

    def __init__(self, name: str, distribution: str, kwargs: dict):
        self.name = name
        self.distribution = distribution
        if self.distribution not in IMPLEMENTED_DISTRIBUTIONS:
            raise NotImplementedError(
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
        if type(self) != type(other):
            raise ValueError(
                "To perform arithmetic operations between InputValues both distributions must return the same type"
            )
        return self.current_sample + other.current_sample

    def __sub__(self, other):
        if type(self) != type(other):
            raise ValueError(
                "To perform arithmetic operations between InputValues both distributions must return the same type"
            )
        return self.current_sample - other.current_sample

    def __mul__(self, other):
        if type(self) != type(other):
            raise ValueError(
                "To perform arithmetic operations between InputValues both distributions must return the same type"
            )
        return self.current_sample * other.current_sample

    def __truediv__(self, other):
        if type(self) != type(other):
            raise ValueError(
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

        Raises:
            KeyError: when at least one of the given kwargs keys does not match with the expected for the current
                distribution type.
            TypeError: when at least one of the given kwargs values does not match with the expected type.
            ValueError: when at least one of the given kwargs values does not match with the expected range of values,
                or does not follow the expected conditions (e.g. for random distributions lower bound must be lower than
                the upper bound value).
            DistributionKwargError: when at least one of the expected kwargs keys for the current distribution is not
                present on the given arguments.
        """
        if self.distribution in ["integer_random", "float_random"]:
            for k in self.current_distribution_kwargs.keys():
                if k not in [
                    "upper",
                    "lower",
                    "upper_inclusive",
                    "lower_inclusive",
                    "lambda",
                ]:
                    raise KeyError
            if "lower" not in self.current_distribution_kwargs.keys():
                raise MissingKwargError
            if "upper" not in self.current_distribution_kwargs.keys():
                raise MissingKwargError
            if not isinstance(self.current_distribution_kwargs["lower"], (int, float)):
                raise TypeKwargError
            if not isinstance(self.current_distribution_kwargs["upper"], (int, float)):
                raise TypeKwargError
            if "lower_inclusive" not in self.current_distribution_kwargs.keys():
                self.current_distribution_kwargs["lower_inclusive"] = True
            if "upper_inclusive" not in self.current_distribution_kwargs.keys():
                self.current_distribution_kwargs["upper_inclusive"] = True
            if (
                self.current_distribution_kwargs["lower"]
                > self.current_distribution_kwargs["upper"]
            ):
                raise ValueError
        elif self.distribution in ["integer_normal", "float_normal"]:
            for k in self.current_distribution_kwargs.keys():
                if k not in ["mu", "rho", "lambda"]:
                    raise KeyError
            if "mu" not in self.current_distribution_kwargs.keys():
                raise MissingKwargError
            if "rho" not in self.current_distribution_kwargs.keys():
                raise MissingKwargError
            if not isinstance(self.current_distribution_kwargs["mu"], (int, float)):
                raise TypeKwargError
            if not isinstance(self.current_distribution_kwargs["rho"], (int, float)):
                raise TypeKwargError
            if self.current_distribution_kwargs["rho"] < 0:
                raise ValueError
        elif self.distribution == "choice":
            if self.current_distribution_kwargs.keys() != ["options"]:
                raise MissingKwargError
            if not isinstance(
                self.current_distribution_kwargs["options"], (list, tuple, set)
            ):
                raise TypeKwargError
        elif self.distribution == "constant":
            if self.current_distribution_kwargs.keys() != ["value"]:
                raise MissingKwargError
            if not isinstance(
                self.current_distribution_kwargs["value"],
                (int, float, bool, str, list, set, tuple),
            ):
                raise TypeKwargError

    def _check_sampled_value(self):
        """ Checks if the current sampled value is inside the allowed values base don the value conditions.

        Raises:
            SampleOutOfBounds: when the sampled value does not match with the conditioning kwargs for the current
                distribution.
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
        return_to_init_kwargs: bool = False,
    ):
        """ Gets a sample of the value based on the given conditions.

        Args:
            new_kwargs: new conditions to consider when sampling a new value.
            update_history: if True, the sampled value will be appended to history_samples list after being checked its
                correctness.
            return_to_init_kwargs: if True, it draws a sample from the given distribution but keeps the original
                distribution kwargs as the current distribution of the object.
        """
        self.current_distribution_kwargs_setter(value=new_kwargs)
        self._check_kwargs()
        self.random_sample(update_history=update_history)
        if return_to_init_kwargs:
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
    all of them (no specific for a single InputVariable). Those added InputVariable objects can be sampled (entirely or
    just some of them), change their parameters (kwards dictionaries), and the object itself can be pickled to a file
    and be read and load again by a new instance.
    """

    def __init__(self, tag: str = None):
        """ Initializes the main attributes inside the object.

        Args:
            tag: name of the current search space if given, otherwise, will be set to None.
        """
        self.tag = tag
        self.search_space = dict()
        self.current_search_space = dict()
        self.history_search_space = list()

    def __repr__(self):
        rep = ""
        for k, v in self.current_search_space.items():
            rep += f"Parameter {k}: {v}\n"
        return rep

    def _update_history_samples(self):
        """ Adds the current sampled search space to the history attribute.
        """
        self.history_search_space.append(self.current_search_space)

    def add(self, input_variable: InputVariable):
        """ Add an InputVariable object to the current SeachSpace instance.

        Args:
            input_variable: InputVariable containing the distribution, its parameters, and its name to be added to
                the current search space.

        Raises:
            KeyError: When the given InputVariable name already exists inside the current search space.
        """
        if input_variable.name in self.search_space.keys():
            raise KeyError(
                f"The given Key to add '{input_variable.name}' already exists in the SearchSpace dictionary"
            )
        self.search_space[input_variable.name] = input_variable

    def remove(self, name: str):
        """ Removes a InputVariable distribution from the instance by specifying its unique name.

        Args:
            name: name of the InputVariable object to be removed.

        Raises:
            KeyError: If the given name of the distribution to remove does not exist.
        """
        if name not in self.search_space.keys():
            KeyError(
                f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
            )
        del self.search_space[name]

    def replace_distribution_kwargs(self, name: str, new_kwargs: dict):
        """ Replaces the kwargs of the given distribution based on its name.

        Args:
            name: name of the distribution to be replaced its kwargs.
            new_kwargs: dictionary containing a new set of kwargs for the selected distribution.


        Raises:
            KeyError: If the given name of the distribution to update does not exist.
        """
        if name not in self.search_space.keys():
            KeyError(
                f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
            )
        self.search_space[name] = InputVariable(
            name=name,
            distribution=self.search_space[name].distribution,
            kwargs=new_kwargs,
        )

    def replace_distribution(self, name: str, new_distribution: InputVariable):
        """ Replaces the whole given distribution.

        Args:
            name: name of the distribution to be replaced its kwargs.
            new_distribution: new distribution to replace the selected one.

        Raises:
            KeyError: If the given name of the distribution to update does not exist.
        """
        if name not in self.search_space.keys():
            KeyError(
                f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
            )
        self.remove(name=name)
        self.add(input_variable=new_distribution)

    def sample_all_random(
        self,
        update_history_samples: bool = True,
        update_input_variable_history: bool = True,
    ) -> dict:
        """ Samples all distributions inside the SearchSpace object. It uses the current distributions inside those
        objects without modifying them. If we want to edit them before being sampled, we need to run the method
        sample_all_new_kwargs() methods inside this same class.

        Args:
            update_history_samples: if True, the new sample will be added to the history attribute inside this class.
            update_input_variable_history: if True, the new sample will be added to the history attribute inside each
                InputVariable class.

        Returns:
            current_search_space: the newly sampled search space.
        """
        for key, value in self.search_space.items():
            self.current_search_space[key] = value.random_sample(
                update_history=update_input_variable_history
            )
        if update_history_samples:
            self._update_history_samples()
        return self.current_search_space

    def sample_one_random(
        self,
        name: str,
        update_history_samples: bool = True,
        replace_history_samples: bool = False,
        update_input_variable_history: bool = True,
    ) -> Union[int, float, bool, str]:
        """ Samples a single value from the requested distribution without changing its kwargs.

        Args:
            name: name of the distribution to be sampled.
            update_history_samples: if True, a new set of sampled will be added to the search space history only
                changing the new sampled distribution.
            replace_history_samples: if True, the new sampled distribution will be replaced in the last value inside the
                search space history attribute.
            update_input_variable_history: if True, the new sample will be added to the history attribute inside each
                InputVariable class.

        Returns:
            current_search_space: the newly sampled value.

        Raises:
            KeyError: if the given name of the distribution to update does not exist.
            ValueError: if both update_history_samples and replace_history_samples arguments are set to True.
        """
        if name not in self.search_space.keys():
            raise KeyError(
                f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
            )
        if update_history_samples and replace_history_samples:
            raise ValueError(
                "Both 'update_history_samples' and 'replace_history_samples' arguments cannot be True"
            )
        single_sample = self.search_space[name].random_sample(
            update_history=update_input_variable_history
        )
        if update_history_samples:
            self.current_search_space[name] = single_sample
            self._update_history_samples()
        elif replace_history_samples:
            self.current_search_space[name] = single_sample
            self.history_search_space[-1][name] = single_sample
        return single_sample

    def sample_all_conditioned(
        self,
        new_kwargs_dict: dict,
        return_to_init_kwargs: bool = False,
        update_history_samples: bool = True,
        update_input_variable_history: bool = True,
    ) -> dict:
        """ Samples all distributions inside the SearchSpace object changing the given distribution kwargs.

        Args:
            new_kwargs_dict: dictionary containing the new distribution kwargs where the key is the name of the
                distribution and the value is the dictionary containing the new kwargs. If a distribution name is not
                given in the new_kwargs_dict keys, it will keep its old distribution kwargs.
            return_to_init_kwargs: if True, it draws a sample from the given distribution but keeps the original
                distribution kwargs as the current distribution of the object.
            update_history_samples: if True, the new sample will be added to the history attribute inside this class.
            update_input_variable_history: if True, the new sample will be added to the history attribute inside each
                InputVariable class.

        Returns:
            current_search_space: the newly sampled search space.
        """
        for key, value in self.search_space.items():
            if key in new_kwargs_dict.keys():
                self.current_search_space[key] = value.conditioned_sample(
                    new_kwargs=new_kwargs_dict[key],
                    update_history=update_input_variable_history,
                    return_to_init_kwargs=return_to_init_kwargs,
                )
            else:
                self.current_search_space[key] = value.random_sample(
                    update_history=update_input_variable_history
                )
        if update_history_samples:
            self._update_history_samples()
        return self.current_search_space

    def sample_one_conditioned(
        self,
        name: str,
        new_kwargs: dict,
        return_to_init_kwargs: bool = False,
        update_history_samples: bool = True,
        replace_history_samples: bool = False,
        update_input_variable_history: bool = True,
    ) -> Union[int, float, bool, str]:
        """ Samples the requested distribution, changing its kwargs before drawing that sample.

        Args:
            name: name of the distribution to be sampled.
            new_kwargs: dictionary containing the new distribution kwargs where the key is the name of the distribution
                and the value is the dictionary containing the new kwargs. If a distribution name is not given in the
                new_kwargs keys, it will keep its old distribution kwargs.
            return_to_init_kwargs: if True, it draws a sample from the given distribution but keeps the original
                distribution kwargs as the current distribution of the object.
            update_history_samples: if True, a new set of sampled will be added to the search space history only
                changing the new sampled distribution.
            replace_history_samples: if True, the new sampled distribution will be replaced in the last value inside the
                search space history attribute.
            update_input_variable_history: if True, the new sample will be added to the history attribute inside each
                InputVariable class.

        Returns:
            current_search_space: the newly sampled value.

        Raises:
            KeyError: if the given name of the distribution to update does not exist.
            ValueError: if both update_history_samples and replace_history_samples arguments are set to True.
        """
        if name not in self.search_space.keys():
            raise KeyError(
                f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
            )
        if update_history_samples and replace_history_samples:
            raise ValueError(
                "Both 'update_history_samples' and 'replace_history_samples' arguments cannot be True"
            )
        single_sample = self.search_space[name].conditioned_sample(
            new_kwargs=new_kwargs,
            update_history=update_input_variable_history,
            return_to_init_kwargs=return_to_init_kwargs,
        )
        if update_history_samples:
            self.current_search_space[name] = single_sample
            self._update_history_samples()
        elif replace_history_samples:
            self.current_search_space[name] = single_sample
            self.history_search_space[-1][name] = single_sample
        return single_sample

    def sample_all_constant(
        self,
        new_values: dict,
        update_history_samples: bool = True,
        update_input_variable_history: bool = True,
    ) -> dict:
        """ Samples all distributions inside the SearchSpace object as constant values.

        Args:
            new_values: dictionary containing the new values for each distribution, where the distribution name is the
                key. If a distribution name is not given in the new_values keys, it will keep its old distribution
                kwargs.
            update_history_samples: if True, the new sample will be added to the history attribute inside this class.
            update_input_variable_history: if True, the new sample will be added to the history attribute inside each
                InputVariable class.

        Returns:
            current_search_space: the newly sampled search space.
        """
        for key, value in self.search_space.items():
            if key in new_values.keys():
                self.current_search_space[key] = value.constant_sample(
                    value=new_values[key], update_history=update_input_variable_history
                )
            else:
                self.current_search_space[key] = value.random_sample(
                    update_history=update_input_variable_history
                )
        if update_history_samples:
            self._update_history_samples()
        return self.current_search_space

    def sample_one_constant(
        self,
        name: str,
        new_value: Union[int, float, bool, str],
        update_history_samples: bool = True,
        replace_history_samples: bool = False,
        update_input_variable_history: bool = True,
    ) -> Union[int, float, bool, str]:
        """ Samples a single constant value from the requested distribution.

        Args:
            name: name of the distribution to be sampled.
            new_value: new value to give to the requested distribution.
            update_history_samples: if True, a new set of sampled will be added to the search space history only
                changing the new sampled distribution.
            replace_history_samples: if True, the new sampled distribution will be replaced in the last value inside the
                search space history attribute.
            update_input_variable_history: if True, the new sample will be added to the history attribute inside each
                InputVariable class.

        Returns:
            current_search_space: the newly sampled value.

        Raises:
            KeyError: if the given name of the distribution to update does not exist.
            ValueError: if both update_history_samples and replace_history_samples arguments are set to True.
        """
        if name not in self.search_space.keys():
            raise KeyError(
                f"The given Key to delete '{name}' does not exist in the SearchSpace dictionary"
            )
        if update_history_samples and replace_history_samples:
            raise ValueError(
                "Both 'update_history_samples' and 'replace_history_samples' arguments cannot be True"
            )
        single_sample = self.search_space[name].constant_sample(
            value=new_value, update_history=update_input_variable_history
        )
        if update_history_samples:
            self.current_search_space[name] = single_sample
            self._update_history_samples()
        elif replace_history_samples:
            self.current_search_space[name] = single_sample
            self.history_search_space[-1][name] = single_sample
        return single_sample

    def dump_current_search_space(self, filename: str, dst: Path = Path(os.getcwd())):
        """ Pickles the current search space dictionary to a file.

        Args:
            filename: name to give to the file.
            dst: destination directory where to store the pickled file. If not given, will be stored in the current
                working directory.

        Raises:
            NotADirectoryError: if the given directory where to save the file does not exist.
        """
        if not os.path.isdir(dst):
            raise NotADirectoryError("The given destination directory does not exist")
        with open(Path(dst + filename), "wb") as f:
            dump(obj=self.current_search_space, file=f)

    def dump_history_search_space(self, filename: str, dst: Path = Path(os.getcwd())):
        """ Pickles the historical search spaces list to a file.

        Args:
            filename: name to give to the file.
            dst: destination directory where to store the pickled file. If not given, will be stored in the current
                working directory.

        Raises:
            NotADirectoryError: if the given directory where to save the file does not exist.
        """
        if not os.path.isdir(dst):
            raise NotADirectoryError("The given destination directory does not exist")
        with open(Path(dst + filename), "wb") as f:
            dump(obj=self.history_search_space, file=f)

    def load_search_space(
        self, filename: str, dst: Path = Path(os.getcwd()), replace_tag: bool = False
    ):
        """ Load a saved SearchSpace object, assigning all attributes form the loaded one to the current instance.

        Args:
            filename: name of the file to load.
            dst: destination directory where to store the pickled file. If not given, will search the file in the
                current working directory.
            replace_tag: if True, the current instance tag will be overwritten from the loaded object.

        Raises:
            NotADirectoryError: if the given directory from where to load the file does not exist.
        """
        if not os.path.isdir(dst):
            raise NotADirectoryError("The given destination directory does not exist")
        with open(Path(dst + filename), "rb") as f:
            loaded_SearchSpace = load(file=f)
        self.search_space = loaded_SearchSpace.search_space
        self.history_search_space = loaded_SearchSpace.history_search_space
        self.current_search_space = loaded_SearchSpace.current_search_space
        if replace_tag:
            self.tag = loaded_SearchSpace.tag


if __name__ == "__main__":
    pass
