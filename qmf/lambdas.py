from typing import Union

from numpy import log10


def power_base_2(input_value: Union[int, float]) -> Union[int, float]:
    output_value = 2 ** input_value
    if isinstance(input_value, int):
        output_value = int(output_value)
    return output_value


def power_base_10(input_value: Union[int, float]) -> Union[int, float]:
    output_value = 10 ** input_value
    if isinstance(input_value, int):
        output_value = int(output_value)
    return output_value


def power_exp_2(input_value: Union[int, float]) -> Union[int, float]:
    output_value = input_value ** 2
    if isinstance(input_value, int):
        output_value = int(output_value)
    return output_value


def power_exp_10(input_value: Union[int, float]) -> Union[int, float]:
    output_value = input_value ** 10
    if isinstance(input_value, int):
        output_value = int(output_value)
    return output_value


def log_10(input_value: Union[int, float]) -> Union[int, float]:
    output_value = log10(input_value)
    if isinstance(input_value, int):
        output_value = int(output_value)
    return output_value
