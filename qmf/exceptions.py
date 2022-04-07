class SampleOutOfBounds:
    """ When the current sampled value is outside the conditioned bounds"""

    ...


class NotSampledYet:
    """When the current input variable has not been sampled yet"""

    ...


class MissingKwargError:
    """ When a required argument for a value distribution is not present"""

    ...


class TypeKwargError:
    """ When one of values has a non valid type for a search space distribution"""

    ...
