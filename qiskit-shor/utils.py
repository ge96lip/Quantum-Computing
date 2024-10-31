from typing import Optional, Union, Tuple, List
import math


def is_power(
    N: int, return_decomposition: bool = False
) -> Union[bool, Tuple[bool, int, int]]:
    """Check if N is a perfect power and optionally return its decomposition.

    Args:
        N: The number to test
        return_decomposition: If True, returns tuple (is_power, base, exponent)

    Returns:
        If return_decomposition is False: returns bool
        If return_decomposition is True: returns (bool, base, exponent)
        where base^exponent = N if N is a perfect power
    """
    if N < 2:
        return (False, 0, 0) if return_decomposition else False

    # Check powers from 2 up to log2(N)
    for p in range(2, int(math.log2(N)) + 1):
        # Find the p-th root of N
        root = round(N ** (1 / p))
        # Check if root^p equals N
        if root**p == N:
            return (True, root, p) if return_decomposition else True

    return (False, 0, 0) if return_decomposition else False


def validate_min(name: str, value: int, min_value: int) -> None:
    """Validates that value is greater than or equal to min_value.

    Args:
        name: Name of the value being validated
        value: Value to validate
        min_value: Minimum allowed value

    Raises:
        ValueError: If value is less than min_value
    """
    if value < min_value:
        raise ValueError(
            f"{name} value {value} must be greater than or equal to {min_value}"
        )
