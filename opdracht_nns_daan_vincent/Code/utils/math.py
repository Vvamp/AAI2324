import math


def sigmoid(val: float) -> float:
    """
    Calculate the sigmoid of a given value.

    Args:
        val (float): The input value.

    Returns:
        float: The sigmoid of the input value.
    """
    return 1 / (1 + pow(math.e, -val))


def sigmoid_derivative(val: float) -> float:
    """
    Calculate the derivative of the sigmoid function.

    Args:
        val (float): The input value.

    Returns:
        float: The derivative of the sigmoid function.
    """
    return sigmoid(val) * (1 - sigmoid(val))
