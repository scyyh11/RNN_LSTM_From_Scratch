# activations.py

import numpy as np

def sigmoid(x):
    """
    Applies the sigmoid activation function.

    The sigmoid function maps input values to the range (0, 1) using the formula:
        sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x (np.ndarray): Input array or scalar.

    Returns:
        np.ndarray: Element-wise sigmoid activation of x.
    """
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Applies the hyperbolic tangent (tanh) activation function.

    The tanh function maps input values to the range (-1, 1) using the formula:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x (np.ndarray): Input array or scalar.

    Returns:
        np.ndarray: Element-wise tanh activation of x.
    """
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)
