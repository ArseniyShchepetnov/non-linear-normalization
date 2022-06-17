"""Difference matrices generation."""
import numpy as np


def invertable_diff2_matrix(length: int) -> np.ndarray:
    """Construct invertable difference matrix of second order difference."""
    D = invertable_diff1_matrix(length)
    D = np.matmul(D, D)
    return D


def invertable_diff1_matrix(length: int) -> np.ndarray:
    """Construct invertable difference matrix of second order difference."""

    I = np.eye(length)
    D = np.zeros(shape=(length, length))

    D[:, :-1] = - np.diff(I, 1, axis=1)
    D[-1, -1] = 1

    return D


def inverse_diff_1d(size: int, **kwargs) -> np.ndarray:
    """Inverse of the first order difference matrix."""

    diff_inv = np.empty(shape=(size, size), **kwargs)
    diff_inv[:, 0] = np.arange(1, size + 1)
    for i in range(1, size):
        diff_inv[:i, i] = 0
        diff_inv[i:, i] = diff_inv[:-i, i - 1]

    return diff_inv
