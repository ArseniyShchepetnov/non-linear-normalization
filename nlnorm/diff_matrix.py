"""Difference matrices generation."""
import numpy as np


def invertable_diff2_matrix(length: int) -> np.ndarray:
    """Construct invertable difference matrix of second order difference."""
    diff_mat = invertable_diff1_matrix(length)
    diff2_mat = np.matmul(diff_mat, diff_mat)
    return diff2_mat


def invertable_diff1_matrix(length: int) -> np.ndarray:
    """Construct invertable difference matrix of first order difference."""
    eye = np.eye(length)
    diff_mat = np.zeros(shape=(length, length))
    diff_mat[:, :-1] = - np.diff(eye, 1, axis=1)
    diff_mat[-1, -1] = 1
    return diff_mat


def inverse_diff_1d(size: int, **kwargs) -> np.ndarray:
    """Inverse of the first order difference matrix."""
    diff_inv = np.empty(shape=(size, size), **kwargs)
    diff_inv[:, 0] = np.arange(1, size + 1)
    for i in range(1, size):
        diff_inv[:i, i] = 0
        diff_inv[i:, i] = diff_inv[:-i, i - 1]
    return diff_inv
