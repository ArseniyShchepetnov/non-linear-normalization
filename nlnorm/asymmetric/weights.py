"""Asymmetric weights functions."""
from typing import Optional

import numpy as np

from nlnorm.outliers import q_outliers


def set_asymmetric_weights(data: np.ndarray,
                           est: np.ndarray,
                           alpha: float,
                           outliers: Optional[float] = None) -> np.ndarray:
    """
    Set asymmetric weights.

    Parameters
    ----------
    data : np.ndarray
        Measured data.
    est : np.ndarray
        Estimated data.
    alpha : float
        Asymmetry factor.
    outliers : Optional[float], optional
        Outliers factor to remove with quantile, by default None.

    Returns
    -------
    np.ndarray
        Asymmetry weights.
    """
    diff = data - est
    weights = np.where(diff > 0, alpha, 1 - alpha)
    if outliers is not None:
        where_outliers = q_outliers(data - est, q_margin=outliers)
        weights = weights * (1 - where_outliers)
    return weights
