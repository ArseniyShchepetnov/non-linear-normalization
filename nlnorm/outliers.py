"""Functions for outlier detection."""
import numpy as np


def q_outliers(sig: np.ndarray, q_margin: float = 0.15) -> np.ndarray:
    """
    Outliers by quantile.

    Parameters
    ----------
    sig : np.ndarray
        Input signal.
    **kwargs :
        Arguments to `numpy.quantile`.

    Returns
    -------
    np.ndarray
        Indicator array of the outlier.
    """
    level_up = np.quantile(sig, q=1 - q_margin)
    level_low = np.quantile(sig, q=q_margin)

    result_up = (sig > level_up)
    result_low = (sig < level_low)

    result = (result_up | result_low).astype('int32')

    return result


def correct_weights_outliers(data: np.ndarray,
                             est: np.ndarray,
                             weights: np.ndarray,
                             outliers: float) -> np.ndarray:
    """Correct weights for outliers."""
    where_outliers = q_outliers(data - est, q_margin=outliers)
    weights = weights * (1 - where_outliers)
    return weights
