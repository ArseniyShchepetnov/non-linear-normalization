"""Asymmetric ridge regression."""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge

from nlnorm.asymmetric.weights import set_asymmetric_weights
from nlnorm.diff_matrix import invertable_diff2_matrix


class AsymmetricRidge:  # pylint: disable=too-few-public-methods
    """Asymmetric ridge regression."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 alpha: float = 0.9,
                 alpha_ridge: float = 0.9,
                 tol: float = 1e-6,
                 max_iter: int = 5,
                 outliers: Optional[float] = None,
                 ridge_params: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        alpha : float, optional
            Asymmetry coefficient, by default 0.9.
        tol : float, optional
            Tolerance of regression coefficient difference, by default 1e-6.
        max_iter : int, optional
            Maximum number of iterations, by default 5.
        outliers : float, optional
            Quantile margin for outliers, by default None.
            None turn off the check.
        """

        self.alpha = alpha
        self.alpha_ridge = alpha_ridge
        self.tol = tol
        self.max_iter = max_iter
        self.outliers = outliers
        self.ridge_params = ridge_params

    def predict(self,
                data: np.ndarray) -> np.ndarray:
        """
        Asymmetric leas squares regression.

        Parameters
        ----------
        data : np.ndarray
            Input signal.

        Returns
        -------
        np.ndarray
            Asymmetry smoothed result.
        """

        if self.ridge_params is None:
            ridge_params = {}
        else:
            ridge_params = self.ridge_params

        diff_inv = np.linalg.inv(invertable_diff2_matrix(len(data)))

        weights = np.ones(shape=(len(data), ))
        ridge = Ridge(alpha=self.alpha_ridge, **ridge_params)

        for _ in range(self.max_iter):
            ridge.fit(diff_inv, data, sample_weight=weights)
            est = ridge.predict(diff_inv)
            weights = set_asymmetric_weights(data,
                                             est,
                                             alpha=self.alpha,
                                             outliers=self.outliers)
        return est
