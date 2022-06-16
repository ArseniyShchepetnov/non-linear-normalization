"""Asymmetric ridge regression."""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge

from nlnorm.diff_matrix import inverse_diff_1d, invertable_diff1_matrix, invertable_diff2_matrix
from nlnorm.outliers import q_outliers


class AsymmetricRidge:
    """Asymmetric ridge regression."""

    def __init__(self,
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

        Dinv = np.linalg.inv(invertable_diff2_matrix(len(data)))

        weights = np.ones(shape=(len(data), ))
        ridge = Ridge(alpha=self.alpha_ridge, **ridge_params)

        for _ in range(self.max_iter):

            ridge.fit(Dinv, data, sample_weight=weights)
            est = ridge.predict(Dinv)

            weights = (
                self.alpha * (data > est) + (1 - self.alpha) * (data < est)
            )
            weights = self.correct_weights_outliers(data, est, weights)

        return est

    def correct_weights_outliers(self,
                                 data: np.ndarray,
                                 est: np.ndarray,
                                 weights: np.ndarray) -> np.ndarray:
        """Correct weights for outliers."""
        if self.outliers is not None:
            where_outliers = q_outliers(data - est, q_margin=self.outliers)
            weights = weights * (1 - where_outliers)
        return weights
