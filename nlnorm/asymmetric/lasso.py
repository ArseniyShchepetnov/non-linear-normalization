"""Asymmetric ridge regression."""
from typing import Any, Dict, Optional, Literal

import numpy as np
from sklearn.linear_model import Lasso

from nlnorm.diff_matrix import invertable_diff1_matrix, invertable_diff2_matrix
from nlnorm.outliers import correct_weights_outliers


class AsymmetricLasso:  # pylint: disable=too-few-public-methods
    """Asymmetric lasso regression."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 alpha: float = 0.9,
                 alpha_lasso: float = 0.9,
                 tol: float = 1e-6,
                 max_iter: int = 5,
                 outliers: Optional[float] = None,
                 lasso_params: Optional[Dict[str, Any]] = None,
                 diff_order: Literal[1, 2] = 1):
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
        self.alpha_lasso = alpha_lasso
        self.tol = tol
        self.max_iter = max_iter
        self.outliers = outliers
        self.lasso_params = lasso_params
        self.diff_order = diff_order

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

        if self.lasso_params is None:
            lasso_params = {}
        else:
            lasso_params = self.lasso_params

        if self.diff_order == 1:
            diff_inv = np.linalg.inv(invertable_diff1_matrix(len(data)))
        elif self.diff_order == 2:
            diff_inv = np.linalg.inv(invertable_diff2_matrix(len(data)))
        else:
            raise ValueError(f"Unknown difference order: {self.diff_order}")

        weights = np.ones(shape=(len(data), ))
        lasso = Lasso(alpha=self.alpha_lasso, **lasso_params)

        for _ in range(self.max_iter):

            lasso.fit(diff_inv, data, sample_weight=weights)
            est = lasso.predict(diff_inv)

            weights = (
                self.alpha * (data > est) + (1 - self.alpha) * (data < est)
            )
            if self.outliers is not None:
                weights = correct_weights_outliers(data,
                                                   est,
                                                   weights,
                                                   outliers=self.outliers)

        return est
