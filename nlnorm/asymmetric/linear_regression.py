"""Asymmetric linear regression."""
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from nlnorm.outliers import q_outliers


class AsymmetricLinearRegression:  # pylint: disable=too-few-public-methods
    """Asymmetric linear regression."""

    def __init__(self,
                 alpha: float = 0.9,
                 tol: float = 1e-6,
                 max_iter: int = 5,
                 outliers: Optional[float] = None):
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
        self.tol = tol
        self.max_iter = max_iter
        self.outliers = outliers

    def predict(self,
                x_data: np.ndarray,
                y_data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_data : np.ndarray
            One-dimensional array.
        y_data : np.ndarray
            One-dimensional array.

        Returns
        -------
        np.ndarray
            Predicted linear regression array.
        """

        weights = np.ones(shape=x_data.shape)
        model = LinearRegression()

        coef_last = None
        for _ in range(self.max_iter):
            lin_reg = model.fit(x_data.reshape(-1, 1),
                                y_data.reshape(-1, 1),
                                sample_weight=weights)
            coef = lin_reg.coef_.squeeze()

            pred = lin_reg.predict(x_data.reshape(-1, 1))
            pred = pred.squeeze()

            if coef_last is not None and abs(coef_last - coef) < self.tol:
                break

            coef_last = coef

            diff = y_data - pred
            weights = np.where(diff > 0, self.alpha, 1 - self.alpha)

            if self.outliers is not None:
                where_outliers = q_outliers(diff, q_margin=self.outliers)
                weights = weights * (1 - where_outliers)

        return pred
