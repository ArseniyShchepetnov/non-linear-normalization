"""Asymmetric linear regression."""
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from nlnorm.asymmetric.weights import set_asymmetric_weights


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

            weights = set_asymmetric_weights(y_data,
                                             pred,
                                             alpha=self.alpha,
                                             outliers=self.outliers)

        return pred
