"""Total variation denoising."""
import numpy as np
import scipy.sparse as sp

from asym_tvd.outliers import q_outliers


def tvd(y: np.ndarray,
        n_iter: int = 5,
        lam: float = 1,
        tol: float = 0.0001) -> np.ndarray:
    """
    Total variation denoising.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    n_iter : int, optional
        Number of iterations, by default 5
    lam : float, optional
        Total variation term contribution, by default 1.
    tol : float, optional
        Tolerance, by default 0.0001

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    # pylint: disable=invalid-name

    cost = np.zeros(shape=(n_iter, ))

    l = y.size
    I = np.eye(l)
    D = sp.csc_matrix(np.diff(I, 1)).transpose()
    DT = D.transpose()
    DDT = D.dot(DT)
    x = y.copy()
    Dx = D.dot(x)
    Dy = D.dot(y)

    for k in range(n_iter):

        F = sp.csc_matrix(np.diag(np.abs(Dx) / lam) + DDT)
        Finv = sp.linalg.inv(F)
        x = y - DT.dot(Finv.dot(Dy))
        Dx = D.dot(x)

        diff = np.abs(x - y)
        cost[k] = 0.5 * np.sum(diff * diff) + lam * np.sum(np.abs(Dx))

        if tol >= cost[k] - cost[k - 1]:
            break

    return x, cost


def tvd_linear(y: np.ndarray,
               n_iter: int = 5,
               lam: float = 1) -> np.ndarray:
    """
    Linear total variation. Peice-wise linear result.

    Parameters
    ----------
    y : np.ndarray
        Input signal
    n_iter : int, optional
        Number of iteratioins, by default 5
    lam : float, optional
        Total variation contribution, by default 1

    Returns
    -------
    np.ndarray
        Peice-wise linear filtered input signal.
    """
    # pylint: disable=invalid-name

    cost = np.zeros(shape=(n_iter, ))

    l = y.size
    I = np.eye(l)
    D = np.diff(np.diff(I, 1,), 1).transpose()

    DT = D.transpose()
    DDT = np.matmul(D, DT)

    x = y.copy()
    Dx = np.matmul(D, x)
    Dy = np.matmul(D, y)

    for k in range(n_iter):
        F = np.diag(np.abs(Dx) / lam) + DDT
        Finv = np.linalg.inv(F)
        x = y - np.matmul(DT, np.matmul(Finv, Dy))
        Dx = np.matmul(D, x)
        cost[k] = 0.5*np.sum(np.power(np.abs(x-y), 2)) + lam*np.sum(np.abs(Dx))

    return x, cost


def tvd_linear_w_weights(y: np.ndarray,
                         w: np.ndarray,
                         x: Optional[np.ndarray] = None,
                         n_iter: int = 5,
                         lam: float = 1) -> np.ndarray:
    """
    Peice-wise linear total variation with weight input.

    TODO: Solve second difference outliers high power problem which
    causes instability.

    Parameters
    ----------
    y : np.ndarray
        Input signal
    w : np.ndarray
        Input weights
    x : np.ndarray, optional
        Initial approximation, by default None.
    n_iter : int, optional
        Number of iterations, by default 5
    lam : float, optional
        Total variation contributions, by default 1

    Returns
    -------
    np.ndarray
        [description]
    """
    # pylint: disable=invalid-name

    cost = np.zeros(shape=(n_iter, ))

    l = y.size
    I = np.eye(l)
    D = np.diff(np.diff(I, 1), 1).transpose()

    DT = D.transpose()
    WDT = np.matmul(np.diag(1 / w), DT)
    DWDT = np.matmul(D, WDT)

    if x is None:
        x = y.copy()

    Dx = np.matmul(D, x)
    Dy = np.matmul(D, y)

    for k in range(n_iter):

        F = sp.csc_matrix(np.diag(np.abs(Dx) / lam) + DWDT)

        Finv = sp.linalg.inv(F)

        x = y - WDT.dot(Finv.dot(Dy))
        Dx = D.dot(x)

        diff = np.power(np.abs(x-y), 2)
        cost[k] = 0.5 * np.sum(w * diff) + lam * np.sum(np.abs(Dx))

    return x, cost


def asym_tvd_linear(y: np.ndarray,
                    p_asym: float = 1,
                    n_iter: int = 5,
                    n_tvd_iter: int = 5,
                    lam: float = 1,
                    outliers: Optional[float] = None) -> np.ndarray:
    """
    Asymmetric peice-wise linear total variation filtering.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    p_asym : float, optional
        Asymmetry factor, by default 1
    n_iter : int, optional
        Number of asymmetry iterations, by default 5
    n_tvd_iter : int, optional
        Number of inner total variation iterations, by default 5
    lam : float, optional
        Total variation contribution, by default 1
    outliers : float, optional
        Quantile margin for outliers, by default None.
        None turn off the check.

    Returns
    -------
    np.ndarray
        Piece-wise linear asymmetric result.
    """
    # pylint: disable=invalid-name
    w = np.ones(shape=(y.size, ))
    x = None

    for _ in range(n_iter):

        x, _ = tvd_linear_w_weights(y, w, x=x, n_iter=n_tvd_iter, lam=lam)

        w = p_asym * (y > x) + (1 - p_asym) * (y < x)

        if outliers is not None:
            w = w * (1 - q_outliers(y - x, q_margin=outliers))

    return x
