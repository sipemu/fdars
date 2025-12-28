"""Regression methods for functional data."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from . import _fdapy
from .fdata import FData


def fpca(
    fdataobj: FData,
    n_components: int | None = None,
) -> dict:
    """Functional Principal Component Analysis (FPCA).

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    n_components : int, optional
        Number of components to keep. If None, keeps all.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'scores': Principal component scores (n_samples, n_components)
        - 'components': Principal component functions (n_components, n_points)
        - 'mean': Mean function
        - 'explained_variance': Variance explained by each component
        - 'explained_variance_ratio': Proportion of variance explained
        - 'singular_values': Singular values from SVD
        - 'centered': Centered functional data
    """
    result = _fdapy.fdata_to_pc_1d(fdataobj.data, fdataobj.argvals, n_components)

    # Convert centered to FData
    result["centered"] = FData(
        data=result["centered"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
    )

    return result


def fpls(
    X: FData,
    y: ArrayLike,
    n_components: int = 2,
) -> dict:
    """Functional Partial Least Squares (FPLS).

    Parameters
    ----------
    X : FData
        Functional predictor data.
    y : array-like, shape (n_samples,)
        Response variable.
    n_components : int, default=2
        Number of PLS components.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'x_scores': X scores (n_samples, n_components)
        - 'x_loadings': X loadings (n_components, n_points)
        - 'weights': PLS weights
        - 'x_mean': Mean of X
        - 'y_mean': Mean of y
        - 'centered_x': Centered X data
    """
    y = np.asarray(y, dtype=np.float64)

    result = _fdapy.fdata_to_pls_1d(X.data, y, X.argvals, n_components)

    # Convert centered_x to FData
    result["centered_x"] = FData(
        data=result["centered_x"],
        argvals=X.argvals.copy(),
        rangeval=X.rangeval,
    )

    return result


def ridge(
    X: ArrayLike,
    y: ArrayLike,
    lambda_: float = 1.0,
    fit_intercept: bool = True,
) -> dict:
    """Ridge regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Response variable.
    lambda_ : float, default=1.0
        Regularization parameter.
    fit_intercept : bool, default=True
        Whether to fit an intercept.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'coefficients': Regression coefficients
        - 'intercept': Intercept (if fit_intercept=True)
        - 'fitted_values': Fitted values
        - 'residuals': Residuals
        - 'r_squared': R-squared value
        - 'dof': Degrees of freedom
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return _fdapy.ridge_regression_fit(X, y, lambda_, fit_intercept)
