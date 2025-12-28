"""Basis function representations for functional data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _fdapy
from .fdata import FData

BasisType = Literal["bspline", "fourier"]
Criterion = Literal["GCV", "AIC", "BIC"]


def bspline_basis(
    argvals: ArrayLike,
    nbasis: int,
) -> NDArray[np.float64]:
    """Compute B-spline basis matrix.

    Parameters
    ----------
    argvals : array-like
        Evaluation points.
    nbasis : int
        Number of basis functions.

    Returns
    -------
    basis : ndarray, shape (n_points, nbasis)
        B-spline basis matrix.
    """
    argvals = np.asarray(argvals, dtype=np.float64)
    return _fdapy.bspline_basis(argvals, nbasis)


def fourier_basis(
    argvals: ArrayLike,
    nbasis: int,
    period: float | None = None,
) -> NDArray[np.float64]:
    """Compute Fourier basis matrix.

    Parameters
    ----------
    argvals : array-like
        Evaluation points.
    nbasis : int
        Number of basis functions (should be odd).
    period : float, optional
        Period of the Fourier basis. If None, uses the range of argvals.

    Returns
    -------
    basis : ndarray, shape (n_points, nbasis)
        Fourier basis matrix.
    """
    argvals = np.asarray(argvals, dtype=np.float64)
    if period is None:
        return _fdapy.fourier_basis(argvals, nbasis)
    return _fdapy.fourier_basis_with_period(argvals, nbasis, period)


def fdata_to_basis(
    fdataobj: FData,
    nbasis: int,
    basis_type: BasisType = "bspline",
) -> NDArray[np.float64]:
    """Project functional data to basis coefficients.

    Parameters
    ----------
    fdataobj : FData
        Functional data to project.
    nbasis : int
        Number of basis functions.
    basis_type : str, default="bspline"
        Type of basis: "bspline" or "fourier".

    Returns
    -------
    coefs : ndarray, shape (n_samples, nbasis)
        Basis coefficients.
    """
    return _fdapy.fdata_to_basis_1d(fdataobj.data, fdataobj.argvals, nbasis, basis_type)


def basis_to_fdata(
    coefs: ArrayLike,
    argvals: ArrayLike,
    nbasis: int,
    basis_type: BasisType = "bspline",
) -> FData:
    """Reconstruct functional data from basis coefficients.

    Parameters
    ----------
    coefs : array-like, shape (n_samples, nbasis)
        Basis coefficients.
    argvals : array-like
        Evaluation points for reconstruction.
    nbasis : int
        Number of basis functions.
    basis_type : str, default="bspline"
        Type of basis: "bspline" or "fourier".

    Returns
    -------
    fdataobj : FData
        Reconstructed functional data.
    """
    coefs = np.asarray(coefs, dtype=np.float64)
    argvals = np.asarray(argvals, dtype=np.float64)

    data = _fdapy.basis_to_fdata_1d(coefs, argvals, nbasis, basis_type)
    return FData(data=data, argvals=argvals)


def pspline(
    fdataobj: FData,
    nbasis: int,
    lambda_: float = 1.0,
    nderiv: int = 2,
) -> dict:
    """Fit P-splines to functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data to smooth.
    nbasis : int
        Number of B-spline basis functions.
    lambda_ : float, default=1.0
        Smoothing parameter.
    nderiv : int, default=2
        Order of penalty derivative.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'coefficients': Basis coefficients
        - 'fitted': Fitted functional data
        - 'edf': Effective degrees of freedom
        - 'gcv': GCV scores
        - 'aic': AIC values
        - 'bic': BIC values
    """
    result = _fdapy.pspline_fit_1d(
        fdataobj.data, fdataobj.argvals, nbasis, lambda_, nderiv
    )

    # Convert fitted to FData
    result["fitted"] = FData(
        data=result["fitted"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
    )

    return result


def fourier_fit(
    fdataobj: FData,
    nbasis: int,
) -> dict:
    """Fit Fourier basis to functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data to fit.
    nbasis : int
        Number of Fourier basis functions.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'coefficients': Fourier coefficients
        - 'fitted': Fitted functional data
        - 'edf': Effective degrees of freedom
        - 'gcv': GCV scores
    """
    result = _fdapy.fourier_fit_1d(fdataobj.data, fdataobj.argvals, nbasis)

    # Convert fitted to FData
    result["fitted"] = FData(
        data=result["fitted"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
    )

    return result


def basis_gcv(
    fdataobj: FData,
    nbasis: int,
    basis_type: BasisType = "bspline",
    lambda_: float = 1.0,
) -> dict:
    """Compute GCV score for basis fitting.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    nbasis : int
        Number of basis functions.
    basis_type : str, default="bspline"
        Type of basis.
    lambda_ : float, default=1.0
        Smoothing parameter (for P-splines).

    Returns
    -------
    result : dict
        Dictionary with 'gcv' (array) and 'mean_gcv' (float).
    """
    return _fdapy.basis_gcv_1d(
        fdataobj.data, fdataobj.argvals, nbasis, basis_type, lambda_
    )


def select_basis_auto(
    fdataobj: FData,
    nbasis_range: tuple[int, int] = (5, 30),
    criterion: Criterion = "GCV",
    lambda_: float = 1.0,
) -> dict:
    """Automatically select optimal basis (Fourier vs P-spline).

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    nbasis_range : tuple, default=(5, 30)
        Range of basis sizes to consider (min, max).
    criterion : str, default="GCV"
        Selection criterion: "GCV", "AIC", or "BIC".
    lambda_ : float, default=1.0
        Smoothing parameter for P-splines.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'basis_type': Selected basis type ("bspline" or "fourier")
        - 'nbasis': Optimal number of basis functions
        - 'coefficients': Optimal coefficients
        - 'fitted': Fitted values
        - 'score': Optimal criterion score
        - 'seasonal_detected': Whether seasonality was detected
    """
    nbasis_min, nbasis_max = nbasis_range

    result = _fdapy.select_basis_auto_1d(
        fdataobj.data, fdataobj.argvals, nbasis_min, nbasis_max, criterion, lambda_
    )

    return result
