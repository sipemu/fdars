"""Utility functions for functional data analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fdata import FData
from . import _fdapy


def integrate(
    fdataobj: FData,
) -> NDArray[np.float64]:
    """Integrate functional data using Simpson's rule.

    Parameters
    ----------
    fdataobj : FData
        Functional data to integrate.

    Returns
    -------
    integrals : ndarray, shape (n_samples,)
        Integral values for each sample.
    """
    return _fdapy.integrate_simpson(fdataobj.data, fdataobj.argvals)


def inner_product(
    fdata1: FData,
    fdata2: FData,
) -> NDArray[np.float64]:
    """Compute inner product between two functional datasets.

    Parameters
    ----------
    fdata1 : FData
        First functional data set, shape (n1, n_points).
    fdata2 : FData
        Second functional data set, shape (n2, n_points).

    Returns
    -------
    inner_prods : ndarray, shape (n1, n2)
        Inner product matrix.
    """
    return _fdapy.inner_product(fdata1.data, fdata2.data, fdata1.argvals)


def gram_matrix(
    fdataobj: FData,
) -> NDArray[np.float64]:
    """Compute Gram matrix (symmetric inner product matrix).

    Parameters
    ----------
    fdataobj : FData
        Functional data.

    Returns
    -------
    gram : ndarray, shape (n_samples, n_samples)
        Gram matrix where gram[i,j] = <f_i, f_j>.
    """
    return _fdapy.inner_product_matrix(fdataobj.data, fdataobj.argvals)
