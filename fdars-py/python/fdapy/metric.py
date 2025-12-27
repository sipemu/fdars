"""Distance metrics for functional data."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .fdata import FData
from . import _fdapy


def metric_lp(
    fdata1: FData,
    fdata2: Optional[FData] = None,
    p: float = 2.0,
) -> NDArray[np.float64]:
    """Compute Lp distance matrix.

    Parameters
    ----------
    fdata1 : FData
        First functional data set.
    fdata2 : FData, optional
        Second functional data set. If None, computes self-distances.
    p : float, default=2.0
        Order of the Lp norm.

    Returns
    -------
    D : ndarray
        Distance matrix. Shape (n1, n2) for cross-distances,
        or (n, n) for self-distances.
    """
    if fdata2 is None:
        if fdata1.fdata2d:
            return _fdapy.metric_lp_self_2d(
                fdata1.data, fdata1.argvals_s, fdata1.argvals_t, p
            )
        return _fdapy.metric_lp_self_1d(fdata1.data, fdata1.argvals, p)
    else:
        if fdata1.fdata2d:
            return _fdapy.metric_lp_cross_2d(
                fdata1.data, fdata2.data, fdata1.argvals_s, fdata1.argvals_t, p
            )
        return _fdapy.metric_lp_cross_1d(fdata1.data, fdata2.data, fdata1.argvals, p)


def metric_hausdorff(
    fdata1: FData,
    fdata2: Optional[FData] = None,
) -> NDArray[np.float64]:
    """Compute Hausdorff distance matrix.

    Parameters
    ----------
    fdata1 : FData
        First functional data set.
    fdata2 : FData, optional
        Second functional data set. If None, computes self-distances.

    Returns
    -------
    D : ndarray
        Hausdorff distance matrix.
    """
    if fdata2 is None:
        if fdata1.fdata2d:
            return _fdapy.metric_hausdorff_self_2d(
                fdata1.data, fdata1.argvals_s, fdata1.argvals_t
            )
        return _fdapy.metric_hausdorff_self_1d(fdata1.data, fdata1.argvals)
    else:
        if fdata1.fdata2d:
            return _fdapy.metric_hausdorff_cross_2d(
                fdata1.data, fdata2.data, fdata1.argvals_s, fdata1.argvals_t
            )
        return _fdapy.metric_hausdorff_cross_1d(fdata1.data, fdata2.data, fdata1.argvals)


def metric_dtw(
    fdata1: FData,
    fdata2: Optional[FData] = None,
    p: float = 2.0,
    w: int = 0,
) -> NDArray[np.float64]:
    """Compute Dynamic Time Warping distance matrix.

    Parameters
    ----------
    fdata1 : FData
        First functional data set.
    fdata2 : FData, optional
        Second functional data set. If None, computes self-distances.
    p : float, default=2.0
        Order of the Lp norm for local distances.
    w : int, default=0
        Warping window size. 0 means no constraint.

    Returns
    -------
    D : ndarray
        DTW distance matrix.
    """
    if fdata2 is None:
        return _fdapy.metric_dtw_self_1d(fdata1.data, fdata1.argvals, p, w)
    else:
        return _fdapy.metric_dtw_cross_1d(fdata1.data, fdata2.data, fdata1.argvals, p, w)


def semimetric_fourier(
    fdata1: FData,
    fdata2: Optional[FData] = None,
    nfreq: int = 5,
) -> NDArray[np.float64]:
    """Compute Fourier semimetric distance matrix.

    Parameters
    ----------
    fdata1 : FData
        First functional data set.
    fdata2 : FData, optional
        Second functional data set. If None, computes self-distances.
    nfreq : int, default=5
        Number of Fourier frequencies to use.

    Returns
    -------
    D : ndarray
        Fourier semimetric distance matrix.
    """
    if fdata2 is None:
        return _fdapy.semimetric_fourier_self_1d(fdata1.data, nfreq)
    else:
        return _fdapy.semimetric_fourier_cross_1d(fdata1.data, fdata2.data, nfreq)


def semimetric_hshift(
    fdata1: FData,
    fdata2: Optional[FData] = None,
    max_shift: Optional[float] = None,
) -> NDArray[np.float64]:
    """Compute horizontal shift semimetric distance matrix.

    Parameters
    ----------
    fdata1 : FData
        First functional data set.
    fdata2 : FData, optional
        Second functional data set. If None, computes self-distances.
    max_shift : float, optional
        Maximum allowed shift. If None, uses 25% of the domain range.

    Returns
    -------
    D : ndarray
        Horizontal shift semimetric distance matrix.
    """
    if fdata2 is None:
        return _fdapy.semimetric_hshift_self_1d(fdata1.data, fdata1.argvals, max_shift)
    else:
        return _fdapy.semimetric_hshift_cross_1d(
            fdata1.data, fdata2.data, fdata1.argvals, max_shift
        )
