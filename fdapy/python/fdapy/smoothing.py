"""Kernel-based smoothing methods for functional data."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fdata import FData
from . import _fdapy

KernelType = Literal["gaussian", "epanechnikov"]


def smooth_nw(
    fdataobj: FData,
    h: float,
    kernel: KernelType = "gaussian",
    eval_points: Optional[ArrayLike] = None,
) -> FData:
    """Nadaraya-Watson kernel smoother.

    Parameters
    ----------
    fdataobj : FData
        Functional data to smooth.
    h : float
        Bandwidth parameter.
    kernel : str, default="gaussian"
        Kernel type: "gaussian" or "epanechnikov".
    eval_points : array-like, optional
        Points at which to evaluate the smoother.
        If None, uses the original argvals.

    Returns
    -------
    smoothed : FData
        Smoothed functional data.
    """
    if eval_points is not None:
        eval_points = np.asarray(eval_points, dtype=np.float64)

    smoothed = _fdapy.nadaraya_watson(
        fdataobj.data, fdataobj.argvals, h, kernel, eval_points
    )

    new_argvals = eval_points if eval_points is not None else fdataobj.argvals.copy()

    return FData(
        data=smoothed,
        argvals=new_argvals,
        names=fdataobj.names.copy(),
    )


def smooth_llr(
    fdataobj: FData,
    h: float,
    kernel: KernelType = "gaussian",
    eval_points: Optional[ArrayLike] = None,
) -> FData:
    """Local linear regression smoother.

    Parameters
    ----------
    fdataobj : FData
        Functional data to smooth.
    h : float
        Bandwidth parameter.
    kernel : str, default="gaussian"
        Kernel type: "gaussian" or "epanechnikov".
    eval_points : array-like, optional
        Points at which to evaluate the smoother.

    Returns
    -------
    smoothed : FData
        Smoothed functional data.
    """
    if eval_points is not None:
        eval_points = np.asarray(eval_points, dtype=np.float64)

    smoothed = _fdapy.local_linear(
        fdataobj.data, fdataobj.argvals, h, kernel, eval_points
    )

    new_argvals = eval_points if eval_points is not None else fdataobj.argvals.copy()

    return FData(
        data=smoothed,
        argvals=new_argvals,
        names=fdataobj.names.copy(),
    )


def smooth_lpr(
    fdataobj: FData,
    h: float,
    degree: int = 2,
    kernel: KernelType = "gaussian",
    eval_points: Optional[ArrayLike] = None,
) -> FData:
    """Local polynomial regression smoother.

    Parameters
    ----------
    fdataobj : FData
        Functional data to smooth.
    h : float
        Bandwidth parameter.
    degree : int, default=2
        Polynomial degree (0, 1, 2, or 3).
    kernel : str, default="gaussian"
        Kernel type: "gaussian" or "epanechnikov".
    eval_points : array-like, optional
        Points at which to evaluate the smoother.

    Returns
    -------
    smoothed : FData
        Smoothed functional data.
    """
    if eval_points is not None:
        eval_points = np.asarray(eval_points, dtype=np.float64)

    smoothed = _fdapy.local_polynomial(
        fdataobj.data, fdataobj.argvals, h, degree, kernel, eval_points
    )

    new_argvals = eval_points if eval_points is not None else fdataobj.argvals.copy()

    return FData(
        data=smoothed,
        argvals=new_argvals,
        names=fdataobj.names.copy(),
    )
