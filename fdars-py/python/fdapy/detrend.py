"""Detrending and decomposition for functional data."""

from __future__ import annotations

from typing import Literal

from . import _fdapy
from .fdata import FData

DetrendMethod = Literal["linear", "polynomial", "diff", "loess", "spline", "auto"]
DecomposeMethod = Literal["additive", "multiplicative"]


def detrend(
    fdataobj: FData,
    method: DetrendMethod = "linear",
    degree: int = 2,
    span: float = 0.75,
    lambda_: float = 1.0,
) -> dict:
    """Detrend functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data to detrend.
    method : str, default="linear"
        Detrending method:
        - "linear": Linear detrending
        - "polynomial": Polynomial detrending
        - "diff": Differencing
        - "loess": LOESS detrending
        - "spline": P-spline detrending
        - "auto": Automatic method selection via AIC
    degree : int, default=2
        Polynomial degree for "polynomial" method.
    span : float, default=0.75
        LOESS span for "loess" method.
    lambda_ : float, default=1.0
        Smoothing parameter for "spline" method.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'detrended': Detrended data as FData
        - 'trend': Estimated trend as FData
        - 'method': Method used
        - 'coefficients': Trend coefficients (for polynomial methods)
        - 'rss': Residual sum of squares
        - 'n_params': Number of parameters
    """
    result = _fdapy.detrend(fdataobj.data, fdataobj.argvals, method, degree, span)

    # Convert to FData objects
    result["detrended"] = FData(
        data=result["detrended"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
        names={**fdataobj.names, "ylab": "Detrended"},
    )

    result["trend"] = FData(
        data=result["trend"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
        names={**fdataobj.names, "ylab": "Trend"},
    )

    return result


def decompose(
    fdataobj: FData,
    period: float,
    method: DecomposeMethod = "additive",
) -> dict:
    """Seasonal decomposition.

    Decomposes functional data into trend, seasonal, and remainder components.

    Parameters
    ----------
    fdataobj : FData
        Functional data to decompose.
    period : float
        Period for seasonal decomposition.
    method : str, default="additive"
        Decomposition method:
        - "additive": Y = Trend + Seasonal + Remainder
        - "multiplicative": Y = Trend * Seasonal * Remainder

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'trend': Trend component as FData
        - 'seasonal': Seasonal component as FData
        - 'remainder': Remainder component as FData
        - 'period': Period used
        - 'method': Method used
    """
    result = _fdapy.decompose(fdataobj.data, fdataobj.argvals, period, method)

    # Convert to FData objects
    result["trend"] = FData(
        data=result["trend"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
        names={**fdataobj.names, "ylab": "Trend"},
    )

    result["seasonal"] = FData(
        data=result["seasonal"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
        names={**fdataobj.names, "ylab": "Seasonal"},
    )

    result["remainder"] = FData(
        data=result["remainder"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
        names={**fdataobj.names, "ylab": "Remainder"},
    )

    return result
