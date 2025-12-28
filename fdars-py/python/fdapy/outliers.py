"""Outlier detection for functional data."""

from __future__ import annotations

from typing import Literal

from . import _fdapy
from .fdata import FData

DepthMethod = Literal["FM", "mode", "RP", "RT", "BD", "MBD"]


def outliers_depth(
    fdataobj: FData,
    quantile: float = 0.01,
    depth_method: DepthMethod = "FM",
) -> dict:
    """Depth-based outlier detection.

    Identifies outliers as observations with depth below a threshold
    determined by a quantile of the depth distribution.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    quantile : float, default=0.01
        Quantile threshold for outlier detection.
        Lower values detect fewer outliers.
    depth_method : str, default="FM"
        Depth method to use: "FM", "mode", "RP", "RT", "BD", "MBD".

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'outliers': Boolean array indicating outliers
        - 'depths': Depth values for all observations
        - 'threshold': Depth threshold used
    """
    return _fdapy.outliers_depth(fdataobj.data, quantile, depth_method)


def outliers_lrt(
    fdataobj: FData,
    trim: float = 0.1,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Likelihood ratio test for outlier detection.

    Uses a bootstrap procedure to determine the threshold for the
    likelihood ratio test statistic.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    trim : float, default=0.1
        Trimming proportion for robust estimation.
    alpha : float, default=0.05
        Significance level for the test.
    n_bootstrap : int, default=200
        Number of bootstrap samples for threshold estimation.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'outliers': Boolean array indicating outliers
        - 'statistics': LRT statistics for all observations
        - 'threshold': Critical value from bootstrap
    """
    return _fdapy.outliers_lrt(
        fdataobj.data, fdataobj.argvals, trim, alpha, n_bootstrap, seed
    )
