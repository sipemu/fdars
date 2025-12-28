"""Seasonal analysis for functional data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import _fdapy
from .fdata import FData

StrengthMethod = Literal["variance", "spectral"]


def estimate_period(
    fdataobj: FData,
    method: Literal["fft", "autocorr"] = "fft",
    min_period: float | None = None,
    max_period: float | None = None,
) -> dict:
    """Estimate dominant period in functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    method : str, default="fft"
        Method for period estimation: "fft" or "autocorr".
    min_period : float, optional
        Minimum period to consider.
    max_period : float, optional
        Maximum period to consider.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'periods': Estimated periods for each sample
        - 'confidences': Confidence values
        - For FFT method also: 'frequencies', 'powers'
    """
    if method == "fft":
        return _fdapy.estimate_period_fft(
            fdataobj.data, fdataobj.argvals, min_period, max_period
        )
    else:
        return _fdapy.estimate_period_autocorr(
            fdataobj.data, fdataobj.argvals, min_period, max_period
        )


def detect_peaks(
    fdataobj: FData,
    min_prominence: float = 0.1,
    min_distance: float | None = None,
) -> dict:
    """Detect peaks in functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    min_prominence : float, default=0.1
        Minimum peak prominence.
    min_distance : float, optional
        Minimum distance between peaks.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'peak_times': List of peak times for each sample
        - 'peak_values': List of peak values for each sample
        - 'prominences': List of prominences for each sample
        - 'inter_peak_distances': Average inter-peak distances
        - 'mean_period': Estimated mean period from peaks
    """
    return _fdapy.detect_peaks(
        fdataobj.data, fdataobj.argvals, min_prominence, min_distance
    )


def detect_multiple_periods(
    fdataobj: FData,
    max_periods: int = 3,
    min_period: float | None = None,
    max_period: float | None = None,
) -> dict:
    """Detect multiple periods in functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    max_periods : int, default=3
        Maximum number of periods to detect.
    min_period : float, optional
        Minimum period to consider.
    max_period : float, optional
        Maximum period to consider.

    Returns
    -------
    result : dict
        Dictionary with 'periods' containing a list of detected periods
        for each sample, each with period, confidence, strength, amplitude.
    """
    return _fdapy.detect_multiple_periods(
        fdataobj.data, fdataobj.argvals, max_periods, min_period, max_period
    )


def seasonal_strength(
    fdataobj: FData,
    period: float,
    method: StrengthMethod = "variance",
) -> NDArray[np.float64]:
    """Compute seasonal strength.

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    period : float
        Period for seasonal decomposition.
    method : str, default="variance"
        Method for strength computation: "variance" or "spectral".

    Returns
    -------
    strengths : ndarray, shape (n_samples,)
        Seasonal strength for each sample.
    """
    return _fdapy.seasonal_strength(fdataobj.data, fdataobj.argvals, period, method)


def detect_seasonality_changes(
    fdataobj: FData,
    period: float,
    window_size: int | None = None,
    threshold: float = 0.3,
) -> dict:
    """Detect changes in seasonality (onset/cessation).

    Parameters
    ----------
    fdataobj : FData
        Functional data.
    period : float
        Expected period of seasonality.
    window_size : int, optional
        Size of sliding window for strength computation.
    threshold : float, default=0.3
        Threshold for detecting significant changes.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'change_points': List of change points for each sample
        - 'strength_curves': Time-varying strength curves
    """
    return _fdapy.detect_seasonality_changes(
        fdataobj.data, fdataobj.argvals, period, window_size, threshold
    )
