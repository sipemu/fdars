"""
fdapy - Functional Data Analysis in Python

A Python package for functional data analysis, providing tools for:
- Functional data representation and manipulation
- Depth measures for functional data
- Distance metrics and semimetrics
- Basis representations (B-splines, Fourier)
- Clustering (k-means, fuzzy c-means)
- Regression (PCA, PLS, ridge)
- Smoothing (kernel, local polynomial)
- Seasonal analysis
- Outlier detection
"""

from fdapy.fdata import FData, fdata
from fdapy.depth import (
    depth,
    depth_fm,
    depth_mode,
    depth_rp,
    depth_rt,
    depth_bd,
    depth_mbd,
    depth_mei,
    depth_fsd,
    depth_kfsd,
)
from fdapy.metric import (
    metric_lp,
    metric_hausdorff,
    metric_dtw,
    semimetric_fourier,
    semimetric_hshift,
)
from fdapy.basis import (
    fdata_to_basis,
    basis_to_fdata,
    pspline,
    fourier_fit,
    basis_gcv,
    select_basis_auto,
    bspline_basis,
    fourier_basis,
)
from fdapy.clustering import kmeans, fcm
from fdapy.regression import fpca, fpls, ridge
from fdapy.smoothing import smooth_nw, smooth_llr, smooth_lpr
from fdapy.seasonal import (
    estimate_period,
    detect_peaks,
    seasonal_strength,
    detect_multiple_periods,
    detect_seasonality_changes,
)
from fdapy.detrend import detrend, decompose
from fdapy.outliers import outliers_depth, outliers_lrt
from fdapy.utility import integrate, inner_product, gram_matrix

__version__ = "0.1.0"

__all__ = [
    # Core
    "FData",
    "fdata",
    # Depth
    "depth",
    "depth_fm",
    "depth_mode",
    "depth_rp",
    "depth_rt",
    "depth_bd",
    "depth_mbd",
    "depth_mei",
    "depth_fsd",
    "depth_kfsd",
    # Metrics
    "metric_lp",
    "metric_hausdorff",
    "metric_dtw",
    "semimetric_fourier",
    "semimetric_hshift",
    # Basis
    "fdata_to_basis",
    "basis_to_fdata",
    "pspline",
    "fourier_fit",
    "basis_gcv",
    "select_basis_auto",
    "bspline_basis",
    "fourier_basis",
    # Clustering
    "kmeans",
    "fcm",
    # Regression
    "fpca",
    "fpls",
    "ridge",
    # Smoothing
    "smooth_nw",
    "smooth_llr",
    "smooth_lpr",
    # Seasonal
    "estimate_period",
    "detect_peaks",
    "seasonal_strength",
    "detect_multiple_periods",
    "detect_seasonality_changes",
    # Detrend
    "detrend",
    "decompose",
    # Outliers
    "outliers_depth",
    "outliers_lrt",
    # Utility
    "integrate",
    "inner_product",
    "gram_matrix",
]
