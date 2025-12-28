"""Tests for outlier detection functions."""

import numpy as np
import pytest

from fdapy import FData
from fdapy.outliers import outliers_depth, outliers_lrt


@pytest.fixture
def fdata_with_outlier():
    """Create functional data with obvious outliers."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 20
    X = np.zeros((n, 50))

    # Normal curves
    for i in range(n - 2):
        X[i, :] = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(50)

    # Outliers (shifted up)
    X[n - 2, :] = np.sin(2 * np.pi * t) + 5.0
    X[n - 1, :] = np.sin(2 * np.pi * t) + 5.0

    return FData(X, argvals=t)


@pytest.fixture
def homogeneous_fdata():
    """Create homogeneous functional data without outliers."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 20
    X = np.zeros((n, 50))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.05 * np.random.randn(50)
    return FData(X, argvals=t)


class TestOutliersDepth:
    """Tests for depth-based outlier detection."""

    def test_returns_dict(self, homogeneous_fdata):
        """Returns expected keys."""
        result = outliers_depth(homogeneous_fdata)
        assert "outliers" in result
        assert "depths" in result
        assert "threshold" in result

    def test_outliers_shape(self, homogeneous_fdata):
        """Outliers array has correct shape."""
        result = outliers_depth(homogeneous_fdata)
        assert result["outliers"].shape == (homogeneous_fdata.n_samples,)

    def test_depths_shape(self, homogeneous_fdata):
        """Depths array has correct shape."""
        result = outliers_depth(homogeneous_fdata)
        assert result["depths"].shape == (homogeneous_fdata.n_samples,)

    def test_depths_in_range(self, homogeneous_fdata):
        """Depths are in valid range [0, 1]."""
        result = outliers_depth(homogeneous_fdata, depth_method="FM")
        assert np.all(result["depths"] >= 0)
        assert np.all(result["depths"] <= 1)

    def test_finds_obvious_outlier(self, fdata_with_outlier):
        """Detects obvious outliers."""
        result = outliers_depth(fdata_with_outlier, quantile=0.1)
        # Last two curves should be detected as outliers
        assert result["outliers"][-1] or result["outliers"][-2]

    def test_depth_methods(self, homogeneous_fdata):
        """Different depth methods work."""
        for method in ["FM", "mode", "BD", "MBD"]:
            result = outliers_depth(homogeneous_fdata, depth_method=method)
            assert "outliers" in result

    def test_quantile_effect(self, homogeneous_fdata):
        """Higher quantile detects more outliers."""
        result_low = outliers_depth(homogeneous_fdata, quantile=0.01)
        result_high = outliers_depth(homogeneous_fdata, quantile=0.2)
        n_low = np.sum(result_low["outliers"])
        n_high = np.sum(result_high["outliers"])
        assert n_high >= n_low


class TestOutliersLRT:
    """Tests for LRT-based outlier detection."""

    def test_returns_dict(self, homogeneous_fdata):
        """Returns expected keys."""
        result = outliers_lrt(homogeneous_fdata, n_bootstrap=50)
        assert "outliers" in result
        assert "statistics" in result
        assert "threshold" in result

    def test_outliers_shape(self, homogeneous_fdata):
        """Outliers array has correct shape."""
        result = outliers_lrt(homogeneous_fdata, n_bootstrap=50)
        assert result["outliers"].shape == (homogeneous_fdata.n_samples,)

    def test_statistics_shape(self, homogeneous_fdata):
        """Statistics array has correct shape."""
        result = outliers_lrt(homogeneous_fdata, n_bootstrap=50)
        assert result["statistics"].shape == (homogeneous_fdata.n_samples,)

    def test_reproducible(self, homogeneous_fdata):
        """Same seed gives same results."""
        result1 = outliers_lrt(homogeneous_fdata, n_bootstrap=50, seed=123)
        result2 = outliers_lrt(homogeneous_fdata, n_bootstrap=50, seed=123)
        assert np.array_equal(result1["outliers"], result2["outliers"])

    def test_threshold_positive(self, homogeneous_fdata):
        """Threshold is positive."""
        result = outliers_lrt(homogeneous_fdata, n_bootstrap=50)
        assert result["threshold"] > 0
