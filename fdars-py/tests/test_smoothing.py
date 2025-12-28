"""Tests for smoothing functions."""

import numpy as np
import pytest

from fdapy import FData
from fdapy.smoothing import smooth_nw, smooth_llr, smooth_lpr


@pytest.fixture
def noisy_fdata():
    """Create noisy functional data for smoothing tests."""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    n = 10
    X = np.zeros((n, 100))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.3 * np.random.randn(100)
    return FData(X, argvals=t)


@pytest.fixture
def constant_fdata():
    """Create constant functional data."""
    t = np.linspace(0, 1, 50)
    X = np.ones((5, 50)) * 3.0
    return FData(X, argvals=t)


class TestNadarayaWatson:
    """Tests for Nadaraya-Watson smoother."""

    def test_returns_fdata(self, noisy_fdata):
        """NW smoother returns FData object."""
        result = smooth_nw(noisy_fdata, h=0.1)
        assert isinstance(result, FData)

    def test_shape_preserved(self, noisy_fdata):
        """Output shape matches input shape."""
        result = smooth_nw(noisy_fdata, h=0.1)
        assert result.data.shape == noisy_fdata.data.shape

    def test_constant_preserved(self, constant_fdata):
        """Constant data remains constant after smoothing."""
        result = smooth_nw(constant_fdata, h=0.1)
        assert np.allclose(result.data, 3.0, atol=0.1)

    def test_reduces_noise(self, noisy_fdata):
        """Smoothing reduces variance."""
        result = smooth_nw(noisy_fdata, h=0.1)
        var_before = np.var(np.diff(noisy_fdata.data, axis=1))
        var_after = np.var(np.diff(result.data, axis=1))
        assert var_after < var_before

    def test_kernel_options(self, noisy_fdata):
        """Both kernel types work."""
        result_gauss = smooth_nw(noisy_fdata, h=0.1, kernel="gaussian")
        result_epan = smooth_nw(noisy_fdata, h=0.1, kernel="epanechnikov")
        assert result_gauss.data.shape == result_epan.data.shape
        # Results should be different
        assert not np.allclose(result_gauss.data, result_epan.data)


class TestLocalLinear:
    """Tests for local linear regression smoother."""

    def test_returns_fdata(self, noisy_fdata):
        """LLR smoother returns FData object."""
        result = smooth_llr(noisy_fdata, h=0.1)
        assert isinstance(result, FData)

    def test_shape_preserved(self, noisy_fdata):
        """Output shape matches input shape."""
        result = smooth_llr(noisy_fdata, h=0.1)
        assert result.data.shape == noisy_fdata.data.shape

    def test_constant_preserved(self, constant_fdata):
        """Constant data remains constant after smoothing."""
        result = smooth_llr(constant_fdata, h=0.15)
        assert np.allclose(result.data, 3.0, atol=0.1)

    def test_linear_preserved(self):
        """Linear data is preserved by local linear smoother."""
        t = np.linspace(0, 1, 50)
        X = np.zeros((3, 50))
        for i in range(3):
            X[i, :] = 2 * t + 1  # Linear function
        fdata = FData(X, argvals=t)
        result = smooth_llr(fdata, h=0.2)
        # Interior points should be close to original
        expected = 2 * t + 1
        assert np.allclose(result.data[0, 10:40], expected[10:40], atol=0.1)


class TestLocalPolynomial:
    """Tests for local polynomial regression smoother."""

    def test_returns_fdata(self, noisy_fdata):
        """LPR smoother returns FData object."""
        result = smooth_lpr(noisy_fdata, h=0.1, degree=2)
        assert isinstance(result, FData)

    def test_shape_preserved(self, noisy_fdata):
        """Output shape matches input shape."""
        result = smooth_lpr(noisy_fdata, h=0.1, degree=2)
        assert result.data.shape == noisy_fdata.data.shape

    def test_degree_1_similar_to_llr(self, noisy_fdata):
        """Degree 1 polynomial is equivalent to local linear."""
        result_llr = smooth_llr(noisy_fdata, h=0.1, kernel="gaussian")
        result_lpr = smooth_lpr(noisy_fdata, h=0.1, degree=1, kernel="gaussian")
        assert np.allclose(result_llr.data, result_lpr.data, atol=1e-6)

    def test_higher_degree(self, noisy_fdata):
        """Higher degree smoothing works."""
        result = smooth_lpr(noisy_fdata, h=0.15, degree=3)
        assert result.data.shape == noisy_fdata.data.shape
        assert not np.any(np.isnan(result.data))
