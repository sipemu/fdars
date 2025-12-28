"""Tests for regression methods."""

import numpy as np
import pytest

from fdapy import FData
from fdapy.regression import fpca, fpls, ridge


@pytest.fixture
def simple_fdata():
    """Create simple functional data."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 20
    X = np.zeros((n, 50))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(50)
    return FData(X, argvals=t)


@pytest.fixture
def fdata_with_response():
    """Create functional data with scalar response."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 25
    X = np.zeros((n, 50))
    y = np.zeros(n)
    for i in range(n):
        coef = np.random.randn()
        X[i, :] = coef * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(50)
        y[i] = coef + 0.1 * np.random.randn()
    return FData(X, argvals=t), y


class TestFPCA:
    """Tests for Functional PCA."""

    def test_returns_dict(self, simple_fdata):
        """FPCA returns expected keys."""
        result = fpca(simple_fdata)
        assert "scores" in result
        assert "components" in result
        assert "mean" in result
        assert "singular_values" in result
        assert "explained_variance_ratio" in result

    def test_scores_shape(self, simple_fdata):
        """Scores have correct shape."""
        result = fpca(simple_fdata)
        n_samples = simple_fdata.n_samples
        assert result["scores"].shape[0] == n_samples

    def test_n_components_limit(self, simple_fdata):
        """n_components limits output."""
        result = fpca(simple_fdata, n_components=3)
        assert result["scores"].shape[1] == 3
        assert result["components"].shape[0] == 3

    def test_singular_values_decreasing(self, simple_fdata):
        """Singular values are in decreasing order."""
        result = fpca(simple_fdata)
        sv = result["singular_values"]
        assert np.all(np.diff(sv) <= 1e-10)

    def test_variance_ratio_sums_to_one(self, simple_fdata):
        """Explained variance ratios sum to 1."""
        result = fpca(simple_fdata)
        ratio_sum = np.sum(result["explained_variance_ratio"])
        assert np.isclose(ratio_sum, 1.0, atol=0.01)

    def test_centered_mean_zero(self, simple_fdata):
        """Centered data has zero mean."""
        result = fpca(simple_fdata)
        centered_mean = result["centered"].mean()
        assert np.allclose(centered_mean, 0.0, atol=1e-10)


class TestFPLS:
    """Tests for Functional PLS."""

    def test_returns_dict(self, fdata_with_response):
        """FPLS returns expected keys."""
        X, y = fdata_with_response
        result = fpls(X, y, n_components=3)
        assert "x_scores" in result
        assert "x_loadings" in result
        assert "weights" in result

    def test_scores_shape(self, fdata_with_response):
        """X scores have correct shape."""
        X, y = fdata_with_response
        n_components = 3
        result = fpls(X, y, n_components=n_components)
        assert result["x_scores"].shape == (X.n_samples, n_components)

    def test_loadings_shape(self, fdata_with_response):
        """X loadings have correct shape."""
        X, y = fdata_with_response
        n_components = 3
        result = fpls(X, y, n_components=n_components)
        assert result["x_loadings"].shape == (n_components, X.n_points)


class TestRidge:
    """Tests for Ridge regression."""

    def test_returns_dict(self):
        """Ridge returns expected keys."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + 0.1 * np.random.randn(50)
        result = ridge(X, y, lambda_=1.0)
        assert "coefficients" in result
        assert "fitted_values" in result
        assert "r_squared" in result

    def test_coefficient_shape(self):
        """Coefficients have correct shape."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + 0.1 * np.random.randn(50)
        result = ridge(X, y, lambda_=1.0)
        assert result["coefficients"].shape == (5,)

    def test_r_squared_range(self):
        """R-squared is in valid range."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + 0.1 * np.random.randn(50)
        result = ridge(X, y, lambda_=1.0)
        assert 0 <= result["r_squared"] <= 1

    def test_regularization_shrinks_coefficients(self):
        """Higher lambda produces smaller coefficients."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + 0.1 * np.random.randn(50)
        result_low = ridge(X, y, lambda_=0.01)
        result_high = ridge(X, y, lambda_=100.0)
        norm_low = np.linalg.norm(result_low["coefficients"])
        norm_high = np.linalg.norm(result_high["coefficients"])
        assert norm_high < norm_low

    def test_intercept(self):
        """Intercept is correctly computed."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = 5.0 + X @ np.array([1, 0, -1]) + 0.1 * np.random.randn(50)
        result = ridge(X, y, lambda_=0.1, fit_intercept=True)
        assert "intercept" in result
        assert np.abs(result["intercept"] - 5.0) < 1.0
