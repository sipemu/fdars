"""Tests for basis functions."""

import numpy as np
import pytest

from fdapy import FData
from fdapy.basis import (
    bspline_basis,
    fourier_basis,
    fdata_to_basis,
    basis_to_fdata,
    pspline,
    fourier_fit,
    basis_gcv,
    select_basis_auto,
)


@pytest.fixture
def simple_fdata():
    """Create simple functional data."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 10
    X = np.zeros((n, 50))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.05 * np.random.randn(50)
    return FData(X, argvals=t)


class TestBSplineBasis:
    """Tests for B-spline basis."""

    def test_shape(self):
        """B-spline basis has correct shape."""
        t = np.linspace(0, 1, 100)
        nbasis = 10
        basis = bspline_basis(t, nbasis)
        assert basis.shape == (100, nbasis)

    def test_nonnegative(self):
        """B-spline basis functions are non-negative."""
        t = np.linspace(0, 1, 100)
        basis = bspline_basis(t, 8)
        assert np.all(basis >= -1e-10)

    def test_partition_of_unity(self):
        """B-spline basis sums to 1 at each point."""
        t = np.linspace(0, 1, 100)
        basis = bspline_basis(t, 10)
        row_sums = np.sum(basis, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


class TestFourierBasis:
    """Tests for Fourier basis."""

    def test_shape(self):
        """Fourier basis has correct shape."""
        t = np.linspace(0, 1, 100)
        nbasis = 11
        basis = fourier_basis(t, nbasis)
        assert basis.shape == (100, nbasis)

    def test_orthogonality(self):
        """Fourier basis is approximately orthogonal."""
        t = np.linspace(0, 1, 200)
        basis = fourier_basis(t, 5)
        # Compute approximate inner products
        gram = basis.T @ basis / len(t)
        # Diagonal should dominate
        diag = np.diag(gram)
        assert np.all(diag > 0.3)


class TestBasisProjection:
    """Tests for basis projection."""

    def test_projection_shape(self, simple_fdata):
        """Basis projection has correct shape."""
        coefs = fdata_to_basis(simple_fdata, nbasis=8, basis_type="bspline")
        assert coefs.shape == (simple_fdata.n_samples, 8)

    def test_roundtrip(self, simple_fdata):
        """Basis roundtrip approximately preserves data."""
        nbasis = 15
        coefs = fdata_to_basis(simple_fdata, nbasis=nbasis, basis_type="bspline")
        reconstructed = basis_to_fdata(
            coefs, simple_fdata.argvals, nbasis, basis_type="bspline"
        )
        # Reconstruction error should be small
        error = np.mean((simple_fdata.data - reconstructed.data) ** 2)
        assert error < 0.1


class TestPSpline:
    """Tests for P-spline smoothing."""

    def test_returns_dict(self, simple_fdata):
        """P-spline returns expected keys."""
        result = pspline(simple_fdata, nbasis=12, lambda_=1.0)
        assert "coefficients" in result
        assert "fitted" in result
        assert "gcv" in result

    def test_fitted_is_fdata(self, simple_fdata):
        """Fitted result is FData object."""
        result = pspline(simple_fdata, nbasis=12, lambda_=1.0)
        assert isinstance(result["fitted"], FData)

    def test_lambda_effect(self, simple_fdata):
        """Higher lambda produces smoother fits."""
        result_low = pspline(simple_fdata, nbasis=15, lambda_=0.01)
        result_high = pspline(simple_fdata, nbasis=15, lambda_=100.0)
        # Higher lambda should produce lower variance in fitted values
        var_low = np.var(np.diff(result_low["fitted"].data, axis=1))
        var_high = np.var(np.diff(result_high["fitted"].data, axis=1))
        assert var_high < var_low


class TestFourierFit:
    """Tests for Fourier fitting."""

    def test_returns_dict(self, simple_fdata):
        """Fourier fit returns expected keys."""
        result = fourier_fit(simple_fdata, nbasis=11)
        assert "coefficients" in result
        assert "fitted" in result

    def test_fitted_is_fdata(self, simple_fdata):
        """Fitted result is FData object."""
        result = fourier_fit(simple_fdata, nbasis=11)
        assert isinstance(result["fitted"], FData)


class TestSelectBasisAuto:
    """Tests for automatic basis selection."""

    def test_returns_dict(self, simple_fdata):
        """Auto selection returns expected keys."""
        result = select_basis_auto(simple_fdata, nbasis_range=(5, 15))
        assert "basis_type" in result
        assert "nbasis" in result

    def test_nbasis_in_range(self, simple_fdata):
        """Selected nbasis is within specified range."""
        result = select_basis_auto(simple_fdata, nbasis_range=(5, 15))
        assert 5 <= result["nbasis"] <= 15

    def test_criterion_options(self, simple_fdata):
        """Different criteria can be used."""
        for criterion in ["GCV", "AIC", "BIC"]:
            result = select_basis_auto(
                simple_fdata, nbasis_range=(5, 10), criterion=criterion
            )
            assert result["nbasis"] >= 5
