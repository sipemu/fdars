"""Tests for FData class and operations."""

import numpy as np
import pytest


class TestFDataCreation:
    """Tests for FData creation."""

    def test_create_from_array(self):
        from fdapy import FData

        X = np.random.randn(10, 100)
        fd = FData(X)
        assert fd.n_samples == 10
        assert fd.n_points == 100

    def test_create_with_argvals(self):
        from fdapy import FData

        X = np.random.randn(10, 100)
        t = np.linspace(0, 1, 100)
        fd = FData(X, argvals=t)
        assert fd.n_samples == 10
        assert fd.n_points == 100
        np.testing.assert_array_equal(fd.argvals, t)

    def test_create_1d_from_vector(self):
        from fdapy import FData

        x = np.random.randn(100)
        fd = FData(x)
        assert fd.n_samples == 1
        assert fd.n_points == 100

    def test_rangeval_auto(self):
        from fdapy import FData

        t = np.linspace(0, 2 * np.pi, 100)
        fd = FData(np.sin(t).reshape(1, -1), argvals=t)
        assert fd.rangeval == (0, 2 * np.pi)

    def test_default_ids(self):
        from fdapy import FData

        fd = FData(np.random.randn(5, 50))
        assert len(fd.id) == 5
        assert fd.id[0] == "obs_0"


class TestFDataOperations:
    """Tests for FData operations."""

    def test_mean(self, sample_fdata_1d):
        mean = sample_fdata_1d.mean()
        assert mean.shape == (sample_fdata_1d.n_points,)

    def test_center(self, sample_fdata_1d):
        centered = sample_fdata_1d.center()
        assert centered.n_samples == sample_fdata_1d.n_samples
        # Mean of centered data should be close to zero
        centered_mean = centered.mean()
        np.testing.assert_allclose(centered_mean, 0, atol=1e-10)

    def test_deriv(self, sample_fdata_1d):
        deriv = sample_fdata_1d.deriv(nderiv=1)
        assert deriv.n_samples == sample_fdata_1d.n_samples
        assert deriv.n_points == sample_fdata_1d.n_points

    def test_norm(self, sample_fdata_1d):
        norms = sample_fdata_1d.norm(p=2.0)
        assert norms.shape == (sample_fdata_1d.n_samples,)
        assert np.all(norms >= 0)

    def test_geometric_median(self, sample_fdata_1d):
        median = sample_fdata_1d.geometric_median()
        assert median.shape == (sample_fdata_1d.n_points,)


class TestFDataIndexing:
    """Tests for FData indexing."""

    def test_single_index(self, sample_fdata_1d):
        fd_single = sample_fdata_1d[0]
        assert fd_single.n_samples == 1

    def test_slice(self, sample_fdata_1d):
        fd_slice = sample_fdata_1d[0:5]
        assert fd_slice.n_samples == 5

    def test_list_index(self, sample_fdata_1d):
        fd_subset = sample_fdata_1d[[0, 2, 4]]
        assert fd_subset.n_samples == 3

    def test_len(self, sample_fdata_1d):
        assert len(sample_fdata_1d) == sample_fdata_1d.n_samples


class TestFDataCopy:
    """Tests for FData copy."""

    def test_copy(self, sample_fdata_1d):
        fd_copy = sample_fdata_1d.copy()
        assert fd_copy.n_samples == sample_fdata_1d.n_samples
        # Modifying copy shouldn't affect original
        fd_copy.data[0, 0] = 999
        assert sample_fdata_1d.data[0, 0] != 999
