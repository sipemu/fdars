"""Tests for distance metrics."""

import numpy as np
import pytest


class TestMetricFunctions:
    """Tests for metric computation."""

    def test_metric_lp_self(self, sample_fdata_1d):
        from fdapy import metric_lp

        D = metric_lp(sample_fdata_1d)
        n = sample_fdata_1d.n_samples
        assert D.shape == (n, n)
        # Distance matrix should be symmetric
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)

    def test_metric_lp_cross(self, sample_fdata_1d):
        from fdapy import metric_lp

        fd1 = sample_fdata_1d[0:5]
        fd2 = sample_fdata_1d[5:10]
        D = metric_lp(fd1, fd2)
        assert D.shape == (5, 5)

    def test_metric_hausdorff(self, sample_fdata_1d):
        from fdapy import metric_hausdorff

        D = metric_hausdorff(sample_fdata_1d)
        n = sample_fdata_1d.n_samples
        assert D.shape == (n, n)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_metric_dtw(self, sample_fdata_1d):
        from fdapy import metric_dtw

        D = metric_dtw(sample_fdata_1d)
        n = sample_fdata_1d.n_samples
        assert D.shape == (n, n)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_semimetric_fourier(self, sample_fdata_1d):
        from fdapy import semimetric_fourier

        D = semimetric_fourier(sample_fdata_1d, nfreq=5)
        n = sample_fdata_1d.n_samples
        assert D.shape == (n, n)


class TestClusteringFunctions:
    """Tests for clustering."""

    def test_kmeans(self, sample_fdata_1d):
        from fdapy import kmeans

        result = kmeans(sample_fdata_1d, n_clusters=3, n_init=5)
        assert "labels" in result
        assert "centers" in result
        assert "inertia" in result
        assert len(result["labels"]) == sample_fdata_1d.n_samples
        assert result["centers"].n_samples == 3

    def test_fcm(self, sample_fdata_1d):
        from fdapy import fcm

        result = fcm(sample_fdata_1d, n_clusters=3)
        assert "labels" in result
        assert "membership" in result
        assert "centers" in result
        assert len(result["labels"]) == sample_fdata_1d.n_samples
        assert result["membership"].shape == (sample_fdata_1d.n_samples, 3)
