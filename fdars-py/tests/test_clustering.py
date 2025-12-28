"""Tests for clustering functions."""

import numpy as np
import pytest

from fdapy import FData
from fdapy.clustering import kmeans, fcm


@pytest.fixture
def two_cluster_fdata():
    """Create functional data with two clear clusters."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n_per_cluster = 15
    X = np.zeros((2 * n_per_cluster, 50))

    # Cluster 1: sine waves
    for i in range(n_per_cluster):
        X[i, :] = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(50)

    # Cluster 2: cosine waves (shifted)
    for i in range(n_per_cluster, 2 * n_per_cluster):
        X[i, :] = np.cos(2 * np.pi * t) + 0.1 * np.random.randn(50)

    return FData(X, argvals=t)


@pytest.fixture
def homogeneous_fdata():
    """Create homogeneous functional data."""
    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 20
    X = np.zeros((n, 50))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(50)
    return FData(X, argvals=t)


class TestKMeans:
    """Tests for k-means clustering."""

    def test_returns_dict(self, homogeneous_fdata):
        """K-means returns expected keys."""
        result = kmeans(homogeneous_fdata, n_clusters=2)
        assert "labels" in result
        assert "centers" in result
        assert "inertia" in result

    def test_labels_shape(self, homogeneous_fdata):
        """Labels have correct shape."""
        result = kmeans(homogeneous_fdata, n_clusters=3)
        assert result["labels"].shape == (homogeneous_fdata.n_samples,)

    def test_labels_values(self, homogeneous_fdata):
        """Labels are in valid range."""
        n_clusters = 3
        result = kmeans(homogeneous_fdata, n_clusters=n_clusters)
        assert np.all(result["labels"] >= 0)
        assert np.all(result["labels"] < n_clusters)

    def test_centers_are_fdata(self, homogeneous_fdata):
        """Centers are FData objects."""
        result = kmeans(homogeneous_fdata, n_clusters=2)
        assert isinstance(result["centers"], FData)

    def test_centers_shape(self, homogeneous_fdata):
        """Centers have correct shape."""
        n_clusters = 3
        result = kmeans(homogeneous_fdata, n_clusters=n_clusters)
        assert result["centers"].n_samples == n_clusters
        assert result["centers"].n_points == homogeneous_fdata.n_points

    def test_finds_clusters(self, two_cluster_fdata):
        """K-means finds the two clusters."""
        result = kmeans(two_cluster_fdata, n_clusters=2, n_init=10)
        labels = result["labels"]
        # First half should be one cluster, second half the other
        first_half = labels[:15]
        second_half = labels[15:]
        # Check most of each half is same cluster
        assert np.sum(first_half == first_half[0]) >= 12
        assert np.sum(second_half == second_half[0]) >= 12

    def test_reproducible(self, homogeneous_fdata):
        """Same seed gives same results."""
        result1 = kmeans(homogeneous_fdata, n_clusters=2, seed=123)
        result2 = kmeans(homogeneous_fdata, n_clusters=2, seed=123)
        assert np.array_equal(result1["labels"], result2["labels"])

    def test_inertia_positive(self, homogeneous_fdata):
        """Inertia is non-negative."""
        result = kmeans(homogeneous_fdata, n_clusters=2)
        assert result["inertia"] >= 0


class TestFCM:
    """Tests for fuzzy c-means clustering."""

    def test_returns_dict(self, homogeneous_fdata):
        """FCM returns expected keys."""
        result = fcm(homogeneous_fdata, n_clusters=2)
        assert "labels" in result
        assert "membership" in result
        assert "centers" in result

    def test_labels_shape(self, homogeneous_fdata):
        """Labels have correct shape."""
        result = fcm(homogeneous_fdata, n_clusters=3)
        assert result["labels"].shape == (homogeneous_fdata.n_samples,)

    def test_membership_shape(self, homogeneous_fdata):
        """Membership matrix has correct shape."""
        n_clusters = 3
        result = fcm(homogeneous_fdata, n_clusters=n_clusters)
        assert result["membership"].shape == (homogeneous_fdata.n_samples, n_clusters)

    def test_membership_sums_to_one(self, homogeneous_fdata):
        """Membership rows sum to 1."""
        result = fcm(homogeneous_fdata, n_clusters=3)
        row_sums = np.sum(result["membership"], axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_membership_nonnegative(self, homogeneous_fdata):
        """Membership values are non-negative."""
        result = fcm(homogeneous_fdata, n_clusters=3)
        assert np.all(result["membership"] >= 0)

    def test_centers_are_fdata(self, homogeneous_fdata):
        """Centers are FData objects."""
        result = fcm(homogeneous_fdata, n_clusters=2)
        assert isinstance(result["centers"], FData)

    def test_fuzziness_effect(self, two_cluster_fdata):
        """Higher fuzziness produces more uncertain memberships."""
        result_low = fcm(two_cluster_fdata, n_clusters=2, m=1.5)
        result_high = fcm(two_cluster_fdata, n_clusters=2, m=3.0)
        # Higher m should have membership values closer to 1/n_clusters
        max_low = np.max(result_low["membership"], axis=1)
        max_high = np.max(result_high["membership"], axis=1)
        assert np.mean(max_high) < np.mean(max_low)
