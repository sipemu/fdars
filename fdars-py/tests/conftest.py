"""Test fixtures for fdapy."""

import numpy as np
import pytest


@pytest.fixture
def sample_fdata_1d():
    """Create sample 1D functional data."""
    from fdapy import FData

    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    n = 20
    X = np.zeros((n, 100))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100)
    return FData(X, argvals=t)


@pytest.fixture
def sample_fdata_2d():
    """Create sample 2D functional data (surfaces)."""
    from fdapy import FData

    np.random.seed(42)
    s = np.linspace(0, 1, 20)
    t = np.linspace(0, 1, 20)
    n = 5
    X = np.zeros((n, 20, 20))
    for i in range(n):
        X[i] = np.outer(np.sin(2 * np.pi * s), np.cos(2 * np.pi * t))
        X[i] += 0.1 * np.random.randn(20, 20)
    return FData(X, argvals=[s, t])


@pytest.fixture
def sample_argvals():
    """Sample evaluation points."""
    return np.linspace(0, 1, 100)


@pytest.fixture
def seasonal_fdata():
    """Create sample functional data with clear seasonality."""
    from fdapy import FData

    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)  # 2 full periods
    n = 10
    X = np.zeros((n, 200))
    for i in range(n):
        X[i, :] = np.sin(t) + 0.1 * np.random.randn(200)
    return FData(X, argvals=t)


@pytest.fixture
def noisy_fdata():
    """Create noisy functional data for smoothing tests."""
    from fdapy import FData

    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    n = 10
    X = np.zeros((n, 100))
    for i in range(n):
        X[i, :] = np.sin(2 * np.pi * t) + 0.3 * np.random.randn(100)
    return FData(X, argvals=t)


@pytest.fixture
def polynomial_fdata():
    """Create polynomial functional data for basis tests."""
    from fdapy import FData

    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n = 8
    X = np.zeros((n, 50))
    for i in range(n):
        coef = np.random.randn(4)
        for j, c in enumerate(coef):
            X[i, :] += c * (t**j)
        X[i, :] += 0.05 * np.random.randn(50)
    return FData(X, argvals=t)


@pytest.fixture
def clustered_fdata():
    """Create functional data with clear cluster structure."""
    from fdapy import FData

    np.random.seed(42)
    t = np.linspace(0, 1, 50)
    n_per_cluster = 10
    n_clusters = 3
    X = np.zeros((n_per_cluster * n_clusters, 50))

    for k in range(n_clusters):
        phase = k * 2 * np.pi / n_clusters
        for i in range(n_per_cluster):
            idx = k * n_per_cluster + i
            X[idx, :] = np.sin(2 * np.pi * t + phase) + 0.1 * np.random.randn(50)

    return FData(X, argvals=t)
