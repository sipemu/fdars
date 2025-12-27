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
