"""Clustering algorithms for functional data."""

from __future__ import annotations

from . import _fdapy
from .fdata import FData


def kmeans(
    fdataobj: FData,
    n_clusters: int,
    max_iter: int = 100,
    n_init: int = 10,
    seed: int = 42,
) -> dict:
    """K-means clustering for functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data to cluster.
    n_clusters : int
        Number of clusters.
    max_iter : int, default=100
        Maximum number of iterations.
    n_init : int, default=10
        Number of random initializations.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'labels': Cluster assignments (n_samples,)
        - 'centers': Cluster centers as FData
        - 'inertia': Sum of squared distances to nearest center
        - 'withinss': Within-cluster sum of squares per cluster
        - 'n_iter': Number of iterations
        - 'converged': Whether the algorithm converged
    """
    result = _fdapy.kmeans_fd(
        fdataobj.data, fdataobj.argvals, n_clusters, max_iter, n_init, seed
    )

    # Convert centers to FData
    result["centers"] = FData(
        data=result["centers"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
    )

    return result


def fcm(
    fdataobj: FData,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """Fuzzy C-means clustering for functional data.

    Parameters
    ----------
    fdataobj : FData
        Functional data to cluster.
    n_clusters : int
        Number of clusters.
    m : float, default=2.0
        Fuzziness parameter (must be > 1).
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'labels': Hard cluster assignments (n_samples,)
        - 'membership': Fuzzy membership matrix (n_samples, n_clusters)
        - 'centers': Cluster centers as FData
        - 'inertia': Weighted sum of squared distances
        - 'withinss': Within-cluster sum of squares per cluster
        - 'n_iter': Number of iterations
        - 'converged': Whether the algorithm converged
    """
    result = _fdapy.fcmeans_fd(
        fdataobj.data, fdataobj.argvals, n_clusters, m, max_iter, tol, seed
    )

    # Convert centers to FData
    result["centers"] = FData(
        data=result["centers"],
        argvals=fdataobj.argvals.copy(),
        rangeval=fdataobj.rangeval,
    )

    return result
