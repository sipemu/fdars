"""Functional data container and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _fdapy


@dataclass
class FData:
    """Container for functional data.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_points) or (n_samples, m1, m2)
        The functional data matrix. For 2D surfaces, provide a 3D array.
    argvals : array-like or list of array-like, optional
        Evaluation points. For 1D: shape (n_points,).
        For 2D: list of two arrays [s_vals, t_vals].
    rangeval : tuple or list of tuples, optional
        Range of argument values.
    names : dict, optional
        Labels for plotting: {'main': '', 'xlab': 't', 'ylab': 'X(t)'}.
    id : list of str, optional
        Sample identifiers.
    metadata : dict or DataFrame, optional
        Additional covariates per sample.

    Attributes
    ----------
    data : ndarray
        The data matrix in row-major order (n_samples, n_points).
    argvals : ndarray
        Evaluation points.
    rangeval : tuple
        (min, max) of argument domain.
    fdata2d : bool
        Whether this is 2D surface data.
    dims : tuple or None
        Grid dimensions (m1, m2) for 2D data.

    Examples
    --------
    >>> import numpy as np
    >>> from fdapy import FData
    >>> t = np.linspace(0, 1, 100)
    >>> X = np.sin(2 * np.pi * t)
    >>> fd = FData(X.reshape(1, -1), argvals=t)
    >>> print(fd)
    FData(n_samples=1, n_points=100)
    """

    data: NDArray[np.float64]
    argvals: NDArray[np.float64] = field(default=None)
    rangeval: Tuple[float, float] = field(default=None)
    names: Dict[str, str] = field(
        default_factory=lambda: {"main": "", "xlab": "t", "ylab": "X(t)"}
    )
    fdata2d: bool = False
    dims: Optional[Tuple[int, int]] = None
    id: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    # For 2D data
    argvals_s: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    argvals_t: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    def __post_init__(self):
        # Handle array conversion
        self.data = np.atleast_2d(np.asarray(self.data, dtype=np.float64))

        # Handle 3D array (surfaces)
        if self.data.ndim == 3:
            n, m1, m2 = self.data.shape
            self.dims = (m1, m2)
            self.fdata2d = True
            self.data = self.data.reshape(n, m1 * m2)

        # Set default argvals
        if self.argvals is None:
            self.argvals = np.arange(self.data.shape[1], dtype=np.float64)
        else:
            if isinstance(self.argvals, (list, tuple)) and len(self.argvals) == 2:
                # 2D case with separate s and t coordinates
                self.argvals_s = np.asarray(self.argvals[0], dtype=np.float64)
                self.argvals_t = np.asarray(self.argvals[1], dtype=np.float64)
                self.fdata2d = True
                if self.dims is None:
                    self.dims = (len(self.argvals_s), len(self.argvals_t))
                # Create flattened argvals for compatibility
                self.argvals = np.arange(self.data.shape[1], dtype=np.float64)
            else:
                self.argvals = np.asarray(self.argvals, dtype=np.float64)

        # Set default rangeval
        if self.rangeval is None:
            self.rangeval = (float(self.argvals.min()), float(self.argvals.max()))

        # Set default IDs
        if self.id is None:
            self.id = [f"obs_{i}" for i in range(len(self))]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        n, m = self.data.shape
        if self.fdata2d:
            return f"FData(n_samples={n}, dims={self.dims})"
        return f"FData(n_samples={n}, n_points={m})"

    def __getitem__(self, key) -> FData:
        """Subset the functional data."""
        if isinstance(key, int):
            key = [key]
        new_data = self.data[key]

        # Handle index selection for IDs
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
        elif hasattr(key, "__iter__"):
            indices = key
        else:
            indices = [key]

        new_id = [self.id[i] for i in indices] if self.id else None

        return FData(
            data=new_data,
            argvals=self.argvals.copy(),
            rangeval=self.rangeval,
            names=self.names.copy(),
            fdata2d=self.fdata2d,
            dims=self.dims,
            id=new_id,
            argvals_s=self.argvals_s.copy() if self.argvals_s is not None else None,
            argvals_t=self.argvals_t.copy() if self.argvals_t is not None else None,
        )

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.data.shape[0]

    @property
    def n_points(self) -> int:
        """Number of evaluation points."""
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the data matrix."""
        return self.data.shape

    def mean(self) -> NDArray[np.float64]:
        """Compute the mean function.

        Returns
        -------
        mean : ndarray, shape (n_points,)
            Mean function across all samples.
        """
        if self.fdata2d:
            return _fdapy.fdata_mean_2d(self.data, self.dims[0], self.dims[1])
        return _fdapy.fdata_mean_1d(self.data)

    def center(self) -> FData:
        """Center the data by subtracting the mean.

        Returns
        -------
        centered : FData
            Centered functional data.
        """
        centered = _fdapy.fdata_center_1d(self.data)
        return FData(
            data=centered,
            argvals=self.argvals.copy(),
            rangeval=self.rangeval,
            names=self.names.copy(),
            fdata2d=self.fdata2d,
            dims=self.dims,
            id=self.id.copy() if self.id else None,
            argvals_s=self.argvals_s.copy() if self.argvals_s is not None else None,
            argvals_t=self.argvals_t.copy() if self.argvals_t is not None else None,
        )

    def deriv(self, nderiv: int = 1) -> FData:
        """Compute numerical derivative.

        Parameters
        ----------
        nderiv : int, default=1
            Order of derivative.

        Returns
        -------
        deriv : FData
            Derivative of functional data.
        """
        if self.fdata2d:
            result = _fdapy.fdata_deriv_2d(
                self.data, self.argvals_s, self.argvals_t, self.dims[0], self.dims[1]
            )
            # Return first partial derivative by default
            deriv_data = result["ds"]
        else:
            deriv_data = _fdapy.fdata_deriv_1d(self.data, self.argvals, nderiv)

        return FData(
            data=deriv_data,
            argvals=self.argvals.copy(),
            rangeval=self.rangeval,
            names={**self.names, "ylab": f"X^({nderiv})(t)"},
            fdata2d=self.fdata2d,
            dims=self.dims,
            argvals_s=self.argvals_s.copy() if self.argvals_s is not None else None,
            argvals_t=self.argvals_t.copy() if self.argvals_t is not None else None,
        )

    def norm(self, p: float = 2.0) -> NDArray[np.float64]:
        """Compute Lp norm for each sample.

        Parameters
        ----------
        p : float, default=2.0
            Order of the Lp norm.

        Returns
        -------
        norms : ndarray, shape (n_samples,)
            Lp norm for each sample.
        """
        return _fdapy.fdata_norm_lp_1d(self.data, self.argvals, p)

    def geometric_median(
        self, max_iter: int = 1000, tol: float = 1e-6
    ) -> NDArray[np.float64]:
        """Compute the geometric median (L1 median).

        Parameters
        ----------
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance.

        Returns
        -------
        median : ndarray, shape (n_points,)
            Geometric median function.
        """
        if self.fdata2d:
            return _fdapy.geometric_median_2d(
                self.data,
                self.argvals_s,
                self.argvals_t,
                self.dims[0],
                self.dims[1],
                max_iter,
                tol,
            )
        return _fdapy.geometric_median_1d(self.data, self.argvals, max_iter, tol)

    def plot(self, ax=None, **kwargs):
        """Plot the functional data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to matplotlib.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        if ax is None:
            fig, ax = plt.subplots()

        alpha = kwargs.pop("alpha", 0.5)
        for i in range(self.n_samples):
            ax.plot(self.argvals, self.data[i], alpha=alpha, **kwargs)

        ax.set_xlabel(self.names.get("xlab", "t"))
        ax.set_ylabel(self.names.get("ylab", "X(t)"))
        if self.names.get("main"):
            ax.set_title(self.names["main"])

        return ax

    def to_pandas(self):
        """Convert to a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with argvals as columns and sample IDs as index.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        df = pd.DataFrame(self.data, columns=self.argvals, index=self.id)
        return df

    def copy(self) -> FData:
        """Create a deep copy of the FData object."""
        return FData(
            data=self.data.copy(),
            argvals=self.argvals.copy(),
            rangeval=self.rangeval,
            names=self.names.copy(),
            fdata2d=self.fdata2d,
            dims=self.dims,
            id=self.id.copy() if self.id else None,
            metadata=self.metadata.copy() if self.metadata else None,
            argvals_s=self.argvals_s.copy() if self.argvals_s is not None else None,
            argvals_t=self.argvals_t.copy() if self.argvals_t is not None else None,
        )


def fdata(
    data: ArrayLike,
    argvals: Optional[ArrayLike] = None,
    rangeval: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> FData:
    """Create a functional data object (convenience function).

    This mirrors R's fdata() constructor.

    Parameters
    ----------
    data : array-like
        The functional data matrix.
    argvals : array-like, optional
        Evaluation points.
    rangeval : tuple, optional
        Range of argument values.
    **kwargs
        Additional arguments passed to FData.

    Returns
    -------
    fd : FData
        Functional data object.
    """
    return FData(data=data, argvals=argvals, rangeval=rangeval, **kwargs)
