"""Depth measures for functional data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import _fdapy
from .fdata import FData

DepthMethod = Literal["FM", "mode", "RP", "RT", "BD", "MBD", "MEI", "FSD", "KFSD"]


def depth(
    fdataobj: FData,
    fdataori: FData | None = None,
    method: DepthMethod = "FM",
    **kwargs,
) -> NDArray[np.float64]:
    """Compute functional data depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.
    method : str, default="FM"
        Depth method: "FM", "mode", "RP", "RT", "BD", "MBD",
        "MEI", "FSD", "KFSD".
    **kwargs
        Additional arguments passed to the specific method.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        Depth values for each sample.

    Examples
    --------
    >>> from fdapy import FData, depth
    >>> import numpy as np
    >>> fd = FData(np.random.randn(10, 100))
    >>> d = depth(fd, method="FM")
    """
    if fdataori is None:
        fdataori = fdataobj

    dispatch = {
        "FM": depth_fm,
        "mode": depth_mode,
        "RP": depth_rp,
        "RT": depth_rt,
        "BD": depth_bd,
        "MBD": depth_mbd,
        "MEI": depth_mei,
        "FSD": depth_fsd,
        "KFSD": depth_kfsd,
    }

    if method not in dispatch:
        raise ValueError(
            f"Unknown depth method: {method}. Choose from: {list(dispatch.keys())}"
        )

    return dispatch[method](fdataobj, fdataori, **kwargs)


def depth_fm(
    fdataobj: FData,
    fdataori: FData | None = None,
    scale: bool = True,
) -> NDArray[np.float64]:
    """Compute Fraiman-Muniz depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.
    scale : bool, default=True
        Whether to scale by integral of weights.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        FM depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    if fdataobj.fdata2d:
        return _fdapy.depth_fm_2d(fdataobj.data, fdataori.data, scale)
    return _fdapy.depth_fm_1d(fdataobj.data, fdataori.data, scale)


def depth_mode(
    fdataobj: FData,
    fdataori: FData | None = None,
    h: float | None = None,
) -> NDArray[np.float64]:
    """Compute Modal depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.
    h : float, optional
        Bandwidth parameter. If None, uses Silverman's rule.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        Modal depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    # Silverman's rule if h not provided
    if h is None:
        n = fdataori.n_samples
        h = max(1.06 * np.std(fdataori.data) * n ** (-0.2), 0.1)

    if fdataobj.fdata2d:
        return _fdapy.depth_mode_2d(fdataobj.data, fdataori.data, h)
    return _fdapy.depth_mode_1d(fdataobj.data, fdataori.data, h)


def depth_rp(
    fdataobj: FData,
    fdataori: FData | None = None,
    n_projections: int = 50,
) -> NDArray[np.float64]:
    """Compute Random Projection depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.
    n_projections : int, default=50
        Number of random projections.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        RP depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    if fdataobj.fdata2d:
        return _fdapy.depth_rp_2d(fdataobj.data, fdataori.data, n_projections)
    return _fdapy.depth_rp_1d(fdataobj.data, fdataori.data, n_projections)


def depth_rt(
    fdataobj: FData,
    fdataori: FData | None = None,
    n_projections: int = 50,
) -> NDArray[np.float64]:
    """Compute Random Tukey depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.
    n_projections : int, default=50
        Number of random projections.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        RT depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    if fdataobj.fdata2d:
        return _fdapy.depth_rt_2d(fdataobj.data, fdataori.data, n_projections)
    return _fdapy.depth_rt_1d(fdataobj.data, fdataori.data, n_projections)


def depth_fsd(
    fdataobj: FData,
    fdataori: FData | None = None,
) -> NDArray[np.float64]:
    """Compute Functional Spatial depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        FSD depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    if fdataobj.fdata2d:
        return _fdapy.depth_fsd_2d(fdataobj.data, fdataori.data)
    return _fdapy.depth_fsd_1d(fdataobj.data, fdataori.data)


def depth_kfsd(
    fdataobj: FData,
    fdataori: FData | None = None,
    h: float | None = None,
) -> NDArray[np.float64]:
    """Compute Kernel Functional Spatial depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.
    h : float, optional
        Bandwidth parameter.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        KFSD depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    if fdataobj.fdata2d:
        return _fdapy.depth_kfsd_2d(fdataobj.data, fdataori.data, h)
    return _fdapy.depth_kfsd_1d(fdataobj.data, fdataori.data, fdataobj.argvals, h)


def depth_bd(
    fdataobj: FData,
    fdataori: FData | None = None,
) -> NDArray[np.float64]:
    """Compute Band depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        Band depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    return _fdapy.depth_bd_1d(fdataobj.data, fdataori.data)


def depth_mbd(
    fdataobj: FData,
    fdataori: FData | None = None,
) -> NDArray[np.float64]:
    """Compute Modified Band depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        MBD depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    return _fdapy.depth_mbd_1d(fdataobj.data, fdataori.data)


def depth_mei(
    fdataobj: FData,
    fdataori: FData | None = None,
) -> NDArray[np.float64]:
    """Compute Modified Epigraph Index depth.

    Parameters
    ----------
    fdataobj : FData
        Functional data to compute depth for.
    fdataori : FData, optional
        Reference sample. If None, uses fdataobj.

    Returns
    -------
    depths : ndarray, shape (n_samples,)
        MEI depth values.
    """
    if fdataori is None:
        fdataori = fdataobj

    return _fdapy.depth_mei_1d(fdataobj.data, fdataori.data)
