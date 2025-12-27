// Depth measures for functional data

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Helper to convert row-major numpy array to column-major flat vector
fn to_col_major(data: &ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (nrow, ncol) = data.dim();
    (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data[[i, j]]))
        .collect()
}

/// Compute Fraiman-Muniz depth (1D)
///
/// Parameters
/// ----------
/// data_obj : ndarray, shape (n_obj, n_points)
///     Data for which to compute depth
/// data_ori : ndarray, shape (n_ori, n_points)
///     Reference sample
/// scale : bool
///     Whether to scale by integral of weights (default True)
///
/// Returns
/// -------
/// depths : ndarray, shape (n_obj,)
///     Depth values
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, scale=true))]
pub fn depth_fm_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    scale: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, n_ori_points) = ori.dim();

    if n_points != n_ori_points || n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::fraiman_muniz_1d(&obj_flat, &ori_flat, n_obj, n_ori, n_points, scale);
    Ok(depths.into_pyarray(py))
}

/// Compute Fraiman-Muniz depth (2D surfaces)
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, scale=true))]
pub fn depth_fm_2d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    scale: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, n_ori_points) = ori.dim();

    if n_points != n_ori_points || n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::fraiman_muniz_2d(&obj_flat, &ori_flat, n_obj, n_ori, n_points, scale);
    Ok(depths.into_pyarray(py))
}

/// Compute Modal depth (1D)
///
/// Parameters
/// ----------
/// data_obj : ndarray, shape (n_obj, n_points)
///     Data for which to compute depth
/// data_ori : ndarray, shape (n_ori, n_points)
///     Reference sample
/// h : float
///     Bandwidth parameter
///
/// Returns
/// -------
/// depths : ndarray, shape (n_obj,)
///     Depth values
#[pyfunction]
pub fn depth_mode_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    h: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::modal_1d(&obj_flat, &ori_flat, n_obj, n_ori, n_points, h);
    Ok(depths.into_pyarray(py))
}

/// Compute Modal depth (2D surfaces)
#[pyfunction]
pub fn depth_mode_2d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    h: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::modal_2d(&obj_flat, &ori_flat, n_obj, n_ori, n_points, h);
    Ok(depths.into_pyarray(py))
}

/// Compute Random Projection depth (1D)
///
/// Parameters
/// ----------
/// data_obj : ndarray, shape (n_obj, n_points)
///     Data for which to compute depth
/// data_ori : ndarray, shape (n_ori, n_points)
///     Reference sample
/// n_projections : int
///     Number of random projections (default 50)
///
/// Returns
/// -------
/// depths : ndarray, shape (n_obj,)
///     Depth values
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, n_projections=50))]
pub fn depth_rp_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    n_projections: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::random_projection_1d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points, n_projections
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Random Projection depth (2D surfaces)
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, n_projections=50))]
pub fn depth_rp_2d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    n_projections: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::random_projection_2d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points, n_projections
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Random Tukey depth (1D)
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, n_projections=50))]
pub fn depth_rt_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    n_projections: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::random_tukey_1d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points, n_projections
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Random Tukey depth (2D surfaces)
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, n_projections=50))]
pub fn depth_rt_2d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    n_projections: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::random_tukey_2d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points, n_projections
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Functional Spatial depth (1D)
#[pyfunction]
pub fn depth_fsd_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::functional_spatial_1d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Functional Spatial depth (2D surfaces)
#[pyfunction]
pub fn depth_fsd_2d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::functional_spatial_2d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Kernel Functional Spatial depth (1D)
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, argvals, h=None))]
pub fn depth_kfsd_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    h: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let argvals_arr = argvals.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 || argvals_arr.len() != n_points {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Compute default bandwidth if not provided
    let bandwidth = h.unwrap_or_else(|| {
        let n = n_ori as f64;
        1.06 * n.powf(-0.2)
    });

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // kernel_functional_spatial_1d(data_obj, data_ori, nobj, nori, n_points, argvals, h)
    let depths = fdars_core::depth::kernel_functional_spatial_1d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points, &argvals_vec, bandwidth
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Kernel Functional Spatial depth (2D surfaces)
#[pyfunction]
#[pyo3(signature = (data_obj, data_ori, h=None))]
pub fn depth_kfsd_2d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
    h: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let bandwidth = h.unwrap_or_else(|| {
        let n = n_ori as f64;
        1.06 * n.powf(-0.2)
    });

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    // kernel_functional_spatial_2d(data_obj, data_ori, nobj, nori, n_points, h)
    let depths = fdars_core::depth::kernel_functional_spatial_2d(
        &obj_flat, &ori_flat, n_obj, n_ori, n_points, bandwidth
    );
    Ok(depths.into_pyarray(py))
}

/// Compute Band depth (1D)
#[pyfunction]
pub fn depth_bd_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::band_1d(&obj_flat, &ori_flat, n_obj, n_ori, n_points);
    Ok(depths.into_pyarray(py))
}

/// Compute Modified Band depth (1D)
#[pyfunction]
pub fn depth_mbd_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::modified_band_1d(&obj_flat, &ori_flat, n_obj, n_ori, n_points);
    Ok(depths.into_pyarray(py))
}

/// Compute Modified Epigraph Index depth (1D)
#[pyfunction]
pub fn depth_mei_1d<'py>(
    py: Python<'py>,
    data_obj: PyReadonlyArray2<'py, f64>,
    data_ori: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj = data_obj.as_array();
    let ori = data_ori.as_array();
    let (n_obj, n_points) = obj.dim();
    let (n_ori, _) = ori.dim();

    if n_obj == 0 || n_ori == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let obj_flat = to_col_major(&obj);
    let ori_flat = to_col_major(&ori);

    let depths = fdars_core::depth::modified_epigraph_index_1d(&obj_flat, &ori_flat, n_obj, n_ori, n_points);
    Ok(depths.into_pyarray(py))
}
