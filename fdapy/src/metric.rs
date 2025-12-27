// Distance metrics for functional data

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Helper to convert row-major numpy array to column-major flat vector
fn to_col_major(data: &ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (nrow, ncol) = data.dim();
    (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data[[i, j]]))
        .collect()
}

/// Helper to convert column-major flat vector to row-major ndarray
fn to_row_major_2d(flat: &[f64], nrow: usize, ncol: usize) -> ndarray::Array2<f64> {
    let mut arr = ndarray::Array2::<f64>::zeros((nrow, ncol));
    for i in 0..nrow {
        for j in 0..ncol {
            arr[[i, j]] = flat[i + j * nrow];
        }
    }
    arr
}

/// Compute Lp distance matrix (self-distances)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// p : float
///     Order of the Lp norm (default 2.0)
///
/// Returns
/// -------
/// D : ndarray, shape (n_samples, n_samples)
///     Symmetric distance matrix
#[pyfunction]
#[pyo3(signature = (data, argvals, p=2.0))]
pub fn metric_lp_self_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || argvals_arr.len() != n_points {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let user_weights: Vec<f64> = vec![1.0; n_points];

    // lp_self_1d(data, n, n_points, argvals, p, user_weights)
    let dist_flat = fdars_core::metric::lp_self_1d(&data_flat, n, n_points, &argvals_vec, p, &user_weights);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute Lp distance matrix (cross-distances)
#[pyfunction]
#[pyo3(signature = (data1, data2, argvals, p=2.0))]
pub fn metric_lp_cross_1d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let argvals_arr = argvals.as_array();
    let (n1, n_points1) = d1.dim();
    let (n2, n_points2) = d2.dim();

    if n1 == 0 || n2 == 0 || n_points1 != n_points2 || argvals_arr.len() != n_points1 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let user_weights: Vec<f64> = vec![1.0; n_points1];

    // lp_cross_1d(data1, data2, n1, n2, n_points, argvals, p, user_weights)
    let dist_flat = fdars_core::metric::lp_cross_1d(&d1_flat, &d2_flat, n1, n2, n_points1, &argvals_vec, p, &user_weights);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}

/// Compute Lp distance matrix for 2D surfaces (self-distances)
#[pyfunction]
#[pyo3(signature = (data, argvals_s, argvals_t, p=2.0))]
pub fn metric_lp_self_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals_s: PyReadonlyArray1<'py, f64>,
    argvals_t: PyReadonlyArray1<'py, f64>,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_s_arr = argvals_s.as_array();
    let argvals_t_arr = argvals_t.as_array();
    let (n, ncol) = data_arr.dim();
    let m1 = argvals_s_arr.len();
    let m2 = argvals_t_arr.len();

    if n == 0 || ncol != m1 * m2 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_s_vec: Vec<f64> = argvals_s_arr.to_vec();
    let argvals_t_vec: Vec<f64> = argvals_t_arr.to_vec();
    let user_weights: Vec<f64> = vec![1.0; m1 * m2];

    let dist_flat = fdars_core::metric::lp_self_2d(&data_flat, n, &argvals_s_vec, &argvals_t_vec, p, &user_weights);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute Lp distance matrix for 2D surfaces (cross-distances)
#[pyfunction]
#[pyo3(signature = (data1, data2, argvals_s, argvals_t, p=2.0))]
pub fn metric_lp_cross_2d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    argvals_s: PyReadonlyArray1<'py, f64>,
    argvals_t: PyReadonlyArray1<'py, f64>,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let argvals_s_arr = argvals_s.as_array();
    let argvals_t_arr = argvals_t.as_array();
    let (n1, _) = d1.dim();
    let (n2, _) = d2.dim();
    let m1 = argvals_s_arr.len();
    let m2 = argvals_t_arr.len();

    if n1 == 0 || n2 == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);
    let argvals_s_vec: Vec<f64> = argvals_s_arr.to_vec();
    let argvals_t_vec: Vec<f64> = argvals_t_arr.to_vec();
    let user_weights: Vec<f64> = vec![1.0; m1 * m2];

    let dist_flat = fdars_core::metric::lp_cross_2d(&d1_flat, &d2_flat, n1, n2, &argvals_s_vec, &argvals_t_vec, p, &user_weights);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}

/// Compute Hausdorff distance matrix (self-distances)
#[pyfunction]
pub fn metric_hausdorff_self_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || argvals_arr.len() != n_points {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // hausdorff_self_1d(data, n, m, argvals)
    let dist_flat = fdars_core::metric::hausdorff_self_1d(&data_flat, n, n_points, &argvals_vec);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute Hausdorff distance matrix (cross-distances)
#[pyfunction]
pub fn metric_hausdorff_cross_1d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let argvals_arr = argvals.as_array();
    let (n1, n_points1) = d1.dim();
    let (n2, n_points2) = d2.dim();

    if n1 == 0 || n2 == 0 || n_points1 != n_points2 || argvals_arr.len() != n_points1 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // hausdorff_cross_1d(data1, data2, n1, n2, m, argvals)
    let dist_flat = fdars_core::metric::hausdorff_cross_1d(&d1_flat, &d2_flat, n1, n2, n_points1, &argvals_vec);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}

/// Compute Hausdorff distance matrix for 2D surfaces (self-distances)
#[pyfunction]
pub fn metric_hausdorff_self_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals_s: PyReadonlyArray1<'py, f64>,
    argvals_t: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_s_arr = argvals_s.as_array();
    let argvals_t_arr = argvals_t.as_array();
    let (n, ncol) = data_arr.dim();
    let m1 = argvals_s_arr.len();
    let m2 = argvals_t_arr.len();

    if n == 0 || ncol != m1 * m2 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_s_vec: Vec<f64> = argvals_s_arr.to_vec();
    let argvals_t_vec: Vec<f64> = argvals_t_arr.to_vec();

    let dist_flat = fdars_core::metric::hausdorff_self_2d(&data_flat, n, &argvals_s_vec, &argvals_t_vec);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute Hausdorff distance matrix for 2D surfaces (cross-distances)
#[pyfunction]
pub fn metric_hausdorff_cross_2d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    argvals_s: PyReadonlyArray1<'py, f64>,
    argvals_t: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let argvals_s_arr = argvals_s.as_array();
    let argvals_t_arr = argvals_t.as_array();
    let (n1, _) = d1.dim();
    let (n2, _) = d2.dim();
    let m1 = argvals_s_arr.len();
    let m2 = argvals_t_arr.len();

    if n1 == 0 || n2 == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);
    let argvals_s_vec: Vec<f64> = argvals_s_arr.to_vec();
    let argvals_t_vec: Vec<f64> = argvals_t_arr.to_vec();

    let dist_flat = fdars_core::metric::hausdorff_cross_2d(&d1_flat, &d2_flat, n1, n2, &argvals_s_vec, &argvals_t_vec);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}

/// Compute DTW distance matrix (self-distances)
#[pyfunction]
#[pyo3(signature = (data, p=2.0, w=0))]
pub fn metric_dtw_self_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    p: f64,
    w: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let dist_flat = fdars_core::metric::dtw_self_1d(&data_flat, n, n_points, p, w);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute DTW distance matrix (cross-distances)
#[pyfunction]
#[pyo3(signature = (data1, data2, p=2.0, w=0))]
pub fn metric_dtw_cross_1d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    p: f64,
    w: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let (n1, m1) = d1.dim();
    let (n2, m2) = d2.dim();

    if n1 == 0 || n2 == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);

    // dtw_cross_1d(data1, data2, n1, n2, m1, m2, p, w)
    let dist_flat = fdars_core::metric::dtw_cross_1d(&d1_flat, &d2_flat, n1, n2, m1, m2, p, w);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}

/// Compute Fourier semimetric distance matrix (self-distances)
#[pyfunction]
#[pyo3(signature = (data, nfreq=5))]
pub fn semimetric_fourier_self_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    nfreq: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let dist_flat = fdars_core::metric::fourier_self_1d(&data_flat, n, n_points, nfreq);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute Fourier semimetric distance matrix (cross-distances)
#[pyfunction]
#[pyo3(signature = (data1, data2, nfreq=5))]
pub fn semimetric_fourier_cross_1d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    nfreq: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let (n1, m1) = d1.dim();
    let (n2, m2) = d2.dim();

    if n1 == 0 || n2 == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);

    let dist_flat = fdars_core::metric::fourier_cross_1d(&d1_flat, &d2_flat, n1, n2, m1, nfreq);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}

/// Compute horizontal shift semimetric distance matrix (self-distances)
#[pyfunction]
#[pyo3(signature = (data, argvals, max_shift=None))]
pub fn semimetric_hshift_self_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    max_shift: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || argvals_arr.len() != n_points {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    // max_shift is in grid points (usize), not absolute time
    let ms: usize = max_shift.map(|f| f as usize).unwrap_or(n_points / 4);

    let dist_flat = fdars_core::metric::hshift_self_1d(&data_flat, n, n_points, &argvals_vec, ms);
    let result = to_row_major_2d(&dist_flat, n, n);

    Ok(result.into_pyarray(py))
}

/// Compute horizontal shift semimetric distance matrix (cross-distances)
#[pyfunction]
#[pyo3(signature = (data1, data2, argvals, max_shift=None))]
pub fn semimetric_hshift_cross_1d<'py>(
    py: Python<'py>,
    data1: PyReadonlyArray2<'py, f64>,
    data2: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    max_shift: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d1 = data1.as_array();
    let d2 = data2.as_array();
    let argvals_arr = argvals.as_array();
    let (n1, m1) = d1.dim();
    let (n2, m2) = d2.dim();

    if n1 == 0 || n2 == 0 || m1 != m2 || argvals_arr.len() != m1 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let d1_flat = to_col_major(&d1);
    let d2_flat = to_col_major(&d2);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    // max_shift is in grid points (usize), not absolute time
    let ms: usize = max_shift.map(|f| f as usize).unwrap_or(m1 / 4);

    let dist_flat = fdars_core::metric::hshift_cross_1d(&d1_flat, &d2_flat, n1, n2, m1, &argvals_vec, ms);
    let result = to_row_major_2d(&dist_flat, n1, n2);

    Ok(result.into_pyarray(py))
}
