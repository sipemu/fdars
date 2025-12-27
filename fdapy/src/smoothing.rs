// Kernel-based smoothing methods

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

/// Nadaraya-Watson kernel smoother
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// h : float
///     Bandwidth parameter
/// kernel : str
///     Kernel type: "gaussian" or "epanechnikov" (default "gaussian")
/// eval_points : ndarray, optional
///     Points at which to evaluate the smoother. If None, uses argvals.
///
/// Returns
/// -------
/// smoothed : ndarray, shape (n_samples, n_eval_points)
///     Smoothed functional data
#[pyfunction]
#[pyo3(signature = (data, argvals, h, kernel="gaussian", eval_points=None))]
pub fn nadaraya_watson<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    h: f64,
    kernel: &str,
    eval_points: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let eval_vec: Vec<f64> = match eval_points {
        Some(ep) => ep.as_array().to_vec(),
        None => argvals_vec.clone(),
    };
    let n_eval = eval_vec.len();
    let kernel_type = if kernel == "epanechnikov" { "epanechnikov" } else { "gaussian" };

    // Smooth each sample individually
    let mut result = ndarray::Array2::<f64>::zeros((n, n_eval));
    for i in 0..n {
        let y_values: Vec<f64> = (0..n_points).map(|j| data_arr[[i, j]]).collect();
        let smoothed = fdars_core::smoothing::nadaraya_watson(
            &argvals_vec, &y_values, &eval_vec, h, kernel_type
        );
        for (j, &val) in smoothed.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result.into_pyarray(py))
}

/// Local linear regression smoother
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// h : float
///     Bandwidth parameter
/// kernel : str
///     Kernel type: "gaussian" or "epanechnikov" (default "gaussian")
/// eval_points : ndarray, optional
///     Points at which to evaluate the smoother. If None, uses argvals.
///
/// Returns
/// -------
/// smoothed : ndarray, shape (n_samples, n_eval_points)
///     Smoothed functional data
#[pyfunction]
#[pyo3(signature = (data, argvals, h, kernel="gaussian", eval_points=None))]
pub fn local_linear<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    h: f64,
    kernel: &str,
    eval_points: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let eval_vec: Vec<f64> = match eval_points {
        Some(ep) => ep.as_array().to_vec(),
        None => argvals_vec.clone(),
    };
    let n_eval = eval_vec.len();
    let kernel_type = if kernel == "epanechnikov" { "epanechnikov" } else { "gaussian" };

    // Smooth each sample individually
    let mut result = ndarray::Array2::<f64>::zeros((n, n_eval));
    for i in 0..n {
        let y_values: Vec<f64> = (0..n_points).map(|j| data_arr[[i, j]]).collect();
        let smoothed = fdars_core::smoothing::local_linear(
            &argvals_vec, &y_values, &eval_vec, h, kernel_type
        );
        for (j, &val) in smoothed.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result.into_pyarray(py))
}

/// Local polynomial regression smoother
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// h : float
///     Bandwidth parameter
/// degree : int
///     Polynomial degree (0, 1, 2, or 3; default 2)
/// kernel : str
///     Kernel type: "gaussian" or "epanechnikov" (default "gaussian")
/// eval_points : ndarray, optional
///     Points at which to evaluate the smoother. If None, uses argvals.
///
/// Returns
/// -------
/// smoothed : ndarray, shape (n_samples, n_eval_points)
///     Smoothed functional data
#[pyfunction]
#[pyo3(signature = (data, argvals, h, degree=2, kernel="gaussian", eval_points=None))]
pub fn local_polynomial<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    h: f64,
    degree: usize,
    kernel: &str,
    eval_points: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let eval_vec: Vec<f64> = match eval_points {
        Some(ep) => ep.as_array().to_vec(),
        None => argvals_vec.clone(),
    };
    let n_eval = eval_vec.len();
    let kernel_type = if kernel == "epanechnikov" { "epanechnikov" } else { "gaussian" };

    // Smooth each sample individually
    let mut result = ndarray::Array2::<f64>::zeros((n, n_eval));
    for i in 0..n {
        let y_values: Vec<f64> = (0..n_points).map(|j| data_arr[[i, j]]).collect();
        let smoothed = fdars_core::smoothing::local_polynomial(
            &argvals_vec, &y_values, &eval_vec, h, degree, kernel_type
        );
        for (j, &val) in smoothed.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result.into_pyarray(py))
}
