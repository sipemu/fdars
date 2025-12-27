// Utility functions for functional data analysis

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

/// Integrate a function using Simpson's rule
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Function values at evaluation points
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
///
/// Returns
/// -------
/// integrals : ndarray, shape (n_samples,)
///     Integral values for each sample
#[pyfunction]
pub fn integrate_simpson<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 || argvals_arr.len() != n_points {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // integrate_simpson works on single curves, so loop over each
    let integrals: Vec<f64> = (0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..n_points).map(|j| data_arr[[i, j]]).collect();
            fdars_core::utility::integrate_simpson(&curve, &argvals_vec)
        })
        .collect();

    Ok(integrals.into_pyarray(py))
}

/// Compute inner product between two functional datasets
///
/// Parameters
/// ----------
/// data1 : ndarray, shape (n1, n_points)
///     First functional data matrix
/// data2 : ndarray, shape (n2, n_points)
///     Second functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
///
/// Returns
/// -------
/// inner_prods : ndarray, shape (n1, n2)
///     Inner product matrix
#[pyfunction]
pub fn inner_product<'py>(
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

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // inner_product works on single curves, so compute cross-product matrix
    let mut result = ndarray::Array2::<f64>::zeros((n1, n2));
    for i in 0..n1 {
        let curve1: Vec<f64> = (0..n_points1).map(|j| d1[[i, j]]).collect();
        for k in 0..n2 {
            let curve2: Vec<f64> = (0..n_points1).map(|j| d2[[k, j]]).collect();
            result[[i, k]] = fdars_core::utility::inner_product(&curve1, &curve2, &argvals_vec);
        }
    }

    Ok(result.into_pyarray(py))
}

/// Compute symmetric inner product matrix for a single dataset
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
///
/// Returns
/// -------
/// gram : ndarray, shape (n_samples, n_samples)
///     Gram matrix (symmetric inner product matrix)
#[pyfunction]
pub fn inner_product_matrix<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 || argvals_arr.len() != n_points {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // inner_product_matrix(data, n, m, argvals)
    let gram_flat = fdars_core::utility::inner_product_matrix(&data_flat, n, n_points, &argvals_vec);
    let result = to_row_major_2d(&gram_flat, n, n);

    Ok(result.into_pyarray(py))
}
