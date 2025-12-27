// Functional data operations: mean, center, derivatives, norms, geometric median

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Helper: compute Simpson's rule integration weights
fn simpsons_weights(argvals: &[f64]) -> Vec<f64> {
    fdars_core::helpers::simpsons_weights(argvals)
}

/// Compute the mean function across all samples (1D)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
///
/// Returns
/// -------
/// mean : ndarray, shape (n_points,)
///     Mean function
#[pyfunction]
pub fn fdata_mean_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let (nrow, ncol) = data_arr.dim();

    if nrow == 0 || ncol == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Convert to column-major flat vector for fdars-core
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data_arr[[i, j]]))
        .collect();

    let mean = fdars_core::fdata::mean_1d(&data_flat, nrow, ncol);
    Ok(mean.into_pyarray(py))
}

/// Compute the mean function across all samples (2D surfaces)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, m1*m2)
///     Functional data matrix with flattened surfaces
/// m1 : int
///     First dimension of surface grid
/// m2 : int
///     Second dimension of surface grid
///
/// Returns
/// -------
/// mean : ndarray, shape (m1*m2,)
///     Mean surface (flattened)
#[pyfunction]
pub fn fdata_mean_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    m1: usize,
    m2: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let (n, ncol) = data_arr.dim();

    if n == 0 || ncol != m1 * m2 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..n).map(move |i| data_arr[[i, j]]))
        .collect();

    let mean = fdars_core::fdata::mean_2d(&data_flat, n, m1 * m2);
    Ok(mean.into_pyarray(py))
}

/// Center functional data by subtracting the mean function
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
///
/// Returns
/// -------
/// centered : ndarray, shape (n_samples, n_points)
///     Centered functional data
#[pyfunction]
pub fn fdata_center_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let (nrow, ncol) = data_arr.dim();

    if nrow == 0 || ncol == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data_arr[[i, j]]))
        .collect();

    let centered_flat = fdars_core::fdata::center_1d(&data_flat, nrow, ncol);

    // Convert back to row-major ndarray
    let mut result = ndarray::Array2::<f64>::zeros((nrow, ncol));
    for i in 0..nrow {
        for j in 0..ncol {
            result[[i, j]] = centered_flat[i + j * nrow];
        }
    }

    Ok(result.into_pyarray(py))
}

/// Compute Lp norm for each sample
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
/// norms : ndarray, shape (n_samples,)
///     Lp norm for each sample
#[pyfunction]
#[pyo3(signature = (data, argvals, p=2.0))]
pub fn fdata_norm_lp_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    p: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (nrow, ncol) = data_arr.dim();

    if nrow == 0 || ncol == 0 || argvals_arr.len() != ncol {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data_arr[[i, j]]))
        .collect();

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let norms = fdars_core::fdata::norm_lp_1d(&data_flat, nrow, ncol, &argvals_vec, p);

    Ok(norms.into_pyarray(py))
}

/// Compute numerical derivative of functional data
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// nderiv : int
///     Order of derivative (default 1)
///
/// Returns
/// -------
/// deriv : ndarray, shape (n_samples, n_points)
///     Derivative of functional data
#[pyfunction]
#[pyo3(signature = (data, argvals, nderiv=1))]
pub fn fdata_deriv_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nderiv: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (nrow, ncol) = data_arr.dim();

    if nrow == 0 || ncol == 0 || argvals_arr.len() != ncol || nderiv < 1 {
        return Ok(ndarray::Array2::<f64>::zeros((nrow, ncol)).into_pyarray(py));
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data_arr[[i, j]]))
        .collect();

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let deriv_flat = fdars_core::fdata::deriv_1d(&data_flat, nrow, ncol, &argvals_vec, nderiv);

    // Convert back to row-major ndarray
    let mut result = ndarray::Array2::<f64>::zeros((nrow, ncol));
    for i in 0..nrow {
        for j in 0..ncol {
            result[[i, j]] = deriv_flat[i + j * nrow];
        }
    }

    Ok(result.into_pyarray(py))
}

/// Compute 2D partial derivatives for surface data
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, m1*m2)
///     Functional data matrix with flattened surfaces
/// argvals_s : ndarray, shape (m1,)
///     Evaluation points for first dimension
/// argvals_t : ndarray, shape (m2,)
///     Evaluation points for second dimension
/// m1 : int
///     First dimension of surface grid
/// m2 : int
///     Second dimension of surface grid
///
/// Returns
/// -------
/// dict with keys 'ds', 'dt', 'dsdt'
///     Partial derivatives
#[pyfunction]
pub fn fdata_deriv_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals_s: PyReadonlyArray1<'py, f64>,
    argvals_t: PyReadonlyArray1<'py, f64>,
    m1: usize,
    m2: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_s_arr = argvals_s.as_array();
    let argvals_t_arr = argvals_t.as_array();
    let (n, ncol) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || ncol != m1 * m2 || argvals_s_arr.len() != m1 || argvals_t_arr.len() != m2 {
        dict.set_item("ds", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        dict.set_item("dt", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        dict.set_item("dsdt", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        return Ok(dict);
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..n).map(move |i| data_arr[[i, j]]))
        .collect();

    let argvals_s_vec: Vec<f64> = argvals_s_arr.to_vec();
    let argvals_t_vec: Vec<f64> = argvals_t_arr.to_vec();

    let result = fdars_core::fdata::deriv_2d(&data_flat, n, &argvals_s_vec, &argvals_t_vec, m1, m2);

    match result {
        Some(res) => {
            // Helper to convert flat column-major to row-major ndarray
            let to_array = |flat: &[f64]| -> ndarray::Array2<f64> {
                let mut arr = ndarray::Array2::<f64>::zeros((n, ncol));
                for i in 0..n {
                    for j in 0..ncol {
                        arr[[i, j]] = flat[i + j * n];
                    }
                }
                arr
            };

            dict.set_item("ds", to_array(&res.ds).into_pyarray(py))?;
            dict.set_item("dt", to_array(&res.dt).into_pyarray(py))?;
            dict.set_item("dsdt", to_array(&res.dsdt).into_pyarray(py))?;
        }
        None => {
            dict.set_item("ds", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            dict.set_item("dt", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            dict.set_item("dsdt", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        }
    }

    Ok(dict)
}

/// Compute the geometric median (L1 median) of functional data (1D)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// max_iter : int
///     Maximum iterations (default 1000)
/// tol : float
///     Convergence tolerance (default 1e-6)
///
/// Returns
/// -------
/// median : ndarray, shape (n_points,)
///     Geometric median function
#[pyfunction]
#[pyo3(signature = (data, argvals, max_iter=1000, tol=1e-6))]
pub fn geometric_median_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (nrow, ncol) = data_arr.dim();

    if nrow == 0 || ncol == 0 || argvals_arr.len() != ncol {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data_arr[[i, j]]))
        .collect();

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let median = fdars_core::fdata::geometric_median_1d(&data_flat, nrow, ncol, &argvals_vec, max_iter, tol);

    Ok(median.into_pyarray(py))
}

/// Compute the geometric median of 2D functional data (surfaces)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, m1*m2)
///     Functional data matrix with flattened surfaces
/// argvals_s : ndarray, shape (m1,)
///     Evaluation points for first dimension
/// argvals_t : ndarray, shape (m2,)
///     Evaluation points for second dimension
/// m1 : int
///     First dimension of surface grid
/// m2 : int
///     Second dimension of surface grid
/// max_iter : int
///     Maximum iterations (default 1000)
/// tol : float
///     Convergence tolerance (default 1e-6)
///
/// Returns
/// -------
/// median : ndarray, shape (m1*m2,)
///     Geometric median surface (flattened)
#[pyfunction]
#[pyo3(signature = (data, argvals_s, argvals_t, m1, m2, max_iter=1000, tol=1e-6))]
pub fn geometric_median_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals_s: PyReadonlyArray1<'py, f64>,
    argvals_t: PyReadonlyArray1<'py, f64>,
    m1: usize,
    m2: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let argvals_s_arr = argvals_s.as_array();
    let argvals_t_arr = argvals_t.as_array();
    let (n, ncol) = data_arr.dim();

    if n == 0 || ncol != m1 * m2 || argvals_s_arr.len() != m1 || argvals_t_arr.len() != m2 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Convert to column-major flat vector
    let data_flat: Vec<f64> = (0..ncol)
        .flat_map(|j| (0..n).map(move |i| data_arr[[i, j]]))
        .collect();

    let argvals_s_vec: Vec<f64> = argvals_s_arr.to_vec();
    let argvals_t_vec: Vec<f64> = argvals_t_arr.to_vec();

    let median = fdars_core::fdata::geometric_median_2d(
        &data_flat, n, m1 * m2, &argvals_s_vec, &argvals_t_vec, max_iter, tol
    );

    Ok(median.into_pyarray(py))
}
