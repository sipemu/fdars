// Basis function representations and fitting

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

/// Compute B-spline basis matrix
#[pyfunction]
pub fn bspline_basis<'py>(
    py: Python<'py>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let argvals_arr = argvals.as_array();
    let n_points = argvals_arr.len();

    if n_points == 0 || nbasis == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    // bspline_basis takes (t, nknots, order) - use order 4 (cubic)
    let order = 4;
    let nknots = nbasis.saturating_sub(order).max(2);
    let basis_flat = fdars_core::basis::bspline_basis(&argvals_vec, nknots, order);

    // Compute actual nbasis from result
    let actual_nbasis = basis_flat.len() / n_points;
    let result = to_row_major_2d(&basis_flat, n_points, actual_nbasis);
    Ok(result.into_pyarray(py))
}

/// Compute Fourier basis matrix
#[pyfunction]
pub fn fourier_basis<'py>(
    py: Python<'py>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let argvals_arr = argvals.as_array();
    let n_points = argvals_arr.len();

    if n_points == 0 || nbasis == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let basis_flat = fdars_core::basis::fourier_basis(&argvals_vec, nbasis);

    let result = to_row_major_2d(&basis_flat, n_points, nbasis);
    Ok(result.into_pyarray(py))
}

/// Compute Fourier basis matrix with explicit period
#[pyfunction]
pub fn fourier_basis_with_period<'py>(
    py: Python<'py>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
    period: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let argvals_arr = argvals.as_array();
    let n_points = argvals_arr.len();

    if n_points == 0 || nbasis == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let basis_flat = fdars_core::basis::fourier_basis_with_period(&argvals_vec, nbasis, period);

    let result = to_row_major_2d(&basis_flat, n_points, nbasis);
    Ok(result.into_pyarray(py))
}

/// Project functional data to basis coefficients
#[pyfunction]
#[pyo3(signature = (data, argvals, nbasis, basis_type="bspline"))]
pub fn fdata_to_basis_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
    basis_type: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 || nbasis == 0 || argvals_arr.len() != n_points {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let btype = if basis_type == "fourier" { 1 } else { 0 };

    // fdata_to_basis_1d(data, n, m, argvals, nbasis, basis_type) -> Option<BasisProjectionResult>
    match fdars_core::basis::fdata_to_basis_1d(&data_flat, n, n_points, &argvals_vec, nbasis, btype)
    {
        Some(result) => {
            let actual_nbasis = result.n_basis;
            let coefs = to_row_major_2d(&result.coefficients, n, actual_nbasis);
            Ok(coefs.into_pyarray(py))
        }
        None => Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py)),
    }
}

/// Reconstruct functional data from basis coefficients
#[pyfunction]
#[pyo3(signature = (coefs, argvals, nbasis, basis_type="bspline"))]
pub fn basis_to_fdata_1d<'py>(
    py: Python<'py>,
    coefs: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
    basis_type: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coefs_arr = coefs.as_array();
    let argvals_arr = argvals.as_array();
    let (n, nb) = coefs_arr.dim();
    let n_points = argvals_arr.len();

    if n == 0 || n_points == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let coefs_flat = to_col_major(&coefs_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let btype = if basis_type == "fourier" { 1 } else { 0 };

    // basis_to_fdata_1d(coefs, nbasis, n, argvals, basis_type, order) -> Vec<f64>
    let order = 4; // cubic B-splines
    let data_flat =
        fdars_core::basis::basis_to_fdata_1d(&coefs_flat, nb, n, &argvals_vec, btype, order);
    let result = to_row_major_2d(&data_flat, n, n_points);

    Ok(result.into_pyarray(py))
}

/// Fit P-splines to functional data
#[pyfunction]
#[pyo3(signature = (data, argvals, nbasis, lambda_=1.0, nderiv=2))]
pub fn pspline_fit_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
    lambda_: f64,
    nderiv: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 || nbasis == 0 {
        dict.set_item(
            "coefficients",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "fitted",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("edf", f64::NAN)?;
        dict.set_item("gcv", f64::NAN)?;
        dict.set_item("aic", f64::NAN)?;
        dict.set_item("bic", f64::NAN)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // pspline_fit_1d(data, n, m, argvals, nbasis, lambda, nderiv) -> Option<PsplineFitResult>
    match fdars_core::basis::pspline_fit_1d(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        nbasis,
        lambda_,
        nderiv,
    ) {
        Some(result) => {
            let actual_nbasis = result.n_basis;
            dict.set_item(
                "coefficients",
                to_row_major_2d(&result.coefficients, n, actual_nbasis).into_pyarray(py),
            )?;
            dict.set_item(
                "fitted",
                to_row_major_2d(&result.fitted, n, n_points).into_pyarray(py),
            )?;
            dict.set_item("edf", result.edf)?;
            dict.set_item("gcv", result.gcv)?;
            dict.set_item("aic", result.aic)?;
            dict.set_item("bic", result.bic)?;
        }
        None => {
            dict.set_item(
                "coefficients",
                ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
            )?;
            dict.set_item(
                "fitted",
                ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
            )?;
            dict.set_item("edf", f64::NAN)?;
            dict.set_item("gcv", f64::NAN)?;
            dict.set_item("aic", f64::NAN)?;
            dict.set_item("bic", f64::NAN)?;
        }
    }

    Ok(dict)
}

/// Fit Fourier basis to functional data
#[pyfunction]
pub fn fourier_fit_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 || nbasis == 0 {
        dict.set_item(
            "coefficients",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "fitted",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("edf", f64::NAN)?;
        dict.set_item("gcv", f64::NAN)?;
        dict.set_item("aic", f64::NAN)?;
        dict.set_item("bic", f64::NAN)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // fourier_fit_1d(data, n, m, argvals, nbasis) -> Option<FourierFitResult>
    match fdars_core::basis::fourier_fit_1d(&data_flat, n, n_points, &argvals_vec, nbasis) {
        Some(result) => {
            dict.set_item(
                "coefficients",
                to_row_major_2d(&result.coefficients, n, nbasis).into_pyarray(py),
            )?;
            dict.set_item(
                "fitted",
                to_row_major_2d(&result.fitted, n, n_points).into_pyarray(py),
            )?;
            dict.set_item("edf", result.edf)?;
            dict.set_item("gcv", result.gcv)?;
            dict.set_item("aic", result.aic)?;
            dict.set_item("bic", result.bic)?;
        }
        None => {
            dict.set_item(
                "coefficients",
                ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
            )?;
            dict.set_item(
                "fitted",
                ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
            )?;
            dict.set_item("edf", f64::NAN)?;
            dict.set_item("gcv", f64::NAN)?;
            dict.set_item("aic", f64::NAN)?;
            dict.set_item("bic", f64::NAN)?;
        }
    }

    Ok(dict)
}

/// Automatically select optimal basis (Fourier vs P-spline)
#[pyfunction]
#[pyo3(signature = (data, argvals, nbasis_min=5, nbasis_max=30, criterion="GCV", lambda_=1.0))]
pub fn select_basis_auto_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis_min: usize,
    nbasis_max: usize,
    criterion: &str,
    lambda_: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("basis_type", "bspline")?;
        dict.set_item("nbasis", 0)?;
        dict.set_item(
            "coefficients",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "fitted",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("score", f64::INFINITY)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let crit: i32 = match criterion {
        "AIC" => 1,
        "BIC" => 2,
        _ => 0, // GCV
    };

    // select_basis_auto_1d(data, n, m, argvals, criterion, nbasis_min, nbasis_max, lambda_pspline, use_seasonal_hint)
    let result = fdars_core::basis::select_basis_auto_1d(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        crit,
        nbasis_min,
        nbasis_max,
        lambda_,
        true,
    );

    // Return info for first curve as representative
    if !result.selections.is_empty() {
        let sel = &result.selections[0];
        let basis_type = if sel.basis_type == 1 {
            "fourier"
        } else {
            "bspline"
        };
        dict.set_item("basis_type", basis_type)?;
        dict.set_item("nbasis", sel.nbasis)?;
        dict.set_item("coefficients", sel.coefficients.clone().into_pyarray(py))?;
        dict.set_item("fitted", sel.fitted.clone().into_pyarray(py))?;
        dict.set_item("score", sel.score)?;
        dict.set_item("seasonal_detected", sel.seasonal_detected)?;
    }

    Ok(dict)
}

/// Compute GCV score for basis fitting
#[pyfunction]
#[pyo3(signature = (data, argvals, nbasis, basis_type="bspline", lambda_=1.0))]
pub fn basis_gcv_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    nbasis: usize,
    basis_type: &str,
    lambda_: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 || nbasis == 0 {
        dict.set_item("gcv", f64::NAN)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // Use pspline or fourier depending on type
    if basis_type == "fourier" {
        match fdars_core::basis::fourier_fit_1d(&data_flat, n, n_points, &argvals_vec, nbasis) {
            Some(result) => {
                dict.set_item("gcv", result.gcv)?;
            }
            None => {
                dict.set_item("gcv", f64::NAN)?;
            }
        }
    } else {
        match fdars_core::basis::pspline_fit_1d(
            &data_flat,
            n,
            n_points,
            &argvals_vec,
            nbasis,
            lambda_,
            2,
        ) {
            Some(result) => {
                dict.set_item("gcv", result.gcv)?;
            }
            None => {
                dict.set_item("gcv", f64::NAN)?;
            }
        }
    }

    Ok(dict)
}
