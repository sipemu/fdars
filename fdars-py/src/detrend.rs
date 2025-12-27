// Detrending and decomposition for functional data

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
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

/// Detrend functional data
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// method : str
///     Detrending method: "linear", "polynomial", "diff", "loess", or "auto"
/// degree : int
///     Polynomial degree for "polynomial" method (default 2)
/// span : float
///     LOESS span for "loess" method (default 0.75)
///
/// Returns
/// -------
/// dict with keys: 'detrended', 'trend', 'method', 'coefficients', 'rss', 'n_params'
#[pyfunction]
#[pyo3(signature = (data, argvals, method="linear", degree=2, span=0.75))]
pub fn detrend<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    method: &str,
    degree: usize,
    span: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item(
            "detrended",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "trend",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("method", method)?;
        dict.set_item("coefficients", pyo3::types::PyList::empty(py))?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    let result = match method {
        "linear" => fdars_core::detrend::detrend_linear(&data_flat, n, n_points, &argvals_vec),
        "polynomial" => {
            fdars_core::detrend::detrend_polynomial(&data_flat, n, n_points, &argvals_vec, degree)
        }
        "diff" => fdars_core::detrend::detrend_diff(&data_flat, n, n_points, 1),
        "loess" => {
            fdars_core::detrend::detrend_loess(&data_flat, n, n_points, &argvals_vec, span, 1)
        }
        "auto" => fdars_core::detrend::auto_detrend(&data_flat, n, n_points, &argvals_vec),
        _ => fdars_core::detrend::detrend_linear(&data_flat, n, n_points, &argvals_vec),
    };

    dict.set_item(
        "detrended",
        to_row_major_2d(&result.detrended, n, n_points).into_pyarray(py),
    )?;
    dict.set_item(
        "trend",
        to_row_major_2d(&result.trend, n, n_points).into_pyarray(py),
    )?;
    dict.set_item("method", &result.method)?;

    // Convert coefficients if present
    match &result.coefficients {
        Some(coefs) => {
            dict.set_item("coefficients", coefs.clone().into_pyarray(py))?;
        }
        None => {
            dict.set_item("coefficients", pyo3::types::PyList::empty(py))?;
        }
    }

    // Include RSS values and n_params
    dict.set_item("rss", result.rss.into_pyarray(py))?;
    dict.set_item("n_params", result.n_params)?;

    Ok(dict)
}

/// Seasonal decomposition (additive or multiplicative)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// period : float
///     Period for seasonal decomposition
/// method : str
///     Decomposition method: "additive" or "multiplicative"
///
/// Returns
/// -------
/// dict with keys: 'trend', 'seasonal', 'remainder', 'period'
#[pyfunction]
#[pyo3(signature = (data, argvals, period, method="additive"))]
pub fn decompose<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    period: f64,
    method: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item(
            "trend",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "seasonal",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "remainder",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("period", period)?;
        dict.set_item("method", method)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // decompose_additive(data, n, m, argvals, period, trend_method, bandwidth, n_harmonics)
    let result = if method == "multiplicative" {
        fdars_core::detrend::decompose_multiplicative(
            &data_flat,
            n,
            n_points,
            &argvals_vec,
            period,
            "loess",
            0.25,
            3,
        )
    } else {
        fdars_core::detrend::decompose_additive(
            &data_flat,
            n,
            n_points,
            &argvals_vec,
            period,
            "loess",
            0.25,
            3,
        )
    };

    dict.set_item(
        "trend",
        to_row_major_2d(&result.trend, n, n_points).into_pyarray(py),
    )?;
    dict.set_item(
        "seasonal",
        to_row_major_2d(&result.seasonal, n, n_points).into_pyarray(py),
    )?;
    dict.set_item(
        "remainder",
        to_row_major_2d(&result.remainder, n, n_points).into_pyarray(py),
    )?;
    dict.set_item("period", result.period)?;
    dict.set_item("method", &result.method)?;

    Ok(dict)
}
