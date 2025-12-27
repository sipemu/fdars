// Regression methods for functional data

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

/// Functional Principal Component Analysis (FPCA)
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// n_components : int or None
///     Number of components to keep. If None, keep all.
///
/// Returns
/// -------
/// dict with keys: 'scores', 'components', 'mean', 'explained_variance', 'explained_variance_ratio'
#[pyfunction]
#[pyo3(signature = (data, argvals, n_components=None))]
pub fn fdata_to_pc_1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    n_components: Option<usize>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("scores", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        dict.set_item("components", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        dict.set_item("mean", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("explained_variance", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("explained_variance_ratio", Vec::<f64>::new().into_pyarray(py))?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let _argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let ncomp = n_components.unwrap_or(n.min(n_points));

    // fdata_to_pc_1d(data, n, m, ncomp) -> Option<FpcaResult>
    let result = match fdars_core::regression::fdata_to_pc_1d(&data_flat, n, n_points, ncomp) {
        Some(r) => r,
        None => {
            dict.set_item("scores", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            dict.set_item("components", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            dict.set_item("mean", Vec::<f64>::new().into_pyarray(py))?;
            dict.set_item("explained_variance", Vec::<f64>::new().into_pyarray(py))?;
            dict.set_item("explained_variance_ratio", Vec::<f64>::new().into_pyarray(py))?;
            return Ok(dict);
        }
    };

    // Compute explained variance ratio
    let total_var: f64 = result.singular_values.iter().map(|x| x * x).sum();
    let exp_var: Vec<f64> = result.singular_values.iter().take(ncomp).map(|x| x * x).collect();
    let exp_var_ratio: Vec<f64> = exp_var.iter().map(|x| x / total_var).collect();

    dict.set_item("scores", to_row_major_2d(&result.scores, n, ncomp).into_pyarray(py))?;
    dict.set_item("components", to_row_major_2d(&result.rotation, ncomp, n_points).into_pyarray(py))?;
    dict.set_item("mean", result.mean.into_pyarray(py))?;
    dict.set_item("singular_values", result.singular_values.into_pyarray(py))?;
    dict.set_item("explained_variance", exp_var.into_pyarray(py))?;
    dict.set_item("explained_variance_ratio", exp_var_ratio.into_pyarray(py))?;
    dict.set_item("centered", to_row_major_2d(&result.centered, n, n_points).into_pyarray(py))?;

    Ok(dict)
}

/// Functional Partial Least Squares (FPLS)
///
/// Parameters
/// ----------
/// X : ndarray, shape (n_samples, n_points)
///     Functional predictor data
/// y : ndarray, shape (n_samples,)
///     Response variable
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// n_components : int
///     Number of PLS components
///
/// Returns
/// -------
/// dict with keys: 'x_scores', 'x_loadings', 'weights', 'coefficients'
#[pyfunction]
#[pyo3(signature = (x, y, argvals, n_components=2))]
pub fn fdata_to_pls_1d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    n_components: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = x_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 || y_arr.len() != n {
        dict.set_item("x_scores", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        dict.set_item("x_loadings", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        dict.set_item("weights", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
        return Ok(dict);
    }

    let x_flat = to_col_major(&x_arr);
    let y_vec: Vec<f64> = y_arr.to_vec();
    let _argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // fdata_to_pls_1d(data, n, m, y, ncomp) -> Option<PlsResult>
    let result = match fdars_core::regression::fdata_to_pls_1d(&x_flat, n, n_points, &y_vec, n_components) {
        Some(r) => r,
        None => {
            dict.set_item("x_scores", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            dict.set_item("x_loadings", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            dict.set_item("weights", ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py))?;
            return Ok(dict);
        }
    };

    dict.set_item("x_scores", to_row_major_2d(&result.scores, n, n_components).into_pyarray(py))?;
    dict.set_item("x_loadings", to_row_major_2d(&result.loadings, n_components, n_points).into_pyarray(py))?;
    dict.set_item("weights", to_row_major_2d(&result.weights, n_components, n_points).into_pyarray(py))?;

    Ok(dict)
}

/// Ridge regression
///
/// Parameters
/// ----------
/// X : ndarray, shape (n_samples, n_features)
///     Feature matrix
/// y : ndarray, shape (n_samples,)
///     Response variable
/// lambda_ : float
///     Regularization parameter
/// fit_intercept : bool
///     Whether to fit an intercept (default True)
///
/// Returns
/// -------
/// dict with keys: 'coefficients', 'intercept', 'fitted_values', 'residuals', 'r_squared'
#[pyfunction]
#[pyo3(signature = (x, y, lambda_, fit_intercept=true))]
pub fn ridge_regression_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    lambda_: f64,
    fit_intercept: bool,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let (n, p) = x_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || p == 0 || y_arr.len() != n {
        dict.set_item("coefficients", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("intercept", 0.0)?;
        dict.set_item("fitted_values", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("residuals", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("r_squared", f64::NAN)?;
        return Ok(dict);
    }

    let x_flat = to_col_major(&x_arr);
    let y_vec: Vec<f64> = y_arr.to_vec();

    let result = fdars_core::regression::ridge_regression_fit(&x_flat, &y_vec, n, p, lambda_, fit_intercept);

    dict.set_item("coefficients", result.coefficients.into_pyarray(py))?;
    dict.set_item("intercept", result.intercept)?;
    dict.set_item("fitted_values", result.fitted_values.into_pyarray(py))?;
    dict.set_item("residuals", result.residuals.into_pyarray(py))?;
    dict.set_item("r_squared", result.r_squared)?;
    dict.set_item("lambda", result.lambda)?;

    Ok(dict)
}
