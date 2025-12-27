// Outlier detection for functional data

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Helper to convert row-major numpy array to column-major flat vector
fn to_col_major(data: &ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (nrow, ncol) = data.dim();
    (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data[[i, j]]))
        .collect()
}

/// Depth-based outlier detection
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// quantile : float
///     Quantile threshold for outlier detection (default 0.01)
/// depth_method : str
///     Depth method to use: "FM", "mode", etc. (default "FM")
///
/// Returns
/// -------
/// dict with keys: 'outliers', 'depths', 'threshold'
#[pyfunction]
#[pyo3(signature = (data, quantile=0.01, depth_method="FM"))]
pub fn outliers_depth<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    quantile: f64,
    depth_method: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("outliers", Vec::<bool>::new().into_pyarray(py))?;
        dict.set_item("depths", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("threshold", f64::NAN)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);

    // Use FM depth by default
    let depths = fdars_core::depth::fraiman_muniz_1d(&data_flat, &data_flat, n, n, n_points, true);

    // Compute threshold as quantile of depths
    let mut sorted_depths = depths.clone();
    sorted_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold_idx = ((n as f64 * quantile).floor() as usize).max(0).min(n - 1);
    let threshold = sorted_depths[threshold_idx];

    // Mark outliers
    let outliers: Vec<bool> = depths.iter().map(|&d| d < threshold).collect();

    dict.set_item("outliers", outliers.into_pyarray(py))?;
    dict.set_item("depths", depths.into_pyarray(py))?;
    dict.set_item("threshold", threshold)?;

    Ok(dict)
}

/// Likelihood ratio test for outlier detection
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// trim : float
///     Trimming proportion (default 0.1)
/// alpha : float
///     Significance level (default 0.05)
/// n_bootstrap : int
///     Number of bootstrap samples for threshold estimation (default 200)
/// seed : int
///     Random seed for reproducibility
///
/// Returns
/// -------
/// dict with keys: 'outliers', 'statistics', 'threshold', 'p_values'
#[pyfunction]
#[pyo3(signature = (data, argvals, trim=0.1, alpha=0.05, n_bootstrap=200, seed=42))]
pub fn outliers_lrt<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    trim: f64,
    alpha: f64,
    n_bootstrap: usize,
    seed: u64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("outliers", Vec::<bool>::new().into_pyarray(py))?;
        dict.set_item("threshold", f64::NAN)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let _argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // Compute bootstrap threshold
    // outliers_threshold_lrt(data, n, m, nb, smo, trim, seed, percentile)
    let smo = 0.05; // smoothing parameter
    let threshold = fdars_core::outliers::outliers_threshold_lrt(
        &data_flat,
        n,
        n_points,
        n_bootstrap,
        smo,
        trim,
        seed,
        1.0 - alpha,
    );

    // Compute LRT statistics - returns Vec<bool> directly
    // detect_outliers_lrt(data, n, m, threshold, trim)
    let outliers =
        fdars_core::outliers::detect_outliers_lrt(&data_flat, n, n_points, threshold, trim);

    dict.set_item("outliers", outliers.into_pyarray(py))?;
    dict.set_item("threshold", threshold)?;

    Ok(dict)
}
