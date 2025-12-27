// fdapy - Functional Data Analysis in Python
// PyO3 bindings for the fdars-core Rust library

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

mod basis;
mod clustering;
mod depth;
mod detrend;
mod fdata;
mod metric;
mod outliers;
mod regression;
mod seasonal;
mod smoothing;
mod utility;

/// fdapy - Functional Data Analysis in Python
///
/// A Python package for functional data analysis, providing tools for:
/// - Functional data representation and manipulation
/// - Depth measures for functional data
/// - Distance metrics and semimetrics
/// - Basis representations (B-splines, Fourier)
/// - Clustering (k-means, fuzzy c-means)
/// - Regression (PCA, PLS, ridge)
/// - Smoothing (kernel, local polynomial)
/// - Seasonal analysis
/// - Outlier detection
#[pymodule]
fn _fdapy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // fdata functions
    m.add_function(wrap_pyfunction!(fdata::fdata_mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::fdata_mean_2d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::fdata_center_1d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::fdata_norm_lp_1d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::fdata_deriv_1d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::fdata_deriv_2d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::geometric_median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(fdata::geometric_median_2d, m)?)?;

    // depth functions
    m.add_function(wrap_pyfunction!(depth::depth_fm_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_fm_2d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_mode_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_mode_2d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_rp_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_rp_2d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_rt_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_rt_2d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_fsd_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_fsd_2d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_kfsd_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_kfsd_2d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_bd_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_mbd_1d, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_mei_1d, m)?)?;

    // metric functions
    m.add_function(wrap_pyfunction!(metric::metric_lp_self_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_lp_cross_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_lp_self_2d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_lp_cross_2d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_hausdorff_self_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_hausdorff_cross_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_hausdorff_self_2d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_hausdorff_cross_2d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_dtw_self_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::metric_dtw_cross_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::semimetric_fourier_self_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::semimetric_fourier_cross_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::semimetric_hshift_self_1d, m)?)?;
    m.add_function(wrap_pyfunction!(metric::semimetric_hshift_cross_1d, m)?)?;

    // basis functions
    m.add_function(wrap_pyfunction!(basis::bspline_basis, m)?)?;
    m.add_function(wrap_pyfunction!(basis::fourier_basis, m)?)?;
    m.add_function(wrap_pyfunction!(basis::fourier_basis_with_period, m)?)?;
    m.add_function(wrap_pyfunction!(basis::fdata_to_basis_1d, m)?)?;
    m.add_function(wrap_pyfunction!(basis::basis_to_fdata_1d, m)?)?;
    m.add_function(wrap_pyfunction!(basis::pspline_fit_1d, m)?)?;
    m.add_function(wrap_pyfunction!(basis::fourier_fit_1d, m)?)?;
    m.add_function(wrap_pyfunction!(basis::select_basis_auto_1d, m)?)?;
    m.add_function(wrap_pyfunction!(basis::basis_gcv_1d, m)?)?;

    // clustering functions
    m.add_function(wrap_pyfunction!(clustering::kmeans_fd, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::fcmeans_fd, m)?)?;

    // regression functions
    m.add_function(wrap_pyfunction!(regression::fdata_to_pc_1d, m)?)?;
    m.add_function(wrap_pyfunction!(regression::fdata_to_pls_1d, m)?)?;
    m.add_function(wrap_pyfunction!(regression::ridge_regression_fit, m)?)?;

    // smoothing functions
    m.add_function(wrap_pyfunction!(smoothing::nadaraya_watson, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::local_linear, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::local_polynomial, m)?)?;

    // seasonal functions
    m.add_function(wrap_pyfunction!(seasonal::estimate_period_fft, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal::estimate_period_autocorr, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal::detect_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal::detect_multiple_periods, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal::seasonal_strength, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal::detect_seasonality_changes, m)?)?;

    // detrend functions
    m.add_function(wrap_pyfunction!(detrend::detrend, m)?)?;
    m.add_function(wrap_pyfunction!(detrend::decompose, m)?)?;

    // outlier functions
    m.add_function(wrap_pyfunction!(outliers::outliers_depth, m)?)?;
    m.add_function(wrap_pyfunction!(outliers::outliers_lrt, m)?)?;

    // utility functions
    m.add_function(wrap_pyfunction!(utility::integrate_simpson, m)?)?;
    m.add_function(wrap_pyfunction!(utility::inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(utility::inner_product_matrix, m)?)?;

    Ok(())
}
