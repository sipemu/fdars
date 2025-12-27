// Seasonal analysis for functional data

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Helper to convert row-major numpy array to column-major flat vector
fn to_col_major(data: &ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (nrow, ncol) = data.dim();
    (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data[[i, j]]))
        .collect()
}

/// Estimate dominant period using FFT
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// min_period : float or None
///     Minimum period to consider
/// max_period : float or None
///     Maximum period to consider
///
/// Returns
/// -------
/// dict with keys: 'periods', 'frequencies', 'powers', 'confidences'
///     Lists of detected periods for each sample
#[pyfunction]
#[pyo3(signature = (data, argvals, min_period=None, max_period=None))]
pub fn estimate_period_fft<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    min_period: Option<f64>,
    max_period: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("period", f64::NAN)?;
        dict.set_item("frequency", f64::NAN)?;
        dict.set_item("power", 0.0)?;
        dict.set_item("confidence", 0.0)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // estimate_period_fft(data, n, m, argvals) -> PeriodEstimate (single result)
    let result = fdars_core::seasonal::estimate_period_fft(&data_flat, n, n_points, &argvals_vec);

    dict.set_item("period", result.period)?;
    dict.set_item("frequency", result.frequency)?;
    dict.set_item("power", result.power)?;
    dict.set_item("confidence", result.confidence)?;

    Ok(dict)
}

/// Estimate period using autocorrelation
#[pyfunction]
#[pyo3(signature = (data, argvals, min_period=None, max_period=None))]
pub fn estimate_period_autocorr<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    min_period: Option<f64>,
    max_period: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("periods", Vec::<f64>::new().into_pyarray(py))?;
        dict.set_item("confidences", Vec::<f64>::new().into_pyarray(py))?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    let range = argvals_vec.last().unwrap() - argvals_vec.first().unwrap();
    let min_p = min_period.unwrap_or(range / (n_points as f64 / 2.0));
    let max_p = max_period.unwrap_or(range);

    // Use half of the data length as max lag
    let max_lag = n_points / 2;
    let result =
        fdars_core::seasonal::estimate_period_acf(&data_flat, n, n_points, &argvals_vec, max_lag);

    // estimate_period_acf returns a single PeriodEstimate, wrap in vectors
    dict.set_item("periods", vec![result.period].into_pyarray(py))?;
    dict.set_item("confidences", vec![result.confidence].into_pyarray(py))?;

    Ok(dict)
}

/// Detect peaks in functional data
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// min_prominence : float
///     Minimum peak prominence (default 0.1)
/// min_distance : float or None
///     Minimum distance between peaks
///
/// Returns
/// -------
/// dict with keys: 'peak_times', 'peak_values', 'prominences', 'mean_period'
#[pyfunction]
#[pyo3(signature = (data, argvals, min_prominence=0.1, min_distance=None))]
pub fn detect_peaks<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    min_prominence: f64,
    min_distance: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("peak_times", pyo3::types::PyList::empty(py))?;
        dict.set_item("peak_values", pyo3::types::PyList::empty(py))?;
        dict.set_item("prominences", pyo3::types::PyList::empty(py))?;
        dict.set_item("mean_period", f64::NAN)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let min_dist = min_distance;
    let min_prom = Some(min_prominence);

    // detect_peaks(data, n, m, argvals, min_distance, min_prominence, smooth_first, smooth_nbasis)
    let result = fdars_core::seasonal::detect_peaks(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        min_dist,
        min_prom,
        false,
        None,
    );

    // Convert peaks to lists of arrays
    let peak_times = pyo3::types::PyList::new(
        py,
        result.peaks.iter().map(|sample_peaks| {
            let times: Vec<f64> = sample_peaks.iter().map(|p| p.time).collect();
            times.into_pyarray(py)
        }),
    )?;
    let peak_values = pyo3::types::PyList::new(
        py,
        result.peaks.iter().map(|sample_peaks| {
            let values: Vec<f64> = sample_peaks.iter().map(|p| p.value).collect();
            values.into_pyarray(py)
        }),
    )?;
    let prominences = pyo3::types::PyList::new(
        py,
        result.peaks.iter().map(|sample_peaks| {
            let proms: Vec<f64> = sample_peaks.iter().map(|p| p.prominence).collect();
            proms.into_pyarray(py)
        }),
    )?;

    // inter_peak_distances is Vec<Vec<f64>>, flatten for simplicity
    let all_distances: Vec<f64> = result.inter_peak_distances.into_iter().flatten().collect();

    dict.set_item("peak_times", peak_times)?;
    dict.set_item("peak_values", peak_values)?;
    dict.set_item("prominences", prominences)?;
    dict.set_item("inter_peak_distances", all_distances.into_pyarray(py))?;
    dict.set_item("mean_period", result.mean_period)?;

    Ok(dict)
}

/// Detect multiple periods in functional data
#[pyfunction]
#[pyo3(signature = (data, argvals, max_periods=3, min_period=None, max_period=None))]
pub fn detect_multiple_periods<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    max_periods: usize,
    min_period: Option<f64>,
    max_period: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("periods", pyo3::types::PyList::empty(py))?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    let min_confidence = 0.5; // default confidence threshold
    let min_strength = 0.1; // default strength threshold

    // detect_multiple_periods(data, n, m, argvals, max_periods, min_confidence, min_strength)
    let results = fdars_core::seasonal::detect_multiple_periods(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        max_periods,
        min_confidence,
        min_strength,
    );

    // Convert to list of dicts (one per detected period)
    let periods_list = pyo3::types::PyList::new(
        py,
        results.iter().map(|p| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("period", p.period).unwrap();
            d.set_item("confidence", p.confidence).unwrap();
            d.set_item("strength", p.strength).unwrap();
            d.set_item("amplitude", p.amplitude).unwrap();
            d.set_item("phase", p.phase).unwrap();
            d.set_item("iteration", p.iteration).unwrap();
            d
        }),
    )?;

    dict.set_item("periods", periods_list)?;

    Ok(dict)
}

/// Compute seasonal strength
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
///     Method for strength computation: "variance" or "spectral"
///
/// Returns
/// -------
/// strengths : ndarray, shape (n_samples,)
///     Seasonal strength for each sample
#[pyfunction]
#[pyo3(signature = (data, argvals, period, method="variance"))]
pub fn seasonal_strength<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    period: f64,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    if n == 0 || n_points == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // These functions return a single f64 for the mean curve, wrap in vec for per-sample use
    let n_harmonics = 3; // default harmonics for variance method
    let strength = if method == "spectral" {
        fdars_core::seasonal::seasonal_strength_spectral(
            &data_flat,
            n,
            n_points,
            &argvals_vec,
            period,
        )
    } else {
        fdars_core::seasonal::seasonal_strength_variance(
            &data_flat,
            n,
            n_points,
            &argvals_vec,
            period,
            n_harmonics,
        )
    };

    // Return single strength value for the mean curve
    Ok(vec![strength].into_pyarray(py))
}

/// Detect changes in seasonality (onset/cessation)
#[pyfunction]
#[pyo3(signature = (data, argvals, period, window_size=None, threshold=0.3))]
pub fn detect_seasonality_changes<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    period: f64,
    window_size: Option<usize>,
    threshold: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 {
        dict.set_item("change_points", pyo3::types::PyList::empty(py))?;
        dict.set_item("strength_curves", pyo3::types::PyList::empty(py))?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();
    let win_size = window_size.unwrap_or((period.ceil() as usize).max(10)) as f64;
    let min_duration = period; // default min_duration is one period

    // detect_seasonality_changes(data, n, m, argvals, period, threshold, window_size, min_duration)
    let result = fdars_core::seasonal::detect_seasonality_changes(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        period,
        threshold,
        win_size,
        min_duration,
    );

    // Convert results - returns a single ChangeDetectionResult
    let change_points = pyo3::types::PyList::new(
        py,
        result.change_points.iter().map(|cp| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("time", cp.time).unwrap();
            d.set_item("change_type", format!("{:?}", cp.change_type))
                .unwrap();
            d.set_item("strength_before", cp.strength_before).unwrap();
            d.set_item("strength_after", cp.strength_after).unwrap();
            d
        }),
    )?;

    dict.set_item("change_points", change_points)?;
    dict.set_item("strength_curve", result.strength_curve.into_pyarray(py))?;

    Ok(dict)
}
