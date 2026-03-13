use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ============================================================================
// STL Decomposition (Cleveland et al., 1990)
// ============================================================================

/// Result of STL decomposition including robustness weights.
#[derive(Debug, Clone)]
pub struct StlResult {
    /// Trend component (n x m)
    pub trend: FdMatrix,
    /// Seasonal component (n x m)
    pub seasonal: FdMatrix,
    /// Remainder/residual component (n x m)
    pub remainder: FdMatrix,
    /// Robustness weights per point (n x m)
    pub weights: FdMatrix,
    /// Period used for decomposition
    pub period: usize,
    /// Seasonal smoothing window
    pub s_window: usize,
    /// Trend smoothing window
    pub t_window: usize,
    /// Number of inner loop iterations performed
    pub inner_iterations: usize,
    /// Number of outer loop iterations performed
    pub outer_iterations: usize,
}

/// Configuration for STL decomposition.
///
/// Collects all tuning parameters for [`stl_decompose_with_config`], with sensible
/// defaults obtained via [`StlConfig::default()`].
///
/// # Example
/// ```no_run
/// use fdars_core::detrend::stl::StlConfig;
///
/// let config = StlConfig {
///     robust: true,
///     s_window: Some(13),
///     ..StlConfig::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct StlConfig {
    /// Seasonal smoothing window (default: `None` for auto = 7).
    pub s_window: Option<usize>,
    /// Trend smoothing window (default: `None` for auto).
    pub t_window: Option<usize>,
    /// Low-pass filter window (default: `None` for auto = period).
    pub l_window: Option<usize>,
    /// Whether to use robust (bisquare) weights (default: false).
    pub robust: bool,
    /// Number of inner loop iterations (default: `None` for auto = 2).
    pub inner_iterations: Option<usize>,
    /// Number of outer loop iterations (default: `None` for auto = 1 or 15 if robust).
    pub outer_iterations: Option<usize>,
}

/// STL Decomposition using a [`StlConfig`] struct.
///
/// This is the config-based alternative to [`stl_decompose`]. It takes data
/// and period directly, and reads all tuning parameters from the config.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `period` — Seasonal period length
/// * `config` — Tuning parameters
pub fn stl_decompose_with_config(data: &FdMatrix, period: usize, config: &StlConfig) -> StlResult {
    stl_decompose(
        data,
        period,
        config.s_window,
        config.t_window,
        config.l_window,
        config.robust,
        config.inner_iterations,
        config.outer_iterations,
    )
}

/// STL Decomposition: Seasonal and Trend decomposition using LOESS.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `period` - Seasonal period length
/// * `s_window` - Seasonal smoothing window (None for auto)
/// * `t_window` - Trend smoothing window (None for auto)
/// * `l_window` - Low-pass filter window (None for auto)
/// * `robust` - Whether to use robust weights
/// * `inner_iterations` - Number of inner loop iterations (None for auto)
/// * `outer_iterations` - Number of outer loop iterations (None for auto)
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::detrend::stl::stl_decompose;
///
/// let n = 3;
/// let m = 40; // must be >= 2 * period
/// let data = FdMatrix::from_column_major(
///     (0..n * m).map(|i| {
///         let t = (i % m) as f64;
///         (t * std::f64::consts::PI / 5.0).sin() + t * 0.01
///     }).collect(),
///     n, m,
/// ).unwrap();
/// let result = stl_decompose(&data, 10, None, None, None, false, None, None);
/// assert_eq!(result.trend.shape(), (n, m));
/// assert_eq!(result.seasonal.shape(), (n, m));
/// assert_eq!(result.remainder.shape(), (n, m));
/// ```
pub fn stl_decompose(
    data: &FdMatrix,
    period: usize,
    s_window: Option<usize>,
    t_window: Option<usize>,
    l_window: Option<usize>,
    robust: bool,
    inner_iterations: Option<usize>,
    outer_iterations: Option<usize>,
) -> StlResult {
    let (n, m) = data.shape();
    if n == 0 || m < 2 * period || period < 2 {
        return StlResult {
            trend: FdMatrix::zeros(n, m),
            seasonal: FdMatrix::zeros(n, m),
            remainder: FdMatrix::from_slice(data.as_slice(), n, m)
                .unwrap_or_else(|_| FdMatrix::zeros(n, m)),
            weights: FdMatrix::from_column_major(vec![1.0; n * m], n, m)
                .unwrap_or_else(|_| FdMatrix::zeros(n, m)),
            period,
            s_window: 0,
            t_window: 0,
            inner_iterations: 0,
            outer_iterations: 0,
        };
    }
    let s_win = s_window.unwrap_or(7).max(3) | 1;
    let t_win = t_window.unwrap_or_else(|| {
        let ratio = 1.5 * period as f64 / (1.0 - 1.5 / s_win as f64);
        let val = ratio.ceil() as usize;
        val.max(3) | 1
    });
    let l_win = l_window.unwrap_or(period) | 1;
    let n_inner = inner_iterations.unwrap_or(2);
    let n_outer = outer_iterations.unwrap_or(if robust { 15 } else { 1 });
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
            stl_single_series(
                &curve, period, s_win, t_win, l_win, robust, n_inner, n_outer,
            )
        })
        .collect();
    let mut trend = FdMatrix::zeros(n, m);
    let mut seasonal = FdMatrix::zeros(n, m);
    let mut remainder = FdMatrix::zeros(n, m);
    let mut weights = FdMatrix::from_column_major(vec![1.0; n * m], n, m)
        .expect("dimension invariant: data.len() == n * m");
    for (i, (t, s, r, w)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[(i, j)] = t[j];
            seasonal[(i, j)] = s[j];
            remainder[(i, j)] = r[j];
            weights[(i, j)] = w[j];
        }
    }
    StlResult {
        trend,
        seasonal,
        remainder,
        weights,
        period,
        s_window: s_win,
        t_window: t_win,
        inner_iterations: n_inner,
        outer_iterations: n_outer,
    }
}

fn stl_single_series(
    data: &[f64],
    period: usize,
    s_window: usize,
    t_window: usize,
    l_window: usize,
    robust: bool,
    n_inner: usize,
    n_outer: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = data.len();
    let mut trend = vec![0.0; m];
    let mut seasonal = vec![0.0; m];
    let mut weights = vec![1.0; m];
    for _outer in 0..n_outer {
        for _inner in 0..n_inner {
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend.iter())
                .map(|(&y, &t)| y - t)
                .collect();
            let cycle_smoothed = smooth_cycle_subseries(&detrended, period, s_window, &weights);
            let low_pass = stl_lowpass_filter(&cycle_smoothed, period, l_window);
            seasonal = cycle_smoothed
                .iter()
                .zip(low_pass.iter())
                .map(|(&c, &l)| c - l)
                .collect();
            let deseasonalized: Vec<f64> = data
                .iter()
                .zip(seasonal.iter())
                .map(|(&y, &s)| y - s)
                .collect();
            trend = weighted_loess(&deseasonalized, t_window, &weights);
        }
        if robust && _outer < n_outer - 1 {
            let remainder: Vec<f64> = data
                .iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&y, &t), &s)| y - t - s)
                .collect();
            weights = compute_robustness_weights(&remainder);
        }
    }
    let remainder: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&y, &t), &s)| y - t - s)
        .collect();
    (trend, seasonal, remainder, weights)
}

fn smooth_cycle_subseries(
    data: &[f64],
    period: usize,
    s_window: usize,
    weights: &[f64],
) -> Vec<f64> {
    let m = data.len();
    let n_cycles = m.div_ceil(period);
    let mut result = vec![0.0; m];
    for pos in 0..period {
        let mut subseries_idx: Vec<usize> = Vec::new();
        let mut subseries_vals: Vec<f64> = Vec::new();
        let mut subseries_weights: Vec<f64> = Vec::new();
        for cycle in 0..n_cycles {
            let idx = cycle * period + pos;
            if idx < m {
                subseries_idx.push(idx);
                subseries_vals.push(data[idx]);
                subseries_weights.push(weights[idx]);
            }
        }
        if subseries_vals.is_empty() {
            continue;
        }
        let smoothed = weighted_loess(&subseries_vals, s_window, &subseries_weights);
        for (i, &idx) in subseries_idx.iter().enumerate() {
            result[idx] = smoothed[i];
        }
    }
    result
}

fn stl_lowpass_filter(data: &[f64], period: usize, _l_window: usize) -> Vec<f64> {
    let ma1 = moving_average(data, period);
    let ma2 = moving_average(&ma1, period);
    moving_average(&ma2, 3)
}

fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    let m = data.len();
    if m == 0 || window == 0 {
        return data.to_vec();
    }
    let half = window / 2;
    let mut result = vec![0.0; m];
    for i in 0..m {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(m);
        let sum: f64 = data[start..end].iter().sum();
        let count = (end - start) as f64;
        result[i] = sum / count;
    }
    result
}

fn weighted_loess(data: &[f64], window: usize, weights: &[f64]) -> Vec<f64> {
    let m = data.len();
    if m == 0 {
        return vec![];
    }
    let half = window / 2;
    let mut result = vec![0.0; m];
    for i in 0..m {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(m);
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wxx = 0.0;
        let mut sum_wxy = 0.0;
        for j in start..end {
            let dist = (j as f64 - i as f64).abs() / (half.max(1) as f64);
            let tricube = if dist < 1.0 {
                (1.0 - dist.powi(3)).powi(3)
            } else {
                0.0
            };
            let w = tricube * weights[j];
            let x = j as f64;
            let y = data[j];
            sum_w += w;
            sum_wx += w * x;
            sum_wy += w * y;
            sum_wxx += w * x * x;
            sum_wxy += w * x * y;
        }
        if sum_w > 1e-10 {
            let denom = sum_w * sum_wxx - sum_wx * sum_wx;
            if denom.abs() > 1e-10 {
                let intercept = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom;
                let slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
                result[i] = intercept + slope * i as f64;
            } else {
                result[i] = sum_wy / sum_w;
            }
        } else {
            result[i] = data[i];
        }
    }
    result
}

fn compute_robustness_weights(residuals: &[f64]) -> Vec<f64> {
    let m = residuals.len();
    if m == 0 {
        return vec![];
    }
    let mut abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
    crate::helpers::sort_nan_safe(&mut abs_residuals);
    let median_idx = m / 2;
    let mad = if m % 2 == 0 {
        (abs_residuals[median_idx - 1] + abs_residuals[median_idx]) / 2.0
    } else {
        abs_residuals[median_idx]
    };
    let h = 6.0 * mad.max(1e-10);
    residuals
        .iter()
        .map(|&r| {
            let u = r.abs() / h;
            if u < 1.0 {
                (1.0 - u * u).powi(2)
            } else {
                0.0
            }
        })
        .collect()
}

/// Wrapper function for functional data STL decomposition.
pub fn stl_fdata(
    data: &FdMatrix,
    _argvals: &[f64],
    period: usize,
    s_window: Option<usize>,
    t_window: Option<usize>,
    robust: bool,
) -> StlResult {
    stl_decompose(data, period, s_window, t_window, None, robust, None, None)
}
