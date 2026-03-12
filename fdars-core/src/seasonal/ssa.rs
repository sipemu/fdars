use crate::matrix::FdMatrix;

use super::compute_mean_curve;

/// Result of Singular Spectrum Analysis.
#[derive(Debug, Clone)]
pub struct SsaResult {
    /// Reconstructed trend component
    pub trend: Vec<f64>,
    /// Reconstructed seasonal/periodic component
    pub seasonal: Vec<f64>,
    /// Noise/residual component
    pub noise: Vec<f64>,
    /// Singular values from SVD (sorted descending)
    pub singular_values: Vec<f64>,
    /// Contribution of each component (proportion of variance)
    pub contributions: Vec<f64>,
    /// Window length used for embedding
    pub window_length: usize,
    /// Number of components extracted
    pub n_components: usize,
    /// Detected period (if any significant periodicity found)
    pub detected_period: f64,
    /// Confidence score for detected period
    pub confidence: f64,
}

/// Singular Spectrum Analysis (SSA) for time series decomposition.
///
/// SSA is a model-free, non-parametric method for decomposing a time series
/// into trend, oscillatory (seasonal), and noise components using singular
/// value decomposition of the trajectory matrix.
///
/// # Algorithm
/// 1. **Embedding**: Convert series into trajectory matrix using sliding windows
/// 2. **Decomposition**: SVD of trajectory matrix
/// 3. **Grouping**: Identify trend vs. periodic vs. noise components
/// 4. **Reconstruction**: Diagonal averaging to recover time series
///
/// # Arguments
/// * `values` - Time series values
/// * `window_length` - Embedding window length (L). If None, uses L = min(n/2, 50).
///   Larger values capture longer-term patterns but need longer series.
/// * `n_components` - Number of components to extract. If None, uses 10.
/// * `trend_components` - Indices of components for trend (0-based). If None, auto-detect.
/// * `seasonal_components` - Indices of components for seasonal. If None, auto-detect.
///
/// # Returns
/// `SsaResult` with decomposed components and diagnostics.
///
/// # Example
/// ```rust
/// use fdars_core::seasonal::ssa;
/// use std::f64::consts::PI;
///
/// // Signal with trend + seasonal + noise
/// let n = 100;
/// let values: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64;
///         0.01 * t + (2.0 * PI * t / 12.0).sin() + 0.1 * (i as f64 * 0.1).sin()
///     })
///     .collect();
///
/// let result = ssa(&values, None, None, None, None);
/// assert!(result.detected_period > 0.0);
/// ```
pub fn ssa(
    values: &[f64],
    window_length: Option<usize>,
    n_components: Option<usize>,
    trend_components: Option<&[usize]>,
    seasonal_components: Option<&[usize]>,
) -> SsaResult {
    let n = values.len();

    // Default window length: min(n/2, 50)
    let l = window_length.unwrap_or_else(|| (n / 2).clamp(2, 50));

    if n < 4 || l < 2 || l > n / 2 {
        return SsaResult {
            trend: values.to_vec(),
            seasonal: vec![0.0; n],
            noise: vec![0.0; n],
            singular_values: vec![],
            contributions: vec![],
            window_length: l,
            n_components: 0,
            detected_period: 0.0,
            confidence: 0.0,
        };
    }

    // Number of columns in trajectory matrix
    let k = n - l + 1;

    // Step 1: Embedding - create trajectory matrix (L x K)
    let trajectory = embed_trajectory(values, l, k);

    // Step 2: SVD decomposition
    let (u, sigma, vt) = svd_decompose(&trajectory, l, k);

    // Determine number of components to use
    let max_components = sigma.len();
    let n_comp = n_components.unwrap_or(10).min(max_components);

    // Compute contributions (proportion of total variance)
    let total_var: f64 = sigma.iter().map(|&s| s * s).sum();
    let contributions: Vec<f64> = sigma
        .iter()
        .take(n_comp)
        .map(|&s| s * s / total_var.max(1e-15))
        .collect();

    // Step 3: Grouping - identify trend and seasonal components
    let (trend_idx, seasonal_idx, detected_period, confidence) =
        if trend_components.is_some() || seasonal_components.is_some() {
            // Use provided groupings
            let t_idx: Vec<usize> = trend_components.map(<[usize]>::to_vec).unwrap_or_default();
            let s_idx: Vec<usize> = seasonal_components
                .map(<[usize]>::to_vec)
                .unwrap_or_default();
            (t_idx, s_idx, 0.0, 0.0)
        } else {
            // Auto-detect groupings
            auto_group_ssa_components(&u, &sigma, l, k, n_comp)
        };

    // Step 4: Reconstruction via diagonal averaging
    let trend = reconstruct_grouped(&u, &sigma, &vt, l, k, n, &trend_idx);
    let seasonal = reconstruct_grouped(&u, &sigma, &vt, l, k, n, &seasonal_idx);

    // Noise is the remainder
    let noise: Vec<f64> = values
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&y, &t), &s)| y - t - s)
        .collect();

    SsaResult {
        trend,
        seasonal,
        noise,
        singular_values: sigma.into_iter().take(n_comp).collect(),
        contributions,
        window_length: l,
        n_components: n_comp,
        detected_period,
        confidence,
    }
}

/// Create trajectory matrix by embedding the time series.
pub(super) fn embed_trajectory(values: &[f64], l: usize, k: usize) -> Vec<f64> {
    // Trajectory matrix is L x K, stored column-major
    let mut trajectory = vec![0.0; l * k];

    for j in 0..k {
        for i in 0..l {
            trajectory[i + j * l] = values[i + j];
        }
    }

    trajectory
}

/// SVD decomposition of trajectory matrix using nalgebra.
pub(super) fn svd_decompose(
    trajectory: &[f64],
    l: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use nalgebra::{DMatrix, SVD};

    // Create nalgebra matrix (column-major)
    let mat = DMatrix::from_column_slice(l, k, trajectory);

    // Compute SVD
    let svd = SVD::new(mat, true, true);

    // Extract components (SVD::new with compute_u/v=true always produces both,
    // but handle gracefully in case of degenerate input)
    let u_mat = match svd.u {
        Some(u) => u,
        None => return (vec![], vec![], vec![]),
    };
    let vt_mat = match svd.v_t {
        Some(vt) => vt,
        None => return (vec![], vec![], vec![]),
    };
    let sigma = svd.singular_values;

    // Convert to flat vectors
    let u: Vec<f64> = u_mat.iter().copied().collect();
    let sigma_vec: Vec<f64> = sigma.iter().copied().collect();
    let vt: Vec<f64> = vt_mat.iter().copied().collect();

    (u, sigma_vec, vt)
}

pub(super) enum SsaComponentKind {
    Trend,
    Seasonal(f64),
    Noise,
}

/// Classify an SSA component as trend, seasonal, or noise.
pub(super) fn classify_ssa_component(u_col: &[f64], trend_count: usize) -> SsaComponentKind {
    if is_trend_component(u_col) && trend_count < 2 {
        SsaComponentKind::Trend
    } else {
        let (is_periodic, period) = is_periodic_component(u_col);
        if is_periodic {
            SsaComponentKind::Seasonal(period)
        } else {
            SsaComponentKind::Noise
        }
    }
}

/// Apply default groupings when auto-detection finds nothing.
pub(super) fn apply_ssa_grouping_defaults(
    trend_idx: &mut Vec<usize>,
    seasonal_idx: &mut Vec<usize>,
    n_comp: usize,
) {
    if trend_idx.is_empty() && n_comp > 0 {
        trend_idx.push(0);
    }
    if seasonal_idx.is_empty() && n_comp >= 3 {
        seasonal_idx.push(1);
        if n_comp > 2 {
            seasonal_idx.push(2);
        }
    }
}

/// Auto-detect trend and seasonal component groupings.
pub(super) fn auto_group_ssa_components(
    u: &[f64],
    sigma: &[f64],
    l: usize,
    _k: usize,
    n_comp: usize,
) -> (Vec<usize>, Vec<usize>, f64, f64) {
    let mut trend_idx = Vec::new();
    let mut seasonal_idx = Vec::new();
    let mut detected_period = 0.0;
    let mut confidence = 0.0;

    for i in 0..n_comp.min(sigma.len()) {
        let u_col: Vec<f64> = (0..l).map(|j| u[j + i * l]).collect();
        match classify_ssa_component(&u_col, trend_idx.len()) {
            SsaComponentKind::Trend => trend_idx.push(i),
            SsaComponentKind::Seasonal(period) => {
                seasonal_idx.push(i);
                if detected_period == 0.0 && period > 0.0 {
                    detected_period = period;
                    confidence = sigma[i] / sigma[0].max(1e-15);
                }
            }
            SsaComponentKind::Noise => {}
        }
    }

    apply_ssa_grouping_defaults(&mut trend_idx, &mut seasonal_idx, n_comp);
    (trend_idx, seasonal_idx, detected_period, confidence)
}

/// Check if a singular vector represents a trend component.
pub(super) fn is_trend_component(u_col: &[f64]) -> bool {
    let n = u_col.len();
    if n < 3 {
        return false;
    }

    // Count sign changes in the vector
    let mut sign_changes = 0;
    for i in 1..n {
        if u_col[i] * u_col[i - 1] < 0.0 {
            sign_changes += 1;
        }
    }

    // Trend components have very few sign changes
    sign_changes <= n / 10
}

/// Check if a singular vector represents a periodic component.
pub(super) fn is_periodic_component(u_col: &[f64]) -> (bool, f64) {
    let n = u_col.len();
    if n < 4 {
        return (false, 0.0);
    }

    // Use autocorrelation to detect periodicity
    let mean: f64 = u_col.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = u_col.iter().map(|&x| x - mean).collect();

    let var: f64 = centered.iter().map(|&x| x * x).sum();
    if var < 1e-15 {
        return (false, 0.0);
    }

    // Find first significant peak in autocorrelation
    let mut best_period = 0.0;
    let mut best_acf = 0.0;

    for lag in 2..n / 2 {
        let mut acf = 0.0;
        for i in 0..(n - lag) {
            acf += centered[i] * centered[i + lag];
        }
        acf /= var;

        if acf > best_acf && acf > 0.3 {
            best_acf = acf;
            best_period = lag as f64;
        }
    }

    let is_periodic = best_acf > 0.3 && best_period > 0.0;
    (is_periodic, best_period)
}

/// Reconstruct time series from grouped components via diagonal averaging.
pub(super) fn reconstruct_grouped(
    u: &[f64],
    sigma: &[f64],
    vt: &[f64],
    l: usize,
    k: usize,
    n: usize,
    group_idx: &[usize],
) -> Vec<f64> {
    if group_idx.is_empty() {
        return vec![0.0; n];
    }

    // Sum of rank-1 matrices for this group
    let mut grouped_matrix = vec![0.0; l * k];

    for &idx in group_idx {
        if idx >= sigma.len() {
            continue;
        }

        let s = sigma[idx];

        // Add s * u_i * v_i^T
        for j in 0..k {
            for i in 0..l {
                let u_val = u[i + idx * l];
                let v_val = vt[idx + j * sigma.len().min(l)]; // v_t is stored as K x min(L,K)
                grouped_matrix[i + j * l] += s * u_val * v_val;
            }
        }
    }

    // Diagonal averaging (Hankelization)
    diagonal_average(&grouped_matrix, l, k, n)
}

/// Diagonal averaging to convert trajectory matrix back to time series.
pub(super) fn diagonal_average(matrix: &[f64], l: usize, k: usize, n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n];
    let mut counts = vec![0.0; n];

    // Average along anti-diagonals
    for j in 0..k {
        for i in 0..l {
            let idx = i + j; // Position in original series
            if idx < n {
                result[idx] += matrix[i + j * l];
                counts[idx] += 1.0;
            }
        }
    }

    // Normalize by counts
    for i in 0..n {
        if counts[i] > 0.0 {
            result[i] /= counts[i];
        }
    }

    result
}

/// Compute SSA for functional data (multiple curves).
///
/// # Arguments
/// * `data` - Column-major matrix (n x m) of functional data
/// * `window_length` - SSA window length. If None, auto-determined.
/// * `n_components` - Number of SSA components. Default: 10.
///
/// # Returns
/// `SsaResult` computed from the mean curve.
pub fn ssa_fdata(
    data: &FdMatrix,
    window_length: Option<usize>,
    n_components: Option<usize>,
) -> SsaResult {
    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Run SSA on mean curve
    ssa(&mean_curve, window_length, n_components, None, None)
}

/// Detect seasonality using SSA.
///
/// # Arguments
/// * `values` - Time series values
/// * `window_length` - SSA window length
/// * `confidence_threshold` - Minimum confidence for positive detection
///
/// # Returns
/// Tuple of (is_seasonal, detected_period, confidence)
pub fn ssa_seasonality(
    values: &[f64],
    window_length: Option<usize>,
    confidence_threshold: Option<f64>,
) -> (bool, f64, f64) {
    let result = ssa(values, window_length, None, None, None);

    let threshold = confidence_threshold.unwrap_or(0.1);
    let is_seasonal = result.confidence >= threshold && result.detected_period > 0.0;

    (is_seasonal, result.detected_period, result.confidence)
}
