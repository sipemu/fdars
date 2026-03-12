use super::helpers::{build_band, percentile_sorted};
use super::{NonConformityScore, ToleranceBand};
use crate::fdata::mean_1d;
use crate::matrix::FdMatrix;
use rand::prelude::*;

// ─── Conformal Prediction Band ──────────────────────────────────────────────

/// Compute a non-conformity score for a single curve against a center function.
fn nonconformity_score(
    data: &FdMatrix,
    i: usize,
    center: &[f64],
    m: usize,
    score_type: NonConformityScore,
) -> f64 {
    match score_type {
        NonConformityScore::SupNorm => (0..m)
            .map(|j| (data[(i, j)] - center[j]).abs())
            .fold(0.0_f64, f64::max),
        NonConformityScore::L2 => {
            let ss: f64 = (0..m).map(|j| (data[(i, j)] - center[j]).powi(2)).sum();
            ss.sqrt()
        }
    }
}

/// Build a subset matrix from selected row indices.
fn subset_rows(data: &FdMatrix, indices: &[usize], m: usize) -> FdMatrix {
    let n_sub = indices.len();
    let mut sub = FdMatrix::zeros(n_sub, m);
    for (new_i, &old_i) in indices.iter().enumerate() {
        for j in 0..m {
            sub[(new_i, j)] = data[(old_i, j)];
        }
    }
    sub
}

/// Compute a distribution-free conformal prediction band.
///
/// Splits data into training and calibration sets, computes a non-conformity
/// score on calibration curves, and uses the conformal quantile to build
/// a prediction band.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `cal_fraction` — Fraction of data used for calibration (e.g., 0.2)
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `score_type` — [`NonConformityScore::SupNorm`] or [`NonConformityScore::L2`]
/// * `seed` — Random seed for the train/calibration split
///
/// # Returns
/// `Some(ToleranceBand)` on success, `None` if inputs are invalid.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{conformal_prediction_band, NonConformityScore};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(40, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = conformal_prediction_band(&data, 0.2, 0.95, NonConformityScore::SupNorm, 42).unwrap();
/// // SupNorm gives constant half-width across all evaluation points
/// let hw = band.half_width[0];
/// assert!(band.half_width.iter().all(|&h| (h - hw).abs() < 1e-12));
/// ```
pub fn conformal_prediction_band(
    data: &FdMatrix,
    cal_fraction: f64,
    coverage: f64,
    score_type: NonConformityScore,
    seed: u64,
) -> Option<ToleranceBand> {
    let (n, m) = data.shape();
    if n < 4
        || m == 0
        || cal_fraction <= 0.0
        || cal_fraction >= 1.0
        || coverage <= 0.0
        || coverage >= 1.0
    {
        return None;
    }

    let n_cal = ((n as f64) * cal_fraction).max(1.0).min((n - 2) as f64) as usize;
    let n_train = n - n_cal;

    // Random permutation for split
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let train_data = subset_rows(data, &indices[..n_train], m);
    let center = mean_1d(&train_data);

    // Compute non-conformity scores on calibration set
    let cal_idx = &indices[n_train..];
    let mut scores: Vec<f64> = cal_idx
        .iter()
        .map(|&i| nonconformity_score(data, i, &center, m, score_type))
        .collect();

    // Conformal quantile: ceil((n_cal + 1) * coverage) / n_cal
    let level = (((n_cal + 1) as f64 * coverage).ceil() / n_cal as f64).min(1.0);
    let q = percentile_sorted(&mut scores, level);

    // Build band depending on score type
    let half_width = match score_type {
        NonConformityScore::SupNorm => vec![q; m],
        NonConformityScore::L2 => vec![q / (m as f64).sqrt(); m],
    };

    Some(build_band(center, half_width))
}
