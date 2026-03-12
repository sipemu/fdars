use crate::fdata::deriv_1d;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::{compute_prominence, Peak, PeakDetectionResult};

/// Optionally smooth data using Fourier basis before peak detection.
fn smooth_for_peaks(
    data: &FdMatrix,
    argvals: &[f64],
    smooth_first: bool,
    smooth_nbasis: Option<usize>,
) -> Vec<f64> {
    if !smooth_first {
        return data.as_slice().to_vec();
    }
    let nbasis = smooth_nbasis
        .unwrap_or_else(|| crate::basis::select_fourier_nbasis_gcv(data, argvals, 5, 25));
    if let Ok(result) = crate::basis::fourier_fit_1d(data, argvals, nbasis) {
        result.fitted.into_vec()
    } else {
        data.as_slice().to_vec()
    }
}

/// Detect peaks in a single curve using derivative zero-crossings.
fn detect_peaks_single_curve(
    curve: &[f64],
    d1: &[f64],
    argvals: &[f64],
    min_dist_points: usize,
    min_prominence: Option<f64>,
    data_range: f64,
) -> (Vec<Peak>, Vec<f64>) {
    let m = curve.len();
    let mut peak_indices = Vec::new();
    for j in 1..m {
        if d1[j - 1] > 0.0 && d1[j] <= 0.0 {
            let idx = if (d1[j - 1] - d1[j]).abs() > 1e-15 {
                j - 1
            } else {
                j
            };

            if peak_indices.is_empty()
                || idx - peak_indices[peak_indices.len() - 1] >= min_dist_points
            {
                peak_indices.push(idx);
            }
        }
    }

    let mut peaks: Vec<Peak> = peak_indices
        .iter()
        .map(|&idx| {
            let prominence = compute_prominence(curve, idx) / data_range;
            Peak {
                time: argvals[idx],
                value: curve[idx],
                prominence,
            }
        })
        .collect();

    if let Some(min_prom) = min_prominence {
        peaks.retain(|p| p.prominence >= min_prom);
    }

    let distances: Vec<f64> = peaks.windows(2).map(|w| w[1].time - w[0].time).collect();

    (peaks, distances)
}

/// Detect peaks in functional data.
///
/// Uses derivative zero-crossings to find local maxima, with optional
/// Fourier basis smoothing and filtering by minimum distance and prominence.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `min_distance` - Minimum time between peaks (None = no constraint)
/// * `min_prominence` - Minimum prominence (0-1 scale, None = no filter)
/// * `smooth_first` - Whether to smooth data before peak detection using Fourier basis
/// * `smooth_nbasis` - Number of Fourier basis functions. If None and smooth_first=true,
///   uses GCV to automatically select optimal nbasis (range 5-25).
pub fn detect_peaks(
    data: &FdMatrix,
    argvals: &[f64],
    min_distance: Option<f64>,
    min_prominence: Option<f64>,
    smooth_first: bool,
    smooth_nbasis: Option<usize>,
) -> PeakDetectionResult {
    let (n, m) = data.shape();
    if n == 0 || m < 3 || argvals.len() != m {
        return PeakDetectionResult {
            peaks: Vec::new(),
            inter_peak_distances: Vec::new(),
            mean_period: f64::NAN,
        };
    }

    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let min_dist_points = min_distance.map(|d| (d / dt).round() as usize).unwrap_or(1);

    let work_data = smooth_for_peaks(data, argvals, smooth_first, smooth_nbasis);

    // Compute first derivative
    let work_mat = FdMatrix::from_column_major(work_data.clone(), n, m).unwrap();
    let deriv1 = deriv_1d(&work_mat, argvals, 1).into_vec();

    // Compute data range for prominence normalization
    let data_range: f64 = {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in work_data.iter() {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        (max_val - min_val).max(1e-15)
    };

    // Find peaks for each sample
    let results: Vec<(Vec<Peak>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| work_data[i + j * n]).collect();
            let d1: Vec<f64> = (0..m).map(|j| deriv1[i + j * n]).collect();
            detect_peaks_single_curve(
                &curve,
                &d1,
                argvals,
                min_dist_points,
                min_prominence,
                data_range,
            )
        })
        .collect();

    let peaks: Vec<Vec<Peak>> = results.iter().map(|(p, _)| p.clone()).collect();
    let inter_peak_distances: Vec<Vec<f64>> = results.iter().map(|(_, d)| d.clone()).collect();

    // Compute mean period from all inter-peak distances
    let all_distances: Vec<f64> = inter_peak_distances.iter().flatten().cloned().collect();
    let mean_period = if all_distances.is_empty() {
        f64::NAN
    } else {
        all_distances.iter().sum::<f64>() / all_distances.len() as f64
    };

    PeakDetectionResult {
        peaks,
        inter_peak_distances,
        mean_period,
    }
}
