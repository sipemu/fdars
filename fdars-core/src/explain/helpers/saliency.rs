//! Saliency map and domain selection helpers.

use crate::matrix::FdMatrix;

/// Compute saliency map: saliency[(i,j)] = sum_k weight_k * (scores[(i,k)] - mean_k) * rotation[(j,k)].
pub(crate) fn compute_saliency_map(
    scores: &FdMatrix,
    mean_scores: &[f64],
    weights: &[f64],
    rotation: &FdMatrix,
    n: usize,
    m: usize,
    ncomp: usize,
) -> FdMatrix {
    let mut saliency_map = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += weights[k] * (scores[(i, k)] - mean_scores[k]) * rotation[(j, k)];
            }
            saliency_map[(i, j)] = val;
        }
    }
    saliency_map
}

/// Mean absolute value per column of an n x m matrix.
pub(crate) fn mean_absolute_column(mat: &FdMatrix, n: usize, m: usize) -> Vec<f64> {
    let mut result = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            result[j] += mat[(i, j)].abs();
        }
        result[j] /= n as f64;
    }
    result
}

/// Reconstruct delta function from delta scores and rotation matrix.
pub(crate) fn reconstruct_delta_function(
    delta_scores: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    m: usize,
) -> Vec<f64> {
    let mut delta_function = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            delta_function[j] += delta_scores[k] * rotation[(j, k)];
        }
    }
    delta_function
}

/// Compute domain selection from beta_t.
pub(crate) fn compute_domain_selection(
    beta_t: &[f64],
    window_width: usize,
    threshold: f64,
) -> Option<super::super::sensitivity::DomainSelectionResult> {
    use super::super::sensitivity::DomainSelectionResult;

    let m = beta_t.len();
    if m == 0 || window_width == 0 || window_width > m || threshold <= 0.0 {
        return None;
    }

    let pointwise_importance: Vec<f64> = beta_t.iter().map(|&b| b * b).collect();
    let total_imp: f64 = pointwise_importance.iter().sum();
    if total_imp == 0.0 {
        return Some(DomainSelectionResult {
            pointwise_importance,
            intervals: vec![],
            window_width,
            threshold,
        });
    }

    // Sliding window with running sum
    let mut window_sum: f64 = pointwise_importance[..window_width].iter().sum();
    let mut raw_intervals: Vec<(usize, usize, f64)> = Vec::new();
    if window_sum / total_imp >= threshold {
        raw_intervals.push((0, window_width - 1, window_sum));
    }
    for start in 1..=(m - window_width) {
        window_sum -= pointwise_importance[start - 1];
        window_sum += pointwise_importance[start + window_width - 1];
        if window_sum / total_imp >= threshold {
            raw_intervals.push((start, start + window_width - 1, window_sum));
        }
    }

    let mut intervals = merge_overlapping_intervals(raw_intervals);
    intervals.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(DomainSelectionResult {
        pointwise_importance,
        intervals,
        window_width,
        threshold,
    })
}

/// Merge overlapping intervals, accumulating importance.
fn merge_overlapping_intervals(
    raw: Vec<(usize, usize, f64)>,
) -> Vec<super::super::sensitivity::ImportantInterval> {
    use super::super::sensitivity::ImportantInterval;

    let mut intervals: Vec<ImportantInterval> = Vec::new();
    for (s, e, imp) in raw {
        if let Some(last) = intervals.last_mut() {
            if s <= last.end_idx + 1 {
                last.end_idx = e;
                last.importance += imp;
                continue;
            }
        }
        intervals.push(ImportantInterval {
            start_idx: s,
            end_idx: e,
            importance: imp,
        });
    }
    intervals
}
