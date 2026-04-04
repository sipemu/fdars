//! Stability analysis and statistical helpers.

// Re-export the canonical quantile from helpers for backward compatibility.
pub(crate) use crate::helpers::quantile_sorted;

/// Sample standard deviation of a slice.
fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    // Bessel's correction for sample std
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Compute average ranks of a slice (1-based, average ranks for ties).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (values[idx[j]] - values[idx[i]]).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation between two equal-length slices.
fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = compute_ranks(a);
    let rb = compute_ranks(b);
    let mean_a: f64 = ra.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da2 = 0.0;
    let mut db2 = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let denom = (da2 * db2).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

/// Mean pairwise Spearman rank correlation across a set of vectors.
fn mean_pairwise_spearman(vectors: &[Vec<f64>]) -> f64 {
    let n = vectors.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += spearman_rank_correlation(&vectors[i], &vectors[j]);
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Compute pointwise mean, std, and coefficient of variation from bootstrap samples.
fn pointwise_mean_std_cv(samples: &[Vec<f64>], length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let mut mean = vec![0.0; length];
    let mut std = vec![0.0; length];
    for j in 0..length {
        let vals: Vec<f64> = samples.iter().map(|s| s[j]).collect();
        mean[j] = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|&v| (v - mean[j]).powi(2)).sum::<f64>() / (n - 1) as f64;
        std[j] = var.sqrt();
    }
    let eps = 1e-15;
    let cv: Vec<f64> = (0..length)
        .map(|j| {
            if mean[j].abs() > eps {
                std[j] / mean[j].abs()
            } else {
                0.0
            }
        })
        .collect();
    (mean, std, cv)
}

/// Compute per-component std from bootstrap coefficient vectors.
fn coefficient_std_from_bootstrap(all_coefs: &[Vec<f64>], ncomp: usize) -> Vec<f64> {
    (0..ncomp)
        .map(|k| {
            let vals: Vec<f64> = all_coefs.iter().map(|c| c[k]).collect();
            sample_std(&vals)
        })
        .collect()
}

/// Build stability result from collected bootstrap data.
pub(crate) fn build_stability_result(
    all_beta_t: &[Vec<f64>],
    all_coefs: &[Vec<f64>],
    all_abs_coefs: &[Vec<f64>],
    all_metrics: &[f64],
    m: usize,
    ncomp: usize,
) -> Option<super::super::advanced::StabilityAnalysisResult> {
    use super::super::advanced::StabilityAnalysisResult;

    let n_success = all_beta_t.len();
    if n_success < 2 {
        return None;
    }
    let (_mean, beta_t_std, beta_t_cv) = pointwise_mean_std_cv(all_beta_t, m);
    let coefficient_std = coefficient_std_from_bootstrap(all_coefs, ncomp);
    let metric_std = sample_std(all_metrics);
    let importance_stability = mean_pairwise_spearman(all_abs_coefs);

    Some(StabilityAnalysisResult {
        beta_t_std,
        coefficient_std,
        metric_std,
        beta_t_cv,
        importance_stability,
        n_boot_success: n_success,
    })
}
