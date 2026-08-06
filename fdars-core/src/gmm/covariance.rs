//! Covariance accumulation and regularization for GMM components.

use super::CovType;

/// Relative regularization factor: the covariance floor is this fraction of the
/// mean per-dimension feature variance. Chosen small so it never distorts a
/// well-estimated component, but large enough (relative to the data scale) to
/// prevent variance-collapse singularities that would otherwise inflate the
/// log-likelihood without bound as `K` grows.
pub(super) const REG_REL: f64 = 1e-6;

/// Compute a data-scaled covariance regularization floor.
///
/// Returns `REG_REL * mean_j Var(feature_j)`, i.e. a small fraction of the
/// average marginal feature variance. Using an absolute constant (e.g. `1e-6`)
/// is scale-dependent and useless when features have variance of order 100+;
/// a relative floor keeps regularization meaningful across data scales.
///
/// Falls back to `REG_REL` if the data has (near-)zero total variance.
pub(super) fn data_scaled_reg(features: &[Vec<f64>], d: usize) -> f64 {
    let n = features.len();
    if n == 0 || d == 0 {
        return REG_REL;
    }
    let nf = n as f64;
    let mut total_var = 0.0;
    for j in 0..d {
        let mut sum = 0.0;
        for f in features {
            sum += f[j];
        }
        let mean = sum / nf;
        let mut ss = 0.0;
        for f in features {
            ss += (f[j] - mean).powi(2);
        }
        total_var += ss / nf;
    }
    let mean_var = total_var / d as f64;
    if mean_var > 0.0 {
        REG_REL * mean_var
    } else {
        REG_REL
    }
}

/// Accumulate full covariance from unit-weighted observations.
pub(super) fn accumulate_full_cov(
    features: &[Vec<f64>],
    indices: &[usize],
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; d * d];
    for &i in indices {
        for r in 0..d {
            let dr = features[i][r] - mean[r];
            for s in r..d {
                let val = dr * (features[i][s] - mean[s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    cov
}

/// Accumulate diagonal covariance from unit-weighted observations.
pub(super) fn accumulate_diag_cov(
    features: &[Vec<f64>],
    indices: &[usize],
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0; d];
    for &i in indices {
        for j in 0..d {
            diag[j] += (features[i][j] - mean[j]).powi(2);
        }
    }
    diag
}

/// Accumulate weighted full covariance from all observations.
pub(super) fn accumulate_full_cov_weighted(
    features: &[Vec<f64>],
    resp: &[f64],
    c: usize,
    k: usize,
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; d * d];
    for i in 0..features.len() {
        let w = resp[i * k + c];
        for r in 0..d {
            let dr = features[i][r] - mean[r];
            for s in r..d {
                let val = w * dr * (features[i][s] - mean[s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    cov
}

/// Accumulate weighted diagonal covariance from all observations.
pub(super) fn accumulate_diag_cov_weighted(
    features: &[Vec<f64>],
    resp: &[f64],
    c: usize,
    k: usize,
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0; d];
    for i in 0..features.len() {
        let w = resp[i * k + c];
        for j in 0..d {
            diag[j] += w * (features[i][j] - mean[j]).powi(2);
        }
    }
    diag
}

/// Normalize covariance by count and add regularization.
pub(super) fn regularize_cov(cov: &mut [f64], scale: f64, d: usize, reg: f64, is_full: bool) {
    for v in cov.iter_mut() {
        *v /= scale;
    }
    if is_full {
        for j in 0..d {
            cov[j * d + j] += reg;
        }
    } else {
        for v in cov.iter_mut() {
            *v += reg;
        }
    }
}

/// Identity-like regularization covariance (for degenerate components).
pub(super) fn identity_cov(d: usize, reg: f64, cov_type: CovType) -> Vec<f64> {
    match cov_type {
        CovType::Full => {
            let mut cov = vec![0.0; d * d];
            for j in 0..d {
                cov[j * d + j] = reg;
            }
            cov
        }
        CovType::Diagonal => vec![reg; d],
    }
}

/// Compute covariances from hard assignments.
pub(super) fn compute_covariances(
    features: &[Vec<f64>],
    assignments: &[usize],
    means: &[Vec<f64>],
    k: usize,
    d: usize,
    cov_type: CovType,
    reg: f64,
) -> Vec<Vec<f64>> {
    let n = features.len();
    let mut covariances = Vec::with_capacity(k);

    for c in 0..k {
        let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
        let nc = members.len().max(1) as f64;

        let mut cov = match cov_type {
            CovType::Full => accumulate_full_cov(features, &members, &means[c], d),
            CovType::Diagonal => accumulate_diag_cov(features, &members, &means[c], d),
        };
        regularize_cov(&mut cov, nc, d, reg, cov_type == CovType::Full);
        covariances.push(cov);
    }
    covariances
}
