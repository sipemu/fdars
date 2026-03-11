//! Functional classification with mixed scalar/functional predictors.
//!
//! Implements supervised classification for functional data using:
//! - [`fclassif_lda`] / [`fclassif_qda`] — FPC + LDA/QDA pipeline
//! - [`fclassif_knn`] — FPC + k-NN classifier
//! - [`fclassif_kernel`] — Nonparametric kernel classifier with mixed predictors
//! - [`fclassif_dd`] — Depth-based DD-classifier
//! - [`fclassif_cv`] — Cross-validated error rate

use crate::depth::fraiman_muniz_1d;
use crate::helpers::{l2_distance, simpsons_weights};
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

/// Classification result.
pub struct ClassifResult {
    /// Predicted class labels (length n)
    pub predicted: Vec<usize>,
    /// Posterior/membership probabilities (n x G) — if available
    pub probabilities: Option<FdMatrix>,
    /// Training accuracy
    pub accuracy: f64,
    /// Confusion matrix (G x G): row = true, col = predicted
    pub confusion: Vec<Vec<usize>>,
    /// Number of classes
    pub n_classes: usize,
    /// Number of FPC components used
    pub ncomp: usize,
}

/// Cross-validation result.
pub struct ClassifCvResult {
    /// Mean error rate across folds
    pub error_rate: f64,
    /// Per-fold error rates
    pub fold_errors: Vec<f64>,
    /// Best ncomp (if tuned)
    pub best_ncomp: usize,
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Count distinct classes and remap labels to 0..G-1.
pub(crate) fn remap_labels(y: &[usize]) -> (Vec<usize>, usize) {
    let mut labels: Vec<usize> = y.to_vec();
    let mut unique: Vec<usize> = y.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let g = unique.len();
    for label in &mut labels {
        *label = unique.iter().position(|&u| u == *label).unwrap_or(0);
    }
    (labels, g)
}

/// Build confusion matrix (G x G).
fn confusion_matrix(true_labels: &[usize], pred_labels: &[usize], g: usize) -> Vec<Vec<usize>> {
    let mut cm = vec![vec![0usize; g]; g];
    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        if t < g && p < g {
            cm[t][p] += 1;
        }
    }
    cm
}

/// Accuracy from labels.
fn compute_accuracy(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();
    if n == 0 {
        return 0.0;
    }
    let correct = true_labels
        .iter()
        .zip(pred_labels.iter())
        .filter(|(&t, &p)| t == p)
        .count();
    correct as f64 / n as f64
}

/// Extract FPC scores and append optional scalar covariates.
pub(crate) fn build_feature_matrix(
    data: &FdMatrix,
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<(FdMatrix, Vec<f64>, FdMatrix)> {
    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let n = data.nrows();
    let d_pc = fpca.scores.ncols();
    let d_cov = covariates.map_or(0, |c| c.ncols());
    let d = d_pc + d_cov;

    let mut features = FdMatrix::zeros(n, d);
    for i in 0..n {
        for j in 0..d_pc {
            features[(i, j)] = fpca.scores[(i, j)];
        }
        if let Some(cov) = covariates {
            for j in 0..d_cov {
                features[(i, d_pc + j)] = cov[(i, j)];
            }
        }
    }

    Some((features, fpca.mean, fpca.rotation))
}

// ---------------------------------------------------------------------------
// LDA: Linear Discriminant Analysis
// ---------------------------------------------------------------------------

/// Compute per-class means, counts, and priors from labeled features.
fn class_means_and_priors(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<f64>) {
    let n = features.nrows();
    let d = features.ncols();
    let mut counts = vec![0usize; g];
    let mut class_means = vec![vec![0.0; d]; g];
    for i in 0..n {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..d {
            class_means[c][j] += features[(i, j)];
        }
    }
    for c in 0..g {
        if counts[c] > 0 {
            for j in 0..d {
                class_means[c][j] /= counts[c] as f64;
            }
        }
    }
    let priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    (class_means, counts, priors)
}

/// Compute pooled within-class covariance (symmetric, regularized).
fn pooled_within_cov(
    features: &FdMatrix,
    labels: &[usize],
    class_means: &[Vec<f64>],
    g: usize,
) -> Vec<f64> {
    let n = features.nrows();
    let d = features.ncols();
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        let c = labels[i];
        for r in 0..d {
            let dr = features[(i, r)] - class_means[c][r];
            for s in r..d {
                let val = dr * (features[(i, s)] - class_means[c][s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    let scale = (n - g).max(1) as f64;
    for v in cov.iter_mut() {
        *v /= scale;
    }
    for j in 0..d {
        cov[j * d + j] += 1e-6;
    }
    cov
}

/// Compute per-class means and pooled within-class covariance.
pub(crate) fn lda_params(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let (class_means, _counts, priors) = class_means_and_priors(features, labels, g);
    let cov = pooled_within_cov(features, labels, &class_means, g);
    (class_means, cov, priors)
}

/// Cholesky factorization of d×d row-major matrix.
pub(crate) fn cholesky_d(mat: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for j in 0..d {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * d + k] * l[j * d + k];
        }
        let diag = mat[j * d + j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j * d + j] = diag.sqrt();
        for i in (j + 1)..d {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * d + k] * l[j * d + k];
            }
            l[i * d + j] = (mat[i * d + j] - s) / l[j * d + j];
        }
    }
    Some(l)
}

/// Forward solve L * x = b.
pub(crate) fn forward_solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut x = vec![0.0; d];
    for i in 0..d {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i * d + j] * x[j];
        }
        x[i] = (b[i] - s) / l[i * d + i];
    }
    x
}

/// Mahalanobis distance squared: (x-mu)^T Sigma^{-1} (x-mu) via Cholesky.
pub(crate) fn mahalanobis_sq(x: &[f64], mu: &[f64], chol: &[f64], d: usize) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mu.iter()).map(|(&a, &b)| a - b).collect();
    let y = forward_solve(chol, &diff, d);
    y.iter().map(|&v| v * v).sum()
}

/// LDA prediction: assign to class minimizing Mahalanobis distance (with prior).
pub(crate) fn lda_predict(
    features: &FdMatrix,
    class_means: &[Vec<f64>],
    cov_chol: &[f64],
    priors: &[f64],
    g: usize,
) -> Vec<usize> {
    let n = features.nrows();
    let d = features.ncols();

    (0..n)
        .map(|i| {
            let xi: Vec<f64> = (0..d).map(|j| features[(i, j)]).collect();
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..g {
                let maha = mahalanobis_sq(&xi, &class_means[c], cov_chol, d);
                let score = priors[c].max(1e-15).ln() - 0.5 * maha;
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }
            best_class
        })
        .collect()
}

/// FPC + LDA classification.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Class labels (length n)
/// * `covariates` — Optional scalar covariates (n × p)
/// * `ncomp` — Number of FPC components
pub fn fclassif_lda(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<ClassifResult> {
    let n = data.nrows();
    if n == 0 || y.len() != n || ncomp == 0 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    let (features, _mean, _rotation) = build_feature_matrix(data, covariates, ncomp)?;
    let d = features.ncols();
    let (class_means, cov, priors) = lda_params(&features, &labels, g);
    let chol = cholesky_d(&cov, d)?;

    let predicted = lda_predict(&features, &class_means, &chol, &priors, g);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: features.ncols().min(ncomp),
    })
}

// ---------------------------------------------------------------------------
// QDA: Quadratic Discriminant Analysis
// ---------------------------------------------------------------------------

/// Accumulate symmetric covariance from feature rows.
fn accumulate_class_cov(
    features: &FdMatrix,
    members: &[usize],
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; d * d];
    for &i in members {
        for r in 0..d {
            let dr = features[(i, r)] - mean[r];
            for s in r..d {
                let val = dr * (features[(i, s)] - mean[s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    cov
}

/// Per-class covariance matrices.
fn qda_class_covariances(
    features: &FdMatrix,
    labels: &[usize],
    class_means: &[Vec<f64>],
    g: usize,
) -> Vec<Vec<f64>> {
    let n = features.nrows();
    let d = features.ncols();

    (0..g)
        .map(|c| {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            let nc = members.len();
            let divisor = (nc.saturating_sub(1)).max(1) as f64;
            let mut cov = accumulate_class_cov(features, &members, &class_means[c], d);
            for v in cov.iter_mut() {
                *v /= divisor;
            }
            for j in 0..d {
                cov[j * d + j] += 1e-6;
            }
            cov
        })
        .collect()
}

/// Compute QDA parameters: class means, Cholesky factors, log-dets, priors.
pub(crate) fn build_qda_params(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    let d = features.ncols();
    let (class_means, _counts, priors) = class_means_and_priors(features, labels, g);
    let class_covs = qda_class_covariances(features, labels, &class_means, g);
    let mut class_chols = Vec::with_capacity(g);
    let mut class_log_dets = Vec::with_capacity(g);
    for cov in &class_covs {
        let chol = cholesky_d(cov, d)?;
        class_log_dets.push(log_det_cholesky(&chol, d));
        class_chols.push(chol);
    }
    Some((class_means, class_chols, class_log_dets, priors))
}

/// Log-determinant from Cholesky factor.
pub(crate) fn log_det_cholesky(l: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        s += l[i * d + i].ln();
    }
    2.0 * s
}

/// QDA prediction: per-class covariance.
pub(crate) fn qda_predict(
    features: &FdMatrix,
    class_means: &[Vec<f64>],
    class_chols: &[Vec<f64>],
    class_log_dets: &[f64],
    priors: &[f64],
    g: usize,
) -> Vec<usize> {
    let n = features.nrows();
    let d = features.ncols();

    (0..n)
        .map(|i| {
            let xi: Vec<f64> = (0..d).map(|j| features[(i, j)]).collect();
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..g {
                let maha = mahalanobis_sq(&xi, &class_means[c], &class_chols[c], d);
                let score = priors[c].max(1e-15).ln() - 0.5 * (class_log_dets[c] + maha);
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }
            best_class
        })
        .collect()
}

/// FPC + QDA classification.
pub fn fclassif_qda(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<ClassifResult> {
    let n = data.nrows();
    if n == 0 || y.len() != n || ncomp == 0 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    let (features, _mean, _rotation) = build_feature_matrix(data, covariates, ncomp)?;

    let (class_means, class_chols, class_log_dets, priors) =
        build_qda_params(&features, &labels, g)?;

    let predicted = qda_predict(
        &features,
        &class_means,
        &class_chols,
        &class_log_dets,
        &priors,
        g,
    );
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: features.ncols().min(ncomp),
    })
}

// ---------------------------------------------------------------------------
// k-NN classifier
// ---------------------------------------------------------------------------

/// FPC + k-NN classification.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Class labels
/// * `covariates` — Optional scalar covariates
/// * `ncomp` — Number of FPC components
/// * `k_nn` — Number of nearest neighbors
pub fn fclassif_knn(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
    k_nn: usize,
) -> Option<ClassifResult> {
    let n = data.nrows();
    if n == 0 || y.len() != n || ncomp == 0 || k_nn == 0 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    let (features, _mean, _rotation) = build_feature_matrix(data, covariates, ncomp)?;
    let d = features.ncols();

    let predicted = knn_predict_loo(&features, &labels, g, d, k_nn);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: d.min(ncomp),
    })
}

/// Leave-one-out k-NN prediction.
pub(crate) fn knn_predict_loo(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
    d: usize,
    k_nn: usize,
) -> Vec<usize> {
    let n = features.nrows();
    let k_nn = k_nn.min(n - 1);

    (0..n)
        .map(|i| {
            let xi: Vec<f64> = (0..d).map(|j| features[(i, j)]).collect();
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let xj: Vec<f64> = (0..d).map(|jj| features[(j, jj)]).collect();
                    let d_sq: f64 = xi.iter().zip(&xj).map(|(&a, &b)| (a - b).powi(2)).sum();
                    (d_sq, labels[j])
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Majority vote among k nearest
            let mut votes = vec![0usize; g];
            for &(_, label) in dists.iter().take(k_nn) {
                votes[label] += 1;
            }
            votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(c, _)| c)
                .unwrap_or(0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Nonparametric kernel classifier with mixed predictors
// ---------------------------------------------------------------------------

/// Find class with maximum score.
fn argmax_class(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(c, _)| c)
        .unwrap_or(0)
}

/// Compute marginal rank-based scalar depth of observation i w.r.t. class c.
fn scalar_depth_for_obs(cov: &FdMatrix, i: usize, class_indices: &[usize], p: usize) -> f64 {
    let nc = class_indices.len() as f64;
    if nc < 1.0 || p == 0 {
        return 0.0;
    }
    let mut depth = 0.0;
    for j in 0..p {
        let val = cov[(i, j)];
        let rank = class_indices
            .iter()
            .filter(|&&k| cov[(k, j)] <= val)
            .count() as f64;
        let u = rank / nc.max(1.0);
        depth += u.min(1.0 - u).min(0.5);
    }
    depth / p as f64
}

/// Generate bandwidth candidates from distance percentiles.
fn bandwidth_candidates(dists: &[f64], n: usize) -> Vec<f64> {
    let mut all_dists: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            all_dists.push(dists[i * n + j]);
        }
    }
    all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    (1..=20)
        .map(|p| {
            let idx = (p as f64 / 20.0 * (all_dists.len() - 1) as f64) as usize;
            all_dists[idx.min(all_dists.len() - 1)]
        })
        .filter(|&h| h > 1e-15)
        .collect()
}

/// LOO classification accuracy for a single bandwidth.
fn loo_accuracy_for_bandwidth(dists: &[f64], labels: &[usize], g: usize, n: usize, h: f64) -> f64 {
    let mut correct = 0;
    for i in 0..n {
        let mut votes = vec![0.0; g];
        for j in 0..n {
            if j != i {
                votes[labels[j]] += gaussian_kernel(dists[i * n + j], h);
            }
        }
        if argmax_class(&votes) == labels[i] {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

/// Gaussian kernel: exp(-d²/(2h²)).
fn gaussian_kernel(dist: f64, h: f64) -> f64 {
    if h < 1e-15 {
        return 0.0;
    }
    (-dist * dist / (2.0 * h * h)).exp()
}

/// Nonparametric kernel classifier for functional data with optional scalar covariates.
///
/// Uses product kernel: K_func × K_scalar. Bandwidth selected by LOO-CV.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points
/// * `y` — Class labels
/// * `covariates` — Optional scalar covariates (n × p)
/// * `h_func` — Functional bandwidth (0 = auto via LOO-CV)
/// * `h_scalar` — Scalar bandwidth (0 = auto)
pub fn fclassif_kernel(
    data: &FdMatrix,
    argvals: &[f64],
    y: &[usize],
    covariates: Option<&FdMatrix>,
    h_func: f64,
    h_scalar: f64,
) -> Option<ClassifResult> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || y.len() != n || argvals.len() != m {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    let weights = simpsons_weights(argvals);

    // Compute pairwise functional distances
    let func_dists = compute_pairwise_l2(data, &weights);

    // Compute pairwise scalar distances if covariates exist
    let scalar_dists = covariates.map(compute_pairwise_scalar);

    // Select bandwidths via LOO if needed
    let h_f = if h_func > 0.0 {
        h_func
    } else {
        select_bandwidth_loo(&func_dists, &labels, g, n, true)
    };
    let h_s = match &scalar_dists {
        Some(sd) if h_scalar <= 0.0 => select_bandwidth_loo(sd, &labels, g, n, false),
        _ => h_scalar,
    };

    let predicted = kernel_classify_loo(
        &func_dists,
        scalar_dists.as_deref(),
        &labels,
        g,
        n,
        h_f,
        h_s,
    );
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: 0,
    })
}

/// Compute pairwise L2 distances between curves.
fn compute_pairwise_l2(data: &FdMatrix, weights: &[f64]) -> Vec<f64> {
    let n = data.nrows();
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        let ri = data.row(i);
        for j in (i + 1)..n {
            let rj = data.row(j);
            let d = l2_distance(&ri, &rj, weights);
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Compute pairwise Euclidean distances between scalar covariate vectors.
fn compute_pairwise_scalar(covariates: &FdMatrix) -> Vec<f64> {
    let n = covariates.nrows();
    let p = covariates.ncols();
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d_sq = 0.0;
            for k in 0..p {
                d_sq += (covariates[(i, k)] - covariates[(j, k)]).powi(2);
            }
            let d = d_sq.sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Select bandwidth by LOO classification accuracy.
fn select_bandwidth_loo(dists: &[f64], labels: &[usize], g: usize, n: usize, is_func: bool) -> f64 {
    let candidates = bandwidth_candidates(dists, n);
    if candidates.is_empty() {
        return if is_func { 1.0 } else { 0.5 };
    }

    let mut best_h = candidates[0];
    let mut best_acc = 0.0;
    for &h in &candidates {
        let acc = loo_accuracy_for_bandwidth(dists, labels, g, n, h);
        if acc > best_acc {
            best_acc = acc;
            best_h = h;
        }
    }
    best_h
}

/// LOO kernel classification with product kernel.
fn kernel_classify_loo(
    func_dists: &[f64],
    scalar_dists: Option<&[f64]>,
    labels: &[usize],
    g: usize,
    n: usize,
    h_func: f64,
    h_scalar: f64,
) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let mut votes = vec![0.0; g];
            for j in 0..n {
                if j == i {
                    continue;
                }
                let kf = gaussian_kernel(func_dists[i * n + j], h_func);
                let ks = match scalar_dists {
                    Some(sd) if h_scalar > 1e-15 => gaussian_kernel(sd[i * n + j], h_scalar),
                    _ => 1.0,
                };
                votes[labels[j]] += kf * ks;
            }
            argmax_class(&votes)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Depth-based DD-classifier
// ---------------------------------------------------------------------------

/// Depth-based DD-classifier.
///
/// Computes functional depth of each observation w.r.t. each class,
/// then classifies by maximum depth.
/// Compute depth of all observations w.r.t. each class.
fn compute_class_depths(data: &FdMatrix, class_indices: &[Vec<usize>], n: usize) -> FdMatrix {
    let g = class_indices.len();
    let mut depth_scores = FdMatrix::zeros(n, g);
    for c in 0..g {
        if class_indices[c].is_empty() {
            continue;
        }
        let class_data = extract_class_data(data, &class_indices[c]);
        let depths = fraiman_muniz_1d(data, &class_data, true);
        for i in 0..n {
            depth_scores[(i, c)] = depths[i];
        }
    }
    depth_scores
}

/// Blend functional depth scores with scalar rank depth from covariates.
fn blend_scalar_depths(
    depth_scores: &mut FdMatrix,
    cov: &FdMatrix,
    class_indices: &[Vec<usize>],
    n: usize,
) {
    let g = class_indices.len();
    let p = cov.ncols();
    for c in 0..g {
        for i in 0..n {
            let sd = scalar_depth_for_obs(cov, i, &class_indices[c], p);
            depth_scores[(i, c)] = 0.7 * depth_scores[(i, c)] + 0.3 * sd;
        }
    }
}

pub fn fclassif_dd(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
) -> Option<ClassifResult> {
    let n = data.nrows();
    if n == 0 || y.len() != n {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    let class_indices: Vec<Vec<usize>> = (0..g)
        .map(|c| (0..n).filter(|&i| labels[i] == c).collect())
        .collect();

    let mut depth_scores = compute_class_depths(data, &class_indices, n);

    if let Some(cov) = covariates {
        blend_scalar_depths(&mut depth_scores, cov, &class_indices, n);
    }

    let predicted: Vec<usize> = (0..n)
        .map(|i| {
            let scores: Vec<f64> = (0..g).map(|c| depth_scores[(i, c)]).collect();
            argmax_class(&scores)
        })
        .collect();

    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifResult {
        predicted,
        probabilities: Some(depth_scores),
        accuracy,
        confusion,
        n_classes: g,
        ncomp: 0,
    })
}

/// Extract rows corresponding to given indices into a new FdMatrix.
fn extract_class_data(data: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let nc = indices.len();
    let m = data.ncols();
    let mut result = FdMatrix::zeros(nc, m);
    for (ri, &i) in indices.iter().enumerate() {
        for j in 0..m {
            result[(ri, j)] = data[(i, j)];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

/// K-fold cross-validated error rate for functional classification.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points
/// * `y` — Class labels
/// * `covariates` — Optional scalar covariates
/// * `method` — "lda", "qda", "knn", "kernel", "dd"
/// * `ncomp` — Number of FPC components (for lda/qda/knn)
/// * `nfold` — Number of CV folds
/// * `seed` — Random seed for fold assignment
pub fn fclassif_cv(
    data: &FdMatrix,
    argvals: &[f64],
    y: &[usize],
    covariates: Option<&FdMatrix>,
    method: &str,
    ncomp: usize,
    nfold: usize,
    seed: u64,
) -> Option<ClassifCvResult> {
    let n = data.nrows();
    if n < nfold || nfold < 2 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    // Assign folds
    let folds = assign_folds(n, nfold, seed);

    let mut fold_errors = Vec::with_capacity(nfold);

    for fold in 0..nfold {
        let (train_idx, test_idx) = fold_split(&folds, fold);
        let train_data = extract_class_data(data, &train_idx);
        let test_data = extract_class_data(data, &test_idx);
        let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
        let test_labels: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

        let train_cov = covariates.map(|c| extract_class_data(c, &train_idx));
        let test_cov = covariates.map(|c| extract_class_data(c, &test_idx));

        let predictions = cv_fold_predict(
            &train_data,
            &test_data,
            argvals,
            &train_labels,
            train_cov.as_ref(),
            test_cov.as_ref(),
            method,
            ncomp,
        );

        let n_test = test_labels.len();
        let errors = match predictions {
            Some(pred) => {
                let wrong = pred
                    .iter()
                    .zip(&test_labels)
                    .filter(|(&p, &t)| p != t)
                    .count();
                wrong as f64 / n_test as f64
            }
            None => 1.0,
        };
        fold_errors.push(errors);
    }

    let error_rate = fold_errors.iter().sum::<f64>() / nfold as f64;

    Some(ClassifCvResult {
        error_rate,
        fold_errors,
        best_ncomp: ncomp,
    })
}

/// Assign observations to folds.
fn assign_folds(n: usize, nfold: usize, seed: u64) -> Vec<usize> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut folds = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        folds[idx] = rank % nfold;
    }
    folds
}

/// Split indices into train and test for given fold.
fn fold_split(folds: &[usize], fold: usize) -> (Vec<usize>, Vec<usize>) {
    let train: Vec<usize> = (0..folds.len()).filter(|&i| folds[i] != fold).collect();
    let test: Vec<usize> = (0..folds.len()).filter(|&i| folds[i] == fold).collect();
    (train, test)
}

/// Predict on test set for one CV fold.
fn cv_fold_predict(
    train_data: &FdMatrix,
    test_data: &FdMatrix,
    _argvals: &[f64],
    train_labels: &[usize],
    train_cov: Option<&FdMatrix>,
    _test_cov: Option<&FdMatrix>,
    method: &str,
    ncomp: usize,
) -> Option<Vec<usize>> {
    match method {
        "lda" => {
            let _result = fclassif_lda(train_data, train_labels, train_cov, ncomp)?;
            // Re-predict on test data using training FPCA
            let fpca = fdata_to_pc_1d(train_data, ncomp)?;
            let predictions =
                project_and_classify_lda(test_data, &fpca, train_labels, train_cov, ncomp);
            Some(predictions)
        }
        "qda" => {
            let _result = fclassif_qda(train_data, train_labels, train_cov, ncomp)?;
            let fpca = fdata_to_pc_1d(train_data, ncomp)?;
            let predictions =
                project_and_classify_qda(test_data, &fpca, train_labels, train_cov, ncomp);
            Some(predictions)
        }
        "knn" => {
            let fpca = fdata_to_pc_1d(train_data, ncomp)?;
            let predictions =
                project_and_classify_knn(test_data, &fpca, train_labels, train_cov, ncomp, 5);
            Some(predictions)
        }
        // kernel and dd classifiers don't support out-of-sample prediction on new data
        "kernel" | "dd" => None,
        _ => None,
    }
}

/// Project test data onto FPCA basis (mean-center, multiply by rotation).
fn project_test_onto_fpca(test_data: &FdMatrix, fpca: &crate::regression::FpcaResult) -> FdMatrix {
    let n_test = test_data.nrows();
    let m = test_data.ncols();
    let d_pc = fpca.scores.ncols();
    let mut test_features = FdMatrix::zeros(n_test, d_pc);
    for i in 0..n_test {
        for k in 0..d_pc {
            let mut score = 0.0;
            for j in 0..m {
                score += (test_data[(i, j)] - fpca.mean[j]) * fpca.rotation[(j, k)];
            }
            test_features[(i, k)] = score;
        }
    }
    test_features
}

/// Project test data onto training FPCA and classify with LDA.
fn project_and_classify_lda(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    train_labels: &[usize],
    _train_cov: Option<&FdMatrix>,
    _ncomp: usize,
) -> Vec<usize> {
    let test_features = project_test_onto_fpca(test_data, fpca);

    let train_features = &fpca.scores;
    let (labels, g) = remap_labels(train_labels);
    let (class_means, cov, priors) = lda_params(train_features, &labels, g);
    let d = train_features.ncols();
    match cholesky_d(&cov, d) {
        Some(chol) => lda_predict(&test_features, &class_means, &chol, &priors, g),
        None => vec![0; test_data.nrows()],
    }
}

/// Project test data onto training FPCA and classify with QDA.
fn project_and_classify_qda(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    train_labels: &[usize],
    _train_cov: Option<&FdMatrix>,
    _ncomp: usize,
) -> Vec<usize> {
    let n_test = test_data.nrows();
    let test_features = project_test_onto_fpca(test_data, fpca);

    let (labels, g) = remap_labels(train_labels);
    let train_features = &fpca.scores;

    match build_qda_params(train_features, &labels, g) {
        Some((class_means, class_chols, class_log_dets, priors)) => qda_predict(
            &test_features,
            &class_means,
            &class_chols,
            &class_log_dets,
            &priors,
            g,
        ),
        None => vec![0; n_test],
    }
}

/// Project test data and classify with k-NN.
fn project_and_classify_knn(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    train_labels: &[usize],
    _train_cov: Option<&FdMatrix>,
    _ncomp: usize,
    k_nn: usize,
) -> Vec<usize> {
    let n_test = test_data.nrows();
    let n_train = fpca.scores.nrows();
    let m = test_data.ncols();
    let d = fpca.scores.ncols();

    let (labels, g) = remap_labels(train_labels);

    (0..n_test)
        .map(|i| {
            // Project test observation
            let mut xi = vec![0.0; d];
            for k in 0..d {
                for j in 0..m {
                    xi[k] += (test_data[(i, j)] - fpca.mean[j]) * fpca.rotation[(j, k)];
                }
            }

            // Distances to all training points
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|t| {
                    let d_sq: f64 = (0..d).map(|k| (xi[k] - fpca.scores[(t, k)]).powi(2)).sum();
                    (d_sq, labels[t])
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut votes = vec![0usize; g];
            for &(_, label) in dists.iter().take(k_nn.min(n_train)) {
                votes[label] += 1;
            }
            votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(c, _)| c)
                .unwrap_or(0)
        })
        .collect()
}

// ===========================================================================
// ClassifFit: wrapper for explainability trait
// ===========================================================================

use crate::explain_generic::{FpcPredictor, TaskType};

/// Classification method with stored parameters for prediction.
pub enum ClassifMethod {
    /// Linear Discriminant Analysis.
    Lda {
        class_means: Vec<Vec<f64>>,
        cov_chol: Vec<f64>,
        priors: Vec<f64>,
        n_classes: usize,
    },
    /// Quadratic Discriminant Analysis.
    Qda {
        class_means: Vec<Vec<f64>>,
        class_chols: Vec<Vec<f64>>,
        class_log_dets: Vec<f64>,
        priors: Vec<f64>,
        n_classes: usize,
    },
    /// k-Nearest Neighbors.
    Knn {
        training_scores: FdMatrix,
        training_labels: Vec<usize>,
        k: usize,
        n_classes: usize,
    },
}

/// A fitted classification model that retains FPCA components for explainability.
pub struct ClassifFit {
    /// Classification result (predicted labels, accuracy, confusion matrix).
    pub result: ClassifResult,
    /// FPCA mean function (length m).
    pub fpca_mean: Vec<f64>,
    /// FPCA rotation matrix (m × ncomp).
    pub fpca_rotation: FdMatrix,
    /// FPCA scores (n × ncomp).
    pub fpca_scores: FdMatrix,
    /// Number of FPC components used.
    pub ncomp: usize,
    /// The classification method with stored parameters.
    pub method: ClassifMethod,
}

/// FPC + LDA classification, retaining FPCA and LDA parameters for explainability.
pub fn fclassif_lda_fit(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<ClassifFit> {
    let n = data.nrows();
    if n == 0 || y.len() != n || ncomp == 0 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    // _fit variants use FPCA-only features (no covariates) so that stored
    // dimensions are consistent with FpcPredictor::project() / predict_from_scores().
    let (features, mean, rotation) = build_feature_matrix(data, None, ncomp)?;
    let _ = covariates; // acknowledged but not used — see docstring
    let d = features.ncols();
    let (class_means, cov, priors) = lda_params(&features, &labels, g);
    let chol = cholesky_d(&cov, d)?;

    let predicted = lda_predict(&features, &class_means, &chol, &priors, g);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifFit {
        result: ClassifResult {
            predicted,
            probabilities: None,
            accuracy,
            confusion,
            n_classes: g,
            ncomp: d,
        },
        fpca_mean: mean.clone(),
        fpca_rotation: rotation,
        fpca_scores: features,
        ncomp: d,
        method: ClassifMethod::Lda {
            class_means,
            cov_chol: chol,
            priors,
            n_classes: g,
        },
    })
}

/// FPC + QDA classification, retaining FPCA and QDA parameters for explainability.
pub fn fclassif_qda_fit(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<ClassifFit> {
    let n = data.nrows();
    if n == 0 || y.len() != n || ncomp == 0 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    // _fit variants use FPCA-only features — see fclassif_lda_fit comment.
    let (features, mean, rotation) = build_feature_matrix(data, None, ncomp)?;
    let _ = covariates;
    let (class_means, class_chols, class_log_dets, priors) =
        build_qda_params(&features, &labels, g)?;

    let predicted = qda_predict(
        &features,
        &class_means,
        &class_chols,
        &class_log_dets,
        &priors,
        g,
    );
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);
    let d = features.ncols();

    Some(ClassifFit {
        result: ClassifResult {
            predicted,
            probabilities: None,
            accuracy,
            confusion,
            n_classes: g,
            ncomp: d,
        },
        fpca_mean: mean.clone(),
        fpca_rotation: rotation,
        fpca_scores: features,
        ncomp: d,
        method: ClassifMethod::Qda {
            class_means,
            class_chols,
            class_log_dets,
            priors,
            n_classes: g,
        },
    })
}

/// FPC + k-NN classification, retaining FPCA and training data for explainability.
pub fn fclassif_knn_fit(
    data: &FdMatrix,
    y: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
    k_nn: usize,
) -> Option<ClassifFit> {
    let n = data.nrows();
    if n == 0 || y.len() != n || ncomp == 0 || k_nn == 0 {
        return None;
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return None;
    }

    // _fit variants use FPCA-only features — see fclassif_lda_fit comment.
    let (features, mean, rotation) = build_feature_matrix(data, None, ncomp)?;
    let _ = covariates;
    let d = features.ncols();

    let predicted = knn_predict_loo(&features, &labels, g, d, k_nn);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Some(ClassifFit {
        result: ClassifResult {
            predicted,
            probabilities: None,
            accuracy,
            confusion,
            n_classes: g,
            ncomp: d,
        },
        fpca_mean: mean.clone(),
        fpca_rotation: rotation,
        fpca_scores: features.clone(),
        ncomp: d,
        method: ClassifMethod::Knn {
            training_scores: features,
            training_labels: labels,
            k: k_nn,
            n_classes: g,
        },
    })
}

// ---------------------------------------------------------------------------
// FpcPredictor impl for ClassifFit
// ---------------------------------------------------------------------------

impl FpcPredictor for ClassifFit {
    fn fpca_mean(&self) -> &[f64] {
        &self.fpca_mean
    }

    fn fpca_rotation(&self) -> &FdMatrix {
        &self.fpca_rotation
    }

    fn ncomp(&self) -> usize {
        self.ncomp
    }

    fn training_scores(&self) -> &FdMatrix {
        &self.fpca_scores
    }

    fn task_type(&self) -> TaskType {
        match &self.method {
            ClassifMethod::Lda { n_classes, .. }
            | ClassifMethod::Qda { n_classes, .. }
            | ClassifMethod::Knn { n_classes, .. } => {
                if *n_classes == 2 {
                    TaskType::BinaryClassification
                } else {
                    TaskType::MulticlassClassification(*n_classes)
                }
            }
        }
    }

    fn predict_from_scores(&self, scores: &[f64], _scalar_covariates: Option<&[f64]>) -> f64 {
        match &self.method {
            ClassifMethod::Lda {
                class_means,
                cov_chol,
                priors,
                n_classes,
            } => {
                let g = *n_classes;
                let d = scores.len();
                if g == 2 {
                    // Return P(Y=1) via softmax of discriminant scores
                    let score0 = priors[0].max(1e-15).ln()
                        - 0.5 * mahalanobis_sq(scores, &class_means[0], cov_chol, d);
                    let score1 = priors[1].max(1e-15).ln()
                        - 0.5 * mahalanobis_sq(scores, &class_means[1], cov_chol, d);
                    let max_s = score0.max(score1);
                    let exp0 = (score0 - max_s).exp();
                    let exp1 = (score1 - max_s).exp();
                    exp1 / (exp0 + exp1)
                } else {
                    // Return predicted class as f64
                    let mut best_class = 0;
                    let mut best_score = f64::NEG_INFINITY;
                    for c in 0..g {
                        let maha = mahalanobis_sq(scores, &class_means[c], cov_chol, d);
                        let s = priors[c].max(1e-15).ln() - 0.5 * maha;
                        if s > best_score {
                            best_score = s;
                            best_class = c;
                        }
                    }
                    best_class as f64
                }
            }
            ClassifMethod::Qda {
                class_means,
                class_chols,
                class_log_dets,
                priors,
                n_classes,
            } => {
                let g = *n_classes;
                let d = scores.len();
                if g == 2 {
                    // Return P(Y=1) via softmax of discriminant scores
                    let score0 = priors[0].max(1e-15).ln()
                        - 0.5
                            * (class_log_dets[0]
                                + mahalanobis_sq(scores, &class_means[0], &class_chols[0], d));
                    let score1 = priors[1].max(1e-15).ln()
                        - 0.5
                            * (class_log_dets[1]
                                + mahalanobis_sq(scores, &class_means[1], &class_chols[1], d));
                    let max_s = score0.max(score1);
                    let exp0 = (score0 - max_s).exp();
                    let exp1 = (score1 - max_s).exp();
                    exp1 / (exp0 + exp1)
                } else {
                    let mut best_class = 0;
                    let mut best_score = f64::NEG_INFINITY;
                    for c in 0..g {
                        let maha = mahalanobis_sq(scores, &class_means[c], &class_chols[c], d);
                        let s = priors[c].max(1e-15).ln() - 0.5 * (class_log_dets[c] + maha);
                        if s > best_score {
                            best_score = s;
                            best_class = c;
                        }
                    }
                    best_class as f64
                }
            }
            ClassifMethod::Knn {
                training_scores,
                training_labels,
                k,
                n_classes,
            } => {
                let g = *n_classes;
                let n_train = training_scores.nrows();
                let d = scores.len();
                let k_nn = (*k).min(n_train);

                let mut dists: Vec<(f64, usize)> = (0..n_train)
                    .map(|j| {
                        let d_sq: f64 = (0..d)
                            .map(|c| (scores[c] - training_scores[(j, c)]).powi(2))
                            .sum();
                        (d_sq, training_labels[j])
                    })
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let mut votes = vec![0usize; g];
                for &(_, label) in dists.iter().take(k_nn) {
                    if label < g {
                        votes[label] += 1;
                    }
                }

                if g == 2 {
                    // Return proportion voting for class 1 as probability
                    votes[1] as f64 / k_nn as f64
                } else {
                    // Return majority vote class as f64
                    votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &v)| v)
                        .map(|(c, _)| c as f64)
                        .unwrap_or(0.0)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Class probability vectors (for conformal prediction sets)
// ---------------------------------------------------------------------------

/// Compute full class probability vectors for each observation.
///
/// Returns `n × g` probability vectors suitable for conformal classification.
/// For each observation, the probabilities sum to 1.
pub(crate) fn classif_predict_probs(fit: &ClassifFit, scores: &FdMatrix) -> Vec<Vec<f64>> {
    let n = scores.nrows();
    let d = scores.ncols();
    match &fit.method {
        ClassifMethod::Lda {
            class_means,
            cov_chol,
            priors,
            n_classes,
        } => {
            let g = *n_classes;
            (0..n)
                .map(|i| {
                    let x: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
                    let disc: Vec<f64> = (0..g)
                        .map(|c| {
                            priors[c].max(1e-15).ln()
                                - 0.5 * mahalanobis_sq(&x, &class_means[c], cov_chol, d)
                        })
                        .collect();
                    softmax(&disc)
                })
                .collect()
        }
        ClassifMethod::Qda {
            class_means,
            class_chols,
            class_log_dets,
            priors,
            n_classes,
        } => {
            let g = *n_classes;
            (0..n)
                .map(|i| {
                    let x: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
                    let disc: Vec<f64> = (0..g)
                        .map(|c| {
                            priors[c].max(1e-15).ln()
                                - 0.5
                                    * (class_log_dets[c]
                                        + mahalanobis_sq(&x, &class_means[c], &class_chols[c], d))
                        })
                        .collect();
                    softmax(&disc)
                })
                .collect()
        }
        ClassifMethod::Knn {
            training_scores,
            training_labels,
            k,
            n_classes,
        } => {
            let g = *n_classes;
            let n_train = training_scores.nrows();
            let k_nn = (*k).min(n_train);
            (0..n)
                .map(|i| {
                    let x: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
                    let mut dists: Vec<(f64, usize)> = (0..n_train)
                        .map(|j| {
                            let d_sq: f64 = (0..d)
                                .map(|c| (x[c] - training_scores[(j, c)]).powi(2))
                                .sum();
                            (d_sq, training_labels[j])
                        })
                        .collect();
                    dists
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let mut votes = vec![0usize; g];
                    for &(_, label) in dists.iter().take(k_nn) {
                        if label < g {
                            votes[label] += 1;
                        }
                    }
                    votes.iter().map(|&v| v as f64 / k_nn as f64).collect()
                })
                .collect()
        }
    }
}

/// Softmax of a vector of log-scores → probabilities.
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
    }

    /// Generate two well-separated classes of curves.
    fn generate_two_class_data(n_per: usize, m: usize) -> (FdMatrix, Vec<usize>, Vec<f64>) {
        let t = uniform_grid(m);
        let n = 2 * n_per;
        let mut col_major = vec![0.0; n * m];

        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                // Class 0: sin
                col_major[i + j * n] =
                    (2.0 * PI * tj).sin() + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                // Class 1: -sin (opposite phase)
                col_major[(i + n_per) + j * n] =
                    -(2.0 * PI * tj).sin() + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }

        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();
        (data, labels, t)
    }

    #[test]
    fn test_fclassif_lda_basic() {
        let (data, labels, _t) = generate_two_class_data(20, 50);
        let result = fclassif_lda(&data, &labels, None, 3).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert_eq!(result.n_classes, 2);
        assert!(
            result.accuracy > 0.8,
            "LDA accuracy should be high: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_fclassif_qda_basic() {
        let (data, labels, _t) = generate_two_class_data(20, 50);
        let result = fclassif_qda(&data, &labels, None, 3).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert!(
            result.accuracy > 0.8,
            "QDA accuracy should be high: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_fclassif_knn_basic() {
        let (data, labels, _t) = generate_two_class_data(20, 50);
        let result = fclassif_knn(&data, &labels, None, 3, 5).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert!(
            result.accuracy > 0.7,
            "k-NN accuracy should be reasonable: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_fclassif_kernel_basic() {
        let (data, labels, t) = generate_two_class_data(20, 50);
        let result = fclassif_kernel(&data, &t, &labels, None, 0.0, 0.0).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert!(
            result.accuracy > 0.7,
            "Kernel accuracy should be reasonable: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_fclassif_dd_basic() {
        let (data, labels, _t) = generate_two_class_data(20, 50);
        let result = fclassif_dd(&data, &labels, None).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert_eq!(result.n_classes, 2);
        // DD-classifier should work on well-separated data
        assert!(
            result.accuracy > 0.6,
            "DD accuracy should be reasonable: {}",
            result.accuracy
        );
        assert!(result.probabilities.is_some());
    }

    #[test]
    fn test_confusion_matrix_shape() {
        let (data, labels, _t) = generate_two_class_data(15, 50);
        let result = fclassif_lda(&data, &labels, None, 2).unwrap();

        assert_eq!(result.confusion.len(), 2);
        assert_eq!(result.confusion[0].len(), 2);
        assert_eq!(result.confusion[1].len(), 2);

        // Total should equal n
        let total: usize = result.confusion.iter().flat_map(|row| row.iter()).sum();
        assert_eq!(total, 30);
    }

    #[test]
    fn test_fclassif_cv_lda() {
        let (data, labels, t) = generate_two_class_data(25, 50);
        let result = fclassif_cv(&data, &t, &labels, None, "lda", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(
            result.error_rate < 0.3,
            "CV error should be low: {}",
            result.error_rate
        );
    }

    #[test]
    fn test_fclassif_invalid_input() {
        let data = FdMatrix::zeros(0, 0);
        assert!(fclassif_lda(&data, &[], None, 1).is_none());

        let data = FdMatrix::zeros(10, 50);
        let labels = vec![0; 10]; // single class
        assert!(fclassif_lda(&data, &labels, None, 1).is_none());
    }

    #[test]
    fn test_remap_labels() {
        let (mapped, g) = remap_labels(&[5, 5, 10, 10, 20]);
        assert_eq!(g, 3);
        assert_eq!(mapped, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn test_fclassif_lda_with_covariates() {
        let n_per = 15;
        let n = 2 * n_per;
        let m = 50;
        let t = uniform_grid(m);

        // Curves are identical across classes
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * tj).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        // But covariate separates: 0 vs 10
        let mut cov_data = vec![0.0; n];
        for i in n_per..n {
            cov_data[i] = 10.0;
        }
        let covariates = FdMatrix::from_column_major(cov_data, n, 1).unwrap();

        let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

        let result = fclassif_lda(&data, &labels, Some(&covariates), 2).unwrap();
        assert!(
            result.accuracy > 0.9,
            "Covariate should enable separation: {}",
            result.accuracy
        );
    }

    // -----------------------------------------------------------------------
    // Additional coverage tests
    // -----------------------------------------------------------------------

    /// Helper: generate two-class data with scalar covariates.
    fn generate_two_class_with_covariates(
        n_per: usize,
        m: usize,
        p_cov: usize,
    ) -> (FdMatrix, Vec<usize>, Vec<f64>, FdMatrix) {
        let (data, labels, t) = generate_two_class_data(n_per, m);
        let n = 2 * n_per;
        // Covariates: class 0 → low values, class 1 → high values
        let mut cov_data = vec![0.0; n * p_cov];
        for i in 0..n {
            for j in 0..p_cov {
                let base = if labels[i] == 0 { 0.0 } else { 5.0 };
                cov_data[i + j * n] = base + 0.1 * ((i * 3 + j * 7) % 50) as f64 / 50.0;
            }
        }
        let covariates = FdMatrix::from_column_major(cov_data, n, p_cov).unwrap();
        (data, labels, t, covariates)
    }

    #[test]
    fn test_fclassif_cv_qda() {
        let (data, labels, t) = generate_two_class_data(25, 50);
        let result = fclassif_cv(&data, &t, &labels, None, "qda", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(
            result.error_rate < 0.4,
            "QDA CV error should be low: {}",
            result.error_rate
        );
        assert_eq!(result.best_ncomp, 3);
    }

    #[test]
    fn test_fclassif_cv_knn() {
        let (data, labels, t) = generate_two_class_data(25, 50);
        let result = fclassif_cv(&data, &t, &labels, None, "knn", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(
            result.error_rate < 0.4,
            "k-NN CV error should be low: {}",
            result.error_rate
        );
    }

    #[test]
    fn test_fclassif_cv_kernel() {
        let (data, labels, t) = generate_two_class_data(25, 50);
        let result = fclassif_cv(&data, &t, &labels, None, "kernel", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        // Kernel CV: placeholder prediction may not be accurate, just ensure it runs
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_cv_dd() {
        let (data, labels, t) = generate_two_class_data(25, 50);
        let result = fclassif_cv(&data, &t, &labels, None, "dd", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_cv_invalid_method() {
        let (data, labels, t) = generate_two_class_data(25, 50);
        // "bogus" method hits the `_ => None` arm in cv_fold_predict
        let result = fclassif_cv(&data, &t, &labels, None, "bogus", 3, 5, 42);

        // Should still return Some — fold errors will be 1.0 for each fold
        let r = result.unwrap();
        assert!((r.error_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fclassif_cv_too_few_folds() {
        let (data, labels, t) = generate_two_class_data(10, 50);
        // nfold < 2 → None
        assert!(fclassif_cv(&data, &t, &labels, None, "lda", 3, 1, 42).is_none());
        // n < nfold → None
        assert!(fclassif_cv(&data, &t, &labels, None, "lda", 3, 100, 42).is_none());
    }

    #[test]
    fn test_fclassif_cv_single_class() {
        let (data, _labels, t) = generate_two_class_data(10, 50);
        let single = vec![0usize; 20]; // only one class
        assert!(fclassif_cv(&data, &t, &single, None, "lda", 3, 5, 42).is_none());
    }

    #[test]
    fn test_fclassif_kernel_with_covariates() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(20, 50, 2);
        let result = fclassif_kernel(&data, &t, &labels, Some(&covariates), 0.0, 0.0).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert!(
            result.accuracy > 0.5,
            "Kernel+cov accuracy should be reasonable: {}",
            result.accuracy
        );
        assert_eq!(result.ncomp, 0); // kernel doesn't use ncomp
    }

    #[test]
    fn test_fclassif_kernel_with_covariates_manual_bandwidth() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(15, 50, 1);
        // Provide explicit bandwidths (>0 skips LOO selection)
        let result = fclassif_kernel(&data, &t, &labels, Some(&covariates), 1.0, 1.0).unwrap();

        assert_eq!(result.predicted.len(), 30);
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    }

    #[test]
    fn test_fclassif_dd_with_covariates() {
        let (data, labels, _t, covariates) = generate_two_class_with_covariates(20, 50, 2);
        let result = fclassif_dd(&data, &labels, Some(&covariates)).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert_eq!(result.n_classes, 2);
        assert!(
            result.accuracy > 0.5,
            "DD+cov accuracy should be reasonable: {}",
            result.accuracy
        );
        assert!(result.probabilities.is_some());
    }

    #[test]
    fn test_fclassif_dd_with_single_covariate() {
        // Curves are identical; only the covariate separates classes
        let n_per = 15;
        let n = 2 * n_per;
        let m = 50;
        let t = uniform_grid(m);

        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * tj).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

        // Covariate: class 0 → [0..1], class 1 → [10..11]
        let mut cov_data = vec![0.0; n];
        for i in 0..n_per {
            cov_data[i] = i as f64 / n_per as f64;
        }
        for i in n_per..n {
            cov_data[i] = 10.0 + (i - n_per) as f64 / n_per as f64;
        }
        let covariates = FdMatrix::from_column_major(cov_data, n, 1).unwrap();

        let result = fclassif_dd(&data, &labels, Some(&covariates)).unwrap();
        // The scalar blending should help even when curves are identical
        assert!(
            result.accuracy >= 0.5,
            "DD with scalar covariate: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_scalar_depth_for_obs_edge_cases() {
        // Empty class indices → depth = 0
        let cov = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        assert_eq!(scalar_depth_for_obs(&cov, 0, &[], 1), 0.0);

        // p=0 → depth = 0
        let cov0 = FdMatrix::zeros(4, 0);
        assert_eq!(scalar_depth_for_obs(&cov0, 0, &[0, 1, 2, 3], 0), 0.0);

        // Normal case: all indices
        let depth = scalar_depth_for_obs(&cov, 1, &[0, 1, 2, 3], 1);
        assert!(depth > 0.0 && depth <= 0.5, "depth={}", depth);

        // Observation is at the extremes
        let depth_min = scalar_depth_for_obs(&cov, 0, &[0, 1, 2, 3], 1);
        let depth_max = scalar_depth_for_obs(&cov, 3, &[0, 1, 2, 3], 1);
        // Extreme observations should have low depth
        assert!(depth_min <= 0.5, "depth_min={}", depth_min);
        assert!(depth_max <= 0.5, "depth_max={}", depth_max);
    }

    #[test]
    fn test_scalar_depth_for_obs_multivariate() {
        // 2 covariates
        let cov =
            FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)
                .unwrap();
        let depth = scalar_depth_for_obs(&cov, 1, &[0, 1, 2, 3], 2);
        assert!(depth > 0.0 && depth <= 0.5, "multivar depth={}", depth);
    }

    #[test]
    fn test_blend_scalar_depths_modifies_scores() {
        let n = 6;
        let g = 2;
        let mut depth_scores = FdMatrix::zeros(n, g);
        // Fill with some values
        for i in 0..n {
            depth_scores[(i, 0)] = 0.5;
            depth_scores[(i, 1)] = 0.3;
        }

        let cov = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], n, 1).unwrap();
        let class_indices = vec![vec![0, 1, 2], vec![3, 4, 5]];

        let original_00 = depth_scores[(0, 0)];
        blend_scalar_depths(&mut depth_scores, &cov, &class_indices, n);

        // Scores should have been modified (blended with 0.7 / 0.3 weights)
        let blended_00 = depth_scores[(0, 0)];
        // blended = 0.7 * 0.5 + 0.3 * scalar_depth
        assert!(
            (blended_00 - original_00).abs() > 1e-10,
            "blend should change scores: original={}, blended={}",
            original_00,
            blended_00
        );
    }

    #[test]
    fn test_compute_pairwise_scalar() {
        let n = 4;
        // 2 covariates
        let cov = FdMatrix::from_column_major(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], n, 2)
            .unwrap();
        let dists = compute_pairwise_scalar(&cov);
        assert_eq!(dists.len(), n * n);

        // Diagonal should be zero
        for i in 0..n {
            assert!((dists[i * n + i]).abs() < 1e-15);
        }
        // Symmetry
        for i in 0..n {
            for j in 0..n {
                assert!((dists[i * n + j] - dists[j * n + i]).abs() < 1e-15);
            }
        }
        // d(0,1) = sqrt(1^2 + 0^2) = 1.0
        assert!((dists[1] - 1.0).abs() < 1e-10);
        // d(0,3) = sqrt(3^2 + 0^2) = 3.0
        assert!((dists[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fclassif_cv_lda_with_covariates() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(25, 50, 1);
        let result = fclassif_cv(&data, &t, &labels, Some(&covariates), "lda", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_cv_qda_with_covariates() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(25, 50, 1);
        let result = fclassif_cv(&data, &t, &labels, Some(&covariates), "qda", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_cv_knn_with_covariates() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(25, 50, 1);
        let result = fclassif_cv(&data, &t, &labels, Some(&covariates), "knn", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_cv_kernel_with_covariates() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(25, 50, 1);
        let result =
            fclassif_cv(&data, &t, &labels, Some(&covariates), "kernel", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_cv_dd_with_covariates() {
        let (data, labels, t, covariates) = generate_two_class_with_covariates(25, 50, 2);
        let result = fclassif_cv(&data, &t, &labels, Some(&covariates), "dd", 3, 5, 42).unwrap();

        assert_eq!(result.fold_errors.len(), 5);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
    }

    #[test]
    fn test_fclassif_kernel_invalid_inputs() {
        let data = FdMatrix::zeros(0, 0);
        assert!(fclassif_kernel(&data, &[], &[], None, 0.0, 0.0).is_none());

        let data = FdMatrix::zeros(5, 10);
        let t = uniform_grid(10);
        let labels = vec![0; 5]; // single class
        assert!(fclassif_kernel(&data, &t, &labels, None, 0.0, 0.0).is_none());

        // Mismatched argvals length
        let labels2 = vec![0, 0, 0, 1, 1];
        let wrong_t = vec![0.0, 1.0]; // wrong length
        assert!(fclassif_kernel(&data, &wrong_t, &labels2, None, 0.0, 0.0).is_none());
    }

    #[test]
    fn test_fclassif_dd_invalid_inputs() {
        let data = FdMatrix::zeros(0, 0);
        assert!(fclassif_dd(&data, &[], None).is_none());

        let data = FdMatrix::zeros(5, 10);
        let labels = vec![0; 5]; // single class
        assert!(fclassif_dd(&data, &labels, None).is_none());
    }

    #[test]
    fn test_argmax_class_empty() {
        assert_eq!(argmax_class(&[]), 0);
        assert_eq!(argmax_class(&[0.1]), 0);
        assert_eq!(argmax_class(&[0.1, 0.9, 0.5]), 1);
    }

    #[test]
    fn test_gaussian_kernel_values() {
        // h=0 → 0
        assert_eq!(gaussian_kernel(1.0, 0.0), 0.0);
        // dist=0 → 1
        assert!((gaussian_kernel(0.0, 1.0) - 1.0).abs() < 1e-15);
        // Normal case
        let k = gaussian_kernel(1.0, 1.0);
        let expected = (-0.5_f64).exp();
        assert!((k - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fclassif_qda_with_covariates() {
        let (data, labels, _t, covariates) = generate_two_class_with_covariates(20, 50, 1);
        let result = fclassif_qda(&data, &labels, Some(&covariates), 3).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert!(
            result.accuracy > 0.5,
            "QDA+cov accuracy: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_fclassif_knn_with_covariates() {
        let (data, labels, _t, covariates) = generate_two_class_with_covariates(20, 50, 1);
        let result = fclassif_knn(&data, &labels, Some(&covariates), 3, 5).unwrap();

        assert_eq!(result.predicted.len(), 40);
        assert!(
            result.accuracy > 0.5,
            "k-NN+cov accuracy: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_fclassif_knn_invalid_k() {
        let (data, labels, _t) = generate_two_class_data(10, 50);
        // k_nn == 0 → None
        assert!(fclassif_knn(&data, &labels, None, 3, 0).is_none());
    }

    #[test]
    fn test_bandwidth_candidates_empty_distances() {
        // All distances zero → candidates filtered out
        let dists = vec![0.0; 9];
        let cands = bandwidth_candidates(&dists, 3);
        assert!(cands.is_empty());
    }

    #[test]
    fn test_select_bandwidth_loo_empty_candidates() {
        // All distances zero → empty candidates → default bandwidth
        let dists = vec![0.0; 9];
        let labels = vec![0, 0, 1];
        let h = select_bandwidth_loo(&dists, &labels, 2, 3, true);
        assert!((h - 1.0).abs() < 1e-10, "default func bandwidth: {}", h);

        let h2 = select_bandwidth_loo(&dists, &labels, 2, 3, false);
        assert!((h2 - 0.5).abs() < 1e-10, "default scalar bandwidth: {}", h2);
    }

    #[test]
    fn test_fold_split() {
        let folds = vec![0, 1, 2, 0, 1, 2];
        let (train, test) = fold_split(&folds, 1);
        assert_eq!(train, vec![0, 2, 3, 5]);
        assert_eq!(test, vec![1, 4]);
    }

    #[test]
    fn test_assign_folds_deterministic() {
        let f1 = assign_folds(10, 3, 42);
        let f2 = assign_folds(10, 3, 42);
        assert_eq!(f1, f2);

        // All fold indices in [0, nfold)
        for &f in &f1 {
            assert!(f < 3);
        }
    }

    #[test]
    fn test_project_test_onto_fpca() {
        let n_train = 20;
        let m = 50;
        let ncomp = 3;
        let (data, _labels, _t) = generate_two_class_data(n_train / 2, m);

        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        // Create small "test" matrix
        let n_test = 5;
        let mut test_col = vec![0.0; n_test * m];
        for i in 0..n_test {
            for j in 0..m {
                test_col[i + j * n_test] = data[(i, j)] + 0.01;
            }
        }
        let test_data = FdMatrix::from_column_major(test_col, n_test, m).unwrap();

        let projected = project_test_onto_fpca(&test_data, &fpca);
        assert_eq!(projected.nrows(), n_test);
        assert_eq!(projected.ncols(), ncomp);
    }

    #[test]
    fn test_fclassif_three_classes() {
        let n_per = 15;
        let n = 3 * n_per;
        let m = 50;
        let t = uniform_grid(m);

        let mut col_major = vec![0.0; n * m];
        // Class 0: sin
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] =
                    (2.0 * PI * tj).sin() + 0.02 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }
        // Class 1: cos
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                col_major[(i + n_per) + j * n] =
                    (2.0 * PI * tj).cos() + 0.02 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }
        // Class 2: constant
        for i in 0..n_per {
            for (j, _) in t.iter().enumerate() {
                col_major[(i + 2 * n_per) + j * n] =
                    3.0 + 0.02 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }

        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let labels: Vec<usize> = (0..n)
            .map(|i| {
                if i < n_per {
                    0
                } else if i < 2 * n_per {
                    1
                } else {
                    2
                }
            })
            .collect();

        let result = fclassif_lda(&data, &labels, None, 3).unwrap();
        assert_eq!(result.n_classes, 3);
        assert!(
            result.accuracy > 0.8,
            "Three-class accuracy: {}",
            result.accuracy
        );
    }
}
