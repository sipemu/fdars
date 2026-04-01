//! Score projection, subsetting, and statistics helpers.

use crate::matrix::FdMatrix;

/// Project data → FPC scores (with integration weights).
pub(crate) fn project_scores(
    data: &FdMatrix,
    mean: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    weights: &[f64],
) -> FdMatrix {
    let (n, m) = data.shape();
    let mut scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (data[(i, j)] - mean[j]) * rotation[(j, k)] * weights[j];
            }
            scores[(i, k)] = s;
        }
    }
    scores
}

/// Subsample rows from an FdMatrix.
pub(crate) fn subsample_rows(data: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let ncols = data.ncols();
    let mut out = FdMatrix::zeros(indices.len(), ncols);
    for (new_i, &orig_i) in indices.iter().enumerate() {
        for j in 0..ncols {
            out[(new_i, j)] = data[(orig_i, j)];
        }
    }
    out
}

/// Clone an FdMatrix of scores.
pub(crate) fn clone_scores_matrix(scores: &FdMatrix, n: usize, ncomp: usize) -> FdMatrix {
    let mut perm = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for c in 0..ncomp {
            perm[(i, c)] = scores[(i, c)];
        }
    }
    perm
}

/// Compute the grid for a single FPC score column.
pub(crate) fn make_grid(scores: &FdMatrix, component: usize, n_grid: usize) -> Vec<f64> {
    let n = scores.nrows();
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for i in 0..n {
        let v = scores[(i, component)];
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if (mx - mn).abs() < 1e-15 {
        mx = mn + 1.0;
    }
    (0..n_grid)
        .map(|g| mn + (mx - mn) * g as f64 / (n_grid - 1) as f64)
        .collect()
}

/// Compute column means of an FdMatrix.
pub(crate) fn compute_column_means(mat: &FdMatrix, ncols: usize) -> Vec<f64> {
    let n = mat.nrows();
    let mut means = vec![0.0; ncols];
    for k in 0..ncols {
        for i in 0..n {
            means[k] += mat[(i, k)];
        }
        means[k] /= n as f64;
    }
    means
}

/// Compute mean scalar covariates from an optional FdMatrix.
pub(crate) fn compute_mean_scalar(
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n: usize,
) -> Vec<f64> {
    if p_scalar == 0 {
        return vec![];
    }
    if let Some(sc) = scalar_covariates {
        (0..p_scalar)
            .map(|j| {
                let mut s = 0.0;
                for i in 0..n {
                    s += sc[(i, j)];
                }
                s / n as f64
            })
            .collect()
    } else {
        vec![0.0; p_scalar]
    }
}

/// Compute score variance for each component (mean-zero scores from FPCA).
pub(crate) fn compute_score_variance(scores: &FdMatrix, n: usize, ncomp: usize) -> Vec<f64> {
    let mut score_variance = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_variance[k] = ss / (n - 1) as f64;
    }
    score_variance
}

/// Compute column means of ICE curves -> PDP.
pub(crate) fn ice_to_pdp(ice_curves: &FdMatrix, n: usize, n_grid: usize) -> Vec<f64> {
    let mut pdp = vec![0.0; n_grid];
    for g in 0..n_grid {
        for i in 0..n {
            pdp[g] += ice_curves[(i, g)];
        }
        pdp[g] /= n as f64;
    }
    pdp
}
