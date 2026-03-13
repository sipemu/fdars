//! Random Projection with Derivatives (RPD) depth.
//!
//! Enriches random projection depth by incorporating derivative information.
//! Each curve is augmented with its finite-difference derivatives up to order
//! `nderiv`, and standard random projection depth is computed on the
//! augmented representation.

use crate::matrix::FdMatrix;

/// Compute RPD depth for 1D functional data.
///
/// Projects curves augmented with derivatives to scalars using random
/// projections and computes average univariate depth.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for (n_obj x m)
/// * `data_ori` - Reference data (n_ori x m)
/// * `argvals` - Evaluation points (length m), used for finite-difference spacing
/// * `nproj` - Number of random projections
/// * `nderiv` - Maximum derivative order to include (0 = no derivatives, same as RP depth)
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::depth::rpd_depth_1d;
///
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let depths = rpd_depth_1d(&data, &data, &argvals, 50, 1);
/// assert_eq!(depths.len(), 5);
/// assert!(depths.iter().all(|&d| d >= 0.0));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn rpd_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: &[f64],
    nproj: usize,
    nderiv: usize,
) -> Vec<f64> {
    rpd_depth_1d_seeded(data_obj, data_ori, argvals, nproj, nderiv, None)
}

/// Compute RPD depth with optional seed for reproducibility.
///
/// See [`rpd_depth_1d`] for details. When `seed` is `Some`, the random
/// projections are generated deterministically.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn rpd_depth_1d_seeded(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: &[f64],
    nproj: usize,
    nderiv: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    let m = data_obj.ncols();

    // Edge case: empty data
    if data_obj.nrows() == 0 || data_ori.nrows() == 0 || m == 0 || nproj == 0 {
        return Vec::new();
    }

    // Edge case: too few grid points for requested derivatives
    if m <= nderiv {
        return vec![0.0; data_obj.nrows()];
    }

    // When nderiv == 0, skip augmentation and delegate directly
    if nderiv == 0 {
        return super::random_depth_core(
            data_obj,
            data_ori,
            nproj,
            seed,
            0.0,
            |acc, d| acc + d,
            |acc, n| acc / n as f64,
        );
    }

    // Compute augmented dimension: m + (m-1) + (m-2) + ... + (m-nderiv)
    let aug_dim: usize = (0..=nderiv).map(|k| m - k).sum();

    let aug_obj = build_augmented(data_obj, argvals, nderiv, aug_dim);
    let aug_ori = build_augmented(data_ori, argvals, nderiv, aug_dim);

    super::random_depth_core(
        &aug_obj,
        &aug_ori,
        nproj,
        seed,
        0.0,
        |acc, d| acc + d,
        |acc, n| acc / n as f64,
    )
}

/// Build augmented matrix by concatenating function values with finite-difference
/// derivatives up to order `nderiv`.
///
/// For each curve, the augmented row is:
/// `[f(t_1), ..., f(t_m), f'(t_1), ..., f'(t_{m-1}), f''(t_1), ..., f''(t_{m-2}), ...]`
fn build_augmented(data: &FdMatrix, argvals: &[f64], nderiv: usize, aug_dim: usize) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();

    // We'll work with row-major intermediate storage, then convert to column-major.
    // aug_data is column-major: aug_data[i + col * n]
    let mut aug_data = vec![0.0; n * aug_dim];

    // Copy original values (derivative order 0)
    for j in 0..m {
        for i in 0..n {
            aug_data[i + j * n] = data[(i, j)];
        }
    }

    // Compute derivatives iteratively.
    // We keep a buffer of the "previous derivative" values per curve.
    // prev_deriv[i * prev_len + j] = d^(k-1)/dt^(k-1) f_i(t_j)
    let mut prev_len = m;
    let mut prev_deriv: Vec<f64> = (0..n * m)
        .map(|idx| {
            let i = idx % n;
            let j = idx / n;
            data[(i, j)]
        })
        .collect();

    let mut col_offset = m; // where to start writing in augmented matrix

    for k in 1..=nderiv {
        let new_len = m - k;
        let mut new_deriv = vec![0.0; n * new_len];

        // The argvals spacing for derivative k uses the midpoints from the
        // previous level. For simplicity, we use the spacing from the
        // original grid points that are k apart:
        // d^k f / dt^k ≈ (d^{k-1}f(t_{j+1}) - d^{k-1}f(t_j)) / (t_{j+1} - t_j)
        //
        // For the k-th derivative, the denominator uses argvals[j+1] - argvals[j]
        // at the level of the (k-1)-th derivative's grid.
        // After the first derivative, the grid points shift. We approximate using
        // the original grid spacing: argvals[j+k] - argvals[j+k-1] for derivative
        // level, but for finite differences we use the spacing between consecutive
        // points of the previous derivative's implicit grid.
        //
        // The standard approach: the k-th finite difference uses
        // (prev[j+1] - prev[j]) / (argvals[j+1] - argvals[j]) where argvals
        // here refers to the spacing appropriate for this derivative level.
        // For the first derivative: dt_j = argvals[j+1] - argvals[j]
        // For the second derivative operating on first-derivative values at
        // midpoints (argvals[j] + argvals[j+1])/2:
        //   dt_j = midpoint[j+1] - midpoint[j] = (argvals[j+2] - argvals[j]) / 2
        //
        // For simplicity and consistency with R's depth.RPD, we use the spacing
        // of consecutive original grid points offset by (k-1):
        //   dt_j = argvals[j + 1 + (k-1)] - argvals[j + (k-1)]
        //        = argvals[j + k] - argvals[j + k - 1]
        // but actually the simplest correct approach for iterated differences is
        // just: (prev[j+1] - prev[j]) / (argvals[j+1] - argvals[j]) at each level.
        // However, the "argvals" for derivative level k are the midpoints from
        // level k-1. Let's just use a constant dt approach: we divide by the
        // spacing of the *previous level's* implicit grid.

        for j in 0..new_len {
            // Spacing: use the difference of the original argvals indices
            // that correspond to the endpoints of this finite difference.
            // For iterated forward differences, the j-th value at level k
            // spans original indices [j, j+k], so we use:
            //   dt = argvals[j + k] - argvals[j + k - 1]
            // But more precisely, for the simple forward-difference iteration,
            // the denominator at each level should be the spacing of the
            // previous derivative's grid. Since the previous derivative's j-th
            // value corresponds to the interval [argvals[j + k-1], argvals[j + k]],
            // we approximate its "position" as the midpoint, and the spacing
            // between consecutive positions is:
            //   (argvals[j+k] + argvals[j+k-1])/2 - (argvals[j+k-1] + argvals[j+k-2])/2
            //   = (argvals[j+k] - argvals[j+k-2]) / 2
            //
            // For simplicity and numerical stability, we just use:
            //   dt = argvals[j+1] - argvals[j]  (for k=1)
            //   dt = argvals[j+2] - argvals[j+1] (for k=2, operating on first-deriv values)
            // i.e., argvals[j + k] - argvals[j + k - 1]
            let dt = argvals[j + k] - argvals[j + k - 1];
            let inv_dt = if dt.abs() < 1e-30 { 0.0 } else { 1.0 / dt };

            for i in 0..n {
                let val = (prev_deriv[i + (j + 1) * n] - prev_deriv[i + j * n]) * inv_dt;
                new_deriv[i + j * n] = val;
                aug_data[i + (col_offset + j) * n] = val;
            }
        }

        col_offset += new_len;
        prev_deriv = new_deriv;
        prev_len = new_len;
    }

    // Suppress unused variable warning
    let _ = prev_len;

    FdMatrix::from_column_major(aug_data, n, aug_dim).unwrap()
}
