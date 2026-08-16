//! Constant (intercept) basis function.
//!
//! Provides the trivial single-function basis whose only member is the
//! constant function `f(t) = 1`. Its evaluation on a grid `t` is a
//! column-major `m × 1` matrix of ones — the intercept column of a
//! regression design matrix. This mirrors R's `create.constant.basis`
//! (`fda`) and `fdata.usc`'s constant basis.

/// Evaluate the constant basis on a grid of evaluation points.
///
/// Returns a column-major `m × 1` matrix (`m = t.len()`) of ones, stored as a
/// flat `Vec<f64>` — matching the `Vec<f64>`-returning, column-major convention
/// of [`bspline_basis`](crate::basis::bspline_basis) and
/// [`fourier_basis`](crate::basis::fourier_basis). The single basis function is
/// the constant `1`, so every entry is `1.0`. Used as the intercept column of a
/// regression design matrix.
///
/// This constructor is infallible: an empty grid yields an empty matrix.
///
/// # Examples
///
/// ```
/// use fdars_core::basis::constant::constant_basis;
///
/// let t = vec![0.0, 0.5, 1.0];
/// let basis = constant_basis(&t);
/// // Column-major: m x 1 all-ones column
/// assert_eq!(basis.len(), 3);
/// assert!(basis.iter().all(|&v| (v - 1.0).abs() < 1e-12));
/// ```
pub fn constant_basis(t: &[f64]) -> Vec<f64> {
    vec![1.0; t.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ones_column_shape() {
        let t = vec![0.0, 0.5, 1.0];
        let basis = constant_basis(&t);
        // m x 1 column-major → length m, every entry 1.0
        assert_eq!(basis.len(), 3);
        for &v in &basis {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn empty_input_no_panic() {
        let basis = constant_basis(&[]);
        assert_eq!(basis.len(), 0);
    }

    #[test]
    fn intercept_only_fit_reproduces_mean() {
        // Design matrix X = constant_basis(t) is an m x 1 ones column.
        // The least-squares solution of X beta = y for the single-column X is
        // beta = (Xᵀy) / (XᵀX) = sum(y) / m = mean(y). An intercept-only fit
        // must therefore reproduce the response mean.
        let t = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![3.0, -1.0, 7.5, 2.0, 0.5];
        let x = constant_basis(&t);
        assert_eq!(x.len(), y.len());

        let xty: f64 = x.iter().zip(&y).map(|(&xi, &yi)| xi * yi).sum();
        let xtx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let beta = xty / xtx;

        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert!((beta - mean_y).abs() < 1e-10, "beta={beta}, mean={mean_y}");
    }
}
