//! Fourier basis functions.

use crate::autodiff::Scalar;
use std::f64::consts::PI;

/// Compute Fourier basis matrix.
///
/// The period is automatically set to the range of evaluation points (t_max - t_min).
/// For explicit period control, use `fourier_basis_with_period`.
///
/// # Examples
///
/// ```
/// use fdars_core::basis::fourier::fourier_basis;
///
/// let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let basis = fourier_basis(&t, 5);
/// // Column-major: n_points x nbasis
/// assert_eq!(basis.len(), 20 * 5);
/// // First basis function is constant 1
/// assert!((basis[0] - 1.0).abs() < 1e-10);
/// ```
pub fn fourier_basis(t: &[f64], nbasis: usize) -> Vec<f64> {
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let period = t_max - t_min;
    fourier_basis_with_period(t, nbasis, period)
}

/// Compute Fourier basis matrix with explicit period.
///
/// This function creates a Fourier basis expansion where the period can be specified
/// independently of the evaluation range. This is essential for seasonal analysis
/// where the seasonal period may differ from the observation window.
///
/// # Arguments
/// * `t` - Evaluation points
/// * `nbasis` - Number of basis functions (1 constant + pairs of sin/cos)
/// * `period` - The period for the Fourier basis
///
/// # Returns
/// Column-major matrix (n_points x nbasis) stored as flat vector
pub fn fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64> {
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
    fourier_basis_eval(t, nbasis, period, t_min)
}

/// Generic-over-`Scalar` Fourier basis evaluation (differentiable through `t`).
///
/// This is the differentiable core (family 1 of DIF-F2 / DOP-01): the evaluation
/// points `t` carry the scalar type `T`, while `period`, `t_min`, and `nbasis` are
/// fixed f64/`usize` basis-structure parameters. At `T = f64` it reproduces the
/// numerics of [`fourier_basis_with_period`] bit-for-bit (`f64::from_f64` is the
/// identity and the `2π·(t-t_min)/period` operation order is preserved).
///
/// The f64 wrappers ([`fourier_basis`], [`fourier_basis_with_period`]) derive
/// `t_min` from `t` and delegate here, so their public signatures are unchanged.
pub fn fourier_basis_eval<T: Scalar>(t: &[T], nbasis: usize, period: f64, t_min: f64) -> Vec<T> {
    let n = t.len();

    let mut basis = vec![T::zero(); n * nbasis];

    for (i, &ti) in t.iter().enumerate() {
        // Preserve the exact f64 operation order `2*PI*(ti - t_min)/period` so the
        // T=f64 path is bit-identical to fourier_basis_with_period (do NOT precompute
        // a `2*PI/period` scale — that reorders the FP ops and breaks parity).
        let x = T::from_f64(2.0 * PI) * (ti - T::from_f64(t_min)) / T::from_f64(period);

        basis[i] = T::one();

        let mut k = 1;
        let mut freq = 1;
        while k < nbasis {
            if k < nbasis {
                basis[i + k * n] = (T::from_f64(f64::from(freq)) * x).sin();
                k += 1;
            }
            if k < nbasis {
                basis[i + k * n] = (T::from_f64(f64::from(freq)) * x).cos();
                k += 1;
            }
            freq += 1;
        }
    }

    basis
}
