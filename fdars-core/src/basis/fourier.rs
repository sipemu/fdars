//! Fourier basis functions.

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
    let n = t.len();
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);

    let mut basis = vec![0.0; n * nbasis];

    for (i, &ti) in t.iter().enumerate() {
        let x = 2.0 * PI * (ti - t_min) / period;

        basis[i] = 1.0;

        let mut k = 1;
        let mut freq = 1;
        while k < nbasis {
            if k < nbasis {
                basis[i + k * n] = (f64::from(freq) * x).sin();
                k += 1;
            }
            if k < nbasis {
                basis[i + k * n] = (f64::from(freq) * x).cos();
                k += 1;
            }
            freq += 1;
        }
    }

    basis
}
