//! Covariance kernels and Gaussian process generation.
//!
//! This module provides a flexible [`CovKernel`] enum for defining covariance
//! functions (Gaussian, Matern, Periodic, etc.) with kernel algebra via
//! [`Sum`](CovKernel::Sum) and [`Product`](CovKernel::Product) combinators,
//! and a [`generate_gaussian_process`] function for drawing sample paths from
//! a Gaussian process with a given kernel and optional mean function.

use crate::error::FdarError;
use crate::linalg::cholesky_d;
use crate::matrix::FdMatrix;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

/// Covariance kernel specification.
///
/// Each variant encodes a family of covariance functions `k(s, t)`.
/// Kernels can be composed with [`Sum`](CovKernel::Sum) and
/// [`Product`](CovKernel::Product) to build richer covariance structures.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CovKernel {
    /// Squared-exponential (RBF) kernel: `variance * exp(-0.5 * ((s-t)/length_scale)^2)`.
    Gaussian { length_scale: f64, variance: f64 },
    /// Exponential (Ornstein-Uhlenbeck) kernel: `variance * exp(-|s-t| / length_scale)`.
    Exponential { length_scale: f64, variance: f64 },
    /// Matern kernel with smoothness parameter `nu`.
    ///
    /// Closed-form expressions are used for `nu = 0.5` (exponential),
    /// `nu = 1.5`, and `nu = 2.5`. For other values of `nu` the general
    /// formula with a gamma-function approximation is used.
    Matern {
        length_scale: f64,
        variance: f64,
        nu: f64,
    },
    /// Brownian motion (Wiener process) kernel: `variance * min(s, t)`.
    Brownian { variance: f64 },
    /// Periodic kernel: `variance * exp(-2 * sin^2(pi * |s-t| / period) / length_scale^2)`.
    Periodic {
        length_scale: f64,
        variance: f64,
        period: f64,
    },
    /// Linear kernel: `variance * (s - offset) * (t - offset)`.
    Linear { variance: f64, offset: f64 },
    /// Polynomial kernel: `(variance * s * t + offset)^degree`.
    Polynomial {
        variance: f64,
        offset: f64,
        degree: u32,
    },
    /// White noise kernel: `variance * delta(s, t)`.
    WhiteNoise { variance: f64 },
    /// Sum of two kernels: `k1(s,t) + k2(s,t)`.
    Sum(Box<CovKernel>, Box<CovKernel>),
    /// Product of two kernels: `k1(s,t) * k2(s,t)`.
    Product(Box<CovKernel>, Box<CovKernel>),
}

impl CovKernel {
    /// Evaluate the covariance function `k(s, t)`.
    pub fn eval(&self, s: f64, t: f64) -> f64 {
        match self {
            CovKernel::Gaussian {
                length_scale,
                variance,
            } => {
                let d = (s - t) / length_scale;
                variance * (-0.5 * d * d).exp()
            }
            CovKernel::Exponential {
                length_scale,
                variance,
            } => {
                let d = (s - t).abs() / length_scale;
                variance * (-d).exp()
            }
            CovKernel::Matern {
                length_scale,
                variance,
                nu,
            } => eval_matern(s, t, *length_scale, *variance, *nu),
            CovKernel::Brownian { variance } => {
                if s >= 0.0 && t >= 0.0 {
                    variance * s.min(t)
                } else {
                    0.0
                }
            }
            CovKernel::Periodic {
                length_scale,
                variance,
                period,
            } => {
                let sin_val = (PI * (s - t).abs() / period).sin();
                variance * (-2.0 * sin_val * sin_val / (length_scale * length_scale)).exp()
            }
            CovKernel::Linear { variance, offset } => variance * (s - offset) * (t - offset),
            CovKernel::Polynomial {
                variance,
                offset,
                degree,
            } => {
                let inner = variance * s * t + offset;
                inner.powi(*degree as i32)
            }
            CovKernel::WhiteNoise { variance } => {
                if (s - t).abs() < 1e-15 {
                    *variance
                } else {
                    0.0
                }
            }
            CovKernel::Sum(k1, k2) => k1.eval(s, t) + k2.eval(s, t),
            CovKernel::Product(k1, k2) => k1.eval(s, t) * k2.eval(s, t),
        }
    }

    /// Validate kernel parameters, returning an error for negative variance or
    /// length scale, or invalid degree.
    fn validate(&self) -> Result<(), FdarError> {
        match self {
            CovKernel::Gaussian {
                length_scale,
                variance,
            } => {
                check_positive("variance", *variance)?;
                check_positive("length_scale", *length_scale)?;
            }
            CovKernel::Exponential {
                length_scale,
                variance,
            } => {
                check_positive("variance", *variance)?;
                check_positive("length_scale", *length_scale)?;
            }
            CovKernel::Matern {
                length_scale,
                variance,
                nu,
            } => {
                check_positive("variance", *variance)?;
                check_positive("length_scale", *length_scale)?;
                check_positive("nu", *nu)?;
            }
            CovKernel::Brownian { variance } => {
                check_positive("variance", *variance)?;
            }
            CovKernel::Periodic {
                length_scale,
                variance,
                period,
            } => {
                check_positive("variance", *variance)?;
                check_positive("length_scale", *length_scale)?;
                check_positive("period", *period)?;
            }
            CovKernel::Linear { variance, .. } => {
                check_positive("variance", *variance)?;
            }
            CovKernel::Polynomial {
                variance, degree, ..
            } => {
                check_positive("variance", *variance)?;
                if *degree == 0 {
                    return Err(FdarError::InvalidParameter {
                        parameter: "degree",
                        message: "must be >= 1".to_string(),
                    });
                }
            }
            CovKernel::WhiteNoise { variance } => {
                check_positive("variance", *variance)?;
            }
            CovKernel::Sum(k1, k2) => {
                k1.validate()?;
                k2.validate()?;
            }
            CovKernel::Product(k1, k2) => {
                k1.validate()?;
                k2.validate()?;
            }
        }
        Ok(())
    }
}

/// Check that a parameter is strictly positive.
fn check_positive(name: &'static str, value: f64) -> Result<(), FdarError> {
    if value <= 0.0 || value.is_nan() {
        return Err(FdarError::InvalidParameter {
            parameter: name,
            message: format!("must be positive, got {value}"),
        });
    }
    Ok(())
}

/// Evaluate the Matern covariance function.
///
/// Uses closed-form expressions for `nu = 0.5`, `1.5`, and `2.5`.
/// For half-integer `nu = p + 0.5`, uses the general half-integer closed form.
/// For other values of `nu`, uses the general formula with the
/// Lanczos gamma approximation and a robust Bessel K_nu implementation.
fn eval_matern(s: f64, t: f64, length_scale: f64, variance: f64, nu: f64) -> f64 {
    let d = (s - t).abs();
    if d < 1e-15 {
        return variance;
    }
    let r = d / length_scale;
    let z = (2.0 * nu).sqrt() * r;

    // Check if nu is a half-integer: nu = p + 0.5 for integer p >= 0
    let twice_nu = 2.0 * nu;
    let twice_nu_rounded = twice_nu.round();
    if (twice_nu - twice_nu_rounded).abs() < 1e-10 && twice_nu_rounded >= 1.0 {
        let twice_nu_int = twice_nu_rounded as u64;
        if twice_nu_int % 2 == 1 {
            // nu is half-integer: use the closed-form polynomial expression
            // For nu = p + 0.5, K_{p+1/2}(z) = sqrt(pi/(2z)) * exp(-z) * sum_{k=0}^p (p+k)!/(k!(p-k)!) * (2z)^{-k}
            let p = (twice_nu_int - 1) / 2;
            return variance * matern_half_integer(z, p as usize);
        }
    }

    // General formula: k(r) = variance * 2^(1-nu) / Gamma(nu) * (sqrt(2*nu)*r)^nu * K_nu(sqrt(2*nu)*r)
    // Work in log-space for numerical stability
    let log_prefactor = (1.0 - nu) * 2.0_f64.ln() - ln_gamma(nu) + nu * z.ln();
    let knu = bessel_knu(nu, z);
    if knu <= 0.0 {
        return 0.0;
    }
    variance * (log_prefactor + knu.ln()).exp()
}

/// Closed-form Matern for half-integer nu = p + 0.5.
///
/// Uses the Rasmussen & Williams formula (eq. 4.17): for nu = p + 1/2,
/// `k(r) = exp(-z) * (p! / (2p)!) * sum_{i=0}^{p} ((p+i)! / (i! * (p-i)!)) * (2z)^{p-i}`
/// where `z = sqrt(2*nu) * r / l`.
///
/// This avoids all Bessel function evaluation and is numerically exact.
fn matern_half_integer(z: f64, p: usize) -> f64 {
    let two_z = 2.0 * z;
    let mut poly = 0.0;
    for i in 0..=p {
        // coefficient = (p+i)! / (i! * (p-i)!)
        let coeff = factorial(p + i) as f64 / (factorial(i) as f64 * factorial(p - i) as f64);
        // (2z)^{p-i}
        let power = two_z.powi((p - i) as i32);
        poly += coeff * power;
    }
    // Prefactor: p! / (2p)!
    let prefactor = factorial(p) as f64 / factorial(2 * p) as f64;
    prefactor * poly * (-z).exp()
}

/// Compute n! (for small n).
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product::<u64>().max(1)
}

/// Lanczos approximation of the log-gamma function.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        let log_pi = PI.ln();
        return log_pi - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x);
    }

    let g = 7.0;
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    let coefficients = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let x = x - 1.0;
    let mut sum = coefficients[0];
    for (i, &c) in coefficients.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
}

/// Compute the modified Bessel function of the second kind K_nu(z).
///
/// Uses the series representation K_nu(z) = (pi/2)(I_{-nu}(z) - I_nu(z)) / sin(nu*pi)
/// for non-integer nu, and the asymptotic expansion for large z.
fn bessel_knu(nu: f64, z: f64) -> f64 {
    if z <= 0.0 {
        return f64::INFINITY;
    }

    // For large z, use asymptotic expansion (good for all nu)
    if z > 50.0 {
        return bessel_knu_asymptotic(nu, z);
    }

    // Check if nu is close to an integer
    let nu_rounded = nu.round();
    let is_integer = (nu - nu_rounded).abs() < 1e-10;

    if is_integer {
        // For integer nu, use the Miller backward recurrence from K_0 and K_1
        let n = nu_rounded.abs() as u32;
        bessel_kn_miller(n, z)
    } else {
        // For non-integer nu, use the series K_nu = (pi/2)(I_{-nu} - I_nu)/sin(nu*pi)
        let sin_nu_pi = (nu * PI).sin();
        let i_neg_nu = bessel_inu_series(-nu, z);
        let i_nu = bessel_inu_series(nu, z);
        (PI / 2.0) * (i_neg_nu - i_nu) / sin_nu_pi
    }
}

/// Asymptotic expansion of K_nu(z) for large z.
fn bessel_knu_asymptotic(nu: f64, z: f64) -> f64 {
    let prefactor = (PI / (2.0 * z)).sqrt() * (-z).exp();
    let mu = 4.0 * nu * nu;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=20 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * z * kf);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    prefactor * sum
}

/// Power series for I_nu(z), the modified Bessel function of the first kind.
fn bessel_inu_series(nu: f64, z: f64) -> f64 {
    let half_z = z / 2.0;
    // First term: (z/2)^nu / Gamma(nu+1)
    let log_first = nu * half_z.ln() - ln_gamma(nu + 1.0);
    let mut term = log_first.exp();
    let mut sum = term;
    let z2_over4 = half_z * half_z;
    for k in 1..=80 {
        let kf = k as f64;
        term *= z2_over4 / (kf * (kf + nu));
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

/// Compute K_n(z) for non-negative integer n using K_0 and K_1 with forward recurrence.
fn bessel_kn_miller(n: u32, z: f64) -> f64 {
    let k0 = bessel_k0_series(z);
    if n == 0 {
        return k0;
    }
    let k1 = bessel_k1_series(z);
    if n == 1 {
        return k1;
    }
    let mut km1 = k0;
    let mut k_cur = k1;
    for i in 1..n {
        let k_next = (2.0 * i as f64 / z) * k_cur + km1;
        km1 = k_cur;
        k_cur = k_next;
    }
    k_cur
}

/// K_0(z) via the Temme series (Abramowitz & Stegun 9.6.13).
fn bessel_k0_series(z: f64) -> f64 {
    if z > 2.0 {
        // Asymptotic
        return bessel_knu_asymptotic(0.0, z);
    }
    // K_0(z) = -[ln(z/2) + gamma] * I_0(z) + sum_{k=0}^inf (z/2)^{2k} * h_k / (k!)^2
    // where h_k = sum_{j=1}^k 1/j (harmonic numbers), h_0 = 0
    let euler_gamma = 0.577_215_664_901_532_9;
    let half_z = z / 2.0;
    let ln_half_z = half_z.ln();

    // I_0(z) series
    let mut i0 = 1.0;
    let mut term_i0 = 1.0;
    let z2_over4 = half_z * half_z;
    for k in 1..=30 {
        let kf = k as f64;
        term_i0 *= z2_over4 / (kf * kf);
        i0 += term_i0;
    }

    // The sum part
    let mut sum_part = 0.0;
    let mut term_s = 1.0; // (z/2)^{2k} / (k!)^2 for k=0
    let mut h_k = 0.0;
    sum_part += term_s * h_k; // k=0 contributes 0
    for k in 1..=30 {
        let kf = k as f64;
        term_s *= z2_over4 / (kf * kf);
        h_k += 1.0 / kf;
        sum_part += term_s * h_k;
    }

    -(ln_half_z + euler_gamma) * i0 + sum_part
}

/// K_1(z) via K_1(z) = (1/z) + ln(z/2)*I_1(z) + series (Abramowitz & Stegun).
fn bessel_k1_series(z: f64) -> f64 {
    if z > 2.0 {
        return bessel_knu_asymptotic(1.0, z);
    }
    // Use the relation: K_1(z) = -dK_0/dz, which from the series gives:
    // K_1(z) = 1/z + (ln(z/2) + gamma - 1/2) * z/2 * ... (complex)
    // Simpler: use the Wronskian relation I_1*K_0 + I_0*K_1 = 1/z
    // => K_1 = (1/z - I_1*K_0) / I_0
    let half_z = z / 2.0;
    let z2_over4 = half_z * half_z;

    // I_0(z)
    let mut i0 = 1.0;
    let mut term = 1.0;
    for k in 1..=30 {
        let kf = k as f64;
        term *= z2_over4 / (kf * kf);
        i0 += term;
    }

    // I_1(z) = (z/2) * sum (z/2)^{2k} / (k! * (k+1)!)
    let mut i1 = half_z;
    term = half_z;
    for k in 1..=30 {
        let kf = k as f64;
        term *= z2_over4 / (kf * (kf + 1.0));
        i1 += term;
    }

    let k0 = bessel_k0_series(z);
    (1.0 / z - i1 * k0) / i0
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the m x m covariance matrix K\[i,j\] = k(argvals\[i\], argvals\[j\]).
///
/// The result is a symmetric positive semi-definite matrix stored in an
/// [`FdMatrix`] with `m` rows and `m` columns (column-major layout).
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `argvals` is empty.
/// * [`FdarError::InvalidParameter`] if kernel parameters are invalid
///   (e.g. negative variance or length scale).
#[must_use = "returns the covariance matrix without modifying the kernel"]
pub fn covariance_matrix(kernel: &CovKernel, argvals: &[f64]) -> Result<FdMatrix, FdarError> {
    if argvals.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 1".to_string(),
            actual: "0".to_string(),
        });
    }
    kernel.validate()?;

    let m = argvals.len();
    let mut data = vec![0.0; m * m];

    // Fill column-major: data[i + j * m] = k(argvals[i], argvals[j])
    for j in 0..m {
        for i in 0..m {
            let val = kernel.eval(argvals[i], argvals[j]);
            data[i + j * m] = val;
        }
    }

    FdMatrix::from_column_major(data, m, m)
}

/// Result of Gaussian process sample generation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GaussianProcessResult {
    /// Sample paths, stored as an n x m matrix (n samples, m evaluation points).
    pub samples: FdMatrix,
    /// Evaluation points (length m).
    pub argvals: Vec<f64>,
    /// Kernel used for generation.
    pub kernel: CovKernel,
    /// Mean function used for generation (length m).
    pub mean_function: Vec<f64>,
}

/// Generate `n` sample paths from a Gaussian process.
///
/// Uses Cholesky decomposition of `K + jitter * I` (where `jitter = 1e-10`)
/// to produce `L`, then each sample is `mean + L * z` where `z ~ N(0, I)`.
///
/// # Arguments
///
/// * `n` — number of sample paths to generate.
/// * `kernel` — covariance kernel specification.
/// * `argvals` — evaluation points (length `m`).
/// * `mean_fn` — optional mean function values (length `m`); defaults to zero.
/// * `seed` — optional RNG seed for reproducibility.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `argvals` is empty, `n == 0`, or
///   `mean_fn` length does not match `argvals`.
/// * [`FdarError::InvalidParameter`] if kernel parameters are invalid.
/// * [`FdarError::ComputationFailed`] if Cholesky decomposition fails.
#[must_use = "returns GP samples without modifying inputs"]
pub fn generate_gaussian_process(
    n: usize,
    kernel: &CovKernel,
    argvals: &[f64],
    mean_fn: Option<&[f64]>,
    seed: Option<u64>,
) -> Result<GaussianProcessResult, FdarError> {
    // Validate inputs
    if argvals.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 1".to_string(),
            actual: "0".to_string(),
        });
    }
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "n",
            expected: ">= 1".to_string(),
            actual: "0".to_string(),
        });
    }

    let m = argvals.len();

    let mean = if let Some(mf) = mean_fn {
        if mf.len() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "mean_fn",
                expected: format!("{m}"),
                actual: format!("{}", mf.len()),
            });
        }
        mf.to_vec()
    } else {
        vec![0.0; m]
    };

    kernel.validate()?;

    // Build covariance matrix in row-major for cholesky_d
    let mut cov_row = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            cov_row[i * m + j] = kernel.eval(argvals[i], argvals[j]);
        }
    }

    // Add jitter for numerical stability
    let jitter = 1e-10;
    for i in 0..m {
        cov_row[i * m + i] += jitter;
    }

    // Cholesky decomposition: cov = L * L^T
    let l = cholesky_d(&cov_row, m).map_err(|_| FdarError::ComputationFailed {
        operation: "Cholesky decomposition",
        detail: "covariance matrix is not positive definite — try adding jitter or checking kernel parameters".to_string(),
    })?;

    // Generate samples
    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(StdRng::from_entropy()),
    };

    // Samples stored column-major: data[i + j * n] = sample i at argval j
    let mut data = vec![0.0; n * m];

    for i in 0..n {
        // Draw z ~ N(0, I) of length m
        let z: Vec<f64> = (0..m)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        // Compute L * z (L is row-major m×m lower triangular)
        for j in 0..m {
            let mut val = mean[j];
            for k in 0..=j {
                val += l[j * m + k] * z[k];
            }
            data[i + j * n] = val;
        }
    }

    let samples = FdMatrix::from_column_major(data, n, m)?;

    Ok(GaussianProcessResult {
        samples,
        argvals: argvals.to_vec(),
        kernel: kernel.clone(),
        mean_function: mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // -----------------------------------------------------------------------
    // Kernel evaluation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gaussian_kernel_eval() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        // k(0,0) = 1
        assert!((k.eval(0.0, 0.0) - 1.0).abs() < TOL);
        // k(0,1) = exp(-0.5)
        assert!((k.eval(0.0, 1.0) - (-0.5_f64).exp()).abs() < TOL);
        // symmetry
        assert!((k.eval(0.3, 0.7) - k.eval(0.7, 0.3)).abs() < TOL);
    }

    #[test]
    fn test_gaussian_kernel_variance_scale() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 2.5,
        };
        assert!((k.eval(0.0, 0.0) - 2.5).abs() < TOL);
        assert!((k.eval(0.0, 1.0) - 2.5 * (-0.5_f64).exp()).abs() < TOL);
    }

    #[test]
    fn test_exponential_kernel_eval() {
        let k = CovKernel::Exponential {
            length_scale: 2.0,
            variance: 1.0,
        };
        assert!((k.eval(0.0, 0.0) - 1.0).abs() < TOL);
        // k(0, 1) = exp(-1/2) = exp(-0.5)
        assert!((k.eval(0.0, 1.0) - (-0.5_f64).exp()).abs() < TOL);
    }

    #[test]
    fn test_matern_05_matches_exponential() {
        let matern = CovKernel::Matern {
            length_scale: 1.5,
            variance: 2.0,
            nu: 0.5,
        };
        let exp = CovKernel::Exponential {
            length_scale: 1.5,
            variance: 2.0,
        };
        let points = [0.0, 0.3, 0.7, 1.0, 2.5];
        for &s in &points {
            for &t in &points {
                assert!(
                    (matern.eval(s, t) - exp.eval(s, t)).abs() < 1e-8,
                    "Matern(0.5) != Exponential at ({s}, {t}): {} vs {}",
                    matern.eval(s, t),
                    exp.eval(s, t)
                );
            }
        }
    }

    #[test]
    fn test_matern_15_eval() {
        let k = CovKernel::Matern {
            length_scale: 1.0,
            variance: 1.0,
            nu: 1.5,
        };
        // At s=t, k=variance
        assert!((k.eval(0.0, 0.0) - 1.0).abs() < TOL);
        // k(0, 1) = (1 + sqrt(3)) * exp(-sqrt(3))
        let sqrt3 = 3.0_f64.sqrt();
        let expected = (1.0 + sqrt3) * (-sqrt3).exp();
        assert!((k.eval(0.0, 1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_matern_25_eval() {
        let k = CovKernel::Matern {
            length_scale: 1.0,
            variance: 1.0,
            nu: 2.5,
        };
        assert!((k.eval(0.0, 0.0) - 1.0).abs() < TOL);
        let sqrt5 = 5.0_f64.sqrt();
        let expected = (1.0 + sqrt5 + 5.0 / 3.0) * (-sqrt5).exp();
        assert!((k.eval(0.0, 1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_brownian_kernel_eval() {
        let k = CovKernel::Brownian { variance: 1.0 };
        assert!((k.eval(0.3, 0.7) - 0.3).abs() < TOL);
        assert!((k.eval(0.7, 0.3) - 0.3).abs() < TOL);
        assert!((k.eval(0.0, 0.5) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_periodic_kernel_eval() {
        let k = CovKernel::Periodic {
            length_scale: 1.0,
            variance: 1.0,
            period: 1.0,
        };
        // k(t, t) = variance
        assert!((k.eval(0.5, 0.5) - 1.0).abs() < TOL);
        // k(0, 1) should equal k(0, 0) because period=1
        assert!((k.eval(0.0, 1.0) - 1.0).abs() < 1e-10);
        // symmetry
        assert!((k.eval(0.2, 0.8) - k.eval(0.8, 0.2)).abs() < TOL);
    }

    #[test]
    fn test_linear_kernel_eval() {
        let k = CovKernel::Linear {
            variance: 1.0,
            offset: 0.0,
        };
        assert!((k.eval(2.0, 3.0) - 6.0).abs() < TOL);
        assert!((k.eval(0.0, 5.0) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_polynomial_kernel_eval() {
        let k = CovKernel::Polynomial {
            variance: 1.0,
            offset: 1.0,
            degree: 2,
        };
        // (1*2*3 + 1)^2 = 49
        assert!((k.eval(2.0, 3.0) - 49.0).abs() < TOL);
    }

    #[test]
    fn test_white_noise_kernel_eval() {
        let k = CovKernel::WhiteNoise { variance: 3.0 };
        assert!((k.eval(1.0, 1.0) - 3.0).abs() < TOL);
        assert!((k.eval(1.0, 1.001) - 0.0).abs() < TOL);
    }

    // -----------------------------------------------------------------------
    // Kernel algebra tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sum_kernel() {
        let k1 = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let k2 = CovKernel::WhiteNoise { variance: 0.5 };
        let sum = CovKernel::Sum(Box::new(k1.clone()), Box::new(k2.clone()));

        // At s=t: 1.0 + 0.5 = 1.5
        assert!((sum.eval(0.0, 0.0) - 1.5).abs() < TOL);
        // At s!=t: gaussian + 0 = gaussian
        let val = sum.eval(0.0, 1.0);
        let expected = k1.eval(0.0, 1.0);
        assert!((val - expected).abs() < TOL);
    }

    #[test]
    fn test_product_kernel() {
        let k1 = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 2.0,
        };
        let k2 = CovKernel::Gaussian {
            length_scale: 2.0,
            variance: 3.0,
        };
        let prod = CovKernel::Product(Box::new(k1.clone()), Box::new(k2.clone()));

        let s = 0.0;
        let t = 0.5;
        let expected = k1.eval(s, t) * k2.eval(s, t);
        assert!((prod.eval(s, t) - expected).abs() < TOL);
    }

    // -----------------------------------------------------------------------
    // Covariance matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_covariance_matrix_symmetric() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let argvals: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let cov = covariance_matrix(&k, &argvals).unwrap();
        let m = argvals.len();
        assert_eq!(cov.nrows(), m);
        assert_eq!(cov.ncols(), m);

        // Check symmetry
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < TOL,
                    "not symmetric at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_covariance_matrix_positive_definite() {
        let k = CovKernel::Gaussian {
            length_scale: 0.5,
            variance: 1.0,
        };
        let argvals: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let cov = covariance_matrix(&k, &argvals).unwrap();
        let m = argvals.len();

        // Convert to row-major for cholesky_d
        let mut row_major = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                row_major[i * m + j] = cov[(i, j)];
            }
        }
        // Add tiny jitter
        for i in 0..m {
            row_major[i * m + i] += 1e-10;
        }

        // Cholesky should succeed on a positive definite matrix
        assert!(cholesky_d(&row_major, m).is_ok());
    }

    #[test]
    fn test_covariance_matrix_diagonal_equals_variance() {
        let variance = 2.5;
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance,
        };
        let argvals = vec![0.0, 0.5, 1.0];
        let cov = covariance_matrix(&k, &argvals).unwrap();
        for i in 0..3 {
            assert!((cov[(i, i)] - variance).abs() < TOL);
        }
    }

    // -----------------------------------------------------------------------
    // GP generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gp_deterministic_with_seed() {
        let k = CovKernel::Gaussian {
            length_scale: 0.5,
            variance: 1.0,
        };
        let argvals: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();

        let r1 = generate_gaussian_process(5, &k, &argvals, None, Some(42)).unwrap();
        let r2 = generate_gaussian_process(5, &k, &argvals, None, Some(42)).unwrap();

        assert_eq!(r1.samples.nrows(), 5);
        assert_eq!(r1.samples.ncols(), 20);

        // Same seed should produce identical results
        for i in 0..5 {
            for j in 0..20 {
                assert!(
                    (r1.samples[(i, j)] - r2.samples[(i, j)]).abs() < TOL,
                    "mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_gp_different_seeds_differ() {
        let k = CovKernel::Gaussian {
            length_scale: 0.5,
            variance: 1.0,
        };
        let argvals: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();

        let r1 = generate_gaussian_process(3, &k, &argvals, None, Some(1)).unwrap();
        let r2 = generate_gaussian_process(3, &k, &argvals, None, Some(2)).unwrap();

        // At least some values should differ
        let mut differ = false;
        for i in 0..3 {
            for j in 0..10 {
                if (r1.samples[(i, j)] - r2.samples[(i, j)]).abs() > 1e-6 {
                    differ = true;
                }
            }
        }
        assert!(differ, "different seeds should produce different samples");
    }

    #[test]
    fn test_gp_mean_and_variance() {
        let variance = 1.0;
        let k = CovKernel::Gaussian {
            length_scale: 0.3,
            variance,
        };
        let argvals: Vec<f64> = (0..5).map(|i| i as f64 * 0.25).collect();
        let n = 10_000;

        let result = generate_gaussian_process(n, &k, &argvals, None, Some(123)).unwrap();

        // Check empirical mean is close to zero and variance close to 1.0
        let m = argvals.len();
        for j in 0..m {
            let col: Vec<f64> = (0..n).map(|i| result.samples[(i, j)]).collect();
            let mean = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            assert!(
                mean.abs() < 0.1,
                "empirical mean at j={j} is {mean}, expected ~0"
            );
            assert!(
                (var - variance).abs() < 0.15,
                "empirical variance at j={j} is {var}, expected ~{variance}"
            );
        }
    }

    #[test]
    fn test_gp_with_mean_function() {
        let k = CovKernel::Gaussian {
            length_scale: 0.3,
            variance: 1.0,
        };
        let argvals = vec![0.0, 0.5, 1.0];
        let mean_fn = vec![10.0, 20.0, 30.0];
        let n = 5000;

        let result = generate_gaussian_process(n, &k, &argvals, Some(&mean_fn), Some(99)).unwrap();
        assert_eq!(result.mean_function, mean_fn);

        // Empirical mean should be close to the specified mean function
        for j in 0..3 {
            let col_mean: f64 = (0..n).map(|i| result.samples[(i, j)]).sum::<f64>() / n as f64;
            assert!(
                (col_mean - mean_fn[j]).abs() < 0.2,
                "empirical mean at j={j} is {col_mean}, expected ~{}",
                mean_fn[j]
            );
        }
    }

    #[test]
    fn test_gp_result_fields() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let argvals = vec![0.0, 0.5, 1.0];
        let result = generate_gaussian_process(3, &k, &argvals, None, Some(0)).unwrap();
        assert_eq!(result.argvals, argvals);
        assert_eq!(result.kernel, k);
        assert_eq!(result.mean_function, vec![0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Error case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_covariance_matrix_empty_argvals() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let err = covariance_matrix(&k, &[]).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "argvals");
            }
            other => panic!("expected InvalidDimension, got {other:?}"),
        }
    }

    #[test]
    fn test_gp_empty_argvals() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let err = generate_gaussian_process(5, &k, &[], None, None).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "argvals");
            }
            other => panic!("expected InvalidDimension, got {other:?}"),
        }
    }

    #[test]
    fn test_gp_n_zero() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let err = generate_gaussian_process(0, &k, &[0.0, 1.0], None, None).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "n");
            }
            other => panic!("expected InvalidDimension, got {other:?}"),
        }
    }

    #[test]
    fn test_gp_wrong_mean_fn_length() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let err = generate_gaussian_process(3, &k, &[0.0, 0.5, 1.0], Some(&[1.0, 2.0]), None)
            .unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "mean_fn");
            }
            other => panic!("expected InvalidDimension, got {other:?}"),
        }
    }

    #[test]
    fn test_negative_variance_error() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: -1.0,
        };
        let err = covariance_matrix(&k, &[0.0, 1.0]).unwrap_err();
        match err {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "variance");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_negative_length_scale_error() {
        let k = CovKernel::Gaussian {
            length_scale: -1.0,
            variance: 1.0,
        };
        let err = covariance_matrix(&k, &[0.0, 1.0]).unwrap_err();
        match err {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "length_scale");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_negative_variance_in_sum_error() {
        let k = CovKernel::Sum(
            Box::new(CovKernel::Gaussian {
                length_scale: 1.0,
                variance: 1.0,
            }),
            Box::new(CovKernel::WhiteNoise { variance: -0.1 }),
        );
        let err = covariance_matrix(&k, &[0.0, 1.0]).unwrap_err();
        match err {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "variance");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Kernel-specific edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_matern_general_nu() {
        // Test with a non-standard nu value (e.g. 3.5)
        let k = CovKernel::Matern {
            length_scale: 1.0,
            variance: 1.0,
            nu: 3.5,
        };
        // At s=t, should be variance
        assert!((k.eval(0.0, 0.0) - 1.0).abs() < TOL);
        // Should be positive and decreasing with distance
        let v1 = k.eval(0.0, 0.5);
        let v2 = k.eval(0.0, 1.0);
        assert!(v1 > 0.0, "Matern should be positive: {v1}");
        assert!(v2 > 0.0, "Matern should be positive: {v2}");
        assert!(v1 > v2, "Matern should decrease with distance");
    }

    #[test]
    fn test_brownian_negative_args() {
        let k = CovKernel::Brownian { variance: 1.0 };
        // Brownian motion is defined for non-negative arguments
        assert!((k.eval(-1.0, 0.5)).abs() < TOL);
    }

    #[test]
    fn test_periodic_kernel_periodicity() {
        let period = 2.0;
        let k = CovKernel::Periodic {
            length_scale: 1.0,
            variance: 1.0,
            period,
        };
        // k(0, period) should be close to k(0, 0)
        assert!((k.eval(0.0, period) - k.eval(0.0, 0.0)).abs() < 1e-10);
        // k(0.3, 0.3 + period) should be close to k(0.3, 0.3)
        assert!((k.eval(0.3, 0.3 + period) - k.eval(0.3, 0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_gp_single_point() {
        let k = CovKernel::Gaussian {
            length_scale: 1.0,
            variance: 1.0,
        };
        let result = generate_gaussian_process(10, &k, &[0.5], None, Some(7)).unwrap();
        assert_eq!(result.samples.nrows(), 10);
        assert_eq!(result.samples.ncols(), 1);
    }

    #[test]
    fn test_gp_with_brownian_kernel() {
        let k = CovKernel::Brownian { variance: 1.0 };
        let argvals: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
        // Brownian kernel on positive argvals should work
        let result = generate_gaussian_process(5, &k, &argvals, None, Some(42)).unwrap();
        assert_eq!(result.samples.nrows(), 5);
        assert_eq!(result.samples.ncols(), 10);
    }

    #[test]
    fn test_gp_with_sum_kernel() {
        let k = CovKernel::Sum(
            Box::new(CovKernel::Gaussian {
                length_scale: 0.5,
                variance: 1.0,
            }),
            Box::new(CovKernel::WhiteNoise { variance: 0.1 }),
        );
        let argvals: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let result = generate_gaussian_process(3, &k, &argvals, None, Some(55)).unwrap();
        assert_eq!(result.samples.nrows(), 3);
        assert_eq!(result.samples.ncols(), 10);
    }

    #[test]
    fn test_polynomial_degree_zero_error() {
        let k = CovKernel::Polynomial {
            variance: 1.0,
            offset: 1.0,
            degree: 0,
        };
        let err = covariance_matrix(&k, &[0.0, 1.0]).unwrap_err();
        match err {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "degree");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }
}
