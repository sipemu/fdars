//! Simulation functions for functional data.
//!
//! This module provides tools for generating synthetic functional data using
//! the Karhunen-Loève expansion and various eigenfunction/eigenvalue configurations.
//!
//! ## Overview
//!
//! Functional data can be simulated using the truncated Karhunen-Loève representation:
//! ```text
//! f_i(t) = μ(t) + Σ_{k=1}^{M} ξ_{ik} φ_k(t)
//! ```
//! where:
//! - μ(t) is the mean function
//! - φ_k(t) are orthonormal eigenfunctions
//! - ξ_{ik} ~ N(0, λ_k) are random scores with variances given by eigenvalues
//!
//! ## Eigenfunction Types
//!
//! - **Fourier**: sin/cos basis functions, suitable for periodic data
//! - **Legendre**: Orthonormal Legendre polynomials on [0,1]
//! - **Wiener**: Eigenfunctions of the Wiener process

use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;

/// Eigenfunction type enum for simulation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EFunType {
    /// Fourier basis: 1, sqrt(2)*cos(2πkt), sqrt(2)*sin(2πkt)
    Fourier = 0,
    /// Orthonormal Legendre polynomials on [0,1]
    Poly = 1,
    /// Higher-order Legendre polynomials (starting at degree 2)
    PolyHigh = 2,
    /// Wiener process eigenfunctions: sqrt(2)*sin((k-0.5)πt)
    Wiener = 3,
}

impl EFunType {
    /// Create from integer (for FFI)
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(EFunType::Fourier),
            1 => Some(EFunType::Poly),
            2 => Some(EFunType::PolyHigh),
            3 => Some(EFunType::Wiener),
            _ => None,
        }
    }
}

/// Eigenvalue decay type for simulation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EValType {
    /// Linear decay: λ_k = 1/k
    Linear = 0,
    /// Exponential decay: λ_k = exp(-k)
    Exponential = 1,
    /// Wiener eigenvalues: λ_k = 1/((k-0.5)π)²
    Wiener = 2,
}

impl EValType {
    /// Create from integer (for FFI)
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(EValType::Linear),
            1 => Some(EValType::Exponential),
            2 => Some(EValType::Wiener),
            _ => None,
        }
    }
}

// =============================================================================
// Eigenfunction Computation
// =============================================================================

/// Compute Fourier eigenfunctions on [0,1].
///
/// The Fourier basis consists of:
/// - φ_1(t) = 1
/// - φ_{2k}(t) = √2 cos(2πkt) for k = 1, 2, ...
/// - φ_{2k+1}(t) = √2 sin(2πkt) for k = 1, 2, ...
///
/// # Arguments
/// * `t` - Evaluation points in [0,1]
/// * `m` - Number of eigenfunctions
///
/// # Returns
/// Column-major matrix of size `len(t) × m` as flat vector
pub fn fourier_eigenfunctions(t: &[f64], m: usize) -> Vec<f64> {
    let n = t.len();
    let mut phi = vec![0.0; n * m];
    let sqrt2 = 2.0_f64.sqrt();

    for (i, &ti) in t.iter().enumerate() {
        // φ_1(t) = 1
        phi[i] = 1.0;

        let mut k = 1; // current eigenfunction index
        let mut freq = 1; // frequency index

        while k < m {
            // sin term: sqrt(2) * sin(2*pi*freq*t)
            if k < m {
                phi[i + k * n] = sqrt2 * (2.0 * PI * freq as f64 * ti).sin();
                k += 1;
            }
            // cos term: sqrt(2) * cos(2*pi*freq*t)
            if k < m {
                phi[i + k * n] = sqrt2 * (2.0 * PI * freq as f64 * ti).cos();
                k += 1;
            }
            freq += 1;
        }
    }
    phi
}

/// Compute Legendre polynomial eigenfunctions on [0,1].
///
/// Uses orthonormalized Legendre polynomials. The normalization factor is
/// √(2n+1) where n is the polynomial degree, which ensures unit L² norm on [0,1].
///
/// # Arguments
/// * `t` - Evaluation points in [0,1]
/// * `m` - Number of eigenfunctions
/// * `high` - If true, start at degree 2 (PolyHigh), otherwise start at degree 0
///
/// # Returns
/// Column-major matrix of size `len(t) × m` as flat vector
pub fn legendre_eigenfunctions(t: &[f64], m: usize, high: bool) -> Vec<f64> {
    let n = t.len();
    let mut phi = vec![0.0; n * m];
    let start_deg = if high { 2 } else { 0 };

    for (i, &ti) in t.iter().enumerate() {
        // Transform from [0,1] to [-1,1]
        let x = 2.0 * ti - 1.0;

        for j in 0..m {
            let deg = start_deg + j;
            // Compute Legendre polynomial P_deg(x)
            let p = legendre_p(x, deg);
            // Normalize: ||P_n||² on [-1,1] = 2/(2n+1), on [0,1] = 1/(2n+1)
            let norm = ((2 * deg + 1) as f64).sqrt();
            phi[i + j * n] = p * norm;
        }
    }
    phi
}

/// Compute Legendre polynomial P_n(x) using recurrence relation.
///
/// The three-term recurrence is:
/// (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
fn legendre_p(x: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut p_prev = 1.0;
    let mut p_curr = x;

    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Compute Wiener process eigenfunctions on [0,1].
///
/// The Wiener (Brownian motion) eigenfunctions are:
/// φ_k(t) = √2 sin((k - 0.5)πt)
///
/// These are the eigenfunctions of the covariance kernel K(s,t) = min(s,t).
///
/// # Arguments
/// * `t` - Evaluation points in [0,1]
/// * `m` - Number of eigenfunctions
///
/// # Returns
/// Column-major matrix of size `len(t) × m` as flat vector
pub fn wiener_eigenfunctions(t: &[f64], m: usize) -> Vec<f64> {
    let n = t.len();
    let mut phi = vec![0.0; n * m];
    let sqrt2 = 2.0_f64.sqrt();

    for (i, &ti) in t.iter().enumerate() {
        for j in 0..m {
            let k = (j + 1) as f64;
            // φ_k(t) = sqrt(2) * sin((k - 0.5) * pi * t)
            phi[i + j * n] = sqrt2 * ((k - 0.5) * PI * ti).sin();
        }
    }
    phi
}

/// Unified eigenfunction computation.
///
/// # Arguments
/// * `t` - Evaluation points
/// * `m` - Number of eigenfunctions
/// * `efun_type` - Type of eigenfunction basis
///
/// # Returns
/// Column-major matrix of size `len(t) × m` as flat vector
pub fn eigenfunctions(t: &[f64], m: usize, efun_type: EFunType) -> Vec<f64> {
    match efun_type {
        EFunType::Fourier => fourier_eigenfunctions(t, m),
        EFunType::Poly => legendre_eigenfunctions(t, m, false),
        EFunType::PolyHigh => legendre_eigenfunctions(t, m, true),
        EFunType::Wiener => wiener_eigenfunctions(t, m),
    }
}

// =============================================================================
// Eigenvalue Computation
// =============================================================================

/// Generate eigenvalue sequence with linear decay.
///
/// λ_k = 1/k for k = 1, ..., m
pub fn eigenvalues_linear(m: usize) -> Vec<f64> {
    (1..=m).map(|k| 1.0 / k as f64).collect()
}

/// Generate eigenvalue sequence with exponential decay.
///
/// λ_k = exp(-k) for k = 1, ..., m
pub fn eigenvalues_exponential(m: usize) -> Vec<f64> {
    (1..=m).map(|k| (-1.0 * k as f64).exp()).collect()
}

/// Generate Wiener process eigenvalues.
///
/// λ_k = 1/((k - 0.5)π)² for k = 1, ..., m
///
/// These are the eigenvalues of the covariance kernel K(s,t) = min(s,t).
pub fn eigenvalues_wiener(m: usize) -> Vec<f64> {
    (1..=m)
        .map(|k| {
            let denom = (k as f64 - 0.5) * PI;
            1.0 / (denom * denom)
        })
        .collect()
}

/// Unified eigenvalue computation.
///
/// # Arguments
/// * `m` - Number of eigenvalues
/// * `eval_type` - Type of eigenvalue decay
///
/// # Returns
/// Vector of m eigenvalues in decreasing order
pub fn eigenvalues(m: usize, eval_type: EValType) -> Vec<f64> {
    match eval_type {
        EValType::Linear => eigenvalues_linear(m),
        EValType::Exponential => eigenvalues_exponential(m),
        EValType::Wiener => eigenvalues_wiener(m),
    }
}

// =============================================================================
// Karhunen-Loève Simulation
// =============================================================================

/// Simulate functional data via Karhunen-Loève expansion.
///
/// Generates n curves using the truncated KL representation:
/// f_i(t) = Σ_{k=1}^{M} ξ_{ik} φ_k(t)
/// where ξ_{ik} ~ N(0, λ_k)
///
/// # Arguments
/// * `n` - Number of curves to generate
/// * `phi` - Eigenfunctions matrix (m × M) in column-major format
/// * `m` - Number of evaluation points
/// * `big_m` - Number of eigenfunctions
/// * `lambda` - Eigenvalues (length M)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Data matrix (n × m) in column-major format
pub fn sim_kl(
    n: usize,
    phi: &[f64],
    m: usize,
    big_m: usize,
    lambda: &[f64],
    seed: Option<u64>,
) -> Vec<f64> {
    // Create RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate scores ξ ~ N(0, λ) for all curves
    // xi is n × big_m in column-major format
    let mut xi = vec![0.0; n * big_m];
    for k in 0..big_m {
        let sd = lambda[k].sqrt();
        for i in 0..n {
            xi[i + k * n] = rng.sample::<f64, _>(normal) * sd;
        }
    }

    // Compute data = xi * phi^T
    // xi: n × M, phi: m × M -> data: n × m
    let mut data = vec![0.0; n * m];

    // Parallelize over columns (evaluation points)
    data.par_chunks_mut(n).enumerate().for_each(|(j, col)| {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..big_m {
                // phi[j + k*m] is φ_k(t_j)
                // xi[i + k*n] is ξ_{ik}
                sum += xi[i + k * n] * phi[j + k * m];
            }
            col[i] = sum;
        }
    });

    data
}

/// Simulate functional data with specified eigenfunction and eigenvalue types.
///
/// Convenience function that combines eigenfunction and eigenvalue generation
/// with KL simulation.
///
/// # Arguments
/// * `n` - Number of curves to generate
/// * `t` - Evaluation points
/// * `big_m` - Number of eigenfunctions/eigenvalues to use
/// * `efun_type` - Type of eigenfunction basis
/// * `eval_type` - Type of eigenvalue decay
/// * `seed` - Optional random seed
///
/// # Returns
/// Data matrix (n × len(t)) in column-major format
pub fn sim_fundata(
    n: usize,
    t: &[f64],
    big_m: usize,
    efun_type: EFunType,
    eval_type: EValType,
    seed: Option<u64>,
) -> Vec<f64> {
    let m = t.len();
    let phi = eigenfunctions(t, big_m, efun_type);
    let lambda = eigenvalues(big_m, eval_type);
    sim_kl(n, &phi, m, big_m, &lambda, seed)
}

// =============================================================================
// Noise Addition
// =============================================================================

/// Add pointwise Gaussian noise to functional data.
///
/// Adds independent N(0, σ²) noise to each point.
///
/// # Arguments
/// * `data` - Data matrix (n × m) in column-major format
/// * `nrow` - Number of rows (observations)
/// * `ncol` - Number of columns (evaluation points)
/// * `sd` - Standard deviation of noise
/// * `seed` - Optional random seed
///
/// # Returns
/// Noisy data matrix (n × m) in column-major format
pub fn add_error_pointwise(
    data: &[f64],
    _nrow: usize,
    _ncol: usize,
    sd: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, sd).unwrap();

    data.iter()
        .map(|&x| x + rng.sample::<f64, _>(normal))
        .collect()
}

/// Add curve-level Gaussian noise to functional data.
///
/// Adds a constant noise term per curve: each observation in curve i
/// has the same noise value.
///
/// # Arguments
/// * `data` - Data matrix (n × m) in column-major format
/// * `nrow` - Number of rows (observations)
/// * `ncol` - Number of columns (evaluation points)
/// * `sd` - Standard deviation of noise
/// * `seed` - Optional random seed
///
/// # Returns
/// Noisy data matrix (n × m) in column-major format
pub fn add_error_curve(
    data: &[f64],
    nrow: usize,
    ncol: usize,
    sd: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, sd).unwrap();

    // Generate one noise value per curve
    let curve_noise: Vec<f64> = (0..nrow).map(|_| rng.sample::<f64, _>(normal)).collect();

    // Add to data
    let mut result = data.to_vec();
    for j in 0..ncol {
        for i in 0..nrow {
            result[i + j * nrow] += curve_noise[i];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourier_eigenfunctions_dimensions() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let phi = fourier_eigenfunctions(&t, 5);
        assert_eq!(phi.len(), 100 * 5);
    }

    #[test]
    fn test_fourier_eigenfunctions_first_is_constant() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let phi = fourier_eigenfunctions(&t, 3);

        // First eigenfunction should be constant 1
        for i in 0..100 {
            assert!((phi[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eigenvalues_linear() {
        let lambda = eigenvalues_linear(5);
        assert_eq!(lambda.len(), 5);
        assert!((lambda[0] - 1.0).abs() < 1e-10);
        assert!((lambda[1] - 0.5).abs() < 1e-10);
        assert!((lambda[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_exponential() {
        let lambda = eigenvalues_exponential(3);
        assert_eq!(lambda.len(), 3);
        assert!((lambda[0] - (-1.0_f64).exp()).abs() < 1e-10);
        assert!((lambda[1] - (-2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_sim_kl_dimensions() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let phi = fourier_eigenfunctions(&t, 5);
        let lambda = eigenvalues_linear(5);

        let data = sim_kl(10, &phi, 50, 5, &lambda, Some(42));
        assert_eq!(data.len(), 10 * 50);
    }

    #[test]
    fn test_sim_fundata_dimensions() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let data = sim_fundata(20, &t, 5, EFunType::Fourier, EValType::Linear, Some(42));
        assert_eq!(data.len(), 20 * 100);
    }

    #[test]
    fn test_add_error_pointwise() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3 matrix
        let noisy = add_error_pointwise(&data, 2, 3, 0.1, Some(42));
        assert_eq!(noisy.len(), 6);
        // Check that values changed but not by too much
        for i in 0..6 {
            assert!((noisy[i] - data[i]).abs() < 1.0);
        }
    }

    #[test]
    fn test_legendre_orthonormality() {
        // Test that Legendre eigenfunctions are approximately orthonormal
        let n = 1000;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let m = 5;
        let phi = legendre_eigenfunctions(&t, m, false);
        let dt = 1.0 / (n - 1) as f64;

        // Check orthonormality
        for j1 in 0..m {
            for j2 in 0..m {
                let mut integral = 0.0;
                for i in 0..n {
                    integral += phi[i + j1 * n] * phi[i + j2 * n] * dt;
                }
                let expected = if j1 == j2 { 1.0 } else { 0.0 };
                assert!(
                    (integral - expected).abs() < 0.05,
                    "Orthonormality check failed for ({}, {}): {} vs {}",
                    j1,
                    j2,
                    integral,
                    expected
                );
            }
        }
    }
}
