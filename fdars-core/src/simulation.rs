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
//! - **Legendre**: Orthonormal Legendre polynomials on \[0,1\]
//! - **Wiener**: Eigenfunctions of the Wiener process

use crate::matrix::FdMatrix;
use crate::maybe_par_chunks_mut_enumerate;
use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;

/// Eigenfunction type enum for simulation
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum EFunType {
    /// Fourier basis: 1, sqrt(2)*cos(2πkt), sqrt(2)*sin(2πkt)
    Fourier = 0,
    /// Orthonormal Legendre polynomials on \[0,1\]
    Poly = 1,
    /// Higher-order Legendre polynomials (starting at degree 2)
    PolyHigh = 2,
    /// Wiener process eigenfunctions: sqrt(2)*sin((k-0.5)πt)
    Wiener = 3,
}

impl EFunType {
    /// Create from integer (for FFI)
    pub fn from_i32(value: i32) -> Result<Self, crate::FdarError> {
        match value {
            0 => Ok(EFunType::Fourier),
            1 => Ok(EFunType::Poly),
            2 => Ok(EFunType::PolyHigh),
            3 => Ok(EFunType::Wiener),
            _ => Err(crate::FdarError::InvalidEnumValue {
                enum_name: "EFunType",
                value,
            }),
        }
    }
}

/// Eigenvalue decay type for simulation
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
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
    pub fn from_i32(value: i32) -> Result<Self, crate::FdarError> {
        match value {
            0 => Ok(EValType::Linear),
            1 => Ok(EValType::Exponential),
            2 => Ok(EValType::Wiener),
            _ => Err(crate::FdarError::InvalidEnumValue {
                enum_name: "EValType",
                value,
            }),
        }
    }
}

// =============================================================================
// Eigenfunction Computation
// =============================================================================

/// Compute Fourier eigenfunctions on \[0,1\].
///
/// The Fourier basis consists of:
/// - φ_1(t) = 1
/// - φ_{2k}(t) = √2 cos(2πkt) for k = 1, 2, ...
/// - φ_{2k+1}(t) = √2 sin(2πkt) for k = 1, 2, ...
///
/// # Arguments
/// * `t` - Evaluation points in \[0,1\]
/// * `m` - Number of eigenfunctions
///
/// # Returns
/// `FdMatrix` of size `len(t) × m`
pub fn fourier_eigenfunctions(t: &[f64], m: usize) -> FdMatrix {
    let n = t.len();
    let mut phi = FdMatrix::zeros(n, m);
    let sqrt2 = 2.0_f64.sqrt();

    for (i, &ti) in t.iter().enumerate() {
        // φ_1(t) = 1
        phi[(i, 0)] = 1.0;

        let mut k = 1; // current eigenfunction index
        let mut freq = 1; // frequency index

        while k < m {
            // sin term: sqrt(2) * sin(2*pi*freq*t)
            if k < m {
                phi[(i, k)] = sqrt2 * (2.0 * PI * f64::from(freq) * ti).sin();
                k += 1;
            }
            // cos term: sqrt(2) * cos(2*pi*freq*t)
            if k < m {
                phi[(i, k)] = sqrt2 * (2.0 * PI * f64::from(freq) * ti).cos();
                k += 1;
            }
            freq += 1;
        }
    }
    phi
}

/// Compute Legendre polynomial eigenfunctions on \[0,1\].
///
/// Uses orthonormalized Legendre polynomials. The normalization factor is
/// √(2n+1) where n is the polynomial degree, which ensures unit L² norm on \[0,1\].
///
/// # Arguments
/// * `t` - Evaluation points in \[0,1\]
/// * `m` - Number of eigenfunctions
/// * `high` - If true, start at degree 2 (PolyHigh), otherwise start at degree 0
///
/// # Returns
/// `FdMatrix` of size `len(t) × m`
pub fn legendre_eigenfunctions(t: &[f64], m: usize, high: bool) -> FdMatrix {
    let n = t.len();
    let mut phi = FdMatrix::zeros(n, m);
    let start_deg = if high { 2 } else { 0 };

    for (i, &ti) in t.iter().enumerate() {
        // Transform from \[0,1\] to \[-1,1\]
        let x = 2.0 * ti - 1.0;

        for j in 0..m {
            let deg = start_deg + j;
            // Compute Legendre polynomial P_deg(x)
            let p = legendre_p(x, deg);
            // Normalize: ||P_n||² on \[-1,1\] = 2/(2n+1), on \[0,1\] = 1/(2n+1)
            let norm = ((2 * deg + 1) as f64).sqrt();
            phi[(i, j)] = p * norm;
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

/// Compute Wiener process eigenfunctions on \[0,1\].
///
/// The Wiener (Brownian motion) eigenfunctions are:
/// φ_k(t) = √2 sin((k - 0.5)πt)
///
/// These are the eigenfunctions of the covariance kernel K(s,t) = min(s,t).
///
/// # Arguments
/// * `t` - Evaluation points in \[0,1\]
/// * `m` - Number of eigenfunctions
///
/// # Returns
/// `FdMatrix` of size `len(t) × m`
pub fn wiener_eigenfunctions(t: &[f64], m: usize) -> FdMatrix {
    let n = t.len();
    let mut phi = FdMatrix::zeros(n, m);
    let sqrt2 = 2.0_f64.sqrt();

    for (i, &ti) in t.iter().enumerate() {
        for j in 0..m {
            let k = (j + 1) as f64;
            // φ_k(t) = sqrt(2) * sin((k - 0.5) * pi * t)
            phi[(i, j)] = sqrt2 * ((k - 0.5) * PI * ti).sin();
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
/// `FdMatrix` of size `len(t) × m`
pub fn eigenfunctions(t: &[f64], m: usize, efun_type: EFunType) -> FdMatrix {
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
    (1..=m).map(|k| (-(k as f64)).exp()).collect()
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
/// * `phi` - Eigenfunctions matrix (m × big_m) as `FdMatrix`
/// * `big_m` - Number of eigenfunctions
/// * `lambda` - Eigenvalues (length big_m)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Data `FdMatrix` of size `n × m`
pub fn sim_kl(
    n: usize,
    phi: &FdMatrix,
    big_m: usize,
    lambda: &[f64],
    seed: Option<u64>,
) -> FdMatrix {
    let m = phi.nrows();

    // Create RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, 1.0).expect("valid distribution parameters");

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
    // xi: n × big_m, phi: m × big_m -> data: n × m
    let mut data = vec![0.0; n * m];

    // Parallelize over columns (evaluation points)
    maybe_par_chunks_mut_enumerate!(data, n, |(j, col)| {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..big_m {
                // phi[(j, k)] is φ_k(t_j)
                // xi[i + k*n] is ξ_{ik}
                sum += xi[i + k * n] * phi[(j, k)];
            }
            col[i] = sum;
        }
    });

    FdMatrix::from_column_major(data, n, m).expect("dimension invariant: data.len() == n * m")
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
/// Data `FdMatrix` of size `n × len(t)`
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
///
/// let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = sim_fundata(5, &t, 4, EFunType::Fourier, EValType::Linear, Some(42));
/// assert_eq!(data.shape(), (5, 20));
/// assert!(data.as_slice().iter().all(|v| v.is_finite()));
/// ```
pub fn sim_fundata(
    n: usize,
    t: &[f64],
    big_m: usize,
    efun_type: EFunType,
    eval_type: EValType,
    seed: Option<u64>,
) -> FdMatrix {
    let phi = eigenfunctions(t, big_m, efun_type);
    let lambda = eigenvalues(big_m, eval_type);
    sim_kl(n, &phi, big_m, &lambda, seed)
}

// =============================================================================
// Noise Addition
// =============================================================================

/// Add pointwise Gaussian noise to functional data.
///
/// Adds independent N(0, σ²) noise to each point.
///
/// # Arguments
/// * `data` - Data `FdMatrix` (n × m)
/// * `sd` - Standard deviation of noise
/// * `seed` - Optional random seed
///
/// # Returns
/// Noisy data `FdMatrix` (n × m)
pub fn add_error_pointwise(data: &FdMatrix, sd: f64, seed: Option<u64>) -> FdMatrix {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, sd).expect("valid distribution parameters: sd > 0");

    let noisy: Vec<f64> = data
        .as_slice()
        .iter()
        .map(|&x| x + rng.sample::<f64, _>(normal))
        .collect();

    FdMatrix::from_column_major(noisy, data.nrows(), data.ncols())
        .expect("dimension invariant: data.len() == n * m")
}

/// Add curve-level Gaussian noise to functional data.
///
/// Adds a constant noise term per curve: each observation in curve i
/// has the same noise value.
///
/// # Arguments
/// * `data` - Data `FdMatrix` (n × m)
/// * `sd` - Standard deviation of noise
/// * `seed` - Optional random seed
///
/// # Returns
/// Noisy data `FdMatrix` (n × m)
pub fn add_error_curve(data: &FdMatrix, sd: f64, seed: Option<u64>) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(0.0, sd).expect("valid distribution parameters: sd > 0");

    // Generate one noise value per curve
    let curve_noise: Vec<f64> = (0..n).map(|_| rng.sample::<f64, _>(normal)).collect();

    // Add to data
    let mut result = data.as_slice().to_vec();
    for j in 0..m {
        for i in 0..n {
            result[i + j * n] += curve_noise[i];
        }
    }
    FdMatrix::from_column_major(result, n, m).expect("dimension invariant: data.len() == n * m")
}

// ─── Functional VAR/VMA + FARMA simulators (FTS-03-04/05, plan 41-02) ─────────

/// Result of a functional VAR/VMA (`sim_fvarma`) simulation.
///
/// Produced by [`sim_fvarma`]. `curves` is the `N × m` simulated series (rows =
/// curves, columns = grid points).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FvarmaResult {
    /// Simulated curve series, shape `N × m`.
    pub curves: FdMatrix,
    /// Autoregressive order p (= number of AR operator kernels).
    pub ar_order: usize,
    /// Moving-average order q (= number of MA operator kernels).
    pub ma_order: usize,
    /// Number of burn-in curves discarded before the kept output.
    pub burn_in: usize,
}

/// Result of a functional ARMA (`sim_farma`) simulation.
///
/// Produced by [`sim_farma`]. Same fields as [`FvarmaResult`]; the two share the
/// underlying recurrence and are bit-identical for identical inputs.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FarmaResult {
    /// Simulated curve series, shape `N × m`.
    pub curves: FdMatrix,
    /// Autoregressive order p (= number of AR operator kernels).
    pub ar_order: usize,
    /// Moving-average order q (= number of MA operator kernels).
    pub ma_order: usize,
    /// Number of burn-in curves discarded before the kept output.
    pub burn_in: usize,
}

/// Validate the grid and that every operator kernel is a flat m×m matrix.
fn validate_operator_kernels(
    argvals: &[f64],
    ar_ops: &[Vec<f64>],
    ma_ops: &[Vec<f64>],
) -> Result<usize, crate::FdarError> {
    let m = argvals.len();
    if m == 0 {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "argvals",
            expected: "non-empty grid".to_string(),
            actual: "0 elements".to_string(),
        });
    }
    for a in ar_ops {
        if a.len() != m * m {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "ar_ops",
                expected: format!("{} elements per kernel (m*m)", m * m),
                actual: format!("{} elements", a.len()),
            });
        }
    }
    for b in ma_ops {
        if b.len() != m * m {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "ma_ops",
                expected: format!("{} elements per kernel (m*m)", m * m),
                actual: format!("{} elements", b.len()),
            });
        }
    }
    Ok(m)
}

/// Shared functional VAR/VMA/FARMA recurrence used by both public entry points.
///
/// Runs `burn_in + n` steps of `X_t = Σ_k A_k·X_{t-k} + ε_t + Σ_k B_k·ε_{t-k}`
/// with i.i.d. N(0,1) innovations per grid point, discards the first `burn_in`
/// curves, and returns the kept `n × m` series. Returns `ComputationFailed` if
/// any curve entry becomes non-finite (non-stationary operator).
fn fvarma_core(
    n: usize,
    argvals: &[f64],
    ar_ops: &[Vec<f64>],
    ma_ops: &[Vec<f64>],
    burn_in: usize,
    seed: u64,
) -> Result<FdMatrix, crate::FdarError> {
    let m = validate_operator_kernels(argvals, ar_ops, ma_ops)?;
    let p = ar_ops.len();
    let q = ma_ops.len();

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("valid distribution parameters");

    // Ring histories: hist_x[k] = X_{t-1-k}, hist_eps[k] = ε_{t-1-k}.
    let mut hist_x: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut hist_eps: Vec<Vec<f64>> = Vec::with_capacity(q);
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(n);

    let total = burn_in + n;
    for step in 0..total {
        let eps_t: Vec<f64> = (0..m).map(|_| rng.sample::<f64, _>(normal)).collect();
        let mut x_new = eps_t.clone();

        // AR terms: X_t += A_k · X_{t-1-k}.
        for (k, a_k) in ar_ops.iter().enumerate() {
            if let Some(x_prev) = hist_x.get(k) {
                for j1 in 0..m {
                    let mut s = 0.0;
                    for j2 in 0..m {
                        s += a_k[j1 + j2 * m] * x_prev[j2];
                    }
                    x_new[j1] += s;
                }
            }
        }
        // MA terms: X_t += B_k · ε_{t-1-k}.
        for (k, b_k) in ma_ops.iter().enumerate() {
            if let Some(e_prev) = hist_eps.get(k) {
                for j1 in 0..m {
                    let mut s = 0.0;
                    for j2 in 0..m {
                        s += b_k[j1 + j2 * m] * e_prev[j2];
                    }
                    x_new[j1] += s;
                }
            }
        }

        // Divergence guard (Pitfall 5): reject non-finite values instead of emitting them.
        // Fires anywhere in the recurrence (burn-in or kept output), so the message
        // stays phase-agnostic.
        if x_new.iter().any(|v| !v.is_finite()) {
            let phase = if step < burn_in { "burn-in" } else { "output" };
            return Err(crate::FdarError::ComputationFailed {
                operation: "sim_fvarma recurrence",
                detail: format!(
                    "curve values diverged to NaN/Inf during {phase} (step {step}); \
                     ensure AR operators have spectral radius < 1"
                ),
            });
        }

        if p > 0 {
            hist_x.insert(0, x_new.clone());
            if hist_x.len() > p {
                hist_x.pop();
            }
        }
        if q > 0 {
            hist_eps.insert(0, eps_t);
            if hist_eps.len() > q {
                hist_eps.pop();
            }
        }

        if step >= burn_in {
            kept.push(x_new);
        }
    }

    // Assemble N×m column-major FdMatrix: data[i + j*n] = kept[i][j].
    let mut data = vec![0.0f64; n * m];
    for (i, curve) in kept.iter().enumerate() {
        for (j, &v) in curve.iter().enumerate() {
            data[i + j * n] = v;
        }
    }
    FdMatrix::from_column_major(data, n, m).map_err(|_| crate::FdarError::ComputationFailed {
        operation: "sim_fvarma assembly",
        detail: format!("could not assemble {n}×{m} curve matrix"),
    })
}

/// Simulate a functional VAR/VMA process from user-supplied operator kernels.
///
/// Generates `n` curves from the recurrence
/// `X_t = Σ_{k=1}^{p} A_k·X_{t-k} + ε_t + Σ_{k=1}^{q} B_k·ε_{t-k}`,
/// where each `A_k` (AR) and `B_k` (MA) is a flat column-major m×m operator kernel
/// applied by matrix-vector product to the grid-discretized curve, and the
/// innovations `ε_t` are i.i.d. standard-normal per grid point. The first
/// `burn_in` curves are discarded so the kept output is approximately stationary.
///
/// # Arguments
///
/// * `n` — number of output curves.
/// * `argvals` — grid points; `m = argvals.len()`.
/// * `ar_ops` — AR operator kernels, each a flat m×m column-major matrix (`p = ar_ops.len()`).
/// * `ma_ops` — MA operator kernels, each a flat m×m column-major matrix (`q = ma_ops.len()`).
/// * `burn_in` — number of leading curves to discard (200 is a reasonable default
///   for moderate operator norms).
/// * `seed` — RNG seed; output is bit-identical for a fixed seed
///   (`StdRng::seed_from_u64`), with no entropy fallback.
///
/// # Errors
///
/// [`crate::FdarError::InvalidDimension`] if `argvals` is empty or any kernel is
/// not m×m; [`crate::FdarError::ComputationFailed`] if the recurrence diverges to
/// NaN/Inf (a non-stationary operator).
///
/// # Stationarity
///
/// Stationarity (companion-matrix spectral radius < 1; sufficient condition
/// `‖A_1‖_HS < 1` for FAR(1)) is the caller's responsibility — it is not enforced,
/// only guarded against numeric divergence.
///
/// # Divergence from `freqdom`
///
/// Innovations use an identity covariance (i.i.d. N(0,1) per grid point);
/// `freqdom::fts.rar` accepts a user-supplied innovation covariance σ.
#[must_use = "the simulated curve series is the return value and should be used"]
pub fn sim_fvarma(
    n: usize,
    argvals: &[f64],
    ar_ops: &[Vec<f64>],
    ma_ops: &[Vec<f64>],
    burn_in: usize,
    seed: u64,
) -> Result<FvarmaResult, crate::FdarError> {
    let curves = fvarma_core(n, argvals, ar_ops, ma_ops, burn_in, seed)?;
    Ok(FvarmaResult {
        curves,
        ar_order: ar_ops.len(),
        ma_order: ma_ops.len(),
        burn_in,
    })
}

/// Simulate a functional ARMA (FARMA) process combining AR and MA operator terms.
///
/// FARMA is the combined AR+MA case of the functional operator recurrence; this is
/// a thin named entry point over the shared [`sim_fvarma`] recurrence and is
/// bit-identical to `sim_fvarma` for identical inputs. See [`sim_fvarma`] for the
/// recurrence, stationarity responsibility, and R-baseline divergence.
///
/// # Errors
///
/// Same as [`sim_fvarma`].
#[must_use = "the simulated curve series is the return value and should be used"]
pub fn sim_farma(
    n: usize,
    argvals: &[f64],
    ar_ops: &[Vec<f64>],
    ma_ops: &[Vec<f64>],
    burn_in: usize,
    seed: u64,
) -> Result<FarmaResult, crate::FdarError> {
    let curves = fvarma_core(n, argvals, ar_ops, ma_ops, burn_in, seed)?;
    Ok(FarmaResult {
        curves,
        ar_order: ar_ops.len(),
        ma_order: ma_ops.len(),
        burn_in,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FTS-03-04/05 simulator helpers + oracles ────────────────────────────

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// Frobenius norm of the lag-`h` sample autocovariance operator.
    fn lag_autocov_fro(data: &FdMatrix, h: usize) -> f64 {
        let (n, m) = data.shape();
        let mut xbar = vec![0.0f64; m];
        for (j, xb) in xbar.iter_mut().enumerate() {
            let mut s = 0.0;
            for i in 0..n {
                s += data[(i, j)];
            }
            *xb = s / n as f64;
        }
        let mut fro = 0.0;
        for j1 in 0..m {
            for j2 in 0..m {
                let mut c = 0.0;
                for i in 0..(n - h) {
                    c += (data[(i, j1)] - xbar[j1]) * (data[(i + h, j2)] - xbar[j2]);
                }
                c /= n as f64;
                fro += c * c;
            }
        }
        fro.sqrt()
    }

    fn scaled_identity(m: usize, s: f64) -> Vec<f64> {
        let mut a = vec![0.0f64; m * m];
        for j in 0..m {
            a[j + j * m] = s;
        }
        a
    }

    #[test]
    fn fvarma_deterministic() {
        let (n, m) = (30, 8);
        let argvals = uniform_grid(m);
        let ar = vec![scaled_identity(m, 0.3)];
        let a = sim_fvarma(n, &argvals, &ar, &[], 50, 42).unwrap();
        let b = sim_fvarma(n, &argvals, &ar, &[], 50, 42).unwrap();
        assert_eq!(a, b, "same seed must give bit-identical output");
        assert_eq!(a.curves.shape(), (n, m));
        assert!(a.curves.as_slice().iter().all(|x| x.is_finite()));
        assert_eq!(a.ar_order, 1);
        assert_eq!(a.ma_order, 0);
    }

    #[test]
    fn fvarma_zero_op_white_noise() {
        // Oracle 4: zero AR operator ⇒ pure i.i.d. innovations, near-zero lag-1 ACF.
        let (n, m) = (500, 6);
        let argvals = uniform_grid(m);
        let ar = vec![vec![0.0f64; m * m]];
        let res = sim_fvarma(n, &argvals, &ar, &[], 0, 7).unwrap();
        assert!(res.curves.as_slice().iter().all(|x| x.is_finite()));
        let c0 = lag_autocov_fro(&res.curves, 0);
        let c1 = lag_autocov_fro(&res.curves, 1);
        assert!(c1 < 0.15 * c0, "lag-1 ACF too large: c1={c1}, c0={c0}");
    }

    #[test]
    fn fvarma_rank1_dependence() {
        // Oracle 6: rank-1 AR operator ⇒ non-trivial lag-1 serial dependence.
        let (n, m) = (400, 10);
        let argvals = uniform_grid(m);
        let raw: Vec<f64> = argvals
            .iter()
            .map(|&t| (std::f64::consts::PI * t).sin())
            .collect();
        let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let phi: Vec<f64> = raw.iter().map(|x| x / norm).collect();
        let mut ar1 = vec![0.0f64; m * m];
        for j1 in 0..m {
            for j2 in 0..m {
                ar1[j1 + j2 * m] = 0.8 * phi[j1] * phi[j2];
            }
        }
        let res = sim_fvarma(n, &argvals, &[ar1], &[], 200, 3).unwrap();
        let c0 = lag_autocov_fro(&res.curves, 0);
        let c1 = lag_autocov_fro(&res.curves, 1);
        assert!(c1 > 0.1 * c0, "lag-1 dependence too weak: c1={c1}, c0={c0}");
    }

    #[test]
    fn fvarma_dimension_errors() {
        let (n, m) = (20, 5);
        let argvals = uniform_grid(m);
        // AR kernel wrong length.
        assert!(matches!(
            sim_fvarma(n, &argvals, &[vec![0.0; m * m - 1]], &[], 10, 1),
            Err(crate::FdarError::InvalidDimension {
                parameter: "ar_ops",
                ..
            })
        ));
        // MA kernel wrong length.
        assert!(matches!(
            sim_fvarma(n, &argvals, &[], &[vec![0.0; m * m + 2]], 10, 1),
            Err(crate::FdarError::InvalidDimension {
                parameter: "ma_ops",
                ..
            })
        ));
        // Empty grid.
        assert!(matches!(
            sim_fvarma(n, &[], &[], &[], 10, 1),
            Err(crate::FdarError::InvalidDimension {
                parameter: "argvals",
                ..
            })
        ));
    }

    #[test]
    fn fvarma_divergence_guard() {
        // Spectral radius ≥ 1 (2×identity) ⇒ geometric blow-up ⇒ ComputationFailed.
        let (n, m) = (10, 4);
        let argvals = uniform_grid(m);
        let ar = vec![scaled_identity(m, 2.0)];
        assert!(matches!(
            sim_fvarma(n, &argvals, &ar, &[], 2000, 1),
            Err(crate::FdarError::ComputationFailed {
                operation: "sim_fvarma recurrence",
                ..
            })
        ));
    }

    #[test]
    fn farma_shape_and_order() {
        let (n, m) = (40, 5);
        let argvals = uniform_grid(m);
        let ar = vec![scaled_identity(m, 0.3)];
        let ma = vec![scaled_identity(m, 0.2)];
        let res = sim_farma(n, &argvals, &ar, &ma, 50, 9).unwrap();
        assert_eq!(res.curves.shape(), (n, m));
        assert!(res.curves.as_slice().iter().all(|x| x.is_finite()));
        assert_eq!(res.ar_order, 1);
        assert_eq!(res.ma_order, 1);
    }

    #[test]
    fn farma_deterministic() {
        let (n, m) = (25, 6);
        let argvals = uniform_grid(m);
        let ar = vec![scaled_identity(m, 0.4)];
        let ma = vec![scaled_identity(m, 0.25)];
        let a = sim_farma(n, &argvals, &ar, &ma, 40, 11).unwrap();
        let b = sim_farma(n, &argvals, &ar, &ma, 40, 11).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn farma_equals_fvarma() {
        // FARMA = combined AR+MA: identical inputs + seed ⇒ identical curves.
        let (n, m) = (30, 6);
        let argvals = uniform_grid(m);
        let ar = vec![scaled_identity(m, 0.35)];
        let ma = vec![scaled_identity(m, 0.2)];
        let f = sim_fvarma(n, &argvals, &ar, &ma, 60, 77).unwrap();
        let g = sim_farma(n, &argvals, &ar, &ma, 60, 77).unwrap();
        assert_eq!(f.curves, g.curves);
    }

    #[test]
    fn test_fourier_eigenfunctions_dimensions() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let phi = fourier_eigenfunctions(&t, 5);
        assert_eq!(phi.nrows(), 100);
        assert_eq!(phi.ncols(), 5);
        assert_eq!(phi.len(), 100 * 5);
    }

    #[test]
    fn test_fourier_eigenfunctions_first_is_constant() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let phi = fourier_eigenfunctions(&t, 3);

        // First eigenfunction should be constant 1
        for i in 0..100 {
            assert!((phi[(i, 0)] - 1.0).abs() < 1e-10);
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

        let data = sim_kl(10, &phi, 5, &lambda, Some(42));
        assert_eq!(data.nrows(), 10);
        assert_eq!(data.ncols(), 50);
        assert_eq!(data.len(), 10 * 50);
    }

    #[test]
    fn test_sim_fundata_dimensions() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let data = sim_fundata(20, &t, 5, EFunType::Fourier, EValType::Linear, Some(42));
        assert_eq!(data.nrows(), 20);
        assert_eq!(data.ncols(), 100);
        assert_eq!(data.len(), 20 * 100);
    }

    #[test]
    fn test_add_error_pointwise() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3 matrix
        let data = FdMatrix::from_column_major(raw.clone(), 2, 3).unwrap();
        let noisy = add_error_pointwise(&data, 0.1, Some(42));
        assert_eq!(noisy.len(), 6);
        // Check that values changed but not by too much
        let noisy_slice = noisy.as_slice();
        for i in 0..6 {
            assert!((noisy_slice[i] - raw[i]).abs() < 1.0);
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
                    integral += phi[(i, j1)] * phi[(i, j2)] * dt;
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

    // ========================================================================
    // Wiener eigenfunction tests
    // ========================================================================

    #[test]
    fn test_wiener_eigenfunctions_dimensions() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let phi = wiener_eigenfunctions(&t, 7);
        assert_eq!(phi.nrows(), 100);
        assert_eq!(phi.ncols(), 7);
        assert_eq!(phi.len(), 100 * 7);
    }

    #[test]
    fn test_wiener_eigenfunctions_orthonormality() {
        // Wiener eigenfunctions: sqrt(2)*sin((k-0.5)*pi*t)
        // Should be orthonormal on [0,1]
        let n = 1000;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let m = 5;
        let phi = wiener_eigenfunctions(&t, m);
        let dt = 1.0 / (n - 1) as f64;

        for j1 in 0..m {
            for j2 in 0..m {
                let mut integral = 0.0;
                for i in 0..n {
                    integral += phi[(i, j1)] * phi[(i, j2)] * dt;
                }
                let expected = if j1 == j2 { 1.0 } else { 0.0 };
                assert!(
                    (integral - expected).abs() < 0.05,
                    "Wiener orthonormality failed for ({}, {}): {} vs {}",
                    j1,
                    j2,
                    integral,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_wiener_eigenfunctions_analytical_form() {
        // φ_k(t) = sqrt(2) * sin((k - 0.5) * pi * t)
        let t = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let phi = wiener_eigenfunctions(&t, 2);
        let sqrt2 = 2.0_f64.sqrt();

        // First eigenfunction: k=1, freq = 0.5*pi
        for (i, &ti) in t.iter().enumerate() {
            let expected = sqrt2 * (0.5 * PI * ti).sin();
            assert!(
                (phi[(i, 0)] - expected).abs() < 1e-10,
                "k=1 at t={}: got {} expected {}",
                ti,
                phi[(i, 0)],
                expected
            );
        }

        // Second eigenfunction: k=2, freq = 1.5*pi
        for (i, &ti) in t.iter().enumerate() {
            let expected = sqrt2 * (1.5 * PI * ti).sin();
            assert!(
                (phi[(i, 1)] - expected).abs() < 1e-10,
                "k=2 at t={}: got {} expected {}",
                ti,
                phi[(i, 1)],
                expected
            );
        }
    }

    // ========================================================================
    // Wiener eigenvalue tests
    // ========================================================================

    #[test]
    fn test_eigenvalues_wiener_decay_rate() {
        // λ_k = 1/((k - 0.5)*pi)^2
        let lambda = eigenvalues_wiener(5);
        assert_eq!(lambda.len(), 5);

        for k in 1..=5 {
            let denom = (k as f64 - 0.5) * PI;
            let expected = 1.0 / (denom * denom);
            assert!(
                (lambda[k - 1] - expected).abs() < 1e-12,
                "Wiener eigenvalue k={}: got {} expected {}",
                k,
                lambda[k - 1],
                expected
            );
        }
    }

    #[test]
    fn test_eigenvalues_wiener_decreasing() {
        // Wiener eigenvalues should decrease monotonically
        let lambda = eigenvalues_wiener(10);

        for i in 1..lambda.len() {
            assert!(
                lambda[i] < lambda[i - 1],
                "Eigenvalues not decreasing at {}: {} >= {}",
                i,
                lambda[i],
                lambda[i - 1]
            );
        }
    }

    // ========================================================================
    // add_error_curve tests
    // ========================================================================

    #[test]
    fn test_add_error_curve_properties() {
        // Curve-level noise: each observation in same curve gets same noise
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 curves x 3 points
        let n = 2;
        let data = FdMatrix::from_column_major(raw.clone(), n, 3).unwrap();
        let noisy = add_error_curve(&data, 0.5, Some(42));

        assert_eq!(noisy.len(), 6);

        // Compute the difference for curve 0 at each point
        let diff0_j0 = noisy[(0, 0)] - raw[0]; // curve 0, point 0
        let diff0_j1 = noisy[(0, 1)] - raw[n]; // curve 0, point 1
        let diff0_j2 = noisy[(0, 2)] - raw[2 * n]; // curve 0, point 2

        // All differences for same curve should be equal (same noise added)
        assert!(
            (diff0_j0 - diff0_j1).abs() < 1e-10,
            "Curve 0 noise differs: {} vs {}",
            diff0_j0,
            diff0_j1
        );
        assert!(
            (diff0_j0 - diff0_j2).abs() < 1e-10,
            "Curve 0 noise differs: {} vs {}",
            diff0_j0,
            diff0_j2
        );

        // Curve 1 should have different noise
        let diff1_j0 = noisy[(1, 0)] - raw[1];
        // Different curves should (with high probability) have different noise
        // We can't guarantee this, but with seed=42 they should differ
        assert!(
            (diff0_j0 - diff1_j0).abs() > 1e-10,
            "Different curves got same noise"
        );
    }

    #[test]
    fn test_add_error_curve_reproducibility() {
        let raw = vec![1.0, 2.0, 3.0, 4.0];
        let data = FdMatrix::from_column_major(raw, 2, 2).unwrap();
        let noisy1 = add_error_curve(&data, 1.0, Some(123));
        let noisy2 = add_error_curve(&data, 1.0, Some(123));

        let s1 = noisy1.as_slice();
        let s2 = noisy2.as_slice();
        for i in 0..4 {
            assert!(
                (s1[i] - s2[i]).abs() < 1e-10,
                "Reproducibility failed at {}: {} vs {}",
                i,
                s1[i],
                s2[i]
            );
        }
    }

    // ========================================================================
    // Enum dispatcher tests
    // ========================================================================

    #[test]
    fn test_efun_type_from_i32() {
        assert_eq!(EFunType::from_i32(0), Ok(EFunType::Fourier));
        assert_eq!(EFunType::from_i32(1), Ok(EFunType::Poly));
        assert_eq!(EFunType::from_i32(2), Ok(EFunType::PolyHigh));
        assert_eq!(EFunType::from_i32(3), Ok(EFunType::Wiener));
        assert!(EFunType::from_i32(-1).is_err());
        assert!(EFunType::from_i32(4).is_err());
        assert!(EFunType::from_i32(100).is_err());
    }

    #[test]
    fn test_eval_type_from_i32() {
        assert_eq!(EValType::from_i32(0), Ok(EValType::Linear));
        assert_eq!(EValType::from_i32(1), Ok(EValType::Exponential));
        assert_eq!(EValType::from_i32(2), Ok(EValType::Wiener));
        assert!(EValType::from_i32(-1).is_err());
        assert!(EValType::from_i32(3).is_err());
        assert!(EValType::from_i32(99).is_err());
    }

    #[test]
    fn test_eigenfunctions_dispatcher() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let m = 4;

        // Test that dispatcher returns correct results for each type
        let phi_fourier = eigenfunctions(&t, m, EFunType::Fourier);
        let phi_fourier_direct = fourier_eigenfunctions(&t, m);
        assert_eq!(phi_fourier, phi_fourier_direct);

        let phi_poly = eigenfunctions(&t, m, EFunType::Poly);
        let phi_poly_direct = legendre_eigenfunctions(&t, m, false);
        assert_eq!(phi_poly, phi_poly_direct);

        let phi_poly_high = eigenfunctions(&t, m, EFunType::PolyHigh);
        let phi_poly_high_direct = legendre_eigenfunctions(&t, m, true);
        assert_eq!(phi_poly_high, phi_poly_high_direct);

        let phi_wiener = eigenfunctions(&t, m, EFunType::Wiener);
        let phi_wiener_direct = wiener_eigenfunctions(&t, m);
        assert_eq!(phi_wiener, phi_wiener_direct);
    }

    #[test]
    fn test_sigma_zero_error() {
        let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        let data = sim_fundata(5, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
        let noisy = add_error_pointwise(&data, 0.0, Some(42));
        // sigma=0 → no noise added, should be identical
        for i in 0..5 {
            for j in 0..20 {
                assert!(
                    (noisy[(i, j)] - data[(i, j)]).abs() < 1e-12,
                    "Zero-sigma error should not change data"
                );
            }
        }
    }

    #[test]
    fn test_ncomp1_eigenfunctions() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let phi = fourier_eigenfunctions(&t, 1);
        assert_eq!(phi.nrows(), t.len());
        assert_eq!(phi.ncols(), 1);
        // First Fourier eigenfunction should be constant
        let first_val = phi[(0, 0)];
        for i in 1..t.len() {
            assert!((phi[(i, 0)] - first_val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deterministic_seed() {
        let t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
        let d1 = sim_fundata(10, &t, 3, EFunType::Fourier, EValType::Linear, Some(123));
        let d2 = sim_fundata(10, &t, 3, EFunType::Fourier, EValType::Linear, Some(123));
        for i in 0..10 {
            for j in 0..30 {
                assert!(
                    (d1[(i, j)] - d2[(i, j)]).abs() < 1e-12,
                    "Same seed should produce identical results"
                );
            }
        }
    }
}
