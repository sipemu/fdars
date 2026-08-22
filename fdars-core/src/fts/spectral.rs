//! Frequency-domain functional time series: spectral density operator and dynamic FPCA.
//!
//! This module adds spectral (frequency-domain) analysis for functional time
//! series, building on the lagged-autocovariance machinery in [`super::acf`]:
//!
//! * [`spectral_density`] (FTS-03-01) — the spectral density operator, the
//!   frequency-domain long-run covariance formed by a Bartlett-weighted DFT
//!   (via `rustfft`) over the lagged autocovariance operators, evaluated at the
//!   Fourier frequencies `θ_j = 2πj/N`.
//! * [`dpca`] (FTS-03-02) — dynamic functional PCA: dynamic eigen-filters and
//!   dynamic scores obtained by eigendecomposing the spectral density operator
//!   per frequency and inverse-FFT-ing the eigenvectors into time-domain filter
//!   taps over a symmetric lag window `[-L, L]`.
//! * [`dpca_reconstruct`] (FTS-03-03) — curve reconstruction from dynamic scores
//!   via inverse dynamic filtering, with an integrated-L2 reconstruction error
//!   that is monotone non-increasing in the number of retained components.
//!
//! # R baseline
//!
//! Matches `freqdom` / `freqdom.fda` (Hörmann, Kidziński, Hallin 2015, *JRSS-B*)
//! by capability. Documented divergences: the `1/2π` pre-factor is omitted
//! (consistent with [`super::long_run_covariance`]); eigendecomposition uses
//! `Re(f̂(θ))` via [`nalgebra::SymmetricEigen`] rather than a complex Hermitian
//! path (nalgebra 0.33 has none without `faer`) — exact for the leading dynamic
//! subspace of a real-lag-window (Bartlett) estimator; dynamic scores are
//! trimmed to the valid interior `t ∈ [L, N-1-L]` (`valid_range`) rather than
//! zero-padded. The estimator, scores, and reconstruction all use the same
//! Simpson-weighted L2 inner product so the DPCA projection is self-consistent.
//!
//! # Conventions
//!
//! All entry points take explicit `argvals`, return `Result<_, FdarError>`, and
//! validate inputs at entry. No randomness — outputs are a deterministic function
//! of the input series.

use super::{DpcaReconstruction, DpcaResult, SpectralDensityResult};
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ─── Input validation (mirrors fts/acf.rs; forecast.rs re-implements it too) ──

/// Validate that `data` is non-empty and `argvals` length matches data columns.
/// Returns `(n, m)` on success.
fn validate_fts_input(data: &FdMatrix, argvals: &[f64]) -> Result<(usize, usize), FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    Ok((n, m))
}

/// Sample mean curve `xbar[j] = (1/n) Σ_i data[(i,j)]`.
fn mean_curve(data: &FdMatrix, n: usize, m: usize) -> Vec<f64> {
    let mut xbar = vec![0.0f64; m];
    let inv_n = 1.0 / n as f64;
    for (j, xb) in xbar.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..n {
            s += data[(i, j)];
        }
        *xb = s * inv_n;
    }
    xbar
}

/// Bartlett lag-window weight `w_h = 1 - h/b`.
#[inline]
fn bartlett_weight(h: usize, bandwidth: usize) -> f64 {
    1.0 - (h as f64) / (bandwidth as f64)
}

// ─── FTS-03-01: spectral density operator ─────────────────────────────────────

/// Estimate the spectral density operator of a functional time series.
///
/// Computes the frequency-domain long-run covariance: for each entry `(j1, j2)`
/// of the lag-`h` autocovariance operator sequence `{C_h}`, a Bartlett-weighted
/// DFT across the lag index (via `rustfft`) yields the complex operator value at
/// every Fourier frequency `θ_k = 2πk/N`. Negative lags use the stationarity
/// identity `C_{-h}[j1,j2] = C_h[j2,j1]`, so the result is Hermitian.
///
/// # Arguments
///
/// * `data` — `N × m` functional time series (rows = curves, columns = grid).
/// * `argvals` — grid points, length `m`.
/// * `bandwidth` — Bartlett lag-window bandwidth; `None` uses `⌊N^{1/3}⌋`
///   (matching [`super::long_run_covariance`]). `Some(0)` is rejected.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] on empty data or `argvals` length mismatch;
/// [`FdarError::InvalidParameter`] if `bandwidth == Some(0)`.
///
/// # Divergence
///
/// The `1/2π` pre-factor is omitted; the frequency grid is the DFT grid
/// `θ_k = 2πk/N` for `k = 0..N`.
#[must_use = "the estimated spectral density operator is the return value and should be used"]
pub fn spectral_density(
    data: &FdMatrix,
    argvals: &[f64],
    bandwidth: Option<usize>,
) -> Result<SpectralDensityResult, FdarError> {
    let (n, m) = validate_fts_input(data, argvals)?;

    let resolved_bandwidth = match bandwidth {
        None => (n as f64).cbrt().floor().max(1.0) as usize,
        Some(0) => {
            return Err(FdarError::InvalidParameter {
                parameter: "bandwidth",
                message: "must be >= 1".to_string(),
            });
        }
        Some(b) => b,
    };
    // Guard against usize underflow in autocovariance_matrix (needs h < n).
    let max_h = resolved_bandwidth.min(n - 1);

    let xbar = mean_curve(data, n, m);
    // Lag operators C_0..=max_h (each flat column-major m×m).
    let mut lag_ops: Vec<Vec<f64>> = Vec::with_capacity(max_h + 1);
    for h in 0..=max_h {
        lag_ops.push(super::acf::autocovariance_matrix(data, &xbar, h, n, m));
    }

    let n_freq = n;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_freq);

    let mut re = vec![vec![0.0f64; m * m]; n_freq];
    let mut im = vec![vec![0.0f64; m * m]; n_freq];

    for j1 in 0..m {
        for j2 in 0..m {
            let mut buf = vec![Complex::new(0.0, 0.0); n_freq];
            // Positive lags h at circular index h: w_h * C_h[j1, j2].
            for (h, c_h) in lag_ops.iter().enumerate() {
                let w_h = bartlett_weight(h, resolved_bandwidth);
                buf[h] += Complex::new(w_h * c_h[j1 + j2 * m], 0.0);
            }
            // Negative lags h at circular index N-h: w_h * C_{-h}[j1,j2] = w_h * C_h[j2,j1].
            for (h, c_h) in lag_ops.iter().enumerate().skip(1) {
                let w_h = bartlett_weight(h, resolved_bandwidth);
                buf[n_freq - h] += Complex::new(w_h * c_h[j2 + j1 * m], 0.0);
            }
            fft.process(&mut buf);
            for k in 0..n_freq {
                re[k][j1 + j2 * m] = buf[k].re;
                im[k][j1 + j2 * m] = buf[k].im;
            }
        }
    }

    let freqs: Vec<f64> = (0..n_freq)
        .map(|k| 2.0 * std::f64::consts::PI * (k as f64) / (n as f64))
        .collect();

    Ok(SpectralDensityResult {
        freqs,
        re,
        im,
        m,
        n_curves: n,
        bandwidth: resolved_bandwidth,
    })
}

// ─── Eigendecomposition helper (real, Simpson-metric-scaled) ──────────────────

/// Eigendecompose the metric-scaled real spectral operator `A = W^{1/2} Re(f̂) W^{1/2}`
/// (`W` = Simpson weights). Returns the top-`ncomp` eigenvalues (descending) and
/// the corresponding **scaled** eigenvectors `ψ_c` (Euclidean-orthonormal),
/// sign-aligned so each vector's largest-magnitude entry is positive. The caller
/// recovers physical filters via `φ_c[j] = ψ_c[j] / sqrt(w[j])`.
fn eigen_at_frequency(
    spec_real: &[f64],
    m: usize,
    ncomp: usize,
    sqrt_w: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Build W^{1/2} Re(f̂) W^{1/2}, symmetrised defensively.
    let mut scaled = vec![0.0f64; m * m];
    for j1 in 0..m {
        for j2 in 0..m {
            scaled[j1 + j2 * m] = spec_real[j1 + j2 * m] * sqrt_w[j1] * sqrt_w[j2];
        }
    }
    let mut mat = DMatrix::from_column_slice(m, m, &scaled);
    for j1 in 0..m {
        for j2 in (j1 + 1)..m {
            let avg = 0.5 * (mat[(j1, j2)] + mat[(j2, j1)]);
            mat[(j1, j2)] = avg;
            mat[(j2, j1)] = avg;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(mat);
    let mut pairs: Vec<(f64, Vec<f64>)> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .map(|(&val, col)| (val, col.iter().copied().collect()))
        .collect();
    // Sort by eigenvalue descending.
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Sign-align: make the largest-magnitude entry positive (Pitfall 2).
    for (_, evec) in &mut pairs {
        let max_abs = evec.iter().copied().fold(0.0f64, |acc, x| acc.max(x.abs()));
        let sign = evec
            .iter()
            .copied()
            .find(|x| x.abs() == max_abs)
            .map_or(1.0, |x| if x < 0.0 { -1.0 } else { 1.0 });
        if sign < 0.0 {
            evec.iter_mut().for_each(|x| *x = -*x);
        }
    }
    let eigenvalues: Vec<f64> = pairs.iter().take(ncomp).map(|(v, _)| *v).collect();
    let eigenvectors: Vec<Vec<f64>> = pairs.into_iter().take(ncomp).map(|(_, e)| e).collect();
    (eigenvalues, eigenvectors)
}

// ─── FTS-03-02: dynamic functional PCA ────────────────────────────────────────

/// Compute dynamic functional PCA (DPCA) from the spectral density operator.
///
/// Eigendecomposes the (Simpson-metric-scaled) spectral density operator at each
/// Fourier frequency, sign-aligns the eigenvectors across frequencies, and
/// inverse-FFTs each eigenvector trajectory into real time-domain filter taps
/// over the symmetric lag window `[-L, L]`. Dynamic scores are the Simpson-weighted
/// time-domain convolution of the curve series with the filters, over the valid
/// interior `t ∈ [L, N-1-L]`.
///
/// # Arguments
///
/// * `data` — `N × m` functional time series.
/// * `argvals` — grid points, length `m`.
/// * `ncomp` — number of dynamic components (`1..=m`).
/// * `bandwidth` — Bartlett bandwidth forwarded to [`spectral_density`].
/// * `filter_lag` — symmetric filter half-width `L`; `None` uses the resolved
///   bandwidth. Must satisfy `L < N/2`.
///
/// # Errors
///
/// [`FdarError::InvalidParameter`] if `ncomp` is not in `1..=m` or `filter_lag`
/// is `>= N/2`; propagates [`spectral_density`] validation errors.
#[must_use = "the DPCA filters and scores are the return value and should be used"]
pub fn dpca(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    bandwidth: Option<usize>,
    filter_lag: Option<usize>,
) -> Result<DpcaResult, FdarError> {
    let (n, m) = validate_fts_input(data, argvals)?;
    if ncomp == 0 || ncomp > m {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("must be in 1..={m}"),
        });
    }

    let sd = spectral_density(data, argvals, bandwidth)?;
    let l = filter_lag.unwrap_or(sd.bandwidth);
    if l >= n / 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "filter_lag",
            message: format!("must be < N/2 = {}", n / 2),
        });
    }

    let weights = simpsons_weights(argvals);
    let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let n_freq = sd.n_curves;

    // Eigendecompose per frequency; collect scaled eigenvectors ψ_c(θ_k).
    let mut eigenvalues = vec![vec![0.0f64; n_freq]; ncomp];
    // freq_vecs[k][c] = ψ_c(θ_k) (length m, scaled/Euclidean-orthonormal).
    let mut freq_vecs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_freq);
    for k in 0..n_freq {
        let (vals, vecs) = eigen_at_frequency(&sd.re[k], m, ncomp, &sqrt_w);
        for c in 0..ncomp {
            eigenvalues[c][k] = vals[c].max(0.0); // clip negative finite-sample eigenvalues
        }
        freq_vecs.push(vecs);
    }

    // Inverse-FFT each ψ_c(·)[j] trajectory → physical filter taps φ_c[j,l] = ψ_c[j,l]/sqrt(w[j]).
    let mut inv_planner = FftPlanner::<f64>::new();
    let ifft = inv_planner.plan_fft_inverse(n_freq);
    let inv_n = 1.0 / (n_freq as f64);
    let n_rows = 2 * l + 1;
    let mut filters: Vec<FdMatrix> = Vec::with_capacity(ncomp);
    for c in 0..ncomp {
        let mut filt = vec![0.0f64; n_rows * m]; // column-major (2L+1) × m
        for j in 0..m {
            let mut buf: Vec<Complex<f64>> = (0..n_freq)
                .map(|k| Complex::new(freq_vecs[k][c][j], 0.0))
                .collect();
            ifft.process(&mut buf);
            let inv_sw = 1.0 / sqrt_w[j];
            for lag in 0..=l {
                let tap = buf[lag].re * inv_n * inv_sw;
                let row_pos = l + lag; // lag +lag
                let row_neg = l - lag; // lag -lag (symmetric filter)
                filt[row_pos + j * n_rows] = tap;
                filt[row_neg + j * n_rows] = tap;
            }
        }
        filters.push(
            FdMatrix::from_column_major(filt, n_rows, m)
                .expect("dimension invariant: filt.len() == (2L+1) * m"),
        );
    }

    // Dynamic scores: Simpson-weighted convolution over the interior t ∈ [L, N-1-L].
    let n_interior = n - 2 * l;
    let mut scores_flat = vec![0.0f64; n_interior * ncomp];
    for (c, filt) in filters.iter().enumerate() {
        for t in l..=(n - 1 - l) {
            let mut s = 0.0;
            for lag_idx in 0..n_rows {
                let lag = lag_idx as isize - l as isize;
                let ct = (t as isize + lag) as usize;
                for j in 0..m {
                    s += filt[(lag_idx, j)] * data[(ct, j)] * weights[j];
                }
            }
            scores_flat[(t - l) + c * n_interior] = s;
        }
    }
    let scores = FdMatrix::from_column_major(scores_flat, n_interior, ncomp)
        .expect("dimension invariant: scores.len() == (N-2L) * ncomp");

    Ok(DpcaResult {
        filters,
        scores,
        eigenvalues,
        n_freqs: n_freq,
        filter_lag: l,
        ncomp,
        valid_range: (l, n - 1 - l),
    })
}

// ─── FTS-03-03: reconstruction from dynamic scores ────────────────────────────

/// Reconstruct curves from DPCA dynamic scores via inverse dynamic filtering.
///
/// For each retained-component count `K = 1..=ncomp`, reconstructs the interior
/// series `X̂_K[t,j] = Σ_{c<K} Σ_l filters[c][l,j] · scores[t'+l, c]` and records
/// the integrated-L2 error over the fully-defined interior. The error is monotone
/// non-increasing in `K` (optimal DPCA projection). `fitted` holds the full-`ncomp`
/// reconstruction over the DPCA interior.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] if `argvals`/`data` shapes are inconsistent with
/// `dpca` (grid mismatch or score-length mismatch).
#[must_use = "the reconstruction and its per-component error are the return value"]
pub fn dpca_reconstruct(
    data: &FdMatrix,
    argvals: &[f64],
    dpca: &DpcaResult,
) -> Result<DpcaReconstruction, FdarError> {
    let (n, m) = validate_fts_input(data, argvals)?;
    let l = dpca.filter_lag;
    let ncomp = dpca.ncomp;
    let n_interior = n - 2 * l;

    // Consistency checks against the supplied DpcaResult.
    if dpca.scores.nrows() != n_interior || dpca.scores.ncols() != ncomp {
        return Err(FdarError::InvalidDimension {
            parameter: "dpca.scores",
            expected: format!("{n_interior} rows × {ncomp} cols (matching data/filter_lag)"),
            actual: format!(
                "{} rows × {} cols",
                dpca.scores.nrows(),
                dpca.scores.ncols()
            ),
        });
    }
    if let Some(f0) = dpca.filters.first() {
        if f0.ncols() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "dpca.filters",
                expected: format!("{m} columns (matching data grid)"),
                actual: format!("{} columns", f0.ncols()),
            });
        }
    }

    let weights = simpsons_weights(argvals);
    let n_rows = 2 * l + 1;

    // full-K fitted curves over the DPCA interior [L, N-1-L]; edges use zero-padded
    // scores (out-of-range score rows contribute 0).
    let mut fitted_flat = vec![0.0f64; n_interior * m];
    for c in 0..ncomp {
        let filt = &dpca.filters[c];
        for t in l..=(n - 1 - l) {
            let row = t - l; // interior row
            for lag_idx in 0..n_rows {
                let lag = lag_idx as isize - l as isize;
                let s_idx = row as isize + lag; // score interior index (t' + lag)
                if s_idx < 0 || s_idx as usize >= n_interior {
                    continue; // zero-padded score at the boundary
                }
                let s = dpca.scores[(s_idx as usize, c)];
                for j in 0..m {
                    fitted_flat[row + j * n_interior] += filt[(lag_idx, j)] * s;
                }
            }
        }
    }
    let fitted = FdMatrix::from_column_major(fitted_flat, n_interior, m)
        .expect("dimension invariant: fitted.len() == (N-2L) * m");

    // Per-K cumulative reconstruction error over the fully-defined window
    // t ∈ [2L, N-1-2L], where every score index (t'+lag) is in range.
    let lo = 2 * l;
    let hi = n.saturating_sub(2 * l + 1);
    let mut reconstruction_error = vec![0.0f64; ncomp];
    for k in 1..=ncomp {
        let mut err = 0.0;
        let mut count = 0usize;
        for t in lo..=hi {
            let row = t - l;
            for j in 0..m {
                let mut xhat = 0.0;
                for c in 0..k {
                    let filt = &dpca.filters[c];
                    for lag_idx in 0..n_rows {
                        let lag = lag_idx as isize - l as isize;
                        let s_idx = (row as isize + lag) as usize;
                        xhat += filt[(lag_idx, j)] * dpca.scores[(s_idx, c)];
                    }
                }
                let d = data[(t, j)] - xhat;
                err += d * d * weights[j];
            }
            count += 1;
        }
        reconstruction_error[k - 1] = if count > 0 { err / count as f64 } else { 0.0 };
    }

    Ok(DpcaReconstruction {
        fitted,
        reconstruction_error,
        valid_range: dpca.valid_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// i.i.d. N(0,1) white-noise curve set, deterministic under `seed`.
    fn white_noise(n: usize, m: usize, seed: u64) -> FdMatrix {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v = vec![0.0f64; n * m];
        for x in &mut v {
            *x = rng.sample::<f64, _>(rand_distr::StandardNormal);
        }
        FdMatrix::from_column_major(v, n, m).unwrap()
    }

    /// Rank-`r` series X_t[j] = Σ_c b_{t,c} ψ_c[j] with AR(1) score processes.
    fn multimode_series(n: usize, m: usize, ar: &[f64], seed: u64) -> FdMatrix {
        let r = ar.len();
        let grid = uniform_grid(m);
        // Smooth orthogonal-ish shapes: cosine modes.
        let shapes: Vec<Vec<f64>> = (0..r)
            .map(|c| {
                grid.iter()
                    .map(|&t| ((c + 1) as f64 * std::f64::consts::PI * t).cos())
                    .collect()
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut b = vec![0.0f64; r];
        let mut v = vec![0.0f64; n * m];
        // Burn-in the AR score processes.
        for _ in 0..200 {
            for c in 0..r {
                b[c] = ar[c] * b[c] + rng.sample::<f64, _>(rand_distr::StandardNormal);
            }
        }
        for i in 0..n {
            for c in 0..r {
                b[c] = ar[c] * b[c] + rng.sample::<f64, _>(rand_distr::StandardNormal);
            }
            for j in 0..m {
                let mut s = 0.0;
                for c in 0..r {
                    s += b[c] * shapes[c][j];
                }
                v[i + j * n] = s;
            }
        }
        FdMatrix::from_column_major(v, n, m).unwrap()
    }

    // ── FTS-03-01 ──────────────────────────────────────────────────────────

    #[test]
    fn tracer_white_noise_flat() {
        let (n, m) = (120, 6);
        let argvals = uniform_grid(m);
        let data = white_noise(n, m, 7);
        let sd = spectral_density(&data, &argvals, None).unwrap();

        // Shape invariants.
        assert_eq!(sd.freqs.len(), n);
        assert_eq!(sd.re.len(), n);
        assert_eq!(sd.im.len(), n);
        assert_eq!(sd.re[0].len(), m * m);

        // Rigorous correctness: FFT path == direct-sum DFT of the weighted lag ops.
        let xbar = mean_curve(&data, n, m);
        let bw = sd.bandwidth;
        let max_h = bw.min(n - 1);
        let lag_ops: Vec<Vec<f64>> = (0..=max_h)
            .map(|h| super::super::acf::autocovariance_matrix(&data, &xbar, h, n, m))
            .collect();
        for &k in &[0usize, 1, 5, 37, n / 2, n - 1] {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            for &(j1, j2) in &[(0usize, 0usize), (1, 3), (4, 2)] {
                let mut val = Complex::new(0.0, 0.0);
                for (h, c_h) in lag_ops.iter().enumerate() {
                    let w = bartlett_weight(h, bw);
                    let e_neg = Complex::new((h as f64 * theta).cos(), -(h as f64 * theta).sin());
                    val += Complex::new(w * c_h[j1 + j2 * m], 0.0) * e_neg;
                    if h > 0 {
                        let e_pos =
                            Complex::new((h as f64 * theta).cos(), (h as f64 * theta).sin());
                        val += Complex::new(w * c_h[j2 + j1 * m], 0.0) * e_pos;
                    }
                }
                assert!((sd.re[k][j1 + j2 * m] - val.re).abs() < 1e-9);
                assert!((sd.im[k][j1 + j2 * m] - val.im).abs() < 1e-9);
            }
        }

        // DC (θ=0) diagonal is the summed real long-run variance and is positive.
        assert!(sd.re[0][0] > 0.0);

        // Near-flatness for white noise: every frequency's diagonal stays close to
        // the mean-over-frequencies (deviation bounded by the finite-sample scale).
        let mean_diag: f64 = (0..n).map(|k| sd.re[k][0]).sum::<f64>() / n as f64;
        let c0_00 = lag_ops[0][0];
        for k in 0..n {
            assert!((sd.re[k][0] - mean_diag).abs() < 0.6 * c0_00.abs());
        }
    }

    #[test]
    fn spectral_density_errors_empty_and_argvals() {
        let argvals = uniform_grid(5);
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(matches!(
            spectral_density(&empty, &[], None),
            Err(FdarError::InvalidDimension {
                parameter: "data",
                ..
            })
        ));
        let data = white_noise(20, 5, 1);
        assert!(matches!(
            spectral_density(&data, &argvals[..3], None),
            Err(FdarError::InvalidDimension {
                parameter: "argvals",
                ..
            })
        ));
        assert!(matches!(
            spectral_density(&data, &argvals, Some(0)),
            Err(FdarError::InvalidParameter {
                parameter: "bandwidth",
                ..
            })
        ));
    }

    #[test]
    fn spectral_density_deterministic() {
        let (n, m) = (30, 5);
        let argvals = uniform_grid(m);
        let d1 = white_noise(n, m, 99);
        let d2 = white_noise(n, m, 99);
        let s1 = spectral_density(&d1, &argvals, None).unwrap();
        let s2 = spectral_density(&d2, &argvals, None).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn spectral_density_hermitian_symmetry() {
        let (n, m) = (60, 6);
        let argvals = uniform_grid(m);
        let data = multimode_series(n, m, &[0.7, 0.4], 11);
        let sd = spectral_density(&data, &argvals, None).unwrap();
        for &k in &[1usize, 7, 23] {
            for &(j1, j2) in &[(0usize, 2usize), (1, 5), (3, 4)] {
                let a = sd.im[k][j1 + j2 * m];
                let b = sd.im[k][j2 + j1 * m];
                assert!((a + b).abs() < 1e-9, "Hermitian im antisymmetry at k={k}");
            }
        }
    }

    // ── FTS-03-02 ──────────────────────────────────────────────────────────

    #[test]
    fn dpca_shapes_and_finiteness() {
        let (n, m, ncomp) = (100, 12, 3);
        let argvals = uniform_grid(m);
        let data = multimode_series(n, m, &[0.7, 0.5, 0.3], 5);
        let res = dpca(&data, &argvals, ncomp, None, None).unwrap();
        let l = res.filter_lag;
        assert_eq!(res.filters.len(), ncomp);
        for f in &res.filters {
            assert_eq!(f.shape(), (2 * l + 1, m));
            assert!(f.as_slice().iter().all(|x| x.is_finite()));
        }
        assert_eq!(res.scores.shape(), (n - 2 * l, ncomp));
        assert!(res.scores.as_slice().iter().all(|x| x.is_finite()));
        assert_eq!(res.eigenvalues.len(), ncomp);
        for ev in &res.eigenvalues {
            assert_eq!(ev.len(), res.n_freqs);
        }
        assert_eq!(res.valid_range, (l, n - 1 - l));
    }

    #[test]
    fn dpca_white_noise_flat_eigenvalues() {
        let (n, m, ncomp) = (120, 8, 2);
        let argvals = uniform_grid(m);
        let data = white_noise(n, m, 3);
        let res = dpca(&data, &argvals, ncomp, None, None).unwrap();
        // Leading eigenvalue is ~constant across frequencies for a flat spectrum.
        let lead = &res.eigenvalues[0];
        let mean: f64 = lead.iter().sum::<f64>() / lead.len() as f64;
        assert!(mean > 0.0);
        for &v in lead {
            assert!((v - mean).abs() < 0.7 * mean);
        }
    }

    #[test]
    fn dpca_parameter_range_errors() {
        let (n, m) = (60, 6);
        let argvals = uniform_grid(m);
        let data = multimode_series(n, m, &[0.6], 2);
        assert!(matches!(
            dpca(&data, &argvals, 0, None, None),
            Err(FdarError::InvalidParameter {
                parameter: "ncomp",
                ..
            })
        ));
        assert!(matches!(
            dpca(&data, &argvals, m + 1, None, None),
            Err(FdarError::InvalidParameter {
                parameter: "ncomp",
                ..
            })
        ));
        assert!(matches!(
            dpca(&data, &argvals, 2, None, Some(n / 2)),
            Err(FdarError::InvalidParameter {
                parameter: "filter_lag",
                ..
            })
        ));
    }

    // ── FTS-03-03 ──────────────────────────────────────────────────────────

    #[test]
    fn dpca_reconstruct_monotone_error() {
        let (n, m, ncomp) = (100, 16, 3);
        let argvals = uniform_grid(m);
        let data = multimode_series(n, m, &[0.7, 0.5, 0.3], 21);
        let res = dpca(&data, &argvals, ncomp, None, Some(2)).unwrap();
        let rec = dpca_reconstruct(&data, &argvals, &res).unwrap();
        assert_eq!(rec.reconstruction_error.len(), ncomp);
        for k in 0..ncomp - 1 {
            assert!(
                rec.reconstruction_error[k] >= rec.reconstruction_error[k + 1] - 1e-9,
                "error not monotone at K={k}: {:?}",
                rec.reconstruction_error
            );
        }
        assert_eq!(rec.valid_range, res.valid_range);
    }

    #[test]
    fn dpca_reconstruct_rank1_exact() {
        // Rank-1 series X_t[j] = a_t * phi[j], a_t an AR(1) process.
        let (n, m) = (80, 20);
        let argvals = uniform_grid(m);
        let phi: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let mut rng = StdRng::seed_from_u64(42);
        let mut a = 0.0f64;
        for _ in 0..200 {
            a = 0.8 * a + rng.sample::<f64, _>(rand_distr::StandardNormal);
        }
        let mut v = vec![0.0f64; n * m];
        for i in 0..n {
            a = 0.8 * a + rng.sample::<f64, _>(rand_distr::StandardNormal);
            for j in 0..m {
                v[i + j * n] = a * phi[j];
            }
        }
        let data = FdMatrix::from_column_major(v, n, m).unwrap();
        let res = dpca(&data, &argvals, 3, None, Some(3)).unwrap();
        let rec = dpca_reconstruct(&data, &argvals, &res).unwrap();
        assert_eq!(rec.fitted.shape(), (n - 2 * res.filter_lag, m));
        // One dynamic component reconstructs the rank-1 series to near machine precision.
        assert!(
            rec.reconstruction_error[0] < 1e-4,
            "rank-1 K=1 error too large: {}",
            rec.reconstruction_error[0]
        );
    }

    #[test]
    fn dpca_reconstruct_dimension_mismatch() {
        let (n, m) = (60, 6);
        let argvals = uniform_grid(m);
        let data = multimode_series(n, m, &[0.6, 0.4], 8);
        let res = dpca(&data, &argvals, 2, None, Some(2)).unwrap();
        // Wrong grid length → dimension error.
        assert!(matches!(
            dpca_reconstruct(&data, &argvals[..m - 1], &res),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
