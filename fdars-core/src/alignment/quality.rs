//! Alignment quality metrics: warp complexity, smoothness, variance decomposition,
//! and pairwise consistency.

use super::pairwise::elastic_align_pair;
use super::srsf::compose_warps;
use super::KarcherMeanResult;
use crate::error::FdarError;
use crate::helpers::{gradient_uniform, l2_distance, simpsons_weights};
use crate::matrix::FdMatrix;

/// Comprehensive alignment quality assessment.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentQuality {
    /// Per-curve geodesic distance from warp to identity.
    pub warp_complexity: Vec<f64>,
    /// Mean warp complexity.
    pub mean_warp_complexity: f64,
    /// Per-curve bending energy ∫(γ'')² dt.
    pub warp_smoothness: Vec<f64>,
    /// Mean warp smoothness (bending energy).
    pub mean_warp_smoothness: f64,
    /// Total variance: (1/n) Σ ∫(f_i - mean_orig)² dt.
    pub total_variance: f64,
    /// Amplitude variance: (1/n) Σ ∫(f_i^aligned - mean_aligned)² dt.
    pub amplitude_variance: f64,
    /// Phase variance: total - amplitude (clamped ≥ 0).
    pub phase_variance: f64,
    /// Phase-to-total variance ratio.
    pub phase_amplitude_ratio: f64,
    /// Pointwise ratio: aligned_var / orig_var per time point.
    pub pointwise_variance_ratio: Vec<f64>,
    /// Mean variance reduction.
    pub mean_variance_reduction: f64,
}

/// Compute warp complexity: geodesic distance from a warp to the identity.
///
/// This is `arccos(⟨ψ, ψ_id⟩)` on the Hilbert sphere.
pub fn warp_complexity(gamma: &[f64], argvals: &[f64]) -> f64 {
    crate::warping::phase_distance(gamma, argvals)
}

/// Compute warp smoothness (bending energy): ∫(γ'')² dt.
pub fn warp_smoothness(gamma: &[f64], argvals: &[f64]) -> f64 {
    let m = gamma.len();
    if m < 3 {
        return 0.0;
    }

    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_prime = gradient_uniform(gamma, h);
    let gam_pprime = gradient_uniform(&gam_prime, h);

    let integrand: Vec<f64> = gam_pprime.iter().map(|&g| g * g).collect();
    crate::helpers::trapz(&integrand, argvals)
}

/// Compute comprehensive alignment quality metrics.
///
/// # Arguments
/// * `data` — Original functional data (n × m)
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
pub fn alignment_quality(
    data: &FdMatrix,
    karcher: &KarcherMeanResult,
    argvals: &[f64],
) -> AlignmentQuality {
    let (n, m) = data.shape();
    let weights = simpsons_weights(argvals);

    // Per-curve warp complexity and smoothness
    let wc: Vec<f64> = (0..n)
        .map(|i| {
            let gamma: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();
            warp_complexity(&gamma, argvals)
        })
        .collect();
    let ws: Vec<f64> = (0..n)
        .map(|i| {
            let gamma: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();
            warp_smoothness(&gamma, argvals)
        })
        .collect();

    let mean_wc = wc.iter().sum::<f64>() / n as f64;
    let mean_ws = ws.iter().sum::<f64>() / n as f64;

    // Compute original mean
    let orig_mean = crate::fdata::mean_1d(data);

    // Total variance
    let total_var: f64 = (0..n)
        .map(|i| {
            let fi = data.row(i);
            let d = l2_distance(&fi, &orig_mean, &weights);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    // Aligned mean
    let aligned_mean = crate::fdata::mean_1d(&karcher.aligned_data);

    // Amplitude variance
    let amp_var: f64 = (0..n)
        .map(|i| {
            let fi = karcher.aligned_data.row(i);
            let d = l2_distance(&fi, &aligned_mean, &weights);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    let phase_var = (total_var - amp_var).max(0.0);
    let ratio = if total_var > 1e-10 {
        phase_var / total_var
    } else {
        0.0
    };

    // Pointwise variance ratio
    let mut pw_ratio = vec![0.0; m];
    for j in 0..m {
        let col_orig = data.column(j);
        let mean_orig_j = col_orig.iter().sum::<f64>() / n as f64;
        let var_orig: f64 = col_orig
            .iter()
            .map(|&v| (v - mean_orig_j).powi(2))
            .sum::<f64>()
            / n as f64;

        let col_aligned = karcher.aligned_data.column(j);
        let mean_aligned_j = col_aligned.iter().sum::<f64>() / n as f64;
        let var_aligned: f64 = col_aligned
            .iter()
            .map(|&v| (v - mean_aligned_j).powi(2))
            .sum::<f64>()
            / n as f64;

        pw_ratio[j] = if var_orig > 1e-15 {
            var_aligned / var_orig
        } else {
            1.0
        };
    }

    let mean_vr = pw_ratio.iter().sum::<f64>() / m as f64;

    AlignmentQuality {
        warp_complexity: wc,
        mean_warp_complexity: mean_wc,
        warp_smoothness: ws,
        mean_warp_smoothness: mean_ws,
        total_variance: total_var,
        amplitude_variance: amp_var,
        phase_variance: phase_var,
        phase_amplitude_ratio: ratio,
        pointwise_variance_ratio: pw_ratio,
        mean_variance_reduction: mean_vr,
    }
}

/// Generate triplet indices (i,j,k) with i<j<k, capped at `max_triplets` (0 = all).
fn triplet_indices(n: usize, max_triplets: usize) -> Vec<(usize, usize, usize)> {
    let total = n * (n - 1) * (n - 2) / 6;
    let cap = if max_triplets > 0 {
        max_triplets.min(total)
    } else {
        total
    };
    (0..n)
        .flat_map(|i| ((i + 1)..n).flat_map(move |j| ((j + 1)..n).map(move |k| (i, j, k))))
        .take(cap)
        .collect()
}

/// Compute the warp deviation for one triplet: ‖γ_ij∘γ_jk − γ_ik‖_L2.
fn triplet_warp_deviation(
    data: &FdMatrix,
    argvals: &[f64],
    weights: &[f64],
    i: usize,
    j: usize,
    k: usize,
    lambda: f64,
) -> f64 {
    let fi = data.row(i);
    let fj = data.row(j);
    let fk = data.row(k);
    let rij = elastic_align_pair(&fi, &fj, argvals, lambda);
    let rjk = elastic_align_pair(&fj, &fk, argvals, lambda);
    let rik = elastic_align_pair(&fi, &fk, argvals, lambda);
    let composed = compose_warps(&rij.gamma, &rjk.gamma, argvals);
    l2_distance(&composed, &rik.gamma, weights)
}

/// Measure pairwise alignment consistency via triplet checks.
///
/// For triplets (i,j,k), checks `γ_ij ∘ γ_jk ≈ γ_ik` by measuring the L2
/// deviation of the composed warp from the direct warp.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight
/// * `max_triplets` — Maximum number of triplets to check (0 = all)
pub fn pairwise_consistency(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    max_triplets: usize,
) -> f64 {
    let n = data.nrows();
    if n < 3 {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    let triplets = triplet_indices(n, max_triplets);
    if triplets.is_empty() {
        return 0.0;
    }

    let total_dev: f64 = triplets
        .iter()
        .map(|&(i, j, k)| triplet_warp_deviation(data, argvals, &weights, i, j, k, lambda))
        .sum();
    total_dev / triplets.len() as f64
}

// ---------------------------------------------------------------------------
// Registration-quality scores (FEAT-07)
// ---------------------------------------------------------------------------
//
// These three functions return `Result<f64, FdarError>` — unlike the raw-f64
// neighbors (`warp_complexity`, `warp_smoothness`, `pairwise_consistency`) in
// this file — so that dimension/parameter validation can be surfaced to the
// caller rather than silently producing NaN. This is an intentional deviation
// from the older neighbors, noted in each function's rustdoc.
//
// All three implement **standalone-energy** forms: they measure the spread or
// structure of the *registered* data in absolute L2 units and do NOT divide by
// the spread of the unregistered data. This differs from scikit-fda's ratio-based
// scorers (`LeastSquares`, `PairwiseCorrelation`, `SobolevLeastSquares`), which
// return a ratio-to-original. The standalone form avoids division-by-zero when
// the original data is nearly constant and is more interpretable as an absolute
// quality measure.

/// Compute the least-squares registration score: mean Simpson-weighted L2 spread
/// of the registered curves around their cross-sectional mean.
///
/// **Formula:** `(1/n) Σᵢ ∫ (registeredᵢ(t) − mean(t))² dt`
///
/// where `mean(t)` is the cross-sectional sample mean and the integral is
/// approximated with Simpson weights over `argvals`.
///
/// # Standalone-energy form
///
/// This is an **absolute** (standalone-energy) measure of residual spread, not a
/// ratio to the unregistered data's spread. Lower is better after registration.
/// This intentionally diverges from scikit-fda's `LeastSquares` scorer which
/// returns a ratio; the standalone form avoids division-by-zero on constant data.
///
/// # Returns `Result`
///
/// Unlike the raw-`f64` quality functions in this module (`warp_complexity`,
/// `warp_smoothness`), this function returns `Result<f64, FdarError>` to surface
/// dimension mismatches via [`FdarError::InvalidDimension`].
///
/// # Arguments
/// * `registered` — Registered functional data (n × m)
/// * `argvals` — Evaluation points (length m)
///
/// # Errors
/// * [`FdarError::InvalidDimension`] — if `registered` is empty or
///   `argvals.len() != m`
/// * [`FdarError::InvalidParameter`] — if `argvals.len() < 2`
pub fn least_squares_score(registered: &FdMatrix, argvals: &[f64]) -> Result<f64, FdarError> {
    let (n, m) = registered.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "registered",
            expected: "non-empty matrix".to_string(),
            actual: format!("{}×{}", n, m),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: m.to_string(),
            actual: argvals.len().to_string(),
        });
    }
    if argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must have at least 2 evaluation points".to_string(),
        });
    }

    let weights = simpsons_weights(argvals);
    let mean = crate::fdata::mean_1d(registered);

    let score = (0..n)
        .map(|i| {
            let fi = registered.row(i);
            fi.iter()
                .zip(mean.iter())
                .zip(weights.iter())
                .map(|((&a, &b), &w)| (a - b) * (a - b) * w)
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;

    Ok(score)
}

/// Compute the Sobolev least-squares registration score: LS spread plus a
/// derivative-penalty term weighted by `lambda`.
///
/// **Formula:** `LS_term + λ · (1/n) Σᵢ ∫ (fᵢ′(t) − mean′(t))² dt`
///
/// where the LS term equals [`least_squares_score`] and the derivative `fᵢ′` is
/// approximated by [`gradient_uniform`] (5-point stencil, same as `warp_smoothness`).
///
/// # Standalone-energy form
///
/// Like [`least_squares_score`], this is an absolute measure, not a ratio to the
/// unregistered data. This diverges from scikit-fda's `SobolevLeastSquares` scorer.
///
/// # Uniform-grid requirement (when `lambda > 0`)
///
/// The derivative term uses [`gradient_uniform`], which requires a **uniform**
/// `argvals` grid. When `lambda > 0` this function validates uniformity and
/// returns [`FdarError::InvalidParameter`] on a non-uniform grid. Use
/// [`gradient_nonuniform`][crate::helpers::gradient_nonuniform] externally if
/// your grid is non-uniform and compose your own Sobolev score.
///
/// # Returns `Result`
///
/// Returns `Result<f64, FdarError>` to surface dimension/parameter validation
/// errors, consistent with the other FEAT-07 score functions.
///
/// # Arguments
/// * `registered` — Registered functional data (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Non-negative weight for the derivative penalty (0.0 reproduces
///   [`least_squares_score`])
///
/// # Errors
/// * [`FdarError::InvalidDimension`] — if `registered` is empty or
///   `argvals.len() != m`
/// * [`FdarError::InvalidParameter`] — if `argvals.len() < 2`, `lambda < 0.0`,
///   or `lambda > 0` with a non-uniform `argvals` grid
pub fn sobolev_least_squares_score(
    registered: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
) -> Result<f64, FdarError> {
    let (n, m) = registered.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "registered",
            expected: "non-empty matrix".to_string(),
            actual: format!("{}×{}", n, m),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: m.to_string(),
            actual: argvals.len().to_string(),
        });
    }
    if argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must have at least 2 evaluation points".to_string(),
        });
    }
    if lambda < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: "lambda must be non-negative".to_string(),
        });
    }

    let weights = simpsons_weights(argvals);
    let mean = crate::fdata::mean_1d(registered);

    // LS term: (1/n) Σᵢ ∫ (fᵢ − mean)² dt
    let ls_term = (0..n)
        .map(|i| {
            let fi = registered.row(i);
            fi.iter()
                .zip(mean.iter())
                .zip(weights.iter())
                .map(|((&a, &b), &w)| (a - b) * (a - b) * w)
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;

    if lambda == 0.0 {
        return Ok(ls_term);
    }

    // Sobolev derivative term: (1/n) Σᵢ ∫ (fᵢ′ − mean′)² dt
    // gradient_uniform assumes a uniform argvals grid. Validate uniformity
    // before computing h so callers on non-uniform grids get an error instead
    // of a silently wrong derivative penalty.
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let uniform = argvals
        .windows(2)
        .all(|w| ((w[1] - w[0]) - h).abs() < 1e-9 * h.abs().max(1e-12));
    if !uniform {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "sobolev_least_squares_score with lambda>0 requires a uniform grid; \
                      use gradient_nonuniform externally for non-uniform grids"
                .to_string(),
        });
    }
    let mean_prime = gradient_uniform(&mean, h);

    let sobol_term = (0..n)
        .map(|i| {
            let fi_row = registered.row(i);
            let fi_prime = gradient_uniform(&fi_row, h);
            fi_prime
                .iter()
                .zip(mean_prime.iter())
                .zip(weights.iter())
                .map(|((&a, &b), &w)| (a - b) * (a - b) * w)
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;

    Ok(ls_term + lambda * sobol_term)
}

/// Compute the pairwise correlation registration score: mean functional Pearson
/// correlation over all n(n−1)/2 unordered curve pairs.
///
/// **Formula:** `mean over (i<k) of [⟨f̃ᵢ, f̃_k⟩_L2 / (‖f̃ᵢ‖_L2 · ‖f̃_k‖_L2)]`
///
/// where `f̃ᵢ = fᵢ − μᵢ` is the mean-centred curve, `μᵢ = ∫ fᵢ dt / ∫ dt` is
/// the Simpson-weighted functional mean, and all inner products and norms are
/// Simpson-weighted. This is the functional analogue of **Pearson correlation**
/// (centred), not cosine similarity (uncentred).
///
/// A zero-variance curve (`‖f̃ᵢ‖ ≈ 0`, i.e. a nearly constant curve) contributes
/// 0 to every pair it participates in (NaN guard).
///
/// Higher scores indicate greater pairwise alignment — use this score to confirm
/// that registration has increased curve-to-curve similarity.
///
/// # Standalone form
///
/// This computes the mean Pearson correlation of the registered curves directly,
/// without dividing by the correlation of the unregistered curves. This diverges
/// from scikit-fda's `PairwiseCorrelation` scorer which returns a ratio.
///
/// # Complexity
///
/// O(n² · m) — suitable for moderate n (e.g., n ≤ 500 with m ≤ 1000).
///
/// # Returns `Result`
///
/// Returns `Result<f64, FdarError>` to surface dimension/parameter validation
/// errors, consistent with the other FEAT-07 score functions.
///
/// # Arguments
/// * `registered` — Registered functional data (n × m), n ≥ 2
/// * `argvals` — Evaluation points (length m, at least 2)
///
/// # Errors
/// * [`FdarError::InvalidDimension`] — if `m == 0`, `m < 2`, or `argvals.len() != m`
/// * [`FdarError::InvalidParameter`] — if `n < 2` (need at least 2 curves to
///   form a pair)
pub fn pairwise_correlation_score(
    registered: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = registered.shape();
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "registered",
            expected: "non-empty matrix (m > 0)".to_string(),
            actual: format!("{}×{}", n, m),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: m.to_string(),
            actual: argvals.len().to_string(),
        });
    }
    if argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must have at least 2 evaluation points".to_string(),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n",
            message: "pairwise correlation requires at least 2 curves".to_string(),
        });
    }

    let weights = simpsons_weights(argvals);
    let weight_sum: f64 = weights.iter().sum();

    // Precompute centred curves and their L2 norms for O(n·m) instead of O(n²·m).
    // μᵢ = (Σⱼ fᵢ(tⱼ) · wⱼ) / (Σⱼ wⱼ)  — Simpson-weighted functional mean.
    // f̃ᵢ = fᵢ − μᵢ  — centred curve (true Pearson, not cosine similarity).
    let centred: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let fi = registered.row(i);
            let mu: f64 = fi
                .iter()
                .zip(weights.iter())
                .map(|(&a, &w)| a * w)
                .sum::<f64>()
                / weight_sum;
            fi.iter().map(|&a| a - mu).collect()
        })
        .collect();

    let norms: Vec<f64> = centred
        .iter()
        .map(|fi_c| {
            fi_c.iter()
                .zip(weights.iter())
                .map(|(&a, &w)| a * a * w)
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let n_pairs = n * (n - 1) / 2;
    let corr_sum: f64 = (0..n)
        .flat_map(|i| (i + 1..n).map(move |k| (i, k)))
        .map(|(i, k)| {
            let denom = norms[i] * norms[k];
            if denom < 1e-15 {
                // At least one curve is nearly constant — skip this pair.
                0.0
            } else {
                let inner: f64 = centred[i]
                    .iter()
                    .zip(centred[k].iter())
                    .zip(weights.iter())
                    .map(|((&a, &b), &w)| a * b * w)
                    .sum();
                inner / denom
            }
        })
        .sum();

    Ok(corr_sum / n_pairs as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    fn gaussian_bump(argvals: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
        argvals
            .iter()
            .map(|&t| (-(t - mu).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect()
    }

    /// Build a matrix of n Gaussian bumps with centres spread around mu=0.5.
    /// `delta` is the half-spread (curve i gets centre 0.5 + (i - n/2)*delta/n).
    fn make_shifted_bumps(n: usize, m: usize, delta: f64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let mu = 0.5 + (i as f64 - (n as f64 - 1.0) / 2.0) * delta / n as f64;
            let bump = gaussian_bump(&argvals, mu, 0.1);
            for j in 0..m {
                data[(i, j)] = bump[j];
            }
        }
        (data, argvals)
    }

    // ---- FEAT-07-A: least_squares_score on identical constant curves = 0.0 --

    #[test]
    fn test_ls_score_identical_curves() {
        let m = 51;
        let argvals = uniform_grid(m);
        // All four rows are the same constant curve → mean == every curve → score = 0
        let mut data = FdMatrix::zeros(4, m);
        for i in 0..4 {
            for j in 0..m {
                data[(i, j)] = 2.5;
            }
        }
        let score = least_squares_score(&data, &argvals).unwrap();
        assert!(score.abs() < 1e-12, "expected 0.0, got {score}");
    }

    // ---- FEAT-07-E: sobolev with lambda=0 equals least_squares_score --------

    #[test]
    fn test_sobolev_score_lambda_zero() {
        let (data, argvals) = make_shifted_bumps(5, 51, 0.1);
        let ls = least_squares_score(&data, &argvals).unwrap();
        let sobol = sobolev_least_squares_score(&data, &argvals, 0.0).unwrap();
        assert!(
            (sobol - ls).abs() < 1e-12,
            "sobolev(lambda=0) should equal least_squares_score: ls={ls}, sobol={sobol}"
        );
    }

    // ---- FEAT-07-F: sobolev with lambda>0 is >= sobolev(lambda=0) -----------

    #[test]
    fn test_sobolev_score_lambda_positive() {
        let (data, argvals) = make_shifted_bumps(5, 51, 0.1);
        let sobol0 = sobolev_least_squares_score(&data, &argvals, 0.0).unwrap();
        let sobol_pos = sobolev_least_squares_score(&data, &argvals, 1.0).unwrap();
        assert!(
            sobol_pos >= sobol0 - 1e-12,
            "sobolev(lambda>0) should be >= sobolev(lambda=0): sobol0={sobol0}, sobol_pos={sobol_pos}"
        );
    }

    // ---- FEAT-07-B: least_squares_score drops after registration -------------
    // (uses least_squares_shift_registration from shift.rs — added in Task 2)

    #[test]
    fn test_ls_score_drops_after_registration() {
        let (data, argvals) = make_shifted_bumps(5, 101, 0.3);
        let max_shift = 0.25;
        let result =
            crate::alignment::shift::least_squares_shift_registration(&data, &argvals, max_shift)
                .unwrap();
        let score_before = least_squares_score(&data, &argvals).unwrap();
        let score_after = least_squares_score(&result.registered_data, &argvals).unwrap();
        assert!(
            score_after < score_before,
            "LS score should drop after registration: before={score_before}, after={score_after}"
        );
    }

    // ---- FEAT-07-C: pairwise_correlation_score rises after registration ------

    #[test]
    fn test_pairwise_corr_rises_after_registration() {
        let (data, argvals) = make_shifted_bumps(5, 101, 0.3);
        let max_shift = 0.25;
        let result =
            crate::alignment::shift::least_squares_shift_registration(&data, &argvals, max_shift)
                .unwrap();
        let score_before = pairwise_correlation_score(&data, &argvals).unwrap();
        let score_after = pairwise_correlation_score(&result.registered_data, &argvals).unwrap();
        assert!(
            score_after > score_before,
            "Pairwise correlation should rise after registration: before={score_before}, after={score_after}"
        );
    }

    // ---- FEAT-07-D: pairwise_correlation_score with n=1 returns Err ---------

    #[test]
    fn test_pairwise_corr_n1_error() {
        let m = 51;
        let argvals = uniform_grid(m);
        let mut single = FdMatrix::zeros(1, m);
        for j in 0..m {
            single[(0, j)] = 1.0;
        }
        let result = pairwise_correlation_score(&single, &argvals);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "expected Err(InvalidParameter), got {result:?}"
        );
    }

    // ---- WR-02: all three score fns reject m<2 (consistent with shift fn) ---

    #[test]
    fn test_score_fns_reject_single_point_grid() {
        // m=1 matrix: argvals has only one evaluation point — the integral would
        // be a bare point-mass (undefined as a functional L2 norm).
        let argvals_1pt = vec![0.5_f64];
        let mut data_1col = FdMatrix::zeros(3, 1);
        for i in 0..3 {
            data_1col[(i, 0)] = 1.0;
        }

        let r1 = least_squares_score(&data_1col, &argvals_1pt);
        assert!(
            matches!(r1, Err(FdarError::InvalidParameter { .. })),
            "least_squares_score m=1 should return Err(InvalidParameter), got {r1:?}"
        );

        let r2 = sobolev_least_squares_score(&data_1col, &argvals_1pt, 0.0);
        assert!(
            matches!(r2, Err(FdarError::InvalidParameter { .. })),
            "sobolev_least_squares_score m=1 should return Err(InvalidParameter), got {r2:?}"
        );

        // pairwise_correlation_score: m=0 is caught by the InvalidDimension guard first;
        // m=1 hits the argvals.len() < 2 InvalidParameter guard.
        let mut data_1col_2rows = FdMatrix::zeros(2, 1);
        data_1col_2rows[(0, 0)] = 1.0;
        data_1col_2rows[(1, 0)] = 2.0;
        let r3 = pairwise_correlation_score(&data_1col_2rows, &argvals_1pt);
        assert!(
            matches!(r3, Err(FdarError::InvalidParameter { .. })),
            "pairwise_correlation_score m=1 should return Err(InvalidParameter), got {r3:?}"
        );
    }
}
