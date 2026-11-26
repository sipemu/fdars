//! Align, then localise where two groups differ — separately in amplitude and
//! in phase (GH #76, ALN-LOCAL).
//!
//! [`elastic_local_difference_test`] answers *where* two groups of curves
//! differ, complementing the global (whether) test
//! [`super::elastic_phase_amplitude_test`]:
//!
//! 1. **Label-free joint alignment.** All curves are aligned to one elastic
//!    Karcher mean ([`super::karcher_mean_with_config`], optimiser options via
//!    [`super::WarpOptimConfig`]); the labels are never seen by the alignment,
//!    so the subsequent permutation / bootstrap inference is not biased by it
//!    (see [`super::elastic_phase_amplitude_test`] for the argument).
//! 2. **Amplitude** — a local two-sample test on the aligned curves
//!    `fᵢ∘γᵢ`.
//! 3. **Phase** — the same local test on the warps, represented either as the
//!    time displacement `γᵢ(t) − t` ([`PhaseRepresentation::Displacement`],
//!    default) or as `ψᵢ = √γᵢ'` on the unit-interval time scale
//!    ([`PhaseRepresentation::Psi`]).
//!
//! The local test is the interval-wise testing procedure (ITP, Pini & Vantini
//! 2016; [`crate::inference::itp_two_pop_regions`], default) or a simultaneous
//! confidence band for the mean difference
//! ([`crate::inference::scb_two_sample_band`]). Either way the result is a list
//! of grid regions `[t_start, t_end]` with the direction of the difference.
//!
//! # Direction conventions
//!
//! Group **A** is the smaller label value, group **B** the larger.
//! * amplitude: `Positive` ⇔ the aligned curves of A lie above those of B;
//! * phase (displacement): `Positive` ⇔ `γ_A(t) − t > γ_B(t) − t`. Since
//!   `fᵢ(γᵢ(t)) ≈ μ(t)`, a feature located at `t` in the template occurs at
//!   `γᵢ(t)` in curve `i`, so `Positive` ⇔ **A's features occur later** than
//!   B's around `t` (`Negative` ⇔ earlier); see [`LocalDifferenceTest::timing`];
//! * phase (ψ): `Positive` ⇔ A's warps run faster (A's clock is ahead in its
//!   rate) around `t`.
//!
//! # Error control
//!
//! ITP: interval-wise error rate ≤ `alpha` for every interval of the domain
//! (any interval on which H0 holds is wholly flagged with probability ≤
//! `alpha`). SCB: family-wise over the grid at level `alpha` (the band has
//! simultaneous coverage `1 − alpha`). Amplitude and phase are tested
//! separately; no correction across the two is applied.

use super::karcher::karcher_mean_with_config;
use super::{KarcherMeanResult, WarpOptimConfig};
use crate::error::FdarError;
use crate::inference::{
    itp_two_pop_regions, scb_two_sample_band, ItpDomainMapping, ItpRegionConfig, ItpResult,
};
use crate::matrix::FdMatrix;
use crate::regions::{difference_regions, DifferenceRegion, SignificanceDirection};
use crate::tolerance::MultiplierDistribution;
use crate::warping::gam_to_psi;

/// Local two-sample test used by [`elastic_local_difference_test`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum LocalTestMethod {
    /// Interval-wise testing procedure on a B-spline basis
    /// ([`crate::inference::itp_two_pop_regions`]); interval-wise error control.
    #[default]
    Itp,
    /// Simultaneous confidence band for the mean difference
    /// ([`crate::inference::scb_two_sample_band`]); family-wise over the grid.
    /// Uses the paired-difference construction of that function (the first
    /// `min(n_A, n_B)` curves of each group, in input order). The multiplier
    /// bootstrap is liberal for small groups: in a seeded null study (200
    /// replicates, 50 grid points) the family-wise rate of flagging any
    /// region was 35/200 (amplitude) and 38/200 (phase) with 10 curves per
    /// group, but 6/200 and 5/200 with 30 per group. The ITP gave 1/200 and
    /// 2/200 (10 per group) and 0/200 (30 per group) — conservative.
    Scb,
}

/// Representation of the warps for the phase test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PhaseRepresentation {
    /// Time displacement `γᵢ(t) − t` (in `argvals` units).
    #[default]
    Displacement,
    /// `ψᵢ = √γᵢ'` of the warp rescaled to `[0, 1]`.
    Psi,
}

/// Timing interpretation of a phase region (displacement representation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Timing {
    /// Group A's features occur later than group B's.
    ALater,
    /// Group A's features occur earlier than group B's.
    AEarlier,
}

/// Configuration for [`elastic_local_difference_test`].
///
/// Construct with `LocalDifferenceConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LocalDifferenceConfig {
    /// Local test (default [`LocalTestMethod::Itp`]).
    pub method: LocalTestMethod,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// ITP: requested number of B-spline basis functions (default 15).
    pub nbasis: usize,
    /// ITP: number of permutations (default 999).
    pub n_perm: usize,
    /// ITP: RNG seed (default 42). The phase test uses `seed + 1`.
    pub seed: u64,
    /// ITP: component → grid mapping (default [`ItpDomainMapping::Dominant`]).
    pub mapping: ItpDomainMapping,
    /// SCB: smoothing bandwidth in `argvals` units (default `None` = 5% of the
    /// domain length).
    pub scb_bandwidth: Option<f64>,
    /// SCB: number of multiplier-bootstrap replicates (default 999).
    pub scb_nb: usize,
    /// SCB: multiplier distribution (default Gaussian).
    pub multiplier: MultiplierDistribution,
    /// Warp representation for the phase test (default displacement).
    pub phase_representation: PhaseRepresentation,
    /// Maximum Karcher-mean iterations (default 20).
    pub max_iter: usize,
    /// Karcher-mean convergence tolerance (default 1e-4).
    pub tol: f64,
    /// Curve-to-template alignment options (default classic DP, λ = 0).
    pub warp: WarpOptimConfig,
}

impl Default for LocalDifferenceConfig {
    fn default() -> Self {
        Self {
            method: LocalTestMethod::Itp,
            alpha: 0.05,
            nbasis: 15,
            n_perm: 999,
            seed: 42,
            mapping: ItpDomainMapping::Dominant,
            scb_bandwidth: None,
            scb_nb: 999,
            multiplier: MultiplierDistribution::Gaussian,
            phase_representation: PhaseRepresentation::Displacement,
            max_iter: 20,
            tol: 1e-4,
            warp: WarpOptimConfig::default(),
        }
    }
}

/// One local two-sample test (amplitude or phase) of
/// [`elastic_local_difference_test`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LocalDifferenceTest {
    /// Regions where the groups differ (`direction` = sign of A − B of the
    /// tested quantity; ITP regions carry `min_adjusted_pvalue`).
    pub regions: Vec<DifferenceRegion>,
    /// Pointwise mean difference A − B of the tested quantity (ITP: raw
    /// sample means; SCB: the smoothed band centre).
    pub effect: Vec<f64>,
    /// ITP: pointwise adjusted p-value function on the grid.
    pub pointwise_adjusted_pvalues: Option<Vec<f64>>,
    /// ITP: the per-component result.
    pub itp: Option<ItpResult>,
    /// SCB: lower band limit for the mean difference.
    pub lower: Option<Vec<f64>>,
    /// SCB: upper band limit for the mean difference.
    pub upper: Option<Vec<f64>>,
}

impl LocalDifferenceTest {
    /// Timing reading of a phase region under
    /// [`PhaseRepresentation::Displacement`]: `Positive` ⇔ [`Timing::ALater`].
    #[must_use]
    pub fn timing(region: &DifferenceRegion) -> Timing {
        match region.direction {
            SignificanceDirection::Negative => Timing::AEarlier,
            _ => Timing::ALater,
        }
    }
}

/// Result of [`elastic_local_difference_test`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LocalDifferenceResult {
    /// Local test on the aligned curves.
    pub amplitude: LocalDifferenceTest,
    /// Local test on the warps.
    pub phase: LocalDifferenceTest,
    /// Tested phase quantity per curve (`γᵢ(t) − t` or `ψᵢ`), `n × m`.
    pub phase_data: FdMatrix,
    /// The label-free joint alignment (template, warps, aligned curves).
    pub alignment: KarcherMeanResult,
    /// `[label of A, label of B]` (A = smaller label).
    pub group_labels: [usize; 2],
    /// `[n_A, n_B]`.
    pub group_sizes: [usize; 2],
}

/// Align all curves jointly (label-free), then localise where two groups
/// differ in amplitude and in phase.
///
/// 1. **Label-free joint alignment.** All curves are aligned to one elastic
///    Karcher mean ([`super::karcher_mean_with_config`], optimiser options via
///    [`super::WarpOptimConfig`]); the labels are never seen by the alignment,
///    so the subsequent permutation / bootstrap inference is not biased by it
///    (see [`super::elastic_phase_amplitude_test`] for the argument).
/// 2. **Amplitude** — a local two-sample test on the aligned curves
///    `fᵢ∘γᵢ`.
/// 3. **Phase** — the same local test on the warps, represented either as the
///    time displacement `γᵢ(t) − t` ([`PhaseRepresentation::Displacement`],
///    default) or as `ψᵢ = √γᵢ'` on the unit-interval time scale
///    ([`PhaseRepresentation::Psi`]).
///
/// The local test is the interval-wise testing procedure (ITP, Pini & Vantini
/// 2016; [`crate::inference::itp_two_pop_regions`], default) or a simultaneous
/// confidence band for the mean difference
/// ([`crate::inference::scb_two_sample_band`]). Either way the result is a list
/// of grid regions `[t_start, t_end]` with the direction of the difference.
///
/// # Direction conventions
///
/// Group **A** is the smaller label value, group **B** the larger.
/// * amplitude: `Positive` ⇔ the aligned curves of A lie above those of B;
/// * phase (displacement): `Positive` ⇔ `γ_A(t) − t > γ_B(t) − t`. Since
///   `fᵢ(γᵢ(t)) ≈ μ(t)`, a feature located at `t` in the template occurs at
///   `γᵢ(t)` in curve `i`, so `Positive` ⇔ **A's features occur later** than
///   B's around `t` (`Negative` ⇔ earlier); see [`LocalDifferenceTest::timing`];
/// * phase (ψ): `Positive` ⇔ A's warps run faster (A's clock is ahead in its
///   rate) around `t`.
///
/// # Error control
///
/// ITP: interval-wise error rate ≤ `alpha` for every interval of the domain
/// (any interval on which H0 holds is wholly flagged with probability ≤
/// `alpha`). SCB: family-wise over the grid at level `alpha` (the band has
/// simultaneous coverage `1 − alpha`). Amplitude and phase are tested
/// separately; no correction across the two is applied.
///
///
/// # Arguments
/// * `data` — curves (`n × m`).
/// * `argvals` — strictly increasing grid (length `m ≥ 3`).
/// * `labels` — group label per curve; exactly two distinct values.
/// * `config` — see [`LocalDifferenceConfig`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] for an `argvals` / `labels` length
/// mismatch, `m < 3`, or a group with fewer than 2 (ITP) / 3 (SCB) curves;
/// [`FdarError::InvalidParameter`] for non-finite data, non-increasing
/// `argvals`, not exactly two labels, `alpha ∉ (0, 1)`, `max_iter = 0`, a
/// non-positive SCB bandwidth or an invalid warp configuration; errors of the
/// underlying ITP / SCB are forwarded.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{elastic_local_difference_test, LocalDifferenceConfig};
/// use fdars_core::FdMatrix;
///
/// let m = 50;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 16;
/// let mut data = FdMatrix::zeros(n, m);
/// let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
/// for i in 0..n {
///     let height = if labels[i] == 0 { 1.0 } else { 2.0 };
///     for j in 0..m {
///         let noise = 0.02 * (((i * 13 + j * 7) % 11) as f64 - 5.0) / 5.0;
///         data[(i, j)] = height * (-((t[j] - 0.5) / 0.15).powi(2)).exp() + noise;
///     }
/// }
/// let mut cfg = LocalDifferenceConfig::default();
/// cfg.n_perm = 199;
/// cfg.max_iter = 5;
/// let res = elastic_local_difference_test(&data, &t, &labels, &cfg).unwrap();
/// assert_eq!(res.group_sizes, [8, 8]);
/// assert_eq!(res.alignment.gammas.nrows(), n);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_local_difference_test(
    data: &FdMatrix,
    argvals: &[f64],
    labels: &[usize],
    config: &LocalDifferenceConfig,
) -> Result<LocalDifferenceResult, FdarError> {
    validate_inputs(data, argvals, labels, config)?;
    let (group_labels, idx_a, idx_b) = split_groups(labels, config)?;

    // Label-free joint alignment: `labels` are not used below except to split
    // the aligned quantities into the two groups.
    let alignment =
        karcher_mean_with_config(data, argvals, config.max_iter, config.tol, &config.warp)?;
    let phase_data = phase_quantity(&alignment.gammas, argvals, config.phase_representation);

    let amplitude = local_test(&alignment.aligned_data, &idx_a, &idx_b, argvals, config, 0)?;
    let phase = local_test(&phase_data, &idx_a, &idx_b, argvals, config, 1)?;

    Ok(LocalDifferenceResult {
        amplitude,
        phase,
        phase_data,
        alignment,
        group_labels,
        group_sizes: [idx_a.len(), idx_b.len()],
    })
}

fn validate_inputs(
    data: &FdMatrix,
    argvals: &[f64],
    labels: &[usize],
    config: &LocalDifferenceConfig,
) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if m < 3 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (matching data columns, at least 3)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if labels.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "labels",
            expected: format!("length {n} (one per curve)"),
            actual: format!("length {}", labels.len()),
        });
    }
    if argvals
        .windows(2)
        .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
    {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must be strictly increasing".to_string(),
        });
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: "must contain only finite values".to_string(),
        });
    }
    validate_config(config)
}

fn validate_config(config: &LocalDifferenceConfig) -> Result<(), FdarError> {
    if !(config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {}", config.alpha),
        });
    }
    if config.max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "must be >= 1".to_string(),
        });
    }
    if let Some(bw) = config.scb_bandwidth {
        if !(bw.is_finite() && bw > 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter: "scb_bandwidth",
                message: format!("must be finite and > 0, got {bw}"),
            });
        }
    }
    config.warp.validate()
}

/// Split curve indices by label; A = smaller label.
fn split_groups(
    labels: &[usize],
    config: &LocalDifferenceConfig,
) -> Result<([usize; 2], Vec<usize>, Vec<usize>), FdarError> {
    let mut distinct: Vec<usize> = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "labels",
            message: format!("need exactly 2 distinct labels, got {}", distinct.len()),
        });
    }
    let idx = |g: usize| -> Vec<usize> { (0..labels.len()).filter(|&i| labels[i] == g).collect() };
    let (idx_a, idx_b) = (idx(distinct[0]), idx(distinct[1]));
    let min_n = if config.method == LocalTestMethod::Scb {
        3
    } else {
        2
    };
    if idx_a.len() < min_n || idx_b.len() < min_n {
        return Err(FdarError::InvalidDimension {
            parameter: "labels",
            expected: format!("at least {min_n} curves per group"),
            actual: format!("{} and {}", idx_a.len(), idx_b.len()),
        });
    }
    Ok(([distinct[0], distinct[1]], idx_a, idx_b))
}

/// `γᵢ(t) − t` or `ψᵢ = √γᵢ'` (warp rescaled to `[0, 1]`) for every curve.
fn phase_quantity(gammas: &FdMatrix, argvals: &[f64], repr: PhaseRepresentation) -> FdMatrix {
    let (n, m) = gammas.shape();
    let mut out = FdMatrix::zeros(n, m);
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let h = 1.0 / (m - 1) as f64;
    for i in 0..n {
        match repr {
            PhaseRepresentation::Displacement => {
                for j in 0..m {
                    out[(i, j)] = gammas[(i, j)] - argvals[j];
                }
            }
            PhaseRepresentation::Psi => {
                let gam01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
                for (j, v) in gam_to_psi(&gam01, h).into_iter().enumerate() {
                    out[(i, j)] = v;
                }
            }
        }
    }
    out
}

/// Whether column `j` of `mat` is (numerically) constant across all rows.
fn degenerate(mat: &FdMatrix, j: usize) -> bool {
    let col = mat.column(j);
    let (lo, hi) = col
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    hi - lo <= 1e-12 * (1.0 + hi.abs().max(lo.abs()))
}

/// Rows `idx` of `mat` as a new matrix.
fn select_rows(mat: &FdMatrix, idx: &[usize]) -> FdMatrix {
    let m = mat.ncols();
    let mut out = FdMatrix::zeros(idx.len(), m);
    for j in 0..m {
        for (r, &i) in idx.iter().enumerate() {
            out[(r, j)] = mat[(i, j)];
        }
    }
    out
}

/// Run the configured local test of A vs B on `quantity`. `seed_offset`
/// decorrelates the amplitude and phase permutations.
fn local_test(
    quantity: &FdMatrix,
    idx_a: &[usize],
    idx_b: &[usize],
    argvals: &[f64],
    config: &LocalDifferenceConfig,
    seed_offset: u64,
) -> Result<LocalDifferenceTest, FdarError> {
    let a = select_rows(quantity, idx_a);
    let b = select_rows(quantity, idx_b);
    match config.method {
        LocalTestMethod::Itp => {
            let cfg = ItpRegionConfig {
                nbasis: config.nbasis,
                n_perm: config.n_perm,
                seed: config.seed.wrapping_add(seed_offset),
                alpha: config.alpha,
                mapping: config.mapping,
            };
            let res = itp_two_pop_regions(&a, &b, argvals, &cfg)?;
            Ok(LocalDifferenceTest {
                regions: res.regions,
                effect: res.mean_difference,
                pointwise_adjusted_pvalues: Some(res.pointwise_adjusted_pvalues),
                itp: Some(res.itp),
                lower: None,
                upper: None,
            })
        }
        LocalTestMethod::Scb => {
            let m = argvals.len();
            let bandwidth = config
                .scb_bandwidth
                .unwrap_or(0.05 * (argvals[m - 1] - argvals[0]));
            let band = scb_two_sample_band(
                &a,
                &b,
                argvals,
                bandwidth,
                config.scb_nb,
                1.0 - config.alpha,
                config.multiplier,
            )?;
            // Grid points where the quantity is identical for all curves
            // (e.g. the warp displacement at the domain end points) carry no
            // information; the band collapses there and must not flag them.
            let significant: Vec<bool> = (0..m)
                .map(|j| (band.lower[j] > 0.0 || band.upper[j] < 0.0) && !degenerate(quantity, j))
                .collect();
            let regions = difference_regions(argvals, &band.estimate, &significant, None)?;
            Ok(LocalDifferenceTest {
                regions,
                effect: band.estimate,
                pointwise_adjusted_pvalues: None,
                itp: None,
                lower: Some(band.lower),
                upper: Some(band.upper),
            })
        }
    }
}

#[cfg(test)]
#[path = "local_difference_tests.rs"]
mod tests;
