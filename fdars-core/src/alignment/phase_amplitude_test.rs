//! Permutation tests of group differences in amplitude and in phase (GH #62, ALN-PHTEST).
//!
//! Given curves with group labels, all curves are aligned **jointly and
//! label-free** to their elastic Karcher mean. This splits each curve into an
//! amplitude part (its SRSF after alignment, an element of the quotient space
//! `L²/Γ`) and a phase part (its warp `γᵢ`, a point on the Hilbert sphere via
//! `ψᵢ = √γᵢ'`). Distance matrices are built separately for the two parts and a
//! one-way PERMANOVA pseudo-F with a seeded permutation p-value is computed on
//! each ([`crate::inference::permanova`]):
//!
//! * **amplitude** — the exact pairwise elastic distance
//!   `inf_γ ‖qᵢ − (qⱼ∘γ)√γ'‖` ([`AmplitudeMetric::Pairwise`], default), or the
//!   faster `‖q̃ᵢ − q̃ⱼ‖_{L²}` between the SRSFs aligned to the common template
//!   ([`AmplitudeMetric::TemplateAligned`]);
//! * **phase** — the Fisher–Rao geodesic `d_P(i, j) = arccos⟨ψᵢ, ψⱼ⟩` between
//!   the warps to the common template;
//! * **joint** (optional) — `d_J² = d_A² / mean(d_A²) + d_P² / mean(d_P²)`, a
//!   scale-free combination (each part normalised by its mean off-diagonal
//!   squared distance; a part with all-zero distances is dropped).
//!
//! # Why aligning before permuting is valid
//!
//! The alignment (Karcher mean, warps, aligned SRSFs) and therefore both
//! distance matrices are functions of the **unlabelled** sample only. Under
//! H0 the labels are exchangeable with respect to the curves, so permuting the
//! labels while holding the distance matrices fixed generates the exact
//! conditional null distribution of the pseudo-F — no re-alignment per
//! permutation is needed, and the permutation p-value is exact (up to Monte-Carlo
//! error in the number of permutations). This would **not** hold if the
//! template were estimated per group, or if labels influenced the alignment in
//! any way — so the alignment here never sees `labels`.
//!
//! # Interpretation
//!
//! As with any PERMANOVA, a rejection means the groups differ in their
//! distribution in that metric space (location, and — with unbalanced groups —
//! possibly dispersion). The amplitude and phase tests share the same
//! permutations (same `seed`), so their p-values are coupled; no multiplicity
//! correction is applied.

use super::karcher::karcher_mean;
use super::pairwise::elastic_self_distance_matrix;
use super::srsf::srsf_transform;
use super::KarcherMeanResult;
use crate::error::FdarError;
use crate::helpers::{l2_distance, simpsons_weights};
use crate::inference::permanova::{validate_group_labels, PermanovaResult, SquaredDistances};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::warping::{gam_to_psi, inner_product_l2, l2_norm_l2};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// How the amplitude distance between two curves is measured.
///
/// Both choices are label-free, so both give an exact permutation test of the
/// full null "no group difference at all". They differ in how well they isolate
/// amplitude from phase when **only the phase** differs between groups:
/// residual misalignment of the discretised template alignment is
/// group-structured in that case and leaks into the template-aligned distances
/// (in a 60-replicate simulation with a ±1 warp-parameter shift between groups
/// of 10 curves on 120 grid points, `TemplateAligned` rejected the true
/// amplitude null at α = 0.05 in 4/60 replicates with mean p ≈ 0.40, while
/// `Pairwise` gave 3/60 with mean p ≈ 0.56 on 60 points). `Pairwise` is therefore
/// the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AmplitudeMetric {
    /// Exact pairwise elastic (amplitude) distance
    /// `inf_γ ‖qᵢ − (qⱼ∘γ)√γ'‖`, minimising over the relative warp of every
    /// pair ([`super::elastic_self_distance_matrix`]; `O(n²)` DP solves).
    #[default]
    Pairwise,
    /// L² distance between the SRSFs of the curves aligned to the common
    /// Karcher template (no extra DP solves beyond the Karcher mean). Faster,
    /// but see the caveat above.
    TemplateAligned,
}

/// Configuration for [`elastic_phase_amplitude_test`].
///
/// Construct with `PhaseAmplitudeTestConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PhaseAmplitudeTestConfig {
    /// Number of label permutations (default 999). `0` skips the permutations
    /// (p-values are then `1.0`).
    pub n_perm: usize,
    /// Base RNG seed; permutation `k` uses `seed + k` (default 42).
    pub seed: u64,
    /// Warp-roughness penalty λ of the elastic alignment (default 0.0).
    pub lambda: f64,
    /// Maximum Karcher-mean iterations (default 20).
    pub max_iter: usize,
    /// Karcher-mean convergence tolerance (default 1e-4).
    pub tol: f64,
    /// Amplitude distance (default [`AmplitudeMetric::Pairwise`]).
    pub amplitude_metric: AmplitudeMetric,
    /// Also run the joint amplitude + phase test (default `true`).
    pub joint: bool,
}

impl Default for PhaseAmplitudeTestConfig {
    fn default() -> Self {
        Self {
            n_perm: 999,
            seed: 42,
            lambda: 0.0,
            max_iter: 20,
            tol: 1e-4,
            amplitude_metric: AmplitudeMetric::Pairwise,
            joint: true,
        }
    }
}

/// Result of [`elastic_phase_amplitude_test`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PhaseAmplitudeTestResult {
    /// PERMANOVA on the amplitude distances.
    pub amplitude: PermanovaResult,
    /// PERMANOVA on the phase (Fisher–Rao warp) distances.
    pub phase: PermanovaResult,
    /// PERMANOVA on the joint distance (when `config.joint`).
    pub joint: Option<PermanovaResult>,
    /// Amplitude distance matrix (n × n).
    pub amplitude_distances: FdMatrix,
    /// Phase distance matrix (n × n), in radians on the Hilbert sphere (`[0, π/2]`).
    pub phase_distances: FdMatrix,
    /// The label-free joint alignment (Karcher mean, warps, aligned curves).
    pub alignment: KarcherMeanResult,
    /// Distinct group labels, in increasing order (group `g` of the tests is `group_labels[g]`).
    pub group_labels: Vec<usize>,
    /// Number of curves per group (same order as `group_labels`).
    pub group_sizes: Vec<usize>,
}

/// Validate data / argvals / config shape.
fn validate_inputs(
    data: &FdMatrix,
    argvals: &[f64],
    config: &PhaseAmplitudeTestConfig,
) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 curves".to_string(),
            actual: format!("{n} curves"),
        });
    }
    if m < 3 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (matching data columns, at least 3)"),
            actual: format!("length {}", argvals.len()),
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
    if !config.lambda.is_finite() || config.lambda < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("must be finite and >= 0, got {}", config.lambda),
        });
    }
    if config.max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "must be >= 1".to_string(),
        });
    }
    Ok(())
}

/// Fill a symmetric matrix from an upper-triangle pair function (parallel over rows).
fn symmetric_from_pairs<F>(n: usize, pair: F) -> FdMatrix
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| ((i + 1)..n).map(|j| pair(i, j)).collect())
        .collect();
    let mut d = FdMatrix::zeros(n, n);
    for (i, row) in rows.iter().enumerate() {
        for (off, &v) in row.iter().enumerate() {
            let j = i + 1 + off;
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
    d
}

/// L² distances between the SRSFs of the template-aligned curves.
fn template_amplitude_distances(aligned: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let q = srsf_transform(aligned, argvals);
    let weights = simpsons_weights(argvals);
    let rows: Vec<Vec<f64>> = (0..q.nrows()).map(|i| q.row(i)).collect();
    symmetric_from_pairs(rows.len(), |i, j| l2_distance(&rows[i], &rows[j], &weights))
}

/// Unit-norm `ψᵢ = √γᵢ'` of every warp, on the normalised `[0, 1]` grid.
fn unit_psis(gammas: &FdMatrix, argvals: &[f64], time: &[f64]) -> Vec<Vec<f64>> {
    let (n, m) = gammas.shape();
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let h = 1.0 / (m - 1) as f64;
    (0..n)
        .map(|i| {
            let gam01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
            let psi = gam_to_psi(&gam01, h);
            let norm = l2_norm_l2(&psi, time);
            if norm > 1e-12 {
                psi.iter().map(|&p| p / norm).collect()
            } else {
                vec![1.0; m]
            }
        })
        .collect()
}

/// Fisher–Rao (Hilbert-sphere geodesic) distances between the warps.
fn phase_distances(gammas: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let m = gammas.ncols();
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let psis = unit_psis(gammas, argvals, &time);
    symmetric_from_pairs(psis.len(), |i, j| {
        inner_product_l2(&psis[i], &psis[j], &time)
            .clamp(-1.0, 1.0)
            .acos()
    })
}

/// Mean off-diagonal squared distance.
fn mean_sq_offdiag(d: &FdMatrix) -> f64 {
    let n = d.nrows();
    let mut s = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            s += d[(i, j)] * d[(i, j)];
        }
    }
    s / (n * (n - 1) / 2) as f64
}

/// Scale-free joint distance `√(d_A²/mean(d_A²) + d_P²/mean(d_P²))`, dropping
/// an all-zero part.
fn joint_distances(amp: &FdMatrix, phase: &FdMatrix) -> FdMatrix {
    let n = amp.nrows();
    let sa = mean_sq_offdiag(amp);
    let sp = mean_sq_offdiag(phase);
    let wa = if sa > 0.0 { 1.0 / sa } else { 0.0 };
    let wp = if sp > 0.0 { 1.0 / sp } else { 0.0 };
    let mut d = FdMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let (a, p) = (amp[(i, j)], phase[(i, j)]);
            d[(i, j)] = (wa * a * a + wp * p * p).sqrt();
        }
    }
    d
}

/// Permutation tests of group differences in **amplitude** and in **phase**
/// in the elastic quotient space.
///
/// All curves are aligned jointly to their elastic Karcher mean **without
/// using the labels**; then a one-way PERMANOVA (pseudo-F, R `vegan::adonis2`
/// semantics) with a seeded label-permutation p-value is computed separately
/// on the amplitude distances, the phase (Fisher–Rao warp) distances and —
/// optionally — a scale-free joint distance. See the module
/// docs for the distances and for why label-free alignment makes the
/// permutation test exact under H0.
///
/// The three tests use the same `seed`, i.e. the same permutations. Permutation
/// `k` reseeds `StdRng` at `seed + k`, so results are deterministic and
/// identical with or without the `parallel` feature. The phase distances assume
/// an (approximately) uniform `argvals` grid, like [`crate::warping::phase_distance`].
///
/// # Arguments
/// * `data` — Curves (n × m), one per row.
/// * `argvals` — Strictly increasing evaluation points (length m ≥ 3).
/// * `labels` — Group label per curve (any `usize` values; at least 2 groups, `n > k`).
/// * `config` — [`PhaseAmplitudeTestConfig`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] for fewer than 3 curves or an `argvals` /
/// `labels` length mismatch; [`FdarError::InvalidParameter`] for non-increasing
/// `argvals`, non-finite data, a negative λ, `max_iter = 0`, fewer than two
/// groups or `n ≤ k`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{elastic_phase_amplitude_test, PhaseAmplitudeTestConfig};
/// use fdars_core::FdMatrix;
///
/// let m = 40;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// // Two groups with the same bump shape; group 1 is warped later in time.
/// let n = 12;
/// let mut data = FdMatrix::zeros(n, m);
/// let mut labels = vec![0usize; n];
/// for i in 0..n {
///     let g = i % 2;
///     labels[i] = g;
///     let s = if g == 0 { -1.5 } else { 1.5 } + 0.1 * (i as f64 - 6.0) / 6.0;
///     let amp = 1.0 + 0.05 * ((i * 7) % 5) as f64;
///     for j in 0..m {
///         let w = ((s * t[j]).exp() - 1.0) / (s.exp() - 1.0); // monotone warp
///         data[(i, j)] = amp * (-((w - 0.5) / 0.12).powi(2) / 2.0).exp();
///     }
/// }
/// let mut cfg = PhaseAmplitudeTestConfig::default();
/// cfg.n_perm = 199;
/// let res = elastic_phase_amplitude_test(&data, &t, &labels, &cfg).unwrap();
/// assert!(res.phase.p_value < 0.05);
/// assert_eq!(res.amplitude_distances.nrows(), n);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_phase_amplitude_test(
    data: &FdMatrix,
    argvals: &[f64],
    labels: &[usize],
    config: &PhaseAmplitudeTestConfig,
) -> Result<PhaseAmplitudeTestResult, FdarError> {
    validate_inputs(data, argvals, config)?;
    let (ids, k) = validate_group_labels(labels, data.nrows())?;

    // Label-free joint alignment: labels are never passed below this line
    // except to the permutation tests.
    let alignment = karcher_mean(data, argvals, config.max_iter, config.tol, config.lambda);
    let amplitude_distances = match config.amplitude_metric {
        AmplitudeMetric::TemplateAligned => {
            template_amplitude_distances(&alignment.aligned_data, argvals)
        }
        AmplitudeMetric::Pairwise => elastic_self_distance_matrix(data, argvals, config.lambda),
    };
    let phase_distances = phase_distances(&alignment.gammas, argvals);

    let run = |d: &FdMatrix| SquaredDistances::new(d).test(&ids, k, config.n_perm, config.seed);
    let amplitude = run(&amplitude_distances);
    let phase = run(&phase_distances);
    let joint = config
        .joint
        .then(|| run(&joint_distances(&amplitude_distances, &phase_distances)));

    let mut group_labels: Vec<usize> = labels.to_vec();
    group_labels.sort_unstable();
    group_labels.dedup();
    let mut group_sizes = vec![0usize; k];
    for &g in &ids {
        group_sizes[g] += 1;
    }

    Ok(PhaseAmplitudeTestResult {
        amplitude,
        phase,
        joint,
        amplitude_distances,
        phase_distances,
        alignment,
        group_labels,
        group_sizes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::permanova;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// Two-bump template.
    fn template(t: f64) -> f64 {
        (-((t - 0.35) / 0.1).powi(2) / 2.0).exp() + 0.7 * (-((t - 0.68) / 0.09).powi(2) / 2.0).exp()
    }

    /// Monotone warp `γ_s(t) = (e^{st} − 1)/(e^s − 1)` (identity at s = 0).
    fn warp(s: f64, t: f64) -> f64 {
        if s.abs() < 1e-12 {
            t
        } else {
            ((s * t).exp() - 1.0) / (s.exp() - 1.0)
        }
    }

    /// Curves `a_i · template(γ_{s_i}(t))` with group-specific centres for the
    /// warp parameter `s` and the amplitude `a`.
    fn simulate(
        rng: &mut StdRng,
        labels: &[usize],
        m: usize,
        s_centre: [f64; 2],
        a_centre: [f64; 2],
    ) -> FdMatrix {
        let t = grid(m);
        let n = labels.len();
        let mut data = FdMatrix::zeros(n, m);
        for (i, &g) in labels.iter().enumerate() {
            let s = s_centre[g] + rng.gen_range(-0.6..0.6);
            let a = a_centre[g] + rng.gen_range(-0.25..0.25);
            for j in 0..m {
                data[(i, j)] = a * template(warp(s, t[j]));
            }
        }
        data
    }

    fn cfg(n_perm: usize) -> PhaseAmplitudeTestConfig {
        PhaseAmplitudeTestConfig {
            n_perm,
            max_iter: 10,
            ..Default::default()
        }
    }

    fn two_groups(n: usize) -> Vec<usize> {
        (0..n).map(|i| i % 2).collect()
    }

    /// Phase-only difference: phase test rejects, amplitude test does not.
    #[test]
    fn phase_only_difference_detected_by_phase_test_only() {
        let labels = two_groups(20);
        let mut rng = StdRng::seed_from_u64(5);
        let data = simulate(&mut rng, &labels, 60, [-1.2, 1.2], [1.0, 1.0]);
        let r = elastic_phase_amplitude_test(&data, &grid(60), &labels, &cfg(199)).unwrap();
        assert!(r.phase.p_value <= 0.01, "phase p {}", r.phase.p_value);
        assert!(
            r.amplitude.p_value > 0.05,
            "amplitude p {}",
            r.amplitude.p_value
        );
        assert!(r.joint.unwrap().p_value <= 0.05);
    }

    /// Amplitude-only difference (scaling): amplitude test rejects, phase test does not.
    #[test]
    fn amplitude_only_difference_detected_by_amplitude_test_only() {
        let labels = two_groups(20);
        let mut rng = StdRng::seed_from_u64(9);
        let data = simulate(&mut rng, &labels, 60, [0.0, 0.0], [1.0, 1.6]);
        let r = elastic_phase_amplitude_test(&data, &grid(60), &labels, &cfg(199)).unwrap();
        assert!(
            r.amplitude.p_value <= 0.01,
            "amplitude p {}",
            r.amplitude.p_value
        );
        assert!(r.phase.p_value > 0.05, "phase p {}", r.phase.p_value);
        assert!(r.joint.unwrap().p_value <= 0.05);
    }

    /// The pairwise amplitude metric also detects an amplitude-only difference
    /// and gives a symmetric, zero-diagonal matrix.
    #[test]
    fn template_aligned_amplitude_metric() {
        let labels = two_groups(12);
        let mut rng = StdRng::seed_from_u64(9);
        let data = simulate(&mut rng, &labels, 30, [0.0, 0.0], [1.0, 1.8]);
        let mut c = cfg(99);
        c.amplitude_metric = AmplitudeMetric::TemplateAligned;
        c.joint = false;
        let r = elastic_phase_amplitude_test(&data, &grid(30), &labels, &c).unwrap();
        assert!(r.joint.is_none());
        assert!(
            r.amplitude.p_value <= 0.05,
            "amplitude p {}",
            r.amplitude.p_value
        );
        let d = &r.amplitude_distances;
        for i in 0..12 {
            assert_eq!(d[(i, i)], 0.0);
            for j in 0..12 {
                assert_eq!(d[(i, j)], d[(j, i)]);
            }
        }
    }

    /// Monte-Carlo size under H0 (same distribution in both groups): rejection
    /// rates at α = 0.05 lie within a 3-sigma binomial band. Exactness follows
    /// from label-free alignment; this checks the implementation honours it.
    #[test]
    fn size_under_null_monte_carlo() {
        let reps = 100;
        let labels = two_groups(12);
        let t = grid(30);
        let mut rng = StdRng::seed_from_u64(2026);
        let (mut rej_a, mut rej_t, mut rej_p, mut rej_j) = (0usize, 0usize, 0usize, 0usize);
        for r in 0..reps {
            let data = simulate(&mut rng, &labels, 30, [0.0, 0.0], [1.0, 1.0]);
            let mut c = cfg(99);
            c.seed = 1000 + r as u64;
            let res = elastic_phase_amplitude_test(&data, &t, &labels, &c).unwrap();
            rej_a += usize::from(res.amplitude.p_value <= 0.05);
            rej_p += usize::from(res.phase.p_value <= 0.05);
            rej_j += usize::from(res.joint.unwrap().p_value <= 0.05);
            c.amplitude_metric = AmplitudeMetric::TemplateAligned;
            c.joint = false;
            let res_t = elastic_phase_amplitude_test(&data, &t, &labels, &c).unwrap();
            rej_t += usize::from(res_t.amplitude.p_value <= 0.05);
        }
        let sd = (0.05 * 0.95 / reps as f64).sqrt();
        for (name, rej) in [
            ("amplitude (pairwise)", rej_a),
            ("amplitude (template)", rej_t),
            ("phase", rej_p),
            ("joint", rej_j),
        ] {
            let rate = rej as f64 / reps as f64;
            assert!((rate - 0.05).abs() <= 3.0 * sd, "{name} size {rate}");
        }
    }

    /// The statistics equal a standalone PERMANOVA on the returned distance
    /// matrices; distances are symmetric with zero diagonal, phase in [0, π/2].
    #[test]
    fn consistent_with_standalone_permanova_and_well_formed() {
        let labels: Vec<usize> = (0..15).map(|i| i % 3).collect();
        let mut rng = StdRng::seed_from_u64(1);
        let data = simulate(
            &mut rng,
            &labels[..].iter().map(|&g| g.min(1)).collect::<Vec<_>>(),
            40,
            [-0.5, 0.5],
            [1.0, 1.2],
        );
        let r = elastic_phase_amplitude_test(&data, &grid(40), &labels, &cfg(149)).unwrap();
        let pa = permanova(&r.amplitude_distances, &labels, 149, 42).unwrap();
        let pp = permanova(&r.phase_distances, &labels, 149, 42).unwrap();
        assert_eq!(pa, r.amplitude);
        assert_eq!(pp, r.phase);
        assert_eq!(r.group_labels, vec![0, 1, 2]);
        assert_eq!(r.group_sizes, vec![5, 5, 5]);
        for i in 0..15 {
            assert_eq!(r.phase_distances[(i, i)], 0.0);
            for j in 0..15 {
                let d = r.phase_distances[(i, j)];
                assert_eq!(d, r.phase_distances[(j, i)]);
                assert!((0.0..=std::f64::consts::FRAC_PI_2 + 1e-12).contains(&d));
            }
        }
    }

    /// Phase distance between two warps equals the closed-form sphere geodesic:
    /// for γ(t) = t^2 and the identity, ⟨ψ, 1⟩ = ∫ √(2t) dt = 2√2/3.
    #[test]
    fn phase_distance_closed_form() {
        let m = 2001;
        let t = grid(m);
        let mut g = FdMatrix::zeros(2, m);
        for j in 0..m {
            g[(0, j)] = t[j];
            g[(1, j)] = t[j] * t[j];
        }
        let d = phase_distances(&g, &t);
        let expected = (2.0 * 2.0_f64.sqrt() / 3.0).acos();
        assert!(
            (d[(0, 1)] - expected).abs() < 2e-3,
            "{} vs {expected}",
            d[(0, 1)]
        );
    }

    /// Deterministic for a fixed seed; labels do not affect the alignment.
    #[test]
    fn deterministic_and_alignment_label_free() {
        let labels = two_groups(10);
        let mut rng = StdRng::seed_from_u64(3);
        let data = simulate(&mut rng, &labels, 30, [-1.0, 1.0], [1.0, 1.0]);
        let t = grid(30);
        let a = elastic_phase_amplitude_test(&data, &t, &labels, &cfg(99)).unwrap();
        let b = elastic_phase_amplitude_test(&data, &t, &labels, &cfg(99)).unwrap();
        assert_eq!(a, b);
        let other: Vec<usize> = (0..10).map(|i| usize::from(i < 5)).collect();
        let c = elastic_phase_amplitude_test(&data, &t, &other, &cfg(99)).unwrap();
        assert_eq!(a.alignment, c.alignment);
        assert_eq!(a.amplitude_distances, c.amplitude_distances);
        assert_eq!(a.phase_distances, c.phase_distances);
    }

    #[test]
    fn input_validation() {
        let t = grid(20);
        let data = FdMatrix::zeros(6, 20);
        let c = cfg(9);
        assert!(elastic_phase_amplitude_test(&data, &t, &[0; 6], &c).is_err());
        assert!(elastic_phase_amplitude_test(&data, &t, &[0, 1, 0], &c).is_err());
        assert!(elastic_phase_amplitude_test(&data, &t[..19], &[0, 1, 0, 1, 0, 1], &c).is_err());
        let mut bad = c.clone();
        bad.lambda = -1.0;
        assert!(elastic_phase_amplitude_test(&data, &t, &[0, 1, 0, 1, 0, 1], &bad).is_err());
        let mut nan = data.clone();
        nan[(0, 0)] = f64::NAN;
        assert!(elastic_phase_amplitude_test(&nan, &t, &[0, 1, 0, 1, 0, 1], &c).is_err());
    }
}
