//! One-way PERMANOVA (permutational multivariate ANOVA on a distance matrix).
//!
//! Implements Anderson's (2001) distance-based pseudo-F for a one-way layout,
//! with the semantics of R `vegan::adonis2` (single factor):
//!
//! ```text
//! SS_T = (1/N)   Σ_{i<j}          d²_ij
//! SS_W = Σ_g (1/n_g) Σ_{i<j, i,j∈g} d²_ij
//! SS_A = SS_T − SS_W
//! F    = (SS_A / (k − 1)) / (SS_W / (N − k)),     R² = SS_A / SS_T
//! ```
//!
//! The identity `tr(G) = SS_T`, `tr(HGH) = SS_A` (Gower-centred `G = −½ C D² C`)
//! holds for **any** symmetric dissimilarity with zero diagonal, so the
//! statistic is exactly vegan's even for non-Euclidean distances.
//!
//! The permutation p-value is `(#{F* ≥ F − √ε} + 1) / (n_perm + 1)`, the same
//! tie tolerance (`EPS = sqrt(.Machine$double.eps)`) as vegan's `anova.cca`.
//! Permutations go through the crate's authoritative
//! [`crate::permutation_test::permutation_pvalue`] scaffold: every permutation
//! reseeds `StdRng` at `seed + perm`, so the result is deterministic and
//! bit-identical with or without the `parallel` feature.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Permutation count at and above which permutations are dispatched in parallel.
const PERMANOVA_PERM_PARALLEL_THRESHOLD: usize = 200;

/// Tie tolerance on the pseudo-F comparison (vegan `anova.cca`: `sqrt(.Machine$double.eps)`).
const PERMANOVA_TIE_EPS: f64 = 1.490_116_119_384_765_6e-8;

/// Result of a one-way PERMANOVA on a distance matrix.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct PermanovaResult {
    /// Pseudo-F statistic `(SS_A/(k−1)) / (SS_W/(N−k))`.
    ///
    /// `+∞` when `SS_W = 0 < SS_A`; `0` when every distance is zero.
    pub statistic: f64,
    /// Permutation p-value `(#{F* ≥ F − √ε} + 1) / (n_perm + 1)`; `1.0` when `n_perm = 0`.
    pub p_value: f64,
    /// Between-group sum of squares `SS_A`.
    pub ss_between: f64,
    /// Within-group (residual) sum of squares `SS_W`.
    pub ss_within: f64,
    /// Total sum of squares `SS_T`.
    pub ss_total: f64,
    /// Proportion of `SS_T` explained by the grouping, `SS_A / SS_T` (0 when `SS_T = 0`).
    pub r_squared: f64,
    /// Between-group degrees of freedom `k − 1`.
    pub df_between: usize,
    /// Residual degrees of freedom `N − k`.
    pub df_within: usize,
    /// Number of permutations used.
    pub n_perm: usize,
}

/// Map arbitrary group labels to contiguous ids `0..k` (in increasing label order).
///
/// Returns `(ids, k)`.
pub(crate) fn contiguous_labels(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut distinct: Vec<usize> = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let ids = labels
        .iter()
        .map(|l| distinct.binary_search(l).unwrap_or(0))
        .collect();
    (ids, distinct.len())
}

/// Validate `labels` against `n` observations; returns contiguous ids and `k`.
pub(crate) fn validate_group_labels(
    labels: &[usize],
    n: usize,
) -> Result<(Vec<usize>, usize), FdarError> {
    if labels.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "labels",
            expected: format!("{n} labels (one per observation)"),
            actual: format!("{} labels", labels.len()),
        });
    }
    let (ids, k) = contiguous_labels(labels);
    if k < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "labels",
            message: format!("need at least 2 distinct groups, got {k}"),
        });
    }
    if n <= k {
        return Err(FdarError::InvalidParameter {
            parameter: "labels",
            message: format!(
                "need more observations than groups for residual degrees of freedom (N = {n}, k = {k})"
            ),
        });
    }
    Ok((ids, k))
}

/// Pre-processed squared distances (upper triangle, row-major `i < j`).
pub(crate) struct SquaredDistances {
    n: usize,
    /// `d²_ij` for `i < j`, in the order `(0,1), (0,2), …, (0,n−1), (1,2), …`.
    upper: Vec<f64>,
    /// `SS_T` (label-independent).
    ss_total: f64,
}

impl SquaredDistances {
    /// Build from a validated symmetric `n × n` distance matrix.
    pub(crate) fn new(dist: &FdMatrix) -> Self {
        let n = dist.nrows();
        let mut upper = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist[(i, j)];
                upper.push(d * d);
            }
        }
        let ss_total = upper.iter().sum::<f64>() / n as f64;
        Self { n, upper, ss_total }
    }

    /// Within-group sum of squares `SS_W` for contiguous group ids `0..k`.
    fn ss_within(&self, ids: &[usize], k: usize) -> f64 {
        let mut sums = vec![0.0; k];
        let mut sizes = vec![0usize; k];
        for &g in ids {
            sizes[g] += 1;
        }
        let mut idx = 0;
        for i in 0..self.n {
            let gi = ids[i];
            for &gj in &ids[(i + 1)..] {
                if gi == gj {
                    sums[gi] += self.upper[idx];
                }
                idx += 1;
            }
        }
        sums.iter()
            .zip(&sizes)
            .filter(|(_, &s)| s > 0)
            .map(|(&sum, &s)| sum / s as f64)
            .sum()
    }

    /// `(F, SS_A, SS_W)` for contiguous group ids `0..k`.
    pub(crate) fn pseudo_f(&self, ids: &[usize], k: usize) -> (f64, f64, f64) {
        let ss_w = self.ss_within(ids, k);
        let ss_a = self.ss_total - ss_w;
        let df_a = (k - 1) as f64;
        let df_w = (self.n - k) as f64;
        let f = if ss_w > 0.0 {
            (ss_a / df_a) / (ss_w / df_w)
        } else if ss_a > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        (f, ss_a, ss_w)
    }

    /// Full PERMANOVA (statistic + permutation p-value) for contiguous ids.
    pub(crate) fn test(
        &self,
        ids: &[usize],
        k: usize,
        n_perm: usize,
        seed: u64,
    ) -> PermanovaResult {
        let (f_obs, ss_a, ss_w) = self.pseudo_f(ids, k);
        let p_value = if n_perm == 0 {
            1.0
        } else {
            crate::permutation_test::permutation_pvalue(
                f_obs - PERMANOVA_TIE_EPS,
                self.n,
                n_perm,
                seed,
                PERMANOVA_PERM_PARALLEL_THRESHOLD,
                |perm_idx: &[usize]| -> f64 {
                    let perm_ids: Vec<usize> = perm_idx.iter().map(|&i| ids[i]).collect();
                    self.pseudo_f(&perm_ids, k).0
                },
            )
        };
        PermanovaResult {
            statistic: f_obs,
            p_value,
            ss_between: ss_a,
            ss_within: ss_w,
            ss_total: self.ss_total,
            r_squared: if self.ss_total > 0.0 {
                ss_a / self.ss_total
            } else {
                0.0
            },
            df_between: k - 1,
            df_within: self.n - k,
            n_perm,
        }
    }
}

/// Check one off-diagonal pair: finite, non-negative and symmetric.
fn check_distance_pair(i: usize, j: usize, a: f64, b: f64) -> Result<(), FdarError> {
    if !a.is_finite() || a < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "dist",
            message: format!("distances must be finite and >= 0, dist[({i}, {j})] = {a}"),
        });
    }
    if (a - b).abs() > 1e-10 * (1.0 + a.abs()) {
        return Err(FdarError::InvalidParameter {
            parameter: "dist",
            message: format!("matrix is not symmetric at ({i}, {j}): {a} vs {b}"),
        });
    }
    Ok(())
}

/// Validate a distance matrix: square, finite, non-negative, symmetric, zero diagonal.
pub(crate) fn validate_distance_matrix(dist: &FdMatrix) -> Result<(), FdarError> {
    let (n, nc) = dist.shape();
    if n != nc {
        return Err(FdarError::InvalidDimension {
            parameter: "dist",
            expected: "a square distance matrix".to_string(),
            actual: format!("{n} x {nc}"),
        });
    }
    for i in 0..n {
        if dist[(i, i)].abs() > 1e-12 {
            return Err(FdarError::InvalidParameter {
                parameter: "dist",
                message: format!("diagonal must be zero, dist[({i}, {i})] = {}", dist[(i, i)]),
            });
        }
        for j in (i + 1)..n {
            check_distance_pair(i, j, dist[(i, j)], dist[(j, i)])?;
        }
    }
    Ok(())
}

/// One-way PERMANOVA (Anderson 2001) on a distance matrix.
///
/// Tests H0: "the groups do not differ in location (centroid) in the metric
/// space" using the distance-based pseudo-F and a label-permutation p-value.
/// The statistic, sums of squares and R² are those of R `vegan::adonis2(d ~ g)`
/// for a single factor; the permutation p-value uses vegan's tie tolerance
/// (`F* ≥ F − √ε`). Pass `n_perm = 0` to skip the permutations (p-value `1.0`).
///
/// Labels may be any `usize` values (they are mapped to contiguous ids
/// internally). The p-value is reproducible for a fixed `seed` (permutation
/// `k` reseeds `StdRng` at `seed + k`) and identical with or without the
/// `parallel` feature.
///
/// Like every PERMANOVA, the test is sensitive to differences in dispersion as
/// well as location when group sizes are unbalanced (Anderson 2001).
///
/// # Arguments
/// * `dist` — Symmetric `N × N` distance matrix with zero diagonal.
/// * `labels` — Group label per observation (length `N`, at least 2 groups, `N > k`).
/// * `n_perm` — Number of permutations (e.g. 999).
/// * `seed` — Base RNG seed.
///
/// # Errors
/// [`FdarError::InvalidDimension`] for a non-square matrix or a label-length
/// mismatch; [`FdarError::InvalidParameter`] for a non-symmetric / negative /
/// non-finite matrix, a non-zero diagonal, fewer than 2 groups, or `N ≤ k`.
///
/// # Examples
///
/// ```
/// use fdars_core::inference::permanova;
/// use fdars_core::FdMatrix;
///
/// // Two well-separated clusters on the line: {0, 1, 2} and {10, 11, 12}.
/// let x: [f64; 6] = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
/// let mut d = FdMatrix::zeros(6, 6);
/// for i in 0..6 {
///     for j in 0..6 {
///         d[(i, j)] = (x[i] - x[j]).abs();
///     }
/// }
/// let res = permanova(&d, &[0, 0, 0, 1, 1, 1], 199, 7).unwrap();
/// // SS_W = 2 * (1 + 4 + 1) / 3 = 4, SS_T = 154, F = 150 / (4 / 4) = 150.
/// assert!((res.statistic - 150.0).abs() < 1e-10);
/// assert!(res.p_value < 0.15); // only 20 distinct splits exist for 3 + 3
/// ```
#[must_use = "returns the PERMANOVA result; examine the statistic and p-value"]
pub fn permanova(
    dist: &FdMatrix,
    labels: &[usize],
    n_perm: usize,
    seed: u64,
) -> Result<PermanovaResult, FdarError> {
    validate_distance_matrix(dist)?;
    let (ids, k) = validate_group_labels(labels, dist.nrows())?;
    Ok(SquaredDistances::new(dist).test(&ids, k, n_perm, seed))
}

#[cfg(test)]
// Golden literals are R's `%.17g` output, kept verbatim.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// Deterministic 15 × 3 point cloud, identical to the R golden script
    /// `.planning/quick/261126-apa-…/golden_vegan_adonis2.R` (R is 1-based: `i + 1`).
    fn points(offset: f64, labels: &[usize]) -> Vec<[f64; 3]> {
        (0..15)
            .map(|i| {
                let mut p = [0.0; 3];
                for (c, pc) in p.iter_mut().enumerate() {
                    *pc = (0.37 * (i + 1) as f64 * (c + 1) as f64 + c as f64).sin();
                }
                p[0] += offset * labels[i] as f64;
                p
            })
            .collect()
    }

    fn dist_matrix(pts: &[[f64; 3]], manhattan: bool) -> FdMatrix {
        let n = pts.len();
        let mut d = FdMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let v = if manhattan {
                    (0..3).map(|c| (pts[i][c] - pts[j][c]).abs()).sum::<f64>()
                } else {
                    (0..3)
                        .map(|c| (pts[i][c] - pts[j][c]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                d[(i, j)] = v;
            }
        }
        d
    }

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    /// Monte-Carlo tolerance for comparing two permutation p-values: 4 combined
    /// standard errors (ours with `n_ours` perms, reference with `n_ref`).
    fn mc_tol(p: f64, n_ours: usize, n_ref: usize) -> f64 {
        let v = p * (1.0 - p);
        4.0 * (v / n_ours as f64 + v / n_ref as f64).sqrt() + 1.0 / n_ours as f64
    }

    struct Golden {
        ss_between: f64,
        ss_within: f64,
        ss_total: f64,
        r2: f64,
        f: f64,
        p: f64,
    }

    fn check_golden(d: &FdMatrix, labels: &[usize], g: &Golden) {
        let n_perm = 9999;
        let r = permanova(d, labels, n_perm, 2026).unwrap();
        assert!(
            rel_close(r.statistic, g.f, 1e-10),
            "F {} vs {}",
            r.statistic,
            g.f
        );
        assert!(rel_close(r.ss_between, g.ss_between, 1e-10));
        assert!(rel_close(r.ss_within, g.ss_within, 1e-10));
        assert!(rel_close(r.ss_total, g.ss_total, 1e-10));
        assert!(rel_close(r.r_squared, g.r2, 1e-10));
        let tol = mc_tol(g.p, n_perm, 9999);
        assert!(
            (r.p_value - g.p).abs() <= tol,
            "p {} vs vegan {} (tol {tol})",
            r.p_value,
            g.p
        );
    }

    /// R vegan 2.7.8: `adonis2(dist(XA) ~ g, permutations = 9999)` (golden_vegan_adonis2.R, case A).
    #[test]
    fn matches_vegan_adonis2_euclidean_three_groups() {
        let labels: Vec<usize> = (0..15).map(|i| i % 3).collect();
        let d = dist_matrix(&points(1.1, &labels), false);
        check_golden(
            &d,
            &labels,
            &Golden {
                ss_between: 11.162270661903836,
                ss_within: 22.737584375062116,
                ss_total: 33.899855036965953,
                r2: 0.32927192903131847,
                f: 2.9455030431850813,
                p: 0.0251,
            },
        );
    }

    /// R vegan 2.7.8: `adonis2(dist(XB, "manhattan") ~ g, permutations = 9999)` (case B,
    /// unbalanced 6/9, non-Euclidean L1 distance).
    #[test]
    fn matches_vegan_adonis2_manhattan_unbalanced() {
        let labels: Vec<usize> = (0..15).map(|i| usize::from(i >= 6)).collect();
        let d = dist_matrix(&points(0.25, &labels), true);
        check_golden(
            &d,
            &labels,
            &Golden {
                ss_between: 10.442126151699718,
                ss_within: 38.309365782784589,
                ss_total: 48.751491934484307,
                r2: 0.21419090446980749,
                f: 3.5434582953367091,
                p: 0.0211,
            },
        );
    }

    /// R vegan 2.7.8: case C, 4 unbalanced groups, no group effect.
    #[test]
    fn matches_vegan_adonis2_four_groups_null() {
        let labels = [0usize, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 3];
        let d = dist_matrix(&points(0.0, &labels), false);
        check_golden(
            &d,
            &labels,
            &Golden {
                ss_between: 0.60112415149541143,
                ss_within: 22.263235964445389,
                ss_total: 22.864360115940801,
                r2: 0.026290880149159028,
                f: 0.099002763674450267,
                p: 0.9999,
            },
        );
    }

    /// Hand-computed: 1-D points {0,1,2 | 10,11,12}.
    #[test]
    fn hand_computed_two_clusters() {
        let x: [f64; 6] = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        let mut d = FdMatrix::zeros(6, 6);
        for i in 0..6 {
            for j in 0..6 {
                d[(i, j)] = (x[i] - x[j]).abs();
            }
        }
        let r = permanova(&d, &[0, 0, 0, 1, 1, 1], 0, 0).unwrap();
        // Within pairs: (1 + 4 + 1) per group → 12. Between gaps 10,11,12,9,10,11,8,9,10
        // → squares sum 912. Σ_{i<j} d² = 924, SS_T = 924 / 6 = 154, SS_W = 6/3 + 6/3 = 4,
        // F = (154 − 4)/1 / (4/4) = 150.
        assert!((r.ss_total - 154.0).abs() < 1e-12);
        assert!((r.ss_within - 4.0).abs() < 1e-12);
        assert!((r.statistic - 150.0).abs() < 1e-10);
        assert_eq!(r.p_value, 1.0);
        assert_eq!((r.df_between, r.df_within), (1, 4));
    }

    /// Relabelling groups and permuting observations leave F and p unchanged
    /// (p up to the different permutation draws → only F is compared exactly).
    #[test]
    fn invariant_to_label_values_and_observation_order() {
        let labels: Vec<usize> = (0..15).map(|i| i % 3).collect();
        let d = dist_matrix(&points(1.1, &labels), false);
        let base = permanova(&d, &labels, 499, 3).unwrap();
        // Arbitrary, non-contiguous label values with the same partition.
        let renamed: Vec<usize> = labels.iter().map(|&g| [17, 4, 99][g]).collect();
        let r2 = permanova(&d, &renamed, 499, 3).unwrap();
        assert!(rel_close(r2.statistic, base.statistic, 1e-12));
        // Reverse the observation order.
        let n = 15;
        let mut d_rev = FdMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                d_rev[(i, j)] = d[(n - 1 - i, n - 1 - j)];
            }
        }
        let lab_rev: Vec<usize> = labels.iter().rev().copied().collect();
        let r3 = permanova(&d_rev, &lab_rev, 499, 3).unwrap();
        assert!(rel_close(r3.statistic, base.statistic, 1e-12));
    }

    /// Deterministic for a fixed seed; the observed labelling counts itself (p ≥ 1/(n_perm+1)).
    #[test]
    fn deterministic_and_bounded() {
        let labels: Vec<usize> = (0..15).map(|i| i % 3).collect();
        let d = dist_matrix(&points(5.0, &labels), false);
        let a = permanova(&d, &labels, 300, 11).unwrap();
        let b = permanova(&d, &labels, 300, 11).unwrap();
        assert_eq!(a, b);
        assert!(a.p_value >= 1.0 / 301.0 && a.p_value <= 0.01);
    }

    /// Degenerate all-zero distances: F = 0, p = 1.
    #[test]
    fn all_zero_distances() {
        let d = FdMatrix::zeros(6, 6);
        let r = permanova(&d, &[0, 0, 1, 1, 2, 2], 99, 1).unwrap();
        assert_eq!(r.statistic, 0.0);
        assert_eq!(r.p_value, 1.0);
        assert_eq!(r.r_squared, 0.0);
    }

    #[test]
    fn input_validation() {
        let d = FdMatrix::zeros(4, 4);
        assert!(permanova(&d, &[0, 0, 0, 0], 9, 0).is_err()); // one group
        assert!(permanova(&d, &[0, 1, 2], 9, 0).is_err()); // length
        assert!(permanova(&d, &[0, 1, 2, 3], 9, 0).is_err()); // N <= k
        assert!(permanova(&FdMatrix::zeros(3, 4), &[0, 1, 1], 9, 0).is_err());
        let mut asym = FdMatrix::zeros(4, 4);
        asym[(0, 1)] = 1.0;
        assert!(permanova(&asym, &[0, 0, 1, 1], 9, 0).is_err());
        let mut neg = FdMatrix::zeros(4, 4);
        neg[(0, 1)] = -1.0;
        neg[(1, 0)] = -1.0;
        assert!(permanova(&neg, &[0, 0, 1, 1], 9, 0).is_err());
    }

    /// Size under H0 on exchangeable data: rejection rate at α = 0.05 within a
    /// 4-sigma binomial band (fixed seeds, 400 replicates).
    #[test]
    fn size_under_null() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let reps = 400;
        let mut rng = StdRng::seed_from_u64(123);
        let mut rejections = 0usize;
        let labels: Vec<usize> = (0..18).map(|i| i % 3).collect();
        for r in 0..reps {
            let pts: Vec<[f64; 3]> = (0..18)
                .map(|_| [rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()])
                .collect();
            let d = dist_matrix(&pts, false);
            let res = permanova(&d, &labels, 199, r as u64).unwrap();
            if res.p_value <= 0.05 {
                rejections += 1;
            }
        }
        let rate = rejections as f64 / reps as f64;
        let sd = (0.05 * 0.95 / reps as f64).sqrt();
        assert!((rate - 0.05).abs() <= 4.0 * sd, "size {rate}");
    }
}
