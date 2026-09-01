//! Asymptotic one-way functional ANOVA via the V-statistic.
//!
//! [`oneway_anova_vstat`] is the asymptotic counterpart to the existing
//! permutation-based [`crate::function_on_scalar::fanova`]. It computes the
//! functional one-way ANOVA V-statistic
//!
//! ```text
//! V = ∫ Σ_g n_g (x̄_g(t) − x̄(t))² dt
//! ```
//!
//! (Simpson-weighted over the grid) and returns an asymptotic p-value via a
//! scaled-χ² (Satterthwaite/Box) approximation of the V-null. `fanova` is left
//! completely unchanged; this function is added alongside it.

use super::dist::chi_square_sf_df;
use super::TestResult;
use crate::error::FdarError;
use crate::function_on_scalar::compute_group_means;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

/// Asymptotic one-way functional ANOVA test (V-statistic).
///
/// Tests the null hypothesis H0 that all groups share a common mean curve. The
/// test statistic is the Simpson-integrated between-group sum of squares
///
/// ```text
/// V = ∫ Σ_g n_g (x̄_g(t) − x̄(t))² dt
/// ```
///
/// where `x̄_g(t)` is the group-`g` mean curve, `x̄(t)` is the overall mean
/// curve, and `n_g` is the group size. Larger `V` is stronger evidence against
/// H0.
///
/// # Asymptotic p-value (scaled-χ² approximation)
///
/// Under H0 the V-null is a weighted sum of χ² variables whose exact
/// distribution depends on the (unknown) functional covariance. Following the
/// standard Box/Satterthwaite approach, this is approximated by a single scaled
/// chi-square `V ≈ β · χ²_d`, matching the first two moments of the null:
///
/// ```text
/// E[V] = (k − 1) · A,      Var[V] = 2 (k − 1) · B
/// β = B / A,               d = (k − 1) · A² / B
/// ```
///
/// where, with `w(t)` the Simpson weights and `Ĉ(t) = ∫-weighted pooled
/// within-group variance at t`,
///
/// ```text
/// A = Σ_t w(t) Ĉ(t)                  (≈ E of the per-df between-group SS)
/// B = 2 Σ_t Σ_s w(t) w(s) Ĉ(t,s)²   (approximated diagonally as Σ_t w(t)² Ĉ(t)²).
/// ```
///
/// The p-value is `P(χ²_d > V / β)` via the (crate-internal) real-df χ²
/// survival function. This is an
/// approximation; for exact size control the permutation form
/// [`crate::function_on_scalar::fanova`] remains available and this function
/// is designed to agree with it in reject/accept direction.
///
/// Returns a [`TestResult`] with `statistic = V`, the asymptotic `p_value`, and
/// `n_perm = 0`.
///
/// # Errors
///
/// Mirrors [`fanova`](crate::function_on_scalar::fanova)'s guards. Returns
/// [`FdarError::InvalidDimension`] if `data` has zero columns, `groups.len()`
/// does not match the number of rows, `argvals.len()` does not match the
/// column count, or `n < 3`. Returns [`FdarError::InvalidParameter`] if fewer
/// than 2 distinct groups are present.
pub fn oneway_anova_vstat(
    data: &FdMatrix,
    groups: &[usize],
    argvals: &[f64],
) -> Result<TestResult, FdarError> {
    let (n, m) = data.shape();
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (grid points)".to_string(),
            actual: "0 columns".to_string(),
        });
    }
    if groups.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "groups",
            expected: format!("{n} elements (matching data rows)"),
            actual: format!("{} elements", groups.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 observations".to_string(),
            actual: format!("{n} observations"),
        });
    }

    let mut labels: Vec<usize> = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let k = labels.len();
    if k < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "groups",
            message: format!("at least 2 distinct groups required, but only {k} found"),
        });
    }

    // Group means (k x m) + overall mean (m) via the shared helper.
    let (group_means, overall_mean) = compute_group_means(data, groups, &labels);

    // Group counts (aligned with `labels`).
    let mut counts = vec![0usize; k];
    for &g in groups {
        let idx = labels.iter().position(|&l| l == g).unwrap_or(0);
        counts[idx] += 1;
    }

    let weights = simpsons_weights(argvals);

    // Between-group weighted sum of squares → V.
    // V = ∫ Σ_g n_g (x̄_g(t) − x̄(t))² dt.
    let mut v_stat = 0.0;
    for t in 0..m {
        let mut between_t = 0.0;
        for g in 0..k {
            let d = group_means[(g, t)] - overall_mean[t];
            between_t += counts[g] as f64 * d * d;
        }
        v_stat += weights[t] * between_t;
    }

    // Pooled within-group variance per grid point Ĉ(t):
    // Ĉ(t) = Σ_i (x_i(t) − x̄_{g(i)}(t))² / (n − k).
    let df_within = (n as f64 - k as f64).max(1.0);
    let mut cov_diag = vec![0.0f64; m];
    for t in 0..m {
        let mut ss = 0.0;
        for i in 0..n {
            let g = labels.iter().position(|&l| l == groups[i]).unwrap_or(0);
            let d = data[(i, t)] - group_means[(g, t)];
            ss += d * d;
        }
        cov_diag[t] = ss / df_within;
    }

    // Scaled-χ² (Box/Satterthwaite) moment match on the between-group V-null.
    // Per-df between-group SS at t has expectation (k−1)·Ĉ(t); integrating with
    // the Simpson weights:
    //   A = Σ_t w(t) Ĉ(t)      (so E[V] = (k−1)·A)
    //   B = Σ_t w(t)² Ĉ(t)²    (diagonal approximation of the covariance term,
    //                            so Var[V] = 2 (k−1)·B)
    let a: f64 = (0..m).map(|t| weights[t] * cov_diag[t]).sum();
    let b: f64 = (0..m).map(|t| (weights[t] * cov_diag[t]).powi(2)).sum();

    let p_value = if a <= 1e-30 || b <= 1e-30 {
        // No within-group variation to calibrate against: any positive V is
        // decisive, V == 0 means identical group means.
        if v_stat > 1e-30 {
            0.0
        } else {
            1.0
        }
    } else {
        let beta = b / a;
        let d = (k as f64 - 1.0) * a * a / b;
        chi_square_sf_df(v_stat / beta, d)
    };

    Ok(TestResult {
        statistic: v_stat,
        p_value,
        n_perm: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(deprecated)]
    // import of the deprecated `fanova`; the tests below pin its old behavior
    use crate::function_on_scalar::fanova;
    use crate::test_helpers::uniform_grid;

    /// Deterministic pseudo-noise in [-1, 1].
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let z = (*seed >> 33) as f64 / (1u64 << 31) as f64;
        z - 1.0
    }

    /// Build a grouped dataset: each group gets a mean shift `shifts[g]`.
    fn make_grouped(
        per_group: usize,
        argvals: &[f64],
        shifts: &[f64],
        seed: u64,
    ) -> (FdMatrix, Vec<usize>) {
        let k = shifts.len();
        let n = per_group * k;
        let m = argvals.len();
        let mut mat = FdMatrix::zeros(n, m);
        let mut groups = vec![0usize; n];
        let mut s = seed;
        let mut row = 0;
        for (g, &shift) in shifts.iter().enumerate() {
            for _ in 0..per_group {
                for (j, &t) in argvals.iter().enumerate() {
                    let base = (2.0 * std::f64::consts::PI * t).sin();
                    mat[(row, j)] = base + shift + 0.2 * noise(&mut s);
                }
                groups[row] = g;
                row += 1;
            }
        }
        (mat, groups)
    }

    // Cross-checks the OLD (deprecated) `fanova` decision vs vstat — pins the old path on purpose.
    #[allow(deprecated)]
    #[test]
    fn vstat_rejects_separated_groups_agrees_with_fanova() {
        let argvals = uniform_grid(30);
        // Clearly-separated group means.
        let (data, groups) = make_grouped(20, &argvals, &[0.0, 1.5, 3.0], 101);
        let res = oneway_anova_vstat(&data, &groups, &argvals).unwrap();
        assert!(
            res.p_value < 0.05,
            "separated groups should reject, got p={} (V={})",
            res.p_value,
            res.statistic
        );
        // Permutation fanova agrees in direction.
        let f = fanova(&data, &groups, 499).unwrap();
        assert!(
            f.p_value < 0.05,
            "fanova should also reject separated groups, got p={}",
            f.p_value
        );
    }

    // Cross-checks the OLD (deprecated) `fanova` decision vs vstat — pins the old path on purpose.
    #[allow(deprecated)]
    #[test]
    fn vstat_fails_to_reject_pooled_groups_agrees_with_fanova() {
        let argvals = uniform_grid(30);
        // All groups drawn from the SAME distribution (no shift).
        let (data, groups) = make_grouped(20, &argvals, &[0.0, 0.0, 0.0], 202);
        let res = oneway_anova_vstat(&data, &groups, &argvals).unwrap();
        assert!(
            res.p_value > 0.05,
            "pooled groups should not reject, got p={} (V={})",
            res.p_value,
            res.statistic
        );
        let f = fanova(&data, &groups, 499).unwrap();
        assert!(
            f.p_value > 0.05,
            "fanova should also fail to reject pooled groups, got p={}",
            f.p_value
        );
    }

    #[test]
    fn vstat_is_deterministic() {
        let argvals = uniform_grid(25);
        let (data, groups) = make_grouped(15, &argvals, &[0.0, 1.0], 303);
        let a = oneway_anova_vstat(&data, &groups, &argvals).unwrap();
        let b = oneway_anova_vstat(&data, &groups, &argvals).unwrap();
        assert_eq!(a.statistic, b.statistic);
        assert_eq!(a.p_value, b.p_value);
    }

    #[test]
    fn vstat_validates_input() {
        let argvals = uniform_grid(20);
        let (data, groups) = make_grouped(10, &argvals, &[0.0, 1.0], 404);
        // Too few distinct groups (all same label).
        let one_group = vec![0usize; groups.len()];
        assert!(matches!(
            oneway_anova_vstat(&data, &one_group, &argvals),
            Err(FdarError::InvalidParameter { .. })
        ));
        // Mismatched argvals length.
        let bad_argvals = uniform_grid(15);
        assert!(matches!(
            oneway_anova_vstat(&data, &groups, &bad_argvals),
            Err(FdarError::InvalidDimension { .. })
        ));
        // Mismatched groups length.
        let short_groups = vec![0usize; groups.len() - 1];
        assert!(matches!(
            oneway_anova_vstat(&data, &short_groups, &argvals),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
