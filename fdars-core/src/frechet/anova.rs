//! Fréchet ANOVA: a group-difference test on metric-space (density) responses.
//!
//! Implements the Dubey–Müller (2019) Fréchet ANOVA statistic `Tₙ` over the
//! 1D-Wasserstein density space, reusing the Wave-1 [`frechet_mean`] /
//! [`frechet_variance`] machinery and the in-crate chi-square survival function.
//! Reports the statistic, an asymptotic χ²(k−1) p-value, and a primary seeded
//! permutation p-value.

use super::mean::{frechet_mean, frechet_variance};
use super::space::WassersteinDensitySpace;
use super::{FrechetAnovaResult, MetricSpace};
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::inference::dist::chi_square_sf;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Compute the Dubey–Müller `Tₙ` statistic and its components for a labelling.
///
/// Returns `(Tn, Fn, Un, group_variances, pooled_variance)`.
///
/// # σ̂ₗ² formula — [ASSUMED]
///
/// `σ̂ₗ² = (1/nₗ) Σᵢ [d²(Yᵢ, μ̂ₗ) − V̂ₗ]²` follows Dubey & Müller (2019,
/// *Biometrika* 106(4)) conventions; the exact variance-estimator form is
/// **[ASSUMED]** from the public R `frechet::DenANOVA` sources — if the
/// asymptotic statistic magnitude diverges from R on real data the formula may
/// need adjustment. The seeded permutation p-value (the primary reported
/// inference) is robust to this assumption.
pub(crate) fn compute_tn_generic<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    labels: &[usize],
    k: usize,
) -> Result<(f64, f64, f64, Vec<f64>, f64), FdarError>
where
    S::Object: Clone,
{
    let n = objects.len();

    // Partition object indices by (contiguous 0..k) group id.
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &g) in labels.iter().enumerate() {
        groups[g].push(i);
    }

    // Pooled Fréchet mean + variance.
    let pooled_mean = frechet_mean(space, objects, None)?;
    let pooled_var = frechet_variance(space, objects, &pooled_mean, None)?;

    let mut group_vars = vec![0.0; k];
    let mut sigma2 = vec![0.0; k];
    let mut lambda = vec![0.0; k];
    for (g, idx) in groups.iter().enumerate() {
        let n_g = idx.len();
        lambda[g] = n_g as f64 / n as f64;
        let subset: Vec<S::Object> = idx.iter().map(|&i| objects[i].clone()).collect();
        let mu_g = frechet_mean(space, &subset, None)?;
        // d²(Yᵢ, μ̂ₗ) for each member.
        let d2: Vec<f64> = subset
            .iter()
            .map(|o| space.distance(o, &mu_g).map(|d| d * d))
            .collect::<Result<Vec<f64>, _>>()?;
        let v_g = d2.iter().sum::<f64>() / n_g as f64;
        // σ̂ₗ² = (1/nₗ) Σ (d² − V̂ₗ)²  ([ASSUMED] — see fn docs).
        let s2 = d2.iter().map(|&d| (d - v_g).powi(2)).sum::<f64>() / n_g as f64;
        group_vars[g] = v_g;
        sigma2[g] = s2.max(NUMERICAL_EPS);
    }

    // Fₙ = V̂ₚ − Σₗ λₗ V̂ₗ.
    let fn_stat = pooled_var - (0..k).map(|g| lambda[g] * group_vars[g]).sum::<f64>();

    // Uₙ = Σⱼ<ₗ (λⱼλₗ / (σ̂ⱼ² σ̂ₗ²)) (V̂ⱼ − V̂ₗ)².
    let mut un = 0.0;
    for j in 0..k {
        for l in (j + 1)..k {
            un += (lambda[j] * lambda[l] / (sigma2[j] * sigma2[l]))
                * (group_vars[j] - group_vars[l]).powi(2);
        }
    }

    // Tₙ = n·Uₙ / Σₗ(λₗ/σ̂ₗ²) + n·Fₙ² / Σₗ(λₗ²σ̂ₗ²).
    let denom_u: f64 = (0..k).map(|g| lambda[g] / sigma2[g]).sum();
    let denom_f: f64 = (0..k).map(|g| lambda[g] * lambda[g] * sigma2[g]).sum();
    let term_u = if denom_u > NUMERICAL_EPS {
        n as f64 * un / denom_u
    } else {
        0.0
    };
    let term_f = if denom_f > NUMERICAL_EPS {
        n as f64 * fn_stat * fn_stat / denom_f
    } else {
        0.0
    };
    let tn = term_u + term_f;

    Ok((tn, fn_stat, un, group_vars, pooled_var))
}

/// Fréchet ANOVA group-difference test on metric-space (density) responses.
///
/// Each response row of `responses` is a density on `argvals`; `group_labels`
/// assigns each to a group. Computes the Dubey–Müller `Tₙ` statistic and returns
/// both a seeded **permutation** p-value (the primary reported inference) and an
/// asymptotic χ²(k−1) p-value (secondary).
///
/// The permutation p-value shuffles the group labels `n_perm` times with a
/// per-iteration seeded RNG (`StdRng::seed_from_u64(seed + k)`), so it is
/// reproducible for a fixed `seed`. Pass `n_perm = 0` to use the default of 999.
///
/// The exact σ̂ₗ² variance estimator is **[ASSUMED]** (see [`compute_tn_generic`]);
/// the permutation p-value does not depend on that assumption.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] for a `group_labels`/response length
/// mismatch or an `argvals` length mismatch, and [`FdarError::InvalidParameter`]
/// for non-monotone `argvals` or fewer than two distinct groups.
#[must_use = "returns the Fréchet ANOVA result; examine the p-values"]
pub fn frechet_anova(
    responses: &FdMatrix,
    argvals: &[f64],
    group_labels: &[usize],
    n_perm: usize,
    seed: u64,
) -> Result<FrechetAnovaResult, FdarError> {
    let (n, m) = responses.shape();
    if group_labels.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "group_labels",
            expected: format!("{n} labels (matching response rows)"),
            actual: format!("{} labels", group_labels.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching response columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    // Distinct groups; require contiguous 0..k labelling for indexing.
    let k = group_labels.iter().copied().max().map_or(0, |mx| mx + 1);
    let distinct: std::collections::BTreeSet<usize> = group_labels.iter().copied().collect();
    if distinct.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "group_labels",
            message: "need at least 2 distinct groups for a Fréchet ANOVA".to_string(),
        });
    }
    if distinct.len() != k || *distinct.iter().next().unwrap() != 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "group_labels",
            message: format!("group labels must be contiguous 0..{k}"),
        });
    }

    let space = WassersteinDensitySpace::new(argvals.to_vec())?;
    let objects: Vec<Vec<f64>> = (0..n).map(|i| responses.row(i)).collect();

    let n_perm = if n_perm == 0 { 999 } else { n_perm };

    let (tn_obs, fn_stat, un_stat, group_vars, pooled_var) =
        compute_tn_generic(&space, &objects, group_labels, k)?;
    let p_asymptotic = chi_square_sf(tn_obs, k - 1);

    // Seeded permutation p-value (per-iteration RNG → thread-count-independent).
    let mut n_ge = 0usize;
    for perm in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
        let mut perm_labels = group_labels.to_vec();
        perm_labels.shuffle(&mut rng);
        // A degenerate permutation (compute error) is skipped conservatively.
        if let Ok((tn_perm, _, _, _, _)) = compute_tn_generic(&space, &objects, &perm_labels, k) {
            if tn_perm >= tn_obs {
                n_ge += 1;
            }
        }
    }
    let p_permutation = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(FrechetAnovaResult {
        statistic: tn_obs,
        p_value_asymptotic: p_asymptotic,
        p_value_permutation: p_permutation,
        n_perm,
        group_frechet_variances: group_vars,
        pooled_frechet_variance: pooled_var,
        fn_statistic: fn_stat,
        un_statistic: un_stat,
        group_labels: group_labels.to_vec(),
    })
}

/// Fréchet ANOVA group-difference test over an arbitrary object [`MetricSpace`]
/// backend (FRE-02-07). Reuses the Dubey–Müller [`compute_tn_generic`] statistic
/// and the identical seeded-permutation p-value machinery as [`frechet_anova`],
/// generalized from densities to any metric-space objects.
///
/// `objects` are the responses (one per observation); `group_labels` assigns each
/// to a contiguous group `0..k`. Pass `n_perm = 0` for the default of 999. The
/// permutation p-value is reproducible for a fixed `seed`
/// (`StdRng::seed_from_u64(seed.wrapping_add(perm))`).
///
/// # Errors
/// [`FdarError::InvalidDimension`] for an empty sample or a
/// `group_labels`/objects length mismatch; [`FdarError::InvalidParameter`] for
/// fewer than two distinct groups or non-contiguous labels.
#[must_use = "returns the Fréchet ANOVA result; examine the p-values"]
pub fn frechet_anova_space<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    group_labels: &[usize],
    n_perm: usize,
    seed: u64,
) -> Result<FrechetAnovaResult, FdarError>
where
    S::Object: Clone,
{
    let n = objects.len();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "objects",
            expected: "at least 1 object".to_string(),
            actual: "0 objects".to_string(),
        });
    }
    if group_labels.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "group_labels",
            expected: format!("{n} labels (matching objects)"),
            actual: format!("{} labels", group_labels.len()),
        });
    }
    let k = group_labels.iter().copied().max().map_or(0, |mx| mx + 1);
    let distinct: std::collections::BTreeSet<usize> = group_labels.iter().copied().collect();
    if distinct.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "group_labels",
            message: "need at least 2 distinct groups for a Fréchet ANOVA".to_string(),
        });
    }
    if distinct.len() != k || *distinct.iter().next().unwrap() != 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "group_labels",
            message: format!("group labels must be contiguous 0..{k}"),
        });
    }

    let n_perm = if n_perm == 0 { 999 } else { n_perm };

    let (tn_obs, fn_stat, un_stat, group_vars, pooled_var) =
        compute_tn_generic(space, objects, group_labels, k)?;
    let p_asymptotic = chi_square_sf(tn_obs, k - 1);

    let mut n_ge = 0usize;
    for perm in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
        let mut perm_labels = group_labels.to_vec();
        perm_labels.shuffle(&mut rng);
        if let Ok((tn_perm, _, _, _, _)) = compute_tn_generic(space, objects, &perm_labels, k) {
            if tn_perm >= tn_obs {
                n_ge += 1;
            }
        }
    }
    let p_permutation = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);

    Ok(FrechetAnovaResult {
        statistic: tn_obs,
        p_value_asymptotic: p_asymptotic,
        p_value_permutation: p_permutation,
        n_perm,
        group_frechet_variances: group_vars,
        pooled_frechet_variance: pooled_var,
        fn_statistic: fn_stat,
        un_statistic: un_stat,
        group_labels: group_labels.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::trapz;

    fn uniform_grid(m: usize, lb: f64, ub: f64) -> Vec<f64> {
        (0..m)
            .map(|j| lb + (ub - lb) * j as f64 / (m - 1) as f64)
            .collect()
    }

    fn gaussian(argvals: &[f64], mu: f64) -> Vec<f64> {
        let raw: Vec<f64> = argvals
            .iter()
            .map(|&x| (-(x - mu).powi(2) / 2.0).exp())
            .collect();
        let integral = trapz(&raw, argvals);
        raw.iter().map(|&d| d / integral).collect()
    }

    /// Build a 2-group response matrix: `n_each` densities N(mu_a,1) then N(mu_b,1).
    fn two_group(argvals: &[f64], n_each: usize, mu_a: f64, mu_b: f64) -> (FdMatrix, Vec<usize>) {
        let m = argvals.len();
        let mut resp = FdMatrix::zeros(2 * n_each, m);
        let mut labels = vec![0usize; 2 * n_each];
        for i in 0..n_each {
            // small deterministic within-group jitter so groups aren't degenerate
            let a = gaussian(
                argvals,
                mu_a + 0.05 * ((i as f64) - n_each as f64 / 2.0) / n_each as f64,
            );
            let b = gaussian(
                argvals,
                mu_b + 0.05 * ((i as f64) - n_each as f64 / 2.0) / n_each as f64,
            );
            for j in 0..m {
                resp[(i, j)] = a[j];
                resp[(n_each + i, j)] = b[j];
            }
            labels[i] = 0;
            labels[n_each + i] = 1;
        }
        (resp, labels)
    }

    #[test]
    fn anova_flags_shifted_groups() {
        let argvals = uniform_grid(81, -5.0, 5.0);
        let (resp, labels) = two_group(&argvals, 12, -1.0, 1.0);
        let res = frechet_anova(&resp, &argvals, &labels, 199, 42).unwrap();
        assert!(
            res.p_value_permutation < 0.05,
            "perm p = {}",
            res.p_value_permutation
        );
    }

    #[test]
    fn anova_ignores_homogeneous_sample() {
        let argvals = uniform_grid(81, -5.0, 5.0);
        let (resp, labels) = two_group(&argvals, 12, 0.0, 0.0);
        let res = frechet_anova(&resp, &argvals, &labels, 199, 7).unwrap();
        assert!(
            res.p_value_permutation > 0.05,
            "perm p = {}",
            res.p_value_permutation
        );
    }

    #[test]
    fn anova_permutation_is_seed_reproducible() {
        let argvals = uniform_grid(61, -5.0, 5.0);
        let (resp, labels) = two_group(&argvals, 10, -0.7, 0.7);
        let a = frechet_anova(&resp, &argvals, &labels, 99, 123).unwrap();
        let b = frechet_anova(&resp, &argvals, &labels, 99, 123).unwrap();
        assert_eq!(a.p_value_permutation, b.p_value_permutation);
        assert_eq!(a.statistic, b.statistic);
    }

    #[test]
    fn anova_rejects_too_few_groups() {
        let argvals = uniform_grid(41, -5.0, 5.0);
        let (resp, _labels) = two_group(&argvals, 5, 0.0, 0.0);
        let labels = vec![0usize; resp.nrows()]; // single group
        assert!(matches!(
            frechet_anova(&resp, &argvals, &labels, 49, 1).unwrap_err(),
            FdarError::InvalidParameter { parameter, .. } if parameter == "group_labels"
        ));
    }

    #[test]
    fn anova_rejects_label_mismatch() {
        let argvals = uniform_grid(41, -5.0, 5.0);
        let (resp, _labels) = two_group(&argvals, 5, -1.0, 1.0);
        let labels = vec![0usize, 1, 0]; // wrong length
        assert!(matches!(
            frechet_anova(&resp, &argvals, &labels, 49, 1).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
    }

    // ── Generic object-space ANOVA (FRE-02-07) ──────────────────────────────

    use crate::frechet::{SpdMatrixSpace, SpdMetric};

    /// A 2-group SPD sample: `n_each` 2×2 matrices near `base_a`·I then near
    /// `base_b`·I, with small deterministic within-group jitter.
    fn spd_two_group(n_each: usize, base_a: f64, base_b: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut objects = Vec::with_capacity(2 * n_each);
        let mut labels = vec![0usize; 2 * n_each];
        for i in 0..n_each {
            let e = 0.02 * ((i as f64) - n_each as f64 / 2.0);
            objects.push(vec![base_a + e, 0.0, 0.0, base_a - 0.5 * e]);
            labels[i] = 0;
        }
        for i in 0..n_each {
            let e = 0.02 * ((i as f64) - n_each as f64 / 2.0);
            objects.push(vec![base_b + e, 0.0, 0.0, base_b - 0.5 * e]);
            labels[n_each + i] = 1;
        }
        (objects, labels)
    }

    #[test]
    fn spd_anova_homogeneous_not_significant() {
        let space = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        // Both groups drawn from the same distribution around identity.
        let (objects, labels) = spd_two_group(12, 1.0, 1.0);
        let res = frechet_anova_space(&space, &objects, &labels, 199, 7).unwrap();
        assert!(
            res.p_value_permutation > 0.05,
            "perm p = {}",
            res.p_value_permutation
        );
    }

    #[test]
    fn spd_anova_separated_significant() {
        let space = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        // Group A near identity, group B near 4·identity.
        let (objects, labels) = spd_two_group(15, 1.0, 4.0);
        let res = frechet_anova_space(&space, &objects, &labels, 199, 42).unwrap();
        assert!(
            res.p_value_permutation < 0.05,
            "perm p = {}",
            res.p_value_permutation
        );
    }

    #[test]
    fn spd_anova_seed_reproducible() {
        let space = SpdMatrixSpace::new(2, SpdMetric::Frobenius).unwrap();
        let (objects, labels) = spd_two_group(10, 1.0, 2.0);
        let a = frechet_anova_space(&space, &objects, &labels, 99, 123).unwrap();
        let b = frechet_anova_space(&space, &objects, &labels, 99, 123).unwrap();
        assert_eq!(a.p_value_permutation, b.p_value_permutation);
        assert_eq!(a.statistic, b.statistic);
    }
}
