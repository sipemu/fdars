//! Permutation-based two-sample functional tests.
//!
//! Provides [`t_perm_test`] (integrated L2-of-difference statistic) and
//! [`f_perm_test`] (integrated F-statistic, the k = 2 case of functional
//! ANOVA). Both build a permutation null by pooling the two samples and
//! relabelling group membership via a Fisher–Yates shuffle seeded
//! deterministically with `StdRng::seed_from_u64(seed)`.

use super::TestResult;
use crate::error::FdarError;
use crate::function_on_scalar::integrated_f_statistic;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Default number of permutations for the permutation tests.
pub const DEFAULT_N_PERM: usize = 999;

/// Validate two functional samples share equal, non-zero column counts, that
/// `argvals` matches that width, and that each sample has at least 2 rows.
///
/// Returns `(n_a, n_b, m)` on success.
fn validate_two_samples(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
) -> Result<(usize, usize, usize), FdarError> {
    let (n_a, m_a) = data_a.shape();
    let (n_b, m_b) = data_b.shape();
    if m_a == 0 || m_b == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (grid points)".to_string(),
            actual: format!("data_a has {m_a} columns, data_b has {m_b} columns"),
        });
    }
    if m_a != m_b {
        return Err(FdarError::InvalidDimension {
            parameter: "data_b",
            expected: format!("{m_a} columns (matching data_a)"),
            actual: format!("{m_b} columns"),
        });
    }
    if argvals.len() != m_a {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m_a} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if n_a < 2 || n_b < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows per sample".to_string(),
            actual: format!("data_a has {n_a} rows, data_b has {n_b} rows"),
        });
    }
    Ok((n_a, n_b, m_a))
}

/// Pool two column-major samples into a single `(n_a + n_b) x m` matrix, with
/// the `n_a` rows of `data_a` first, followed by the `n_b` rows of `data_b`.
fn pool_two_samples(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    n_a: usize,
    n_b: usize,
    m: usize,
) -> FdMatrix {
    let mut pooled = FdMatrix::zeros(n_a + n_b, m);
    for j in 0..m {
        for i in 0..n_a {
            pooled[(i, j)] = data_a[(i, j)];
        }
        for i in 0..n_b {
            pooled[(n_a + i, j)] = data_b[(i, j)];
        }
    }
    pooled
}

/// Integrated L2 distance between two sample-mean curves.
///
/// `sqrt( ∫ (mean_a(t) - mean_b(t))^2 dt )`, integrated with Simpson's weights.
fn integrated_l2_mean_diff(
    pooled: &FdMatrix,
    labels: &[usize],
    n_a: usize,
    m: usize,
    weights: &[f64],
) -> f64 {
    let mut mean_a = vec![0.0; m];
    let mut mean_b = vec![0.0; m];
    let n_b = labels.len() - n_a;
    for (i, &lab) in labels.iter().enumerate() {
        if lab == 0 {
            for j in 0..m {
                mean_a[j] += pooled[(i, j)];
            }
        } else {
            for j in 0..m {
                mean_b[j] += pooled[(i, j)];
            }
        }
    }
    for j in 0..m {
        mean_a[j] /= n_a as f64;
        mean_b[j] /= n_b as f64;
    }
    let mut acc = 0.0;
    for j in 0..m {
        let d = mean_a[j] - mean_b[j];
        acc += d * d * weights[j];
    }
    acc.sqrt()
}

/// Fisher–Yates in-place shuffle of `v` using `rng`.
fn shuffle_labels(v: &mut [usize], rng: &mut StdRng) {
    use rand::Rng;
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

/// Functional two-sample permutation *t*-test (`fda::tperm.fd`).
///
/// Tests the null hypothesis that `data_a` and `data_b` are drawn from
/// populations with the same mean curve. The test statistic is the integrated
/// L2 distance between the two sample-mean curves,
/// `sqrt( ∫ (mean_a - mean_b)^2 dt )`, integrated with Simpson's weights over
/// `argvals`. The permutation null pools all `n_a + n_b` curves, relabels group
/// membership via a Fisher–Yates shuffle, and recomputes the statistic; the
/// p-value is `(#{perm >= observed} + 1) / (n_perm + 1)`.
///
/// # Arguments
/// * `data_a` - First sample (`n_a x m`).
/// * `data_b` - Second sample (`n_b x m`).
/// * `argvals` - Evaluation points (length `m`).
/// * `n_perm` - Number of permutations (typical default: [`DEFAULT_N_PERM`] = 999).
/// * `seed` - Deterministic RNG seed (`StdRng::seed_from_u64(seed)`).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the two samples have unequal or
/// zero column counts, if `argvals.len()` does not match the column count, or
/// if either sample has fewer than 2 rows. Returns
/// [`FdarError::InvalidParameter`] if `n_perm == 0`.
pub fn t_perm_test(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    n_perm: usize,
    seed: u64,
) -> Result<TestResult, FdarError> {
    let (n_a, n_b, m) = validate_two_samples(data_a, data_b, argvals)?;
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "must be >= 1".to_string(),
        });
    }

    let weights = simpsons_weights(argvals);
    let pooled = pool_two_samples(data_a, data_b, n_a, n_b, m);

    let mut labels: Vec<usize> = (0..(n_a + n_b)).map(|i| usize::from(i >= n_a)).collect();
    let observed = integrated_l2_mean_diff(&pooled, &labels, n_a, m, &weights);

    // NOT migrated to permutation_test::permutation_pvalue — uses a single ADVANCING StdRng across all
    // permutations; per-perm reseed would change the p-value (Phase-49 CONS-02 Plan A).
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_ge = 0usize;
    for _ in 0..n_perm {
        shuffle_labels(&mut labels, &mut rng);
        let perm_stat = integrated_l2_mean_diff(&pooled, &labels, n_a, m, &weights);
        if perm_stat >= observed {
            n_ge += 1;
        }
    }

    let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
    Ok(TestResult {
        statistic: observed,
        p_value,
        n_perm,
    })
}

/// Functional two-sample permutation *F*-test (`fda::Fperm.fd`).
///
/// The k = 2 case of functional ANOVA: assembles a two-group problem from
/// `data_a` (label 0) and `data_b` (label 1) and computes the integrated
/// F-statistic via the shared `integrated_f_statistic` core (the same core used
/// by [`crate::function_on_scalar::fanova`]). The permutation null relabels the
/// pooled group membership via a seeded Fisher–Yates shuffle; the p-value is
/// `(#{perm >= observed} + 1) / (n_perm + 1)`.
///
/// # Arguments
/// * `data_a` - First sample (`n_a x m`).
/// * `data_b` - Second sample (`n_b x m`).
/// * `argvals` - Evaluation points (length `m`), used only for input validation
///   (the integrated F-statistic is a mean over grid points).
/// * `n_perm` - Number of permutations (typical default: [`DEFAULT_N_PERM`] = 999).
/// * `seed` - Deterministic RNG seed (`StdRng::seed_from_u64(seed)`).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the two samples have unequal or
/// zero column counts, if `argvals.len()` does not match the column count, or
/// if either sample has fewer than 2 rows. Returns
/// [`FdarError::InvalidParameter`] if `n_perm == 0`.
pub fn f_perm_test(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    n_perm: usize,
    seed: u64,
) -> Result<TestResult, FdarError> {
    let (n_a, n_b, m) = validate_two_samples(data_a, data_b, argvals)?;
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "must be >= 1".to_string(),
        });
    }

    let pooled = pool_two_samples(data_a, data_b, n_a, n_b, m);
    let labels_dedup = [0usize, 1usize];

    // Group vector: 0 for the first n_a rows, 1 for the next n_b.
    let mut groups: Vec<usize> = (0..(n_a + n_b)).map(|i| usize::from(i >= n_a)).collect();
    let observed = integrated_f_statistic(&pooled, &groups, &labels_dedup);

    // NOT migrated to permutation_test::permutation_pvalue — uses a single ADVANCING StdRng across all
    // permutations; per-perm reseed would change the p-value (Phase-49 CONS-02 Plan A).
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_ge = 0usize;
    for _ in 0..n_perm {
        shuffle_labels(&mut groups, &mut rng);
        let perm_stat = integrated_f_statistic(&pooled, &groups, &labels_dedup);
        if perm_stat >= observed {
            n_ge += 1;
        }
    }

    let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
    Ok(TestResult {
        statistic: observed,
        p_value,
        n_perm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    /// Deterministic sample: `n` curves of width `m` = `argvals.len()`, each a
    /// smooth base curve plus a per-row/per-column perturbation, shifted by
    /// `shift`.
    fn make_sample(n: usize, argvals: &[f64], shift: f64, seed: u64) -> FdMatrix {
        let m = argvals.len();
        let mut mat = FdMatrix::zeros(n, m);
        // Simple deterministic pseudo-random generator (LCG) for reproducible noise.
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for i in 0..n {
            for (j, &t) in argvals.iter().enumerate() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0; // ~[-1, 1)
                mat[(i, j)] = (2.0 * std::f64::consts::PI * t).sin() + 0.1 * noise + shift;
            }
        }
        mat
    }

    #[test]
    fn t_perm_separated_small_p() {
        let argvals = uniform_grid(25);
        let a = make_sample(15, &argvals, 0.0, 1);
        let b = make_sample(15, &argvals, 5.0, 2); // large constant shift
        let res = t_perm_test(&a, &b, &argvals, 199, 42).unwrap();
        assert!(
            res.p_value < 0.05,
            "separated samples should give small p, got {}",
            res.p_value
        );
    }

    #[test]
    fn t_perm_null_large_p() {
        let argvals = uniform_grid(25);
        let a = make_sample(15, &argvals, 0.0, 10);
        let b = make_sample(15, &argvals, 0.0, 20); // same generator, no shift
        let res = t_perm_test(&a, &b, &argvals, 199, 7).unwrap();
        assert!(
            res.p_value > 0.1,
            "null samples should give large p, got {}",
            res.p_value
        );
    }

    #[test]
    fn t_perm_deterministic() {
        let argvals = uniform_grid(20);
        let a = make_sample(10, &argvals, 0.0, 3);
        let b = make_sample(12, &argvals, 1.0, 4);
        let r1 = t_perm_test(&a, &b, &argvals, 99, 123).unwrap();
        let r2 = t_perm_test(&a, &b, &argvals, 99, 123).unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical result");
    }

    #[test]
    fn t_perm_invalid_input() {
        let argvals = uniform_grid(20);
        let a = make_sample(10, &argvals, 0.0, 5);
        // Mismatched column counts.
        let argvals_b = uniform_grid(15);
        let b = make_sample(10, &argvals_b, 0.0, 6);
        assert!(matches!(
            t_perm_test(&a, &b, &argvals, 99, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
        // n_perm = 0 rejected.
        let b2 = make_sample(10, &argvals, 0.0, 7);
        assert!(matches!(
            t_perm_test(&a, &b2, &argvals, 0, 1),
            Err(FdarError::InvalidParameter { .. })
        ));
        // Too few rows.
        let a_small = make_sample(1, &argvals, 0.0, 8);
        assert!(matches!(
            t_perm_test(&a_small, &b2, &argvals, 99, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn f_perm_separated_small_p() {
        let argvals = uniform_grid(25);
        let a = make_sample(15, &argvals, 0.0, 11);
        let b = make_sample(15, &argvals, 5.0, 12);
        let res = f_perm_test(&a, &b, &argvals, 199, 42).unwrap();
        assert!(
            res.p_value < 0.05,
            "separated samples should give small p, got {}",
            res.p_value
        );
    }

    #[test]
    fn f_perm_null_large_p() {
        let argvals = uniform_grid(25);
        let a = make_sample(15, &argvals, 0.0, 30);
        let b = make_sample(15, &argvals, 0.0, 40);
        let res = f_perm_test(&a, &b, &argvals, 199, 7).unwrap();
        assert!(
            res.p_value > 0.1,
            "null samples should give large p, got {}",
            res.p_value
        );
    }

    #[test]
    fn f_perm_deterministic() {
        let argvals = uniform_grid(20);
        let a = make_sample(10, &argvals, 0.0, 3);
        let b = make_sample(12, &argvals, 1.0, 4);
        let r1 = f_perm_test(&a, &b, &argvals, 99, 555).unwrap();
        let r2 = f_perm_test(&a, &b, &argvals, 99, 555).unwrap();
        assert_eq!(r1, r2);
    }

    // Cross-checks the OLD (deprecated) `fanova` decision vs f_perm_test — pins the old path.
    #[allow(deprecated)]
    #[test]
    fn f_perm_agrees_with_fanova_decision() {
        use crate::function_on_scalar::fanova;
        let argvals = uniform_grid(25);
        let a = make_sample(15, &argvals, 0.0, 111);
        let b = make_sample(15, &argvals, 5.0, 112);
        // Stack as a 2-group fanova problem.
        let n_a = 15usize;
        let n_b = 15usize;
        let m = argvals.len();
        let mut pooled = FdMatrix::zeros(n_a + n_b, m);
        for j in 0..m {
            for i in 0..n_a {
                pooled[(i, j)] = a[(i, j)];
            }
            for i in 0..n_b {
                pooled[(n_a + i, j)] = b[(i, j)];
            }
        }
        let groups: Vec<usize> = (0..(n_a + n_b)).map(|i| usize::from(i >= n_a)).collect();
        let fa = fanova(&pooled, &groups, 199).unwrap();
        let fp = f_perm_test(&a, &b, &argvals, 199, 42).unwrap();
        // Both should reject at 0.05.
        assert!(fa.p_value < 0.05);
        assert!(fp.p_value < 0.05);
    }

    #[test]
    fn f_perm_invalid_input() {
        let argvals = uniform_grid(20);
        let a = make_sample(10, &argvals, 0.0, 5);
        let b2 = make_sample(10, &argvals, 0.0, 7);
        assert!(matches!(
            f_perm_test(&a, &b2, &argvals, 0, 1),
            Err(FdarError::InvalidParameter { .. })
        ));
        let a_small = make_sample(1, &argvals, 0.0, 8);
        assert!(matches!(
            f_perm_test(&a_small, &b2, &argvals, 99, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
