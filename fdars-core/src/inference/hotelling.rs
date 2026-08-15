//! FPC-basis Hotelling-T² two-sample mean test.
//!
//! [`two_sample_mean_test`] projects both samples onto a shared FPC basis
//! (fitted on the pooled data via [`crate::regression::fdata_to_pc_1d`]),
//! forms the Hotelling-T² statistic on the difference of the two group
//! score-means (reusing [`crate::spm::stats::hotelling_t2`]), and converts it
//! to a p-value via the asymptotic chi-square(`ncomp`) upper tail.

use super::TestResult;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use crate::spm::stats::hotelling_t2;

/// Regularized lower incomplete gamma function P(a, x) via the series
/// expansion (converges for x < a + 1). Numerical-Recipes style.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    // Uses the series P(a,x) = x^a e^{-x} / Γ(a) · Σ_{n≥0} x^n / (a(a+1)...(a+n))
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..200 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized upper incomplete gamma function Q(a, x) via the continued
/// fraction (converges for x >= a + 1). Numerical-Recipes style.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Log-gamma via the Lanczos approximation.
///
/// The coefficients are the published Lanczos g = 7, n = 9 series; their full
/// precision is intentional, so `clippy::excessive_precision` is allowed here.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        let t = x + G + 0.5;
        for (i, &c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Chi-square survival function: P(X > x) for X ~ χ²(k degrees of freedom).
///
/// Equivalent to the regularized upper incomplete gamma Q(k/2, x/2).
fn chi_square_sf(x: f64, k: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let a = k as f64 / 2.0;
    let xx = x / 2.0;
    if xx < a + 1.0 {
        1.0 - gamma_p_series(a, xx)
    } else {
        gamma_q_cf(a, xx)
    }
}

/// Mean score vector (length ncomp) over the rows of a score matrix.
fn mean_scores(scores: &FdMatrix) -> Vec<f64> {
    let (n, ncomp) = scores.shape();
    let mut mean = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut s = 0.0;
        for i in 0..n {
            s += scores[(i, k)];
        }
        mean[k] = s / n as f64;
    }
    mean
}

/// Functional two-sample mean-equality test via Hotelling-T² on a shared FPC
/// basis (`fda.usc`-style mean equality).
///
/// Both samples are projected onto a common FPC basis fitted on the pooled
/// data. The Hotelling-T² statistic is formed on the difference of the two
/// group score-means, scaled by the effective sample size
/// `sqrt(n_a · n_b / (n_a + n_b))` so that under the null the statistic is
/// asymptotically χ²(`ncomp`). The p-value is the χ²(`ncomp`) upper-tail
/// probability of the observed statistic.
///
/// The eigenvalues fed to [`hotelling_t2`] are derived from the pooled FPCA
/// singular values via `eigenvalue = sv² / (n_pooled − 1)` (the mfpca
/// convention).
///
/// # Arguments
/// * `data_a` - First sample (`n_a x m`).
/// * `data_b` - Second sample (`n_b x m`).
/// * `argvals` - Evaluation points (length `m`).
/// * `ncomp` - Number of FPC components for the shared basis.
///
/// Returns a [`TestResult`] with `n_perm = 0` (non-permutation path).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the two samples have unequal or
/// zero column counts, if `argvals.len()` does not match the column count, or
/// if either sample has fewer than 2 rows. Returns
/// [`FdarError::InvalidParameter`] if `ncomp < 1`. Propagates errors from
/// [`fdata_to_pc_1d`] / [`hotelling_t2`].
pub fn two_sample_mean_test(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
) -> Result<TestResult, FdarError> {
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
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {ncomp}"),
        });
    }

    // Pool the two samples (data_a rows first, then data_b rows).
    let n_pooled = n_a + n_b;
    let m = m_a;
    let mut pooled = FdMatrix::zeros(n_pooled, m);
    for j in 0..m {
        for i in 0..n_a {
            pooled[(i, j)] = data_a[(i, j)];
        }
        for i in 0..n_b {
            pooled[(n_a + i, j)] = data_b[(i, j)];
        }
    }

    // Fit a shared FPC basis on the pooled data.
    let fpca = fdata_to_pc_1d(&pooled, ncomp, argvals)?;
    // fdata_to_pc_1d clamps ncomp to min(n, m); use the realized component count.
    let eff_ncomp = fpca.singular_values.len();

    // Express both samples in the shared coordinate system.
    let scores_a = fpca.project(data_a)?;
    let scores_b = fpca.project(data_b)?;
    let mean_a = mean_scores(&scores_a);
    let mean_b = mean_scores(&scores_b);

    // Effective sample-size scaling: under H0 the scaled mean-difference has
    // unit-variance components (in eigenvalue units), giving an asymptotic
    // χ²(ncomp) statistic.
    let scale = ((n_a as f64) * (n_b as f64) / (n_pooled as f64)).sqrt();
    let diff: Vec<f64> = (0..eff_ncomp)
        .map(|k| scale * (mean_a[k] - mean_b[k]))
        .collect();
    let diff_row = FdMatrix::from_column_major(diff, 1, eff_ncomp)?;

    // Eigenvalues from pooled singular values (mfpca convention sv^2/(n-1)).
    let eigenvalues: Vec<f64> = fpca
        .singular_values
        .iter()
        .map(|&sv| (sv * sv / (n_pooled as f64 - 1.0)).max(1e-15))
        .collect();

    let t2 = hotelling_t2(&diff_row, &eigenvalues)?[0];
    let p_value = chi_square_sf(t2, eff_ncomp);

    Ok(TestResult {
        statistic: t2,
        p_value,
        n_perm: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    fn make_sample(n: usize, argvals: &[f64], shift: f64, seed: u64) -> FdMatrix {
        let m = argvals.len();
        let mut mat = FdMatrix::zeros(n, m);
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for i in 0..n {
            for (j, &t) in argvals.iter().enumerate() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                mat[(i, j)] = (2.0 * std::f64::consts::PI * t).sin() + 0.2 * noise + shift;
            }
        }
        mat
    }

    #[test]
    fn chi_square_sf_sane() {
        // χ²(1): P(X > 3.841) ≈ 0.05; P(X > 0) = 1.
        assert!((chi_square_sf(3.8415, 1) - 0.05).abs() < 1e-3);
        assert!((chi_square_sf(0.0, 3) - 1.0).abs() < 1e-12);
        // χ²(2): P(X > 5.991) ≈ 0.05.
        assert!((chi_square_sf(5.9915, 2) - 0.05).abs() < 1e-3);
        // Monotone decreasing.
        assert!(chi_square_sf(1.0, 3) > chi_square_sf(5.0, 3));
    }

    #[test]
    fn mean_test_differ_rejects() {
        let argvals = uniform_grid(30);
        let a = make_sample(30, &argvals, 0.0, 101);
        let b = make_sample(30, &argvals, 2.0, 102); // clearly different mean
        let res = two_sample_mean_test(&a, &b, &argvals, 3).unwrap();
        assert!(
            res.p_value < 0.05,
            "differing means should reject, got p={}",
            res.p_value
        );
    }

    #[test]
    fn mean_test_coincide_fails_to_reject() {
        let argvals = uniform_grid(30);
        let a = make_sample(30, &argvals, 0.0, 201);
        let b = make_sample(30, &argvals, 0.0, 202); // same generator
        let res = two_sample_mean_test(&a, &b, &argvals, 3).unwrap();
        assert!(
            res.p_value > 0.05,
            "coinciding means should not reject, got p={}",
            res.p_value
        );
    }

    #[test]
    fn mean_test_invalid_input() {
        let argvals = uniform_grid(20);
        let a = make_sample(10, &argvals, 0.0, 5);
        // ncomp = 0 rejected.
        let b = make_sample(10, &argvals, 0.0, 6);
        assert!(matches!(
            two_sample_mean_test(&a, &b, &argvals, 0),
            Err(FdarError::InvalidParameter { .. })
        ));
        // Mismatched columns.
        let argvals_b = uniform_grid(15);
        let b2 = make_sample(10, &argvals_b, 0.0, 7);
        assert!(matches!(
            two_sample_mean_test(&a, &b2, &argvals, 3),
            Err(FdarError::InvalidDimension { .. })
        ));
        // Too few rows.
        let a_small = make_sample(1, &argvals, 0.0, 8);
        assert!(matches!(
            two_sample_mean_test(&a_small, &b, &argvals, 3),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
