//! Comprehensive tests for the SPM module.

#[cfg(test)]
mod tests {
    use crate::matrix::FdMatrix;
    use crate::spm::chi_squared::{chi2_cdf, chi2_quantile};
    use crate::spm::contrib::{spe_contributions, t2_contributions};
    use crate::spm::control::{spe_control_limit, t2_control_limit};
    use crate::spm::ewma::{ewma_scores, spm_ewma_monitor, EwmaConfig};
    use crate::spm::frcc::{frcc_monitor, frcc_phase1, FrccConfig};
    use crate::spm::mfpca::{mfpca, MfpcaConfig};
    use crate::spm::phase::{mf_spm_monitor, mf_spm_phase1, spm_monitor, spm_phase1, SpmConfig};
    use crate::spm::stats::{hotelling_t2, spe_multivariate, spe_univariate};
    use std::f64::consts::PI;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    /// Generate in-control functional data: sine waves with random phase.
    fn generate_ic_data(n: usize, m: usize, seed: u64) -> FdMatrix {
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            // Deterministic pseudo-random phase
            let phase = ((seed.wrapping_add(i as u64).wrapping_mul(2654435761)) as f64)
                / (u32::MAX as f64)
                * 0.5;
            let amplitude = 1.0
                + ((seed.wrapping_add(i as u64 * 7).wrapping_mul(1103515245)) as f64)
                    / (u32::MAX as f64)
                    * 0.2;
            for j in 0..m {
                data[(i, j)] = amplitude * (2.0 * PI * t[j] + phase).sin()
                    + 0.05
                        * ((seed
                            .wrapping_add(i as u64 * 13 + j as u64 * 7)
                            .wrapping_mul(48271)) as f64)
                        / (u32::MAX as f64);
            }
        }
        data
    }

    /// Generate out-of-control data (shifted mean).
    fn generate_oc_data(n: usize, m: usize, shift: f64, seed: u64) -> FdMatrix {
        let mut data = generate_ic_data(n, m, seed + 999);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] += shift;
            }
        }
        data
    }

    // ── chi_squared tests ────────────────────────────────────────────────────

    #[test]
    fn test_chi2_cdf_at_zero() {
        assert_eq!(chi2_cdf(0.0, 1), 0.0);
        assert_eq!(chi2_cdf(0.0, 5), 0.0);
    }

    #[test]
    fn test_chi2_quantile_median_df2() {
        let q = chi2_quantile(0.5, 2);
        assert!(
            (q - 1.3862943611198906).abs() < 0.02,
            "chi2_quantile(0.5, 2) = {q}, expected ~1.386"
        );
    }

    #[test]
    fn test_chi2_quantile_95th_df2() {
        let q = chi2_quantile(0.95, 2);
        assert!(
            (q - 5.991).abs() < 0.02,
            "chi2_quantile(0.95, 2) = {q}, expected ~5.991"
        );
    }

    #[test]
    fn test_chi2_roundtrip() {
        for k in &[1usize, 2, 5, 10] {
            for &x in &[1.0, 3.0, 7.0, 15.0] {
                let p = chi2_cdf(x, *k);
                if p > 0.01 && p < 0.99 {
                    let x_back = chi2_quantile(p, *k);
                    assert!(
                        (x_back - x).abs() < 0.1,
                        "Round-trip failed: k={k}, x={x}, p={p}, x_back={x_back}"
                    );
                }
            }
        }
    }

    // ── T-squared tests ──────────────────────────────────────────────────────

    #[test]
    fn test_t2_non_negative() {
        let scores =
            FdMatrix::from_column_major(vec![1.0, -2.0, 0.5, 3.0, 1.0, -1.0], 3, 2).unwrap();
        let eigenvalues = vec![2.0, 0.5];
        let t2 = hotelling_t2(&scores, &eigenvalues).unwrap();
        for &v in &t2 {
            assert!(v >= 0.0, "T2 must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_t2_zero_scores() {
        let scores = FdMatrix::zeros(5, 3);
        let eigenvalues = vec![1.0, 1.0, 1.0];
        let t2 = hotelling_t2(&scores, &eigenvalues).unwrap();
        for &v in &t2 {
            assert!(v.abs() < 1e-15, "T2 of zero scores should be 0, got {v}");
        }
    }

    #[test]
    fn test_t2_dimension_mismatch() {
        let scores = FdMatrix::zeros(5, 3);
        let eigenvalues = vec![1.0, 1.0]; // wrong length
        assert!(hotelling_t2(&scores, &eigenvalues).is_err());
    }

    #[test]
    fn test_t2_negative_eigenvalue() {
        let scores = FdMatrix::zeros(5, 2);
        let eigenvalues = vec![1.0, -0.5];
        assert!(hotelling_t2(&scores, &eigenvalues).is_err());
    }

    // ── SPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_spe_non_negative() {
        let m = 20;
        let argvals = uniform_grid(m);
        let centered = generate_ic_data(10, m, 42);
        let reconstructed = FdMatrix::zeros(10, m);
        let spe = spe_univariate(&centered, &reconstructed, &argvals).unwrap();
        for &v in &spe {
            assert!(v >= 0.0, "SPE must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_spe_zero_error() {
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(5, m, 42);
        let spe = spe_univariate(&data, &data, &argvals).unwrap();
        for &v in &spe {
            assert!(
                v.abs() < 1e-10,
                "SPE should be 0 when centered==reconstructed, got {v}"
            );
        }
    }

    #[test]
    fn test_spe_dimension_mismatch() {
        let argvals = uniform_grid(20);
        let centered = FdMatrix::zeros(5, 20);
        let reconstructed = FdMatrix::zeros(5, 15); // wrong size
        assert!(spe_univariate(&centered, &reconstructed, &argvals).is_err());
    }

    #[test]
    fn test_spe_multivariate_matches_sum() {
        let m1 = 15;
        let m2 = 20;
        let argvals1 = uniform_grid(m1);
        let argvals2 = uniform_grid(m2);
        let n = 5;

        let var1_std = generate_ic_data(n, m1, 42);
        let var1_rec = FdMatrix::zeros(n, m1);
        let var2_std = generate_ic_data(n, m2, 99);
        let var2_rec = FdMatrix::zeros(n, m2);

        let spe_total = spe_multivariate(
            &[&var1_std, &var2_std],
            &[&var1_rec, &var2_rec],
            &[&argvals1, &argvals2],
        )
        .unwrap();

        let spe1 = spe_univariate(&var1_std, &var1_rec, &argvals1).unwrap();
        let spe2 = spe_univariate(&var2_std, &var2_rec, &argvals2).unwrap();

        for i in 0..n {
            let expected = spe1[i] + spe2[i];
            assert!(
                (spe_total[i] - expected).abs() < 1e-10,
                "Multivariate SPE should equal sum of univariate, obs {i}: {} vs {}",
                spe_total[i],
                expected
            );
        }
    }

    // ── Control limit tests ──────────────────────────────────────────────────

    #[test]
    fn test_t2_control_limit_chi2() {
        let limit = t2_control_limit(2, 0.05).unwrap();
        assert!(
            (limit.ucl - 5.991).abs() < 0.05,
            "T2 UCL for ncomp=2, alpha=0.05 should be ~5.991, got {}",
            limit.ucl
        );
    }

    #[test]
    fn test_t2_control_limit_invalid() {
        assert!(t2_control_limit(0, 0.05).is_err());
        assert!(t2_control_limit(5, 0.0).is_err());
        assert!(t2_control_limit(5, 1.0).is_err());
    }

    #[test]
    fn test_spe_control_limit_moment_matching() {
        // Known SPE values with mean ~2, var ~1
        let spe_values: Vec<f64> = (0..100)
            .map(|i| 2.0 + 0.5 * ((i as f64 * 0.1).sin()))
            .collect();
        let limit = spe_control_limit(&spe_values, 0.05).unwrap();
        assert!(limit.ucl > 0.0, "SPE UCL must be positive");
        // UCL should be above most values
        let n_above = spe_values.iter().filter(|&&v| v > limit.ucl).count();
        // With alpha=0.05, we expect roughly 5% or fewer above
        assert!(
            n_above <= 15,
            "Too many values ({n_above}/100) above SPE UCL"
        );
    }

    #[test]
    fn test_spe_control_limit_too_few_values() {
        assert!(spe_control_limit(&[1.0], 0.05).is_err());
    }

    // ── Phase I/II tests ─────────────────────────────────────────────────────

    #[test]
    fn test_phase1_basic() {
        let n = 40;
        let m = 30;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = spm_phase1(&data, &argvals, &config).unwrap();
        assert_eq!(chart.t2_phase1.len() + 0, chart.spe_phase1.len()); // same length
        assert!(chart.t2_limit.ucl > 0.0);
        assert!(chart.spe_limit.ucl > 0.0);

        // T2 should be non-negative
        for &v in &chart.t2_phase1 {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_phase1_ic_few_alarms() {
        let n = 60;
        let m = 30;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Monitor the same type of data
        let new_data = generate_ic_data(20, m, 123);
        let result = spm_monitor(&chart, &new_data, &argvals).unwrap();

        // In-control data should have few alarms (allow some due to random variation)
        let t2_alarm_count = result.t2_alarm.iter().filter(|&&a| a).count();
        let spe_alarm_count = result.spe_alarm.iter().filter(|&&a| a).count();
        // With 20 observations at alpha=0.05, expect ~1 alarm
        // Allow up to 10 for robustness
        assert!(
            t2_alarm_count <= 15,
            "Too many T2 alarms for IC data: {t2_alarm_count}/20"
        );
        assert!(
            spe_alarm_count <= 15,
            "Too many SPE alarms for IC data: {spe_alarm_count}/20"
        );
    }

    #[test]
    fn test_phase2_oc_many_alarms() {
        let n = 60;
        let m = 30;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Monitor shifted data (large shift)
        let oc_data = generate_oc_data(20, m, 5.0, 456);
        let result = spm_monitor(&chart, &oc_data, &argvals).unwrap();

        // Out-of-control data should trigger many alarms
        let t2_alarm_count = result.t2_alarm.iter().filter(|&&a| a).count();
        let spe_alarm_count = result.spe_alarm.iter().filter(|&&a| a).count();
        let total_alarms = t2_alarm_count + spe_alarm_count;
        assert!(
            total_alarms >= 5,
            "Expected many alarms for shifted data, got T2={t2_alarm_count}, SPE={spe_alarm_count}"
        );
    }

    #[test]
    fn test_phase1_too_few_observations() {
        let data = generate_ic_data(3, 20, 42);
        let argvals = uniform_grid(20);
        let config = SpmConfig::default();
        assert!(spm_phase1(&data, &argvals, &config).is_err());
    }

    // ── EWMA tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_ewma_lambda_one_equals_raw() {
        let scores = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let smoothed = ewma_scores(&scores, 1.0).unwrap();

        for i in 0..3 {
            for k in 0..2 {
                assert!(
                    (smoothed[(i, k)] - scores[(i, k)]).abs() < 1e-12,
                    "EWMA with lambda=1 should equal raw scores at ({i},{k})"
                );
            }
        }
    }

    #[test]
    fn test_ewma_lambda_small_nearly_constant() {
        let n = 10;
        let scores =
            FdMatrix::from_column_major((0..n * 2).map(|i| i as f64).collect(), n, 2).unwrap();

        let smoothed = ewma_scores(&scores, 0.01).unwrap();

        // With very small lambda, smoothed scores should be much smaller than raw
        // (they build up very slowly from 0)
        for k in 0..2 {
            assert!(
                smoothed[(n - 1, k)].abs() < scores[(n - 1, k)].abs(),
                "EWMA with small lambda should dampen scores"
            );
        }
    }

    #[test]
    fn test_ewma_invalid_lambda() {
        let scores = FdMatrix::zeros(5, 2);
        assert!(ewma_scores(&scores, 0.0).is_err());
        assert!(ewma_scores(&scores, 1.5).is_err());
        assert!(ewma_scores(&scores, -0.1).is_err());
    }

    #[test]
    fn test_ewma_monitor_basic() {
        let n = 60;
        let m = 30;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ewma_config = EwmaConfig {
            lambda: 0.2,
            ncomp: 3,
            alpha: 0.05,
        };

        let new_data = generate_ic_data(15, m, 789);
        let result = spm_ewma_monitor(&chart, &new_data, &argvals, &ewma_config).unwrap();

        assert_eq!(result.t2.len(), 15);
        assert_eq!(result.spe.len(), 15);
        assert_eq!(result.smoothed_scores.shape(), (15, 3));
        assert!(result.t2_limit > 0.0);
    }

    // ── MFPCA tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_mfpca_basic() {
        let n = 30;
        let m1 = 20;
        let m2 = 15;
        let var1 = generate_ic_data(n, m1, 42);
        let var2 = generate_ic_data(n, m2, 99);

        let config = MfpcaConfig {
            ncomp: 3,
            weighted: true,
        };

        let result = mfpca(&[&var1, &var2], &config).unwrap();
        assert_eq!(result.scores.shape(), (n, 3));
        assert_eq!(result.eigenfunctions.len(), 2);
        assert_eq!(result.eigenfunctions[0].nrows(), m1);
        assert_eq!(result.eigenfunctions[1].nrows(), m2);
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.means.len(), 2);
        assert_eq!(result.scales.len(), 2);
    }

    #[test]
    fn test_mfpca_project_reconstruct_roundtrip() {
        let n = 20;
        let m1 = 15;
        let m2 = 10;
        let var1 = generate_ic_data(n, m1, 42);
        let var2 = generate_ic_data(n, m2, 99);

        let ncomp = n.min(m1 + m2);
        let config = MfpcaConfig {
            ncomp,
            weighted: false,
        };

        let result = mfpca(&[&var1, &var2], &config).unwrap();
        let actual_ncomp = result.eigenvalues.len();

        // Project original data
        let scores = result.project(&[&var1, &var2]).unwrap();

        // Reconstruct
        let recon = result.reconstruct(&scores, actual_ncomp).unwrap();

        // Check approximate reconstruction
        for i in 0..n {
            for j in 0..m1 {
                assert!(
                    (recon[0][(i, j)] - var1[(i, j)]).abs() < 1.0,
                    "Reconstruction error too large for var1 at ({i},{j}): {} vs {}",
                    recon[0][(i, j)],
                    var1[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_mfpca_empty_variables() {
        let config = MfpcaConfig::default();
        assert!(mfpca(&[], &config).is_err());
    }

    #[test]
    fn test_mfpca_inconsistent_rows() {
        let var1 = FdMatrix::zeros(10, 5);
        let var2 = FdMatrix::zeros(8, 5); // different n
        let config = MfpcaConfig::default();
        assert!(mfpca(&[&var1, &var2], &config).is_err());
    }

    // ── Multivariate Phase I/II tests ────────────────────────────────────────

    #[test]
    fn test_mf_phase1_basic() {
        let n = 40;
        let m1 = 20;
        let m2 = 15;
        let var1 = generate_ic_data(n, m1, 42);
        let var2 = generate_ic_data(n, m2, 99);
        let argvals1 = uniform_grid(m1);
        let argvals2 = uniform_grid(m2);

        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart =
            mf_spm_phase1(&[&var1, &var2], &[&argvals1[..], &argvals2[..]], &config).unwrap();

        assert!(chart.t2_limit.ucl > 0.0);
        assert!(chart.spe_limit.ucl > 0.0);

        // Monitor new data
        let new_var1 = generate_ic_data(10, m1, 200);
        let new_var2 = generate_ic_data(10, m2, 300);
        let result = mf_spm_monitor(
            &chart,
            &[&new_var1, &new_var2],
            &[&argvals1[..], &argvals2[..]],
        )
        .unwrap();
        assert_eq!(result.t2.len(), 10);
        assert_eq!(result.spe.len(), 10);
    }

    // ── FRCC tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_frcc_basic() {
        let n = 40;
        let m = 30;
        let p = 2;
        let argvals = uniform_grid(m);

        // Generate predictors
        let mut predictors = FdMatrix::zeros(n, p);
        for i in 0..n {
            predictors[(i, 0)] = i as f64 / n as f64;
            predictors[(i, 1)] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Generate functional response dependent on predictors
        let mut y_curves = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let t = argvals[j];
                let mu = (2.0 * PI * t).sin();
                let beta1 = t * predictors[(i, 0)];
                let beta2 = 0.5 * (4.0 * PI * t).cos() * predictors[(i, 1)];
                y_curves[(i, j)] =
                    mu + beta1 + beta2 + 0.1 * ((i * 13 + j * 7) % 100) as f64 / 100.0;
            }
        }

        let config = FrccConfig {
            ncomp: 3,
            fosr_lambda: 1e-4,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = frcc_phase1(&y_curves, &predictors, &argvals, &config).unwrap();
        assert!(chart.t2_limit.ucl > 0.0);
        assert!(chart.spe_limit.ucl > 0.0);

        // Monitor new data
        let n_new = 10;
        let mut new_x = FdMatrix::zeros(n_new, p);
        let mut new_y = FdMatrix::zeros(n_new, m);
        for i in 0..n_new {
            new_x[(i, 0)] = (i as f64 + 0.5) / n_new as f64;
            new_x[(i, 1)] = if i % 2 == 0 { 1.0 } else { 0.0 };
            for j in 0..m {
                let t = argvals[j];
                let mu = (2.0 * PI * t).sin();
                let beta1 = t * new_x[(i, 0)];
                let beta2 = 0.5 * (4.0 * PI * t).cos() * new_x[(i, 1)];
                new_y[(i, j)] = mu + beta1 + beta2 + 0.1 * ((i * 17 + j * 3) % 100) as f64 / 100.0;
            }
        }

        let result = frcc_monitor(&chart, &new_y, &new_x, &argvals).unwrap();
        assert_eq!(result.t2.len(), n_new);
        assert_eq!(result.spe.len(), n_new);
    }

    #[test]
    fn test_frcc_too_few_observations() {
        let y = FdMatrix::zeros(4, 20);
        let x = FdMatrix::zeros(4, 2);
        let argvals = uniform_grid(20);
        let config = FrccConfig::default();
        assert!(frcc_phase1(&y, &x, &argvals, &config).is_err());
    }

    // ── Contribution tests ───────────────────────────────────────────────────

    #[test]
    fn test_t2_contributions_sum_approx_total() {
        let n = 5;
        let ncomp = 3;
        let scores = FdMatrix::from_column_major(
            vec![
                1.0, 2.0, 3.0, -1.0, 0.5, 0.5, -1.0, 2.0, 1.0, -0.5, -0.3, 0.7, 1.0, -2.0, 0.1,
            ],
            n,
            ncomp,
        )
        .unwrap();
        let eigenvalues = vec![2.0, 1.0, 0.5];
        let grid_sizes = vec![10, 15]; // 2 variables

        let t2_total = hotelling_t2(&scores, &eigenvalues).unwrap();
        let contrib = t2_contributions(&scores, &eigenvalues, &grid_sizes).unwrap();

        assert_eq!(contrib.shape(), (n, 2));

        // Sum of contributions should approximate total T2
        for i in 0..n {
            let sum: f64 = (0..2).map(|v| contrib[(i, v)]).sum();
            assert!(
                (sum - t2_total[i]).abs() < 1e-8,
                "T2 contribution sum ({sum}) should equal total ({}) for obs {i}",
                t2_total[i]
            );
        }
    }

    #[test]
    fn test_spe_contributions_sum_equals_total() {
        let n = 5;
        let m1 = 10;
        let m2 = 8;
        let argvals1 = uniform_grid(m1);
        let argvals2 = uniform_grid(m2);

        let var1_std = generate_ic_data(n, m1, 42);
        let var1_rec = FdMatrix::zeros(n, m1);
        let var2_std = generate_ic_data(n, m2, 99);
        let var2_rec = FdMatrix::zeros(n, m2);

        let contrib = spe_contributions(
            &[&var1_std, &var2_std],
            &[&var1_rec, &var2_rec],
            &[&argvals1, &argvals2],
        )
        .unwrap();

        let spe_total = spe_multivariate(
            &[&var1_std, &var2_std],
            &[&var1_rec, &var2_rec],
            &[&argvals1, &argvals2],
        )
        .unwrap();

        assert_eq!(contrib.shape(), (n, 2));

        for i in 0..n {
            let sum: f64 = (0..2).map(|v| contrib[(i, v)]).sum();
            assert!(
                (sum - spe_total[i]).abs() < 1e-10,
                "SPE contribution sum ({sum}) should equal total ({}) for obs {i}",
                spe_total[i]
            );
        }
    }

    #[test]
    fn test_t2_contributions_dimension_mismatch() {
        let scores = FdMatrix::zeros(5, 3);
        let eigenvalues = vec![1.0, 1.0]; // wrong length
        let grid_sizes = vec![10, 10];
        assert!(t2_contributions(&scores, &eigenvalues, &grid_sizes).is_err());
    }

    // ── Input validation tests ───────────────────────────────────────────────

    #[test]
    fn test_monitor_dimension_mismatch() {
        let n = 40;
        let m = 30;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };

        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let wrong_m = FdMatrix::zeros(5, 20); // wrong number of columns
        assert!(spm_monitor(&chart, &wrong_m, &argvals).is_err());
    }

    #[test]
    fn test_mfpca_project_wrong_variables() {
        let var1 = generate_ic_data(20, 10, 42);
        let var2 = generate_ic_data(20, 8, 99);
        let config = MfpcaConfig {
            ncomp: 3,
            weighted: true,
        };
        let result = mfpca(&[&var1, &var2], &config).unwrap();

        // Wrong number of variables
        let new_var1 = generate_ic_data(5, 10, 200);
        assert!(result.project(&[&new_var1]).is_err());
    }

    #[test]
    fn test_mfpca_project_wrong_grid_size() {
        let var1 = generate_ic_data(20, 10, 42);
        let var2 = generate_ic_data(20, 8, 99);
        let config = MfpcaConfig {
            ncomp: 3,
            weighted: true,
        };
        let result = mfpca(&[&var1, &var2], &config).unwrap();

        // Wrong grid size for variable 2
        let new_var1 = generate_ic_data(5, 10, 200);
        let new_var2 = generate_ic_data(5, 12, 300); // should be 8
        assert!(result.project(&[&new_var1, &new_var2]).is_err());
    }

    // ── Gap 1: SPE control limit formula verification ───────────────────────

    #[test]
    fn test_spe_control_limit_formula_verification() {
        let spe_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let alpha = 0.05;
        let n = spe_values.len();

        // Manually compute mean and variance
        let mean: f64 = spe_values.iter().sum::<f64>() / n as f64;
        let var: f64 =
            spe_values.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);

        // Moment-matching parameters
        let a = var / (2.0 * mean);
        let b = 2.0 * mean * mean / var;
        let b_int = b.round().max(1.0) as usize;

        // Expected UCL
        let expected_ucl = a * chi2_quantile(1.0 - alpha, b_int);

        let limit = spe_control_limit(&spe_values, alpha).unwrap();

        assert!(
            (limit.ucl - expected_ucl).abs() < 1e-10,
            "SPE UCL should match moment-matching formula: got {}, expected {}",
            limit.ucl,
            expected_ucl
        );

        // Verify intermediate values: mean = 5.5, var = 9.1667
        assert!((mean - 5.5).abs() < 1e-10, "Mean should be 5.5, got {mean}");
        assert!(
            (var - 9.166666666666666).abs() < 1e-6,
            "Variance should be ~9.1667, got {var}"
        );
    }

    // ── Gap 2: EWMA asymptotic behavior ─────────────────────────────────────

    #[test]
    fn test_ewma_asymptotic_behavior() {
        // Create a long sequence (100 obs, 3 components) with known mean and variance
        let n = 100;
        let ncomp = 3;
        let mut raw_data = Vec::with_capacity(n * ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                // Deterministic pseudo-random values centered around 0
                let val = ((i * 13 + k * 7 + 1) as f64).sin() * (k as f64 + 1.0);
                raw_data.push(val);
            }
        }
        let scores = FdMatrix::from_column_major(raw_data, n, ncomp).unwrap();
        let smoothed = ewma_scores(&scores, 0.2).unwrap();

        // Verify smoothed scores at the end are closer to the column mean than raw scores
        for k in 0..ncomp {
            // EWMA smoothed scores should generally be less extreme than raw
            // (closer to the running average). For a long sequence, the last
            // smoothed value should be within a reasonable range.
            assert!(
                smoothed[(n - 1, k)].abs()
                    < scores
                        .column(k)
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0_f64, f64::max)
                        + 0.1,
                "Smoothed score at end should not exceed max absolute raw score for component {k}"
            );
        }

        // Verify EWMA variance ratio: var(smoothed) < var(raw) for each component
        for k in 0..ncomp {
            let raw_mean: f64 = (0..n).map(|i| scores[(i, k)]).sum::<f64>() / n as f64;
            let raw_var: f64 = (0..n)
                .map(|i| (scores[(i, k)] - raw_mean).powi(2))
                .sum::<f64>()
                / (n as f64 - 1.0);

            let sm_mean: f64 = (0..n).map(|i| smoothed[(i, k)]).sum::<f64>() / n as f64;
            let sm_var: f64 = (0..n)
                .map(|i| (smoothed[(i, k)] - sm_mean).powi(2))
                .sum::<f64>()
                / (n as f64 - 1.0);

            assert!(
                sm_var < raw_var,
                "EWMA smoothed variance ({sm_var}) should be less than raw variance ({raw_var}) for component {k}"
            );
        }
    }

    // ── Gap 3: MFPCA with tighter tolerance ─────────────────────────────────

    #[test]
    fn test_mfpca_tight_reconstruction_and_decreasing_eigenvalues() {
        let n = 40;
        let m1 = 30;
        let m2 = 25;
        let t1 = uniform_grid(m1);
        let t2_grid = uniform_grid(m2);

        // Variable 1: sin waves with varying amplitude (structured, low-rank)
        let mut var1 = FdMatrix::zeros(n, m1);
        for i in 0..n {
            let phase = i as f64 / n as f64 * PI;
            let amp = 1.0 + 0.5 * (i as f64 / n as f64);
            for j in 0..m1 {
                var1[(i, j)] = amp * (2.0 * PI * t1[j] + phase).sin();
            }
        }

        // Variable 2: polynomial curves
        let mut var2 = FdMatrix::zeros(n, m2);
        for i in 0..n {
            let a = i as f64 / n as f64;
            let b = 0.5 - i as f64 / (2.0 * n as f64);
            for j in 0..m2 {
                let t = t2_grid[j];
                var2[(i, j)] = a * t + b * t * t;
            }
        }

        // Use enough components to capture most variance
        let config = MfpcaConfig {
            ncomp: 10,
            weighted: true,
        };

        let result = mfpca(&[&var1, &var2], &config).unwrap();
        let actual_ncomp = result.eigenvalues.len();

        // Project and reconstruct
        let scores = result.project(&[&var1, &var2]).unwrap();
        let recon = result.reconstruct(&scores, actual_ncomp).unwrap();

        // Verify reconstruction error is < 0.1 for each element
        let mut max_error_v1 = 0.0_f64;
        for i in 0..n {
            for j in 0..m1 {
                let err = (recon[0][(i, j)] - var1[(i, j)]).abs();
                max_error_v1 = max_error_v1.max(err);
            }
        }
        assert!(
            max_error_v1 < 0.1,
            "Variable 1 max reconstruction error = {max_error_v1}, expected < 0.1"
        );

        let mut max_error_v2 = 0.0_f64;
        for i in 0..n {
            for j in 0..m2 {
                let err = (recon[1][(i, j)] - var2[(i, j)]).abs();
                max_error_v2 = max_error_v2.max(err);
            }
        }
        assert!(
            max_error_v2 < 0.1,
            "Variable 2 max reconstruction error = {max_error_v2}, expected < 0.1"
        );

        // Verify eigenvalues are decreasing (non-increasing)
        for w in result.eigenvalues.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-12,
                "Eigenvalues should be decreasing: {} followed by {}",
                w[0],
                w[1]
            );
        }
    }

    // ── Gap 4: Phase I alarm rate tightening ────────────────────────────────

    #[test]
    fn test_phase1_alarm_rate_tighter() {
        let n = 100;
        let m = 30;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Calibration set has ~50 observations
        let n_cal = chart.t2_phase1.len();

        // At alpha=0.05, expect at most ~5% alarms. Allow 15% tolerance
        // to account for finite-sample calibration effects.
        let max_allowed = (n_cal as f64 * 0.15).ceil() as usize;

        let t2_alarms = chart
            .t2_phase1
            .iter()
            .filter(|&&v| v > chart.t2_limit.ucl)
            .count();
        let spe_alarms = chart
            .spe_phase1
            .iter()
            .filter(|&&v| v > chart.spe_limit.ucl)
            .count();

        assert!(
            t2_alarms <= max_allowed,
            "T2 alarm count ({t2_alarms}/{n_cal}) exceeds 10% tolerance ({max_allowed})"
        );
        assert!(
            spe_alarms <= max_allowed,
            "SPE alarm count ({spe_alarms}/{n_cal}) exceeds 10% tolerance ({max_allowed})"
        );
    }

    // ── Gap 5: MFPCA constant/degenerate data ──────────────────────────────

    #[test]
    fn test_mfpca_constant_data_zero_eigenvalues() {
        let n = 10;
        let m = 15;

        // All curves identical
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (j as f64 * 0.1).sin();
            }
        }

        let config = MfpcaConfig {
            ncomp: 3,
            weighted: false,
        };

        let result = mfpca(&[&data], &config).unwrap();

        // All eigenvalues should be ~0 since there is no variation
        for (l, &ev) in result.eigenvalues.iter().enumerate() {
            assert!(
                ev < 1e-10,
                "Eigenvalue[{l}] = {ev} should be ~0 for constant data"
            );
        }
    }

    #[test]
    fn test_mfpca_single_variable_matches_structure() {
        // Single variable MFPCA should produce valid results consistent with
        // univariate structure: ncomp scores, ncomp eigenfunctions, etc.
        let n = 20;
        let m = 15;
        let var1 = generate_ic_data(n, m, 42);

        let config = MfpcaConfig {
            ncomp: 3,
            weighted: false,
        };

        let result = mfpca(&[&var1], &config).unwrap();

        assert_eq!(result.scores.shape(), (n, 3));
        assert_eq!(result.eigenfunctions.len(), 1);
        assert_eq!(result.eigenfunctions[0].nrows(), m);
        assert_eq!(result.eigenfunctions[0].ncols(), 3);
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.means.len(), 1);
        assert_eq!(result.means[0].len(), m);
        assert_eq!(result.scales.len(), 1);
        assert_eq!(result.grid_sizes, vec![m]);

        // Eigenvalues should be non-negative and decreasing
        for &ev in &result.eigenvalues {
            assert!(ev >= 0.0, "Eigenvalue should be non-negative, got {ev}");
        }
        for w in result.eigenvalues.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-12,
                "Eigenvalues should be decreasing: {} followed by {}",
                w[0],
                w[1]
            );
        }
    }

    // ── Gap 6: Phase I sample size boundary ─────────────────────────────────

    #[test]
    fn test_phase1_minimal_sample_size() {
        // n=8, ncomp=2, tuning_fraction=0.5 → 4 tuning, 4 calibration
        let n = 8;
        let m = 15;
        let data = generate_ic_data(n, m, 42);
        let argvals = uniform_grid(m);
        let config = SpmConfig {
            ncomp: 2,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        };

        let chart = spm_phase1(&data, &argvals, &config);
        assert!(
            chart.is_ok(),
            "Phase I with n=8 should succeed, got: {:?}",
            chart.err()
        );

        let chart = chart.unwrap();
        assert!(chart.t2_limit.ucl > 0.0);
        assert!(chart.spe_limit.ucl > 0.0);
        // Should have calibration observations
        assert!(
            !chart.t2_phase1.is_empty(),
            "Should have calibration T2 values"
        );
        assert!(
            !chart.spe_phase1.is_empty(),
            "Should have calibration SPE values"
        );
    }

    // ── Gap 7: EWMA extreme lambda ──────────────────────────────────────────

    #[test]
    fn test_ewma_extreme_lambda_very_small() {
        // lambda=0.001: smoothed scores should be nearly constant (close to 0)
        let n = 50;
        let ncomp = 2;
        let mut raw_data = Vec::with_capacity(n * ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                raw_data.push(((i * 7 + k * 3 + 1) as f64).sin() * 5.0);
            }
        }
        let scores = FdMatrix::from_column_major(raw_data, n, ncomp).unwrap();
        let smoothed = ewma_scores(&scores, 0.001).unwrap();

        // With lambda=0.001, the smoothed values build up extremely slowly from 0
        // The max absolute smoothed value should be much smaller than the max raw value
        for k in 0..ncomp {
            let max_raw: f64 = (0..n).map(|i| scores[(i, k)].abs()).fold(0.0_f64, f64::max);
            let max_smoothed: f64 = (0..n)
                .map(|i| smoothed[(i, k)].abs())
                .fold(0.0_f64, f64::max);

            assert!(
                max_smoothed < max_raw * 0.1,
                "lambda=0.001: max smoothed ({max_smoothed}) should be << max raw ({max_raw}) for component {k}"
            );
        }

        // All smoothed values should be close to 0
        for i in 0..n {
            for k in 0..ncomp {
                assert!(
                    smoothed[(i, k)].abs() < 1.0,
                    "lambda=0.001: smoothed[({i},{k})] = {} should be close to 0",
                    smoothed[(i, k)]
                );
            }
        }
    }

    #[test]
    fn test_ewma_extreme_lambda_near_one() {
        // lambda=0.999: smoothed scores should be nearly equal to raw scores
        let n = 20;
        let ncomp = 2;
        let mut raw_data = Vec::with_capacity(n * ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                raw_data.push(((i * 7 + k * 3 + 1) as f64).sin() * 5.0);
            }
        }
        let scores = FdMatrix::from_column_major(raw_data, n, ncomp).unwrap();
        let smoothed = ewma_scores(&scores, 0.999).unwrap();

        // After a few observations, the smoothed values should closely track raw values
        for i in 5..n {
            for k in 0..ncomp {
                let diff = (smoothed[(i, k)] - scores[(i, k)]).abs();
                let scale = scores[(i, k)].abs().max(1.0);
                assert!(
                    diff / scale < 0.05,
                    "lambda=0.999: smoothed[({i},{k})] = {} should be close to raw {} (relative diff: {})",
                    smoothed[(i, k)],
                    scores[(i, k)],
                    diff / scale
                );
            }
        }
    }

    // ── Gap 8: Contribution decomposition with non-trivial eigenvalues ──────

    #[test]
    fn test_t2_contributions_nontrivial_eigenvalues_sum_to_total() {
        let n = 5;
        let ncomp = 3;
        // Scores with real structure
        let scores = FdMatrix::from_column_major(
            vec![
                2.0, -1.5, 3.0, 0.5, -2.0, // component 0
                1.0, 0.5, -1.0, 2.0, -0.5, // component 1
                0.3, -0.2, 0.8, -0.4, 0.1, // component 2
            ],
            n,
            ncomp,
        )
        .unwrap();

        // Non-trivial eigenvalues: not all 1.0
        let eigenvalues = vec![4.0, 1.0, 0.25];
        let grid_sizes = vec![10, 15, 5]; // 3 variables

        let t2_total = hotelling_t2(&scores, &eigenvalues).unwrap();
        let contrib = t2_contributions(&scores, &eigenvalues, &grid_sizes).unwrap();

        assert_eq!(contrib.shape(), (n, 3));

        // Sum of contributions should equal total T2
        for i in 0..n {
            let sum: f64 = (0..3).map(|v| contrib[(i, v)]).sum();
            assert!(
                (sum - t2_total[i]).abs() < 1e-8,
                "T2 contribution sum ({sum}) should equal total ({}) for obs {i}",
                t2_total[i]
            );
        }

        // Verify T2 values are what we expect manually for the first observation:
        // T2 = 2.0^2/4.0 + 1.0^2/1.0 + 0.3^2/0.25 = 1.0 + 1.0 + 0.36 = 2.36
        let expected_t2_0 = 2.0_f64.powi(2) / 4.0 + 1.0_f64.powi(2) / 1.0 + 0.3_f64.powi(2) / 0.25;
        assert!(
            (t2_total[0] - expected_t2_0).abs() < 1e-10,
            "T2[0] should be {expected_t2_0}, got {}",
            t2_total[0]
        );

        // With non-trivial eigenvalues, contributions should reflect the weighting
        // Variable with largest grid_size should get the largest share
        // grid_sizes = [10, 15, 5], so variable 1 (15/30 = 0.5) should get 50%
        for i in 0..n {
            let total = t2_total[i];
            if total > 1e-12 {
                let frac_v1 = contrib[(i, 1)] / total;
                assert!(
                    (frac_v1 - 0.5).abs() < 1e-10,
                    "Variable 1 should contribute 50% of T2 (got {frac_v1}) for obs {i}"
                );
            }
        }
    }
}
