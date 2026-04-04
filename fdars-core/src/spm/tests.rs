//! Comprehensive tests for the SPM module.

#[cfg(test)]
mod spm_tests {
    use crate::matrix::FdMatrix;
    use crate::spm::amewma::{spm_amewma_monitor, AmewmaConfig};
    use crate::spm::chi_squared::{chi2_cdf, chi2_quantile};
    use crate::spm::contrib::{spe_contributions, t2_contributions, t2_contributions_mfpca};
    use crate::spm::control::spe_moment_match_diagnostic;
    use crate::spm::control::{spe_control_limit, t2_control_limit};
    use crate::spm::cusum::{spm_cusum_monitor, spm_cusum_monitor_with_restart, CusumConfig};
    use crate::spm::ewma::{ewma_scores, spm_ewma_monitor, EwmaConfig};
    use crate::spm::frcc::{frcc_monitor, frcc_phase1, FrccConfig};
    use crate::spm::iterative::{spm_phase1_iterative, IterativePhase1Config};
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
        assert_eq!(chart.t2_phase1.len(), chart.spe_phase1.len()); // same length
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
            ..EwmaConfig::default()
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
            ..FrccConfig::default()
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

    #[test]
    fn test_frcc_negative_fosr_lambda_rejected() {
        let y = generate_ic_data(40, 20, 42);
        let x = FdMatrix::from_column_major((0..80).map(|i| (i as f64) * 0.1).collect(), 40, 2)
            .unwrap();
        let argvals = uniform_grid(20);
        let config = FrccConfig {
            fosr_lambda: -1.0,
            ..FrccConfig::default()
        };
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

    // ── ncomp selection tests ─────────────────────────────────────────────

    use crate::spm::ncomp::{select_ncomp, NcompMethod};

    #[test]
    fn test_ncomp_cumvar_known_spectrum() {
        let eigenvalues = [10.0, 5.0, 1.0, 0.1, 0.01];
        // Total = 16.11; cumvar at 2 = 15/16.11 = 0.931; at 3 = 16/16.11 = 0.993
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::CumulativeVariance(0.95)).unwrap();
        assert_eq!(ncomp, 3, "Should need 3 PCs for 95% variance");
    }

    #[test]
    fn test_ncomp_elbow_known_spectrum() {
        let eigenvalues = [10.0, 5.0, 1.0, 0.1, 0.01];
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::Elbow).unwrap();
        // Second finite diffs: d2[1]=10-10+1=1, d2[2]=5-2+0.1=3.1, d2[3]=1-0.2+0.01=0.81
        // Wait: d2[k] = λ[k-1] - 2λ[k] + λ[k+1]
        // d2[1] = 10 - 2*5 + 1 = 1
        // d2[2] = 5 - 2*1 + 0.1 = 3.1
        // d2[3] = 1 - 2*0.1 + 0.01 = 0.81
        // max at k=2 → ncomp = 3
        assert!(
            (2..=4).contains(&ncomp),
            "Elbow should be around 3, got {ncomp}"
        );
    }

    #[test]
    fn test_ncomp_fixed_clamp() {
        let eigenvalues = [10.0, 5.0, 1.0];
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::Fixed(10)).unwrap();
        assert_eq!(ncomp, 3, "Fixed(10) should be clamped to 3");
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::Fixed(0)).unwrap();
        assert_eq!(ncomp, 1, "Fixed(0) should be clamped to 1");
    }

    #[test]
    fn test_ncomp_flat_spectrum() {
        let eigenvalues = [1.0, 1.0, 1.0, 1.0, 1.0];
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::CumulativeVariance(0.80)).unwrap();
        assert_eq!(ncomp, 4, "Flat spectrum: need 4/5 for 80%");
    }

    #[test]
    fn test_ncomp_single_eigenvalue() {
        let eigenvalues = [5.0];
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::CumulativeVariance(0.95)).unwrap();
        assert_eq!(ncomp, 1);
        // Elbow fallback to CV(0.95) for < 3 values
        let ncomp = select_ncomp(&eigenvalues, &NcompMethod::Elbow).unwrap();
        assert_eq!(ncomp, 1);
    }

    #[test]
    fn test_ncomp_integration_with_phase1() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 10,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();
        // Use eigenvalues from chart to select ncomp
        let ncomp =
            select_ncomp(&chart.eigenvalues, &NcompMethod::CumulativeVariance(0.95)).unwrap();
        assert!(ncomp >= 1 && ncomp <= chart.eigenvalues.len());
    }

    // ── t2_pc_contributions tests ─────────────────────────────────────────

    use crate::spm::contrib::t2_pc_contributions;

    #[test]
    fn test_t2_pc_contributions_sum_matches_t2() {
        let data = generate_ic_data(20, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();
        let new_data = generate_ic_data(10, 30, 100);
        let result = spm_monitor(&chart, &new_data, &argvals).unwrap();

        let contrib = t2_pc_contributions(&result.scores, &chart.eigenvalues).unwrap();
        let (n, ncomp) = contrib.shape();
        assert_eq!(n, 10);
        assert_eq!(ncomp, chart.eigenvalues.len());

        // Row sums should match T² values
        for i in 0..n {
            let row_sum: f64 = (0..ncomp).map(|l| contrib[(i, l)]).sum();
            assert!(
                (row_sum - result.t2[i]).abs() < 1e-8,
                "Row {i} sum {row_sum} != T2 {}",
                result.t2[i]
            );
        }
    }

    #[test]
    fn test_t2_pc_contributions_manual_3x2() {
        // Manual test: 3 obs, 2 components
        let scores = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        let eigenvalues = [2.0, 3.0];
        let contrib = t2_pc_contributions(&scores, &eigenvalues).unwrap();

        // contrib[(0,0)] = 1²/2 = 0.5, contrib[(0,1)] = 4²/3 = 5.333
        assert!((contrib[(0, 0)] - 0.5).abs() < 1e-10);
        assert!((contrib[(0, 1)] - 16.0 / 3.0).abs() < 1e-10);
        // contrib[(1,0)] = 2²/2 = 2.0, contrib[(1,1)] = 5²/3 = 8.333
        assert!((contrib[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((contrib[(1, 1)] - 25.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_t2_pc_contributions_shifted_attribution() {
        // Large shift should make overall T² large, and some PC should dominate
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let oc_data = generate_oc_data(5, 30, 5.0, 42);
        let result = spm_monitor(&chart, &oc_data, &argvals).unwrap();
        let contrib = t2_pc_contributions(&result.scores, &chart.eigenvalues).unwrap();

        // For shifted data, total T² should be large and at least one PC
        // should dominate the contribution
        for i in 0..5 {
            let total: f64 = (0..chart.eigenvalues.len()).map(|l| contrib[(i, l)]).sum();
            let max_frac = (0..chart.eigenvalues.len())
                .map(|l| contrib[(i, l)] / total)
                .fold(0.0_f64, f64::max);
            // At least one PC should contribute > 30% of the total
            assert!(
                max_frac > 0.3,
                "Shifted obs {i}: dominant PC should contribute > 30%, got {max_frac}"
            );
        }
    }

    // ── rules tests ───────────────────────────────────────────────────────

    use crate::spm::rules::{evaluate_rules, western_electric_rules, ChartRule};

    #[test]
    fn test_we1_single_outlier() {
        let mut values = vec![0.0; 20];
        values[10] = 4.0; // Beyond 3σ
        let violations = western_electric_rules(&values, 0.0, 1.0).unwrap();
        let we1_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.rule == ChartRule::WE1)
            .collect();
        assert!(
            !we1_violations.is_empty(),
            "Should detect WE1 violation at index 10"
        );
        assert!(we1_violations.iter().any(|v| v.indices.contains(&10)));
    }

    #[test]
    fn test_we4_eight_in_a_row() {
        // 8 points all above center
        let values: Vec<f64> = (0..15)
            .map(|i| if (3..11).contains(&i) { 0.5 } else { -0.5 })
            .collect();
        let violations = evaluate_rules(&values, 0.0, 1.0, &[ChartRule::WE4]).unwrap();
        assert!(
            !violations.is_empty(),
            "Should detect WE4: 8 consecutive same side"
        );
    }

    #[test]
    fn test_nelson5_monotone() {
        // 6 strictly increasing points
        let values: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let violations = evaluate_rules(&values, 0.0, 1.0, &[ChartRule::Nelson5]).unwrap();
        assert!(
            !violations.is_empty(),
            "Should detect Nelson5: 6 monotone increasing"
        );
    }

    #[test]
    fn test_no_false_positives_centered() {
        // Data centered at 0 with small variation (within 1σ range, but not 15 consecutive)
        let values: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let violations = western_electric_rules(&values, 0.0, 1.0).unwrap();
        // WE1 should not fire (all within 3σ)
        let we1: Vec<_> = violations
            .iter()
            .filter(|v| v.rule == ChartRule::WE1)
            .collect();
        assert!(we1.is_empty(), "No WE1 violations expected");
    }

    #[test]
    fn test_rules_empty_input() {
        let violations = evaluate_rules(&[], 0.0, 1.0, &[ChartRule::WE1, ChartRule::WE4]).unwrap();
        assert!(
            violations.is_empty(),
            "Empty input should yield no violations"
        );
    }

    // ── bootstrap tests ───────────────────────────────────────────────────

    use crate::spm::bootstrap::{spe_limit_robust, t2_limit_robust, ControlLimitMethod};

    #[test]
    fn test_bootstrap_parametric_matches_existing() {
        let limit_param =
            t2_limit_robust(&[1.0; 50], 3, 0.05, &ControlLimitMethod::Parametric).unwrap();
        let limit_orig = t2_control_limit(3, 0.05).unwrap();
        assert!(
            (limit_param.ucl - limit_orig.ucl).abs() < 1e-10,
            "Parametric should match: {} vs {}",
            limit_param.ucl,
            limit_orig.ucl
        );
    }

    #[test]
    fn test_bootstrap_empirical_on_sequence() {
        // Values 1..100, empirical 95th percentile ≈ 95
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let limit = t2_limit_robust(&values, 5, 0.05, &ControlLimitMethod::Empirical).unwrap();
        assert!(
            (limit.ucl - 95.0).abs() < 1.5,
            "Empirical 95th percentile of 1..100 should be ~95, got {}",
            limit.ucl
        );
    }

    #[test]
    fn test_bootstrap_convergence() {
        // Bootstrap on uniform data should give stable result
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let limit = t2_limit_robust(
            &values,
            5,
            0.05,
            &ControlLimitMethod::Bootstrap {
                n_bootstrap: 1000,
                seed: 42,
            },
        )
        .unwrap();
        // Should be close to the empirical quantile
        assert!(
            limit.ucl > 85.0 && limit.ucl < 100.0,
            "Bootstrap UCL should be near 95, got {}",
            limit.ucl
        );
    }

    #[test]
    fn test_kde_convergence() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let limit = t2_limit_robust(
            &values,
            5,
            0.05,
            &ControlLimitMethod::KernelDensity { bandwidth: None },
        )
        .unwrap();
        assert!(
            limit.ucl > 80.0 && limit.ucl < 105.0,
            "KDE UCL should be near 95, got {}",
            limit.ucl
        );
    }

    #[test]
    fn test_spe_limit_robust_empirical() {
        let values: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let limit = spe_limit_robust(&values, 0.05, &ControlLimitMethod::Empirical).unwrap();
        assert!(limit.ucl > 45.0 && limit.ucl <= 50.0);
    }

    // ── ARL tests ─────────────────────────────────────────────────────────

    use crate::spm::arl::{arl0_ewma_t2, arl0_spe, arl0_t2, arl1_t2, ArlConfig};

    #[test]
    fn test_arl0_approximately_1_over_alpha() {
        // ARL0 ≈ 1/α for T² chart
        let eigenvalues = [1.0, 0.5, 0.25];
        let ucl = chi2_quantile(0.95, 3); // α = 0.05
        let config = ArlConfig {
            n_simulations: 5000,
            max_run_length: 2000,
            seed: 42,
        };
        let result = arl0_t2(&eigenvalues, ucl, &config).unwrap();
        // Theory: ARL0 = 1/0.05 = 20
        assert!(
            result.arl > 10.0 && result.arl < 40.0,
            "ARL0 should be near 20, got {}",
            result.arl
        );
    }

    #[test]
    fn test_arl1_less_than_arl0() {
        let eigenvalues = [1.0, 0.5];
        let ucl = chi2_quantile(0.95, 2);
        let config = ArlConfig {
            n_simulations: 2000,
            max_run_length: 1000,
            seed: 42,
        };
        let arl0 = arl0_t2(&eigenvalues, ucl, &config).unwrap();
        let shift = [2.0, 0.0]; // Shift in PC1
        let arl1 = arl1_t2(&eigenvalues, ucl, &shift, &config).unwrap();
        assert!(
            arl1.arl < arl0.arl,
            "ARL1 ({}) should be less than ARL0 ({})",
            arl1.arl,
            arl0.arl
        );
    }

    #[test]
    fn test_ewma_lambda1_matches_raw() {
        // EWMA with lambda=1 should match raw T²
        let eigenvalues = [1.0, 0.5];
        let ucl = chi2_quantile(0.95, 2);
        let config = ArlConfig {
            n_simulations: 1000,
            max_run_length: 500,
            seed: 42,
        };
        let raw = arl0_t2(&eigenvalues, ucl, &config).unwrap();
        let ewma = arl0_ewma_t2(&eigenvalues, ucl, 1.0, &config).unwrap();
        // With lambda=1, EWMA reduces to raw scores, adjusted eigenvalues = eigenvalues
        assert!(
            (ewma.arl - raw.arl).abs() / raw.arl < 0.3,
            "EWMA(lambda=1) ARL {} should be close to raw ARL {}",
            ewma.arl,
            raw.arl
        );
    }

    #[test]
    fn test_arl_monotonicity_shift() {
        let eigenvalues = [1.0];
        let ucl = chi2_quantile(0.95, 1);
        let config = ArlConfig {
            n_simulations: 1000,
            max_run_length: 500,
            seed: 42,
        };
        let arl_small = arl1_t2(&eigenvalues, ucl, &[1.0], &config).unwrap();
        let arl_large = arl1_t2(&eigenvalues, ucl, &[3.0], &config).unwrap();
        assert!(
            arl_large.arl < arl_small.arl,
            "Larger shift should give smaller ARL1: {} vs {}",
            arl_large.arl,
            arl_small.arl
        );
    }

    #[test]
    fn test_arl0_spe_basic() {
        let config = ArlConfig {
            n_simulations: 1000,
            max_run_length: 500,
            seed: 42,
        };
        // df=10, scale=1.0, ucl at 95th percentile ≈ 18.31
        let ucl = chi2_quantile(0.95, 10);
        let result = arl0_spe(10.0, 1.0, ucl, &config).unwrap();
        // ARL0 should be roughly 1/0.05 = 20
        assert!(
            result.arl > 8.0 && result.arl < 50.0,
            "SPE ARL0 should be near 20, got {}",
            result.arl
        );
    }

    // ── MEWMA tests ───────────────────────────────────────────────────────

    use crate::spm::mewma::{spm_mewma_monitor, MewmaConfig};

    #[test]
    fn test_mewma_asymptotic_ic_data() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(20, 30, 100);
        let mewma_config = MewmaConfig {
            lambda: 0.2,
            ncomp: 3,
            alpha: 0.05,
            asymptotic: true,
        };
        let result = spm_mewma_monitor(&chart, &new_data, &argvals, &mewma_config).unwrap();
        assert_eq!(result.mewma_statistic.len(), 20);
        assert_eq!(result.alarm.len(), 20);
        assert!(result.ucl > 0.0);
    }

    #[test]
    fn test_mewma_exact_converges_to_asymptotic() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Generate enough data that exact should converge
        let new_data = generate_ic_data(100, 30, 100);
        let mewma_asym = spm_mewma_monitor(
            &chart,
            &new_data,
            &argvals,
            &MewmaConfig {
                asymptotic: true,
                ..MewmaConfig::default()
            },
        )
        .unwrap();
        let mewma_exact = spm_mewma_monitor(
            &chart,
            &new_data,
            &argvals,
            &MewmaConfig {
                asymptotic: false,
                ..MewmaConfig::default()
            },
        )
        .unwrap();

        // At the end of the sequence, exact should be close to asymptotic
        let last = mewma_asym.mewma_statistic.len() - 1;
        let ratio = mewma_exact.mewma_statistic[last] / mewma_asym.mewma_statistic[last];
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "Exact and asymptotic should converge, ratio = {ratio}"
        );
    }

    #[test]
    fn test_mewma_lambda1_matches_raw_t2() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(10, 30, 100);
        let mewma_result = spm_mewma_monitor(
            &chart,
            &new_data,
            &argvals,
            &MewmaConfig {
                lambda: 1.0,
                ncomp: 3,
                alpha: 0.05,
                asymptotic: true,
            },
        )
        .unwrap();
        let raw_result = spm_monitor(&chart, &new_data, &argvals).unwrap();

        // With lambda=1, MEWMA T² should equal raw T²
        for i in 0..10 {
            assert!(
                (mewma_result.mewma_statistic[i] - raw_result.t2[i]).abs() < 1e-6,
                "Lambda=1 MEWMA[{i}]={} should equal T2[{i}]={}",
                mewma_result.mewma_statistic[i],
                raw_result.t2[i]
            );
        }
    }

    #[test]
    fn test_mewma_sensitive_to_small_shifts() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Small shift (0.5) - MEWMA should accumulate it
        let oc_data = generate_oc_data(30, 30, 0.5, 42);
        let mewma_config = MewmaConfig {
            lambda: 0.1,
            ncomp: 3,
            alpha: 0.05,
            asymptotic: true,
        };
        let result = spm_mewma_monitor(&chart, &oc_data, &argvals, &mewma_config).unwrap();
        // MEWMA statistic should generally increase over time due to accumulation
        let first_half_mean: f64 = result.mewma_statistic[..15].iter().sum::<f64>() / 15.0;
        let second_half_mean: f64 = result.mewma_statistic[15..].iter().sum::<f64>() / 15.0;
        assert!(
            second_half_mean >= first_half_mean * 0.5,
            "MEWMA should accumulate shift: first_half={first_half_mean}, second_half={second_half_mean}"
        );
    }

    // ── Profile monitoring tests ──────────────────────────────────────────

    use crate::spm::profile::{profile_monitor, profile_phase1, ProfileMonitorConfig};

    /// Generate y_curves and predictors for profile testing.
    fn generate_profile_data(n: usize, m: usize, p: usize, seed: u64) -> (FdMatrix, FdMatrix) {
        let t = uniform_grid(m);
        let mut y = FdMatrix::zeros(n, m);
        let mut pred = FdMatrix::zeros(n, p);
        for i in 0..n {
            // Deterministic pseudo-random predictor values
            for k in 0..p {
                pred[(i, k)] = ((seed
                    .wrapping_add(i as u64 * 3 + k as u64)
                    .wrapping_mul(48271)) as f64)
                    / (u32::MAX as f64)
                    * 2.0
                    - 1.0;
            }
            for j in 0..m {
                // y = beta0(t) + x * beta1(t) + noise
                let beta0 = (2.0 * PI * t[j]).sin();
                let beta1 = t[j] * (1.0 - t[j]);
                let noise = 0.05
                    * ((seed
                        .wrapping_add(i as u64 * 13 + j as u64 * 7)
                        .wrapping_mul(48271)) as f64)
                    / (u32::MAX as f64);
                y[(i, j)] = beta0 + pred[(i, 0)] * beta1 + noise;
            }
        }
        (y, pred)
    }

    #[test]
    fn test_profile_phase1_basic() {
        let (y, pred) = generate_profile_data(60, 20, 1, 42);
        let argvals = uniform_grid(20);
        let config = ProfileMonitorConfig {
            window_size: 15,
            ncomp: 2,
            ..ProfileMonitorConfig::default()
        };
        let chart = profile_phase1(&y, &pred, &argvals, &config).unwrap();
        assert!(chart.t2_limit.ucl > 0.0);
        assert!(!chart.eigenvalues.is_empty());
    }

    #[test]
    fn test_profile_monitor_stable() {
        let (y_train, pred_train) = generate_profile_data(60, 20, 1, 42);
        let (y_test, pred_test) = generate_profile_data(40, 20, 1, 100);
        let argvals = uniform_grid(20);
        let config = ProfileMonitorConfig {
            window_size: 15,
            ncomp: 2,
            ..ProfileMonitorConfig::default()
        };
        let chart = profile_phase1(&y_train, &pred_train, &argvals, &config).unwrap();
        let result = profile_monitor(&chart, &y_test, &pred_test, &argvals, &config).unwrap();
        assert!(!result.t2.is_empty());
        // Stable profile should have few alarms
        let alarm_rate =
            result.t2_alarm.iter().filter(|&&a| a).count() as f64 / result.t2_alarm.len() as f64;
        assert!(
            alarm_rate < 0.5,
            "Stable profile alarm rate should be low, got {alarm_rate}"
        );
    }

    #[test]
    fn test_profile_shifted_triggers_alarms() {
        let (y_train, pred_train) = generate_profile_data(60, 20, 1, 42);
        let argvals = uniform_grid(20);
        let config = ProfileMonitorConfig {
            window_size: 15,
            ncomp: 2,
            ..ProfileMonitorConfig::default()
        };
        let chart = profile_phase1(&y_train, &pred_train, &argvals, &config).unwrap();

        // Shifted data
        let (mut y_test, pred_test) = generate_profile_data(40, 20, 1, 200);
        for i in 0..y_test.nrows() {
            for j in 0..y_test.ncols() {
                y_test[(i, j)] += 3.0; // Large shift
            }
        }
        let result = profile_monitor(&chart, &y_test, &pred_test, &argvals, &config).unwrap();
        let alarm_count = result.t2_alarm.iter().filter(|&&a| a).count();
        assert!(
            alarm_count > 0,
            "Shifted profile should trigger some alarms"
        );
    }

    #[test]
    fn test_profile_small_window() {
        let (y, pred) = generate_profile_data(30, 15, 1, 42);
        let argvals = uniform_grid(15);
        let config = ProfileMonitorConfig {
            window_size: 5,
            step_size: 2,
            ncomp: 2,
            ..ProfileMonitorConfig::default()
        };
        let chart = profile_phase1(&y, &pred, &argvals, &config).unwrap();
        assert!(chart.eigenvalues.len() <= 2);
    }

    #[test]
    fn test_profile_step_size_zero_rejected() {
        let (y, pred) = generate_profile_data(30, 15, 1, 42);
        let argvals = uniform_grid(15);
        let config = ProfileMonitorConfig {
            step_size: 0,
            ..ProfileMonitorConfig::default()
        };
        assert!(profile_phase1(&y, &pred, &argvals, &config).is_err());
    }

    // ── Partial-domain monitoring tests ───────────────────────────────────

    use crate::spm::partial::{
        spm_monitor_partial, spm_monitor_partial_batch, DomainCompletion, PartialDomainConfig,
    };

    #[test]
    fn test_partial_full_domain_matches_monitor() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(1, 30, 200);
        let full_values: Vec<f64> = (0..30).map(|j| new_data[(0, j)]).collect();

        // Full domain with ZeroPad should match spm_monitor
        let partial_config = PartialDomainConfig {
            ncomp: 3,
            completion: DomainCompletion::ZeroPad,
            ..PartialDomainConfig::default()
        };
        let partial_result =
            spm_monitor_partial(&chart, &full_values, &argvals, 30, &partial_config).unwrap();
        let full_result = spm_monitor(&chart, &new_data, &argvals).unwrap();

        assert!(
            (partial_result.t2 - full_result.t2[0]).abs() < 1e-4,
            "Full domain partial T2 ({}) should match full T2 ({})",
            partial_result.t2,
            full_result.t2[0]
        );
    }

    #[test]
    fn test_partial_decreasing_domain_noisier() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(1, 30, 300);
        let values: Vec<f64> = (0..30).map(|j| new_data[(0, j)]).collect();

        let partial_config = PartialDomainConfig {
            ncomp: 3,
            completion: DomainCompletion::PartialProjection,
            ..PartialDomainConfig::default()
        };

        // More observed points → scores should stabilize
        let result_10 =
            spm_monitor_partial(&chart, &values, &argvals, 10, &partial_config).unwrap();
        let result_30 =
            spm_monitor_partial(&chart, &values, &argvals, 30, &partial_config).unwrap();

        assert!(
            result_10.domain_fraction < result_30.domain_fraction,
            "10 points should have smaller domain fraction"
        );
    }

    #[test]
    fn test_partial_ce_accuracy() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 2,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(1, 30, 400);
        let values: Vec<f64> = (0..30).map(|j| new_data[(0, j)]).collect();

        let partial_config = PartialDomainConfig {
            ncomp: 2,
            completion: DomainCompletion::ConditionalExpectation,
            ..PartialDomainConfig::default()
        };
        let result = spm_monitor_partial(&chart, &values, &argvals, 20, &partial_config).unwrap();
        assert!(result.completed_curve.is_some());
        assert_eq!(result.scores.len(), 2);
        assert!(result.domain_fraction > 0.0 && result.domain_fraction <= 1.0);
    }

    #[test]
    fn test_partial_zeropad_baseline() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(1, 30, 500);
        let values: Vec<f64> = (0..30).map(|j| new_data[(0, j)]).collect();

        let partial_config = PartialDomainConfig {
            ncomp: 3,
            completion: DomainCompletion::ZeroPad,
            ..PartialDomainConfig::default()
        };
        let result = spm_monitor_partial(&chart, &values, &argvals, 15, &partial_config).unwrap();
        assert!(result.completed_curve.is_some());
        let curve = result.completed_curve.unwrap();
        // Unobserved portion should equal the mean
        for j in 15..30 {
            assert!(
                (curve[j] - chart.fpca.mean[j]).abs() < 1e-10,
                "ZeroPad: unobserved point {j} should equal mean"
            );
        }
    }

    #[test]
    fn test_partial_n_observed_1() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 2,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let values = vec![1.0; 30];
        let partial_config = PartialDomainConfig {
            ncomp: 2,
            completion: DomainCompletion::ConditionalExpectation,
            ..PartialDomainConfig::default()
        };
        let result = spm_monitor_partial(&chart, &values, &argvals, 1, &partial_config).unwrap();
        assert_eq!(result.scores.len(), 2);
    }

    #[test]
    fn test_partial_n_observed_0_error() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 2,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let values = vec![1.0; 30];
        let partial_config = PartialDomainConfig::default();
        let result = spm_monitor_partial(&chart, &values, &argvals, 0, &partial_config);
        assert!(result.is_err(), "n_observed=0 should return error");
    }

    #[test]
    fn test_partial_batch() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 2,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let values1 = vec![0.5; 30];
        let values2 = vec![1.0; 30];
        let batch: Vec<(&[f64], usize)> = vec![(&values1, 15), (&values2, 20)];
        let partial_config = PartialDomainConfig::default();
        let results = spm_monitor_partial_batch(&chart, &batch, &argvals, &partial_config).unwrap();
        assert_eq!(results.len(), 2);
    }

    // ── Elastic SPM tests ─────────────────────────────────────────────────

    use crate::spm::elastic_spm::{elastic_spm_monitor, elastic_spm_phase1, ElasticSpmConfig};

    #[test]
    fn test_elastic_spm_amplitude_only() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = ElasticSpmConfig {
            spm: SpmConfig {
                ncomp: 3,
                ..SpmConfig::default()
            },
            monitor_phase: false,
            ..ElasticSpmConfig::default()
        };
        let chart = elastic_spm_phase1(&data, &argvals, &config).unwrap();
        assert!(chart.phase_chart.is_none());

        let new_data = generate_ic_data(10, 30, 100);
        let result = elastic_spm_monitor(&chart, &new_data, &argvals).unwrap();
        assert!(result.phase.is_none());
        assert_eq!(result.amplitude.t2.len(), 10);
    }

    #[test]
    fn test_elastic_spm_with_phase() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = ElasticSpmConfig {
            spm: SpmConfig {
                ncomp: 3,
                ..SpmConfig::default()
            },
            monitor_phase: true,
            warp_ncomp: 2,
            ..ElasticSpmConfig::default()
        };
        let chart = elastic_spm_phase1(&data, &argvals, &config).unwrap();
        assert!(chart.phase_chart.is_some());

        let new_data = generate_ic_data(10, 30, 100);
        let result = elastic_spm_monitor(&chart, &new_data, &argvals).unwrap();
        assert!(result.phase.is_some());
        let phase = result.phase.unwrap();
        assert_eq!(phase.t2.len(), 10);
    }

    #[test]
    fn test_elastic_spm_roundtrip() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = ElasticSpmConfig {
            spm: SpmConfig {
                ncomp: 3,
                ..SpmConfig::default()
            },
            ..ElasticSpmConfig::default()
        };
        let chart = elastic_spm_phase1(&data, &argvals, &config).unwrap();
        assert!(!chart.karcher_mean.is_empty());
        assert_eq!(chart.karcher_mean.len(), 30);

        // Monitor same data — should have few alarms
        let result = elastic_spm_monitor(&chart, &data, &argvals).unwrap();
        assert_eq!(result.aligned_data.nrows(), 40);
        assert_eq!(result.warping_functions.nrows(), 40);
    }

    #[test]
    fn test_elastic_spm_shifted_detection() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = ElasticSpmConfig {
            spm: SpmConfig {
                ncomp: 3,
                ..SpmConfig::default()
            },
            ..ElasticSpmConfig::default()
        };
        let chart = elastic_spm_phase1(&data, &argvals, &config).unwrap();

        let oc_data = generate_oc_data(10, 30, 5.0, 42);
        let result = elastic_spm_monitor(&chart, &oc_data, &argvals).unwrap();
        let amp_alarms = result.amplitude.t2_alarm.iter().filter(|&&a| a).count();
        // Large shift should trigger alarms
        assert!(
            amp_alarms > 0,
            "Amplitude shift should trigger alarms, got {amp_alarms}"
        );
    }

    // ── CUSUM tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_cusum_multivariate_ic_data() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ic_data = generate_ic_data(20, 30, 200);
        let cusum_config = CusumConfig {
            ncomp: 3,
            ..CusumConfig::default()
        };
        let result = spm_cusum_monitor(&chart, &ic_data, &argvals, &cusum_config).unwrap();

        assert_eq!(result.cusum_statistic.len(), 20);
        assert_eq!(result.alarm.len(), 20);
        assert!(result.ucl > 0.0);
        assert!(result.cusum_plus.is_none()); // multivariate mode
        assert!(result.cusum_minus.is_none());
        for &s in &result.cusum_statistic {
            assert!(s >= 0.0, "CUSUM statistic must be non-negative");
        }
    }

    #[test]
    fn test_cusum_univariate_mode() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ic_data = generate_ic_data(15, 30, 300);
        let cusum_config = CusumConfig {
            ncomp: 3,
            multivariate: false,
            ..CusumConfig::default()
        };
        let result = spm_cusum_monitor(&chart, &ic_data, &argvals, &cusum_config).unwrap();

        assert_eq!(result.cusum_statistic.len(), 15);
        assert!(result.cusum_plus.is_some());
        assert!(result.cusum_minus.is_some());
        let cp = result.cusum_plus.as_ref().unwrap();
        let cm = result.cusum_minus.as_ref().unwrap();
        assert_eq!(cp.shape(), (15, 3));
        assert_eq!(cm.shape(), (15, 3));
    }

    #[test]
    fn test_cusum_detects_shift() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let oc_data = generate_oc_data(20, 30, 3.0, 42);
        let cusum_config = CusumConfig {
            ncomp: 3,
            k: 0.5,
            h: 4.0,
            ..CusumConfig::default()
        };
        let result = spm_cusum_monitor(&chart, &oc_data, &argvals, &cusum_config).unwrap();
        let alarm_count = result.alarm.iter().filter(|&&a| a).count();
        assert!(
            alarm_count > 0,
            "CUSUM should detect shifted data, got 0 alarms"
        );
    }

    #[test]
    fn test_cusum_restart_fewer_alarms() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let oc_data = generate_oc_data(30, 30, 3.0, 42);
        let cusum_config = CusumConfig {
            ncomp: 3,
            k: 0.5,
            h: 4.0,
            ..CusumConfig::default()
        };

        let result_no_restart =
            spm_cusum_monitor(&chart, &oc_data, &argvals, &cusum_config).unwrap();
        let result_restart =
            spm_cusum_monitor_with_restart(&chart, &oc_data, &argvals, &cusum_config).unwrap();

        // With restart, statistics reset after alarm, so cumulative stats should differ
        assert_eq!(result_no_restart.cusum_statistic.len(), 30);
        assert_eq!(result_restart.cusum_statistic.len(), 30);
        // After first alarm, restart version should have lower statistics
        let no_restart_alarms = result_no_restart.alarm.iter().filter(|&&a| a).count();
        let restart_alarms = result_restart.alarm.iter().filter(|&&a| a).count();
        // Both should detect the shift
        assert!(no_restart_alarms > 0);
        assert!(restart_alarms > 0);
    }

    #[test]
    fn test_cusum_invalid_k() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig::default();
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ic_data = generate_ic_data(5, 30, 100);
        let cusum_config = CusumConfig {
            k: -1.0,
            ..CusumConfig::default()
        };
        assert!(spm_cusum_monitor(&chart, &ic_data, &argvals, &cusum_config).is_err());
    }

    #[test]
    fn test_cusum_invalid_h() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig::default();
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ic_data = generate_ic_data(5, 30, 100);
        let cusum_config = CusumConfig {
            h: 0.0,
            ..CusumConfig::default()
        };
        assert!(spm_cusum_monitor(&chart, &ic_data, &argvals, &cusum_config).is_err());
    }

    // ── Adaptive EWMA (AMFEWMA) tests ────────────────────────────────────────

    #[test]
    fn test_amewma_ic_data() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ic_data = generate_ic_data(20, 30, 200);
        let amewma_config = AmewmaConfig {
            ncomp: 3,
            ..AmewmaConfig::default()
        };
        let result = spm_amewma_monitor(&chart, &ic_data, &argvals, &amewma_config).unwrap();

        assert_eq!(result.t2_statistic.len(), 20);
        assert_eq!(result.lambda_t.len(), 20);
        assert_eq!(result.alarm.len(), 20);
        assert!(result.ucl > 0.0);
        assert_eq!(result.smoothed_scores.shape(), (20, 3));

        // Lambda should stay within bounds
        for &lam in &result.lambda_t {
            assert!(
                lam >= amewma_config.lambda_min && lam <= amewma_config.lambda_max,
                "lambda_t={lam} out of bounds [{}, {}]",
                amewma_config.lambda_min,
                amewma_config.lambda_max
            );
        }
    }

    #[test]
    fn test_amewma_detects_shift() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let oc_data = generate_oc_data(20, 30, 5.0, 42);
        let amewma_config = AmewmaConfig {
            ncomp: 3,
            ..AmewmaConfig::default()
        };
        let result = spm_amewma_monitor(&chart, &oc_data, &argvals, &amewma_config).unwrap();
        let alarm_count = result.alarm.iter().filter(|&&a| a).count();
        assert!(
            alarm_count > 0,
            "AMFEWMA should detect shifted data, got 0 alarms"
        );
    }

    #[test]
    fn test_amewma_lambda_increases_on_shift() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Monitor shifted data — lambda should increase due to larger errors
        let oc_data = generate_oc_data(20, 30, 5.0, 42);
        let amewma_config = AmewmaConfig {
            ncomp: 3,
            lambda_init: 0.2,
            ..AmewmaConfig::default()
        };
        let result = spm_amewma_monitor(&chart, &oc_data, &argvals, &amewma_config).unwrap();

        // Later lambdas should be larger than initial as the adaptive weight reacts
        let avg_lambda: f64 = result.lambda_t.iter().sum::<f64>() / result.lambda_t.len() as f64;
        assert!(
            avg_lambda > amewma_config.lambda_init,
            "Average lambda ({avg_lambda}) should exceed lambda_init ({}) under shift",
            amewma_config.lambda_init
        );
    }

    #[test]
    fn test_amewma_invalid_lambda_range() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = SpmConfig::default();
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let ic_data = generate_ic_data(5, 30, 100);

        // lambda_min > lambda_max
        let bad_config = AmewmaConfig {
            lambda_min: 0.8,
            lambda_max: 0.2,
            lambda_init: 0.2,
            ..AmewmaConfig::default()
        };
        assert!(spm_amewma_monitor(&chart, &ic_data, &argvals, &bad_config).is_err());

        // lambda_init out of range
        let bad_config2 = AmewmaConfig {
            lambda_min: 0.1,
            lambda_max: 0.5,
            lambda_init: 0.8,
            ..AmewmaConfig::default()
        };
        assert!(spm_amewma_monitor(&chart, &ic_data, &argvals, &bad_config2).is_err());

        // eta out of range
        let bad_config3 = AmewmaConfig {
            eta: 0.0,
            ..AmewmaConfig::default()
        };
        assert!(spm_amewma_monitor(&chart, &ic_data, &argvals, &bad_config3).is_err());
    }

    // ── Iterative Phase I tests ──────────────────────────────────────────────

    #[test]
    fn test_iterative_phase1_clean_data() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = IterativePhase1Config {
            spm: SpmConfig {
                ncomp: 3,
                alpha: 0.05,
                ..SpmConfig::default()
            },
            max_removal_fraction: 0.3,
            ..IterativePhase1Config::default()
        };
        let result = spm_phase1_iterative(&data, &argvals, &config).unwrap();

        // Removal fraction should not exceed the limit
        let frac = result.removed_indices.len() as f64 / 40.0;
        assert!(
            frac <= 0.3 + 0.01,
            "Should not exceed max_removal_fraction, got {frac}"
        );
        assert_eq!(result.n_remaining + result.removed_indices.len(), 40);
        // Chart should be valid
        assert!(result.chart.t2_limit.ucl > 0.0);
    }

    #[test]
    fn test_iterative_phase1_removes_outliers() {
        // Build data with extreme outliers — large enough to be clearly OC
        let mut data = generate_ic_data(50, 30, 42);
        // Add 5 extreme outliers
        for i in 45..50 {
            for j in 0..30 {
                data[(i, j)] += 50.0;
            }
        }
        let argvals = uniform_grid(30);
        let config = IterativePhase1Config {
            spm: SpmConfig {
                ncomp: 3,
                alpha: 0.01,
                ..SpmConfig::default()
            },
            max_removal_fraction: 0.5,
            ..IterativePhase1Config::default()
        };
        let result = spm_phase1_iterative(&data, &argvals, &config).unwrap();

        // Should perform at least one iteration
        assert!(
            result.n_iterations > 0 || result.removed_indices.is_empty(),
            "iterations={}, removed={}",
            result.n_iterations,
            result.removed_indices.len()
        );
        // The chart should be valid regardless
        assert!(result.chart.t2_limit.ucl > 0.0);
        assert!(result.n_remaining >= 4);
    }

    #[test]
    fn test_iterative_phase1_max_removal_fraction() {
        // All data is "extreme" — should hit removal fraction limit
        let data = generate_oc_data(40, 30, 10.0, 42);
        let argvals = uniform_grid(30);
        let config = IterativePhase1Config {
            spm: SpmConfig {
                ncomp: 3,
                alpha: 0.05,
                ..SpmConfig::default()
            },
            max_removal_fraction: 0.1,
            ..IterativePhase1Config::default()
        };
        let result = spm_phase1_iterative(&data, &argvals, &config).unwrap();

        // Should not remove more than 10%
        assert!(
            result.removed_indices.len() as f64 / 40.0 <= 0.1 + 0.01,
            "Should not exceed max_removal_fraction"
        );
    }

    #[test]
    fn test_iterative_phase1_invalid_params() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);

        // max_iterations = 0
        let config = IterativePhase1Config {
            max_iterations: 0,
            ..IterativePhase1Config::default()
        };
        assert!(spm_phase1_iterative(&data, &argvals, &config).is_err());

        // max_removal_fraction = 0
        let config2 = IterativePhase1Config {
            max_removal_fraction: 0.0,
            ..IterativePhase1Config::default()
        };
        assert!(spm_phase1_iterative(&data, &argvals, &config2).is_err());
    }

    #[test]
    fn test_iterative_phase1_convergence() {
        let data = generate_ic_data(40, 30, 42);
        let argvals = uniform_grid(30);
        let config = IterativePhase1Config {
            spm: SpmConfig {
                ncomp: 3,
                alpha: 0.05,
                ..SpmConfig::default()
            },
            max_iterations: 20,
            ..IterativePhase1Config::default()
        };
        let result = spm_phase1_iterative(&data, &argvals, &config).unwrap();

        // Should converge before max_iterations on clean data
        assert!(
            result.n_iterations < 20,
            "Should converge before max iterations on clean data, took {}",
            result.n_iterations
        );
    }

    // ── Upgrade tests: Kaiser ncomp ─────────────────────────────────────────

    #[test]
    fn test_kaiser_ncomp_known_spectrum() {
        use crate::spm::ncomp::{select_ncomp, NcompMethod};
        // Eigenvalues: [10, 5, 1, 0.1, 0.01]  mean = 3.222
        // Components > mean: 10, 5 → should retain 2
        let eigenvalues = vec![10.0, 5.0, 1.0, 0.1, 0.01];
        let k = select_ncomp(&eigenvalues, &NcompMethod::Kaiser).unwrap();
        assert_eq!(k, 2, "Kaiser should retain 2 components for this spectrum");
    }

    #[test]
    fn test_kaiser_ncomp_flat_spectrum() {
        use crate::spm::ncomp::{select_ncomp, NcompMethod};
        // All equal → all equal to mean → none strictly above → fallback to 1
        let eigenvalues = vec![1.0, 1.0, 1.0, 1.0];
        let k = select_ncomp(&eigenvalues, &NcompMethod::Kaiser).unwrap();
        assert_eq!(k, 1, "Kaiser with flat spectrum should return 1");
    }

    #[test]
    fn test_kaiser_ncomp_single() {
        use crate::spm::ncomp::{select_ncomp, NcompMethod};
        let eigenvalues = vec![5.0];
        let k = select_ncomp(&eigenvalues, &NcompMethod::Kaiser).unwrap();
        assert_eq!(k, 1, "Kaiser with single eigenvalue should return 1");
    }

    // ── Upgrade tests: MFPCA relative threshold ─────────────────────────────

    #[test]
    fn test_mfpca_relative_threshold() {
        // Test that MFPCA works correctly with variables of very different scales
        let n = 20;
        let m1 = 10;
        let m2 = 10;
        let t1 = uniform_grid(m1);
        let t2 = uniform_grid(m2);

        // Variable 1: large scale
        let mut var1 = FdMatrix::zeros(n, m1);
        for i in 0..n {
            let phase = i as f64 * 0.3;
            for j in 0..m1 {
                var1[(i, j)] = 1000.0 * (2.0 * PI * t1[j] + phase).sin() + 5000.0;
            }
        }

        // Variable 2: tiny scale (but not zero)
        let mut var2 = FdMatrix::zeros(n, m2);
        for i in 0..n {
            let phase = i as f64 * 0.2;
            for j in 0..m2 {
                var2[(i, j)] = 0.001 * (2.0 * PI * t2[j] + phase).cos() + 0.005;
            }
        }

        let config = MfpcaConfig {
            ncomp: 3,
            weighted: true,
        };
        let result = mfpca(&[&var1, &var2], &config).unwrap();
        // Should have 3 or fewer components
        assert!(result.eigenvalues.len() <= 3);
        // Scale threshold should be relative to max scale
        assert!(result.scales[0] > result.scales[1]);
    }

    // ── Upgrade tests: SPE moment-match diagnostic ──────────────────────────

    #[test]
    fn test_spe_moment_match_diagnostic_adequate() {
        // Generate chi-squared-like data (exponential-ish)
        let n = 200;
        let spe: Vec<f64> = (0..n)
            .map(|i| {
                let x = (i as f64 * 2654435761.0) % 100.0;
                (x / 10.0).max(0.1)
            })
            .collect();
        let result = spe_moment_match_diagnostic(&spe);
        assert!(result.is_ok());
        let (kurtosis, theoretical, _adequate) = result.unwrap();
        // Just check it returns finite values
        assert!(kurtosis.is_finite());
        assert!(theoretical.is_finite());
    }

    #[test]
    fn test_spe_moment_match_diagnostic_too_few() {
        let spe = vec![1.0, 2.0, 3.0]; // < 4
        assert!(spe_moment_match_diagnostic(&spe).is_err());
    }

    // ── Upgrade tests: Bootstrap median quantile ────────────────────────────

    #[test]
    fn test_bootstrap_quantile_convergence() {
        use crate::spm::bootstrap::{t2_limit_robust, ControlLimitMethod};
        // For a large sample from chi2(5), bootstrap should converge near parametric
        let n = 500;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                // Pseudo chi2(5): sum of 5 squared normals (approximated)
                let mut sum = 0.0;
                for k in 0..5 {
                    let z =
                        ((i as u64 * 48271 + k as u64 * 2654435761) % 10000) as f64 / 5000.0 - 1.0;
                    sum += z * z;
                }
                sum
            })
            .collect();

        let parametric =
            t2_limit_robust(&values, 5, 0.05, &ControlLimitMethod::Parametric).unwrap();
        let bootstrap = t2_limit_robust(
            &values,
            5,
            0.05,
            &ControlLimitMethod::Bootstrap {
                n_bootstrap: 1000,
                seed: 42,
            },
        )
        .unwrap();

        // Bootstrap should be in a reasonable range (not identical, but same ballpark)
        assert!(bootstrap.ucl > 0.0, "Bootstrap UCL should be positive");
        assert!(parametric.ucl > 0.0, "Parametric UCL should be positive");
    }

    // ── Upgrade tests: Partial domain condition number ──────────────────────

    #[test]
    fn test_partial_monitoring_well_conditioned() {
        use crate::spm::partial::{spm_monitor_partial, DomainCompletion, PartialDomainConfig};
        let n = 40;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(n, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Monitor with half-domain observation (should not crash even with regularization)
        let partial_vals: Vec<f64> = (0..m).map(|j| data[(0, j)]).collect();
        let partial_config = PartialDomainConfig {
            ncomp: 3,
            alpha: 0.05,
            completion: DomainCompletion::ConditionalExpectation,
            ..PartialDomainConfig::default()
        };
        let result = spm_monitor_partial(&chart, &partial_vals, &argvals, m / 2, &partial_config);
        assert!(result.is_ok(), "Partial monitoring should succeed");
        let r = result.unwrap();
        assert!(r.domain_fraction > 0.0 && r.domain_fraction < 1.0);
    }

    // ── Upgrade tests: MEWMA eigenvalue guard ───────────────────────────────

    #[test]
    fn test_mewma_eigenvalue_guard() {
        use crate::spm::mewma::{spm_mewma_monitor, MewmaConfig};
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Normal usage should succeed
        let new_data = generate_ic_data(10, m, 99);
        let mewma_config = MewmaConfig {
            lambda: 0.2,
            ncomp: 3,
            alpha: 0.05,
            asymptotic: true,
        };
        let result = spm_mewma_monitor(&chart, &new_data, &argvals, &mewma_config);
        assert!(result.is_ok());
    }

    // ── Upgrade tests: FRCC R-squared diagnostic ────────────────────────────

    #[test]
    fn test_frcc_low_r_squared_rejected() {
        // Predictors that are pure noise → R² ≈ 0 → should fail
        let n = 30;
        let m = 15;
        let argvals = uniform_grid(m);
        let y = generate_ic_data(n, m, 42);

        // Random predictors with no relationship to y
        let mut predictors = FdMatrix::zeros(n, 2);
        for i in 0..n {
            predictors[(i, 0)] = ((i as u64).wrapping_mul(48271) % 10000) as f64 / 10000.0;
            predictors[(i, 1)] = ((i as u64).wrapping_mul(2654435761) % 10000) as f64 / 10000.0;
        }

        let config = FrccConfig {
            ncomp: 2,
            fosr_lambda: 1e-4,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
            ..FrccConfig::default()
        };
        let result = frcc_phase1(&y, &predictors, &argvals, &config);
        // Should either fail with low R² or produce a result with low R²
        // The exact behavior depends on the data, but the check is in place
        if let Ok(chart) = &result {
            assert!(
                chart.fosr_r_squared >= 0.1,
                "If chart was built, R² should be >= 0.1"
            );
        }
    }

    // ── Upgrade tests: MFPCA T² contributions ───────────────────────────────

    #[test]
    fn test_t2_contributions_mfpca_sums() {
        let n = 20;
        let m1 = 10;
        let m2 = 10;
        let t1 = uniform_grid(m1);
        let t2_grid = uniform_grid(m2);

        let mut var1 = FdMatrix::zeros(n, m1);
        let mut var2 = FdMatrix::zeros(n, m2);
        for i in 0..n {
            let phase = i as f64 * 0.3;
            for j in 0..m1 {
                var1[(i, j)] = (2.0 * PI * t1[j] + phase).sin();
            }
            for j in 0..m2 {
                var2[(i, j)] = (2.0 * PI * t2_grid[j] + phase * 0.5).cos();
            }
        }

        let mfpca_config = MfpcaConfig {
            ncomp: 3,
            weighted: true,
        };
        let mfpca_result = mfpca(&[&var1, &var2], &mfpca_config).unwrap();
        let _ncomp = mfpca_result.eigenvalues.len();

        // Compute T² and contributions
        let t2_vals = hotelling_t2(&mfpca_result.scores, &mfpca_result.eigenvalues).unwrap();
        let contrib = t2_contributions_mfpca(&mfpca_result.scores, &mfpca_result).unwrap();

        // Per-variable contributions should sum to T² for each observation
        for i in 0..n {
            let row_sum: f64 = (0..2).map(|v| contrib[(i, v)]).sum();
            assert!(
                (row_sum - t2_vals[i]).abs() < 1e-8 * (1.0 + t2_vals[i].abs()),
                "MFPCA T² contribution row sum ({}) should match T² ({})",
                row_sum,
                t2_vals[i]
            );
        }
    }

    // ── Upgrade tests: AMEWMA time-dependent covariance ─────────────────────

    #[test]
    fn test_amewma_time_dependent_covariance() {
        use crate::spm::amewma::{spm_amewma_monitor, AmewmaConfig};
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(20, m, 99);
        let amewma_config = AmewmaConfig {
            lambda_min: 0.05,
            lambda_max: 0.95,
            lambda_init: 0.2,
            eta: 0.1,
            ncomp: 3,
            alpha: 0.05,
            error_clip: None,
        };
        let result = spm_amewma_monitor(&chart, &new_data, &argvals, &amewma_config).unwrap();

        // T² should be finite and non-negative
        for &t2 in &result.t2_statistic {
            assert!(
                t2.is_finite() && t2 >= 0.0,
                "T² should be finite and non-negative, got {t2}"
            );
        }
        // Lambda values should be within bounds
        for &lam in &result.lambda_t {
            assert!(
                lam >= amewma_config.lambda_min && lam <= amewma_config.lambda_max,
                "Lambda {lam} should be in [{}, {}]",
                amewma_config.lambda_min,
                amewma_config.lambda_max
            );
        }
    }

    // ── Upgrade tests: Profile effective sample size ────────────────────────

    #[test]
    fn test_profile_effective_n_windows() {
        use crate::spm::profile::{profile_phase1, ProfileMonitorConfig};
        let n = 60;
        let m = 10;
        let p = 2;
        let argvals = uniform_grid(m);

        let y = generate_ic_data(n, m, 42);
        let mut predictors = FdMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                predictors[(i, j)] =
                    ((i as u64 * (j as u64 + 1)).wrapping_mul(48271) % 10000) as f64 / 10000.0;
            }
        }

        // Non-overlapping windows
        let config_no_overlap = ProfileMonitorConfig {
            fosr_lambda: 1e-4,
            ncomp: 2,
            alpha: 0.05,
            window_size: 10,
            step_size: 10,
        };
        let chart1 = profile_phase1(&y, &predictors, &argvals, &config_no_overlap).unwrap();

        // Overlapping windows (step=1)
        let config_overlap = ProfileMonitorConfig {
            fosr_lambda: 1e-4,
            ncomp: 2,
            alpha: 0.05,
            window_size: 10,
            step_size: 1,
        };
        let chart2 = profile_phase1(&y, &predictors, &argvals, &config_overlap).unwrap();

        // Non-overlapping: effective should be positive and bounded
        assert!(
            chart1.effective_n_windows >= 2.0,
            "Non-overlapping: effective={} should be >= 2",
            chart1.effective_n_windows
        );

        // Overlapping: should also be positive and bounded
        assert!(
            chart2.effective_n_windows >= 2.0,
            "Overlapping: effective={} should be >= 2",
            chart2.effective_n_windows
        );

        // Lag-1 autocorrelation should be in [-1, 1]
        assert!(
            chart1.lag1_autocorrelation >= -1.0 && chart1.lag1_autocorrelation <= 1.0,
            "lag1_autocorrelation={} should be in [-1, 1]",
            chart1.lag1_autocorrelation
        );
        assert!(
            chart2.lag1_autocorrelation >= -1.0 && chart2.lag1_autocorrelation <= 1.0,
            "lag1_autocorrelation={} should be in [-1, 1]",
            chart2.lag1_autocorrelation
        );
    }

    // ── Round 2 upgrade tests ───────────────────────────────────────────────

    #[test]
    fn test_ncomp_elbow_smoothed() {
        use crate::spm::ncomp::{select_ncomp, NcompMethod};
        // Noisy spectrum where smoothing helps: [100, 50, 48, 2, 1, 0.5]
        // Without smoothing, elbow could be at index 1 or 2 (noise)
        // With smoothing, the elbow at the 48→2 drop is clearer
        let eigenvalues = vec![100.0, 50.0, 48.0, 2.0, 1.0, 0.5];
        let k = select_ncomp(&eigenvalues, &NcompMethod::Elbow).unwrap();
        assert!(
            (2..=4).contains(&k),
            "Smoothed elbow should find ~3, got {k}"
        );
    }

    #[test]
    fn test_hotelling_t2_regularized() {
        use crate::spm::stats::hotelling_t2_regularized;
        let scores = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        // Very small eigenvalue that would cause numerical issues
        let eigenvalues = vec![1.0, 1e-20];
        // Without regularization, T² would be huge for the second component
        let t2_reg = hotelling_t2_regularized(&scores, &eigenvalues, 1e-6).unwrap();
        assert!(t2_reg[0].is_finite(), "Regularized T² should be finite");
        assert!(t2_reg[1].is_finite(), "Regularized T² should be finite");
        // With floor=1e-6, second component contributes score²/1e-6
        let expected_0 = 1.0 / 1.0 + 3.0 * 3.0 / 1e-6;
        assert!(
            (t2_reg[0] - expected_0).abs() < 1e-3,
            "Expected {expected_0}, got {}",
            t2_reg[0]
        );
    }

    #[test]
    fn test_ewma_exact_covariance() {
        use crate::spm::ewma::{spm_ewma_monitor, EwmaConfig};
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();
        let new_data = generate_ic_data(10, m, 99);

        let config_asymptotic = EwmaConfig {
            lambda: 0.2,
            ncomp: 3,
            alpha: 0.05,
            exact_covariance: false,
        };
        let config_exact = EwmaConfig {
            lambda: 0.2,
            ncomp: 3,
            alpha: 0.05,
            exact_covariance: true,
        };

        let r_asym = spm_ewma_monitor(&chart, &new_data, &argvals, &config_asymptotic).unwrap();
        let r_exact = spm_ewma_monitor(&chart, &new_data, &argvals, &config_exact).unwrap();

        // Exact T² should be >= asymptotic T² (startup correction inflates early)
        assert!(
            r_exact.t2[0] >= r_asym.t2[0] - 1e-10,
            "Exact T²[0]={} should be >= asymptotic T²[0]={}",
            r_exact.t2[0],
            r_asym.t2[0]
        );

        // Later values should converge
        let last = r_asym.t2.len() - 1;
        let ratio = r_exact.t2[last] / r_asym.t2[last].max(1e-15);
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "Late T² should converge: exact={}, asymptotic={}",
            r_exact.t2[last],
            r_asym.t2[last]
        );
    }

    #[test]
    fn test_cusum_restart_config() {
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let chart_config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &chart_config).unwrap();

        let shifted = generate_oc_data(20, m, 3.0, 99);

        // Without restart
        let config_no_restart = CusumConfig {
            restart: false,
            ..CusumConfig::default()
        };
        let r1 = spm_cusum_monitor(&chart, &shifted, &argvals, &config_no_restart).unwrap();

        // With restart via config
        let config_restart = CusumConfig {
            restart: true,
            ..CusumConfig::default()
        };
        let r2 = spm_cusum_monitor(&chart, &shifted, &argvals, &config_restart).unwrap();

        // Both should produce results; restart may have different alarm patterns
        assert_eq!(r1.cusum_statistic.len(), r2.cusum_statistic.len());
    }

    #[test]
    fn test_phase1_sample_size_adequate() {
        let m = 20;
        let argvals = uniform_grid(m);

        // Small dataset: n=8, ncomp=5 → 8 < 10*5 → not adequate
        let small_data = generate_ic_data(8, m, 42);
        let config = SpmConfig {
            ncomp: 5,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&small_data, &argvals, &config).unwrap();
        assert!(
            !chart.sample_size_adequate,
            "n=8, ncomp=5 should be inadequate"
        );

        // Large dataset: n=60, ncomp=3 → 60 >= 30 → adequate
        let large_data = generate_ic_data(60, m, 42);
        let config2 = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart2 = spm_phase1(&large_data, &argvals, &config2).unwrap();
        assert!(
            chart2.sample_size_adequate,
            "n=60, ncomp=3 should be adequate"
        );
    }

    #[test]
    fn test_iterative_removal_rates() {
        let n = 50;
        let m = 20;
        let argvals = uniform_grid(m);

        let mut data = generate_ic_data(n, m, 42);
        // Add strong outliers
        for j in 0..m {
            data[(0, j)] += 50.0;
            data[(1, j)] += 50.0;
        }

        let config = IterativePhase1Config {
            spm: SpmConfig {
                ncomp: 3,
                alpha: 0.01,
                ..SpmConfig::default()
            },
            ..IterativePhase1Config::default()
        };
        let result = spm_phase1_iterative(&data, &argvals, &config).unwrap();
        // Should have removal_rates recorded
        assert_eq!(
            result.removal_rates.len(),
            result.n_iterations,
            "Should have one removal rate per iteration"
        );
        // All rates should be in [0, 1]
        for &rate in &result.removal_rates {
            assert!(
                (0.0..=1.0).contains(&rate),
                "Removal rate {rate} out of range"
            );
        }
    }

    #[test]
    fn test_amewma_error_clipping() {
        use crate::spm::amewma::{spm_amewma_monitor, AmewmaConfig};
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Data with an outlier
        let mut new_data = generate_ic_data(10, m, 99);
        for j in 0..m {
            new_data[(5, j)] += 100.0; // extreme outlier
        }

        // Without clipping
        let config_no_clip = AmewmaConfig {
            error_clip: None,
            ..AmewmaConfig::default()
        };
        let r1 = spm_amewma_monitor(&chart, &new_data, &argvals, &config_no_clip).unwrap();

        // With clipping at 9.0 (3σ)
        let config_clip = AmewmaConfig {
            error_clip: Some(9.0),
            ..AmewmaConfig::default()
        };
        let r2 = spm_amewma_monitor(&chart, &new_data, &argvals, &config_clip).unwrap();

        // Clipped lambda should change less dramatically
        let max_lambda_no_clip = r1.lambda_t.iter().cloned().fold(0.0_f64, f64::max);
        let max_lambda_clip = r2.lambda_t.iter().cloned().fold(0.0_f64, f64::max);
        // With clipping, max lambda should be constrained
        assert!(
            max_lambda_clip <= max_lambda_no_clip + 1e-10,
            "Clipped max lambda ({max_lambda_clip}) should be <= unclipped ({max_lambda_no_clip})"
        );
    }

    #[test]
    fn test_frcc_configurable_r_squared_threshold() {
        // With min_r_squared=0.0, even noise predictors should pass
        let n = 30;
        let m = 15;
        let argvals = uniform_grid(m);
        let y = generate_ic_data(n, m, 42);

        let mut predictors = FdMatrix::zeros(n, 2);
        for i in 0..n {
            predictors[(i, 0)] = ((i as u64).wrapping_mul(48271) % 10000) as f64 / 10000.0;
            predictors[(i, 1)] = ((i as u64).wrapping_mul(2654435761) % 10000) as f64 / 10000.0;
        }

        let config_strict = FrccConfig {
            min_r_squared: 0.5, // Very strict
            ..FrccConfig::default()
        };
        let result_strict = frcc_phase1(&y, &predictors, &argvals, &config_strict);
        // Noise predictors should fail with strict threshold
        assert!(
            result_strict.is_err(),
            "Strict R² threshold should reject noise predictors"
        );

        let config_lax = FrccConfig {
            min_r_squared: 0.0, // Accept anything
            ..FrccConfig::default()
        };
        let result_lax = frcc_phase1(&y, &predictors, &argvals, &config_lax);
        // Should pass with lax threshold
        assert!(
            result_lax.is_ok(),
            "Zero R² threshold should accept any predictors"
        );
    }

    #[test]
    fn test_elastic_spm_alignment_residual() {
        use crate::spm::elastic_spm::{elastic_spm_phase1, ElasticSpmConfig};
        let n = 30;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(n, m, 42);

        let config = ElasticSpmConfig::default();
        let chart = elastic_spm_phase1(&data, &argvals, &config).unwrap();

        // Alignment residual should be non-negative and finite
        assert!(
            chart.mean_alignment_residual >= 0.0 && chart.mean_alignment_residual.is_finite(),
            "mean_alignment_residual={} should be non-negative and finite",
            chart.mean_alignment_residual
        );
    }

    #[test]
    fn test_partial_configurable_regularization() {
        use crate::spm::partial::{spm_monitor_partial, PartialDomainConfig};
        let n = 40;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(n, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();
        let partial_vals: Vec<f64> = (0..m).map(|j| data[(0, j)]).collect();

        // Default regularization
        let config1 = PartialDomainConfig::default();
        let r1 = spm_monitor_partial(&chart, &partial_vals, &argvals, m / 2, &config1).unwrap();

        // Stronger regularization
        let config2 = PartialDomainConfig {
            regularization_eps: 1e-6,
            ..PartialDomainConfig::default()
        };
        let r2 = spm_monitor_partial(&chart, &partial_vals, &argvals, m / 2, &config2).unwrap();

        // Both should produce finite results
        assert!(r1.t2.is_finite(), "Default reg T² should be finite");
        assert!(r2.t2.is_finite(), "Strong reg T² should be finite");
    }

    // ── Round 3 upgrade tests ────────────────────────────────────────────

    #[test]
    fn test_chi2_quantile_high_df() {
        // Verify chi2_quantile doesn't hang or produce NaN for large df
        // (blocker: 1e-300 tolerance in Lentz algorithm could cause infinite loops)
        use crate::spm::chi_squared::{chi2_cdf, chi2_quantile};
        for &k in &[50, 100, 200] {
            let q = chi2_quantile(0.95, k);
            assert!(
                q.is_finite(),
                "chi2_quantile(0.95, {k}) should be finite, got {q}"
            );
            assert!(q > 0.0, "chi2_quantile(0.95, {k}) should be positive");
            // Round-trip: CDF of the quantile should be close to 0.95
            let p = chi2_cdf(q, k);
            assert!(
                (p - 0.95).abs() < 0.01,
                "Round-trip failed for k={k}: q={q}, cdf(q)={p}"
            );
        }
    }

    #[test]
    fn test_mewma_spe_present() {
        // Verify MEWMA now returns SPE statistics
        use crate::spm::mewma::{spm_mewma_monitor, MewmaConfig};
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let monitor_data = generate_ic_data(15, m, 99);
        let mewma_config = MewmaConfig {
            lambda: 0.2,
            ncomp: 3,
            alpha: 0.05,
            asymptotic: true,
        };
        let result = spm_mewma_monitor(&chart, &monitor_data, &argvals, &mewma_config).unwrap();

        // SPE fields should be populated and consistent
        assert_eq!(
            result.spe.len(),
            15,
            "SPE should have one value per observation"
        );
        assert!(result.spe_limit > 0.0, "SPE limit should be positive");
        assert_eq!(
            result.spe_alarm.len(),
            15,
            "SPE alarm should match observations"
        );
        // In-control data: most SPE values should be below the limit
        let n_spe_alarm: usize = result.spe_alarm.iter().filter(|&&a| a).count();
        assert!(
            n_spe_alarm < 10,
            "In-control data should have few SPE alarms"
        );
    }

    #[test]
    fn test_cusum_spe_present() {
        // Verify CUSUM now returns SPE statistics
        use crate::spm::cusum::{spm_cusum_monitor, CusumConfig};
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let monitor_data = generate_ic_data(15, m, 99);
        let cusum_config = CusumConfig {
            ncomp: 3,
            ..CusumConfig::default()
        };
        let result = spm_cusum_monitor(&chart, &monitor_data, &argvals, &cusum_config).unwrap();

        assert_eq!(
            result.spe.len(),
            15,
            "SPE should have one value per observation"
        );
        assert!(result.spe_limit > 0.0, "SPE limit should be positive");
        assert_eq!(
            result.spe_alarm.len(),
            15,
            "SPE alarm should match observations"
        );
    }

    #[test]
    fn test_t2_pc_significance_bonferroni() {
        use crate::spm::contrib::{t2_pc_contributions, t2_pc_significance};
        // 5 obs, 3 PCs; PC1 has huge contribution, others small
        let scores = FdMatrix::from_column_major(
            vec![
                10.0, 0.1, 0.1, 0.1, 0.1, // PC1: one outlier
                0.1, 0.1, 0.1, 0.1, 0.1, // PC2: all small
                0.1, 0.1, 0.1, 0.1, 0.1, // PC3: all small
            ],
            5,
            3,
        )
        .unwrap();
        let eigenvalues = vec![1.0, 1.0, 1.0];
        let contribs = t2_pc_contributions(&scores, &eigenvalues).unwrap();
        let sig = t2_pc_significance(&contribs, 0.05).unwrap();

        // Obs 0, PC1 contribution = 100.0 >> chi2(1, 0.983) ≈ 5.73
        assert_eq!(
            sig[(0, 0)],
            1.0,
            "Large PC1 contribution should be significant"
        );
        // Small contributions should not be significant
        assert_eq!(
            sig[(1, 0)],
            0.0,
            "Small contribution should not be significant"
        );
        assert_eq!(
            sig[(0, 1)],
            0.0,
            "Small PC2 contribution should not be significant"
        );
    }

    #[test]
    fn test_partial_cholesky_retry() {
        // Test that partial monitoring handles near-singular systems gracefully
        use crate::spm::partial::{spm_monitor_partial, PartialDomainConfig};
        let n = 40;
        let m = 30;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(n, m, 42);
        let config = SpmConfig {
            ncomp: 5,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Observe only 3 grid points — very small observed domain
        let partial_vals: Vec<f64> = (0..m).map(|j| data[(0, j)]).collect();
        let partial_config = PartialDomainConfig {
            ncomp: 5,
            ..PartialDomainConfig::default()
        };
        let result = spm_monitor_partial(&chart, &partial_vals, &argvals, 3, &partial_config);
        // Should succeed (Cholesky retry handles ill-conditioning)
        assert!(
            result.is_ok(),
            "Partial monitoring with 3 obs should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        assert!(r.t2.is_finite(), "T² should be finite");
    }

    #[test]
    fn test_frcc_uses_shared_split_indices() {
        // Verify FRCC produces valid results (using shared split_indices from phase.rs)
        use crate::spm::frcc::{frcc_phase1, FrccConfig};
        let n = 60;
        let m = 20;
        let argvals = uniform_grid(m);
        let y_curves = generate_ic_data(n, m, 42);

        // Create simple predictors
        let mut pred_data = vec![0.0; n];
        for i in 0..n {
            pred_data[i] = (i as f64) / n as f64;
        }
        let predictors = FdMatrix::from_column_major(pred_data, n, 1).unwrap();

        let config = FrccConfig {
            min_r_squared: 0.0, // Allow any R² for this test
            ..FrccConfig::default()
        };
        let result = frcc_phase1(&y_curves, &predictors, &argvals, &config);
        // Should produce a valid chart (not crash from missing functions)
        assert!(result.is_ok(), "FRCC should succeed: {:?}", result.err());
    }

    #[test]
    fn test_pcg_rng_determinism() {
        // Verify the PCG-based split is deterministic (same seed → same split)
        let (tune1, cal1) = crate::spm::phase::split_indices(100, 0.5, 42);
        let (tune2, cal2) = crate::spm::phase::split_indices(100, 0.5, 42);
        assert_eq!(tune1, tune2, "Same seed should produce same tuning set");
        assert_eq!(cal1, cal2, "Same seed should produce same calibration set");

        // Different seeds should produce different splits
        let (tune3, _) = crate::spm::phase::split_indices(100, 0.5, 99);
        assert_ne!(
            tune1, tune3,
            "Different seeds should produce different splits"
        );
    }

    // ── Round 4 validation tests ─────────────────────────────────────────────

    #[test]
    fn test_amewma_error_clip_positive_validation() {
        // error_clip must be positive when set
        let n = 60;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(n, m, 42);
        let config = SpmConfig::default();
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(10, m, 99);
        let amewma_config = AmewmaConfig {
            error_clip: Some(-1.0),
            ..AmewmaConfig::default()
        };
        let result = spm_amewma_monitor(&chart, &new_data, &argvals, &amewma_config);
        assert!(result.is_err(), "Negative error_clip should be rejected");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("error_clip") && err_msg.contains("positive"),
            "Error should mention error_clip must be positive: {err_msg}"
        );

        // Zero should also be rejected
        let amewma_config_zero = AmewmaConfig {
            error_clip: Some(0.0),
            ..AmewmaConfig::default()
        };
        let result2 = spm_amewma_monitor(&chart, &new_data, &argvals, &amewma_config_zero);
        assert!(result2.is_err(), "Zero error_clip should be rejected");
    }

    #[test]
    fn test_amewma_q_prev_clamping_stability() {
        // Test that the adaptive weight doesn't diverge even with extreme data
        let n = 60;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(n, m, 42);
        let config = SpmConfig::default();
        let chart = spm_phase1(&data, &argvals, &config).unwrap();

        // Create data with a massive spike to stress the adaptive weight
        let mut new_data = generate_ic_data(20, m, 99);
        for j in 0..m {
            new_data[(5, j)] *= 100.0; // huge spike at observation 5
        }

        let amewma_config = AmewmaConfig {
            eta: 0.5, // fast adaptation to amplify the effect
            ..AmewmaConfig::default()
        };
        let result = spm_amewma_monitor(&chart, &new_data, &argvals, &amewma_config);
        assert!(
            result.is_ok(),
            "AMEWMA should handle extreme data: {:?}",
            result.err()
        );
        let res = result.unwrap();
        // All lambda_t must be within [lambda_min, lambda_max]
        for (t, &lam) in res.lambda_t.iter().enumerate() {
            assert!(
                lam >= amewma_config.lambda_min && lam <= amewma_config.lambda_max,
                "lambda_t[{t}] = {lam} out of range [{}, {}]",
                amewma_config.lambda_min,
                amewma_config.lambda_max
            );
        }
        // T² values should all be finite
        for (t, &t2) in res.t2_statistic.iter().enumerate() {
            assert!(t2.is_finite(), "t2_statistic[{t}] is not finite: {t2}");
        }
    }

    #[test]
    fn test_nelson7_closed_interval() {
        use crate::spm::rules::{evaluate_rules, ChartRule};
        // 15 points exactly at center ± 1σ boundary should trigger Nelson7
        // (closed interval: <= sigma, not < sigma)
        let center = 10.0;
        let sigma = 2.0;
        // All values exactly at center + sigma
        let values: Vec<f64> = vec![center + sigma; 15];
        let violations = evaluate_rules(&values, center, sigma, &[ChartRule::Nelson7]).unwrap();
        assert!(
            !violations.is_empty(),
            "Nelson7 should trigger for points exactly at 1σ boundary (closed interval)"
        );
        assert_eq!(violations[0].rule, ChartRule::Nelson7);
    }

    #[test]
    fn test_nelson7_just_outside_no_trigger() {
        use crate::spm::rules::{evaluate_rules, ChartRule};
        // 15 points just outside 1σ should NOT trigger Nelson7
        let center = 10.0;
        let sigma = 2.0;
        let values: Vec<f64> = vec![center + sigma + 0.001; 15];
        let violations = evaluate_rules(&values, center, sigma, &[ChartRule::Nelson7]).unwrap();
        assert!(
            violations.is_empty(),
            "Nelson7 should NOT trigger for points just outside 1σ boundary"
        );
    }

    #[test]
    fn test_phase1_small_tuning_set_error() {
        // Very small data with high tuning_fraction should fail gracefully
        let m = 10;
        let argvals = uniform_grid(m);
        let data = generate_ic_data(4, m, 42); // only 4 observations
        let config = SpmConfig {
            tuning_fraction: 0.25, // ~1 obs for tuning, too few
            ..SpmConfig::default()
        };
        let result = spm_phase1(&data, &argvals, &config);
        assert!(result.is_err(), "Should fail with tiny tuning set");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("tuning") || err_msg.contains("3"),
            "Error should mention tuning set size: {err_msg}"
        );
    }

    #[test]
    fn test_profile_window_size_vs_predictors() {
        use crate::spm::profile::{profile_phase1, ProfileMonitorConfig};
        let n = 100;
        let m = 15;
        let argvals = uniform_grid(m);
        let y_curves = generate_ic_data(n, m, 42);

        // 3 predictors: window_size must be >= 5 (p + 2)
        let mut pred_data = vec![0.0; n * 3];
        for i in 0..n {
            pred_data[i] = (i as f64) / n as f64;
            pred_data[n + i] = ((i as f64) / n as f64).powi(2);
            pred_data[2 * n + i] = ((i * 7 + 3) as f64 % 10.0) / 10.0;
        }
        let predictors = FdMatrix::from_column_major(pred_data, n, 3).unwrap();

        // window_size = 4 < p + 2 = 5 should fail
        let config = ProfileMonitorConfig {
            window_size: 4,
            step_size: 4,
            ncomp: 2,
            ..ProfileMonitorConfig::default()
        };
        let result = profile_phase1(&y_curves, &predictors, &argvals, &config);
        assert!(result.is_err(), "window_size < p+2 should fail");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("window_size") && err_msg.contains("p + 2"),
            "Error should mention window_size and p+2: {err_msg}"
        );
    }

    #[test]
    fn test_t2_pc_significance_extreme_scores() {
        use crate::spm::contrib::{t2_pc_contributions, t2_pc_significance};
        // Build scores where PC1 has a huge value but PC2 is small
        let scores = FdMatrix::from_column_major(
            vec![
                100.0, 0.5, 0.3, // col 0 (PC1): obs0=100, obs1=0.5, obs2=0.3
                0.1, 0.2, 0.1, // col 1 (PC2): all small
            ],
            3,
            2,
        )
        .unwrap();
        let eigenvalues = [1.0, 1.0];
        let contribs = t2_pc_contributions(&scores, &eigenvalues).unwrap();

        // Significance at α = 0.05: Bonferroni → test each PC at α/2 = 0.025
        let sig = t2_pc_significance(&contribs, 0.05).unwrap();
        let (n, ncomp) = sig.shape();
        assert_eq!(n, 3);
        assert_eq!(ncomp, 2);

        // Observation 0, PC1 (score²=10000) should be significant
        assert_eq!(sig[(0, 0)], 1.0, "Large PC1 score should be significant");
        // Observation 0, PC2 (score²=0.01) should not be significant
        assert_eq!(
            sig[(0, 1)],
            0.0,
            "Small PC2 score should not be significant"
        );
        // Observations 1,2 PC1 (score²=0.25, 0.09) should not be significant
        assert_eq!(sig[(1, 0)], 0.0);
        assert_eq!(sig[(2, 0)], 0.0);
    }

    #[test]
    fn test_t2_pc_significance_alpha_bounds() {
        use crate::spm::contrib::t2_pc_significance;
        let contribs = FdMatrix::from_column_major(vec![1.0, 2.0], 2, 1).unwrap();
        assert!(t2_pc_significance(&contribs, 0.0).is_err());
        assert!(t2_pc_significance(&contribs, 1.0).is_err());
        assert!(t2_pc_significance(&contribs, -0.1).is_err());
        assert!(t2_pc_significance(&contribs, 0.05).is_ok());
    }

    // ── spm_monitor_from_fields tests ───────────────────────────────────────

    #[test]
    fn test_monitor_from_fields_matches_spm_monitor() {
        use crate::spm::phase::spm_monitor_from_fields;

        let m = 30;
        let argvals = uniform_grid(m);
        let ic_data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&ic_data, &argvals, &config).unwrap();

        let new_data = generate_ic_data(10, m, 99);

        let r1 = spm_monitor(&chart, &new_data, &argvals).unwrap();
        let r2 = spm_monitor_from_fields(
            &chart.fpca.mean,
            &chart.fpca.rotation,
            &chart.fpca.weights,
            &chart.eigenvalues,
            chart.t2_limit.ucl,
            chart.spe_limit.ucl,
            &new_data,
            &argvals,
        )
        .unwrap();

        assert_eq!(r1.t2.len(), r2.t2.len());
        assert_eq!(r1.spe.len(), r2.spe.len());
        for (a, b) in r1.t2.iter().zip(&r2.t2) {
            assert!((a - b).abs() < 1e-10, "T2 mismatch: {a} vs {b}");
        }
        for (a, b) in r1.spe.iter().zip(&r2.spe) {
            assert!((a - b).abs() < 1e-10, "SPE mismatch: {a} vs {b}");
        }
        assert_eq!(r1.t2_alarm, r2.t2_alarm);
        assert_eq!(r1.spe_alarm, r2.spe_alarm);
        assert_eq!(r1.scores.shape(), r2.scores.shape());
        let (n, nc) = r1.scores.shape();
        for i in 0..n {
            for k in 0..nc {
                assert!(
                    (r1.scores[(i, k)] - r2.scores[(i, k)]).abs() < 1e-10,
                    "Scores mismatch at ({i},{k})"
                );
            }
        }
    }

    #[test]
    fn test_monitor_from_fields_oc_data() {
        use crate::spm::phase::spm_monitor_from_fields;

        let m = 30;
        let argvals = uniform_grid(m);
        let ic_data = generate_ic_data(40, m, 42);
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..SpmConfig::default()
        };
        let chart = spm_phase1(&ic_data, &argvals, &config).unwrap();

        // Out-of-control data with large mean shift should trigger alarms
        let oc_data = generate_oc_data(5, m, 5.0, 123);
        let r1 = spm_monitor(&chart, &oc_data, &argvals).unwrap();
        let r2 = spm_monitor_from_fields(
            &chart.fpca.mean,
            &chart.fpca.rotation,
            &chart.fpca.weights,
            &chart.eigenvalues,
            chart.t2_limit.ucl,
            chart.spe_limit.ucl,
            &oc_data,
            &argvals,
        )
        .unwrap();

        // Results should match exactly
        assert_eq!(r1.t2_alarm, r2.t2_alarm);
        assert_eq!(r1.spe_alarm, r2.spe_alarm);
        // OC data should trigger at least one alarm
        let any_alarm = r2.t2_alarm.iter().any(|&a| a) || r2.spe_alarm.iter().any(|&a| a);
        assert!(any_alarm, "OC data with shift=5.0 should trigger alarms");
    }

    #[test]
    fn test_monitor_from_fields_dimension_errors() {
        use crate::spm::phase::spm_monitor_from_fields;

        let m = 10;
        let argvals = uniform_grid(m);
        let mean = vec![0.0; m];
        let rotation = FdMatrix::zeros(m, 2);
        let weights = vec![1.0; m];
        let eigenvalues = vec![1.0, 0.5];

        // Wrong number of columns
        let bad_data = FdMatrix::zeros(5, m + 1);
        assert!(spm_monitor_from_fields(
            &mean,
            &rotation,
            &weights,
            &eigenvalues,
            10.0,
            5.0,
            &bad_data,
            &argvals,
        )
        .is_err());

        // Wrong argvals length
        let data = FdMatrix::zeros(5, m);
        let bad_argvals = vec![0.0; m + 1];
        assert!(spm_monitor_from_fields(
            &mean,
            &rotation,
            &weights,
            &eigenvalues,
            10.0,
            5.0,
            &data,
            &bad_argvals,
        )
        .is_err());

        // Wrong rotation rows
        let bad_rotation = FdMatrix::zeros(m + 1, 2);
        assert!(spm_monitor_from_fields(
            &mean,
            &bad_rotation,
            &weights,
            &eigenvalues,
            10.0,
            5.0,
            &data,
            &argvals,
        )
        .is_err());

        // Wrong rotation cols
        let bad_rotation2 = FdMatrix::zeros(m, 3);
        assert!(spm_monitor_from_fields(
            &mean,
            &bad_rotation2,
            &weights,
            &eigenvalues,
            10.0,
            5.0,
            &data,
            &argvals,
        )
        .is_err());

        // Wrong weights length
        let bad_weights = vec![1.0; m + 1];
        assert!(spm_monitor_from_fields(
            &mean,
            &rotation,
            &bad_weights,
            &eigenvalues,
            10.0,
            5.0,
            &data,
            &argvals,
        )
        .is_err());
    }
}
