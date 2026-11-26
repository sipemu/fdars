//! Tests of the two-step GFPCA, `npc_criterion`, periodic templates and
//! piecewise-linear warps (ALN-EXPFAM, GH #60, part 2).
//!
//! Golden values: R registr 2.2.1 (gfpca_twoStep with mgcv 1.9-4, gamm4 0.3.0,
//! lme4 2.0.1, Matrix 1.7-5; pbs 1.1) via
//! `.planning/quick/261126-8kh-aln-expfam-60-exponential-family-registr/golden_twostep.R`
//! (see `golden_twostep.rs`), on the portable LCG data of [`sim`].

// Configs are built as external users must for `#[non_exhaustive]` structs.
#![allow(clippy::field_reassign_with_default)]

use super::basis::{PeriodicBs, TemplateBasis};
use super::golden_twostep::*;
use super::pl2::{hinv, Pl2Problem};
use super::tests::{
    abs_cos, data_sum, max_abs_diff, registered_with, sim, split, subspace_min_cos,
};
use super::twostep::{coarsen_index, latent_covariance};
use super::warp::{ObsLoss, TemplateScale};
use super::*;

fn rel_max(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    max_abs_diff(a, b) / scale
}

fn col(m: &FdMatrix, k: usize) -> Vec<f64> {
    (0..m.nrows()).map(|r| m[(r, k)]).collect()
}

fn two_step_cfg(npc: usize) -> GfpcaConfig {
    GfpcaConfig {
        npc,
        ..GfpcaConfig::default()
    }
}

// --------------------------------------------------------- coarsening ----

#[test]
fn coarsen_index_matches_registr() {
    // registr:::coarsen_index(x, 4L)
    let cases: [(&[f64], &[f64]); 3] = [
        (
            &[0.123456, 0.5, 0.99999, 0.0004321],
            &[0.1235, 0.5, 1.0, 0.0004],
        ),
        (&[12.34567, 3.15259, -2.71928], &[12.35, 3.15, -2.72]),
        (&[0.0, 1.0 / 3.0, 1.0], &[0.0, 0.333, 1.0]),
    ];
    for (x, want) in cases {
        let got = coarsen_index(x, 4);
        assert!(max_abs_diff(&got, want) < 1e-12, "{got:?} vs {want:?}");
    }
}

// ------------------------------------------------------ two-step GFPCA ----

struct TwoStepGolden {
    seed: u64,
    n: usize,
    code: u8,
    family: ExpFamily,
    sum: f64,
    grid: &'static [f64],
    cov_sub: &'static [f64],
    alpha: &'static [f64],
    efun: &'static [f64],
    evals: &'static [f64],
    eval_sum: f64,
    scores: &'static [f64],
    theta: &'static [f64],
}

fn two_step_goldens() -> [TwoStepGolden; 4] {
    [
        TwoStepGolden {
            seed: 101,
            n: 15,
            code: 0,
            family: ExpFamily::Gaussian,
            sum: TG_DATA_SUM,
            grid: &TG_GRID,
            cov_sub: &TG_COV_SUB,
            alpha: &TG_ALPHA,
            efun: &TG_EFUN,
            evals: &TG_EVAL,
            eval_sum: TG_EVAL_SUM,
            scores: &TG_SCORES,
            theta: &TG_THETA,
        },
        TwoStepGolden {
            seed: 202,
            n: 40,
            code: 1,
            family: ExpFamily::Binomial,
            sum: TB_DATA_SUM,
            grid: &TB_GRID,
            cov_sub: &TB_COV_SUB,
            alpha: &TB_ALPHA,
            efun: &TB_EFUN,
            evals: &TB_EVAL,
            eval_sum: TB_EVAL_SUM,
            scores: &TB_SCORES,
            theta: &TB_THETA,
        },
        TwoStepGolden {
            seed: 303,
            n: 20,
            code: 2,
            family: ExpFamily::Poisson,
            sum: TP_DATA_SUM,
            grid: &TP_GRID,
            cov_sub: &TP_COV_SUB,
            alpha: &TP_ALPHA,
            efun: &TP_EFUN,
            evals: &TP_EVAL,
            eval_sum: TP_EVAL_SUM,
            scores: &TP_SCORES,
            theta: &TP_THETA,
        },
        TwoStepGolden {
            seed: 404,
            n: 20,
            code: 3,
            family: ExpFamily::Gamma,
            sum: TGA_DATA_SUM,
            grid: &TGA_GRID,
            cov_sub: &TGA_COV_SUB,
            alpha: &TGA_ALPHA,
            efun: &TGA_EFUN,
            evals: &TGA_EVAL,
            eval_sum: TGA_EVAL_SUM,
            scores: &TGA_SCORES,
            theta: &TGA_THETA,
        },
    ]
}

/// Covariance step (registr `cov_hall`: mgcv P-spline GLM mean with GCV/UBRE,
/// te(ps) smooth with GCV, latent transform, nearPD) against R.
#[test]
fn two_step_covariance_matches_r() {
    for g in two_step_goldens() {
        let (d, _) = sim(g.seed, g.n, 30, g.code, 0);
        assert!((data_sum(&d) - g.sum).abs() < 1e-9);
        let cd = gfpca::curve_data(&d, "data").unwrap();
        let (grid, cov) = latent_covariance(&cd, g.family, &two_step_cfg(2)).unwrap();
        assert!(max_abs_diff(&grid, g.grid) < 1e-12, "{:?}", g.family);
        let idx: Vec<usize> = (0..grid.len()).step_by(3).collect();
        let sub: Vec<f64> = idx
            .iter()
            .flat_map(|&b| idx.iter().map(move |&a| (a, b)))
            .map(|(a, b)| cov[(a, b)])
            .collect();
        // Observed: 2e-7 (gaussian) … 4e-6 (binomial); the residual is the
        // tolerance of mgcv's Newton smoothing-parameter search.
        let err = rel_max(&sub, g.cov_sub);
        assert!(err < 2e-5, "{:?}: covariance rel err {err:e}", g.family);
    }
}

#[test]
fn two_step_matches_r() {
    for g in two_step_goldens() {
        let (d, _) = sim(g.seed, g.n, 30, g.code, 0);
        let fit = gfpca_two_step(&d, g.family, &two_step_cfg(2)).unwrap();
        let dd = g.grid.len();
        assert_eq!(fit.fpca_type, FpcaType::TwoStep);
        assert!(max_abs_diff(&fit.t_vec, g.grid) < 1e-12);
        // Eigen-decomposition of the covariance: cov matches to ≤ 4e-6.
        for k in 0..2 {
            let c = abs_cos(&col(&fit.efunctions, k), &g.efun[k * dd..(k + 1) * dd]);
            assert!(c > 1.0 - 1e-8, "{:?} efun {k}: cos {c}", g.family);
        }
        assert!(rel_max(&fit.evalues, g.evals) < 2e-5, "{:?}", g.family);
        assert!((fit.evalues_sum.unwrap() - g.eval_sum).abs() / g.eval_sum < 2e-5);
        // Mixed model: gaussian REML is exact up to optimiser tolerance; the
        // Laplace fits differ by lme4's Nelder–Mead stage-2 tolerance (observed
        // ≤ 1.9e-4 relative).
        let (tol_mean, tol_theta) = if g.family == ExpFamily::Gaussian {
            (1e-6, 1e-5)
        } else {
            (5e-4, 5e-4)
        };
        let alpha_err = rel_max(&fit.mean, g.alpha);
        assert!(alpha_err < tol_mean, "{:?}: alpha {alpha_err:e}", g.family);
        let mut scores = Vec::new();
        for k in 0..2 {
            let a = col(&fit.efunctions, k);
            let sign = a
                .iter()
                .zip(&g.efun[k * dd..(k + 1) * dd])
                .map(|(x, y)| x * y)
                .sum::<f64>()
                .signum();
            scores.extend((0..g.n).map(|i| sign * fit.scores[(i, k)]));
        }
        let score_err = rel_max(&scores, g.scores);
        assert!(score_err < tol_mean, "{:?}: scores {score_err:e}", g.family);
        // lme4 orders the `||` terms last-to-first: theta = (psi2, psi1, smooth).
        let vc = fit.variance_components.as_ref().unwrap();
        for k in 0..2 {
            let r = g.theta[1 - k];
            assert!(
                (vc[k] - r).abs() / r < tol_theta,
                "{:?} theta {k}",
                g.family
            );
        }
    }
}

#[test]
fn two_step_invariants() {
    let (d, _) = sim(202, 40, 30, 1, 0);
    let fit = gfpca_two_step(&d, ExpFamily::Binomial, &two_step_cfg(2)).unwrap();
    let dd = fit.t_vec.len();
    // Orthonormal eigenvectors, decreasing eigenvalues.
    for a in 0..2 {
        for b in 0..2 {
            let dot: f64 = (0..dd)
                .map(|j| fit.efunctions[(j, a)] * fit.efunctions[(j, b)])
                .sum();
            assert!((dot - f64::from(u8::from(a == b))).abs() < 1e-10);
        }
    }
    assert!(fit.evalues[0] >= fit.evalues[1]);
    // yhat = mean + scores · efunctionsᵀ; fitted = yhat at the observed times.
    let yhat = fit.yhat.as_ref().unwrap();
    for i in 0..40 {
        for j in 0..dd {
            let v = fit.mean[j]
                + (0..2)
                    .map(|k| fit.scores[(i, k)] * fit.efunctions[(j, k)])
                    .sum::<f64>();
            assert!((yhat[(i, j)] - v).abs() < 1e-12);
        }
        assert_eq!(fit.fitted[i].len(), 30);
        for j in 0..30 {
            assert!((fit.fitted[i][j] - yhat[(i, j)]).abs() < 1e-12);
        }
    }
    // Subject coefficients are the least-squares spline fits of the latent
    // curves (registr `registr_oneCurve`): residuals orthogonal to the basis.
    let basis = TemplateBasis::new(fit.domain[0], fit.domain[1], &fit.knots, false);
    for i in 0..40 {
        let c = fit.subject_coefs.row(i);
        let mut normal = vec![0.0; c.len()];
        for j in 0..dd {
            let b = basis.eval(fit.t_vec[j]);
            let r = yhat[(i, j)] - b.iter().zip(&c).map(|(x, y)| x * y).sum::<f64>();
            normal.iter_mut().zip(&b).for_each(|(nv, bv)| *nv += bv * r);
        }
        assert!(normal.iter().all(|v| v.abs() < 1e-9), "{normal:?}");
    }
}

/// Recovery: gaussian curves with two known orthogonal components.
#[test]
fn two_step_recovers_known_components() {
    use std::f64::consts::PI;
    let m = 25;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let phi1: Vec<f64> = t.iter().map(|&s| (2.0 * PI * s).sin()).collect();
    let phi2: Vec<f64> = t.iter().map(|&s| (2.0 * PI * s).cos()).collect();
    let mut rng = 7_u64;
    let mut unif = || {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut ts, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..40 {
        let a = 2.0 * (unif() - 0.5) * 3.0;
        let b = 2.0 * (unif() - 0.5);
        ts.push(t.clone());
        ys.push(
            (0..m)
                .map(|j| 1.0 + t[j] + a * phi1[j] + b * phi2[j] + 0.02 * (unif() - 0.5))
                .collect::<Vec<f64>>(),
        );
    }
    let d = IrregFdata::from_lists(&ts, &ys);
    let fit = gfpca_two_step(&d, ExpFamily::Gaussian, &two_step_cfg(2)).unwrap();
    let mut truth = Vec::new();
    for p in [&phi1, &phi2] {
        let n = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        truth.extend(p.iter().map(|v| v / n));
    }
    let est: Vec<f64> = (0..2).flat_map(|k| col(&fit.efunctions, k)).collect();
    assert!(subspace_min_cos(&est, &truth, m, 2) > 0.995);
    let mean_err = fit
        .mean
        .iter()
        .zip(&t)
        .map(|(a, s)| (a - 1.0 - s).abs())
        .fold(0.0, f64::max);
    assert!(mean_err < 0.1, "mean error {mean_err}");
}

// ----------------------------------------- register_fpca(two-step) e2e ----

#[test]
fn register_fpca_two_step_matches_r() {
    let cases: [(
        u64,
        usize,
        u8,
        ExpFamily,
        &[f64],
        &[f64],
        usize,
        &[f64],
        &[f64],
        &[f64],
        &[f64],
    ); 3] = [
        (
            101,
            15,
            0,
            ExpFamily::Gaussian,
            &RG_BETA,
            &RG_REGLOSS,
            RG_ITER,
            &RG_ALPHA,
            &RG_TVEC,
            &RG_EFUN,
            &RG_EVAL,
        ),
        (
            202,
            40,
            1,
            ExpFamily::Binomial,
            &RB_BETA,
            &RB_REGLOSS,
            RB_ITER,
            &RB_ALPHA,
            &RB_TVEC,
            &RB_EFUN,
            &RB_EVAL,
        ),
        (
            303,
            20,
            2,
            ExpFamily::Poisson,
            &RP_BETA,
            &RP_REGLOSS,
            RP_ITER,
            &RP_ALPHA,
            &RP_TVEC,
            &RP_EFUN,
            &RP_EVAL,
        ),
    ];
    for (seed, n, code, family, beta, regloss, iters, alpha, tvec, efun, evals) in cases {
        let (d, _) = sim(seed, n, 30, code, 0);
        let mut c = RegisterFpcaConfig::default();
        c.family = family;
        c.fpca_type = FpcaType::TwoStep;
        c.fpca.npc = 2;
        c.fpca.index_significant_digits = Some(3);
        let fit = register_fpca(&d, None, &c).unwrap();
        let rb = split(beta, 4);
        let db = fit
            .registration
            .hinv_beta
            .iter()
            .zip(&rb)
            .map(|(a, b)| max_abs_diff(a, b))
            .fold(0.0, f64::max);
        let loss_rel: Vec<f64> = fit
            .registration_loss
            .iter()
            .zip(regloss)
            .map(|(a, b)| (a - b) / b)
            .collect();
        let dd = tvec.len();
        let fd = fit.fpca.t_vec.len();
        let sub = if fd == dd {
            subspace_min_cos(fit.fpca.efunctions.as_slice(), efun, dd, 2)
        } else {
            f64::NAN
        };
        let ratio = (fit.fpca.evalues[0] / fit.fpca.evalues[1]) / (evals[0] / evals[1]);
        let alpha_err = if fd == dd {
            rel_max(&fit.fpca.mean, alpha)
        } else {
            f64::NAN
        };
        assert_eq!(fit.iterations, iters, "{family:?}");
        assert_eq!(fd, dd, "{family:?}");
        assert!(max_abs_diff(&fit.fpca.t_vec, tvec) < 1e-12);
        assert!(
            loss_rel[0].abs() < 1e-5,
            "{family:?}: initial loss {loss_rel:?}"
        );
        if family == ExpFamily::Gaussian {
            // Observed: |Δβ| 1.8e-6, loss 3e-6, subspace 1 − 7e-9, α 2.5e-7.
            assert!(db < 1e-4, "{family:?}: {db}");
            assert!(loss_rel.iter().all(|r| r.abs() < 1e-4), "{loss_rel:?}");
            assert!(
                sub > 1.0 - 1e-6 && (ratio - 1.0).abs() < 1e-3,
                "{sub} {ratio}"
            );
            assert!(alpha_err < 1e-5, "{alpha_err}");
        } else {
            // The initial registration already differs from R's Nelder–Mead /
            // barrier optimum (|Δβ| ≤ 1e-2, see tests.rs), which propagates; the
            // re-registrations reach losses at or below R's (observed binomial
            // −0.06 % / −0.24 %, poisson −0.011 %). FPCA agreement on identical
            // registered data is checked by `two_step_on_r_registered_data_matches_r`.
            assert!(db < 0.05, "{family:?}: {db}");
            assert!(
                loss_rel[1..].iter().all(|&r| r < 1e-6),
                "{family:?}: {loss_rel:?}"
            );
        }
    }
}

/// The final two-step GFPCA of `register_fpca` on R's registered data (sparse
/// irregular grids, regular cross-product branch) against R's `fpca_obj`.
#[test]
fn two_step_on_r_registered_data_matches_r() {
    let cases: [(
        u64,
        usize,
        u8,
        ExpFamily,
        &[f64],
        &[f64],
        &[f64],
        &[f64],
        &[f64],
    ); 2] = [
        (
            202,
            40,
            1,
            ExpFamily::Binomial,
            &RB_BETA,
            &RB_TVEC,
            &RB_EFUN,
            &RB_EVAL,
            &RB_ALPHA,
        ),
        (
            303,
            20,
            2,
            ExpFamily::Poisson,
            &RP_BETA,
            &RP_TVEC,
            &RP_EFUN,
            &RP_EVAL,
            &RP_ALPHA,
        ),
    ];
    for (seed, n, code, family, beta, tvec, efun, evals, alpha) in cases {
        let (d, _) = sim(seed, n, 30, code, 0);
        let reg = registered_with(&d, &split(beta, 4));
        let mut c = two_step_cfg(2);
        c.index_significant_digits = Some(3);
        let fit = gfpca_two_step(&reg, family, &c).unwrap();
        let dd = tvec.len();
        assert_eq!(fit.t_vec.len(), dd);
        assert!(max_abs_diff(&fit.t_vec, tvec) < 1e-12);
        // Observed: binomial subspace 1 − 1e-6, evalues 8.4e-4, α 6.4e-4 (lme4
        // Nelder–Mead tolerance); poisson 1 − 1e-13, 1e-6, 7e-6.
        let sub = subspace_min_cos(fit.efunctions.as_slice(), efun, dd, 2);
        assert!(sub > 1.0 - 1e-5, "{family:?}: {sub}");
        assert!(rel_max(&fit.evalues, evals) < 2e-3, "{family:?}");
        assert!(rel_max(&fit.mean, alpha) < 2e-3, "{family:?}");
    }
}

// ------------------------------------------------------- npc_criterion ----

#[test]
fn npc_criterion_rules() {
    // shares 0.5, 0.3, 0.1, 0.05, 0.05 (registr determine_npc: cumulative >=)
    let ev = [5.0, 3.0, 1.0, 0.5, 0.5];
    assert_eq!(NpcCriterion::new(0.8).choose(&ev), 2);
    assert_eq!(NpcCriterion::new(0.85).choose(&ev), 3);
    assert_eq!(NpcCriterion::with_min_share(0.85, 0.2).choose(&ev), 2);
    assert_eq!(NpcCriterion::with_min_share(0.85, 0.01).choose(&ev), 3);
    assert_eq!(NpcCriterion::with_min_share(0.85, 0.6).choose(&ev), 1);
    // fpca_gauss / bfpca: cumulative share strictly above the threshold
    assert_eq!(NpcCriterion::new(0.8).choose_variational(&ev), 3);
    assert_eq!(NpcCriterion::new(0.79).choose_variational(&ev), 2);
    let mut c = GfpcaConfig::default();
    c.npc_criterion = Some(NpcCriterion::new(1.5));
    assert!(c.validate().is_err());
}

#[test]
fn npc_criterion_two_step_matches_r() {
    let (d, _) = sim(303, 20, 30, 2, 0);
    let mut c = GfpcaConfig::default();
    c.npc_criterion = Some(NpcCriterion::with_min_share(0.95, 0.1));
    let fit = gfpca_two_step(&d, ExpFamily::Poisson, &c).unwrap();
    assert_eq!(fit.efunctions.ncols(), NC_TS_NPC);
    assert!(rel_max(&fit.evalues, &NC_TS_EVAL) < 2e-5);
    c.npc_criterion = Some(NpcCriterion::new(0.95));
    let fit = gfpca_two_step(&d, ExpFamily::Poisson, &c).unwrap();
    assert_eq!(fit.efunctions.ncols(), NC_TS1_NPC);
}

#[test]
fn npc_criterion_variational_matches_r() {
    let (d, _) = sim(101, 15, 50, 0, 0);
    let mut c = GfpcaConfig::default();
    c.npc_criterion = Some(NpcCriterion::new(0.9));
    c.psi_init = Some(FdMatrix::from_column_major(NC_PSI0.to_vec(), 20, 20).unwrap());
    let fit = gfpca_variational(&d, ExpFamily::Gaussian, &c).unwrap();
    assert_eq!(fit.efunctions.ncols(), NC_VE_NPC);
    assert_eq!(fit.alpha_coefs.len(), 20);
    assert!(
        rel_max(&fit.evalues, &NC_VE_EVAL) < 1e-6,
        "{:?}",
        fit.evalues
    );
    assert!((fit.evalues_sum.unwrap() - NC_VE_EVAL_SUM).abs() / NC_VE_EVAL_SUM < 1e-6);
    assert!(max_abs_diff(&fit.mean, &NC_VE_MU) < 1e-6);
    for k in 0..NC_VE_NPC {
        assert!(
            abs_cos(
                &col(&fit.efunctions, k),
                &NC_VE_EFUN[k * 100..(k + 1) * 100]
            ) > 1.0 - 1e-8
        );
    }
}

#[test]
fn npc_criterion_register_fpca_matches_r() {
    let (d, _) = sim(101, 15, 50, 0, 0);
    let mut c = RegisterFpcaConfig::default();
    c.fpca.npc_criterion = Some(NpcCriterion::new(0.9));
    let fit = register_fpca(&d, None, &c).unwrap();
    // registr uses Kt = 20 throughout (template knots included).
    assert_eq!(fit.registration.template_coefs.len(), 20);
    assert_eq!(fit.fpca.efunctions.ncols(), NC_RF_NPC);
    let db = fit
        .registration
        .hinv_beta
        .iter()
        .zip(split(&NC_RF_BETA, 4))
        .map(|(a, b)| max_abs_diff(a, &b))
        .fold(0.0, f64::max);
    // Different (seeded) EM starts in R and Rust, cf. the variational
    // register_fpca comparisons in tests.rs.
    assert!(db < 0.05, "{db}");
    let r_final = NC_RF_REGLOSS[NC_RF_REGLOSS.len() - 1];
    let ours = fit.registration_loss[fit.registration_loss.len() - 1];
    assert!((fit.registration_loss[0] - NC_RF_REGLOSS[0]).abs() / NC_RF_REGLOSS[0] < 1e-5);
    assert!((ours - r_final) / r_final < 5e-3, "{ours} vs {r_final}");
}

// ------------------------------------------------------------ periodic ----

#[test]
fn periodic_basis_matches_pbs() {
    let b = PeriodicBs::new(0.0, 1.0, &[0.2, 0.35, 0.6, 0.8]);
    assert_eq!(b.nbasis(), 5);
    for (r, &x) in PB_X.iter().enumerate() {
        let v = b.eval(x);
        for k in 0..5 {
            assert!(
                (v[k] - PB_BASIS[r + k * PB_X.len()]).abs() < 1e-12,
                "x {x} k {k}"
            );
        }
    }
    for (r, &x) in PB_X[1..7].iter().enumerate() {
        let v = b.deriv(x);
        for k in 0..5 {
            assert!((v[k] - PB_DERIV_FD[r + k * 6]).abs() < 1e-5, "x {x} k {k}");
        }
    }
    // Periodic: values and first two derivatives agree at both ends; partition of unity.
    let (a, z) = (b.eval(0.0), b.eval(1.0));
    let (da, dz) = (b.deriv(0.0), b.deriv(1.0));
    assert!(max_abs_diff(&a, &z) < 1e-12 && max_abs_diff(&da, &dz) < 1e-9);
    assert!((b.eval(0.437).iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn periodic_registration_and_fpca_match_r() {
    let (d, _) = sim(101, 15, 50, 0, 0);
    let mut c = RegisterFpcaConfig::default();
    c.fpca.periodic = true;
    c.fpca.npc = 2;
    let reg = register_expfam(&d, None, &c).unwrap();
    // R minimises with Nelder–Mead (no gradient for periodic = TRUE).
    assert!(
        reg.loss <= PER_INIT_LOSS * (1.0 + 1e-6),
        "{} vs {PER_INIT_LOSS}",
        reg.loss
    );
    let rb = split(&PER_INIT_BETA, 4);
    let db = reg
        .hinv_beta
        .iter()
        .zip(&rb)
        .map(|(a, b)| max_abs_diff(a, b))
        .fold(0.0, f64::max);
    assert!(db < 0.02, "{db}");
    // fpca_gauss(periodic = TRUE) on R's registered data from R's psi start.
    let registered = registered_with(&d, &rb);
    let mut g = GfpcaConfig::default();
    g.npc = 2;
    g.periodic = true;
    g.psi_init = Some(FdMatrix::from_column_major(PER_PSI0.to_vec(), 8, 2).unwrap());
    let fit = gfpca_variational(&registered, ExpFamily::Gaussian, &g).unwrap();
    assert!(max_abs_diff(&fit.knots, &PER_FPCA_KNOTS) < 1e-12);
    assert!(max_abs_diff(&fit.mean, &PER_FPCA_MU) < 1e-7);
    assert!(rel_max(&fit.evalues, &PER_FPCA_EVAL) < 1e-6);
    for k in 0..2 {
        assert!(
            abs_cos(
                &col(&fit.efunctions, k),
                &PER_FPCA_EFUN[k * 100..(k + 1) * 100]
            ) > 1.0 - 1e-10
        );
    }
    // Template knots: Kt − 1 = 7 quantiles (pbs), Kt = 8 basis functions.
    assert_eq!(reg.template_knots.len(), 7);
    assert_eq!(reg.template_coefs.len(), 8);
    let full = register_fpca(&d, None, &c).unwrap();
    let r_final = PER_FULL_REGLOSS[PER_FULL_REGLOSS.len() - 1];
    let ours = full.registration_loss[full.registration_loss.len() - 1];
    assert!((ours - r_final) / r_final < 1e-2, "{ours} vs {r_final}");
    let db = full
        .registration
        .hinv_beta
        .iter()
        .zip(split(&PER_FULL_BETA, 4))
        .map(|(a, b)| max_abs_diff(a, &b))
        .fold(0.0, f64::max);
    assert!(db < 0.05, "{db}");
}

// ------------------------------------------------- piecewise_linear2 ----

#[test]
fn piecewise_linear_hinv_matches_registr() {
    // registr piecewise_linear2_hinv(grid, c(0.25, 0.3, 0.75, 0.9))
    let b = [0.25, 0.3, 0.75, 0.9];
    for (t, want) in [(0.0, 0.0), (0.1, 0.12), (0.5, 0.6), (0.8, 0.92), (1.0, 1.0)] {
        assert!((hinv(t, &b).0 - want).abs() < 1e-14, "{t}");
    }
}

#[test]
fn piecewise_linear_gradients_match_finite_differences() {
    let basis = TemplateBasis::Bs(super::basis::CubicBs::new(0.0, 1.0, &[0.3, 0.5, 0.7]));
    let coefs: Vec<f64> = (0..7)
        .map(|k| 1.0 + 0.4 * ((k as f64) * 1.3).sin())
        .collect();
    let s: Vec<f64> = (0..20).map(|j| (j as f64 + 0.37) / 20.3).collect();
    let y: Vec<f64> = s.iter().map(|&t| 1.2 + (3.0 * t).sin().abs()).collect();
    let beta = [0.27, 0.33, 0.71, 0.8];
    for (family, prior) in [
        (ExpFamily::Gaussian, None),
        (ExpFamily::Binomial, Some(0.1)),
        (ExpFamily::Poisson, None),
        (ExpFamily::Gamma, None),
    ] {
        let yy: Vec<f64> = if family == ExpFamily::Binomial {
            y.iter().map(|v| f64::from(u8::from(*v > 1.6))).collect()
        } else {
            y.clone()
        };
        let prob = Pl2Problem {
            s: &s,
            y: &yy,
            template_basis: &basis,
            template_coefs: &coefs,
            loss: ObsLoss {
                family,
                scale: TemplateScale::Link,
            },
            prior_sd: prior,
        };
        let (_, g, gs) = prob.loss_grad(&beta, 1.7);
        for k in 0..4 {
            let h = 1e-6;
            let (mut bp, mut bm) = (beta, beta);
            bp[k] += h;
            bm[k] -= h;
            let fd = (prob.loss_grad(&bp, 1.7).0 - prob.loss_grad(&bm, 1.7).0) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() < 1e-5 * (1.0 + fd.abs()),
                "{family:?} {k}: {fd} vs {}",
                g[k]
            );
        }
        let fd_s =
            (prob.loss_grad(&beta, 1.7 + 1e-6).0 - prob.loss_grad(&beta, 1.7 - 1e-6).0) / 2e-6;
        assert!(
            (fd_s - gs).abs() < 1e-5 * (1.0 + fd_s.abs()),
            "{family:?} scale"
        );
    }
}

#[test]
fn piecewise_linear_registration_matches_r() {
    let cases: [(u64, usize, usize, u8, ExpFamily, Option<f64>, &[f64], f64); 3] = [
        (
            101,
            15,
            50,
            0,
            ExpFamily::Gaussian,
            None,
            &PLG_BETA,
            PLG_LOSS,
        ),
        (
            202,
            40,
            60,
            1,
            ExpFamily::Binomial,
            None,
            &PLB_BETA,
            PLB_LOSS,
        ),
        (
            202,
            40,
            60,
            1,
            ExpFamily::Binomial,
            Some(0.1),
            &PLBP_BETA,
            PLBP_LOSS,
        ),
    ];
    for (seed, n, m, code, family, prior_sd, r_beta, r_loss) in cases {
        let (d, _) = sim(seed, n, m, code, 0);
        let mut c = RegisterFpcaConfig::default();
        c.family = family;
        c.warping = WarpingType::PiecewiseLinear2;
        c.prior_sd = prior_sd;
        let reg = register_expfam(&d, None, &c).unwrap();
        let rb = split(r_beta, 4);
        // registr's loss at R's coefficients, evaluated by fdars (template GLM
        // and loss incl. priors).
        let basis = TemplateBasis::new(0.0, 1.0, &reg.template_knots, false);
        let s: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let at_r: f64 = (0..n)
            .map(|i| {
                let prob = Pl2Problem {
                    s: &s,
                    y: d.get_obs(i).1,
                    template_basis: &basis,
                    template_coefs: &reg.template_coefs,
                    loss: ObsLoss {
                        family,
                        scale: TemplateScale::Link,
                    },
                    prior_sd,
                };
                prob.loss_grad(&rb[i], 1.0).0
            })
            .sum();
        assert!(
            (at_r - r_loss).abs() / r_loss < 1e-8,
            "{family:?}: {at_r} vs {r_loss}"
        );
        // R: constrOptim + Nelder–Mead from (0.25, 0.3, 0.75, 0.8); fdars
        // (L-BFGS + Nelder–Mead from that start and from the best point of a
        // coarse knot grid) reaches the same or a lower loss.
        assert!(
            reg.loss <= r_loss * (1.0 + 1e-6),
            "{family:?}: {} vs {r_loss}",
            reg.loss
        );
        for (b, th) in reg.hinv_beta.iter().zip(&reg.t_hat) {
            assert!(b[0] >= 0.1 - 1e-12 && b[2] - b[0] >= 0.1 - 1e-12 && b[2] <= 0.9 + 1e-12);
            assert!(b[1] >= 0.1 - 1e-12 && b[3] - b[1] >= 0.1 - 1e-12 && b[3] <= 0.9 + 1e-12);
            assert!(th.windows(2).all(|w| w[1] >= w[0]));
            assert!(th[0].abs() < 1e-12 && (th[th.len() - 1] - 1.0).abs() < 1e-12);
        }
        let db = reg
            .hinv_beta
            .iter()
            .zip(&rb)
            .map(|(a, b)| max_abs_diff(a, b))
            .fold(0.0, f64::max);
        // Observed: gaussian 36.06 vs 36.78, binomial 1152.08 vs 1158.59 (other
        // local optima of a multi-modal loss); with priors 990.8307 vs 990.8310
        // and |Δβ| 0.011.
        if prior_sd.is_some() {
            assert!(db < 0.02, "{db}");
        }
    }
    // Priors only with piecewise-linear warps and binary data (registr).
    let (d, _) = sim(101, 4, 20, 0, 0);
    let mut c = RegisterFpcaConfig::default();
    c.prior_sd = Some(0.1);
    assert!(register_expfam(&d, None, &c).is_err());
    c.warping = WarpingType::PiecewiseLinear2;
    assert!(register_expfam(&d, None, &c).is_err());
    c.prior_sd = None;
    c.incompleteness = Some(CurveIncompleteness::Leading);
    assert!(register_expfam(&d, None, &c).is_err());
}

#[test]
fn piecewise_linear_register_fpca_runs_on_scaled_domain() {
    let (d, _) = sim(202, 30, 40, 1, 0);
    let mut c = RegisterFpcaConfig::default();
    c.family = ExpFamily::Binomial;
    c.warping = WarpingType::PiecewiseLinear2;
    c.max_iterations = 3;
    let fit = register_fpca(&d, None, &c).unwrap();
    assert!(fit.registration_loss.len() >= 2);
    for th in &fit.registration.t_hat {
        assert!(th.windows(2).all(|w| w[1] >= w[0]));
        assert!(th[0].abs() < 1e-12 && (th[th.len() - 1] - 1.0).abs() < 1e-12);
    }
    assert!(fit.registration.hinv_inner_knots.iter().all(Vec::is_empty));
}
