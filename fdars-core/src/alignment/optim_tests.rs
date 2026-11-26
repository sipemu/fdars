//! Tests for the configurable warp optimiser (ALN-RBFGS #42, ALN-PEN #43).

// Configs are built the way external callers must (sealed struct: `default()`
// + field assignment), so this lint is silenced for the test module.
#![allow(clippy::field_reassign_with_default)]

use super::golden::*;
use super::*;
use crate::alignment::dp_alignment_core_banded;
use crate::alignment::rbfgs::rbfgs_data_cost;
use crate::alignment::{
    elastic_align_pair, elastic_align_pair_penalized, elastic_align_pair_with_config,
    elastic_cross_distance_matrix, elastic_cross_distance_matrix_with_config, elastic_distance,
    elastic_distance_with_config, elastic_self_distance_matrix,
    elastic_self_distance_matrix_with_config, karcher_mean, karcher_mean_with_band,
    karcher_mean_with_config, WarpPenaltyType,
};
use crate::matrix::FdMatrix;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn bump(t: f64, c: f64, s: f64) -> f64 {
    (-(t - c).powi(2) / s).exp()
}

fn case_a(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = grid(m);
    let q1 = t
        .iter()
        .map(|&s| 3.0 * bump(s, 0.35, 0.01) - 2.0 * bump(s, 0.7, 0.01) + 0.2)
        .collect();
    let q2 = t
        .iter()
        .map(|&s| 3.0 * bump(s, 0.45, 0.01) - 2.0 * bump(s, 0.78, 0.01) + 0.2)
        .collect();
    (t, q1, q2)
}

fn case_b(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = grid(m);
    let pi = std::f64::consts::PI;
    let q1 = t.iter().map(|&s| (4.0 * pi * s).sin() + 0.5).collect();
    let q2 = t
        .iter()
        .map(|&s| (4.0 * pi * s.powf(1.3)).sin() + 0.5)
        .collect();
    (t, q1, q2)
}

fn rbfgs_cfg() -> WarpOptimConfig {
    let mut c = WarpOptimConfig::default();
    c.method = WarpOptimMethod::Rbfgs;
    c
}

fn maxabs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ─── Bit-identity of the default path ───────────────────────────────────────

fn fnv(h: &mut u64, xs: &[f64]) {
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

fn probe_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let c = 0.3 + 0.05 * i as f64;
        for j in 0..m {
            let s = t[j];
            d[(i, j)] = (-(s - c).powi(2) / 0.01).exp()
                + 0.5 * (-(s - c - 0.3).powi(2) / 0.02).exp()
                + 0.1 * (3.0 * s + i as f64).sin();
        }
    }
    (d, t)
}

fn hash_pair(r: &crate::alignment::AlignmentResult) -> u64 {
    let mut h = FNV0;
    fnv(&mut h, &r.gamma);
    fnv(&mut h, &r.f_aligned);
    fnv(&mut h, &[r.distance]);
    h
}

fn hash_karcher(k: &crate::alignment::KarcherMeanResult) -> u64 {
    let mut h = FNV0;
    fnv(&mut h, &k.mean);
    fnv(&mut h, &k.mean_srsf);
    fnv(&mut h, k.gammas.as_slice());
    fnv(&mut h, k.aligned_data.as_slice());
    fnv(&mut h, &[k.n_iter as f64, f64::from(u8::from(k.converged))]);
    h
}

/// FNV-1a hashes of the `f64` bit patterns produced by the classic entry
/// points **before** this change (recorded by running the same probe on
/// commit c1f088a4 in a scratch worktree). Proves the refactor that threads
/// `WarpOptimConfig` through the DP / pairwise / Karcher code is bit-identical,
/// and that the default `WarpOptimConfig` reproduces those results exactly.
#[test]
fn classic_and_default_config_are_bit_identical_to_pre_change() {
    let (d, t) = probe_data(6, 60);
    let (d2, _) = probe_data(3, 60);
    let expected: [(f64, [u64; 6]); 2] = [
        (
            0.0,
            [
                0xfc91_63b5_3316_0432,
                0x7411_8ee8_0312_7059,
                0xabce_ac94_b41d_64f8,
                0x36fc_238e_0c4d_d892,
                0xc8f3_cd47_a334_8897,
                0xff39_05a6_00e7_2727,
            ],
        ),
        (
            0.3,
            [
                0xb112_16db_fb2e_2ef5,
                0xf9bf_fe8c_0d65_8dfd,
                0x2a29_0d67_995d_bc70,
                0xe81d_36cd_5ec3_a269,
                0x8979_7554_acf4_fff1,
                0x5149_2896_5956_315e,
            ],
        ),
    ];
    for (lam, [h_pair, h_self, h_cross, h_k, h_kb, h_pen]) in expected {
        let mut cfg = WarpOptimConfig::default();
        cfg.lambda = lam;
        let (f1, f2) = (d.row(0), d.row(3));

        let classic = elastic_align_pair(&f1, &f2, &t, lam);
        assert_eq!(hash_pair(&classic), h_pair, "pair lam={lam}");
        let viacfg = elastic_align_pair_with_config(&f1, &f2, &t, &cfg).unwrap();
        assert_eq!(viacfg, classic);
        assert_eq!(
            elastic_distance_with_config(&f1, &f2, &t, &cfg)
                .unwrap()
                .to_bits(),
            elastic_distance(&f1, &f2, &t, lam).to_bits()
        );

        let s = elastic_self_distance_matrix(&d, &t, lam);
        let mut h = FNV0;
        fnv(&mut h, s.as_slice());
        assert_eq!(h, h_self, "self lam={lam}");
        assert_eq!(
            elastic_self_distance_matrix_with_config(&d, &t, &cfg).unwrap(),
            s
        );

        let c = elastic_cross_distance_matrix(&d, &d2, &t, lam);
        let mut h = FNV0;
        fnv(&mut h, c.as_slice());
        assert_eq!(h, h_cross, "cross lam={lam}");
        assert_eq!(
            elastic_cross_distance_matrix_with_config(&d, &d2, &t, &cfg).unwrap(),
            c
        );

        // m = 60 > 50 and max_iter = 10 exercise the coarse-to-fine phases.
        let k = karcher_mean(&d, &t, 10, 1e-4, lam);
        assert_eq!(hash_karcher(&k), h_k, "karcher lam={lam}");
        assert_eq!(karcher_mean_with_config(&d, &t, 10, 1e-4, &cfg).unwrap(), k);

        let kb = karcher_mean_with_band(&d, &t, 10, 1e-4, lam, Some(0.2));
        let mut h = FNV0;
        fnv(&mut h, &kb.mean);
        fnv(&mut h, kb.gammas.as_slice());
        assert_eq!(h, h_kb, "karcher_band lam={lam}");

        let p = elastic_align_pair_penalized(&f1, &f2, &t, lam, WarpPenaltyType::SecondOrder);
        let mut h = FNV0;
        fnv(&mut h, &p.gamma);
        fnv(&mut h, &[p.distance]);
        assert_eq!(h, h_pen, "penalized SecondOrder lam={lam}");
    }
}

#[test]
fn default_config_bit_identical_on_nonuniform_grid() {
    let t: Vec<f64> = (0..45).map(|i| (i as f64 / 44.0).powf(1.4) * 2.0).collect();
    let f1: Vec<f64> = t.iter().map(|&s| (2.0 * s).sin() + 0.2 * s).collect();
    let f2: Vec<f64> = t.iter().map(|&s| (2.0 * s + 0.4).sin()).collect();
    for lam in [0.0, 0.2] {
        let mut cfg = WarpOptimConfig::default();
        cfg.lambda = lam;
        assert_eq!(
            elastic_align_pair_with_config(&f1, &f2, &t, &cfg).unwrap(),
            elastic_align_pair(&f1, &f2, &t, lam)
        );
    }
}

// ─── ALN-PEN: in-objective penalties in DP ───────────────────────────────────

fn identity_dev(g: &[f64], t: &[f64]) -> f64 {
    maxabs(g, t)
}

#[test]
fn dp_l2psi_zero_lambda_is_classic_and_large_lambda_is_identity() {
    let (t, q1, q2) = case_a(41);
    let mut cfg = WarpOptimConfig::default();
    cfg.penalty = WarpObjectivePenalty::L2Psi;
    let g0 = optimal_warp(&q1, &q2, &t, &cfg, None);
    assert_eq!(g0, dp_alignment_core_banded(&q1, &q2, &t, 0.0, None));

    let mut devs = Vec::new();
    for lam in [0.0, 0.05, 0.5, 1e4] {
        cfg.lambda = lam;
        let g = optimal_warp(&q1, &q2, &t, &cfg, None);
        devs.push(identity_dev(&g, &t));
    }
    // More penalty, warp no farther from the identity; a huge one forces it.
    for w in devs.windows(2) {
        assert!(w[1] <= w[0] + 1e-12, "{devs:?}");
    }
    assert!(devs[0] > 0.05, "{devs:?}");
    assert!(devs[3] < 1e-12, "{devs:?}");
}

#[test]
fn dp_l2psi_edge_penalty_is_hand_computable() {
    // One edge of slope 4 over dt = 0.5: λ(√4 − 1)²·0.5 = 0.5λ.
    let argvals = [0.0, 0.5, 1.0, 1.5, 2.0];
    let p = dp_l2psi_penalty(&argvals, 0, 1, 0, 4, 2.0);
    assert!((p - 1.0).abs() < 1e-15, "{p}");
    // The first-order (l2gam) penalty of the same edge: λ(4 − 1)²·0.5 = 9.
    let p = crate::alignment::dp_lambda_penalty(&argvals, 0, 1, 0, 4, 2.0);
    assert!((p - 9.0).abs() < 1e-15, "{p}");
}

#[test]
fn dp_rejects_non_additive_penalties_with_positive_lambda() {
    let t = grid(20);
    let f: Vec<f64> = t.iter().map(|&s| s * s).collect();
    for pen in [
        WarpObjectivePenalty::Roughness,
        WarpObjectivePenalty::Geodesic,
    ] {
        let mut cfg = WarpOptimConfig::default();
        cfg.penalty = pen;
        cfg.lambda = 0.1;
        assert!(matches!(
            elastic_align_pair_with_config(&f, &f, &t, &cfg),
            Err(FdarError::InvalidParameter {
                parameter: "penalty",
                ..
            })
        ));
        // lam = 0: the penalty drops out, accepted and equal to classic DP.
        cfg.lambda = 0.0;
        assert_eq!(
            elastic_align_pair_with_config(&f, &f, &t, &cfg).unwrap(),
            elastic_align_pair(&f, &f, &t, 0.0)
        );
        // RBFGS accepts them.
        cfg.lambda = 0.1;
        cfg.method = WarpOptimMethod::Rbfgs;
        assert!(elastic_align_pair_with_config(&f, &f, &t, &cfg).is_ok());
    }
}

#[test]
fn config_validation_errors() {
    let t = grid(10);
    let f: Vec<f64> = t.to_vec();
    let mut cfg = WarpOptimConfig::default();
    cfg.lambda = -1.0;
    assert!(cfg.validate().is_err());
    cfg.lambda = f64::NAN;
    assert!(cfg.validate().is_err());
    let mut cfg = WarpOptimConfig::default();
    cfg.scl = 0;
    assert!(cfg.validate().is_err());
    let cfg = WarpOptimConfig::default();
    assert!(elastic_align_pair_with_config(&f, &f[..9], &t, &cfg).is_err());
    let bad_t: Vec<f64> = (0..10)
        .map(|i| if i == 5 { 0.1 } else { i as f64 })
        .collect();
    assert!(elastic_align_pair_with_config(&f, &f, &bad_t, &cfg).is_err());
    assert!(karcher_mean_with_config(&FdMatrix::zeros(0, 10), &t, 3, 1e-3, &cfg).is_err());
    assert!(elastic_self_distance_matrix_with_config(&FdMatrix::zeros(2, 9), &t, &cfg).is_err());
}

// ─── ALN-RBFGS: optimiser behaviour ──────────────────────────────────────────

fn assert_valid_warp(g: &[f64], t: &[f64]) {
    assert_eq!(g.len(), t.len());
    assert_eq!(g[0], t[0]);
    assert_eq!(g[g.len() - 1], t[t.len() - 1]);
    assert!(g.windows(2).all(|w| w[1] >= w[0]), "not monotone");
}

#[test]
fn rbfgs_cost_le_dp_cost_and_warps_valid() {
    // The same discretised cost (RBFGS's) evaluated at both warps.
    for (t, q1, q2) in [case_a(41), case_b(41), case_a(101), case_b(151)] {
        let g_rb = optimal_warp(&q1, &q2, &t, &rbfgs_cfg(), None);
        let g_dp = optimal_warp(&q1, &q2, &t, &WarpOptimConfig::default(), None);
        assert_valid_warp(&g_rb, &t);
        let c_rb = rbfgs_data_cost(&q1, &q2, &t, &g_rb);
        let c_dp = rbfgs_data_cost(&q1, &q2, &t, &g_dp);
        assert!(c_rb <= c_dp, "m={}: rbfgs {c_rb} > dp {c_dp}", t.len());
    }
}

/// fdasrsf 2.7.5 `optimum_reparam(q1, t, q2, method="RBFGS")` (embedded warps,
/// see `optim_golden.rs` / `golden_rbfgs.py`). RBFGS is a local optimiser and
/// fdars seeds it with its own 35-move DP instead of fdasrsf's 63-slope DP2,
/// so exact equality is not expected. Documented tolerances: warps within
/// 0.015 (sup norm), final cost within 5% of fdasrsf's (both evaluated with
/// the same discretised cost), and below fdasrsf's own DP warp's cost.
#[test]
fn rbfgs_matches_fdasrsf_rbfgs_golden() {
    for (name, (t, q1, q2), fda_rb, fda_dp) in [
        (
            "A41",
            case_a(41),
            &FDASRSF_RBFGS_A41[..],
            &FDASRSF_DP_A41[..],
        ),
        (
            "B41",
            case_b(41),
            &FDASRSF_RBFGS_B41[..],
            &FDASRSF_DP_B41[..],
        ),
        (
            "A101",
            case_a(101),
            &FDASRSF_RBFGS_A101[..],
            &FDASRSF_DP_A101[..],
        ),
    ] {
        let g = optimal_warp(&q1, &q2, &t, &rbfgs_cfg(), None);
        let dev = maxabs(&g, fda_rb);
        assert!(dev < 0.015, "{name}: sup |γ − γ_fdasrsf| = {dev}");
        let c = rbfgs_data_cost(&q1, &q2, &t, &g);
        let c_ref = rbfgs_data_cost(&q1, &q2, &t, fda_rb);
        let c_dp_ref = rbfgs_data_cost(&q1, &q2, &t, fda_dp);
        assert!(
            (c / c_ref - 1.0).abs() < 0.05,
            "{name}: cost {c} vs fdasrsf {c_ref}"
        );
        assert!(c < c_dp_ref, "{name}: cost {c} vs fdasrsf DP {c_dp_ref}");
    }
}

#[test]
fn rbfgs_recovers_known_warp() {
    // f2 = g ∘ h⁻¹, so aligning f2 to f1 = g recovers γ = h exactly in theory.
    let m = 101;
    let t = grid(m);
    let a = 1.2_f64;
    let h = |s: f64| ((a * s).exp() - 1.0) / (a.exp() - 1.0);
    let hinv = |s: f64| (1.0 + s * (a.exp() - 1.0)).ln() / a;
    let g = |s: f64| bump(s, 0.3, 0.008) + 0.6 * bump(s, 0.65, 0.01) + 0.3 * s;
    let f1: Vec<f64> = t.iter().map(|&s| g(s)).collect();
    let f2: Vec<f64> = t.iter().map(|&s| g(hinv(s))).collect();
    let truth: Vec<f64> = t.iter().map(|&s| h(s)).collect();

    let rb = elastic_align_pair_with_config(&f1, &f2, &t, &rbfgs_cfg()).unwrap();
    let dp = elastic_align_pair(&f1, &f2, &t, 0.0);
    assert_valid_warp(&rb.gamma, &t);
    let err_rb = maxabs(&rb.gamma, &truth);
    let err_dp = maxabs(&dp.gamma, &truth);
    assert!(err_rb < 0.01, "rbfgs warp error {err_rb} (dp {err_dp})");
    // Aligned curve reproduces the template.
    let amp = maxabs(&rb.f_aligned, &f1);
    assert!(amp < 0.05, "aligned curve error {amp}");
}

#[test]
fn rbfgs_identical_curves_give_identity() {
    let t = grid(50);
    let f: Vec<f64> = t.iter().map(|&s| (5.0 * s).sin() + s).collect();
    let r = elastic_align_pair_with_config(&f, &f, &t, &rbfgs_cfg()).unwrap();
    assert!(maxabs(&r.gamma, &t) < 1e-8, "{}", maxabs(&r.gamma, &t));
    assert!(r.distance < 1e-6, "{}", r.distance);
}

fn bending(g: &[f64], t: &[f64]) -> f64 {
    let h = t[1] - t[0];
    (1..g.len() - 1)
        .map(|j| ((g[j + 1] - 2.0 * g[j] + g[j - 1]) / (h * h)).powi(2) * h)
        .sum()
}

#[test]
fn rbfgs_in_objective_penalties_pull_towards_identity() {
    let (t, q1, q2) = case_a(61);
    let phase = |g: &[f64]| crate::warping::phase_distance(g, &t);
    for pen in [
        WarpObjectivePenalty::L2Gam,
        WarpObjectivePenalty::L2Psi,
        WarpObjectivePenalty::Roughness,
        WarpObjectivePenalty::Geodesic,
    ] {
        let mut cfg = rbfgs_cfg();
        cfg.penalty = pen;
        let g0 = optimal_warp(&q1, &q2, &t, &cfg, None);
        cfg.lambda = 1e5;
        let gbig = optimal_warp(&q1, &q2, &t, &cfg, None);
        assert_valid_warp(&gbig, &t);
        let (d0, dbig) = (identity_dev(&g0, &t), identity_dev(&gbig, &t));
        assert!(
            d0 > 0.05,
            "{pen:?}: unpenalised warp too close to identity {d0}"
        );
        assert!(dbig < 0.25 * d0, "{pen:?}: {dbig} vs {d0}");
        assert!(phase(&gbig) < phase(&g0), "{pen:?}");
    }
}

#[test]
fn rbfgs_roughness_penalty_reduces_bending_energy() {
    let (t, q1, q2) = case_b(61);
    let mut cfg = rbfgs_cfg();
    cfg.penalty = WarpObjectivePenalty::Roughness;
    let mut energies = Vec::new();
    for lam in [0.0, 1.0, 100.0] {
        cfg.lambda = lam;
        let g = optimal_warp(&q1, &q2, &t, &cfg, None);
        energies.push(bending(&g, &t));
    }
    assert!(energies[1] < energies[0], "{energies:?}");
    assert!(energies[2] < energies[1], "{energies:?}");
}

#[test]
fn rbfgs_karcher_mean_and_distance_matrix() {
    let (d, t) = probe_data(6, 60);
    let cfg = rbfgs_cfg();
    let k = karcher_mean_with_config(&d, &t, 10, 1e-4, &cfg).unwrap();
    let kd = karcher_mean(&d, &t, 10, 1e-4, 0.0);
    assert_eq!(k.mean.len(), 60);
    for i in 0..d.nrows() {
        assert_valid_warp(&k.gammas.row(i), &t);
    }
    // Same template up to discretisation: means close in sup norm relative to range.
    let range = kd.mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - kd.mean.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(maxabs(&k.mean, &kd.mean) < 0.15 * range);

    let dm = elastic_self_distance_matrix_with_config(&d, &t, &cfg).unwrap();
    let dm_dp = elastic_self_distance_matrix(&d, &t, 0.0);
    for i in 0..6 {
        assert_eq!(dm[(i, i)], 0.0);
        for j in 0..6 {
            assert_eq!(dm[(i, j)], dm[(j, i)]);
            if i != j {
                assert!(dm[(i, j)] > 0.0);
                // RBFGS distances on the same scale as DP's.
                assert!((dm[(i, j)] / dm_dp[(i, j)] - 1.0).abs() < 0.5);
            }
        }
    }
    let cross = elastic_cross_distance_matrix_with_config(&d, &d, &t, &cfg).unwrap();
    for i in 0..6 {
        for j in 0..6 {
            if i < j {
                assert_eq!(cross[(i, j)], dm[(i, j)]);
            }
        }
    }
}

#[test]
fn rbfgs_is_deterministic() {
    let (t, q1, q2) = case_b(80);
    let a = optimal_warp(&q1, &q2, &t, &rbfgs_cfg(), None);
    let b = optimal_warp(&q1, &q2, &t, &rbfgs_cfg(), None);
    assert_eq!(a, b);
}
