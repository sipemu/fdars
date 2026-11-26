//! Tests for the RBFGS / DP-option coverage of the remaining alignment entry
//! points (ALN-CAV #67): N-d shared-warp alignment, landmark-constrained
//! alignment, robust Karcher, `align_to_target`, closed-curve Karcher, the
//! Sakoe–Chiba band under RBFGS, the RBFGS coarse-DP seed options and the
//! DP ↔ RBFGS λ conversion.

// Configs are built the way external callers must (sealed struct: `default()`
// + field assignment), so this lint is silenced for the test module.
#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::alignment::{
    align_to_target, curve_shape_align_pair, elastic_align_pair_constrained, elastic_align_pair_nd,
    elastic_align_pair_with_config, karcher_mean_closed, karcher_mean_nd, karcher_mean_with_config,
    karcher_median, robust_karcher_mean, CurveShapeConfig, RobustKarcherConfig,
};
use crate::matrix::{FdCurveSet, FdMatrix};

// ─── Probe data ─────────────────────────────────────────────────────────────

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn fnv(h: &mut u64, xs: &[f64]) {
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// `n` bumpy curves with shifted features (full-rank, non-degenerate).
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

/// One R^2 curve as an `FdCurveSet` with a single observation.
fn curve_2d(t: &[f64], shift: f64) -> FdCurveSet {
    let m = t.len();
    let x: Vec<f64> = t
        .iter()
        .map(|&s| s + 0.4 * (-(s - 0.4 - shift).powi(2) / 0.02).exp())
        .collect();
    let y: Vec<f64> = t
        .iter()
        .map(|&s| (5.0 * (s - shift)).sin() + 0.3 * s)
        .collect();
    FdCurveSet {
        dims: vec![
            FdMatrix::from_slice(&x, 1, m).unwrap(),
            FdMatrix::from_slice(&y, 1, m).unwrap(),
        ],
    }
}

fn periodic_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let ph = 0.07 * i as f64;
        for j in 0..m {
            let s = t[j];
            d[(i, j)] = (tau * (s + ph)).sin() + 0.5 * (2.0 * tau * (s + ph)).cos();
        }
    }
    (d, t)
}

fn curve_shape_pair(m: usize) -> (FdMatrix, FdMatrix) {
    let mk = |g: &dyn Fn(f64) -> f64| {
        let mut b = FdMatrix::zeros(2, m);
        for j in 0..m {
            let s = g(j as f64 / (m - 1) as f64);
            b[(0, j)] = s;
            b[(1, j)] = 0.3 * (6.0 * s).sin();
        }
        b
    };
    (mk(&|s| s), mk(&|s| s + 0.25 * s * (1.0 - s)))
}

fn rbfgs_cfg() -> WarpOptimConfig {
    let mut c = WarpOptimConfig::default();
    c.method = WarpOptimMethod::Rbfgs;
    c
}

// ─── Bit-identity with the pre-#67 code ─────────────────────────────────────

/// Hashes of every pre-existing entry point this change touches, each run on
/// the probe data with `lambda ∈ {0, 0.05}`.
fn pre_change_hashes() -> Vec<u64> {
    let mut out = Vec::new();
    let (d, t) = probe_data(5, 50);
    for lam in [0.0, 0.05] {
        // N-d pair + Karcher.
        let mut h = FNV0;
        let c1 = curve_2d(&t, 0.0);
        let c2 = curve_2d(&t, 0.06);
        let r = elastic_align_pair_nd(&c1, &c2, &t, lam);
        fnv(&mut h, &r.gamma);
        for f in &r.f_aligned {
            fnv(&mut h, f);
        }
        fnv(&mut h, &[r.distance]);
        let set: Vec<FdCurveSet> = (0..4).map(|i| curve_2d(&t, 0.03 * i as f64)).collect();
        let k = karcher_mean_nd(&set, &t, 4, 1e-4, lam).unwrap();
        for f in &k.mean {
            fnv(&mut h, f);
        }
        fnv(&mut h, k.gammas.as_slice());
        fnv(&mut h, &[k.n_iter as f64]);
        out.push(h);

        // Landmark-constrained.
        let mut h = FNV0;
        let f1 = d.row(0);
        let f2 = d.row(3);
        let r = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.3, 0.45), (0.62, 0.7)], lam);
        fnv(&mut h, &r.gamma);
        fnv(&mut h, &r.f_aligned);
        fnv(&mut h, &[r.distance]);
        out.push(h);

        // Robust Karcher (median + trimmed) and align_to_target.
        let mut h = FNV0;
        let mut rc = RobustKarcherConfig::default();
        rc.lambda = lam;
        rc.max_iter = 4;
        let med = karcher_median(&d, &t, &rc).unwrap();
        fnv(&mut h, &med.mean);
        fnv(&mut h, med.gammas.as_slice());
        fnv(&mut h, &med.weights);
        let tr = robust_karcher_mean(&d, &t, &rc).unwrap();
        fnv(&mut h, &tr.mean);
        fnv(&mut h, tr.gammas.as_slice());
        fnv(&mut h, &tr.weights);
        let at = align_to_target(&d, &d.row(2), &t, lam);
        fnv(&mut h, at.gammas.as_slice());
        fnv(&mut h, at.aligned_data.as_slice());
        fnv(&mut h, &at.distances);
        out.push(h);

        // Closed Karcher.
        let mut h = FNV0;
        let (pd, pt) = periodic_data(4, 40);
        let kc = karcher_mean_closed(&pd, &pt, 3, 1e-4, lam).unwrap();
        fnv(&mut h, &kc.mean);
        fnv(&mut h, kc.gammas.as_slice());
        fnv(
            &mut h,
            &kc.rotations.iter().map(|&r| r as f64).collect::<Vec<_>>(),
        );
        out.push(h);

        // RBFGS defaults: 1-D pair (L2Gam / L2Psi), Karcher, curve shape.
        let mut h = FNV0;
        for pen in [WarpObjectivePenalty::L2Gam, WarpObjectivePenalty::L2Psi] {
            let mut c = rbfgs_cfg();
            c.lambda = lam;
            c.penalty = pen;
            let r = elastic_align_pair_with_config(&d.row(0), &d.row(4), &t, &c).unwrap();
            fnv(&mut h, &r.gamma);
            fnv(&mut h, &[r.distance]);
        }
        let mut c = rbfgs_cfg();
        c.lambda = lam;
        let km = karcher_mean_with_config(&d, &t, 3, 1e-4, &c).unwrap();
        fnv(&mut h, &km.mean);
        fnv(&mut h, km.gammas.as_slice());
        let (b1, b2) = curve_shape_pair(41);
        let mut sc = CurveShapeConfig::default();
        sc.warp_optim.method = WarpOptimMethod::Rbfgs;
        sc.lambda = lam;
        let a = curve_shape_align_pair(&b1, &b2, &sc).unwrap();
        fnv(&mut h, &a.gamma);
        fnv(&mut h, &[a.distance]);
        out.push(h);
    }
    out
}

/// FNV-1a hashes recorded on commit 79430c46 (before any #67 edit).
const PRE_CHANGE_HASHES: [u64; 10] = [
    0xb2b5_d73c_5a93_19c0,
    0xf8b9_1642_c731_a0ab,
    0xc1d1_4356_972e_cae3,
    0xbf25_c0f7_fb19_2fe0,
    0x3be9_b941_56dc_f5e4,
    0x814d_ba49_eb12_23eb,
    0xa72d_3c1e_4f55_5a58,
    0xbb54_6d66_da38_a3d1,
    0xe4ab_f1f3_4a31_e5a6,
    0x931e_3f4f_2294_5007,
];

#[test]
fn touched_entry_points_bit_identical_to_pre_change() {
    let h = pre_change_hashes();
    eprintln!("pre-change hashes: {h:#x?}");
    assert_eq!(h, PRE_CHANGE_HASHES);
}

// ─── Shared helpers ─────────────────────────────────────────────────────────

use crate::alignment::rbfgs::{rbfgs_data_cost, rbfgs_objective};
use crate::alignment::srsf::srsf_single;
use crate::alignment::{
    align_to_target_with_config, elastic_align_pair_banded,
    elastic_align_pair_constrained_with_config, elastic_align_pair_nd_with_config,
    elastic_distance_nd_with_config, karcher_mean_banded, karcher_mean_closed_with_config,
    karcher_mean_nd_with_config, karcher_median_with_config, robust_karcher_mean_with_config,
    AlignmentResultNd,
};

fn maxabs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn assert_valid_warp(g: &[f64], t: &[f64]) {
    assert_eq!(g[0], t[0]);
    assert_eq!(g[g.len() - 1], t[t.len() - 1]);
    assert!(g.windows(2).all(|w| w[1] >= w[0]), "warp not monotone");
}

/// γ0(t) = t + 0.4·t(1 − t) and its analytic inverse on [0, 1].
fn gam0(t: f64) -> f64 {
    t + 0.4 * t * (1.0 - t)
}
fn gam0_inv(s: f64) -> f64 {
    (1.4 - (1.96 - 1.6 * s).max(0.0).sqrt()) / 0.8
}

fn bumps(s: f64) -> f64 {
    (-(s - 0.3).powi(2) / 0.006).exp() + 0.7 * (-(s - 0.68).powi(2) / 0.01).exp() + 0.1 * s
}

/// `f1 = bumps`, `f2 = f1∘γ0`: the warp aligning `f2` to `f1` is `γ0⁻¹`.
fn warped_pair(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = grid(m);
    let f1: Vec<f64> = t.iter().map(|&s| bumps(s)).collect();
    let f2: Vec<f64> = t.iter().map(|&s| bumps(gam0(s))).collect();
    let ginv: Vec<f64> = t.iter().map(|&s| gam0_inv(s)).collect();
    (t, f1, f2, ginv)
}

fn nd_curve(t: &[f64], g: &dyn Fn(f64) -> f64) -> FdCurveSet {
    let m = t.len();
    let x: Vec<f64> = t.iter().map(|&s| (6.0 * g(s)).sin()).collect();
    let y: Vec<f64> = t
        .iter()
        .map(|&s| (4.0 * g(s)).cos() + 0.5 * bumps(g(s)))
        .collect();
    FdCurveSet {
        dims: vec![
            FdMatrix::from_slice(&x, 1, m).unwrap(),
            FdMatrix::from_slice(&y, 1, m).unwrap(),
        ],
    }
}

fn nd_srsfs(c: &FdCurveSet, t: &[f64]) -> Vec<Vec<f64>> {
    c.dims.iter().map(|d| srsf_single(&d.row(0), t)).collect()
}

fn nd_objective(a: &FdCurveSet, b: &FdCurveSet, t: &[f64], g: &[f64]) -> f64 {
    let (q1, q2) = (nd_srsfs(a, t), nd_srsfs(b, t));
    let c1: Vec<&[f64]> = q1.iter().map(Vec::as_slice).collect();
    let c2: Vec<&[f64]> = q2.iter().map(Vec::as_slice).collect();
    rbfgs_objective(&c1, &c2, t, g, 0.0, WarpObjectivePenalty::L2Gam, 3)
}

// ─── N-d shared-warp alignment ──────────────────────────────────────────────

#[test]
fn nd_default_config_equals_classic() {
    let t = grid(50);
    let c1 = curve_2d(&t, 0.0);
    let c2 = curve_2d(&t, 0.06);
    for lam in [0.0, 0.05] {
        let mut cfg = WarpOptimConfig::default();
        cfg.lambda = lam;
        let a = elastic_align_pair_nd_with_config(&c1, &c2, &t, &cfg).unwrap();
        assert_eq!(a, elastic_align_pair_nd(&c1, &c2, &t, lam));
        assert_eq!(
            elastic_distance_nd_with_config(&c1, &c2, &t, &cfg).unwrap(),
            a.distance
        );
        let set: Vec<FdCurveSet> = (0..4).map(|i| curve_2d(&t, 0.03 * i as f64)).collect();
        assert_eq!(
            karcher_mean_nd_with_config(&set, &t, 4, 1e-4, &cfg).unwrap(),
            karcher_mean_nd(&set, &t, 4, 1e-4, lam).unwrap()
        );
    }
}

#[test]
fn nd_rbfgs_lowers_objective_and_recovers_known_warp() {
    let m = 81;
    let t = grid(m);
    let f1 = nd_curve(&t, &|s| s);
    let f2 = nd_curve(&t, &gam0);
    let ginv: Vec<f64> = t.iter().map(|&s| gam0_inv(s)).collect();
    let dp = elastic_align_pair_nd_with_config(&f1, &f2, &t, &WarpOptimConfig::default()).unwrap();
    let rb = elastic_align_pair_nd_with_config(&f1, &f2, &t, &rbfgs_cfg()).unwrap();
    let mut dp2 = WarpOptimConfig::default();
    dp2.nbhd_dim = 10;
    dp2.dp_cost = DpEdgeCost::PiecewiseLinear;
    let d2 = elastic_align_pair_nd_with_config(&f1, &f2, &t, &dp2).unwrap();
    for r in [&dp, &rb, &d2] {
        assert_valid_warp(&r.gamma, &t);
    }
    let (o_dp, o_rb) = (
        nd_objective(&f1, &f2, &t, &dp.gamma),
        nd_objective(&f1, &f2, &t, &rb.gamma),
    );
    let (e_dp, e_rb) = (maxabs(&dp.gamma, &ginv), maxabs(&rb.gamma, &ginv));
    eprintln!("nd: obj dp {o_dp:.3e} rb {o_rb:.3e}; |γ−γ*| dp {e_dp:.4} rb {e_rb:.4}; dist dp {} rb {} dp2 {}", dp.distance, rb.distance, d2.distance);
    assert!(o_rb <= o_dp, "RBFGS objective {o_rb} > DP {o_dp}");
    assert!(e_rb < e_dp && e_rb < 2e-3, "recovery dp {e_dp} rb {e_rb}");
    assert!(rb.distance < dp.distance);
    assert_ne!(d2.gamma, dp.gamma, "DP options must reach the R^d DP");
}

#[test]
fn nd_karcher_rbfgs_and_band() {
    let t = grid(50);
    let set: Vec<FdCurveSet> = (0..4).map(|i| curve_2d(&t, 0.03 * i as f64)).collect();
    let km = karcher_mean_nd_with_config(&set, &t, 4, 1e-4, &rbfgs_cfg()).unwrap();
    for i in 0..4 {
        assert_valid_warp(&km.gammas.row(i), &t);
    }
    // Band: DP and RBFGS warps stay in the corridor |γ(t) − t| ≤ R·h.
    let f1 = nd_curve(&t, &|s| s);
    let f2 = nd_curve(&t, &gam0);
    let bound = super::super::band_radius(0.05, 50).unwrap() as f64 / 49.0 + 1e-12;
    for method in [WarpOptimMethod::Dp, WarpOptimMethod::Rbfgs] {
        let mut cfg = WarpOptimConfig::default();
        cfg.method = method;
        let free = elastic_align_pair_nd_with_config(&f1, &f2, &t, &cfg).unwrap();
        cfg.band_frac = 0.05;
        let r: AlignmentResultNd = elastic_align_pair_nd_with_config(&f1, &f2, &t, &cfg).unwrap();
        assert_valid_warp(&r.gamma, &t);
        let dev = maxabs(&r.gamma, &t);
        assert!(maxabs(&free.gamma, &t) > bound, "band must be active");
        assert!(dev <= bound, "{method:?}: |γ − t| = {dev} > {bound}");
    }
}

// ─── Landmark-constrained alignment ─────────────────────────────────────────

/// Landmark positions that are exact grid points of `grid(m)` for `m = 101`:
/// γ0⁻¹(0.5) = 0.4 … use the warp's own values at grid points.
fn landmark_case() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
    let (t, f1, f2, ginv) = warped_pair(101);
    // γ0⁻¹(0.4) ≈ 0.3142…, γ0⁻¹(0.8) ≈ 0.6838…: snap to the grid as the
    // constrained aligner does; the true warp nearly passes through them.
    let lm = vec![(0.4, gam0_inv(0.4)), (0.8, gam0_inv(0.8))];
    (t, f1, f2, ginv, lm)
}

fn snap(t: &[f64], x: f64) -> usize {
    (0..t.len())
        .min_by(|&a, &b| (t[a] - x).abs().partial_cmp(&(t[b] - x).abs()).unwrap())
        .unwrap()
}

#[test]
fn constrained_default_config_equals_classic() {
    let (d, t) = probe_data(5, 50);
    let (f1, f2) = (d.row(0), d.row(3));
    let lm = [(0.3, 0.45), (0.62, 0.7)];
    for lam in [0.0, 0.05] {
        let mut cfg = WarpOptimConfig::default();
        cfg.lambda = lam;
        let a = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &cfg).unwrap();
        assert_eq!(a, elastic_align_pair_constrained(&f1, &f2, &t, &lm, lam));
        let e = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &[], &cfg).unwrap();
        assert_eq!(e, elastic_align_pair_constrained(&f1, &f2, &t, &[], lam));
    }
}

#[test]
fn constrained_landmarks_exact_for_all_optimisers() {
    let (t, f1, f2, _, lm) = landmark_case();
    let mut dp2 = WarpOptimConfig::default();
    dp2.nbhd_dim = 10;
    dp2.dp_cost = DpEdgeCost::PiecewiseLinear;
    dp2.dp_scl = 2;
    let mut psi = rbfgs_cfg();
    psi.penalty = WarpObjectivePenalty::L2Psi;
    psi.lambda = 0.05;
    let mut rough = rbfgs_cfg();
    rough.penalty = WarpObjectivePenalty::Roughness;
    rough.lambda = 1e-4;
    let mut gam = rbfgs_cfg();
    gam.lambda = 0.05;
    for cfg in [dp2, rbfgs_cfg(), psi, rough, gam] {
        let r = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &cfg).unwrap();
        assert_valid_warp(&r.gamma, &t);
        assert_eq!(r.enforced_landmarks.len(), 2);
        for &(a, b) in &lm {
            let (c, rr) = (snap(&t, a), snap(&t, b));
            assert_eq!(
                r.gamma[c], t[rr],
                "{:?}: γ({}) != {}",
                cfg.method, t[c], t[rr]
            );
        }
    }
}

#[test]
fn constrained_rbfgs_lowers_objective_and_recovers_known_warp() {
    let (t, f1, f2, ginv, lm) = landmark_case();
    let (q1, q2) = (srsf_single(&f1, &t), srsf_single(&f2, &t));
    let dp =
        elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &WarpOptimConfig::default())
            .unwrap();
    let rb = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &rbfgs_cfg()).unwrap();
    let (o_dp, o_rb) = (
        rbfgs_data_cost(&q1, &q2, &t, &dp.gamma),
        rbfgs_data_cost(&q1, &q2, &t, &rb.gamma),
    );
    let (e_dp, e_rb) = (maxabs(&dp.gamma, &ginv), maxabs(&rb.gamma, &ginv));
    eprintln!(
        "lm: obj dp {o_dp:.3e} rb {o_rb:.3e}; |γ−γ*| dp {e_dp:.4} rb {e_rb:.4}; dist {} {}",
        dp.distance, rb.distance
    );
    assert!(o_rb <= o_dp, "RBFGS objective {o_rb} > DP {o_dp}");
    assert!(e_rb < e_dp && e_rb < 5e-3, "recovery dp {e_dp} rb {e_rb}");
    // Penalised: the global objective (λ, L2Gam) of the RBFGS warp is below
    // the constrained DP's (seeded from it, segment objectives restricted from
    // the global one).
    for (pen, lam) in [
        (WarpObjectivePenalty::L2Gam, 0.05),
        (WarpObjectivePenalty::L2Psi, 0.2),
    ] {
        let mut c = rbfgs_cfg();
        c.penalty = pen;
        c.lambda = lam;
        let r = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &c).unwrap();
        let mut cd = WarpOptimConfig::default();
        cd.penalty = pen;
        cd.lambda = lam;
        let d = elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &cd).unwrap();
        let obj = |g: &[f64]| rbfgs_objective(&[&q1], &[&q2], &t, g, lam, pen, 3);
        eprintln!(
            "lm {pen:?}: rb {:.4e} dp {:.4e}",
            obj(&r.gamma),
            obj(&d.gamma)
        );
        assert!(obj(&r.gamma) <= obj(&d.gamma));
    }
}

/// When the unconstrained optimum already passes through the landmark (a
/// mirror-symmetric pair: γ(1/2) = 1/2), the constrained RBFGS warp equals the
/// unconstrained one for every λ — the segment objectives are the global one
/// restricted (the λ normalisation uses the whole curves' norms). Measured:
/// gap ≈ 1e-7 (λ = 0.3) / 1e-10 (λ = 2), while scaling λ by 4 or ¼ moves the
/// warp by ≈ 1e-2.
#[test]
fn constrained_rbfgs_matches_unconstrained_when_landmark_is_optimal() {
    let m = 101;
    let t = grid(m);
    let sym = |s: f64| (-(s - 0.25).powi(2) / 0.004).exp() + (-(s - 0.75).powi(2) / 0.004).exp();
    // f2 = f1∘γ0 with a mirror-symmetric γ0 (γ0(1−t) = 1 − γ0(t)).
    let g = |s: f64| s + 0.05 * (2.0 * std::f64::consts::PI * s).sin();
    // No flat stretches (q ≠ 0 almost everywhere), so the optimum is unique.
    let tau = 2.0 * std::f64::consts::PI;
    let f1: Vec<f64> = t.iter().map(|&s| sym(s) + 0.4 * (tau * s).cos()).collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&s| sym(g(s)) + 0.4 * (tau * g(s)).cos())
        .collect();
    for lam in [0.0, 0.3, 2.0] {
        let mut c = rbfgs_cfg();
        c.lambda = lam;
        c.penalty = WarpObjectivePenalty::L2Psi;
        let free = elastic_align_pair_with_config(&f1, &f2, &t, &c).unwrap();
        assert!(
            (free.gamma[50] - 0.5).abs() < 2e-3,
            "free γ(1/2) = {}",
            free.gamma[50]
        );
        let con =
            elastic_align_pair_constrained_with_config(&f1, &f2, &t, &[(0.5, 0.5)], &c).unwrap();
        let diff = maxabs(&free.gamma, &con.gamma);
        // The same λ with a mis-normalised segment weight (×4 / ×¼) moves the
        // free warp by far more than the constrained/free gap.
        let mut c4 = c.clone();
        c4.lambda = lam * 4.0;
        let f4 = elastic_align_pair_with_config(&f1, &f2, &t, &c4).unwrap();
        let mut c25 = c.clone();
        c25.lambda = lam / 4.0;
        let f25 = elastic_align_pair_with_config(&f1, &f2, &t, &c25).unwrap();
        eprintln!(
            "sym lam={lam}: |free − con| = {diff:.2e}, dev = {:.3}, |free − free(4λ)| {:.2e} |free − free(λ/4)| {:.2e}",
            maxabs(&free.gamma, &t), maxabs(&free.gamma, &f4.gamma), maxabs(&free.gamma, &f25.gamma)
        );
        if lam == 0.0 {
            // Unpenalised: both stop at the solver tolerance of a flat optimum.
            assert!(diff < 5e-3, "lam=0: {diff}");
        } else {
            assert!(diff < 1e-5, "lam={lam}: {diff}");
            let sens = maxabs(&free.gamma, &f4.gamma).min(maxabs(&free.gamma, &f25.gamma));
            assert!(
                diff < 1e-3 * sens,
                "lam={lam}: gap {diff} vs λ-sensitivity {sens}"
            );
        }
    }
}

#[test]
fn constrained_with_config_rejects_unsupported_options() {
    let (t, f1, f2, _, lm) = landmark_case();
    let mut c = rbfgs_cfg();
    c.penalty = WarpObjectivePenalty::Geodesic;
    c.lambda = 0.1;
    assert!(elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &c).is_err());
    let mut c = WarpOptimConfig::default();
    c.band_frac = 0.1;
    assert!(elastic_align_pair_constrained_with_config(&f1, &f2, &t, &lm, &c).is_err());
    assert!(elastic_align_pair_constrained_with_config(
        &f1,
        &f2,
        &t,
        &[(f64::NAN, 0.3)],
        &WarpOptimConfig::default()
    )
    .is_err());
    assert!(elastic_align_pair_constrained_with_config(&f1, &f2[..50], &t, &lm, &c).is_err());
}

// ─── Robust Karcher, align_to_target, closed Karcher ────────────────────────

/// Pure phase variation of one template (`n` curves `bumps∘γ_i`).
fn phase_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let a = 0.3 * (i as f64 / (n - 1) as f64 - 0.5);
        for j in 0..m {
            let s = t[j];
            d[(i, j)] = bumps(s + a * s * (1.0 - s));
        }
    }
    (d, t)
}

#[test]
fn robust_and_target_default_config_equal_classic() {
    let (d, t) = probe_data(5, 50);
    for lam in [0.0, 0.05] {
        let mut rc = RobustKarcherConfig::default();
        rc.lambda = lam;
        rc.max_iter = 4;
        let mut w = WarpOptimConfig::default();
        w.lambda = lam;
        assert_eq!(
            karcher_median_with_config(&d, &t, &rc, &w).unwrap(),
            karcher_median(&d, &t, &rc).unwrap()
        );
        assert_eq!(
            robust_karcher_mean_with_config(&d, &t, &rc, &w).unwrap(),
            robust_karcher_mean(&d, &t, &rc).unwrap()
        );
        // config.lambda = 0 defers to warp.lambda.
        let mut rc0 = rc.clone();
        rc0.lambda = 0.0;
        assert_eq!(
            karcher_median_with_config(&d, &t, &rc0, &w).unwrap(),
            karcher_median(&d, &t, &rc).unwrap()
        );
        assert_eq!(
            align_to_target_with_config(&d, &d.row(2), &t, &w).unwrap(),
            align_to_target(&d, &d.row(2), &t, lam)
        );
        let (pd, pt) = periodic_data(4, 40);
        assert_eq!(
            karcher_mean_closed_with_config(&pd, &pt, 3, 1e-4, &w).unwrap(),
            karcher_mean_closed(&pd, &pt, 3, 1e-4, lam).unwrap()
        );
    }
    let mut rc = RobustKarcherConfig::default();
    rc.lambda = 0.1;
    let mut w = WarpOptimConfig::default();
    w.lambda = 0.2;
    assert!(karcher_median_with_config(&d, &t, &rc, &w).is_err());
    assert!(robust_karcher_mean_with_config(&d, &t, &rc, &w).is_err());
}

#[test]
fn align_to_target_rbfgs_lowers_every_objective() {
    let (d, t) = phase_data(5, 81);
    let target = d.row(2);
    let qt = srsf_single(&target, &t);
    let dp = align_to_target_with_config(&d, &target, &t, &WarpOptimConfig::default()).unwrap();
    let rb = align_to_target_with_config(&d, &target, &t, &rbfgs_cfg()).unwrap();
    for i in 0..5 {
        let qi = srsf_single(&d.row(i), &t);
        assert_valid_warp(&rb.gammas.row(i), &t);
        let (o_dp, o_rb) = (
            rbfgs_data_cost(&qt, &qi, &t, &dp.gammas.row(i)),
            rbfgs_data_cost(&qt, &qi, &t, &rb.gammas.row(i)),
        );
        assert!(o_rb <= o_dp + 1e-12, "curve {i}: rb {o_rb} > dp {o_dp}");
    }
    let (sd, sr): (f64, f64) = (dp.distances.iter().sum(), rb.distances.iter().sum());
    assert!(sr < sd, "Σ distances rb {sr} vs dp {sd}");
}

#[test]
fn robust_karcher_rbfgs_variants() {
    let (d, t) = phase_data(6, 61);
    let mut rc = RobustKarcherConfig::default();
    rc.max_iter = 4;
    rc.trim_fraction = 0.2;
    for (name, w) in [("dp", WarpOptimConfig::default()), ("rbfgs", rbfgs_cfg())] {
        let med = karcher_median_with_config(&d, &t, &rc, &w).unwrap();
        let tr = robust_karcher_mean_with_config(&d, &t, &rc, &w).unwrap();
        for r in [&med, &tr] {
            for i in 0..6 {
                assert_valid_warp(&r.gammas.row(i), &t);
            }
        }
        assert!((med.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(tr.weights.iter().filter(|&&x| x == 0.0).count(), 2);
        // Weiszfeld objective: Σ elastic distances to the median under the
        // same optimiser; amplitude residual of the aligned curves.
        let sum_d: f64 = (0..6)
            .map(|i| {
                crate::alignment::elastic_distance_with_config(&med.mean, &d.row(i), &t, &w)
                    .unwrap()
            })
            .sum();
        eprintln!("{name}: Σ d(median, f_i) = {sum_d:.4}");
    }
    // Pure phase variation: RBFGS reaches a lower Weiszfeld objective.
    let obj = |w: &WarpOptimConfig| -> f64 {
        let med = karcher_median_with_config(&d, &t, &rc, w).unwrap();
        (0..6)
            .map(|i| {
                crate::alignment::elastic_distance_with_config(&med.mean, &d.row(i), &t, w).unwrap()
            })
            .sum()
    };
    assert!(obj(&rbfgs_cfg()) < obj(&WarpOptimConfig::default()));
}

#[test]
fn closed_karcher_rbfgs_valid_and_lower_residual() {
    let (pd, pt) = periodic_data(4, 61);
    let dp =
        karcher_mean_closed_with_config(&pd, &pt, 3, 1e-4, &WarpOptimConfig::default()).unwrap();
    let rb = karcher_mean_closed_with_config(&pd, &pt, 3, 1e-4, &rbfgs_cfg()).unwrap();
    for i in 0..4 {
        assert_valid_warp(&rb.gammas.row(i), &pt);
    }
    // Rotated copies of one periodic curve: the aligned curves agree with
    // each other; RBFGS leaves a smaller spread than the slope-quantised DP.
    let spread = |r: &crate::alignment::ClosedKarcherMeanResult| -> f64 {
        let mut s = 0.0;
        for i in 1..4 {
            s += maxabs(&r.aligned_data.row(i), &r.aligned_data.row(0));
        }
        s
    };
    eprintln!("closed spread dp {} rb {}", spread(&dp), spread(&rb));
    assert!(spread(&rb) <= spread(&dp));
}

// ─── Sakoe–Chiba band under every optimiser ─────────────────────────────────

#[test]
fn band_frac_dp_equals_classic_banded_functions() {
    let (d, t) = probe_data(5, 60);
    let (f1, f2) = (d.row(0), d.row(4));
    for lam in [0.0, 0.05] {
        let mut cfg = WarpOptimConfig::default();
        cfg.lambda = lam;
        cfg.band_frac = 0.1;
        assert_eq!(
            elastic_align_pair_with_config(&f1, &f2, &t, &cfg).unwrap(),
            elastic_align_pair_banded(&f1, &f2, &t, lam, 0.1)
        );
        assert_eq!(
            karcher_mean_with_config(&d, &t, 12, 1e-4, &cfg).unwrap(),
            karcher_mean_banded(&d, &t, 12, 1e-4, lam, 0.1)
        );
    }
    let mut cfg = WarpOptimConfig::default();
    cfg.band_frac = -0.1;
    assert!(cfg.validate().is_err());
    cfg.band_frac = f64::NAN;
    assert!(cfg.validate().is_err());
}

#[test]
fn rbfgs_band_is_respected_and_never_worse_than_banded_dp() {
    let (t, f1, f2, ginv) = warped_pair(81);
    let (q1, q2) = (srsf_single(&f1, &t), srsf_single(&f2, &t));
    assert!(maxabs(&ginv, &t) > 0.09);
    let unbanded = elastic_align_pair_with_config(&f1, &f2, &t, &rbfgs_cfg()).unwrap();
    for frac in [0.02, 0.04, 0.08] {
        let b = super::super::band_radius(frac, 81).unwrap() as f64 / 80.0;
        let mut rc = rbfgs_cfg();
        rc.band_frac = frac;
        let mut dc = WarpOptimConfig::default();
        dc.band_frac = frac;
        let rb = elastic_align_pair_with_config(&f1, &f2, &t, &rc).unwrap();
        let dp = elastic_align_pair_with_config(&f1, &f2, &t, &dc).unwrap();
        assert_valid_warp(&rb.gamma, &t);
        let (dev_rb, dev_dp) = (maxabs(&rb.gamma, &t), maxabs(&dp.gamma, &t));
        let (o_rb, o_dp) = (
            rbfgs_data_cost(&q1, &q2, &t, &rb.gamma),
            rbfgs_data_cost(&q1, &q2, &t, &dp.gamma),
        );
        eprintln!("band {frac}: b {b:.4} dev rb {dev_rb:.4} dp {dev_dp:.4}; obj rb {o_rb:.3e} dp {o_dp:.3e}");
        assert!(dev_rb <= b + 1e-12 && dev_dp <= b + 1e-12);
        assert!(
            o_rb <= o_dp * (1.0 + 1e-9),
            "band {frac}: rb {o_rb} > dp {o_dp}"
        );
    }
    // A band wider than the optimum only changes the start (input-grid DP
    // instead of the coarse DP): both reach the same optimum.
    let mut rc = rbfgs_cfg();
    rc.band_frac = 0.5;
    let wide = elastic_align_pair_with_config(&f1, &f2, &t, &rc).unwrap();
    let gap = maxabs(&wide.gamma, &unbanded.gamma);
    eprintln!(
        "wide band vs unbanded: {gap:.2e}; recovery {:.2e}",
        maxabs(&wide.gamma, &ginv)
    );
    // (Both stop at fdasrsf's 30-iteration cap; the gap is solver tolerance.)
    assert!(gap < 5e-3, "{gap}");
    assert!(maxabs(&wide.gamma, &ginv) < 5e-3);
}

// ─── RBFGS coarse-DP seed honours the DP options ────────────────────────────

/// With `max_iter = 0` RBFGS returns (a re-integration of) its DP seed. On a
/// uniform grid of ≤ 51 points the coarse grid is the input grid, so the seed
/// must be the DP warp with the configured options.
#[test]
fn rbfgs_seed_uses_dp_options() {
    let (t, f1, f2, _) = warped_pair(41);
    let (q1, q2) = (srsf_single(&f1, &t), srsf_single(&f2, &t));
    let mut opts = WarpOptimConfig::default();
    opts.nbhd_dim = 10;
    opts.dp_cost = DpEdgeCost::PiecewiseLinear;
    opts.dp_scl = 2;
    let dp_opt = optimal_warp(&q1, &q2, &t, &opts, None);
    let dp_def = optimal_warp(&q1, &q2, &t, &WarpOptimConfig::default(), None);
    assert!(
        maxabs(&dp_opt, &dp_def) > 5e-3,
        "options must change the DP warp"
    );
    let mut r_opt = opts.clone();
    r_opt.method = WarpOptimMethod::Rbfgs;
    r_opt.max_iter = 0;
    let mut r_def = rbfgs_cfg();
    r_def.max_iter = 0;
    let s_opt = optimal_warp(&q1, &q2, &t, &r_opt, None);
    let s_def = optimal_warp(&q1, &q2, &t, &r_def, None);
    eprintln!(
        "seed: |s_opt − dp_opt| {:.2e} |s_opt − dp_def| {:.2e} |s_def − dp_def| {:.2e}",
        maxabs(&s_opt, &dp_opt),
        maxabs(&s_opt, &dp_def),
        maxabs(&s_def, &dp_def)
    );
    assert!(maxabs(&s_opt, &dp_opt) < 0.5 * maxabs(&s_opt, &dp_def));
    assert!(maxabs(&s_def, &dp_def) < 0.5 * maxabs(&s_def, &dp_opt));
    // The full solver from the finer seed is a valid warp with an objective
    // no worse than that DP warp.
    r_opt.max_iter = 30;
    let g = optimal_warp(&q1, &q2, &t, &r_opt, None);
    assert_valid_warp(&g, &t);
    assert!(rbfgs_data_cost(&q1, &q2, &t, &g) <= rbfgs_data_cost(&q1, &q2, &t, &dp_opt));
}

// ─── λ scale: DP ↔ RBFGS conversion ─────────────────────────────────────────

/// DP objective of the grid path `path` (rows = q2, cols = q1), exactly the
/// sum of the classic DP edge costs (unit-Euclidean-norm SRSFs, `argvals`
/// units) with the `L2Gam` / `L2Psi` edge penalty.
fn dp_path_objective(
    q1: &[f64],
    q2: &[f64],
    t: &[f64],
    path: &[(usize, usize)],
    lam: f64,
    pen: WarpObjectivePenalty,
) -> f64 {
    let e = |q: &[f64]| q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q1n: Vec<f64> = q1.iter().map(|v| v / e(q1)).collect();
    let q2n: Vec<f64> = q2.iter().map(|v| v / e(q2)).collect();
    path.windows(2)
        .map(|w| {
            let ((sr, sc), (tr, tc)) = (w[0], w[1]);
            let p = match pen {
                WarpObjectivePenalty::L2Psi => {
                    super::super::dp_l2psi_penalty(t, sc, tc, sr, tr, lam)
                }
                _ => super::super::dp_lambda_penalty(t, sc, tc, sr, tr, lam),
            };
            super::super::dp_edge_weight(&q1n, &q2n, t, sc, tc, sr, tr) + p
        })
        .sum()
}

/// `a·b` and `(a − b)²` of the identity in [`WarpLambdaScale`]:
/// `a = n1/E1`, `b = n2/E2` (`n` the trapezoid L² norm on the `scl`-refined
/// grid of [0, 1], `E` the Euclidean norm on the input grid).
fn scale_ab(q1: &[f64], q2: &[f64], scl: usize) -> (f64, f64) {
    let ab: Vec<f64> = [q1, q2]
        .iter()
        .map(|q| {
            let f = super::super::refine_srsf_linear(q, scl);
            let h = 1.0 / (f.len() - 1) as f64;
            let n2: f64 = f
                .iter()
                .enumerate()
                .map(|(k, v)| if k == 0 || k == f.len() - 1 { 0.5 } else { 1.0 } * h * v * v)
                .sum();
            n2.sqrt() / q.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect();
    (ab[0] * ab[1], (ab[0] - ab[1]).powi(2))
}

#[test]
fn lambda_factor_tends_to_one() {
    let mut last = f64::INFINITY;
    for m in [21, 81, 321] {
        let (t, f1, f2, _) = warped_pair(m);
        let (q1, q2) = (srsf_single(&f1, &t), srsf_single(&f2, &t));
        let phi = WarpOptimConfig::default().rbfgs_lambda_factor(&q1, &q2);
        eprintln!("m={m}: φ = {phi:.6}");
        assert!((phi - 1.0).abs() < last);
        last = (phi - 1.0).abs();
    }
    assert!(last < 0.01);
}

/// The identity of [`WarpLambdaScale`]: on a fixed warp, with `λ_rb = φ·λ_dp`,
/// `(E_dp(γ) − (a − b)²)/(a·b) = E_rb(γ)` for every λ (here `S = 1`). The
/// residual is the discretisation of the two objectives (piecewise-constant
/// DP edges vs. trapezoid/spline RBFGS, kinks of the piecewise-linear path).
/// The penalty part alone (`ΔE` from λ = 0) isolates "the same λ means the
/// same strength". Measured: m = 241 data ratio 1.0002, penalty ratio
/// ≈ 1.003; a wrong scale (`λ_rb = (m − 1)·λ`) gives ≈ 0.004. At m = 25
/// (`φ ≈ 1.09`) the converted penalty ratio is 1.03 vs 1.13 without `φ`.
#[test]
fn lambda_conversion_makes_objectives_agree_on_a_fixed_warp() {
    for m in [25usize, 241] {
        let (t, f1, f2, _) = warped_pair(m);
        let (q1, q2) = (srsf_single(&f1, &t), srsf_single(&f2, &t));
        // A fixed DP path with slopes 2, 1/2, 2: moves (2,1)×k, (1,2)×2k, (2,1)×k.
        let k = (m - 1) / 6;
        let mut path = vec![(0usize, 0usize)];
        for (dr, dc, n) in [(2, 1, k), (1, 2, 2 * k), (2, 1, k)] {
            for _ in 0..n {
                let &(r, c) = path.last().unwrap();
                path.push((r + dr, c + dc));
            }
        }
        assert_eq!(*path.last().unwrap(), (m - 1, m - 1));
        let gamma = super::super::dp_path_to_gamma(&path, &t);
        let cfg = WarpOptimConfig::default();
        let phi = cfg.rbfgs_lambda_factor(&q1, &q2);
        let (ab, amb) = scale_ab(&q1, &q2, cfg.scl);
        for pen in [WarpObjectivePenalty::L2Gam, WarpObjectivePenalty::L2Psi] {
            let ratio = |lam_dp: f64, lam_rb: f64| {
                let e_dp = dp_path_objective(&q1, &q2, &t, &path, lam_dp, pen) - amb;
                let e_rb = rbfgs_objective(&[&q1], &[&q2], &t, &gamma, lam_rb, pen, cfg.scl);
                e_dp / (ab * e_rb)
            };
            let r0 = ratio(0.0, 0.0);
            // Penalty part alone: ΔE_dp / (a·b·ΔE_rb) — "same λ, same strength".
            let pen_ratio = |lam_dp: f64, lam_rb: f64| {
                let d_dp = dp_path_objective(&q1, &q2, &t, &path, lam_dp, pen)
                    - dp_path_objective(&q1, &q2, &t, &path, 0.0, pen);
                let d_rb = rbfgs_objective(&[&q1], &[&q2], &t, &gamma, lam_rb, pen, cfg.scl)
                    - rbfgs_objective(&[&q1], &[&q2], &t, &gamma, 0.0, pen, cfg.scl);
                d_dp / (ab * d_rb)
            };
            let (conv, native) = (pen_ratio(1.0, phi), pen_ratio(1.0, 1.0));
            let wrong = pen_ratio(1.0, (m - 1) as f64);
            eprintln!(
                "m={m} {pen:?}: φ {phi:.4} data ratio {r0:.5}; penalty ratio conv {conv:.5} native {native:.5} wrong {wrong:.4}; total λ=5 {:.5}",
                ratio(5.0, 5.0 * phi)
            );
            // Exactly linear in λ on both sides.
            assert!((pen_ratio(0.2, 0.2 * phi) - conv).abs() < 1e-9);
            // Residual: central differences of the kinked path in the RBFGS
            // penalty (3 kinks; ≈ 3 % of 72 fine intervals at m = 25).
            let tol = if m == 25 { 0.05 } else { 0.01 };
            assert!((conv - 1.0).abs() < tol, "m={m} {pen:?}: {conv}");
            assert!((wrong - 1.0).abs() > 0.5);
            if m == 25 {
                assert!((native - 1.0).abs() > 3.0 * (conv - 1.0).abs());
            } else {
                assert!((r0 - 1.0).abs() < 1e-3);
                assert!((ratio(5.0, 5.0 * phi) - 1.0).abs() < 0.01);
            }
        }
    }
}

#[test]
fn lambda_scale_conversion_is_applied_exactly() {
    let (t, f1, f2, _) = warped_pair(41);
    let (q1, q2) = (srsf_single(&f1, &t), srsf_single(&f2, &t));
    let phi = WarpOptimConfig::default().rbfgs_lambda_factor(&q1, &q2);
    for pen in [WarpObjectivePenalty::L2Gam, WarpObjectivePenalty::L2Psi] {
        // Fdasrsf-scale λ for DP = Native DP with λ/φ.
        let mut a = WarpOptimConfig::default();
        a.penalty = pen;
        a.lambda = 0.3;
        a.lambda_scale = WarpLambdaScale::Fdasrsf;
        let mut b = a.clone();
        b.lambda_scale = WarpLambdaScale::Native;
        b.lambda = 0.3 / phi;
        assert_eq!(
            optimal_warp(&q1, &q2, &t, &a, None),
            optimal_warp(&q1, &q2, &t, &b, None)
        );
        // Dp-scale λ for DP and Fdasrsf-scale λ for RBFGS are Native.
        a.lambda_scale = WarpLambdaScale::Dp;
        b.lambda = 0.3;
        assert_eq!(
            optimal_warp(&q1, &q2, &t, &a, None),
            optimal_warp(&q1, &q2, &t, &b, None)
        );
        // Dp-scale λ for RBFGS = Native RBFGS with φ·λ.
        a.method = WarpOptimMethod::Rbfgs;
        b.method = WarpOptimMethod::Rbfgs;
        b.lambda = 0.3 * phi;
        assert_eq!(
            optimal_warp(&q1, &q2, &t, &a, None),
            optimal_warp(&q1, &q2, &t, &b, None)
        );
    }
}

/// Curve shapes: the R^d DP works on raw SRVFs, so RBFGS's native λ is
/// `1/φ ≈ (m − 1)·‖q1‖‖q2‖` times stronger. With `lambda_scale = Dp` both
/// optimisers pull the warp towards the identity by a comparable amount.
#[test]
fn curve_shape_lambda_scale_equalises_penalty_strength() {
    let (b1, b2) = curve_shape_pair(41);
    let dev = |cfg: &CurveShapeConfig| {
        let a = curve_shape_align_pair(&b1, &b2, cfg).unwrap();
        a.gamma
            .iter()
            .enumerate()
            .map(|(j, g)| (g - j as f64 / 40.0).abs())
            .fold(0.0, f64::max)
    };
    let mut c = CurveShapeConfig::default();
    c.lambda = 0.5;
    let d_dp = dev(&c);
    c.warp_optim.method = WarpOptimMethod::Rbfgs;
    let d_native = dev(&c);
    c.warp_optim.lambda_scale = WarpLambdaScale::Dp;
    let d_conv = dev(&c);
    c.lambda = 0.0;
    let d_free = dev(&c);
    eprintln!(
        "curve dev: free {d_free:.4} dp {d_dp:.4} rbfgs native {d_native:.4} converted {d_conv:.4}"
    );
    assert!((d_conv - d_dp).abs() < 0.5 * (d_native - d_dp).abs());
    let mut bad = CurveShapeConfig::default();
    bad.warp_optim.band_frac = 0.1;
    assert!(bad.validate().is_err());
}
