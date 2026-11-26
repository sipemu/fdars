//! Bit-identity tests for exact DP pruning being on by default (ALN-CAV, GH #66).
//!
//! Every hash below is an FNV-1a hash of the `f64` bit patterns produced by the
//! entry point **before** pruning was enabled by default (recorded by running
//! this exact probe on commit 7d8cefe2, where every classic path ran the
//! unpruned DP fill). After the change each of these paths prunes (the gate
//! [`dp_prune_is_exact`](super::dp_prune_is_exact) holds for all of them except
//! the negative-λ probe), so equal hashes prove pruning is bit-identical across
//! bands, penalties, non-uniform grids, closed curves, R^d curves, planar
//! curve-shape alignment, landmark-constrained and partial matching.

#![allow(clippy::field_reassign_with_default)]

use super::optim::{DpEdgeCost, WarpObjectivePenalty, WarpOptimConfig, WarpOptimMethod};
use super::{
    coprime_nbhd, curve_geodesic, curve_shape_align_pair, dp_grid_solve_moves, dp_prune_is_exact,
    elastic_align_pair, elastic_align_pair_banded, elastic_align_pair_closed,
    elastic_align_pair_constrained, elastic_align_pair_nd, elastic_align_pair_with_config,
    elastic_cross_distance_matrix_with_config, elastic_partial_match, elastic_self_distance_matrix,
    elastic_self_distance_matrix_with_band, karcher_mean, karcher_mean_closed, karcher_mean_nd,
    karcher_mean_with_band, karcher_mean_with_config, CurveShapeConfig, DpMoves,
    PartialMatchConfig,
};
use crate::matrix::{FdCurveSet, FdMatrix};

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn nonuniform_grid(m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| (i as f64 / (m - 1) as f64).powf(1.4) * 2.0)
        .collect()
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

fn hash(parts: &[&[f64]]) -> u64 {
    let mut h = FNV0;
    for p in parts {
        fnv(&mut h, p);
    }
    h
}

/// Full-rank, non-degenerate probe data on an arbitrary grid.
fn probe_on(t: &[f64], n: usize) -> FdMatrix {
    let m = t.len();
    let (a, b) = (t[0], t[m - 1]);
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let c = 0.3 + 0.05 * i as f64;
        for j in 0..m {
            let s = (t[j] - a) / (b - a);
            d[(i, j)] = (-(s - c).powi(2) / 0.01).exp()
                + 0.5 * (-(s - c - 0.3).powi(2) / 0.02).exp()
                + 0.1 * (3.0 * s + i as f64).sin();
        }
    }
    d
}

fn periodic_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = FdMatrix::zeros(n, m);
    for i in 0..n {
        let ph = 0.07 * i as f64;
        for j in 0..m {
            let s = t[j] + ph;
            d[(i, j)] = (tau * s).sin() + 0.4 * (2.0 * tau * s).cos() + 0.05 * i as f64;
        }
    }
    (d, t)
}

/// `d = 2` curve set (`n` curves) with curve-specific phase.
fn curve_set(t: &[f64], n: usize) -> FdCurveSet {
    let m = t.len();
    let mut x = FdMatrix::zeros(n, m);
    let mut y = FdMatrix::zeros(n, m);
    for i in 0..n {
        let c = 0.35 + 0.06 * i as f64;
        for j in 0..m {
            let s = t[j];
            x[(i, j)] = (-(s - c).powi(2) / 0.015).exp() + 0.2 * s;
            y[(i, j)] = (5.0 * s + 0.3 * i as f64).sin() * 0.6;
        }
    }
    FdCurveSet::from_dims(vec![x, y]).unwrap()
}

fn planar_curve(m: usize, ph: f64, amp: f64) -> FdMatrix {
    let mut b = FdMatrix::zeros(2, m);
    for j in 0..m {
        let s = j as f64 / (m - 1) as f64;
        let w = s + ph * (std::f64::consts::PI * s).sin() * 0.15;
        b[(0, j)] = w + 0.1 * (3.0 * w).cos();
        b[(1, j)] = amp * (6.0 * w).sin() + 0.2 * w * w;
    }
    b
}

fn closed_planar_curve(m: usize, a: f64, b: f64, wobble: f64) -> FdMatrix {
    let mut e = FdMatrix::zeros(2, m);
    let tau = 2.0 * std::f64::consts::PI;
    for j in 0..m {
        let th = tau * j as f64 / (m - 1) as f64;
        let r = 1.0 + wobble * (3.0 * th).cos();
        e[(0, j)] = a * r * th.cos();
        e[(1, j)] = b * r * th.sin();
    }
    e
}

/// Compare computed `(name, hash)` rows against the recorded table, reporting
/// every mismatch (and every computed hash, to make re-recording explicit).
fn check_all(got: &[(String, u64)], expected: &[(&str, u64)]) {
    for (n, h) in got {
        eprintln!("(\"{n}\", {h:#x}),");
    }
    assert_eq!(got.len(), expected.len(), "probe count");
    let bad: Vec<String> = got
        .iter()
        .zip(expected)
        .filter(|((gn, gh), (en, eh))| gn != en || gh != eh)
        .map(|((gn, gh), (en, eh))| format!("{gn}: got {gh:#x}; expected {en}: {eh:#x}"))
        .collect();
    assert!(bad.is_empty(), "bit-identity broken:\n{}", bad.join("\n"));
}

fn pair_hash(r: &crate::alignment::AlignmentResult) -> u64 {
    hash(&[&r.gamma, &r.f_aligned, &[r.distance]])
}

fn karcher_hash(k: &crate::alignment::KarcherMeanResult) -> u64 {
    hash(&[
        &k.mean,
        &k.mean_srsf,
        k.gammas.as_slice(),
        k.aligned_data.as_slice(),
        &[k.n_iter as f64, f64::from(u8::from(k.converged))],
    ])
}

// ─── 1. Classic 1-D entry points: uniform / non-uniform grid, band, λ ───────

/// Recorded on the pre-change (unpruned) code, commit 7d8cefe2.
const CLASSIC_1D_EXPECTED: &[(&str, u64)] = &[
    ("pair/uni/lam=0", 0xd064_e6cc_a9db_5c3b),
    ("pair_banded/uni/lam=0", 0xe9f7_649a_6b5c_fe76),
    ("self_dm/uni/lam=0", 0x7a23_030a_9166_a035),
    ("self_dm_band/uni/lam=0", 0x1f3c_78c8_9c81_e981),
    ("karcher/uni/lam=0", 0x73f5_8f79_e7e3_f0e0),
    ("karcher_band/uni/lam=0", 0x94d6_a829_b024_86fa),
    ("pair/uni/lam=0.25", 0xc04f_8912_16e5_8cbb),
    ("pair_banded/uni/lam=0.25", 0x4a40_96ba_19d2_710c),
    ("self_dm/uni/lam=0.25", 0xdc12_ce4c_9ee8_0bd1),
    ("self_dm_band/uni/lam=0.25", 0x0b65_fc2f_b03e_9449),
    ("karcher/uni/lam=0.25", 0xed81_8ef7_f932_a57b),
    ("karcher_band/uni/lam=0.25", 0x571c_5e39_b488_513b),
    ("pair/uni/lam=-0.5", 0xd064_e6cc_a9db_5c3b),
    ("pair/nonuni/lam=0", 0x3e52_cdca_8309_d8a2),
    ("pair_banded/nonuni/lam=0", 0xa202_4914_7205_0b07),
    ("self_dm/nonuni/lam=0", 0xe6e7_5dc9_eb41_b6f5),
    ("self_dm_band/nonuni/lam=0", 0x42ef_29c6_deb6_2369),
    ("karcher/nonuni/lam=0", 0xa177_4508_9f78_6281),
    ("karcher_band/nonuni/lam=0", 0x6409_8f2f_5acc_ee28),
    ("pair/nonuni/lam=0.25", 0x6229_75cf_d35e_4273),
    ("pair_banded/nonuni/lam=0.25", 0x2485_7e99_2bd0_de58),
    ("self_dm/nonuni/lam=0.25", 0xcab3_a295_1712_4465),
    ("self_dm_band/nonuni/lam=0.25", 0xe223_a9c6_498e_318d),
    ("karcher/nonuni/lam=0.25", 0x585c_029c_43e9_4f4a),
    ("karcher_band/nonuni/lam=0.25", 0x7e1d_35b4_945e_89e3),
    ("pair/nonuni/lam=-0.5", 0x3e52_cdca_8309_d8a2),
];

fn classic_1d_probes() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for (gname, t) in [("uni", grid(55)), ("nonuni", nonuniform_grid(47))] {
        let d = probe_on(&t, 5);
        let (f1, f2) = (d.row(0), d.row(3));
        for lam in [0.0, 0.25] {
            let tag = |what: &str| format!("{what}/{gname}/lam={lam}");
            out.push((
                tag("pair"),
                pair_hash(&elastic_align_pair(&f1, &f2, &t, lam)),
            ));
            let r = elastic_align_pair_banded(&f1, &f2, &t, lam, 0.06);
            out.push((tag("pair_banded"), pair_hash(&r)));
            let s = elastic_self_distance_matrix(&d, &t, lam);
            out.push((tag("self_dm"), hash(&[s.as_slice()])));
            let s = elastic_self_distance_matrix_with_band(&d, &t, lam, Some(0.06));
            out.push((tag("self_dm_band"), hash(&[s.as_slice()])));
            out.push((
                tag("karcher"),
                karcher_hash(&karcher_mean(&d, &t, 6, 1e-4, lam)),
            ));
            let k = karcher_mean_with_band(&d, &t, 6, 1e-4, lam, Some(0.06));
            out.push((tag("karcher_band"), karcher_hash(&k)));
        }
        // Negative λ (accepted by the classic functions, penalty inactive):
        // the gate never prunes it, so this pins the unpruned path.
        let r = elastic_align_pair(&f1, &f2, &t, -0.5);
        out.push((format!("pair/{gname}/lam=-0.5"), pair_hash(&r)));
    }
    out
}

#[test]
fn classic_1d_paths_bit_identical_to_unpruned() {
    check_all(&classic_1d_probes(), CLASSIC_1D_EXPECTED);
}

// ─── 2. Configurable DP: penalties, neighbourhoods, cost models, RBFGS start ─

/// Recorded on the pre-change (unpruned) code, commit 7d8cefe2.
const CONFIG_EXPECTED: &[(&str, u64)] = &[
    ("pair/l2gam/uni", 0x48f5_9812_cdd6_880b),
    ("cross_dm/l2gam/uni", 0x1d5e_8d28_8b6b_d501),
    ("karcher/l2gam/uni", 0x141f_588c_980d_d6a0),
    ("pair/l2psi/uni", 0x9b3d_bab5_3c3d_25bf),
    ("cross_dm/l2psi/uni", 0x3455_72d0_1c4c_1a69),
    ("karcher/l2psi/uni", 0x9bd7_67bb_ba52_f3ae),
    ("pair/dp2/uni", 0xf5cb_8909_c980_ef0b),
    ("cross_dm/dp2/uni", 0xd124_3964_ed68_ac79),
    ("karcher/dp2/uni", 0xe27d_c145_5511_6c1f),
    ("pair/scl3/uni", 0xda14_9c51_6c56_e973),
    ("cross_dm/scl3/uni", 0xca82_a5e7_3995_24a8),
    ("karcher/scl3/uni", 0xd0ee_6059_4214_f09e),
    ("pair/rbfgs_l2psi/uni", 0xbb41_f783_22d3_3786),
    ("cross_dm/rbfgs_l2psi/uni", 0x58a1_693c_17e9_0696),
    ("pair/rbfgs_l2gam/uni", 0xdc19_a616_961f_cd06),
    ("cross_dm/rbfgs_l2gam/uni", 0xe102_83d8_820b_01fe),
    ("pair/l2gam/nonuni", 0x6cb5_daf0_08dc_ade6),
    ("cross_dm/l2gam/nonuni", 0x3b4a_5b9f_84be_1a29),
    ("karcher/l2gam/nonuni", 0x18f8_7ba0_fd25_6d50),
    ("pair/l2psi/nonuni", 0x3777_dfe1_1d1a_810e),
    ("cross_dm/l2psi/nonuni", 0x95b5_8a0d_f715_2912),
    ("karcher/l2psi/nonuni", 0x0133_8a05_31aa_115e),
    ("pair/dp2/nonuni", 0xdea6_eab9_e692_a26f),
    ("cross_dm/dp2/nonuni", 0x22a4_305b_4584_df81),
    ("karcher/dp2/nonuni", 0x6a56_daf8_ff11_d551),
    ("pair/scl3/nonuni", 0x9983_0e7f_7618_b476),
    ("cross_dm/scl3/nonuni", 0x20bd_adb4_8a6c_a95e),
    ("karcher/scl3/nonuni", 0x0b2d_e109_e2eb_b44a),
    ("pair/rbfgs_l2psi/nonuni", 0x4242_b998_c802_d37d),
    ("cross_dm/rbfgs_l2psi/nonuni", 0xdd73_1b68_cc74_83c8),
    ("pair/rbfgs_l2gam/nonuni", 0x4bcc_6f6c_0b80_b627),
    ("cross_dm/rbfgs_l2gam/nonuni", 0xfb3c_cdfb_638b_5f86),
];

fn config_probes(prune: bool) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for (gname, t) in [("uni", grid(50)), ("nonuni", nonuniform_grid(43))] {
        let d = probe_on(&t, 4);
        let (f1, f2) = (d.row(0), d.row(2));
        let variants: [(
            &str,
            WarpOptimMethod,
            WarpObjectivePenalty,
            f64,
            usize,
            usize,
            DpEdgeCost,
        ); 6] = [
            (
                "l2gam",
                WarpOptimMethod::Dp,
                WarpObjectivePenalty::L2Gam,
                0.3,
                7,
                1,
                DpEdgeCost::PiecewiseConstant,
            ),
            (
                "l2psi",
                WarpOptimMethod::Dp,
                WarpObjectivePenalty::L2Psi,
                0.3,
                7,
                1,
                DpEdgeCost::PiecewiseConstant,
            ),
            (
                "dp2",
                WarpOptimMethod::Dp,
                WarpObjectivePenalty::L2Psi,
                0.1,
                10,
                1,
                DpEdgeCost::PiecewiseLinear,
            ),
            (
                "scl3",
                WarpOptimMethod::Dp,
                WarpObjectivePenalty::L2Gam,
                0.0,
                5,
                3,
                DpEdgeCost::PiecewiseConstant,
            ),
            (
                "rbfgs_l2psi",
                WarpOptimMethod::Rbfgs,
                WarpObjectivePenalty::L2Psi,
                0.05,
                7,
                1,
                DpEdgeCost::PiecewiseConstant,
            ),
            (
                "rbfgs_l2gam",
                WarpOptimMethod::Rbfgs,
                WarpObjectivePenalty::L2Gam,
                0.05,
                7,
                1,
                DpEdgeCost::PiecewiseConstant,
            ),
        ];
        for (name, method, penalty, lambda, nbhd, scl, cost) in variants {
            let mut cfg = WarpOptimConfig::default();
            cfg.dp_prune = prune;
            cfg.method = method;
            cfg.penalty = penalty;
            cfg.lambda = lambda;
            cfg.nbhd_dim = nbhd;
            cfg.dp_scl = scl;
            cfg.dp_cost = cost;
            let tag = |what: &str| format!("{what}/{name}/{gname}");
            let r = elastic_align_pair_with_config(&f1, &f2, &t, &cfg).unwrap();
            out.push((tag("pair"), pair_hash(&r)));
            let c = elastic_cross_distance_matrix_with_config(&d, &d, &t, &cfg).unwrap();
            out.push((tag("cross_dm"), hash(&[c.as_slice()])));
            if method == WarpOptimMethod::Dp {
                let k = karcher_mean_with_config(&d, &t, 4, 1e-4, &cfg).unwrap();
                out.push((tag("karcher"), karcher_hash(&k)));
            }
        }
    }
    out
}

#[test]
fn config_dp_paths_bit_identical_to_unpruned() {
    // Default (pruned) and the explicit `dp_prune = false` switch both
    // reproduce the unpruned pre-change hashes.
    assert!(WarpOptimConfig::default().dp_prune);
    check_all(&config_probes(true), CONFIG_EXPECTED);
    check_all(&config_probes(false), CONFIG_EXPECTED);
}

// ─── 3. Closed, R^d, curve-shape, constrained, partial, geodesic ────────────

/// Recorded on the pre-change (unpruned) code, commit 7d8cefe2.
const OTHER_EXPECTED: &[(&str, u64)] = &[
    ("closed_pair/lam=0", 0x635f_fb6f_e705_5ded),
    ("closed_karcher/lam=0", 0x058d_6f96_3ac3_f5b7),
    ("closed_pair/lam=0.3", 0xbec8_6844_5f47_568e),
    ("closed_karcher/lam=0.3", 0xce19_4b1c_53ed_b630),
    ("nd_pair/uni/lam=0", 0xd0b1_613d_6b70_b75e),
    ("nd_karcher/uni/lam=0", 0x38f3_354d_e49e_cab7),
    ("nd_pair/uni/lam=0.2", 0x05f7_7954_3ba2_fc5c),
    ("nd_karcher/uni/lam=0.2", 0xd888_f75f_9090_e464),
    ("nd_pair/nonuni/lam=0", 0xa145_3632_96e1_6d5d),
    ("nd_karcher/nonuni/lam=0", 0xc1d0_04a1_b0f2_d6ac),
    ("nd_pair/nonuni/lam=0.2", 0x2258_252c_8be4_4e82),
    ("nd_karcher/nonuni/lam=0.2", 0xbbef_cf77_b5c6_67e3),
    ("curve_shape/open", 0x44d6_71d2_a615_c103),
    ("curve_shape/open_pen", 0x829a_33c9_de5f_4ec9),
    ("curve_shape/closed", 0x5a58_5024_cd43_cb5b),
    ("constrained/lam=0", 0xf83a_62b5_9fb1_b781),
    ("partial/lam=0", 0x015e_1384_d3a4_c294),
    ("geodesic/lam=0", 0x24c3_4044_0bfa_8958),
    ("constrained/lam=0.2", 0xb4f4_7e1e_82e0_1cc7),
    ("partial/lam=0.2", 0xbb35_588f_ecfc_e4cc),
    ("geodesic/lam=0.2", 0x33ed_7b8c_0f37_9ac2),
];

fn other_probes() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let (pd, pt) = periodic_data(4, 40);
    for lam in [0.0, 0.3] {
        let r = elastic_align_pair_closed(&pd.row(0), &pd.row(2), &pt, lam).unwrap();
        out.push((
            format!("closed_pair/lam={lam}"),
            hash(&[
                &r.gamma,
                &r.f_aligned,
                &[r.distance, r.optimal_rotation as f64],
            ]),
        ));
        let k = karcher_mean_closed(&pd, &pt, 3, 1e-6, lam).unwrap();
        let rots: Vec<f64> = k.rotations.iter().map(|&r| r as f64).collect();
        out.push((
            format!("closed_karcher/lam={lam}"),
            hash(&[
                &k.mean,
                &k.mean_srsf,
                k.gammas.as_slice(),
                k.aligned_data.as_slice(),
                &rots,
            ]),
        ));
    }

    for (gname, t) in [("uni", grid(41)), ("nonuni", nonuniform_grid(41))] {
        let cs = curve_set(&t, 3);
        let one = |i: usize| {
            FdCurveSet::from_dims(
                cs.dims
                    .iter()
                    .map(|dm| {
                        let mut m1 = FdMatrix::zeros(1, dm.ncols());
                        for j in 0..dm.ncols() {
                            m1[(0, j)] = dm[(i, j)];
                        }
                        m1
                    })
                    .collect(),
            )
            .unwrap()
        };
        for lam in [0.0, 0.2] {
            let r = elastic_align_pair_nd(&one(0), &one(2), &t, lam);
            let mut parts: Vec<&[f64]> = vec![&r.gamma];
            for f in &r.f_aligned {
                parts.push(f);
            }
            let dist = [r.distance];
            parts.push(&dist);
            out.push((format!("nd_pair/{gname}/lam={lam}"), hash(&parts)));
            let sets: Vec<FdCurveSet> = (0..3).map(one).collect();
            let k = karcher_mean_nd(&sets, &t, 3, 1e-4, lam).unwrap();
            let mut parts: Vec<&[f64]> = vec![k.gammas.as_slice()];
            for f in &k.mean {
                parts.push(f);
            }
            out.push((format!("nd_karcher/{gname}/lam={lam}"), hash(&parts)));
        }
    }

    for (name, b1, b2, closed, lam) in [
        (
            "open",
            planar_curve(40, 0.0, 0.3),
            planar_curve(40, 1.0, 0.35),
            false,
            0.0,
        ),
        (
            "open_pen",
            planar_curve(40, 0.0, 0.3),
            planar_curve(40, 1.0, 0.35),
            false,
            0.05,
        ),
        (
            "closed",
            closed_planar_curve(37, 1.5, 0.8, 0.1),
            closed_planar_curve(37, 1.2, 1.0, 0.2),
            true,
            0.0,
        ),
    ] {
        let mut cfg = CurveShapeConfig::default();
        cfg.closed = closed;
        cfg.lambda = lam;
        let r = curve_shape_align_pair(&b1, &b2, &cfg).unwrap();
        out.push((
            format!("curve_shape/{name}"),
            hash(&[
                &r.gamma,
                r.q2_aligned.as_slice(),
                r.rotation.as_slice(),
                &[r.distance, r.seed as f64],
            ]),
        ));
    }

    let t = grid(60);
    let d = probe_on(&t, 4);
    for lam in [0.0, 0.2] {
        let r = elastic_align_pair_constrained(
            &d.row(0),
            &d.row(3),
            &t,
            &[(0.35, 0.5), (0.7, 0.8)],
            lam,
        );
        out.push((
            format!("constrained/lam={lam}"),
            hash(&[&r.gamma, &r.f_aligned, &[r.distance]]),
        ));
        let mut pc = PartialMatchConfig::default();
        pc.lambda = lam;
        let tt = grid(25);
        let tmpl: Vec<f64> = tt
            .iter()
            .map(|&s| (-(s - 0.5).powi(2) / 0.02).exp())
            .collect();
        let r = elastic_partial_match(&tmpl, &d.row(1), &tt, &t, &pc).unwrap();
        out.push((
            format!("partial/lam={lam}"),
            hash(&[
                &r.gamma,
                &[r.distance, r.start_index as f64, r.end_index as f64],
            ]),
        ));
        let g = curve_geodesic(&d.row(0), &d.row(2), &t, 5, lam).unwrap();
        out.push((
            format!("geodesic/lam={lam}"),
            hash(&[g.curves.as_slice(), g.warps.as_slice(), &g.distances]),
        ));
    }
    out
}

#[test]
fn other_dp_paths_bit_identical_to_unpruned() {
    check_all(&other_probes(), OTHER_EXPECTED);
}

// ─── 4. Engine: the gate and the pruned fill itself ─────────────────────────

#[test]
fn prune_gate_accepts_only_nonnegative_lambda_on_increasing_grids() {
    let t = grid(10);
    assert!(dp_prune_is_exact(&t, 0.0));
    assert!(dp_prune_is_exact(&t, 3.5));
    assert!(dp_prune_is_exact(&nonuniform_grid(12), 0.1));
    assert!(!dp_prune_is_exact(&t, -1e-12));
    assert!(!dp_prune_is_exact(&t, f64::NAN));
    let mut flat = t.clone();
    flat[4] = flat[3];
    assert!(!dp_prune_is_exact(&flat, 0.0));
    let rev: Vec<f64> = t.iter().rev().copied().collect();
    assert!(!dp_prune_is_exact(&rev, 0.0));
    let mut bad = t.clone();
    bad[9] = f64::NAN;
    assert!(!dp_prune_is_exact(&bad, 0.0));
    bad[9] = f64::INFINITY;
    assert!(!dp_prune_is_exact(&bad, 0.0));
}

/// Deterministic 64-bit LCG (Knuth MMIX constants) → `[0, 1)`.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Random edge-cost table keyed by `(sr, sc, tr, tc)` (pure function of the
/// edge, as the DP requires): `kind` 0 = continuous `≥ 0`, 1 = small integers
/// `0..4` (many exact ties, stressing tie-breaking), 2 = may be negative.
fn random_cost(seed: u64, kind: u8) -> impl Fn(usize, usize, usize, usize) -> f64 {
    move |sr, sc, tr, tc| {
        let key = (sr as u64) ^ ((sc as u64) << 16) ^ ((tr as u64) << 32) ^ ((tc as u64) << 48);
        let mut g = Lcg(seed ^ key.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        g.next();
        let u = g.next();
        match kind {
            0 => u * u * 3.0,
            1 => (u * 4.0).floor(),
            _ => u * 2.0 - 1.4,
        }
    }
}

/// Pruning never changes the DP path for non-negative edge costs — random
/// continuous and heavily tied integer costs, square and rectangular grids,
/// full and banded fills, both neighbourhood sizes.
#[test]
fn pruned_fill_matches_unpruned_for_nonnegative_costs() {
    let n10 = coprime_nbhd(10);
    let n7 = coprime_nbhd(7);
    let mut cases = 0;
    for seed in 0..40u64 {
        for kind in [0u8, 1] {
            for (nr, nc) in [(17, 17), (23, 14), (9, 31)] {
                for band in [None, Some(3)] {
                    for nbhd in [&*n7, &*n10] {
                        let cost = random_cost(seed, kind);
                        let off = dp_grid_solve_moves(
                            nr,
                            nc,
                            band,
                            DpMoves { nbhd, prune: false },
                            &cost,
                        );
                        let on =
                            dp_grid_solve_moves(nr, nc, band, DpMoves { nbhd, prune: true }, &cost);
                        assert_eq!(on, off, "seed={seed} kind={kind} {nr}x{nc} {band:?}");
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 40 * 2 * 3 * 2 * 2);
}

/// Why the gate exists: with negative edge costs pruning is *not* exact (a
/// predecessor at or above the current best could still win through a
/// negative edge), so the engine must never prune them.
#[test]
fn pruning_can_differ_with_negative_costs() {
    let n7 = coprime_nbhd(7);
    let differs = (0..40u64).any(|seed| {
        let cost = random_cost(seed, 2);
        let off = dp_grid_solve_moves(
            15,
            15,
            None,
            DpMoves {
                nbhd: &n7,
                prune: false,
            },
            &cost,
        );
        let on = dp_grid_solve_moves(
            15,
            15,
            None,
            DpMoves {
                nbhd: &n7,
                prune: true,
            },
            &cost,
        );
        on != off
    });
    assert!(
        differs,
        "expected a counterexample with negative edge costs"
    );
}
