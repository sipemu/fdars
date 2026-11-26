//! Tests for R^d curve shape analysis (ALN-CURVE, GH #48).
//!
//! Golden values: fdasrsf 2.7.5 (`.planning/quick/261126-1n9-*/golden_curve_shape.py`).
//! fdasrsf discretises differently (SRVF gradient step 1/T and Riemann-sum inner
//! product `Σ/T`, its own C DP `coptimum_reparam_curve`, coordinate-space
//! rotation in `elastic_distance_curve`), so distances are compared within a
//! tolerance set by fdasrsf's own resolution sensitivity: its distance for the
//! open golden pair moves by 1e-3 between T = 50…400 and the closed pair by
//! 2e-3, so we require agreement within 5e-3. Observed (m = 100): open 2-D
//! 0.10563 vs 0.10641, open 3-D 0.12491 vs 0.12387, closed 0.36366 vs 0.36458;
//! with rotation fdars is *below* fdasrsf (0.10222 vs 0.10593) because fdasrsf
//! rotates in coordinate space rather than SRVF space.
#![allow(clippy::field_reassign_with_default)]

use super::geometry::{self, Grid};
use super::*;
use std::f64::consts::PI;

fn linspace(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

fn from_rows(rows: &[Vec<f64>]) -> FdMatrix {
    let (d, m) = (rows.len(), rows[0].len());
    let mut b = FdMatrix::zeros(d, m);
    for (k, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            b[(k, j)] = v;
        }
    }
    b
}

fn open1(m: usize) -> FdMatrix {
    let t = linspace(m);
    from_rows(&[
        t.clone(),
        t.iter()
            .map(|&t| 0.3 * (2.0 * PI * t).sin() + 0.1 * t * t)
            .collect(),
    ])
}

fn open2(m: usize) -> FdMatrix {
    let s: Vec<f64> = linspace(m)
        .iter()
        .map(|&t| t + 0.15 * t * (1.0 - t))
        .collect();
    from_rows(&[
        s.iter().map(|&s| s + 0.05 * (PI * s).sin()).collect(),
        s.iter()
            .map(|&s| 0.35 * (2.0 * PI * s).sin() + 0.2 * s.powi(3))
            .collect(),
    ])
}

fn open3d_1(m: usize) -> FdMatrix {
    let t = linspace(m);
    from_rows(&[
        t.clone(),
        t.iter().map(|&t| 0.3 * (2.0 * PI * t).sin()).collect(),
        t.iter().map(|&t| 0.2 * (PI * t).cos()).collect(),
    ])
}

fn open3d_2(m: usize) -> FdMatrix {
    let s: Vec<f64> = linspace(m).iter().map(|&t| t.powf(1.2)).collect();
    from_rows(&[
        s.clone(),
        s.iter()
            .map(|&s| 0.25 * (2.0 * PI * s).sin() + 0.05 * s)
            .collect(),
        s.iter().map(|&s| 0.25 * (PI * s).cos()).collect(),
    ])
}

fn closed1(m: usize) -> FdMatrix {
    let th: Vec<f64> = linspace(m).iter().map(|&t| 2.0 * PI * t).collect();
    from_rows(&[
        th.iter().map(|&a| 1.5 * a.cos()).collect(),
        th.iter().map(|&a| 0.8 * a.sin()).collect(),
    ])
}

fn closed2(m: usize) -> FdMatrix {
    let th: Vec<f64> = linspace(m).iter().map(|&t| 2.0 * PI * t).collect();
    let r: Vec<f64> = th.iter().map(|&a| 1.0 + 0.2 * (3.0 * a).cos()).collect();
    from_rows(&[
        th.iter().zip(&r).map(|(&a, &r)| r * a.cos()).collect(),
        th.iter().zip(&r).map(|(&a, &r)| r * a.sin()).collect(),
    ])
}

/// Closed contour without rotational symmetry (so the optimal seed is unique).
fn closed3(m: usize) -> FdMatrix {
    let th: Vec<f64> = linspace(m).iter().map(|&t| 2.0 * PI * t).collect();
    let r: Vec<f64> = th
        .iter()
        .map(|&a| 1.0 + 0.2 * (3.0 * a).cos() + 0.1 * a.sin())
        .collect();
    from_rows(&[
        th.iter().zip(&r).map(|(&a, &r)| r * a.cos()).collect(),
        th.iter().zip(&r).map(|(&a, &r)| r * a.sin()).collect(),
    ])
}

fn family(m: usize, i: usize) -> FdMatrix {
    let t = linspace(m);
    let a = 0.2 + 0.05 * i as f64;
    from_rows(&[
        t.clone(),
        t.iter()
            .map(|&t| a * (PI * t).sin() + 0.1 * i as f64 * t * t)
            .collect(),
    ])
}

fn rot2(a: f64) -> FdMatrix {
    from_rows(&[vec![a.cos(), -a.sin()], vec![a.sin(), a.cos()]])
}

/// Rotation about the unit axis `(1, 2, 2)/3` by angle `a` (Rodrigues).
fn rot3(a: f64) -> FdMatrix {
    let u = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let (c, s) = (a.cos(), a.sin());
    let mut r = FdMatrix::zeros(3, 3);
    let k = [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            r[(i, j)] = c * id + s * k[i][j] + (1.0 - c) * u[i] * u[j];
        }
    }
    r
}

/// `s · R β + c`.
fn rigid(beta: &FdMatrix, r: &FdMatrix, s: f64, c: &[f64]) -> FdMatrix {
    let mut out = geometry::scaled(&geometry::rotate(r, beta), s);
    for j in 0..out.ncols() {
        for (v, ck) in out.column_mut(j).iter_mut().zip(c) {
            *v += ck;
        }
    }
    out
}

fn max_abs_diff(a: &FdMatrix, b: &FdMatrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn transpose(r: &FdMatrix) -> FdMatrix {
    let d = r.nrows();
    let mut t = FdMatrix::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            t[(i, j)] = r[(j, i)];
        }
    }
    t
}

fn rigid_cfg() -> CurveShapeConfig {
    let mut c = CurveShapeConfig::default();
    c.reparam = false;
    c
}

fn closed_cfg() -> CurveShapeConfig {
    let mut c = CurveShapeConfig::default();
    c.closed = true;
    c
}

// ─── Geometry ───────────────────────────────────────────────────────────────

#[test]
fn srvf_roundtrip_reconstructs_curve() {
    let m = 201;
    let b = open1(m);
    let q = curve_srvf(&b, false).unwrap();
    let rec = srvf_to_curve(&q).unwrap();
    let mut shifted = b.clone();
    for j in 0..m {
        for k in 0..2 {
            shifted[(k, j)] = b[(k, j)] - b[(k, 0)];
        }
    }
    assert!(max_abs_diff(&rec, &shifted) < 2e-3);
    // ‖q‖² equals the curve length
    let grid = Grid::new(m);
    let len = geometry::curve_length(&b, &grid);
    assert!((geometry::inner(&q, &q, &grid) - len).abs() < 1e-12);
    // srvf(R β) = R srvf(β) and is translation invariant
    let r = rot2(0.7);
    let q_r = curve_srvf(&rigid(&b, &r, 1.0, &[3.0, -1.0]), false).unwrap();
    assert!(max_abs_diff(&q_r, &geometry::rotate(&r, &q)) < 1e-12);
}

#[test]
fn procrustes_recovers_exact_rotation_2d_3d() {
    for (d, r0) in [(2usize, rot2(2.3)), (3, rot3(1.1))] {
        let b = if d == 2 { open1(60) } else { open3d_1(60) };
        let q2 = curve_srvf(&b, true).unwrap();
        let q1 = geometry::rotate(&r0, &q2);
        let r = optimal_rotation(&q1, &q2, false).unwrap();
        assert!(max_abs_diff(&r, &r0) < 1e-12, "d={d}");
    }
}

#[test]
fn procrustes_reflection_handling() {
    let q2 = curve_srvf(&open1(60), true).unwrap();
    let refl = from_rows(&[vec![1.0, 0.0], vec![0.0, -1.0]]);
    let q1 = geometry::rotate(&refl, &q2);
    let r_refl = optimal_rotation(&q1, &q2, true).unwrap();
    assert!(max_abs_diff(&r_refl, &refl) < 1e-12);
    let r_rot = optimal_rotation(&q1, &q2, false).unwrap();
    let det = r_rot[(0, 0)] * r_rot[(1, 1)] - r_rot[(0, 1)] * r_rot[(1, 0)];
    assert!(
        (det - 1.0).abs() < 1e-12,
        "SO(2) constraint violated: det = {det}"
    );
}

#[test]
fn closure_projection_closes_and_is_idempotent() {
    let m = 101;
    let grid = Grid::new(m);
    // open arcs in 2-D and 3-D
    for b in [open1(m), open3d_1(m)] {
        let q = curve_srvf(&b, true).unwrap();
        let r0 = closure_residual(&q).unwrap();
        assert!(r0.iter().map(|v| v * v).sum::<f64>().sqrt() > 0.1);
        let qc = project_closed_srvf(&q).unwrap();
        let r = closure_residual(&qc).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-9), "residual {r:?}");
        assert!((geometry::norm(&qc, &grid) - 1.0).abs() < 1e-12);
        // reconstructed curve is closed
        let rec = srvf_to_curve(&qc).unwrap();
        for k in 0..b.nrows() {
            assert!((rec[(k, m - 1)] - rec[(k, 0)]).abs() < 1e-9);
        }
        // idempotent
        let qcc = project_closed_srvf(&qc).unwrap();
        assert!(max_abs_diff(&qc, &qcc) < 1e-9);
    }
    // a sampled ellipse is already (almost) closed: projection barely moves it
    let q = curve_srvf(&closed1(m), true).unwrap();
    let qc = project_closed_srvf(&q).unwrap();
    assert!(geometry::norm(&geometry::axpby(1.0, &q, -1.0, &qc), &grid) < 1e-2);
}

#[test]
fn shift_closed_roundtrip() {
    let m = 41;
    let b = closed2(m);
    let s = geometry::shift_closed(&geometry::shift_closed(&b, 9), m - 1 - 9);
    assert!(max_abs_diff(&s, &b) < 1e-15);
}

// ─── Pairwise alignment / distance ─────────────────────────────────────────

#[test]
fn rigid_motion_scale_recovered_exactly() {
    for (b, r0) in [(open1(50), rot2(-1.9)), (open3d_1(50), rot3(0.8))] {
        let d = b.nrows();
        let b2 = rigid(&b, &r0, 2.7, &vec![0.5; d]);
        let res = curve_shape_align_pair(&b, &b2, &rigid_cfg()).unwrap();
        assert!(res.distance < 1e-10, "distance {}", res.distance);
        assert!(max_abs_diff(&res.rotation, &transpose(&r0)) < 1e-10);
        assert!(max_abs_diff(&res.beta2_aligned, &res.beta1) < 1e-10);
        // with reparametrisation enabled the DP keeps the identity
        let res = curve_shape_align_pair(&b, &b2, &CurveShapeConfig::default()).unwrap();
        assert!(res.distance < 1e-10, "distance {}", res.distance);
        assert!(res
            .gamma
            .iter()
            .zip(linspace(50))
            .all(|(g, t)| (g - t).abs() < 1e-12));
    }
}

#[test]
fn distance_invariant_to_reparametrisation_and_rotation() {
    let m = 80;
    let t = linspace(m);
    let gamma: Vec<f64> = t.iter().map(|&t| t + 0.25 * t * (1.0 - t)).collect();
    let grid = Grid::new(m);
    let b1 = open1(m);
    let b2 = open2(m);
    let base = curve_shape_distance(&b1, &b2, &CurveShapeConfig::default()).unwrap();
    // β1∘γ0 is the same shape as β1
    let b1w = geometry::warp_curve(&b1, &grid, &gamma);
    let d_self = curve_shape_distance(
        &b1,
        &rigid(&b1w, &rot2(0.9), 0.4, &[1.0, 1.0]),
        &CurveShapeConfig::default(),
    )
    .unwrap();
    let d_rigid = curve_shape_distance(&b1, &b1w, &rigid_cfg()).unwrap();
    // DP discretisation floor: the shared coprime-neighbourhood DP engine leaves a
    // residual ~0.02 at m = 80 for this warp (the 1-D `elastic_align_pair` finds the
    // same warp, see `curve_dp_matches_1d_engine`); fdasrsf's own floor for the same
    // pair is 0.012 at T = 80 and 0.030 at T = 40.
    assert!(d_self < 0.03, "self distance under warp {d_self}");
    assert!(
        d_rigid > 2.0 * d_self,
        "warp should matter without reparam: {d_rigid}"
    );
    // moving β2 within its orbit leaves the distance (nearly) unchanged
    let b2w = rigid(
        &geometry::warp_curve(&b2, &grid, &gamma),
        &rot2(-2.2),
        3.0,
        &[-4.0, 2.0],
    );
    let moved = curve_shape_distance(&b1, &b2w, &CurveShapeConfig::default()).unwrap();
    assert!((moved - base).abs() < 0.01, "base {base} moved {moved}");
    assert!(base > 0.05);
}

#[test]
fn distance_is_nearly_symmetric() {
    let cfg = CurveShapeConfig::default();
    let d12 = curve_shape_distance(&open1(80), &open2(80), &cfg).unwrap();
    let d21 = curve_shape_distance(&open2(80), &open1(80), &cfg).unwrap();
    assert!((d12 - d21).abs() < 5e-3, "{d12} vs {d21}");
}

#[test]
fn alternation_never_worse_than_single_pass() {
    let mut one = CurveShapeConfig::default();
    one.align_iter = 1;
    let d1 = curve_shape_distance(
        &open1(60),
        &rigid(&open2(60), &rot2(0.5), 1.0, &[0.0, 0.0]),
        &one,
    )
    .unwrap();
    let d3 = {
        let mut c = one.clone();
        c.align_iter = 3;
        curve_shape_distance(
            &open1(60),
            &rigid(&open2(60), &rot2(0.5), 1.0, &[0.0, 0.0]),
            &c,
        )
        .unwrap()
    };
    assert!(d3 <= d1 + 1e-15);
}

#[test]
fn size_and_shape_distance_and_length_term() {
    let m = 50;
    let b = open1(m);
    let b2 = rigid(&b, &rot2(0.3), 2.0, &[0.0, 0.0]);
    let grid = Grid::new(m);
    let len = geometry::curve_length(&b, &grid);
    // scale kept: d(β, 2Rβ) = ‖q − √2 q‖ = (√2 − 1)√L
    let mut cfg = rigid_cfg();
    cfg.scale = false;
    let d = curve_shape_distance(&b, &b2, &cfg).unwrap();
    assert!((d - (2f64.sqrt() - 1.0) * len.sqrt()).abs() < 1e-10, "{d}");
    cfg.reparam = true;
    let d = curve_shape_distance(&b, &b2, &cfg).unwrap();
    assert!((d - (2f64.sqrt() - 1.0) * len.sqrt()).abs() < 1e-9, "{d}");
    // scale removed + length term: |ln(√L/√(2L))| = ln(2)/2
    let mut cfg = rigid_cfg();
    cfg.include_length = true;
    let d = curve_shape_distance(&b, &b2, &cfg).unwrap();
    assert!((d - 0.5 * 2f64.ln()).abs() < 1e-10, "{d}");
}

#[test]
fn closed_seed_rotation_scale_recovered_exactly() {
    let m = 61;
    let b = closed3(m);
    let b2 = rigid(&geometry::shift_closed(&b, 9), &rot2(1.3), 0.5, &[2.0, 2.0]);
    let mut cfg = closed_cfg();
    cfg.reparam = false;
    let res = curve_shape_align_pair(&b, &b2, &cfg).unwrap();
    assert_eq!(res.seed, m - 1 - 9);
    assert!(res.distance < 1e-9, "distance {}", res.distance);
    assert!(max_abs_diff(&res.rotation, &rot2(-1.3)) < 1e-9);
    // without seed search the start-point mismatch is visible
    cfg.seed_search = false;
    let d = curve_shape_distance(&b, &b2, &cfg).unwrap();
    assert!(d > 0.1, "{d}");
}

#[test]
fn closed_alignment_outputs_are_closed_and_seed_invariant() {
    let m = 61;
    let t = linspace(m);
    let grid = Grid::new(m);
    let gamma: Vec<f64> = t.iter().map(|&t| t + 0.2 * t * (1.0 - t)).collect();
    let b1 = closed1(m);
    let b2 = closed2(m);
    let cfg = closed_cfg();
    let res = curve_shape_align_pair(&b1, &b2, &cfg).unwrap();
    for q in [&res.q1, &res.q2_aligned] {
        assert!(closure_residual(q).unwrap().iter().all(|v| v.abs() < 1e-9));
    }
    let moved = rigid(
        &geometry::shift_closed(&geometry::warp_curve(&b2, &grid, &gamma), 17),
        &rot2(2.0),
        1.7,
        &[0.3, 0.0],
    );
    let d_moved = curve_shape_distance(&b1, &moved, &cfg).unwrap();
    assert!(
        (d_moved - res.distance).abs() < 0.015,
        "{} vs {d_moved}",
        res.distance
    );
    // the same shape at another seed / rotation / parametrisation is at ~0
    let b2_moved = rigid(
        &geometry::shift_closed(&geometry::warp_curve(&b2, &grid, &gamma), 23),
        &rot2(-0.7),
        1.0,
        &[0.0, 0.0],
    );
    let d_self = curve_shape_distance(&b2, &b2_moved, &cfg).unwrap();
    // fdasrsf 2.7.5 elastic_distance_curve gives 0.087 for this very configuration
    // (T = 61; 0.078 at T = 121): the residual is the DP / sub-sample-seed floor.
    assert!(d_self < 0.06, "{d_self}");
}

#[test]
fn golden_open_distances_fdasrsf() {
    // fdasrsf 2.7.5 elastic_distance_curve(beta1, beta2, closed=0, rotation, scale=False), T=100.
    // With rotation=True fdasrsf rotates in coordinate space before the DP, which is
    // suboptimal for the SRVF metric: fdars (SRVF Procrustes + alternation) must not
    // exceed it by more than the tolerance.
    const ROT: f64 = 0.10592897026672783;
    const NOROT: f64 = 0.10640984514487875;
    const NOROT_3D: f64 = 0.12387121779690997;
    let cfg = CurveShapeConfig::default();
    let mut norot = cfg.clone();
    norot.rotation = false;
    let d_rot = curve_shape_distance(&open1(100), &open2(100), &cfg).unwrap();
    let d_norot = curve_shape_distance(&open1(100), &open2(100), &norot).unwrap();
    let d_3d = curve_shape_distance(&open3d_1(100), &open3d_2(100), &norot).unwrap();
    assert!((d_norot - NOROT).abs() < 5e-3, "{d_norot} vs {NOROT}");
    assert!((d_3d - NOROT_3D).abs() < 5e-3, "{d_3d} vs {NOROT_3D}");
    assert!(
        d_rot <= ROT + 5e-3 && (d_rot - ROT).abs() < 1e-2,
        "{d_rot} vs {ROT}"
    );
    assert!(d_rot <= d_norot + 1e-12);
}

#[test]
fn golden_closed_distances_fdasrsf() {
    // fdasrsf 2.7.5 elastic_distance_curve(closed1, closed2, closed=1, rotation=True), T=100;
    // second value with closed2 rotated by 0.4 rad and seed-shifted by 12 (shift_f).
    const D: f64 = 0.36457588053483153;
    const D_MOVED: f64 = 0.36519176353473226;
    let cfg = closed_cfg();
    let d = curve_shape_distance(&closed1(100), &closed2(100), &cfg).unwrap();
    let moved = geometry::rotate(&rot2(0.4), &geometry::shift_closed(&closed2(100), 12));
    let dm = curve_shape_distance(&closed1(100), &moved, &cfg).unwrap();
    assert!((d - D).abs() < 5e-3, "{d} vs {D}");
    assert!((dm - D_MOVED).abs() < 5e-3, "{dm} vs {D_MOVED}");
    // fdars recovers the exact sample shift and rotation: identical distance
    assert!((dm - d).abs() < 1e-9, "{dm} vs {d}");
}

#[test]
fn self_distance_matrix_matches_pairs() {
    let curves = curves_to_set(&[open1(40), open2(40), family(40, 2)]).unwrap();
    let cfg = CurveShapeConfig::default();
    let dm = curve_shape_self_distance_matrix(&curves, &cfg).unwrap();
    for i in 0..3 {
        assert_eq!(dm[(i, i)], 0.0);
        for j in 0..3 {
            assert_eq!(dm[(i, j)], dm[(j, i)]);
        }
    }
    let d01 = curve_shape_distance(&open1(40), &open2(40), &cfg).unwrap();
    assert!((dm[(0, 1)] - d01).abs() < 1e-14);
}

#[test]
fn invalid_inputs_are_rejected() {
    let cfg = CurveShapeConfig::default();
    assert!(curve_shape_distance(&open1(40), &open1(41), &cfg).is_err());
    assert!(curve_shape_distance(&open1(3), &open1(3), &cfg).is_err());
    let flat = FdMatrix::zeros(2, 20);
    assert!(matches!(
        curve_shape_distance(&flat, &open1(20), &cfg),
        Err(FdarError::ComputationFailed { .. })
    ));
    let one_d = from_rows(&[linspace(20)]);
    let mut c = closed_cfg();
    assert!(curve_shape_distance(&one_d, &one_d, &c).is_err());
    c.seed_step = 0;
    assert!(c.validate().is_err());
    let mut c = CurveShapeConfig::default();
    c.lambda = -1.0;
    assert!(curve_shape_distance(&open1(20), &open1(20), &c).is_err());
    let mut nan = open1(20);
    nan[(0, 3)] = f64::NAN;
    assert!(curve_shape_distance(&nan, &open1(20), &CurveShapeConfig::default()).is_err());
    assert!(curve_from_set(&curves_to_set(&[open1(20)]).unwrap(), 1).is_err());
    assert!(curves_to_set(&[open1(20), open1(21)]).is_err());
    // 1-D open curves are allowed (rotation group {+1})
    assert!(curve_shape_distance(&one_d, &one_d, &CurveShapeConfig::default()).unwrap() < 1e-12);
}

// ─── Karcher mean / covariance / PCA ───────────────────────────────────────

const FDASRSF_MEAN_X: [f64; 50] = [
    -0.43974848,
    -0.42026392,
    -0.40077718,
    -0.38129068,
    -0.36181023,
    -0.34234236,
    -0.32289345,
    -0.30346974,
    -0.28407728,
    -0.26491002,
    -0.24616818,
    -0.2276649,
    -0.20920596,
    -0.19079504,
    -0.17243564,
    -0.15413098,
    -0.13588401,
    -0.11772649,
    -0.09974774,
    -0.0819791,
    -0.06433413,
    -0.04675394,
    -0.02923878,
    -0.01178935,
    0.00559393,
    0.02291082,
    0.04016133,
    0.05737349,
    0.07463008,
    0.09203836,
    0.10959287,
    0.12715418,
    0.14464132,
    0.16205553,
    0.17939849,
    0.19667246,
    0.21388022,
    0.23102512,
    0.24811101,
    0.26514225,
    0.28228857,
    0.29971643,
    0.31726385,
    0.33477366,
    0.35225349,
    0.3697112,
    0.38715486,
    0.40459265,
    0.42203182,
    0.43947491,
];
const FDASRSF_MEAN_Y: [f64; 50] = [
    -0.12981165,
    -0.11693399,
    -0.10406742,
    -0.09124869,
    -0.07853378,
    -0.06598241,
    -0.05365316,
    -0.0416033,
    -0.02988852,
    -0.01867369,
    -0.00811667,
    0.00186554,
    0.01135626,
    0.02031657,
    0.02870993,
    0.03650226,
    0.04366213,
    0.05015093,
    0.05591771,
    0.06093755,
    0.06522683,
    0.06879639,
    0.07163747,
    0.07374024,
    0.07509773,
    0.07570585,
    0.07556337,
    0.07466834,
    0.07300915,
    0.0705578,
    0.06729876,
    0.06325344,
    0.05845112,
    0.05291521,
    0.04667274,
    0.03975411,
    0.03219296,
    0.02402591,
    0.01529235,
    0.00603426,
    -0.00381515,
    -0.01432563,
    -0.02535008,
    -0.0367308,
    -0.04840839,
    -0.06032186,
    -0.07240885,
    -0.08460581,
    -0.0968529,
    -0.10911087,
];

#[test]
fn karcher_mean_of_one_shape_in_many_poses() {
    let m = 40;
    let b = open1(m);
    let poses = [(0.0, 1.0), (1.0, 2.0), (-2.0, 0.5), (2.5, 3.0)];
    let curves: Vec<FdMatrix> = poses
        .iter()
        .map(|&(a, s)| rigid(&b, &rot2(a), s, &[a, s]))
        .collect();
    let set = curves_to_set(&curves).unwrap();
    let km = curve_shape_karcher_mean(&set, &rigid_cfg(), 10, 1e-8).unwrap();
    assert!(km.converged);
    assert!(
        km.distances.iter().all(|&d| d < 1e-10),
        "{:?}",
        km.distances
    );
    // the mean SRVF is exactly the (rotated) unit SRVF of the shape …
    let q = curve_srvf(&b, true).unwrap();
    let r = optimal_rotation(&km.mean_srvf, &q, false).unwrap();
    assert!(max_abs_diff(&km.mean_srvf, &geometry::rotate(&r, &q)) < 1e-10);
    // … and the reconstructed mean curve matches up to the SRVF round-trip error
    let d = curve_shape_distance(&km.mean, &b, &rigid_cfg()).unwrap();
    assert!(d < 1e-2, "{d}");
    // geometric mean length
    let grid = Grid::new(m);
    let l0 = geometry::curve_length(&b, &grid);
    let expect = l0 * (poses.iter().map(|p| p.1.ln()).sum::<f64>() / 4.0).exp();
    assert!((km.mean_length - expect).abs() < 1e-10 * expect);
}

#[test]
fn karcher_flat_geometry_is_exact_srvf_average() {
    // scale kept: curves sᵢ·Rᵢβ have aligned SRVFs √sᵢ q, so the flat Karcher mean
    // is mean(√sᵢ)·q exactly.
    let m = 40;
    let b = open1(m);
    let scales = [1.0, 2.0, 4.0];
    let curves: Vec<FdMatrix> = scales
        .iter()
        .enumerate()
        .map(|(i, &s)| rigid(&b, &rot2(i as f64), s, &[0.0, 0.0]))
        .collect();
    let mut cfg = rigid_cfg();
    cfg.scale = false;
    let km = curve_shape_karcher_mean(&curves_to_set(&curves).unwrap(), &cfg, 5, 1e-10).unwrap();
    let grid = Grid::new(m);
    let q = geometry::srvf_raw(&geometry::center(&b, &grid), &grid);
    let c = scales.iter().map(|s: &f64| s.sqrt()).sum::<f64>() / 3.0;
    let r = optimal_rotation(&km.mean_srvf, &q, false).unwrap();
    let target = geometry::scaled(&geometry::rotate(&r, &q), c);
    assert!(max_abs_diff(&km.mean_srvf, &target) < 1e-10);
    assert!(km.converged);
}

#[test]
fn golden_karcher_mean_fdasrsf() {
    // fdasrsf 2.7.5: fdacurve(beta, mode="O", N=50).karcher_mean(rotation=True) on
    // family(50, i), i = 0..3 (8-digit output). fdasrsf energy history
    // E = [0.0612, 0.0309, 0.0178] (‖v̄‖).
    let m = 50;
    let set = curves_to_set(&(0..4).map(|i| family(m, i)).collect::<Vec<_>>()).unwrap();
    let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 30, 1e-3).unwrap();
    let golden = from_rows(&[FDASRSF_MEAN_X.to_vec(), FDASRSF_MEAN_Y.to_vec()]);
    let cfg = CurveShapeConfig::default();
    let energy = |mean: &FdMatrix| -> f64 {
        (0..4)
            .map(|i| {
                curve_shape_distance(mean, &family(m, i), &cfg)
                    .unwrap()
                    .powi(2)
            })
            .sum()
    };
    let (e_ours, e_ref) = (energy(&km.mean), energy(&golden));
    let d = curve_shape_distance(&km.mean, &golden, &cfg).unwrap();
    // fdasrsf stops early (‖v̄‖ = 0.018 at its last step), so fdars must reach an
    // energy no larger than fdasrsf's mean and land close to it.
    assert!(
        e_ours <= e_ref * 1.001,
        "energy ours {e_ours} vs fdasrsf {e_ref}"
    );
    assert!(d < 0.05, "mean distance to fdasrsf mean {d}");
    // energy decreases monotonically
    assert!(km.energy.windows(2).all(|w| w[1] <= w[0]));
}

#[test]
fn karcher_closed_contours() {
    let m = 61;
    let grid = Grid::new(m);
    let t = linspace(m);
    let curves: Vec<FdMatrix> = (0..3)
        .map(|i| {
            let gamma: Vec<f64> = t
                .iter()
                .map(|&t| t + 0.1 * i as f64 * t * (1.0 - t))
                .collect();
            let b = geometry::warp_curve(&closed2(m), &grid, &gamma);
            rigid(
                &geometry::shift_closed(&b, 5 * i),
                &rot2(0.8 * i as f64),
                1.0 + i as f64,
                &[0.0, 0.0],
            )
        })
        .collect();
    let km =
        curve_shape_karcher_mean(&curves_to_set(&curves).unwrap(), &closed_cfg(), 5, 1e-3).unwrap();
    assert!(closure_residual(&km.mean_srvf)
        .unwrap()
        .iter()
        .all(|v| v.abs() < 1e-9));
    // all three are the same shape: residual distances are the DP floor
    assert!(km.distances.iter().all(|&d| d < 0.05), "{:?}", km.distances);
    let d = curve_shape_distance(&km.mean, &closed2(m), &closed_cfg()).unwrap();
    assert!(d < 0.05, "{d}");
    // shooting vectors are tangent to the closed manifold at the mean
    let normals = geometry::closure_normals(&km.mean_srvf);
    for i in 0..3 {
        let mut v = FdMatrix::zeros(2, m);
        for k in 0..2 {
            for j in 0..m {
                v[(k, j)] = km.shooting_vectors[(i, k * m + j)];
            }
        }
        for b in &normals {
            assert!(geometry::inner(&v, b, &grid).abs() < 1e-10);
        }
        assert!(geometry::inner(&v, &km.mean_srvf, &grid).abs() < 1e-10);
    }
}

#[test]
fn covariance_and_pca_properties() {
    let m = 40;
    let set = curves_to_set(&(0..6).map(|i| family(m, i)).collect::<Vec<_>>()).unwrap();
    let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 20, 1e-3).unwrap();
    let cov = curve_shape_karcher_covariance(&km).unwrap();
    let p = 2 * m;
    assert_eq!(cov.shape(), (p, p));
    for a in 0..p {
        assert!(cov[(a, a)] >= 0.0);
        for b in 0..p {
            assert_eq!(cov[(a, b)], cov[(b, a)]);
        }
    }
    let pca = curve_shape_pca(&km, 3).unwrap();
    let grid = Grid::new(m);
    // L²-orthonormal components
    for a in 0..3 {
        for b in 0..3 {
            let ip = geometry::inner(&pca.components[a], &pca.components[b], &grid);
            let e = if a == b { 1.0 } else { 0.0 };
            assert!((ip - e).abs() < 1e-8, "<phi{a}, phi{b}> = {ip}");
        }
    }
    // eigenvalues = score variances, decreasing; near one-parameter family
    for k in 0..3 {
        let var: f64 = (0..6).map(|i| pca.scores[(i, k)].powi(2)).sum::<f64>() / 5.0;
        assert!((var - pca.eigenvalues[k]).abs() < 1e-10 * (1.0 + var));
        if k > 0 {
            assert!(pca.eigenvalues[k] <= pca.eigenvalues[k - 1]);
        }
    }
    assert!(
        pca.cumulative_variance[1] > 0.9,
        "{:?}",
        pca.cumulative_variance
    );
    // scores are inner products with the components
    for i in 0..6 {
        let mut v = FdMatrix::zeros(2, m);
        for k in 0..2 {
            for j in 0..m {
                v[(k, j)] = km.shooting_vectors[(i, k * m + j)] - pca.mean_tangent[(k, j)];
            }
        }
        let s = geometry::inner(&v, &pca.components[0], &grid);
        assert!((s - pca.scores[(i, 0)]).abs() < 1e-10);
    }
    // the mode shape at coef 0 is the mean (v̄ ≈ 0 at convergence)
    let mode0 = pca.mode_shape(&km, 0, 0.0).unwrap();
    assert!(curve_shape_distance(&mode0, &km.mean, &rigid_cfg()).unwrap() < 0.02);
    let mode_p = pca.mode_shape(&km, 0, 2.0).unwrap();
    assert!(curve_shape_distance(&mode_p, &km.mean, &rigid_cfg()).unwrap() > 1e-3);
    assert!(pca.mode_shape(&km, 5, 0.0).is_err());
    assert!(curve_shape_pca(&km, 0).is_err());
}

#[test]
fn curve_dp_matches_1d_engine() {
    // For d = 1 the R^d DP is the shared 1-D engine: same warp as elastic_align_pair.
    for m in [40usize, 80] {
        let t = linspace(m);
        let gamma0: Vec<f64> = t.iter().map(|&t| t + 0.25 * t * (1.0 - t)).collect();
        let f: Vec<f64> = t
            .iter()
            .map(|&t| 0.3 * (2.0 * PI * t).sin() + 0.1 * t * t + t)
            .collect();
        let fw = crate::alignment::reparameterize_curve(&f, &t, &gamma0);
        let r1 = crate::alignment::elastic_align_pair(&f, &fw, &t, 0.0);
        let mut c = CurveShapeConfig::default();
        c.scale = false;
        let rc = curve_shape_align_pair(&from_rows(&[f]), &from_rows(&[fw]), &c).unwrap();
        let diff = r1
            .gamma
            .iter()
            .zip(&rc.gamma)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-12, "m={m}: warp differs by {diff}");
    }
}
