//! Tests for the R^d curve downstream analyses (ALN-NDDOWN, GH #54): Karcher
//! median, TSRVF, depth, outlier detection, box plot and bootstrap CI.
//!
//! Golden scripts: `.planning/quick/261126-43c-*/golden_curve_boxplot.R` and
//! `golden_curve_depth.R` (fdasrvf 2.5.0, outputs in the matching `.out` files).
#![allow(clippy::field_reassign_with_default, clippy::excessive_precision)]

use super::geometry::{self, Grid};
use super::karcher::{exp_map, log_map};
use super::*;
use crate::alignment::OutlierThreshold;
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

/// Open planar curve `(t, a sin(πt) + c sin(2π t^p))` evaluated at `s(t)`.
fn bump(m: usize, a: f64, c: f64, p: f64) -> FdMatrix {
    let t = linspace(m);
    from_rows(&[
        t.clone(),
        t.iter()
            .map(|&t| a * (PI * t).sin() + c * (2.0 * PI * t.powf(p)).sin())
            .collect(),
    ])
}

/// One-parameter family of open curves (index 0..n).
fn family(m: usize, i: usize) -> FdMatrix {
    bump(m, 0.15 + 0.05 * i as f64, 0.0, 1.0)
}

/// A zig-zag open curve (gross shape outlier w.r.t. `family`).
fn zigzag(m: usize) -> FdMatrix {
    let t = linspace(m);
    from_rows(&[
        t.clone(),
        t.iter().map(|&t| 0.25 * (8.0 * PI * t).sin()).collect(),
    ])
}

/// Closed contour `r(θ) = 1 + e cos 3θ + f sin θ`.
fn blob(m: usize, e: f64, f: f64) -> FdMatrix {
    let th: Vec<f64> = linspace(m).iter().map(|&t| 2.0 * PI * t).collect();
    let r: Vec<f64> = th
        .iter()
        .map(|&a| 1.0 + e * (3.0 * a).cos() + f * a.sin())
        .collect();
    from_rows(&[
        th.iter().zip(&r).map(|(&a, &r)| r * a.cos()).collect(),
        th.iter().zip(&r).map(|(&a, &r)| r * a.sin()).collect(),
    ])
}

fn rot2(a: f64) -> FdMatrix {
    from_rows(&[vec![a.cos(), -a.sin()], vec![a.sin(), a.cos()]])
}

/// `s · R β + c`.
fn rigid(beta: &FdMatrix, angle: f64, s: f64, c: [f64; 2]) -> FdMatrix {
    let mut out = geometry::scaled(&geometry::rotate(&rot2(angle), beta), s);
    for j in 0..out.ncols() {
        for (v, ck) in out.column_mut(j).iter_mut().zip(c) {
            *v += ck;
        }
    }
    out
}

/// Resample a curve at `γ(t) = t + ε t(1−t)` (linear interpolation).
fn reparam(beta: &FdMatrix, eps: f64) -> FdMatrix {
    let m = beta.ncols();
    let grid = Grid::new(m);
    let gam: Vec<f64> = grid.t.iter().map(|&t| t + eps * t * (1.0 - t)).collect();
    geometry::warp_curve(beta, &grid, &gam)
}

fn set(curves: &[FdMatrix]) -> FdCurveSet {
    curves_to_set(curves).unwrap()
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn argmin(v: &[f64]) -> usize {
    (0..v.len()).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn closed_cfg() -> CurveShapeConfig {
    let mut c = CurveShapeConfig::default();
    c.closed = true;
    c
}

// ─── Karcher median ─────────────────────────────────────────────────────────

#[test]
fn median_of_one_shape_in_many_poses_is_that_shape() {
    let m = 40;
    let base = family(m, 2);
    let curves: Vec<FdMatrix> = (0..5)
        .map(|i| {
            rigid(
                &base,
                0.4 * i as f64,
                1.0 + 0.3 * i as f64,
                [i as f64, -1.0],
            )
        })
        .collect();
    let cfg = CurveShapeConfig::default();
    let med = curve_shape_karcher_median(&set(&curves), &cfg, 10, 1e-6).unwrap();
    // every pose aligns exactly onto the median (as for the Karcher mean test)
    assert!(
        med.distances.iter().all(|&x| x < 1e-10),
        "{:?}",
        med.distances
    );
    // the reconstructed median curve carries the SRVF → curve discretisation error
    let d = curve_shape_distance(&med.mean, &base, &cfg).unwrap();
    assert!(d < 1e-2, "median distance to the common shape {d}");
}

#[test]
fn median_is_robust_to_a_gross_outlier() {
    let m = 40;
    let mut curves: Vec<FdMatrix> = (0..7).map(|i| family(m, i)).collect();
    let clean = curve_shape_karcher_mean(&set(&curves), &CurveShapeConfig::default(), 20, 1e-4)
        .unwrap()
        .mean;
    curves.push(zigzag(m));
    let cfg = CurveShapeConfig::default();
    let mean = curve_shape_karcher_mean(&set(&curves), &cfg, 20, 1e-4).unwrap();
    let med = curve_shape_karcher_median(&set(&curves), &cfg, 20, 1e-4).unwrap();
    let dm = curve_shape_distance(&mean.mean, &clean, &cfg).unwrap();
    let dmed = curve_shape_distance(&med.mean, &clean, &cfg).unwrap();
    assert!(
        dmed < 0.5 * dm,
        "median moved {dmed} vs mean {dm} by the outlier"
    );
    // the median minimises Σ dᵢ: its energy is no larger than at the mean
    let sum_at_mean: f64 = mean.distances.iter().sum();
    let sum_at_med: f64 = med.distances.iter().sum();
    assert!(
        sum_at_med <= sum_at_mean + 1e-9,
        "{sum_at_med} > {sum_at_mean}"
    );
    // energy history is non-increasing
    assert!(med.energy.windows(2).all(|w| w[1] <= w[0] + 1e-12));
}

// ─── TSRVF ──────────────────────────────────────────────────────────────────

/// Reshape a `k·m + j` tangent row into a `d × m` field.
fn row_field(row: &[f64], d: usize, m: usize) -> FdMatrix {
    let mut v = FdMatrix::zeros(d, m);
    for k in 0..d {
        for j in 0..m {
            v[(k, j)] = row[k * m + j];
        }
    }
    v
}

fn check_tsrvf(curves: &FdCurveSet, cfg: &CurveShapeConfig) {
    let km = curve_shape_karcher_mean(curves, cfg, 15, 1e-4).unwrap();
    let ts = curve_shape_tsrvf(&km, curves).unwrap();
    // reproduces the Karcher shooting vectors exactly
    assert_eq!(ts.tangent_vectors, km.shooting_vectors);
    assert_eq!(ts.distances, km.distances);
    let (d, m) = km.mean_srvf.shape();
    let grid = Grid::new(m);
    let mu = &km.mean_srvf;
    let normals = geometry::closure_normals(mu);
    for i in 0..curves.ncurves() {
        let v = row_field(&ts.tangent_vectors.row(i), d, m);
        let nv = geometry::norm(&v, &grid);
        if cfg.scale {
            // tangent to the pre-shape sphere
            assert!(geometry::inner(&v, mu, &grid).abs() < 1e-10 * nv.max(1.0));
        }
        if cfg.scale && !cfg.closed {
            // the sphere log map has length = geodesic distance (for closed
            // contours the closure-tangent projection shortens it slightly)
            assert!(
                (nv - ts.distances[i]).abs() < 1e-9,
                "{nv} vs {}",
                ts.distances[i]
            );
        }
        if cfg.closed {
            for b in &normals {
                let c = geometry::inner(&v, b, &grid) / geometry::norm(b, &grid);
                assert!(
                    c.abs() < 1e-10 * nv.max(1.0),
                    "closure normal component {c}"
                );
            }
        }
    }
}

#[test]
fn tsrvf_open_reproduces_shooting_vectors_and_is_tangent() {
    let m = 40;
    let curves: Vec<FdMatrix> = (0..5).map(|i| family(m, i)).collect();
    check_tsrvf(&set(&curves), &CurveShapeConfig::default());
    let mut flat = CurveShapeConfig::default();
    flat.scale = false;
    check_tsrvf(&set(&curves), &flat);
}

#[test]
fn tsrvf_closed_tangent_orthogonal_to_closure_normals() {
    let m = 41;
    let curves: Vec<FdMatrix> = (0..4)
        .map(|i| {
            rigid(
                &blob(m, 0.1 + 0.04 * i as f64, 0.05),
                0.3 * i as f64,
                1.0,
                [0.0; 2],
            )
        })
        .collect();
    let mut cfg = closed_cfg();
    cfg.seed_step = 8;
    check_tsrvf(&set(&curves), &cfg);
}

#[test]
fn sphere_exp_inverts_log() {
    let m = 40;
    let cfg = CurveShapeConfig::default();
    let grid = Grid::new(m);
    let mu = curve_srvf(&family(m, 1), true).unwrap();
    let q = curve_srvf(&family(m, 4), true).unwrap();
    let v = log_map(&mu, &q, &cfg, &grid);
    let back = exp_map(&mu, &v, &cfg, &grid);
    let err = geometry::norm(&geometry::axpby(1.0, &back, -1.0, &q), &grid);
    assert!(err < 1e-12, "exp(log(q)) error {err}");
}

#[test]
fn tsrvf_of_new_curve_is_pose_invariant_and_inverts() {
    let m = 40;
    let curves: Vec<FdMatrix> = (0..5).map(|i| family(m, i)).collect();
    let cfg = CurveShapeConfig::default();
    let km = curve_shape_karcher_mean(&set(&curves), &cfg, 15, 1e-4).unwrap();
    let posed = rigid(&curves[3], 1.1, 2.5, [3.0, -4.0]);
    let ts = curve_shape_tsrvf(&km, &set(&[posed])).unwrap();
    let diff = ts
        .tangent_vectors
        .row(0)
        .iter()
        .zip(km.shooting_vectors.row(3))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-8, "tangent of posed copy differs by {diff}");
    // v = 0 maps to the mean; v_i maps back to a curve of the shape of curve i
    let zero = vec![0.0; 2 * m];
    let mean_curve = curve_shape_tangent_to_curve(&km, &zero).unwrap();
    let e: f64 = mean_curve
        .as_slice()
        .iter()
        .zip(km.mean.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(e < 1e-12, "v = 0 does not give the mean ({e})");
    let back = curve_shape_tangent_to_curve(&km, &ts.tangent_vectors.row(0)).unwrap();
    let d_back = curve_shape_distance(&back, &curves[3], &cfg).unwrap();
    assert!(
        d_back < 0.1 * km.distances[3].max(1e-3),
        "reconstruction off by {d_back}"
    );
}

#[test]
fn tsrvf_rejects_mismatched_input() {
    let m = 30;
    let curves: Vec<FdMatrix> = (0..3).map(|i| family(m, i)).collect();
    let km =
        curve_shape_karcher_mean(&set(&curves), &CurveShapeConfig::default(), 5, 1e-3).unwrap();
    let other = set(&[family(m + 1, 0)]);
    assert!(curve_shape_tsrvf(&km, &other).is_err());
    assert!(curve_shape_tangent_to_curve(&km, &[0.0; 3]).is_err());
    let mut bad = vec![0.0; 2 * m];
    bad[0] = f64::NAN;
    assert!(curve_shape_tangent_to_curve(&km, &bad).is_err());
}

// ─── Depth ──────────────────────────────────────────────────────────────────

/// fdasrvf 2.5.0 `curve_depth(beta, mode = "O", rotated = FALSE, scale = TRUE)`
/// on the curves of `golden_curve_depth.R` at T = 480 (its finest resolution).
const R_AMP_480: [f64; 6] = [
    0.8092339249,
    0.8163734744,
    0.8591319212,
    0.8328248666,
    0.8283027796,
    0.6454621310,
];
const R_PHASE_480: [f64; 6] = [
    0.8482157195,
    0.8546421656,
    0.8857884294,
    0.8661167105,
    0.8389484578,
    0.6759792606,
];

fn golden_depth_curves(m: usize) -> Vec<FdMatrix> {
    let a = [0.20, 0.25, 0.30, 0.35, 0.40, 0.60];
    let c = [0.05, -0.03, 0.02, 0.04, -0.05, 0.30];
    let p = [1.0, 1.2, 0.9, 1.1, 1.0, 1.5];
    (0..6).map(|i| bump(m, a[i], c[i], p[i])).collect()
}

fn ranking(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    idx
}

#[test]
fn golden_depth_fdasrvf() {
    let mut cfg = CurveShapeConfig::default();
    cfg.rotation = false;
    let res = curve_shape_depth(&set(&golden_depth_curves(60)), &cfg).unwrap();
    // fdasrvf discretises differently (Riemann inner product Σ/T, its own SRVF
    // gradient and DPQ2 grid) and its own depths drift with the resolution: from
    // T = 60 to 480 by up to 9.2e-3 (amplitude) and 1.1e-2 (phase), from T = 240
    // to 480 by 1.5e-3 / 3.5e-3. fdars at T = 60 is compared with fdasrvf at
    // T = 480 within 3e-3 (amplitude); observed max deviation 2.1e-3.
    for (ours, r) in res.amplitude_depth.iter().zip(R_AMP_480) {
        assert!(
            (ours - r).abs() < 3e-3,
            "amplitude depth {ours} vs fdasrvf {r}"
        );
    }
    assert_eq!(ranking(&res.amplitude_depth), ranking(&R_AMP_480));
    // Phase: fdasrvf takes the norm of `log_1(√γ')` without renormalising √γ'
    // and its DP warps differ in detail, so the phase depths agree only to
    // ~2e-2 (observed: max deviation 2.2e-2, on the outlying curve 5 — twice
    // fdasrvf's own T-drift). The ordering of all six curves agrees exactly.
    assert_eq!(ranking(&res.phase_depth), ranking(&R_PHASE_480));
}

#[test]
fn depth_is_invariant_to_poses_of_all_curves() {
    let m = 40;
    let curves = golden_depth_curves(m);
    let cfg = CurveShapeConfig::default();
    let base = curve_shape_depth(&set(&curves), &cfg).unwrap();
    let posed: Vec<FdMatrix> = curves
        .iter()
        .enumerate()
        .map(|(i, c)| {
            rigid(
                c,
                0.7 * i as f64 - 1.0,
                0.5 + 0.4 * i as f64,
                [i as f64, 2.0],
            )
        })
        .collect();
    let moved = curve_shape_depth(&set(&posed), &cfg).unwrap();
    for (a, b) in base.amplitude_depth.iter().zip(&moved.amplitude_depth) {
        assert!((a - b).abs() < 1e-8, "amplitude depth {a} vs {b}");
    }
    for (a, b) in base.phase_depth.iter().zip(&moved.phase_depth) {
        assert!((a - b).abs() < 1e-8, "phase depth {a} vs {b}");
    }
    // reparametrising every curve changes depth only at the DP resolution
    let warped: Vec<FdMatrix> = curves
        .iter()
        .enumerate()
        .map(|(i, c)| reparam(c, 0.4 * ((i % 3) as f64 - 1.0)))
        .collect();
    let rw = curve_shape_depth(&set(&warped), &cfg).unwrap();
    for (a, b) in base.amplitude_depth.iter().zip(&rw.amplitude_depth) {
        assert!(
            (a - b).abs() < 1e-2,
            "reparametrised amplitude depth {a} vs {b}"
        );
    }
    assert_eq!(argmax(&base.amplitude_depth), argmax(&rw.amplitude_depth));
}

#[test]
fn depth_extremes_and_ranges() {
    let m = 40;
    let mut curves: Vec<FdMatrix> = (0..7).map(|i| family(m, i)).collect();
    // curve 1 traversed with a strongly non-uniform speed: the phase outlier
    curves[1] = reparam(&curves[1], 0.9);
    let cfg = CurveShapeConfig::default();
    let res = curve_shape_depth(&set(&curves), &cfg).unwrap();
    assert_eq!(argmin(&res.phase_depth), 1, "{:?}", res.phase_depth);
    // a zig-zag curve is the amplitude (shape) outlier
    curves[6] = zigzag(m);
    let res = curve_shape_depth(&set(&curves), &cfg).unwrap();
    assert_eq!(argmin(&res.amplitude_depth), 6, "{:?}", res.amplitude_depth);
    // centre of the remaining family (curves 0..5, family index 3 central)
    assert_eq!(argmax(&res.amplitude_depth[..6]), 3);
    for i in 0..7 {
        assert!(res.amplitude_depth[i] > 0.0 && res.amplitude_depth[i] <= 1.0);
        assert!((0.0..=1.0).contains(&res.phase_depth[i]));
        assert_eq!(res.amplitude_distances[(i, i)], 0.0);
        for j in 0..7 {
            assert_eq!(
                res.amplitude_distances[(i, j)],
                res.amplitude_distances[(j, i)]
            );
            assert!(res.phase_distances[(i, j)] <= PI / 2.0 + 1e-12);
        }
    }
}

#[test]
fn depth_hand_computed_formula() {
    // three curves: the R-style median of each row (diagonal 0 included) is the
    // middle value of {0, d_ij, d_ik}
    let m = 30;
    let curves: Vec<FdMatrix> = (0..3).map(|i| family(m, 2 * i)).collect();
    let res = curve_shape_depth(&set(&curves), &CurveShapeConfig::default()).unwrap();
    let dist = &res.amplitude_distances;
    let px = &res.phase_distances;
    for i in 0..3 {
        let others: Vec<usize> = (0..3).filter(|&j| j != i).collect();
        let med_a = dist[(i, others[0])].min(dist[(i, others[1])]);
        let med_p = px[(i, others[0])].min(px[(i, others[1])]);
        assert!((res.amplitude_depth[i] - 1.0 / (1.0 + med_a)).abs() < 1e-15);
        let p = 1.0 / (1.0 + med_p);
        let want = (2.0 + PI) / PI * (p - 2.0 / (2.0 + PI));
        assert!((res.phase_depth[i] - want).abs() < 1e-15);
    }
}

// ─── Outliers ───────────────────────────────────────────────────────────────

#[test]
fn planted_outlier_is_flagged_in_any_pose() {
    let m = 40;
    let mut curves: Vec<FdMatrix> = (0..9)
        .map(|i| {
            rigid(
                &family(m, i % 5),
                0.2 * i as f64,
                1.0 + 0.1 * i as f64,
                [0.0; 2],
            )
        })
        .collect();
    curves[4] = rigid(&zigzag(m), 2.0, 3.0, [5.0, 5.0]);
    for centre in [CurveShapeCentre::Median, CurveShapeCentre::Mean] {
        let mut cfg = CurveShapeOutlierConfig::default();
        cfg.centre = centre;
        let res = curve_shape_outlier_detection(&set(&curves), &cfg).unwrap();
        assert_eq!(
            res.outlier_indices,
            vec![4],
            "{centre:?}: {:?}",
            res.distances
        );
        assert!(res.distances[4] > res.threshold);
        assert_eq!(res.phase_distances.len(), 9);
    }
    let mut q = CurveShapeOutlierConfig::default();
    q.threshold_method = OutlierThreshold::Quantile;
    q.alpha = 0.1;
    let res = curve_shape_outlier_detection(&set(&curves), &q).unwrap();
    assert!(res.outlier_indices.contains(&4));
}

#[test]
fn closed_contour_outlier() {
    let m = 41;
    let mut curves: Vec<FdMatrix> = (0..8)
        .map(|i| {
            rigid(
                &blob(m, 0.1 + 0.01 * i as f64, 0.05),
                0.5 * i as f64,
                1.0,
                [0.0; 2],
            )
        })
        .collect();
    curves[2] = blob(m, 0.0, 0.6); // a strongly lopsided contour
    let mut cfg = CurveShapeOutlierConfig::default();
    cfg.shape = closed_cfg();
    cfg.shape.seed_step = 8;
    let res = curve_shape_outlier_detection(&set(&curves), &cfg).unwrap();
    assert_eq!(res.outlier_indices, vec![2], "{:?}", res.distances);
}

#[test]
fn outlier_config_validation() {
    let m = 30;
    let curves: Vec<FdMatrix> = (0..4).map(|i| family(m, i)).collect();
    let mut cfg = CurveShapeOutlierConfig::default();
    cfg.threshold_method = OutlierThreshold::TukeyFence { k: -1.0 };
    assert!(curve_shape_outlier_detection(&set(&curves), &cfg).is_err());
    let mut cfg = CurveShapeOutlierConfig::default();
    cfg.threshold_method = OutlierThreshold::Quantile;
    cfg.alpha = 1.5;
    assert!(curve_shape_outlier_detection(&set(&curves), &cfg).is_err());
    assert!(curve_shape_depth(&set(&curves[..1]), &CurveShapeConfig::default()).is_err());
}

// ─── Box plot ───────────────────────────────────────────────────────────────

/// Analytic "aligned" SRVF fields of `golden_curve_boxplot.R`.
fn golden_box_fields(m: usize) -> (FdMatrix, Vec<FdMatrix>) {
    let grid = Grid::new(m);
    let unit = |q: FdMatrix| geometry::scaled(&q, 1.0 / geometry::norm(&q, &grid));
    let a = [
        0.10, -0.15, 0.20, -0.05, 0.30, -0.25, 0.05, 0.12, -0.18, 1.6, -1.1,
    ];
    let b = [
        0.05, 0.10, -0.12, 0.20, -0.08, 0.02, -0.22, 0.15, 0.07, 0.9, -1.3,
    ];
    let t = &grid.t;
    let qs = (0..a.len())
        .map(|i| {
            unit(from_rows(&[
                t.iter().map(|&t| 1.0 + a[i] * (PI * t).sin()).collect(),
                t.iter()
                    .map(|&t| b[i] * (PI * t).cos() + 0.1 * a[i] * t)
                    .collect(),
            ]))
        })
        .collect();
    let mu = unit(from_rows(&[
        vec![1.0; m],
        t.iter().map(|&t| 0.05 * (2.0 * PI * t).sin()).collect(),
    ]));
    (mu, qs)
}

#[test]
fn golden_boxplot_fdasrvf_curvebox_data() {
    // fdasrvf 2.5.0 fdasrvf:::curvebox_data(alpha = 0.05, ka = 1) with a trapezoidal
    // innerprod_q2 (see golden_curve_boxplot.R / .out); indices converted to 0-based.
    const DIST_SCALE: [f64; 11] = [
        3.494052433263952e-02,
        7.664353833539146e-02,
        1.245611533530653e-01,
        1.193076647672048e-01,
        1.169869921958534e-01,
        9.538373872062494e-02,
        1.824039607603079e-01,
        7.627160262322323e-02,
        7.387188496623143e-02,
        3.454294189371993e-01,
        1.292544200279526e+00,
    ];
    const DIST_FLAT: [f64; 11] = [
        3.493874699317578e-02,
        7.662478046600033e-02,
        1.244806428775183e-01,
        1.192369163838911e-01,
        1.169202919880810e-01,
        9.534758421797571e-02,
        1.821511992561578e-01,
        7.625311650213797e-02,
        7.385508932162904e-02,
        3.437145973454733e-01,
        1.204428973447688e+00,
    ];
    let m = 21;
    let (mu, qs) = golden_box_fields(m);
    let qset = set(&qs);
    for (scale, golden) in [(true, DIST_SCALE), (false, DIST_FLAT)] {
        let bp = curve_shape_boxplot_from_aligned(
            &mu,
            &qset,
            &qset,
            scale,
            &CurveShapeBoxplotConfig::default(),
        )
        .unwrap();
        for (ours, r) in bp.distances.iter().zip(golden) {
            assert!((ours - r).abs() < 1e-12, "scale {scale}: {ours} vs {r}");
        }
        assert_eq!(
            [bp.q1_index, bp.q3_index, bp.q1a_index, bp.q3a_index],
            [1, 4, 6, 9],
            "scale {scale}"
        );
        assert_eq!(bp.outlier_indices, vec![10, 9]);
        assert_eq!((bp.min_index, bp.max_index), (3, 2));
        assert!(bp.whisker_distance < bp.distances[9]);
        assert_eq!(bp.q3, qs[4]);
        assert!(bp.r50 <= bp.ralpha);
    }
}

#[test]
fn boxplot_from_median_flags_planted_outlier() {
    let m = 40;
    let mut curves: Vec<FdMatrix> = (0..11).map(|i| family(m, i % 6)).collect();
    curves[7] = zigzag(m);
    let cfg = CurveShapeConfig::default();
    let med = curve_shape_karcher_median(&set(&curves), &cfg, 20, 1e-4).unwrap();
    let bp = curve_shape_boxplot(&med, &CurveShapeBoxplotConfig::default()).unwrap();
    assert_eq!(bp.outlier_indices, vec![7], "{:?}", bp.distances);
    assert!(bp.r50 <= bp.ralpha && bp.whisker_distance >= bp.r50);
    assert_ne!(bp.q1_index, bp.q3_index);
    assert_eq!(bp.median.shape(), (2, m));
    let mut bad = CurveShapeBoxplotConfig::default();
    bad.alpha = 0.0;
    assert!(curve_shape_boxplot(&med, &bad).is_err());
}

// ─── Bootstrap CI ───────────────────────────────────────────────────────────

#[test]
fn bootstrap_ci_covers_true_shape_and_is_reproducible() {
    let m = 30;
    // symmetric perturbations of the template amplitude 0.3 (true shape),
    // posed randomly
    let deltas = [
        -0.06, 0.05, -0.02, 0.03, 0.06, -0.04, 0.01, -0.05, 0.04, -0.01, 0.02, -0.03,
    ];
    let curves: Vec<FdMatrix> = deltas
        .iter()
        .enumerate()
        .map(|(i, &dl)| {
            let c = bump(m, 0.3 + dl, 0.02 * ((i % 3) as f64 - 1.0), 1.0);
            rigid(&c, 0.3 * i as f64, 1.0 + 0.05 * i as f64, [0.0, i as f64])
        })
        .collect();
    let truth = bump(m, 0.3, 0.0, 1.0);
    let shape = CurveShapeConfig::default();
    let mut ci = CurveShapeCiConfig::default();
    ci.n_bootstrap = 40;
    ci.seed = 7;
    let res = curve_shape_confidence_interval(&set(&curves), &shape, &ci).unwrap();
    assert_eq!(res.bootstrap_means.len(), 40);
    let d_truth = curve_shape_distance(&res.mean.mean, &truth, &shape).unwrap();
    assert!(
        d_truth <= res.radius,
        "true shape at {d_truth} outside the radius {}",
        res.radius
    );
    for k in 0..2 {
        for j in 0..m {
            assert!(res.lower_band[(k, j)] <= res.upper_band[(k, j)]);
        }
    }
    let again = curve_shape_confidence_interval(&set(&curves), &shape, &ci).unwrap();
    assert_eq!(res, again);
    let mut bad = ci.clone();
    bad.confidence_level = 1.0;
    assert!(curve_shape_confidence_interval(&set(&curves), &shape, &bad).is_err());
}
