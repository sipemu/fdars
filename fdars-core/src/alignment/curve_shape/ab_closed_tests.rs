//! Tests of closed planar contours and the branch search under the general
//! (a,b) elastic metric (GH #68).
//!
//! geomstats 2.8.0 (the reference of the open-curve (a,b) tests) has no
//! closed-curve support for `ElasticMetric` (no closure condition, projection
//! or seed search in `geometry/discrete_curves.py`), so the closed case is
//! validated by exact identities, finite differences and invariants.

use super::geometry::{self, Grid};
use super::*;
use std::f64::consts::PI;

/// `(a, b)` pairs with λ = a/(2b) ∈ {1/4, 1/2, 3/2, 2}.
const AB_GENERAL: [(f64, f64); 4] = [(1.0, 2.0), (1.0, 1.0), (3.0, 1.0), (2.0, 0.5)];

fn sample(m: usize, f: impl Fn(f64) -> (f64, f64)) -> FdMatrix {
    let mut out = FdMatrix::zeros(2, m);
    for j in 0..m {
        let (x, y) = f(j as f64 / (m - 1) as f64);
        out[(0, j)] = x;
        out[(1, j)] = y;
    }
    out
}

/// Polar closed contour `r(θ)`, sampled with the first point repeated.
fn polar(m: usize, r: impl Fn(f64) -> f64) -> FdMatrix {
    sample(m, |t| {
        let th = 2.0 * PI * t;
        (r(th) * th.cos(), r(th) * th.sin())
    })
}

fn ellipse(m: usize) -> FdMatrix {
    sample(m, |t| {
        let th = 2.0 * PI * t;
        (1.5 * th.cos(), 0.8 * th.sin())
    })
}

fn trefoil_bump(m: usize) -> FdMatrix {
    polar(m, |a| 1.0 + 0.2 * (3.0 * a).cos())
}

/// No rotational symmetry (unique optimal seed).
fn asym(m: usize) -> FdMatrix {
    polar(m, |a| 1.0 + 0.2 * (3.0 * a).cos() + 0.1 * a.sin())
}

fn arc(m: usize) -> FdMatrix {
    sample(m, |t| {
        let th = 1.5 * PI * t;
        (th.cos(), th.sin())
    })
}

fn elastic(a: f64, b: f64) -> CurveShapeMetric {
    CurveShapeMetric::Elastic { a, b }
}

fn closed_cfg(metric: CurveShapeMetric) -> CurveShapeConfig {
    CurveShapeConfig {
        closed: true,
        metric,
        ..CurveShapeConfig::default()
    }
}

fn rot2(a: f64) -> FdMatrix {
    let (s, c) = a.sin_cos();
    let mut r = FdMatrix::zeros(2, 2);
    r[(0, 0)] = c;
    r[(1, 0)] = s;
    r[(0, 1)] = -s;
    r[(1, 1)] = c;
    r
}

/// `s · R β + c`.
fn rigid(beta: &FdMatrix, r: &FdMatrix, s: f64, c: [f64; 2]) -> FdMatrix {
    let mut out = geometry::scaled(&geometry::rotate(r, beta), s);
    for j in 0..out.ncols() {
        out[(0, j)] += c[0];
        out[(1, j)] += c[1];
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

fn res_norm(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum::<f64>().sqrt()
}

// ─── Closure functional and its gradient ────────────────────────────────────

#[test]
fn closure_residual_is_the_rebuilt_displacement() {
    let m = 81;
    let c = sample(m, |t| (t + 0.3 * t * t, 0.4 * (4.0 * t).sin()));
    for &(a, b) in &AB_GENERAL {
        let f = ab_transform(&c, a, b).unwrap();
        let rec = ab_transform_inverse(&f, a, b).unwrap();
        let r = ab_closure_residual(&f, a, b).unwrap();
        for k in 0..2 {
            let disp = rec[(k, m - 1)] - rec[(k, 0)];
            assert!(
                (r[k] - disp).abs() < 1e-12,
                "a={a} b={b}: {} vs {disp}",
                r[k]
            );
        }
    }
    // a = 1, b = 1/2: the SRVF closure residual
    let q = curve_srvf(&c, false).unwrap();
    let r1 = ab_closure_residual(&q, 1.0, 0.5).unwrap();
    let r0 = closure_residual(&q).unwrap();
    assert!((r1[0] - r0[0]).abs() < 1e-12 && (r1[1] - r0[1]).abs() < 1e-12);
}

#[test]
fn closure_normals_are_the_residual_gradients() {
    // central finite differences of G_λ along smooth directions vs ⟨b_k, δ⟩
    let m = 101;
    let grid = Grid::new(m);
    let c = sample(m, |t| (t + 0.2 * t * t, 0.3 * (3.0 * t).sin()));
    let dirs = [
        sample(m, |t| ((2.0 * t).cos(), 0.5 * t)),
        sample(m, |t| (0.3 * (5.0 * t).sin(), 1.0 - t * t)),
    ];
    for lam in [0.25, 0.5, 1.0, 1.5, 2.0] {
        let f = geometry::f_transform(&c, &grid, lam, 0.5);
        let normals = geometry::closure_normals_ab(&f, lam);
        for dir in &dirs {
            let eps = 1e-6;
            let gp = geometry::closure_vec_ab(&geometry::axpby(1.0, &f, eps, dir), &grid, lam);
            let gm = geometry::closure_vec_ab(&geometry::axpby(1.0, &f, -eps, dir), &grid, lam);
            for k in 0..2 {
                let fd = (gp[k] - gm[k]) / (2.0 * eps);
                let an = geometry::inner(&normals[k], dir, &grid);
                assert!(
                    (fd - an).abs() < 1e-7 * (1.0 + an.abs()),
                    "λ={lam} k={k}: fd {fd} vs {an}"
                );
            }
        }
    }
}

#[test]
fn lambda_one_functional_and_normals_equal_the_srvf_ones() {
    let m = 61;
    let grid = Grid::new(m);
    let q = curve_srvf(&arc(m), true).unwrap();
    let g_ab = geometry::closure_vec_ab(&q, &grid, 1.0);
    let g = geometry::closure_vec(&q, &grid);
    assert!((g_ab[0] - g[0]).abs() < 1e-13 && (g_ab[1] - g[1]).abs() < 1e-13);
    for (x, y) in geometry::closure_normals_ab(&q, 1.0)
        .iter()
        .zip(&geometry::closure_normals(&q))
    {
        assert!(max_abs_diff(x, y) < 1e-12);
    }
}

// ─── Projection ─────────────────────────────────────────────────────────────

#[test]
fn projection_closes_is_idempotent_and_keeps_the_norm() {
    let m = 101;
    let grid = Grid::new(m);
    let opens = [
        arc(m),
        sample(m, |t| (t, 0.3 * (2.0 * PI * t).sin())),
        sample(m, |t| {
            ((1.2 * t).cos() + t, 0.5 * t * t - 0.2 * (5.0 * t).sin())
        }),
    ];
    for &(a, b) in &AB_GENERAL {
        for c in &opens {
            let f = ab_transform(c, a, b).unwrap();
            assert!(res_norm(&ab_closure_residual(&f, a, b).unwrap()) > 1e-2);
            let fc = ab_project_closed(&f, a, b).unwrap();
            let r = ab_closure_residual(&fc, a, b).unwrap();
            let n0 = geometry::norm(&f, &grid);
            assert!(res_norm(&r) < 1e-9 * n0 * n0, "a={a} b={b}: {r:?}");
            assert!((geometry::norm(&fc, &grid) - n0).abs() < 1e-12 * n0);
            let rec = ab_transform_inverse(&fc, a, b).unwrap();
            for k in 0..2 {
                assert!((rec[(k, m - 1)] - rec[(k, 0)]).abs() < 1e-9);
            }
            let fcc = ab_project_closed(&fc, a, b).unwrap();
            assert!(
                max_abs_diff(&fc, &fcc) < 1e-12,
                "not idempotent a={a} b={b}"
            );
        }
        // an already (almost) closed contour barely moves
        let f = ab_transform(&ellipse(m), a, b).unwrap();
        let fc = ab_project_closed(&f, a, b).unwrap();
        let rel =
            geometry::norm(&geometry::axpby(1.0, &f, -1.0, &fc), &grid) / geometry::norm(&f, &grid);
        assert!(rel < 1e-3, "a={a} b={b}: moved {rel}");
    }
}

#[test]
fn projection_at_lambda_one_is_the_srvf_projector() {
    let m = 101;
    let grid = Grid::new(m);
    let q = curve_srvf(&arc(m), true).unwrap();
    // public API: bit-identical dispatch for λ = 1 (any b)
    let p_srvf = project_closed_srvf(&q).unwrap();
    assert_eq!(ab_project_closed(&q, 1.0, 0.5).unwrap(), p_srvf);
    assert_eq!(
        ab_project_closed(&geometry::scaled(&q, 3.0), 3.0, 1.5).unwrap(),
        project_closed_srvf(&geometry::scaled(&q, 3.0)).unwrap()
    );
    // the general Gauss–Newton path at λ = 1 also lands on the SRVF closed
    // manifold; for a nearly closed contour both projections are O(residual)
    // moves and agree to second order
    let (p_gn, ok) = geometry::project_closed_ab(&q, &grid, 1.0);
    assert!(ok && res_norm(&geometry::closure_vec(&p_gn, &grid)) < 1e-9);
    let qe = curve_srvf(&ellipse(m), true).unwrap();
    let move_ = |p: &FdMatrix| geometry::norm(&geometry::axpby(1.0, &qe, -1.0, p), &grid);
    let (pe_gn, _) = geometry::project_closed_ab(&qe, &grid, 1.0);
    let pe = project_closed_srvf(&qe).unwrap();
    let diff = geometry::norm(&geometry::axpby(1.0, &pe_gn, -1.0, &pe), &grid);
    assert!(diff < 0.05 * move_(&pe), "{diff} vs move {}", move_(&pe));
}

// ─── Alignment of closed contours ───────────────────────────────────────────

#[test]
fn elastic_one_half_closed_reproduces_srvf_closed() {
    let m = 61;
    // shapes without rotational symmetry (no near-tied seeds)
    let c1 = asym(m);
    let c2 = polar(m, |a| 1.0 + 0.15 * (2.0 * a).cos() + 0.1 * (a + 0.4).sin());
    for reparam in [false, true] {
        let mut s = closed_cfg(CurveShapeMetric::Srvf);
        s.reparam = reparam;
        let mut e = s.clone();
        e.metric = elastic(1.0, 0.5);
        let rs = curve_shape_align_pair(&c1, &c2, &s).unwrap();
        let re = curve_shape_align_pair(&c1, &c2, &e).unwrap();
        assert_eq!(rs.seed, re.seed);
        assert!((rs.distance - re.distance).abs() < 1e-10);
        assert!(max_abs_diff(&rs.rotation, &re.rotation) < 1e-10);
        let dg = rs
            .gamma
            .iter()
            .zip(&re.gamma)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(dg < 1e-10);
        assert!(max_abs_diff(&rs.beta2_aligned, &re.beta2_aligned) < 1e-10);
        assert!(max_abs_diff(&rs.q2_aligned, &re.q2_aligned) < 1e-10);
    }
}

#[test]
fn closed_seed_rotation_scale_recovered_exactly() {
    let m = 61;
    let b = asym(m);
    let b2 = rigid(&geometry::shift_closed(&b, 9), &rot2(1.3), 0.5, [2.0, 2.0]);
    for &(a, bb) in &AB_GENERAL {
        let mut cfg = closed_cfg(elastic(a, bb));
        cfg.reparam = false;
        let res = curve_shape_align_pair(&b, &b2, &cfg).unwrap();
        assert_eq!(res.seed, m - 1 - 9, "a={a} b={bb}");
        assert!(res.distance < 1e-9, "a={a} b={bb}: {}", res.distance);
        // outputs live on the closed manifold
        for q in [&res.q1, &res.q2_aligned] {
            let r = ab_closure_residual(q, a, bb).unwrap();
            assert!(res_norm(&r) < 1e-9, "{r:?}");
        }
        if a / (2.0 * bb) <= 1.0 {
            // F is injective modulo rotation for λ ≤ 1: the rotation is recovered
            assert!(max_abs_diff(&res.rotation, &rot2(-1.3)) < 1e-9);
        }
        cfg.seed_search = false;
        assert!(curve_shape_distance(&b, &b2, &cfg).unwrap() > 0.05);
    }
}

#[test]
fn closed_distance_invariant_to_seed_rotation_and_reparametrisation() {
    let m = 61;
    let grid = Grid::new(m);
    let t: Vec<f64> = grid.t.clone();
    let gamma: Vec<f64> = t.iter().map(|&t| t + 0.2 * t * (1.0 - t)).collect();
    let (b1, b2) = (ellipse(m), trefoil_bump(m));
    for &(a, b) in &[(1.0, 1.0), (1.0, 2.0), (3.0, 1.0)] {
        let cfg = closed_cfg(elastic(a, b));
        let d = curve_shape_distance(&b1, &b2, &cfg).unwrap();
        assert!(d > 0.1);
        let moved = rigid(
            &geometry::shift_closed(&geometry::warp_curve(&b2, &grid, &gamma), 17),
            &rot2(2.0),
            1.7,
            [0.3, 0.0],
        );
        let dm = curve_shape_distance(&b1, &moved, &cfg).unwrap();
        // the DP / sub-sample-seed floor of the SRVF test (0.015), scaled with λ
        assert!((dm - d).abs() < 0.03, "a={a} b={b}: {d} vs {dm}");
        // the same shape at another seed / rotation / parametrisation
        let self_moved = rigid(
            &geometry::shift_closed(&geometry::warp_curve(&b2, &grid, &gamma), 23),
            &rot2(-0.7),
            1.0,
            [0.0, 0.0],
        );
        let ds = curve_shape_distance(&b2, &self_moved, &cfg).unwrap();
        assert!(ds < 0.08 && ds < 0.3 * d, "a={a} b={b}: self {ds} vs {d}");
        // symmetric up to the DP floor
        let d21 = curve_shape_distance(&b2, &b1, &cfg).unwrap();
        assert!((d21 - d).abs() < 0.03, "{d} vs {d21}");
    }
}

#[test]
fn aligned_transforms_rebuild_closed_curves() {
    let m = 61;
    for &(a, b) in &AB_GENERAL {
        let cfg = closed_cfg(elastic(a, b));
        let res = curve_shape_align_pair(&ellipse(m), &trefoil_bump(m), &cfg).unwrap();
        for q in [&res.q1, &res.q2_aligned] {
            let rec = ab_transform_inverse(q, a, b).unwrap();
            for k in 0..2 {
                assert!((rec[(k, m - 1)] - rec[(k, 0)]).abs() < 1e-9, "a={a} b={b}");
            }
        }
    }
}

// ─── Karcher statistics of closed contours ──────────────────────────────────

#[test]
fn karcher_mean_of_closed_contours() {
    let m = 61;
    let grid = Grid::new(m);
    let curves: Vec<FdMatrix> = (0..3)
        .map(|i| {
            let gamma: Vec<f64> = grid
                .t
                .iter()
                .map(|&t| t + 0.1 * i as f64 * t * (1.0 - t))
                .collect();
            let b = geometry::warp_curve(&trefoil_bump(m), &grid, &gamma);
            rigid(
                &geometry::shift_closed(&b, 5 * i),
                &rot2(0.8 * i as f64),
                1.0 + i as f64,
                [0.0, 0.0],
            )
        })
        .collect();
    let set = curves_to_set(&curves).unwrap();
    for &(a, b) in &[(1.0, 1.0), (3.0, 1.0)] {
        let cfg = closed_cfg(elastic(a, b));
        let lam = a / (2.0 * b);
        let km = curve_shape_karcher_mean(&set, &cfg, 5, 1e-3).unwrap();
        let r = geometry::closure_vec_ab(&km.mean_srvf, &grid, lam);
        assert!(res_norm(&r) < 1e-9, "a={a} b={b}: {r:?}");
        assert!(km.distances.iter().all(|&d| d < 0.08), "{:?}", km.distances);
        let d = curve_shape_distance(&km.mean, &trefoil_bump(m), &cfg).unwrap();
        assert!(d < 0.08, "a={a} b={b}: mean-to-template {d}");
        // shooting vectors are tangent to the closed manifold and the sphere
        let normals = geometry::closure_normals_ab(&km.mean_srvf, lam);
        for i in 0..3 {
            let mut v = FdMatrix::zeros(2, m);
            for k in 0..2 {
                for j in 0..m {
                    v[(k, j)] = km.shooting_vectors[(i, k * m + j)];
                }
            }
            for nb in &normals {
                assert!(geometry::inner(&v, nb, &grid).abs() < 1e-10);
            }
            assert!(geometry::inner(&v, &km.mean_srvf, &grid).abs() < 1e-10);
        }
    }
}

#[test]
fn karcher_closed_elastic_one_half_matches_srvf() {
    let m = 41;
    let curves = vec![ellipse(m), trefoil_bump(m), asym(m)];
    let set = curves_to_set(&curves).unwrap();
    let s = curve_shape_karcher_mean(&set, &closed_cfg(CurveShapeMetric::Srvf), 4, 1e-6).unwrap();
    let e = curve_shape_karcher_mean(&set, &closed_cfg(elastic(1.0, 0.5)), 4, 1e-6).unwrap();
    assert_eq!(s.n_iter, e.n_iter);
    assert_eq!(s.seeds, e.seeds);
    assert!(max_abs_diff(&s.mean, &e.mean) < 1e-8);
    for (x, y) in s.distances.iter().zip(&e.distances) {
        assert!((x - y).abs() < 1e-8);
    }
}

#[test]
fn downstream_runs_on_closed_elastic_contours() {
    let m = 41;
    let curves = vec![
        ellipse(m),
        trefoil_bump(m),
        asym(m),
        polar(m, |a| 1.0 + 0.15 * (2.0 * a).sin()),
    ];
    let set = curves_to_set(&curves).unwrap();
    let cfg = closed_cfg(elastic(1.0, 1.0));
    let km = curve_shape_karcher_mean(&set, &cfg, 4, 1e-4).unwrap();
    let pca = curve_shape_pca(&km, 2).unwrap();
    let mode = pca.mode_shape(&km, 0, 1.0).unwrap();
    // a mode shape is rebuilt from a point of the closed manifold
    for k in 0..2 {
        assert!((mode[(k, m - 1)] - mode[(k, 0)]).abs() < 1e-8);
    }
    let ts = curve_shape_tsrvf(&km, &set).unwrap();
    let back = curve_shape_tangent_to_curve(&km, &ts.tangent_vectors.row(0)).unwrap();
    assert!(curve_shape_distance(&back, &curves[0], &cfg).unwrap() < 0.1);
    let depth = curve_shape_depth(&set, &cfg).unwrap();
    assert!(depth.amplitude_depth.iter().all(|&v| v > 0.0 && v <= 1.0));
}

// ─── Branch search (rotation = false) ───────────────────────────────────────

#[test]
fn branch_search_minimises_over_the_f_angle_ambiguity() {
    let m = 61;
    // initial tangents straddling the branch cut at ±π
    let c1 = sample(m, |t| (-t, 0.05 * t + 0.05 * (4.0 * t).sin()));
    let c2 = sample(m, |t| {
        (-t - 0.1 * t * t, -0.05 * t - 0.04 * (3.0 * t).sin())
    });
    for &(a, b) in &AB_GENERAL {
        let mut cfg = CurveShapeConfig {
            metric: elastic(a, b),
            rotation: false,
            reparam: false,
            ..CurveShapeConfig::default()
        };
        let d0 = curve_shape_distance(&c1, &c2, &cfg).unwrap();
        cfg.branch_search = true;
        let res = curve_shape_align_pair(&c1, &c2, &cfg).unwrap();
        // k = 0 is a candidate: never worse
        assert!(res.distance <= d0 + 1e-12, "a={a} b={b}");
        // the curve itself is not rotated
        assert_eq!(res.rotation, geometry::identity(2));
        assert!(
            max_abs_diff(
                &res.beta2_aligned,
                &geometry::center(
                    &geometry::scaled(&c2, 1.0 / geometry::curve_length(&c2, &Grid::new(m))),
                    &Grid::new(m),
                )
            ) < 1e-12
        );
        // the best relative branch is the one aligning the initial tangents:
        // the same distance as for the curves rotated by π (no branch cut)
        let mut plain = cfg.clone();
        plain.branch_search = false;
        let flip = rot2(PI);
        let dr = curve_shape_distance(
            &rigid(&c1, &flip, 1.0, [0.0; 2]),
            &rigid(&c2, &flip, 1.0, [0.0; 2]),
            &plain,
        )
        .unwrap();
        assert!(
            (res.distance - dr).abs() < 1e-9,
            "a={a} b={b}: {} vs {dr}",
            res.distance
        );
        let lam = a / (2.0 * b);
        if (lam - lam.round()).abs() > 1e-12 {
            assert!(
                d0 > res.distance + 0.1,
                "a={a} b={b}: {d0} vs {}",
                res.distance
            );
        }
    }
    // ignored for the SRVF and with rotation = true
    for metric in [CurveShapeMetric::Srvf, elastic(1.0, 1.0)] {
        let mut cfg = CurveShapeConfig {
            metric,
            ..CurveShapeConfig::default()
        };
        if metric == CurveShapeMetric::Srvf {
            cfg.rotation = false;
        }
        let r0 = curve_shape_align_pair(&c1, &c2, &cfg).unwrap();
        cfg.branch_search = true;
        let r1 = curve_shape_align_pair(&c1, &c2, &cfg).unwrap();
        assert_eq!(r0, r1);
    }
}

#[test]
fn closed_elastic_validation() {
    let m = 41;
    let mut cfg = closed_cfg(elastic(1.0, 1.0));
    assert!(cfg.validate().is_ok());
    assert!(curve_shape_distance(&ellipse(m), &asym(m), &cfg).is_ok());
    cfg.metric = elastic(1.0, 0.0);
    assert!(cfg.validate().is_err());
    let f = ab_transform(&ellipse(m), 1.0, 1.0).unwrap();
    assert!(ab_closure_residual(&f, 0.0, 1.0).is_err());
    assert!(ab_project_closed(&FdMatrix::zeros(3, m), 1.0, 1.0).is_err());
    assert!(ab_closure_residual(&FdMatrix::zeros(2, 2), 1.0, 1.0).is_err());
}
