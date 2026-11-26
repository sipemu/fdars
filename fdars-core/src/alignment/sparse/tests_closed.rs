//! Tests for closed sparse elastic means (goldens: R elasdics 1.1.3,
//! `compute_elastic_mean(closed = TRUE)`, `%.17g` verbatim).
#![allow(clippy::excessive_precision, clippy::field_reassign_with_default)]

use super::closed_mean::rotate_srv;
use super::discrete::compute_distance;
use super::golden::SPIRAL;
use super::golden_closed::*;
use super::polygon::SrvPoly;
use super::smooth::{find_optimal_t_smooth, smooth_distance, LinSpline};
use super::tests::{
    assert_vec_close, curve2d, interleave, knots, random_grid, rel_l2, spiral_curves, srv_of,
};
use super::*;
use std::f64::consts::PI;

fn shape_curves() -> Vec<SparseCurve> {
    SHAPES_X1
        .iter()
        .zip(SHAPES_X2.iter())
        .map(|(a, b)| SparseCurve::new(curve2d(a, b)))
        .collect()
}

fn closed_cfg(nk: usize, mean_type: SparseMeanType) -> SparseElasticMeanConfig {
    let mut cfg = SparseElasticMeanConfig::default();
    cfg.knots = knots(nk);
    cfg.mean_type = mean_type;
    cfg.closed = true;
    cfg
}

fn closed_dist(a: &SparseCurve, b: &SparseCurve) -> f64 {
    let mut cfg = SparseAlignConfig::default();
    cfg.closed = true;
    sparse_elastic_align_pair(a, b, &cfg).unwrap().distance
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// R's `t_optim` (original point order, closing point repeated) back in the
/// rotated order used for the alignment, ending at 1.
fn rerotate(r_topt: &[f64], s0: usize) -> Vec<f64> {
    let n = r_topt.len() - 1;
    let mut t: Vec<f64> = (0..n).map(|k| r_topt[(k + s0) % n]).collect();
    t.push(1.0);
    t
}

/// The point R maps to parameter 0 is the chosen starting point.
fn r_start_indices(r_topt: &[&[f64]]) -> Vec<usize> {
    r_topt
        .iter()
        .map(|t| t[..t.len() - 1].iter().position(|&v| v == 0.0).unwrap())
        .collect()
}

struct ClosedGolden<'a> {
    coef_x1: &'a [f64],
    coef_x2: &'a [f64],
    endpoint: &'a [f64],
    topt: &'a [&'a [f64]],
}

fn spiral_golden(mean_type: SparseMeanType) -> ClosedGolden<'static> {
    match mean_type {
        SparseMeanType::Smooth => ClosedGolden {
            coef_x1: &SPIRAL_SMOOTH_COEF_X1,
            coef_x2: &SPIRAL_SMOOTH_COEF_X2,
            endpoint: &SPIRAL_SMOOTH_ENDPOINT,
            topt: &SPIRAL_SMOOTH_TOPT,
        },
        SparseMeanType::Polygon => ClosedGolden {
            coef_x1: &SPIRAL_POLY_COEF_X1,
            coef_x2: &SPIRAL_POLY_COEF_X2,
            endpoint: &SPIRAL_POLY_ENDPOINT,
            topt: &SPIRAL_POLY_TOPT,
        },
    }
}

fn shapes_golden(mean_type: SparseMeanType) -> ClosedGolden<'static> {
    match mean_type {
        SparseMeanType::Smooth => ClosedGolden {
            coef_x1: &SHAPES_SMOOTH_COEF_X1,
            coef_x2: &SHAPES_SMOOTH_COEF_X2,
            endpoint: &SHAPES_SMOOTH_ENDPOINT,
            topt: &SHAPES_SMOOTH_TOPT,
        },
        SparseMeanType::Polygon => ClosedGolden {
            coef_x1: &SHAPES_POLY_COEF_X1,
            coef_x2: &SHAPES_POLY_COEF_X2,
            endpoint: &SHAPES_POLY_ENDPOINT,
            topt: &SHAPES_POLY_TOPT,
        },
    }
}

/// Distance of a rotated SRV at `t` to the closed mean with full coefficients `c`.
fn dist_to(mean_type: SparseMeanType, kn: &[f64], c: &[f64], srv: &SrvPoly, t: &[f64]) -> f64 {
    match mean_type {
        SparseMeanType::Smooth => smooth_distance(&LinSpline { d: 2, knots: kn, c }, srv, t),
        SparseMeanType::Polygon => {
            let mean = SrvPoly {
                d: 2,
                t: kn.to_vec(),
                q: c.to_vec(),
            };
            compute_distance(&mean, srv, t, false)
        }
    }
}

// ─── elasdics 1.1.3 goldens: compute_elastic_mean(closed = TRUE) ────────────

#[test]
fn golden_closed_polygon_means() {
    // R: compute_elastic_mean(data_curves, knots = seq(0, 1, length = 15),
    //    type = "polygon", closed = TRUE) (help-page spirals) and the closed shapes
    //    with 13 knots. The polygon aligner is deterministic, R's line search is
    //    Brent (ported) and the closure-penalty gradient is analytic here vs
    //    numDeriv there, so everything agrees to round-off.
    for (curves, nk, g) in [
        (spiral_curves(), 15, spiral_golden(SparseMeanType::Polygon)),
        (shape_curves(), 13, shapes_golden(SparseMeanType::Polygon)),
    ] {
        let res = sparse_elastic_mean(&curves, &closed_cfg(nk, SparseMeanType::Polygon)).unwrap();
        assert!(res.converged && res.closed);
        assert_eq!(res.coefs.shape(), (2, nk - 1));
        let r_coefs = interleave(g.coef_x1, g.coef_x2);
        assert_vec_close(res.coefs.as_slice(), &r_coefs, 1e-9, "coefs");
        assert_vec_close(&res.closure_residual(), g.endpoint, 1e-9, "closure");
        assert_eq!(res.start_indices, r_start_indices(g.topt));
        for (i, curve) in curves.iter().enumerate() {
            assert_vec_close(&res.params_optim[i], g.topt[i], 1e-9, "t_optim");
            // distance of R's solution, recomputed exactly, equals ours
            let rot = rotate_srv(&srv_of(curve, true), res.start_indices[i]);
            let t_r = rerotate(g.topt[i], res.start_indices[i]);
            let d_r = dist_to(SparseMeanType::Polygon, &res.knots, &r_coefs, &rot, &t_r);
            assert!((res.distances[i] - d_r).abs() < 1e-9);
        }
    }
}

#[test]
fn golden_closed_smooth_means() {
    // R: compute_elastic_mean(data_curves, knots = seq(0, 1, length = 11),
    //    closed = TRUE) (help-page example) and the closed shapes with 9 knots.
    // R's smooth alignment runs constrOptim with reltol = eps·100/i = 1, 0.5, ... on
    // integrate(rel.tol = 0.1) and converges after 1–2 iterations, so its warps are
    // only coarsely optimised (one shape curve is not moved at all) and its means
    // differ from ours by 8–9 % of the SRV norm. Well-defined comparisons:
    //  - the deterministic pre-alignment picks the same starting points;
    //  - on the exact objective (R's coefficients and warps re-scored by our exact
    //    integrator), our Fréchet objective is no worse than R's. (Re-aligning
    //    every curve optimally to R's mean can beat both: the eps = 0.01 stopping
    //    rule ends either fit at a different, not fully optimised point, so "which
    //    mean is better" is not a well-defined golden.);
    //  - per-curve distances obey the triangle inequality
    //    |d(q, μ) − d(q, μ_R)| ≤ d(μ, μ_R) (d(q, μ_R) with optimal warps);
    //  - sanity bound only: the mean curves are within 0.15·‖μ_R‖ (observed 8.2 %
    //    and 10.6 %); as R's warps are essentially left at the pre-alignment, a
    //    sharper mean-to-mean golden is not meaningful here — the sharp checks are
    //    the objective comparison above and the round-off-level polygon goldens;
    //  - both closure residuals are small relative to the curve length ‖μ‖².
    for (curves, nk, g) in [
        (spiral_curves(), 11, spiral_golden(SparseMeanType::Smooth)),
        (shape_curves(), 9, shapes_golden(SparseMeanType::Smooth)),
    ] {
        let res = sparse_elastic_mean(&curves, &closed_cfg(nk, SparseMeanType::Smooth)).unwrap();
        assert!(res.converged && res.closed);
        assert_eq!(res.coefs.shape(), (2, nk - 1));
        assert_eq!(res.start_indices, r_start_indices(g.topt));
        let r_coefs = interleave(g.coef_x1, g.coef_x2);
        let mut r_full = r_coefs.clone();
        r_full.extend_from_slice(&r_coefs[..2]);
        let mean_r = LinSpline {
            d: 2,
            knots: &res.knots,
            c: &r_full,
        };
        let norm_r = mean_r.norm_sq().sqrt();
        let mut obj_r = 0.0;
        let mut d_r_star = Vec::new();
        for (i, curve) in curves.iter().enumerate() {
            let rot = rotate_srv(&srv_of(curve, true), res.start_indices[i]);
            let t_r = rerotate(g.topt[i], res.start_indices[i]);
            let d_r = smooth_distance(&mean_r, &rot, &t_r);
            let t_star = find_optimal_t_smooth(&mean_r, &rot, &t_r, 1e-8);
            let d_star = smooth_distance(&mean_r, &rot, &t_star);
            assert!(d_star <= d_r + 1e-12);
            obj_r += d_r * d_r;
            d_r_star.push(d_star);
        }
        let obj: f64 = res.distances.iter().map(|x| x * x).sum();
        assert!(obj <= obj_r + 1e-9, "objective {obj} vs R {obj_r}");
        let grid = knots(801);
        let r_res = SparseElasticMeanResult {
            coefs: curve2d(g.coef_x1, g.coef_x2),
            ..res.clone()
        };
        let ours = SparseCurve::new(res.mean_curve(&grid, false).unwrap());
        let theirs = SparseCurve::new(r_res.mean_curve(&grid, false).unwrap());
        let d_means = closed_dist(&theirs, &ours);
        assert!(d_means < 0.15 * norm_r, "{d_means} vs ‖μ_R‖ {norm_r}");
        for (ours, theirs) in res.distances.iter().zip(&d_r_star) {
            assert!(
                (ours - theirs).abs() <= d_means + 1e-3,
                "{ours} vs {theirs}"
            );
        }
        assert!(rel_l2(res.coefs.as_slice(), &r_coefs) < 0.3);
        // elasdics' penalised fit closes the mean only approximately
        let len = norm_r * norm_r;
        assert!(norm(&res.closure_residual()) < 0.01 * len);
        assert!(norm(g.endpoint) < 0.01 * len);
    }
}

// ─── invariants ─────────────────────────────────────────────────────────────

/// Points of the closed planar shape at parameters `s` (in `[0, 1)`).
fn shape_at(s: &[f64]) -> FdMatrix {
    let x1: Vec<f64> = s
        .iter()
        .map(|t| (2.0 * PI * t).cos() + 0.3 * (4.0 * PI * t).cos())
        .collect();
    let x2: Vec<f64> = s.iter().map(|t| 0.6 * (2.0 * PI * t).sin()).collect();
    curve2d(&x1, &x2)
}

/// `m` irregular parameters of a closed curve, starting at `offset`.
fn closed_grid(m: usize, seed: u64, offset: f64) -> Vec<f64> {
    let g = random_grid(m + 1, seed);
    g[..m].iter().map(|v| (v + offset) % 1.0).collect()
}

#[test]
fn closure_steps_close_the_mean() {
    for mean_type in [SparseMeanType::Smooth, SparseMeanType::Polygon] {
        let nk = if mean_type == SparseMeanType::Smooth {
            9
        } else {
            13
        };
        let mut cfg = closed_cfg(nk, mean_type);
        let base = sparse_elastic_mean(&shape_curves(), &cfg).unwrap();
        cfg.closure_steps = 30;
        let res = sparse_elastic_mean(&shape_curves(), &cfg).unwrap();
        let e = norm(&res.closure_residual());
        assert!(e < 1e-10, "{mean_type:?}: closure residual {e}");
        assert!(e < norm(&base.closure_residual()));
        // the extra steps move the (already nearly closed) mean only slightly
        for (a, b) in res.distances.iter().zip(&base.distances) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // and the closed mean curve then ends where it starts
        let pts = res.mean_curve(&[0.0, 1.0], false).unwrap();
        assert!(pts[(0, 1)].abs() < 1e-10 && pts[(1, 1)].abs() < 1e-10);
    }
}

#[test]
fn closed_mean_recovers_curve_from_different_grids_and_starts() {
    // one closed shape, observed at 120–200 irregular points, each observation
    // starting at a different point of the shape
    let curves: Vec<SparseCurve> = [
        (120, 1u64, 0.0),
        (150, 2, 0.37),
        (200, 3, 0.81),
        (170, 4, 0.55),
    ]
    .iter()
    .map(|&(m, seed, off)| SparseCurve::new(shape_at(&closed_grid(m, seed, off))))
    .collect();
    let truth = SparseCurve::new(shape_at(&closed_grid(1500, 9, 0.2)));
    // the observation polygons are 0.05–0.08 from the shape (‖q‖ ≈ 2.1)
    let obs: Vec<f64> = curves.iter().map(|c| closed_dist(&truth, c)).collect();
    let best_obs = obs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(obs.iter().all(|&d| d < 0.1), "{obs:?}");
    // smooth: the mean is closer to the shape than every observation (observed
    // 0.034 vs ≥ 0.05); polygon: limited by the 60-segment knot polygon (0.115)
    for (mean_type, nk, tol) in [
        (SparseMeanType::Smooth, 41, best_obs),
        (SparseMeanType::Polygon, 61, 0.15),
    ] {
        let mut cfg = closed_cfg(nk, mean_type);
        cfg.closure_steps = 30;
        let res = sparse_elastic_mean(&curves, &cfg).unwrap();
        assert!(norm(&res.closure_residual()) < 1e-10);
        let grid = knots(601);
        let mean = SparseCurve::new(res.mean_curve(&grid, false).unwrap());
        let d = closed_dist(&truth, &mean);
        assert!(
            d < tol,
            "{mean_type:?}: distance of the mean to the shape {d}"
        );
        assert!(
            res.distances.iter().all(|&x| x < 0.25),
            "{:?}",
            res.distances
        );
        if mean_type == SparseMeanType::Smooth {
            // relabelling every curve's points (incl. the pre-alignment reference)
            // moves the mean's starting point, hence the knots along the curve: the
            // mean is the same closed curve up to spline resolution — again closer
            // to the shape than every observation (observed 0.031 vs 0.034), the two
            // means 0.022 apart, and the distances agree to within that (triangle
            // inequality, + optimiser slack)
            let relabelled: Vec<SparseCurve> = curves.iter().map(|c| relabel(c, 17)).collect();
            let res2 = sparse_elastic_mean(&relabelled, &cfg).unwrap();
            let mean2 = SparseCurve::new(res2.mean_curve(&grid, false).unwrap());
            let d2 = closed_dist(&truth, &mean2);
            assert!(
                d2 < best_obs,
                "relabelled: distance of the mean to the shape {d2}"
            );
            let d12 = closed_dist(&mean, &mean2);
            assert!(d12 <= d + d2 + 1e-3);
            for (x, y) in res.distances.iter().zip(&res2.distances) {
                assert!((x - y).abs() <= d12 + 2e-3, "{x} vs {y}");
            }
        }
    }
}

/// Curve with its observed points cyclically relabelled to start at point `k`.
fn relabel(c: &SparseCurve, k: usize) -> SparseCurve {
    let (d, m) = c.points.shape();
    let mut v = Vec::with_capacity(d * m);
    for j in 0..m {
        v.extend_from_slice(c.points.column((j + k) % m));
    }
    SparseCurve::new(FdMatrix::from_column_major(v, d, m).unwrap())
}

#[test]
fn closed_mean_is_invariant_to_starting_points() {
    let curves = shape_curves();
    // curve 4 is the pre-alignment reference (R and we both map its first point to
    // the mean's parameter 0); the starting point of every other curve is
    // re-chosen by the pre-alignment, so relabelling their points changes nothing
    let shift = |i: usize| (3 * i + 2) % curves[i].points.ncols();
    let rotated: Vec<SparseCurve> = curves
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i < 4 {
                relabel(c, shift(i))
            } else {
                c.clone()
            }
        })
        .collect();
    for (mean_type, nk) in [(SparseMeanType::Smooth, 9), (SparseMeanType::Polygon, 13)] {
        let cfg = closed_cfg(nk, mean_type);
        let a = sparse_elastic_mean(&curves, &cfg).unwrap();
        assert_eq!(a.start_indices[4], 0);
        let b = sparse_elastic_mean(&rotated, &cfg).unwrap();
        // up to round-off of the re-shifted arc-length parametrisation, propagated
        // through the iterative aligners
        assert_vec_close(b.coefs.as_slice(), a.coefs.as_slice(), 1e-6, "coefs");
        assert_vec_close(&b.distances, &a.distances, 1e-6, "distances");
        for (i, c) in curves.iter().enumerate().take(4) {
            let (m, k) = (c.points.ncols(), shift(i));
            // the same physical point is chosen as start ...
            assert_eq!((b.start_indices[i] + k) % m, a.start_indices[i]);
            // ... and every point keeps its optimal parameter
            for j in 0..m {
                let (pa, pb) = (a.params_optim[i][(j + k) % m], b.params_optim[i][j]);
                assert!((pa - pb).abs() < 1e-6, "curve {i} point {j}: {pa} vs {pb}");
            }
        }
    }
}

#[test]
fn closed_distances_are_reparametrisation_invariant() {
    // the closed elastic distance does not depend on the parametrisation of the
    // observed points (arc length vs. a given, strongly non-uniform one)
    let template = SparseCurve::new(shape_at(&closed_grid(40, 11, 0.0)));
    let s = closed_grid(25, 12, 0.3);
    let pts = shape_at(&s);
    let m = s.len();
    let param: Vec<f64> = (0..m).map(|j| (j as f64 / m as f64).powf(1.6)).collect();
    let d_arc = closed_dist(&template, &SparseCurve::new(pts.clone()));
    let d_par = closed_dist(&template, &SparseCurve::with_param(pts, param));
    assert!(
        (d_arc - d_par).abs() < 1e-3 * d_arc.max(1.0),
        "{d_arc} vs {d_par}"
    );
}

#[test]
fn closed_mean_in_three_dimensions() {
    // d-general (elasdics' closed means are 2-D only): a closed space curve
    let space = |s: &[f64]| {
        let mut v = Vec::new();
        for t in s {
            let a = 2.0 * PI * t;
            v.extend([a.cos(), a.sin(), 0.4 * (2.0 * a).sin()]);
        }
        FdMatrix::from_column_major(v, 3, s.len()).unwrap()
    };
    let curves: Vec<SparseCurve> = [(60, 1u64, 0.1), (80, 2, 0.6), (70, 3, 0.35)]
        .iter()
        .map(|&(m, seed, off)| SparseCurve::new(space(&closed_grid(m, seed, off))))
        .collect();
    let truth = SparseCurve::new(space(&closed_grid(2000, 5, 0.0)));
    let mut cfg = closed_cfg(31, SparseMeanType::Smooth);
    cfg.closure_steps = 30;
    let res = sparse_elastic_mean(&curves, &cfg).unwrap();
    assert_eq!(res.coefs.shape(), (3, 30));
    assert!(norm(&res.closure_residual()) < 1e-10);
    let grid = knots(1001);
    let mean = SparseCurve::new(res.mean_curve(&grid, false).unwrap());
    let d = closed_dist(&truth, &mean);
    assert!(d < 0.05, "3-D closed mean distance to the truth {d}");
}

#[test]
fn closed_mean_srv_is_periodic_and_mean_curve_matches() {
    let res = sparse_elastic_mean(&shape_curves(), &closed_cfg(9, SparseMeanType::Smooth)).unwrap();
    let ends = res.srv_eval(&[0.0, 1.0]);
    assert_eq!(ends.column(0), ends.column(1));
    // mean_curve(1) − mean_curve(0) is the closure residual
    let pts = res.mean_curve(&[0.0, 1.0], false).unwrap();
    assert_vec_close(pts.column(1), &res.closure_residual(), 1e-14, "end point");
}

#[test]
fn closed_invalid_inputs_are_rejected() {
    let curves = spiral_curves();
    let mut cfg = closed_cfg(2, SparseMeanType::Polygon);
    assert!(sparse_elastic_mean(&curves, &cfg).is_err());
    cfg.knots = knots(5);
    for pen in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        cfg.pen_factor = pen;
        assert!(sparse_elastic_mean(&curves, &cfg).is_err());
    }
    // the open default ignores pen_factor
    cfg.closed = false;
    assert!(sparse_elastic_mean(&curves, &cfg).is_ok());
    // a parametrisation ending at 1 on a curve that does not close is rejected
    let (x1, x2) = SPIRAL[0];
    let m = x1.len();
    let param: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let open = SparseCurve::with_param(curve2d(x1, x2), param);
    let cfg = closed_cfg(5, SparseMeanType::Smooth);
    assert!(sparse_elastic_mean(&[open], &cfg).is_err());
}
