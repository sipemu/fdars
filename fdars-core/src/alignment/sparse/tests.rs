//! Tests for sparse elastic alignment / means (goldens: R elasdics 1.1.3, `%.17g` verbatim).
#![allow(clippy::excessive_precision, clippy::field_reassign_with_default)]

use super::golden::*;
use super::polygon::{prepare_curve, srv_from_polygon, SrvPoly};
use super::smooth::{find_optimal_t_smooth, smooth_distance, LinSpline};
use super::*;
use crate::alignment::karcher_mean;

pub(super) fn curve2d(x1: &[f64], x2: &[f64]) -> FdMatrix {
    let v: Vec<f64> = x1.iter().zip(x2).flat_map(|(a, b)| [*a, *b]).collect();
    FdMatrix::from_column_major(v, 2, x1.len()).unwrap()
}

fn example_curves() -> (FdMatrix, FdMatrix) {
    // elasdics `align_curves` example
    let c1 = curve2d(&[1.0, 0.5, -1.0, -1.0], &[1.0, -0.5, -1.0, 1.0]);
    let x1: Vec<f64> = (1..=6)
        .map(|i| if i % 2 == 1 { 0.1 } else { 0.7 } * (i as f64).sin())
        .collect();
    let x2: Vec<f64> = (1..=6).map(|i| (i as f64).cos()).collect();
    (c1, curve2d(&x1, &x2))
}

pub(super) fn assert_vec_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "{what}[{k}]: {x} vs {y}");
    }
}

pub(super) fn spiral_curves() -> Vec<SparseCurve> {
    SPIRAL
        .iter()
        .map(|(x1, x2)| SparseCurve::new(curve2d(x1, x2)))
        .collect()
}

pub(super) fn knots(n: usize) -> Vec<f64> {
    let mut k: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    k[n - 1] = 1.0;
    k
}

/// Points of a smooth planar curve at parameters `s`.
fn spiral_at(s: &[f64]) -> FdMatrix {
    let x1: Vec<f64> = s.iter().map(|t| t * (6.0 * t).cos()).collect();
    let x2: Vec<f64> = s.iter().map(|t| t * (6.0 * t).sin()).collect();
    curve2d(&x1, &x2)
}

/// Deterministic pseudo-random sorted grid in [0, 1] with `m` points.
pub(super) fn random_grid(m: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut inc: Vec<f64> = (0..m - 1)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            0.3 + ((state >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect();
    let total: f64 = inc.iter().sum();
    let mut g = vec![0.0];
    let mut acc = 0.0;
    for v in &mut inc {
        acc += *v / total;
        g.push(acc);
    }
    g[m - 1] = 1.0;
    g
}

/// Point-major interleaving of two coordinate vectors.
pub(super) fn interleave(x1: &[f64], x2: &[f64]) -> Vec<f64> {
    x1.iter().zip(x2).flat_map(|(a, b)| [*a, *b]).collect()
}

/// Piecewise-constant SRV of a prepared curve.
pub(super) fn srv_of(c: &SparseCurve, closed: bool) -> SrvPoly {
    srv_from_polygon(&prepare_curve(c, closed, "c").unwrap(), "c").unwrap()
}

/// Relative L2 difference `‖a − b‖/‖b‖`.
pub(super) fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den).sqrt()
}

pub(super) fn dist(a: &SparseCurve, b: &SparseCurve) -> f64 {
    sparse_elastic_align_pair(a, b, &SparseAlignConfig::default())
        .unwrap()
        .distance
}

// ─── elasdics 1.1.3 goldens: align_curves ───────────────────────────────────

#[test]
fn golden_align_open_example() {
    // R: align_curves(data_curve1, data_curve2) (help-page example)
    let (c1, c2) = example_curves();
    let res = sparse_elastic_align_pair(
        &SparseCurve::new(c1),
        &SparseCurve::new(c2),
        &SparseAlignConfig::default(),
    )
    .unwrap();
    assert_vec_close(
        &res.param,
        &[
            0.0,
            0.256_428_702_492_646_27,
            0.452_975_653_475_148_51,
            0.601_442_331_777_276_8,
            0.841_239_097_805_397_63,
            1.0,
        ],
        1e-14,
        "param",
    );
    assert_vec_close(
        &res.param_optim,
        &[
            0.0,
            0.225_339_834_146_827_57,
            0.509_141_272_569_949_34,
            0.612_574_113_277_206_9,
            0.829_396_645_937_195_86,
            1.0,
        ],
        1e-12,
        "t_optim",
    );
    assert!((res.distance - 1.312_178_275_818_898_7).abs() < 1e-12);
    assert!(!res.closed && !res.removed_duplicates);
}

#[test]
fn golden_align_open_template_param() {
    // R: data_curve1$t <- 0:3/3; align_curves(data_curve1, data_curve2)
    let (c1, c2) = example_curves();
    let c1 = SparseCurve::with_param(c1, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
    let res = sparse_elastic_align_pair(&c1, &SparseCurve::new(c2), &SparseAlignConfig::default())
        .unwrap();
    assert_vec_close(
        &res.param_optim,
        &[
            0.0,
            0.245_208_965_616_832_23,
            0.554_093_012_417_339_1,
            0.666_666_666_666_666_63,
            0.853_216_352_761_965_06,
            1.0,
        ],
        1e-12,
        "t_optim",
    );
    assert!((res.distance - 1.312_178_282_188_295_9).abs() < 1e-12);
}

#[test]
fn golden_align_closed_example() {
    // R: align_curves(c1, c2, closed = TRUE), c1 = (sin, cos)(0:12/5), c2 = 4-gon
    let s: Vec<f64> = (0..=12).map(|k| k as f64 / 5.0).collect();
    let c1 = curve2d(
        &s.iter().map(|v| v.sin()).collect::<Vec<_>>(),
        &s.iter().map(|v| v.cos()).collect::<Vec<_>>(),
    );
    let c2 = curve2d(&[1.0, 0.5, -1.0, -1.0], &[1.0, -0.5, -1.0, 1.0]);
    let mut cfg = SparseAlignConfig::default();
    cfg.closed = true;
    let res =
        sparse_elastic_align_pair(&SparseCurve::new(c1), &SparseCurve::new(c2), &cfg).unwrap();
    // closure point appended to both curves
    assert_eq!(res.template_points.ncols(), 14);
    assert_eq!(res.points.ncols(), 5);
    let tp = [
        0.0,
        0.046_869_266_600_047_463,
        0.093_738_533_200_094_94,
        0.140_607_799_800_142_4,
        0.187_477_066_400_189_88,
        0.234_346_333_000_237_33,
        0.281_215_599_600_284_8,
        0.328_084_866_200_332_28,
        0.374_954_132_800_379_76,
        0.421_823_399_400_427_3,
        0.468_692_666_000_474_7,
        0.515_561_932_600_522_2,
        0.562_431_199_200_569_6,
        1.0,
    ];
    assert_vec_close(&res.template_param, &tp, 1e-14, "template t");
    assert_vec_close(
        &res.param_optim,
        &[
            0.234_346_333_000_237_33,
            0.468_692_666_000_474_66,
            0.562_431_199_200_569_6,
            0.0,
            0.234_346_333_000_237_33,
        ],
        1e-12,
        "t_optim",
    );
    // elasdics' internal (pre-wrap) distance from find_optimal_t_discrete_closed;
    // align_curves itself reports 2.3393052762599029 after wrapping (deviation 1).
    assert!((res.distance - 1.368_614_660_077_007_7).abs() < 1e-12);
}

// ─── elasdics 1.1.3 goldens: compute_elastic_mean ───────────────────────────

#[test]
fn golden_polygon_mean_spirals() {
    // R: compute_elastic_mean(data_curves, knots = seq(0, 1, length = 15), type = "poly")
    let mut cfg = SparseElasticMeanConfig::default();
    cfg.knots = knots(15);
    cfg.mean_type = SparseMeanType::Polygon;
    let res = sparse_elastic_mean(&spiral_curves(), &cfg).unwrap();
    assert!(res.converged);
    assert_eq!(res.coefs.shape(), (2, 14));
    let x1: Vec<f64> = (0..14).map(|j| res.coefs[(0, j)]).collect();
    let x2: Vec<f64> = (0..14).map(|j| res.coefs[(1, j)]).collect();
    assert_vec_close(&x1, &POLY_COEF_X1, 1e-9, "coef x1");
    assert_vec_close(&x2, &POLY_COEF_X2, 1e-9, "coef x2");
    for i in 0..4 {
        assert_vec_close(&res.params_optim[i], POLY_TOPT[i], 1e-9, "t_optim");
    }
    assert_vec_close(&res.distances, &POLY_FINAL_DIST, 1e-9, "dist to mean");
}

#[test]
fn golden_smooth_mean_spirals() {
    // R: compute_elastic_mean(data_curves, knots = seq(0, 1, length = 11)) (smooth).
    //
    // elasdics aligns with constrOptim(reltol = eps·100/i) on H_k integrated by
    // integrate(rel.tol = 0.1); this crate integrates exactly and searches each warp
    // coordinate. Both are local ascents, so coefficients / warps agree only to the
    // optimiser accuracy, and a tight fixed coefficient tolerance would hinge on
    // floating-point details of either optimiser. The assertions therefore target
    // quantities that are well defined despite optimiser differences:
    //  (a) our exact integrator reproduces R's high-accuracy (rel.tol = 1e-13)
    //      recomputation of its own final distances, so R's solution is scored on
    //      exactly the objective we minimise;
    //  (b) the Fréchet objective Σ d² of our fit is no worse than R's (at R's
    //      coefficients and warps). (Comparing with every curve optimally re-aligned
    //      to R's mean is not a well-defined golden: the eps = 0.01 stopping rule
    //      ends both fits at different, not fully optimised points.);
    //  (c) the two mean curves are elastically close: R's warps carry up to
    //      rel.tol = 0.1 relative error in each H_k, i.e. ≈ rel.tol/2 = 5 % in each
    //      √H_k term of the cross product, so R's mean is only determined to ≈ 5 % of
    //      its SRV norm — margin 0.05·‖μ_R‖ ≈ 0.126 (observed 0.097);
    //  (d) per-curve distances obey the triangle inequality
    //      |d(q_i, μ) − d(q_i, μ_R)| ≤ d(μ, μ_R) (+ optimiser slack 1e-3);
    //  (e) loose sanity bounds on coefficients (rel. L2 < 0.1, observed 0.049) and
    //      warps (0.02, observed 0.006).
    let mut cfg = SparseElasticMeanConfig::default();
    cfg.knots = knots(11);
    let curves = spiral_curves();
    let res = sparse_elastic_mean(&curves, &cfg).unwrap();
    assert!(res.converged);
    assert_eq!(res.coefs.shape(), (2, 11));
    let r_coefs = interleave(&SMOOTH_COEF_X1, &SMOOTH_COEF_X2);
    let mean_r = LinSpline {
        d: 2,
        knots: &res.knots,
        c: &r_coefs,
    };
    let srvs: Vec<_> = curves.iter().map(|c| srv_of(c, false)).collect();
    // (a) + (b)
    let mut obj_r = 0.0;
    let mut d_r_star = Vec::new();
    for (i, srv) in srvs.iter().enumerate() {
        let d_r = smooth_distance(&mean_r, srv, SMOOTH_TOPT[i]);
        assert!(
            (d_r - SMOOTH_FINAL_DIST[i]).abs() < 1e-9,
            "{d_r} vs R {}",
            SMOOTH_FINAL_DIST[i]
        );
        obj_r += d_r * d_r;
        // optimal warp to R's mean, for the triangle inequality (d)
        let t_star = find_optimal_t_smooth(&mean_r, srv, SMOOTH_TOPT[i], 1e-8);
        let d_star = smooth_distance(&mean_r, srv, &t_star);
        assert!(d_star <= d_r + 1e-12);
        d_r_star.push(d_star);
    }
    let obj: f64 = res.distances.iter().map(|x| x * x).sum();
    assert!(obj <= obj_r + 1e-9, "objective {obj} vs R {obj_r}");
    // (c)
    let norm_r = mean_r.norm_sq().sqrt();
    let grid = knots(401);
    let ours = SparseCurve::with_param(res.mean_curve(&grid, false).unwrap(), grid.clone());
    let r_res = SparseElasticMeanResult {
        coefs: curve2d(&SMOOTH_COEF_X1, &SMOOTH_COEF_X2),
        ..res.clone()
    };
    let theirs = SparseCurve::with_param(r_res.mean_curve(&grid, false).unwrap(), grid);
    let d_means = dist(&theirs, &ours);
    assert!(
        d_means < 0.05 * norm_r,
        "elastic distance between means {d_means} (‖μ_R‖ = {norm_r})"
    );
    // (d)
    for (ours, theirs) in res.distances.iter().zip(&d_r_star) {
        assert!(
            (ours - theirs).abs() <= d_means + 1e-3,
            "dist {ours} vs R {theirs} (d_means {d_means})"
        );
    }
    // (e)
    let rel = rel_l2(res.coefs.as_slice(), &r_coefs);
    assert!(rel < 0.1, "relative coefficient difference {rel}");
    for i in 0..4 {
        assert_vec_close(&res.params_optim[i], SMOOTH_TOPT[i], 0.02, "t_optim");
    }
}

// ─── invariants ─────────────────────────────────────────────────────────────

#[test]
fn collinear_points_do_not_change_distances() {
    let (c1, c2) = example_curves();
    // insert midpoints into every segment of curve 1: same polygon
    let m = c1.ncols();
    let mut v = Vec::new();
    for j in 0..m {
        v.extend_from_slice(c1.column(j));
        if j + 1 < m {
            for c in 0..2 {
                v.push(0.5 * (c1[(c, j)] + c1[(c, j + 1)]));
            }
        }
    }
    let c1_dense = FdMatrix::from_column_major(v, 2, 2 * m - 1).unwrap();
    let a = SparseCurve::new(c1);
    let b = SparseCurve::new(c1_dense);
    assert!(dist(&a, &b) < 1e-7, "same polygon: {}", dist(&a, &b));
    assert!(dist(&b, &a) < 1e-7);
    let c = SparseCurve::new(c2);
    assert!((dist(&a, &c) - dist(&b, &c)).abs() < 1e-6);
}

#[test]
fn reparametrisation_invariance_of_distance() {
    let (c1, c2) = example_curves();
    let base = dist(&SparseCurve::new(c1.clone()), &SparseCurve::new(c2.clone()));
    // reparametrise the template (non-arc-length t) ...
    for t in [
        vec![0.0, 0.1, 0.2, 1.0],
        vec![0.0, 0.6, 0.9, 1.0],
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    ] {
        let d = dist(
            &SparseCurve::with_param(c1.clone(), t),
            &SparseCurve::new(c2.clone()),
        );
        assert!((d - base).abs() < 1e-6, "template reparam: {d} vs {base}");
    }
    // ... and the aligned curve
    let t2 = vec![0.0, 0.05, 0.1, 0.5, 0.9, 1.0];
    let d = dist(&SparseCurve::new(c1), &SparseCurve::with_param(c2, t2));
    assert!((d - base).abs() < 1e-6, "curve reparam: {d} vs {base}");
}

#[test]
fn self_distance_zero_and_translation_invariance() {
    let pts = spiral_at(&random_grid(25, 3));
    let a = SparseCurve::new(pts.clone());
    assert!(dist(&a, &a) < 1e-10);
    let mut shifted = pts.clone();
    for j in 0..shifted.ncols() {
        shifted[(0, j)] += 3.0;
        shifted[(1, j)] -= 1.5;
    }
    let other = SparseCurve::new(spiral_at(&random_grid(17, 9)));
    let d1 = dist(&other, &a);
    let d2 = dist(&other, &SparseCurve::new(shifted));
    assert!((d1 - d2).abs() < 1e-10);
}

#[test]
fn same_curve_on_different_grids_mean_recovers_curve() {
    // the same smooth curve sampled at 120–200 irregular points per observation
    let curves: Vec<SparseCurve> = [(120, 1u64), (150, 2), (200, 3), (135, 4), (170, 5)]
        .iter()
        .map(|&(m, seed)| SparseCurve::new(spiral_at(&random_grid(m, seed))))
        .collect();
    let truth = SparseCurve::new(spiral_at(&knots(2001)));
    // ‖q‖ of the truth = sqrt(curve length) ≈ 2.6
    // observation polygons themselves are within ~0.035 of the curve
    let worst_obs = curves.iter().map(|c| dist(&truth, c)).fold(0.0, f64::max);
    assert!(worst_obs < 0.05);
    for (mean_type, nk) in [(SparseMeanType::Polygon, 61), (SparseMeanType::Smooth, 41)] {
        let mut cfg = SparseElasticMeanConfig::default();
        cfg.knots = knots(nk);
        cfg.mean_type = mean_type;
        let res = sparse_elastic_mean(&curves, &cfg).unwrap();
        let grid = knots(1001);
        let mean = SparseCurve::with_param(res.mean_curve(&grid, false).unwrap(), grid);
        let d = dist(&truth, &mean);
        let tol = if mean_type == SparseMeanType::Smooth {
            0.02
        } else {
            0.1
        };
        assert!(d < tol, "{mean_type:?}: distance of mean to the curve {d}");
        // every observation is close to the mean (polygon discretisation error only)
        assert!(
            res.distances.iter().all(|&x| x < 0.25),
            "{:?}",
            res.distances
        );
    }
}

#[test]
fn dense_limit_agrees_with_karcher_mean() {
    // 1-D functions on a dense common grid: sparse polygon mean vs dense SRSF
    // Karcher mean, compared modulo warping by the elastic distance.
    let m = 101;
    let argvals = knots(m);
    let n = 6;
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let a = 0.6 * (i as f64 / (n - 1) as f64 - 0.5); // warp strength
        let h = 1.0 + 0.1 * i as f64; // amplitude
        for (j, &t) in argvals.iter().enumerate() {
            let g = t + a * t * (1.0 - t);
            data[(i, j)] =
                h * (-(g - 0.35).powi(2) / 0.01).exp() + 0.7 * (-(g - 0.7).powi(2) / 0.005).exp();
        }
    }
    let km = karcher_mean(&data, &argvals, 20, 1e-4, 0.0);
    let curves: Vec<SparseCurve> = (0..n)
        .map(|i| {
            SparseCurve::with_param(
                FdMatrix::from_column_major(data.row(i), 1, m).unwrap(),
                argvals.clone(),
            )
        })
        .collect();
    let mut cfg = SparseElasticMeanConfig::default();
    cfg.knots = knots(51);
    cfg.mean_type = SparseMeanType::Smooth;
    let res = sparse_elastic_mean(&curves, &cfg).unwrap();
    let grid = knots(401);
    let sparse_mean = SparseCurve::with_param(res.mean_curve(&grid, false).unwrap(), grid);
    let dense_mean = SparseCurve::with_param(
        FdMatrix::from_column_major(km.mean.clone(), 1, m).unwrap(),
        argvals.clone(),
    );
    let d = dist(&dense_mean, &sparse_mean);
    // reference scale: SRSF norm of the mean, ‖q‖ = sqrt(total variation)
    let scale = km
        .mean
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum::<f64>()
        .sqrt();
    let obj = |mu: &SparseCurve| curves.iter().map(|c| dist(mu, c).powi(2)).sum::<f64>();
    assert!(
        d / scale < 0.06,
        "dense-limit relative distance {} (d = {d})",
        d / scale
    );
    // and the sparse mean is a comparably good Fréchet mean (Σ d² within 25 %)
    let (o_sparse, o_dense) = (obj(&sparse_mean), obj(&dense_mean));
    assert!(
        o_sparse < 1.25 * o_dense,
        "objective {o_sparse} vs dense {o_dense}"
    );
}

#[test]
fn irreg_fdata_functions() {
    // a bump observed on three different irregular grids
    let f = |t: f64| (-(t - 0.5).powi(2) / 0.02).exp();
    let grids = [
        random_grid(20, 11),
        random_grid(27, 12),
        random_grid(35, 13),
    ];
    let argvals: Vec<Vec<f64>> = grids
        .iter()
        .map(|g| g.iter().map(|t| 2.0 * t).collect())
        .collect();
    let values: Vec<Vec<f64>> = grids
        .iter()
        .map(|g| g.iter().map(|&t| f(t)).collect())
        .collect();
    let data = crate::irreg_fdata::IrregFdata::from_lists(&argvals, &values);
    let curves = sparse_curves_from_irreg(&data).unwrap();
    assert_eq!(curves.len(), 3);
    assert_vec_close(curves[0].param.as_ref().unwrap(), &grids[0], 1e-15, "param");
    let mut cfg = SparseElasticMeanConfig::default();
    cfg.knots = knots(21);
    let res = sparse_elastic_mean(&curves, &cfg).unwrap();
    let grid = knots(501);
    let mean = SparseCurve::with_param(res.mean_curve(&grid, false).unwrap(), grid.clone());
    let truth = SparseCurve::with_param(
        FdMatrix::from_column_major(grid.iter().map(|&t| f(t)).collect(), 1, grid.len()).unwrap(),
        grid,
    );
    let d = dist(&truth, &mean);
    assert!(d < 0.1, "1-D mean distance to truth {d}");
}

#[test]
fn exact_spline_integrals_match_quadrature() {
    let knots = [0.0, 0.3, 0.55, 1.0];
    let c = [1.0, -0.5, -0.2, 0.8, 0.9, 0.1, -1.2, 0.4];
    let sp = LinSpline {
        d: 2,
        knots: &knots,
        c: &c,
    };
    let v = [0.7, -1.1];
    let (a, b) = (0.12, 0.93);
    let n = 200_000;
    let mut buf = [0.0; 2];
    let h = (b - a) / n as f64;
    let mut quad = 0.0;
    let mut nrm = 0.0;
    for k in 0..n {
        let x = a + (k as f64 + 0.5) * h;
        sp.eval(x, &mut buf);
        quad += h * (v[0] * buf[0] + v[1] * buf[1]).max(0.0).powi(2);
    }
    for k in 0..n {
        let x = (k as f64 + 0.5) / n as f64;
        sp.eval(x, &mut buf);
        nrm += (buf[0].powi(2) + buf[1].powi(2)) / n as f64;
    }
    assert!((sp.pos_sq_integral(&v, a, b) - quad).abs() < 1e-8);
    assert!((sp.norm_sq() - nrm).abs() < 1e-8);
}

#[test]
fn smooth_mean_curve_integrates_srv() {
    // mean_curve is ∫ p|p|: its finite differences reproduce p|p|
    let mut cfg = SparseElasticMeanConfig::default();
    cfg.knots = knots(11);
    cfg.max_iter = 2;
    let res = sparse_elastic_mean(&spiral_curves(), &cfg).unwrap();
    let (x, h) = (0.437, 1e-5);
    let pts = res.mean_curve(&[x - h, x + h], false).unwrap();
    let q = res.srv_eval(&[x]);
    let nq = (q[(0, 0)].powi(2) + q[(1, 0)].powi(2)).sqrt();
    for c in 0..2 {
        let fd = (pts[(c, 1)] - pts[(c, 0)]) / (2.0 * h);
        assert!((fd - q[(c, 0)] * nq).abs() < 1e-5);
    }
    let centred = res.mean_curve(&knots(11), true).unwrap();
    for c in 0..2 {
        let s: f64 = (0..11).map(|j| centred[(c, j)]).sum();
        assert!(s.abs() < 1e-12);
    }
}

#[test]
fn duplicates_are_removed() {
    let c = curve2d(&[0.0, 1.0, 1.0, 2.0], &[0.0, 0.0, 0.0, 1.0]);
    let other = curve2d(&[0.0, 1.0, 2.0], &[0.0, 0.0, 1.0]);
    let res = sparse_elastic_align_pair(
        &SparseCurve::new(other),
        &SparseCurve::new(c),
        &SparseAlignConfig::default(),
    )
    .unwrap();
    assert!(res.removed_duplicates);
    assert_eq!(res.points.ncols(), 3);
    assert!(res.distance < 1e-10);
}

#[test]
fn invalid_inputs_are_rejected() {
    let (c1, c2) = example_curves();
    let a = SparseCurve::new(c1.clone());
    let b = SparseCurve::new(c2);
    let mut cfg = SparseAlignConfig::default();
    cfg.eps = 0.0;
    assert!(sparse_elastic_align_pair(&a, &b, &cfg).is_err());
    // dimension mismatch
    let one_d = SparseCurve::new(FdMatrix::from_column_major(vec![0.0, 1.0, 0.5], 1, 3).unwrap());
    assert!(sparse_elastic_align_pair(&a, &one_d, &SparseAlignConfig::default()).is_err());
    // bad parametrisations
    for t in [
        vec![0.0, 0.5, 0.4, 1.0],
        vec![0.1, 0.2, 0.3, 1.0],
        vec![0.0, 0.2, 0.3, 0.9],
        vec![0.0, 1.0],
    ] {
        let c = SparseCurve::with_param(c1.clone(), t);
        assert!(sparse_elastic_align_pair(&c, &b, &SparseAlignConfig::default()).is_err());
    }
    // knots / max_iter / empty input
    let mut mcfg = SparseElasticMeanConfig::default();
    mcfg.knots = vec![0.0, 0.5, 0.5, 1.0];
    assert!(sparse_elastic_mean(std::slice::from_ref(&a), &mcfg).is_err());
    let mut mcfg = SparseElasticMeanConfig::default();
    mcfg.max_iter = 0;
    assert!(sparse_elastic_mean(std::slice::from_ref(&a), &mcfg).is_err());
    assert!(sparse_elastic_mean(&[], &SparseElasticMeanConfig::default()).is_err());
    assert!(sparse_elastic_mean(&[a, one_d], &SparseElasticMeanConfig::default()).is_err());
    // non-finite points
    let nan =
        SparseCurve::new(FdMatrix::from_column_major(vec![0.0, f64::NAN, 1.0], 1, 3).unwrap());
    assert!(sparse_elastic_mean(&[nan], &SparseElasticMeanConfig::default()).is_err());
}
