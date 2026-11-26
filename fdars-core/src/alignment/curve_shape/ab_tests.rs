//! Tests of the general (a,b) elastic metric (Kurtek–Needham F-transform).
//!
//! Golden values: geomstats 2.8.0 (numpy backend, numpy 2.3.5),
//! `.planning/quick/261126-85x-*/golden_geomstats_ab.py`:
//! `FTransform(Euclidean(2), k, a, b)(curve)` and
//! `DiscreteCurvesStartingAtOrigin(2, k).equip_with_metric(ElasticMetric, a=a, b=b)`
//! `.metric.dist(c1, c2)`.

use super::geometry::{self, Grid};
use super::*;
use std::f64::consts::PI;

/// (a, b) pairs of the golden script: λ = 1 (SRVF), 0.25, 0.5, 1.5, 2.
const AB: [(f64, f64); 5] = [(1.0, 0.5), (1.0, 2.0), (1.0, 1.0), (3.0, 1.0), (2.0, 0.5)];

// Velocities (k − 1)(p_{i+1} − p_i) of the 9-point spiral r = 0.3 + t,
// angle 3πt + 2.5 (initial tangent angle near −1.88, 1.5 turns: the lift
// leaves (−π, π]).
const VX: [f64; 8] = [
    -0.999557204011635,
    3.552867517289969,
    4.603036405781406,
    -1.4033803194817764,
    -7.612619351232587,
    -4.530895846552262,
    5.997774780170951,
    10.647412297036684,
];
const VY: [f64; 8] = [
    -3.1741915159155374,
    -2.6167239266346627,
    3.0243907057557795,
    6.457510730445301,
    1.2329849551465797,
    -7.564111217854194,
    -7.906537171685861,
    2.886233996211949,
];

/// geomstats `FTransform(..., a, b)(spiral)` for the pairs of [`AB`]: (fx, fy).
const F_GOLD: [([f64; 8], [f64; 8]); 5] = [
    (
        [
            -0.5479306300018546,
            1.691363299163953,
            1.961366835816869,
            -0.545924099178515,
            -2.741293696070555,
            -1.5258654777648222,
            1.903907443809858,
            3.205704208515508,
        ],
        [
            -1.740007225281223,
            -1.245706683971701,
            1.2887014365933869,
            2.512013799477914,
            0.44399617647319933,
            -2.547358528684779,
            -2.5098166449496118,
            0.8689822663289046,
        ],
    ),
    (
        [
            6.509150409342484,
            8.296783987769084,
            9.288448321606957,
            9.275876501123417,
            8.163483793360818,
            5.980102236281974,
            2.878334221034441,
            -0.8785728775758797,
        ],
        [
            -3.297970718147451,
            -1.3278910709494318,
            1.359451320922926,
            4.437348121255285,
            7.533041356647329,
            10.262319042106231,
            12.267839355232585,
            13.256501202270476,
        ],
    ),
    (
        [
            2.157913768116537,
            3.991331223925299,
            4.496831698674422,
            3.226411669750022,
            0.4454309859145439,
            -2.9279361616487782,
            -5.643016556068512,
            -6.5846916326772424,
        ],
        [
            -2.9419076543415583,
            -1.3112035959076482,
            1.3451207020477598,
            4.002905637237309,
            5.536143618814738,
            5.166855286622572,
            2.8022378077849712,
            -0.8766497072964521,
        ],
    ),
    (
        [
            -3.4542199302071412,
            2.436175757809251,
            3.019568146055883,
            -4.596785088345479,
            -1.3248329692799898,
            5.937059678983183,
            -1.1779014682937068,
            -6.1259730038827165,
        ],
        [
            -1.174639157809807,
            -3.4227222332618075,
            3.593473701115007,
            2.302727221120644,
            -5.39371046079271,
            -0.14326928415507775,
            6.1894039062283195,
            -2.568877841091867,
        ],
    ),
    (
        [
            -1.4950864672210815,
            0.6231212204823416,
            0.9315496153163625,
            -2.3387773206450313,
            2.6350427001919683,
            -1.4012185549010787,
            -0.848923208345928,
            2.8666892002505753,
        ],
        [
            1.045260529656351,
            -2.0060445899418937,
            2.1540488786292835,
            -1.0669428734790756,
            -0.8765692790669819,
            2.617993027358645,
            -3.0337060721487785,
            1.6774273995779598,
        ],
    ),
];

/// geomstats `ElasticMetric(a, b).dist(c1, c2)` (no reparametrisation, no
/// rotation) for the pairs of [`AB`], at k = 9 and k = 2001 sampling points.
const DIST_K9: [f64; 5] = [
    1.7051724945883746,
    2.2814865387198457,
    2.0139265464379794,
    3.9972002417679904,
    1.8910801294011281,
];
const DIST_K2001: [f64; 5] = [
    1.7213667800948134,
    2.32368014891253,
    2.0409717915204184,
    4.012483364599253,
    1.875840702613627,
];

fn mat2(x: &[f64], y: &[f64]) -> FdMatrix {
    let m = x.len();
    let mut out = FdMatrix::zeros(2, m);
    for j in 0..m {
        out[(0, j)] = x[j];
        out[(1, j)] = y[j];
    }
    out
}

fn sample(m: usize, f: impl Fn(f64) -> (f64, f64)) -> FdMatrix {
    let mut out = FdMatrix::zeros(2, m);
    for j in 0..m {
        let (x, y) = f(j as f64 / (m - 1) as f64);
        out[(0, j)] = x;
        out[(1, j)] = y;
    }
    out
}

/// Golden script curves.
fn gc1(t: f64) -> (f64, f64) {
    (t, 0.3 * (2.0 * PI * t).sin())
}
fn gc2(t: f64) -> (f64, f64) {
    ((1.2 * t).cos() + t, 0.5 * t * t - 0.2 * (5.0 * t).sin())
}

fn elastic(a: f64, b: f64) -> CurveShapeMetric {
    CurveShapeMetric::Elastic { a, b }
}

fn cfg_with(
    metric: CurveShapeMetric,
    rotation: bool,
    reparam: bool,
    scale: bool,
) -> CurveShapeConfig {
    CurveShapeConfig {
        metric,
        rotation,
        reparam,
        scale,
        ..CurveShapeConfig::default()
    }
}

fn rotate_curve(beta: &FdMatrix, phi: f64) -> FdMatrix {
    let (s, c) = phi.sin_cos();
    let mut out = beta.clone();
    for j in 0..beta.ncols() {
        let (x, y) = (beta[(0, j)], beta[(1, j)]);
        out[(0, j)] = c * x - s * y;
        out[(1, j)] = s * x + c * y;
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

/// Left Riemann L² distance of geomstats' discretisation.
fn riemann_dist(f1: &FdMatrix, f2: &FdMatrix) -> f64 {
    let k = f1.ncols();
    let s: f64 = (0..k)
        .map(|j| {
            (0..2)
                .map(|r| (f1[(r, j)] - f2[(r, j)]).powi(2))
                .sum::<f64>()
        })
        .sum();
    (s / k as f64).sqrt()
}

/// Forward-difference velocities of geomstats (`k − 1` columns).
fn forward_velocity(p: &FdMatrix) -> FdMatrix {
    let k = p.ncols();
    let mut v = FdMatrix::zeros(2, k - 1);
    for j in 0..k - 1 {
        for r in 0..2 {
            v[(r, j)] = (k - 1) as f64 * (p[(r, j + 1)] - p[(r, j)]);
        }
    }
    v
}

// ─── Golden: transform, inverse and no-reparam distance ─────────────────────

#[test]
fn f_transform_matches_geomstats_pointwise() {
    let v = mat2(&VX, &VY);
    for (&(a, b), (fx, fy)) in AB.iter().zip(F_GOLD.iter()) {
        let f = geometry::f_from_velocity(&v, a, b);
        let gold = mat2(fx, fy);
        let err = max_abs_diff(&f, &gold);
        assert!(err < 1e-12, "a={a} b={b}: max err {err}");
    }
}

#[test]
fn f_inverse_is_left_inverse_for_small_ratio_and_right_inverse_for_large() {
    let v = mat2(&VX, &VY);
    let theta0 = VY[0].atan2(VX[0]);
    for &(a, b) in &AB {
        let lam = a / (2.0 * b);
        let f = geometry::f_from_velocity(&v, a, b);
        let back = geometry::velocity_from_f(&f, a, b);
        // right inverse: the recovered velocity always has transform f
        let f2 = geometry::f_from_velocity(&back, a, b);
        assert!(max_abs_diff(&f, &f2) < 1e-12, "a={a} b={b}");
        // left inverse whenever λθ(0) stays in (−π, π] (always for λ ≤ 1);
        // geomstats' inverse fails for the same case (a=2, b=1/2, λθ0 ≈ −3.75).
        let in_branch = (lam * theta0).abs() <= PI;
        assert_eq!(in_branch, (a, b) != (2.0, 0.5));
        if in_branch {
            assert!(max_abs_diff(&v, &back) < 1e-12, "a={a} b={b}");
        } else {
            assert!(max_abs_diff(&v, &back) > 0.1);
        }
    }
}

#[test]
fn discrete_distance_matches_geomstats_k9() {
    // Exact reproduction of geomstats' discretisation (forward differences,
    // left Riemann sum) with our pointwise transform.
    let k = 9;
    let (p1, p2) = (sample(k, gc1), sample(k, gc2));
    let (v1, v2) = (forward_velocity(&p1), forward_velocity(&p2));
    for (&(a, b), &gold) in AB.iter().zip(&DIST_K9) {
        let d = riemann_dist(
            &geometry::f_from_velocity(&v1, a, b),
            &geometry::f_from_velocity(&v2, a, b),
        );
        assert!((d - gold).abs() < 1e-12, "a={a} b={b}: {d} vs {gold}");
    }
}

#[test]
fn curve_shape_distance_matches_geomstats_fine_grid() {
    // Our quadrature (central differences, trapezoid on m nodes) vs geomstats
    // (forward differences, left Riemann sum on m − 1 segments): both second
    // order, observed gap ≤ 9e-7 relative at m = 2001.
    let m = 2001;
    let (c1, c2) = (sample(m, gc1), sample(m, gc2));
    for (&(a, b), &gold) in AB.iter().zip(&DIST_K2001) {
        let cfg = cfg_with(elastic(a, b), false, false, false);
        let d = curve_shape_distance(&c1, &c2, &cfg).unwrap();
        assert!(
            ((d - gold) / gold).abs() < 1e-5,
            "a={a} b={b}: {d} vs {gold}"
        );
    }
}

// ─── Hand-computed closed forms ─────────────────────────────────────────────

#[test]
fn bending_only_distance_closed_form() {
    // Unit-speed segment (θ = 0) vs unit-speed arc of curvature κ (θ = κt):
    // d² = 4b² ∫₀¹ |e^{iλκt} − 1|² dt = 8b²(1 − sin(λκ)/(λκ)).
    let m = 2001;
    let kappa = 2.0;
    let seg = sample(m, |t| (t, 0.0));
    let arc = sample(m, |t| {
        ((kappa * t).sin() / kappa, (1.0 - (kappa * t).cos()) / kappa)
    });
    for &(a, b) in &AB {
        let lam = a / (2.0 * b);
        let want = (8.0 * b * b * (1.0 - (lam * kappa).sin() / (lam * kappa))).sqrt();
        let d = curve_shape_distance(&seg, &arc, &cfg_with(elastic(a, b), false, false, false))
            .unwrap();
        assert!(
            ((d - want) / want).abs() < 1e-5,
            "a={a} b={b}: {d} vs {want}"
        );
    }
}

#[test]
fn stretching_only_distance_closed_form() {
    // Same straight segment, speeds 1 and 1 + t (no bending): F differs only in
    // modulus, d = 2b ‖1 − √(1+t)‖_{L²}, independent of a.
    let m = 2001;
    let s1 = sample(m, |t| (t, 0.0));
    let s2 = sample(m, |t| (t + 0.5 * t * t, 0.0));
    // ∫₀¹ (1 − √(1+t))² dt = 1 + 3/2 − (4/3)(2^{3/2} − 1)
    let l2 = (1.0 + 1.5 - 4.0 / 3.0 * (2f64.powf(1.5) - 1.0)).sqrt();
    for &(a, b) in &AB {
        let d =
            curve_shape_distance(&s1, &s2, &cfg_with(elastic(a, b), false, false, false)).unwrap();
        let want = 2.0 * b * l2;
        assert!(
            ((d - want) / want).abs() < 1e-5,
            "a={a} b={b}: {d} vs {want}"
        );
    }
}

// ─── SRVF special case and default bit-identity ─────────────────────────────

#[test]
fn ab_transform_one_half_is_srvf() {
    let beta = sample(60, gc2);
    let f = ab_transform(&beta, 1.0, 0.5).unwrap();
    let q = curve_srvf(&beta, false).unwrap();
    assert!(max_abs_diff(&f, &q) < 1e-13);
    // (a, b) → (ka, kb) keeps λ and scales F by k
    let f3 = ab_transform(&beta, 3.0, 1.5).unwrap();
    assert!(max_abs_diff(&f3, &geometry::scaled(&q, 3.0)) < 1e-12);
}

#[test]
fn elastic_one_half_reproduces_srvf_alignment() {
    let m = 61;
    let c1 = sample(m, |t| (t, 0.3 * (3.0 * t).sin()));
    let c2 = sample(m, |t| (t * t + 0.2, 0.25 * (5.0 * t).sin() + 0.1 * t));
    for scale in [true, false] {
        for rotation in [true, false] {
            let s = cfg_with(CurveShapeMetric::Srvf, rotation, true, scale);
            let e = cfg_with(elastic(1.0, 0.5), rotation, true, scale);
            let rs = curve_shape_align_pair(&c1, &c2, &s).unwrap();
            let re = curve_shape_align_pair(&c1, &c2, &e).unwrap();
            assert!((rs.distance - re.distance).abs() < 1e-10);
            assert!(max_abs_diff(&rs.rotation, &re.rotation) < 1e-10);
            let dg: f64 = rs
                .gamma
                .iter()
                .zip(&re.gamma)
                .map(|(x, y)| (x - y).abs())
                .sum();
            assert!(dg < 1e-10);
            assert!(max_abs_diff(&rs.beta2_aligned, &re.beta2_aligned) < 1e-10);
        }
    }
}

#[test]
fn default_metric_is_srvf() {
    let cfg = CurveShapeConfig::default();
    assert_eq!(cfg.metric, CurveShapeMetric::Srvf);
    assert_eq!(CurveShapeMetric::default(), CurveShapeMetric::Srvf);
    assert_eq!(cfg.metric.ratio(), 1.0);
}

// ─── Invariances and metric axioms ──────────────────────────────────────────

/// Open test curves with genuinely different shapes.
fn shapes(m: usize) -> Vec<FdMatrix> {
    vec![
        sample(m, |t| (t, 0.3 * (3.0 * t).sin())),
        sample(m, |t| (t + 0.2 * t * t, 0.25 * (5.0 * t).sin() - 0.1 * t)),
        sample(m, |t| ((1.5 * t).cos(), (1.5 * t).sin() + 0.2 * t)),
        sample(m, |t| (t, 0.4 * t * (1.0 - t) + 0.05 * (9.0 * t).cos())),
    ]
}

#[test]
fn rotation_invariance_and_recovered_rotation() {
    let m = 81;
    let sh = shapes(m);
    for &(a, b) in &AB {
        let lam = a / (2.0 * b);
        let cfg = cfg_with(elastic(a, b), true, true, true);
        let d0 = curve_shape_distance(&sh[0], &sh[1], &cfg).unwrap();
        for phi in [0.4, -1.1, 2.5] {
            let d = curve_shape_distance(&sh[0], &rotate_curve(&sh[1], phi), &cfg).unwrap();
            assert!((d - d0).abs() < 1e-9, "a={a} b={b} phi={phi}: {d} vs {d0}");
        }
        // a rotated copy is at distance 0, and the reported curve rotation
        // undoes it (ψ = −λφ stays in (−π, π] here, so no branch shift)
        let phi = 0.4;
        let res = curve_shape_align_pair(&sh[2], &rotate_curve(&sh[2], phi), &cfg).unwrap();
        assert!(res.distance < 1e-6, "a={a} b={b}: {}", res.distance);
        assert!((lam * phi).abs() < PI);
        let back = res.rotation[(1, 0)].atan2(res.rotation[(0, 0)]);
        assert!((back + phi).abs() < 1e-6, "a={a} b={b}: {back}");
        assert!(max_abs_diff(&res.beta2_aligned, &res.beta1) < 1e-6);
    }
}

#[test]
fn reparametrisation_invariance() {
    // β∘γ with a smooth monotone warp is (up to DP discretisation) the same
    // shape; the no-reparam distance is clearly positive.
    let m = 101;
    let gam = |t: f64| t + 0.25 * (PI * t).sin() / PI; // γ' = 1 + 0.25 cos(πt) > 0
    let c = |t: f64| (t, 0.3 * (3.0 * t).sin() + 0.1 * t * t);
    let c1 = sample(m, c);
    let c2 = sample(m, |t| c(gam(t)));
    for &(a, b) in &AB {
        let with =
            curve_shape_distance(&c1, &c2, &cfg_with(elastic(a, b), true, true, true)).unwrap();
        let without =
            curve_shape_distance(&c1, &c2, &cfg_with(elastic(a, b), true, false, true)).unwrap();
        assert!(without > 0.02, "a={a} b={b}: {without}");
        // The shared DP engine has a grid-independent residual floor for this
        // warp (≈0.02 for the SRVF, a = 1, b = 1/2, at m = 51…401; cf. the
        // SRVF test `curve_shape::tests`, bound 0.03), growing to ≈0.04 at
        // λ = 2 where F turns twice as fast.
        assert!(with < 0.05, "a={a} b={b}: {with}");
        assert!(with < without / 3.0, "a={a} b={b}: {with} vs {without}");
    }
}

#[test]
fn reparametrisation_acts_like_on_srvfs() {
    // F(c∘γ)(t) = F(c)(γ(t))·√γ'(t): sample both sides analytically. The
    // discrepancy is pure discretisation (central differences + linear
    // interpolation, O(h²)): it must be small and shrink ~16× from m = 501 to
    // m = 2001 (a wrong action would leave an O(1) gap).
    let gam = |t: f64| t + 0.3 * (PI * t).sin() / PI;
    let dgam = |t: f64| 1.0 + 0.3 * (PI * t).cos();
    let c = |t: f64| ((2.0 * t).sin(), 0.5 * t * t + 0.3 * (4.0 * t).cos());
    let gap = |m: usize, a: f64, b: f64| {
        let grid = Grid::new(m);
        let fw = ab_transform(&sample(m, |t| c(gam(t))), a, b).unwrap();
        let f = ab_transform(&sample(m, c), a, b).unwrap();
        let warped: Vec<f64> = grid.t.iter().map(|&t| gam(t)).collect();
        let fg = geometry::warp_curve(&f, &grid, &warped);
        let scale = f.as_slice().iter().fold(0.0f64, |s, v| s.max(v.abs()));
        (2..m - 2)
            .flat_map(|j| (0..2).map(move |r| (r, j)))
            .map(|(r, j)| (fw[(r, j)] - fg[(r, j)] * dgam(grid.t[j]).sqrt()).abs())
            .fold(0.0, f64::max)
            / scale
    };
    for &(a, b) in &AB {
        let (coarse, fine) = (gap(501, a, b), gap(2001, a, b));
        assert!(fine < 1e-4, "a={a} b={b}: {fine}");
        assert!(fine < coarse / 10.0, "a={a} b={b}: {coarse} -> {fine}");
    }
}

#[test]
fn reparam_distance_never_exceeds_rigid_distance() {
    let sh = shapes(61);
    for &(a, b) in &AB {
        for scale in [true, false] {
            for i in 0..sh.len() {
                for j in 0..sh.len() {
                    let with = cfg_with(elastic(a, b), true, true, scale);
                    let without = cfg_with(elastic(a, b), true, false, scale);
                    let dw = curve_shape_distance(&sh[i], &sh[j], &with).unwrap();
                    let dn = curve_shape_distance(&sh[i], &sh[j], &without).unwrap();
                    assert!(dw <= dn + 1e-12, "a={a} b={b}: {dw} > {dn}");
                }
            }
        }
    }
}

#[test]
fn rigid_distance_is_a_metric() {
    // Without reparametrisation the quotient by rotations (an isometric group
    // action on F-space) is exact, so symmetry and the triangle inequality hold
    // exactly (up to rounding) — on random triples.
    let m = 41;
    let mut state = 12345u64;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut curves = Vec::new();
    for _ in 0..6 {
        let c: Vec<f64> = (0..6).map(|_| rnd()).collect();
        curves.push(sample(m, move |t| {
            (
                t + 0.2 * c[0] * (PI * t).sin() + 0.1 * c[1] * t * t,
                0.4 * c[2] * (2.0 * PI * t).sin()
                    + 0.3 * c[3] * t
                    + 0.2 * c[4] * (3.0 * t).cos()
                    + 0.1 * c[5] * t * t,
            )
        }));
    }
    for &(a, b) in &AB {
        for (rotation, scale) in [(true, true), (true, false), (false, false), (false, true)] {
            let cfg = cfg_with(elastic(a, b), rotation, false, scale);
            let n = curves.len();
            let mut d = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    d[i][j] = curve_shape_distance(&curves[i], &curves[j], &cfg).unwrap();
                }
            }
            for i in 0..n {
                assert!(d[i][i] < 1e-7);
                for j in 0..n {
                    assert!((d[i][j] - d[j][i]).abs() < 1e-10, "asym {i},{j}");
                    for k in 0..n {
                        assert!(d[i][k] <= d[i][j] + d[j][k] + 1e-10, "triangle {i},{j},{k}");
                    }
                }
            }
        }
    }
}

#[test]
fn distance_matrix_matches_pairwise() {
    let sh = shapes(41);
    let set = curves_to_set(&sh).unwrap();
    let cfg = cfg_with(elastic(1.0, 1.0), true, true, true);
    let dm = curve_shape_self_distance_matrix(&set, &cfg).unwrap();
    for i in 0..sh.len() {
        for j in i + 1..sh.len() {
            let d = curve_shape_distance(&sh[i], &sh[j], &cfg).unwrap();
            assert!((dm[(i, j)] - d).abs() < 1e-12 && dm[(j, i)] == dm[(i, j)]);
        }
    }
}

// ─── Inverse at curve level ─────────────────────────────────────────────────

#[test]
fn ab_transform_inverse_recovers_curve() {
    let m = 401;
    let beta = sample(m, |t| (t + 0.3, 0.4 * (4.0 * t).sin() + 0.2));
    for &(a, b) in &AB {
        let f = ab_transform(&beta, a, b).unwrap();
        let back = ab_transform_inverse(&f, a, b).unwrap();
        // the recovered curve has the same transform (right inverse) ...
        let f2 = ab_transform(&back, a, b).unwrap();
        // (interior samples: re-differentiating with one-sided end differences
        // is only first order at the two ends)
        let interior = (2..m - 2)
            .flat_map(|j| (0..2).map(move |r| (r, j)))
            .map(|(r, j)| (f[(r, j)] - f2[(r, j)]).abs())
            .fold(0.0, f64::max);
        assert!(interior < 1e-4, "a={a} b={b}: {interior}");
        // ... and is β − β(0) (initial tangent angle ≈ 1.0 rad, inside the
        // branch for all λ ≤ 2); error = central-difference + trapezoid O(h²).
        let mut shifted = beta.clone();
        for j in 0..m {
            shifted[(0, j)] -= beta[(0, 0)];
            shifted[(1, j)] -= beta[(1, 0)];
        }
        assert!(max_abs_diff(&back, &shifted) < 1e-3, "a={a} b={b}");
    }
}

// ─── Karcher mean under the (a,b) metric ────────────────────────────────────

#[test]
fn karcher_mean_recovers_common_shape() {
    // Rotated, scaled, translated and warped copies of one shape: the Karcher
    // mean is that shape and all curves are (nearly) at distance 0.
    let m = 81;
    let base = |t: f64| (t, 0.3 * (3.0 * t).sin() + 0.1 * t * t);
    let template = sample(m, base);
    let curves: Vec<FdMatrix> = (0..5)
        .map(|i| {
            let e = 0.15 * (i as f64 - 2.0);
            let gam = move |t: f64| t + e * (PI * t).sin() / PI;
            let c = sample(m, |t| base(gam(t)));
            let mut c = rotate_curve(&c, 0.3 * i as f64 - 0.5);
            for j in 0..m {
                c[(0, j)] = (1.0 + 0.2 * i as f64) * c[(0, j)] + i as f64;
                c[(1, j)] = (1.0 + 0.2 * i as f64) * c[(1, j)] - 1.0;
            }
            c
        })
        .collect();
    let set = curves_to_set(&curves).unwrap();
    for &(a, b) in &[(1.0, 1.0), (1.0, 2.0), (3.0, 1.0)] {
        let cfg = cfg_with(elastic(a, b), true, true, true);
        let km = curve_shape_karcher_mean(&set, &cfg, 20, 1e-4).unwrap();
        // residuals at the DP floor (see `reparametrisation_invariance`)
        assert!(km.distances.iter().all(|&d| d < 0.08), "{:?}", km.distances);
        let d = curve_shape_distance(&km.mean, &template, &cfg).unwrap();
        assert!(d < 0.05, "a={a} b={b}: mean-to-template {d}");
        // the mean is returned at unit length (scale = true)
        let len = geometry::curve_length(&km.mean, &Grid::new(m));
        assert!((len - 1.0).abs() < 1e-2, "length {len}");
    }
}

#[test]
fn karcher_mean_elastic_one_half_matches_srvf() {
    let sh = shapes(41);
    let set = curves_to_set(&sh).unwrap();
    let s = curve_shape_karcher_mean(
        &set,
        &cfg_with(CurveShapeMetric::Srvf, true, true, true),
        5,
        1e-6,
    )
    .unwrap();
    let e = curve_shape_karcher_mean(
        &set,
        &cfg_with(elastic(1.0, 0.5), true, true, true),
        5,
        1e-6,
    )
    .unwrap();
    assert_eq!(s.n_iter, e.n_iter);
    assert!(max_abs_diff(&s.mean, &e.mean) < 1e-8);
    for (x, y) in s.distances.iter().zip(&e.distances) {
        assert!((x - y).abs() < 1e-8);
    }
}

#[test]
fn downstream_runs_under_elastic_metric() {
    let sh = shapes(41);
    let set = curves_to_set(&sh).unwrap();
    let cfg = cfg_with(elastic(1.0, 1.0), true, true, true);
    let km = curve_shape_karcher_mean(&set, &cfg, 5, 1e-4).unwrap();
    let pca = curve_shape_pca(&km, 2).unwrap();
    let mode = pca.mode_shape(&km, 0, 0.0).unwrap();
    // coef = 0 gives (approximately — exp of the residual mean tangent) the mean
    assert!(curve_shape_distance(&mode, &km.mean, &cfg).unwrap() < 0.1);
    let med = curve_shape_karcher_median(&set, &cfg, 5, 1e-4).unwrap();
    let bp = curve_shape_boxplot(&med, &CurveShapeBoxplotConfig::default()).unwrap();
    // the box-plot median is reconstructed with the F-transform inverse
    assert!(max_abs_diff(&bp.median, &med.mean) < 1e-12);
}

// ─── Validation ─────────────────────────────────────────────────────────────

#[test]
fn elastic_metric_validation() {
    let c = sample(20, gc1);
    let mut cfg = cfg_with(elastic(1.0, 1.0), true, true, true);
    cfg.closed = true;
    assert!(cfg.validate().is_err());
    assert!(curve_shape_distance(&c, &c, &cfg).is_err());
    for (a, b) in [
        (0.0, 1.0),
        (1.0, -1.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
    ] {
        let cfg = cfg_with(elastic(a, b), true, true, true);
        assert!(cfg.validate().is_err());
        assert!(ab_transform(&c, a, b).is_err());
    }
    // planar curves only
    let c3 = FdMatrix::zeros(3, 20);
    let cfg = cfg_with(elastic(1.0, 1.0), true, true, true);
    assert!(matches!(
        curve_shape_distance(&c3, &c3, &cfg),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(ab_transform(&c3, 1.0, 1.0).is_err());
    assert!(ab_transform_inverse(&c3, 1.0, 1.0).is_err());
    let set = curves_to_set(&[c3.clone(), c3]).unwrap();
    assert!(curve_shape_karcher_mean(&set, &cfg, 3, 1e-3).is_err());
}
