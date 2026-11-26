//! Tests of the exact piecewise-linear `(a, b)` elastic distance
//! (Bauer et al. 2024 §4.1, GH #68).
//!
//! Independent references: closed forms (one segment against a polygon,
//! scaled copies, subdivisions), an independent brute-force dynamic program
//! over straight-line matchings evaluated *exactly* on the rectangular weight
//! (every such matching is admissible, so its value is a lower bound of the
//! supremum that converges as the grid is refined), and the F-transform DP of
//! [`curve_shape_distance`] on densely sampled polygons.

use super::*;
use std::f64::consts::PI;

/// `d × n` vertex matrix from vertex rows.
fn poly(pts: &[[f64; 2]]) -> FdMatrix {
    let mut out = FdMatrix::zeros(2, pts.len());
    for (j, p) in pts.iter().enumerate() {
        out[(0, j)] = p[0];
        out[(1, j)] = p[1];
    }
    out
}

/// Polygon from turning angles and segment lengths, starting at the origin.
fn turtle(angles: &[f64], lengths: &[f64]) -> FdMatrix {
    let mut pts = vec![[0.0, 0.0]];
    for (&a, &l) in angles.iter().zip(lengths) {
        let p = pts[pts.len() - 1];
        pts.push([p[0] + l * a.cos(), p[1] + l * a.sin()]);
    }
    poly(&pts)
}

fn lam(a: f64, b: f64) -> f64 {
    a / (2.0 * b)
}

fn kappa(theta: f64, lam: f64) -> f64 {
    if lam * theta < PI / 2.0 {
        (lam * theta).cos()
    } else {
        0.0
    }
}

fn seg_data(c: &FdMatrix) -> (Vec<f64>, Vec<f64>) {
    let n = c.ncols();
    let mut ang = Vec::new();
    let mut len = Vec::new();
    for j in 1..n {
        let (dx, dy) = (c[(0, j)] - c[(0, j - 1)], c[(1, j)] - c[(1, j - 1)]);
        ang.push(dy.atan2(dx));
        len.push(dx.hypot(dy));
    }
    (ang, len)
}

fn angle_between(a1: f64, a2: f64) -> f64 {
    let d = (a1 - a2).rem_euclid(2.0 * PI);
    d.min(2.0 * PI - d)
}

/// Independent lower bound of the supremum `S`: best monotone path through a
/// `n1 × n2` node grid (uniform nodes ∪ polygon vertices) with straight moves
/// `(di, dj)`, `0 ≤ di, dj ≤ k`, each evaluated exactly by splitting it at the
/// cell boundaries.
fn brute_force_similarity(c1: &FdMatrix, c2: &FdMatrix, lam: f64, n: usize, k: usize) -> f64 {
    let (a1, l1) = seg_data(c1);
    let (a2, l2) = seg_data(c2);
    let cum = |l: &[f64]| {
        let mut s = vec![0.0];
        for &v in l {
            s.push(s[s.len() - 1] + v);
        }
        s
    };
    let (s, t) = (cum(&l1), cum(&l2));
    let (big1, big2) = (s[s.len() - 1], t[t.len() - 1]);
    let nodes = |br: &[f64], total: f64| {
        let mut v: Vec<f64> = (0..=n).map(|i| total * i as f64 / n as f64).collect();
        v.extend_from_slice(br);
        v.sort_by(f64::total_cmp);
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-14 * total);
        v
    };
    let (xs, ys) = (nodes(&s, big1), nodes(&t, big2));
    let cell = |br: &[f64], x: f64| br.iter().rposition(|&b| b <= x).unwrap().min(br.len() - 2);
    let w = |x: f64, y: f64| kappa(angle_between(a1[cell(&s, x)], a2[cell(&t, y)]), lam);
    // exact value of the straight move (x0, y0) → (x1, y1)
    let seg_value = |x0: f64, y0: f64, x1: f64, y1: f64| -> f64 {
        if x1 == x0 || y1 == y0 {
            return 0.0;
        }
        let mut us = vec![0.0, 1.0];
        for &b in &s {
            if b > x0 && b < x1 {
                us.push((b - x0) / (x1 - x0));
            }
        }
        for &b in &t {
            if b > y0 && b < y1 {
                us.push((b - y0) / (y1 - y0));
            }
        }
        us.sort_by(f64::total_cmp);
        us.windows(2)
            .map(|p| {
                let (ua, ub) = (p[0], p[1]);
                let um = 0.5 * (ua + ub);
                let (dx, dy) = ((ub - ua) * (x1 - x0), (ub - ua) * (y1 - y0));
                w(x0 + um * (x1 - x0), y0 + um * (y1 - y0)) * (dx * dy).sqrt()
            })
            .sum()
    };
    let (nx, ny) = (xs.len(), ys.len());
    let mut h = vec![f64::NEG_INFINITY; nx * ny];
    h[0] = 0.0;
    for j in 0..ny {
        for i in 0..nx {
            if i == 0 && j == 0 {
                continue;
            }
            let mut best = f64::NEG_INFINITY;
            for di in 0..=k.min(i) {
                for dj in 0..=k.min(j) {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let prev = h[(i - di) + nx * (j - dj)];
                    if prev == f64::NEG_INFINITY {
                        continue;
                    }
                    let v = prev + seg_value(xs[i - di], ys[j - dj], xs[i], ys[j]);
                    best = best.max(v);
                }
            }
            h[i + nx * j] = best;
        }
    }
    h[nx * ny - 1]
}

fn dist_from_similarity(sim: f64, l1: f64, l2: f64, b: f64) -> f64 {
    2.0 * b * (l1 + l2 - 2.0 * sim).max(0.0).sqrt()
}

/// Test polygons: gently turning (all tangent angles within ±0.7 rad).
fn poly_a() -> FdMatrix {
    turtle(&[0.1, 0.5, -0.2, 0.3], &[1.0, 0.6, 0.9, 0.5])
}
fn poly_b() -> FdMatrix {
    turtle(&[-0.3, 0.2, 0.6, -0.1, 0.4], &[0.7, 0.8, 0.4, 0.6, 0.5])
}
fn poly_c() -> FdMatrix {
    turtle(&[0.7, -0.6, 0.0], &[0.5, 1.1, 0.8])
}

const AB: [(f64, f64); 5] = [(1.0, 0.5), (1.0, 2.0), (1.0, 1.0), (3.0, 1.0), (2.0, 0.5)];

// ─── Closed forms ───────────────────────────────────────────────────────────

#[test]
fn two_single_segments() {
    for &(a, b) in &AB {
        for theta in [0.0f64, 0.4, 1.0, 2.0, 3.0] {
            let c1 = poly(&[[0.0, 0.0], [1.3, 0.0]]);
            let c2 = poly(&[
                [1.0, 1.0],
                [1.0 + 0.7 * theta.cos(), 1.0 + 0.7 * theta.sin()],
            ]);
            let d = ab_elastic_distance_pl(&c1, &c2, a, b).unwrap();
            let k = kappa(theta, lam(a, b));
            let expect = 2.0 * b * (1.3 + 0.7 - 2.0 * k * (1.3f64 * 0.7).sqrt()).sqrt();
            assert!(
                (d - expect).abs() < 1e-12,
                "a={a} b={b} θ={theta}: {d} vs {expect}"
            );
        }
    }
}

#[test]
fn one_segment_against_a_polygon() {
    // S = √(L Σ_j κ_j² ℓ_j) (Cauchy–Schwarz over the allocation of the segment)
    let line = poly(&[[0.0, 0.0], [1.7, 0.0]]);
    let targets = [
        turtle(&[0.3, -0.5, 1.2, 0.1], &[0.4, 0.9, 0.3, 0.7]),
        turtle(&[2.5, 0.2, -2.9], &[0.5, 0.5, 0.6]), // includes κ = 0 cells
    ];
    for &(a, b) in &AB {
        for c2 in &targets {
            let (ang, len) = seg_data(c2);
            let sum: f64 = ang
                .iter()
                .zip(&len)
                .map(|(&th, &l)| kappa(th.abs(), lam(a, b)).powi(2) * l)
                .sum();
            let l2: f64 = len.iter().sum();
            let expect = dist_from_similarity((1.7 * sum).sqrt(), 1.7, l2, b);
            for (x, y) in [(&line, c2), (c2, &line)] {
                let r = ab_elastic_match_pl(x, y, a, b).unwrap();
                assert!(
                    (r.distance - expect).abs() < 1e-12,
                    "a={a} b={b}: {} vs {expect}",
                    r.distance
                );
            }
        }
    }
}

#[test]
fn scaled_copy_and_subdivision() {
    let c = poly_a();
    for &(a, b) in &AB {
        // c2 = 2.5·c + shift: d = 2b |√L1 − √L2|
        let mut c2 = c.clone();
        for j in 0..c2.ncols() {
            c2[(0, j)] = 2.5 * c[(0, j)] + 1.0;
            c2[(1, j)] = 2.5 * c[(1, j)] - 3.0;
        }
        let r = ab_elastic_match_pl(&c, &c2, a, b).unwrap();
        let expect = 2.0 * b * (r.length1.sqrt() - r.length2.sqrt()).abs();
        assert!(
            (r.distance - expect).abs() < 1e-7,
            "{} vs {expect}",
            r.distance
        );
        // subdividing segments (extra collinear vertices) changes nothing
        let mut pts = Vec::new();
        for j in 0..c.ncols() {
            if j > 0 {
                for f in [0.3, 0.55] {
                    pts.push([
                        c[(0, j - 1)] + f * (c[(0, j)] - c[(0, j - 1)]),
                        c[(1, j - 1)] + f * (c[(1, j)] - c[(1, j - 1)]),
                    ]);
                }
            }
            pts.push([c[(0, j)], c[(1, j)]]);
        }
        // and repeated vertices are ignored
        pts.insert(3, pts[3]);
        let d = ab_elastic_distance_pl(&c, &poly(&pts), a, b).unwrap();
        assert!(d < 1e-7, "subdivision: {d}");
    }
}

// ─── Metric properties ──────────────────────────────────────────────────────

#[test]
fn symmetric_translation_invariant_and_scaling() {
    let curves = [poly_a(), poly_b(), poly_c()];
    for &(a, b) in &AB {
        for x in &curves {
            for y in &curves {
                let dxy = ab_elastic_distance_pl(x, y, a, b).unwrap();
                let dyx = ab_elastic_distance_pl(y, x, a, b).unwrap();
                assert!((dxy - dyx).abs() < 1e-12, "{dxy} vs {dyx}");
                let mut ys = y.clone();
                for j in 0..ys.ncols() {
                    ys[(0, j)] += 5.0;
                }
                let dt = ab_elastic_distance_pl(x, &ys, a, b).unwrap();
                assert!((dt * dt - dxy * dxy).abs() < 1e-12);
                // d(s c1, s c2) = √s d(c1, c2)
                let s = 3.0;
                let ds =
                    ab_elastic_distance_pl(&geometry_scaled(x, s), &geometry_scaled(y, s), a, b)
                        .unwrap();
                assert!((ds * ds - s * dxy * dxy).abs() < 1e-12 * (1.0 + s * dxy * dxy));
            }
        }
        // triangle inequality
        let d = |i: usize, j: usize| ab_elastic_distance_pl(&curves[i], &curves[j], a, b).unwrap();
        assert!(d(0, 2) <= d(0, 1) + d(1, 2) + 1e-12);
        assert!(d(0, 1) <= d(0, 2) + d(2, 1) + 1e-12);
    }
}

fn geometry_scaled(c: &FdMatrix, s: f64) -> FdMatrix {
    let data: Vec<f64> = c.as_slice().iter().map(|v| v * s).collect();
    FdMatrix::from_column_major(data, c.nrows(), c.ncols()).unwrap()
}

#[test]
fn matching_path_reproduces_the_similarity() {
    for &(a, b) in &AB {
        for (c1, c2) in [
            (poly_a(), poly_b()),
            (poly_c(), poly_a()),
            (poly_b(), poly_c()),
        ] {
            let r = ab_elastic_match_pl(&c1, &c2, a, b).unwrap();
            assert_eq!(r.path[0], (0.0, 0.0));
            assert_eq!(*r.path.last().unwrap(), (1.0, 1.0));
            // monotone, and the exact value of the polyline equals the similarity
            let (a1, l1) = seg_data(&c1);
            let (a2, l2) = seg_data(&c2);
            let cum = |l: &[f64]| {
                let mut s = vec![0.0];
                for &v in l {
                    s.push(s[s.len() - 1] + v);
                }
                s
            };
            let (s, t) = (cum(&l1), cum(&l2));
            let cell =
                |br: &[f64], x: f64| br.iter().rposition(|&v| v <= x).unwrap().min(br.len() - 2);
            let mut val = 0.0;
            for p in r.path.windows(2) {
                let (x0, y0) = (p[0].0 * r.length1, p[0].1 * r.length2);
                let (x1, y1) = (p[1].0 * r.length1, p[1].1 * r.length2);
                assert!(x1 >= x0 - 1e-15 && y1 >= y0 - 1e-15, "not monotone");
                let (dx, dy) = ((x1 - x0).max(0.0), (y1 - y0).max(0.0));
                if dx > 0.0 && dy > 0.0 {
                    let (xm, ym) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
                    let k = kappa(angle_between(a1[cell(&s, xm)], a2[cell(&t, ym)]), lam(a, b));
                    val += k * (dx * dy).sqrt();
                }
            }
            assert!(
                (val - r.similarity).abs() < 1e-9 * r.similarity.max(1.0),
                "a={a} b={b}: path value {val} vs {}",
                r.similarity
            );
        }
    }
}

// ─── Against independent dynamic programs ───────────────────────────────────

#[test]
fn exact_dominates_and_is_approached_by_a_brute_force_dp() {
    for &(a, b) in &AB {
        for (c1, c2) in [(poly_a(), poly_b()), (poly_c(), poly_b())] {
            let r = ab_elastic_match_pl(&c1, &c2, a, b).unwrap();
            let mut gaps = Vec::new();
            // node spacing and slope set both refined (moves up to k nodes)
            for (n, k) in [(8, 4), (16, 8), (48, 24)] {
                let sim = brute_force_similarity(&c1, &c2, lam(a, b), n, k);
                // every DP path is admissible: never above the exact supremum
                assert!(
                    sim <= r.similarity + 1e-12,
                    "a={a} b={b} n={n}: {sim} > {}",
                    r.similarity
                );
                gaps.push(r.similarity - sim);
            }
            assert!(
                gaps[2] <= gaps[1] + 1e-12 && gaps[1] <= gaps[0] + 1e-12,
                "a={a} b={b}: {gaps:?}"
            );
            let prev_gap = gaps[2];
            assert!(
                prev_gap < 2e-3 * r.similarity,
                "a={a} b={b}: brute-force gap {prev_gap} (S = {})",
                r.similarity
            );
            let d_bf = dist_from_similarity(r.similarity - prev_gap, r.length1, r.length2, b);
            assert!(d_bf >= r.distance - 1e-12);
        }
    }
}

#[test]
fn two_by_two_segments_hand_case() {
    // c1: two unit segments at angles 0, π/2; c2: two unit segments at π/4, π/4 + δ.
    // Brute force on a fine grid (n = 64 nodes, moves ≤ 32) agrees with the exact value.
    let c1 = turtle(&[0.0, PI / 2.0], &[1.0, 1.0]);
    let c2 = turtle(&[PI / 4.0, PI / 4.0 + 0.3], &[0.8, 1.4]);
    for &(a, b) in &AB {
        let r = ab_elastic_match_pl(&c1, &c2, a, b).unwrap();
        let sim = brute_force_similarity(&c1, &c2, lam(a, b), 64, 32);
        assert!(sim <= r.similarity + 1e-12);
        assert!(
            r.similarity - sim < 1e-4,
            "a={a} b={b}: {} vs {sim}",
            r.similarity
        );
    }
}

#[test]
fn exact_lower_bounds_the_curve_shape_dp_and_is_approached() {
    // Densely sampled polygons (uniform in arc length, vertices included) under
    // the F-transform DP of curve_shape (rotation = false, scale = false). The
    // tangent angles stay within ±0.7 rad, so the lifted angle difference is
    // below π/(2λ) for every λ ≤ 2 and the DP integrand never exceeds the
    // exact rectangular weight: d_exact ≤ d_DP up to the sampling error.
    let sample_poly = |c: &FdMatrix, m: usize| -> FdMatrix {
        let (ang, len) = seg_data(c);
        let total: f64 = len.iter().sum();
        let mut out = FdMatrix::zeros(2, m);
        for j in 0..m {
            let mut s = total * j as f64 / (m - 1) as f64;
            let (mut x, mut y) = (c[(0, 0)], c[(1, 0)]);
            for (&th, &l) in ang.iter().zip(&len) {
                let step = s.min(l);
                x += step * th.cos();
                y += step * th.sin();
                s -= step;
                if s <= 0.0 {
                    break;
                }
            }
            out[(0, j)] = x;
            out[(1, j)] = y;
        }
        out
    };
    for &(a, b) in &[(1.0, 0.5), (1.0, 1.0), (3.0, 1.0)] {
        let (c1, c2) = (poly_a(), poly_b());
        let exact = ab_elastic_distance_pl(&c1, &c2, a, b).unwrap();
        let cfg = CurveShapeConfig {
            metric: CurveShapeMetric::Elastic { a, b },
            rotation: false,
            scale: false,
            ..CurveShapeConfig::default()
        };
        let mut gaps = Vec::new();
        for m in [101, 401] {
            let d = curve_shape_distance(&sample_poly(&c1, m), &sample_poly(&c2, m), &cfg).unwrap();
            gaps.push((d - exact) / exact);
        }
        // The sampled polygon's transform (central differences) rounds the
        // corners, an O(1/m) error that can push the DP slightly below the
        // exact value; it shrinks with m, and the remaining gap is the DP
        // slope-quantisation floor (single-digit percent, cf. Bauer et al. Fig. 2).
        assert!(gaps[1].abs() < gaps[0].abs(), "a={a} b={b}: {gaps:?}");
        assert!(gaps[0] > -0.03 && gaps[1] > -5e-3, "a={a} b={b}: {gaps:?}");
        assert!(gaps[1] < 0.05, "a={a} b={b}: {gaps:?}");
    }
}

// ─── Higher dimensions and validation ───────────────────────────────────────

#[test]
fn space_curves_and_srvf_lambda() {
    // R³ polygons: symmetric, and a = 1, b = 1/2 uses κ = cos θ clipped at 0
    let mut c1 = FdMatrix::zeros(3, 4);
    let mut c2 = FdMatrix::zeros(3, 3);
    let p1 = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.2],
        [1.5, 1.2, 1.0],
    ];
    let p2 = [[0.0, 0.0, 0.0], [0.5, 0.6, 0.1], [0.9, 1.4, 0.9]];
    for (j, p) in p1.iter().enumerate() {
        for k in 0..3 {
            c1[(k, j)] = p[k];
        }
    }
    for (j, p) in p2.iter().enumerate() {
        for k in 0..3 {
            c2[(k, j)] = p[k];
        }
    }
    let d12 = ab_elastic_distance_pl(&c1, &c2, 1.0, 0.5).unwrap();
    let d21 = ab_elastic_distance_pl(&c2, &c1, 1.0, 0.5).unwrap();
    assert!(d12 > 0.0 && (d12 - d21).abs() < 1e-12);
    assert!(ab_elastic_distance_pl(&c1, &c1, 1.0, 0.5).unwrap() < 1e-7);
}

#[test]
fn pl_validation() {
    let c = poly_a();
    assert!(ab_elastic_distance_pl(&c, &c, 0.0, 1.0).is_err());
    assert!(ab_elastic_distance_pl(&c, &c, 1.0, f64::NAN).is_err());
    assert!(ab_elastic_distance_pl(&c, &FdMatrix::zeros(3, 4), 1.0, 1.0).is_err());
    assert!(ab_elastic_distance_pl(&c, &FdMatrix::zeros(2, 1), 1.0, 1.0).is_err());
    assert!(matches!(
        ab_elastic_distance_pl(&c, &FdMatrix::zeros(2, 3), 1.0, 1.0),
        Err(FdarError::ComputationFailed { .. })
    ));
    let mut bad = c.clone();
    bad[(0, 1)] = f64::INFINITY;
    assert!(ab_elastic_distance_pl(&bad, &c, 1.0, 1.0).is_err());
}
