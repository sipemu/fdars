use super::*;
use crate::metric::dtw::dtw_distance_windowed;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const X10: [f64; 10] = [0.5, -1.2, 0.3, 2.0, 1.1, -0.4, 0.0, 0.9, -2.1, 0.7];
const Y10: [f64; 10] = [0.1, 0.4, -0.8, 1.5, 2.2, 0.3, -0.6, 0.2, 0.5, -1.0];

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1.0)
}

fn random_walk(rng: &mut StdRng, n: usize) -> Vec<f64> {
    let mut v = rng.gen_range(-1.0..1.0);
    (0..n)
        .map(|_| {
            v += rng.gen_range(-1.0..1.0);
            v
        })
        .collect()
}

fn naive_envelope(y: &[f64], rows: &[(usize, usize)]) -> (Vec<f64>, Vec<f64>) {
    rows.iter()
        .map(|&(lo, hi)| {
            let s = &y[lo..hi];
            (
                s.iter().copied().fold(f64::INFINITY, f64::min),
                s.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .unzip()
}

/// Windows exercised by the invariant sweeps (only feasible ones are used).
fn windows() -> Vec<DtwWindow> {
    vec![
        DtwWindow::None,
        DtwWindow::SakoeChiba { radius: 0 },
        DtwWindow::SakoeChiba { radius: 1 },
        DtwWindow::SakoeChiba { radius: 2 },
        DtwWindow::SakoeChiba { radius: 5 },
        DtwWindow::Itakura { max_slope: 2.0 },
        DtwWindow::Itakura { max_slope: 3.0 },
        DtwWindow::SakoeChibaFraction { window: 0.1 },
        DtwWindow::SakoeChibaFraction { window: 0.3 },
        DtwWindow::ItakuraFraction { max_slope: 0.5 },
    ]
}

// --- tslearn / dtaidistance goldens -------------------------------------------

#[test]
fn tslearn_lb_envelope_golden() {
    // tslearn 0.10.0: tslearn.metrics.lb_envelope(Y10, radius=r) -> (down, up)
    let golden: [(usize, [f64; 10], [f64; 10]); 4] = [
        (0, Y10, Y10),
        (
            1,
            [0.1, -0.8, -0.8, -0.8, 0.3, -0.6, -0.6, -0.6, -1.0, -1.0],
            [0.4, 0.4, 1.5, 2.2, 2.2, 2.2, 0.3, 0.5, 0.5, 0.5],
        ),
        (
            2,
            [-0.8, -0.8, -0.8, -0.8, -0.8, -0.6, -0.6, -1.0, -1.0, -1.0],
            [0.4, 1.5, 2.2, 2.2, 2.2, 2.2, 2.2, 0.5, 0.5, 0.5],
        ),
        (
            3,
            [-0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -1.0, -1.0, -1.0, -1.0],
            [1.5, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 0.5, 0.5],
        ),
    ];
    for (r, lo, up) in golden {
        let env = lb_envelope(&Y10, r).unwrap();
        assert_eq!(env.lower, lo.to_vec(), "radius {r}");
        assert_eq!(env.upper, up.to_vec(), "radius {r}");
        let w = lb_envelope_windowed(&Y10, 10, DtwWindow::SakoeChiba { radius: r }).unwrap();
        assert_eq!(w, env);
    }
    // tslearn docstring example.
    let env = lb_envelope(&[1.0, 2.0, 3.0, 2.0, 1.0], 1).unwrap();
    assert_eq!(env.lower, vec![1.0, 1.0, 2.0, 1.0, 1.0]);
    assert_eq!(env.upper, vec![2.0, 3.0, 3.0, 3.0, 2.0]);
}

#[test]
fn tslearn_lb_keogh_golden() {
    // tslearn 0.10.0: float(tslearn.metrics.lb_keogh(X10, Y10, radius=r)) (= sqrt of
    // our p = 2 value) and tslearn.metrics.dtw(X10, Y10, global_constraint=
    // "sakoe_chiba", sakoe_chiba_radius=r) (= sqrt of our DTW).
    let golden = [
        (0usize, 4.047221268969612, 4.047221268969612),
        (1, 1.2569805089976533, 2.6057628441590768),
        (2, 1.2569805089976533, 2.6057628441590768),
        (3, 1.1874342087037917, 2.6057628441590768),
    ];
    for (r, ts_keogh, ts_dtw) in golden {
        let lb = lb_keogh(&X10, &lb_envelope(&Y10, r).unwrap(), 2.0).unwrap();
        assert!(
            close(lb.sqrt(), ts_keogh, 1e-14),
            "r={r}: {} vs {ts_keogh}",
            lb.sqrt()
        );
        assert!(close(lb, ts_keogh * ts_keogh, 1e-13));
        let d =
            dtw_distance_windowed(&X10, &Y10, 2.0, DtwWindow::SakoeChiba { radius: r }).unwrap();
        assert!(close(d.sqrt(), ts_dtw, 1e-14));
        assert!(lb <= d);
    }
    // tslearn docstring: lb_keogh([0]*5, [1, 2, 3, 2, 1], radius=1) = 2.8284… (= sqrt 8).
    let env = lb_envelope(&[1.0, 2.0, 3.0, 2.0, 1.0], 1).unwrap();
    assert_eq!(lb_keogh(&[0.0; 5], &env, 2.0).unwrap(), 8.0);
}

#[test]
fn dtaidistance_lb_keogh_unequal_lengths_golden() {
    // dtaidistance 2.5.1: dtw.lb_keogh(s1, s2, window=w) ** 2 (its window = radius + 1,
    // squared output of its sqrt), for unequal lengths.
    let golden = [
        (1usize, 0.15999999999999992, 8.02),
        (2, 0.15999999999999992, 5.93),
        (3, 0.15999999999999992, 1.86),
    ];
    for (w, short_q, short_c) in golden {
        let win = DtwWindow::SakoeChiba { radius: w - 1 };
        // s1 = X10[:7], s2 = Y10
        let env = lb_envelope_windowed(&Y10, 7, win).unwrap();
        assert!(close(
            lb_keogh(&X10[..7], &env, 2.0).unwrap(),
            short_q,
            1e-13
        ));
        // s1 = X10, s2 = Y10[:6]
        let env = lb_envelope_windowed(&Y10[..6], 10, win).unwrap();
        assert!(close(lb_keogh(&X10, &env, 2.0).unwrap(), short_c, 1e-13));
    }
}

// --- LB_Webb against the authors' reference implementation --------------------

// Data literals (3.14 is a sample value, not π).
#[allow(clippy::approx_constant)]
const S16: [[f64; 16]; 4] = [
    [
        -0.59, -0.92, -2.44, -4.33, -3.27, -2.99, -3.49, -2.83, -0.97, -2.79, -2.44, -3.42, -3.1,
        -3.44, -4.56, -4.49,
    ],
    [
        1.14, 1.7, 1.02, 1.69, 2.88, 3.76, 3.94, 4.14, 4.72, 4.09, 2.86, 1.82, 1.97, -0.23, 0.36,
        1.28,
    ],
    [
        1.09, 2.25, 2.36, 3.14, 3.13, 4.72, 2.86, 3.11, 1.99, 1.28, 0.85, 0.06, -0.61, -0.12, 0.48,
        2.56,
    ],
    [
        -0.85, -1.55, -2.0, -2.43, -1.58, -2.38, -2.84, -1.37, -2.93, -2.79, -1.04, 0.74, 1.12,
        0.77, 0.51, 0.48,
    ],
];

/// `(w, query, candidate, lbKeogh, lbWebb, DTW)` from the authors' Java code
/// (github.com/GIWebb/DTWBounds, `distance.Bounds.lbKeogh` / `lbWebb` with
/// `simpleGetEnvelopes` envelopes and `bsf = Double.MAX_VALUE`;
/// `distance.elastic.DTW.distance(q, t, w)`), squared Euclidean cost.
const WEBB_GOLDEN: [(usize, usize, usize, f64, f64, f64); 36] = [
    (1, 0, 1, 368.7482, 386.02809999999994, 447.15959999999995),
    (
        1,
        0,
        2,
        303.43379999999996,
        354.00340000000006,
        398.07419999999996,
    ),
    (1, 0, 3, 91.10329999999999, 96.4995, 113.14),
    (1, 1, 0, 366.9139, 408.0728, 447.15959999999995),
    (1, 1, 2, 17.1727, 18.4579, 20.874899999999997),
    (
        1,
        1,
        3,
        204.99059999999997,
        266.12760000000003,
        285.73339999999996,
    ),
    (
        1,
        2,
        0,
        309.75139999999993,
        351.7893000000001,
        398.07419999999996,
    ),
    (
        1,
        2,
        1,
        15.643199999999998,
        17.965700000000002,
        20.874899999999997,
    ),
    (1, 2, 3, 173.88349999999997, 221.3204, 239.78980000000004),
    (1, 3, 0, 87.53140000000002, 104.28340000000001, 113.14),
    (1, 3, 1, 232.2534, 238.5236, 285.73339999999996),
    (1, 3, 2, 178.5525, 203.88360000000003, 239.78980000000004),
    (2, 0, 1, 290.9435, 324.0569, 424.5482999999999),
    (
        2,
        0,
        2,
        247.04999999999998,
        322.20220000000006,
        385.71740000000005,
    ),
    (2, 0, 3, 74.31299999999999, 81.4186, 111.2138),
    (2, 1, 0, 292.3231, 349.50970000000007, 424.5482999999999),
    (2, 1, 2, 6.1739, 8.213399999999998, 10.956699999999998),
    (2, 1, 3, 178.6406, 229.24110000000002, 263.4302),
    (2, 2, 0, 239.849, 302.4753000000001, 385.71740000000005),
    (2, 2, 1, 4.834, 8.2721, 10.956699999999998),
    (
        2,
        2,
        3,
        155.94590000000002,
        194.46880000000002,
        231.47470000000004,
    ),
    (2, 3, 0, 67.1035, 93.6108, 111.2138),
    (2, 3, 1, 176.44070000000002, 190.40319999999997, 263.4302),
    (
        2,
        3,
        2,
        138.18500000000003,
        168.98760000000001,
        231.47470000000004,
    ),
    (3, 0, 1, 240.17639999999994, 311.9614, 411.15599999999995),
    (3, 0, 2, 202.19009999999997, 294.4379, 364.05260000000004),
    (3, 0, 3, 58.988600000000005, 74.1542, 109.635),
    (
        3,
        1,
        0,
        238.98149999999998,
        364.47890000000007,
        411.15599999999995,
    ),
    (3, 1, 2, 0.9652999999999999, 3.4408, 7.384100000000001),
    (3, 1, 3, 148.76039999999998, 197.2569, 239.89419999999998),
    (3, 2, 0, 185.378, 313.2965000000001, 364.05260000000004),
    (3, 2, 1, 0.7841000000000001, 3.4408, 7.384100000000001),
    (3, 2, 3, 135.713, 171.38690000000003, 217.06600000000003),
    (3, 3, 0, 51.7503, 84.6581, 109.635),
    (3, 3, 1, 132.32920000000001, 175.7881, 239.89419999999998),
    (3, 3, 2, 102.2177, 155.2554, 217.06600000000003),
];

#[test]
fn lb_webb_matches_reference_implementation() {
    let cost = LocalCost::new(2.0);
    for &(w, qi, ti, keogh, webb, dtw) in &WEBB_GOLDEN {
        let (a, b) = (&S16[qi][..], &S16[ti][..]);
        let window = DtwWindow::SakoeChiba { radius: w };
        let lat = Lattice::new(16, 16, window).unwrap();
        let q = WebbQuery::new(a, &lat);
        let c = WebbCandidate::new(b, &lat);
        // The reference uses the left/right-path variant for w >= 1 and n >= 6.
        let base = min_lr_paths(a, b, &lat, cost).unwrap();
        let ours = webb_pass(&q, b, &c, &lat, cost, (3, base), f64::INFINITY);
        assert!(
            close(ours, webb, 1e-12),
            "w={w} q={qi} t={ti}: {ours} vs {webb}"
        );
        let k = lb_keogh(a, &lb_envelope(b, w).unwrap(), 2.0).unwrap();
        assert!(close(k, keogh, 1e-12));
        let d = dtw_distance_windowed(a, b, 2.0, window).unwrap();
        assert!(close(d, dtw, 1e-12));
        let pub_webb = lb_webb(a, b, 2.0, window).unwrap();
        assert!(pub_webb >= ours && pub_webb >= k && pub_webb <= d);
    }
    // w = 0 (diagonal band): the reference skips the end paths; LB_Webb_NoLR is
    // then exact (the diagonal path is the only path).
    for qi in 0..4 {
        for ti in 0..4 {
            let (a, b) = (&S16[qi][..], &S16[ti][..]);
            let w0 = DtwWindow::SakoeChiba { radius: 0 };
            let d = dtw_distance_windowed(a, b, 2.0, w0).unwrap();
            assert!(close(lb_webb(a, b, 2.0, w0).unwrap(), d, 1e-12));
        }
    }
}

// --- envelopes ------------------------------------------------------------------

#[test]
fn envelopes_match_naive_scan() {
    let mut rng = StdRng::seed_from_u64(58);
    for &(n, m) in &[
        (1usize, 1usize),
        (1, 5),
        (5, 1),
        (12, 12),
        (17, 9),
        (9, 23),
        (40, 40),
    ] {
        let y = random_walk(&mut rng, m);
        for w in windows() {
            let Ok(rows) = w.row_bounds(n, m) else {
                continue;
            };
            let env = lb_envelope_windowed(&y, n, w).unwrap();
            let (lo, up) = naive_envelope(&y, &rows);
            assert_eq!(env.lower, lo, "{w:?} {n}x{m}");
            assert_eq!(env.upper, up, "{w:?} {n}x{m}");
            // Column ranges: exact admissible rows of every column.
            let lat = Lattice::new(n, m, w).unwrap();
            let mask = w.mask(n, m).unwrap();
            for j in 0..m {
                let (a, b) = lat.cols[j];
                for (i, row) in mask.iter().enumerate() {
                    assert_eq!(row[j], a <= i && i < b, "{w:?} {n}x{m} col {j} row {i}");
                }
            }
        }
        for r in [0usize, 1, 3, usize::MAX] {
            let x = random_walk(&mut rng, n);
            let env = lb_envelope(&x, r).unwrap();
            let rows: Vec<(usize, usize)> = (0..n)
                .map(|i| {
                    (
                        i.saturating_sub(r),
                        i.saturating_add(r).saturating_add(1).min(n),
                    )
                })
                .collect();
            assert_eq!((env.lower, env.upper), naive_envelope(&x, &rows));
        }
    }
}

#[test]
fn sliding_extrema_non_monotone_fallback() {
    let v = [3.0, -1.0, 4.0, 1.0, 5.0];
    let bounds = [(2, 5), (0, 2), (1, 4)];
    let (lo, up) = sliding_extrema(&v, &bounds);
    assert_eq!(lo, vec![1.0, -1.0, -1.0]);
    assert_eq!(up, vec![5.0, 3.0, 4.0]);
}

// --- LB_Kim -----------------------------------------------------------------------

#[test]
fn lb_kim_hand_cases() {
    // first+last = (0-1)^2 + (2-0)^2 = 5; max: (3-1)^2 = 4; min: (0-0)^2 = 0.
    assert_eq!(
        lb_kim(&[0.0, 3.0, 2.0], &[1.0, 0.0, 0.0], 2.0).unwrap(),
        5.0
    );
    // max term dominates: (3 - 0.5)^2.
    let x = [0.0, 3.0, 1.0, 2.0];
    let y = [0.5, 0.0, 0.25, 0.0];
    assert_eq!(lb_kim(&x, &y, 2.0).unwrap(), 6.25);
    assert_eq!(lb_kim(&x, &y, 1.0).unwrap(), 2.5); // max(0.5 + 2, 2.5, 0)
                                                   // single-element series: one corner cell, equals DTW.
    assert_eq!(lb_kim(&[2.0], &[5.0], 2.0).unwrap(), 9.0);
    assert_eq!(lb_kim(&[2.0], &[5.0, 2.0], 2.0).unwrap(), 9.0);
}

// --- invariants ----------------------------------------------------------------

#[test]
fn bounds_sandwich_dtw_on_random_pairs() {
    let mut rng = StdRng::seed_from_u64(2021);
    let sizes = [
        (16usize, 16usize),
        (30, 30),
        (7, 7),
        (20, 13),
        (11, 25),
        (3, 3),
        (6, 9),
    ];
    let mut tighter = 0usize;
    let mut checked = 0usize;
    for _rep in 0..12 {
        for &(n, m) in &sizes {
            let x = random_walk(&mut rng, n);
            let y = random_walk(&mut rng, m);
            for w in windows() {
                if w.row_bounds(n, m).is_err() {
                    continue;
                }
                for &p in &[1.0, 2.0, 1.5, 3.0] {
                    let d = dtw_distance_windowed(&x, &y, p, w).unwrap();
                    let tol = 1e-12 * d.max(1.0);
                    let kim = lb_kim(&x, &y, p).unwrap();
                    let keogh = lb_keogh(&x, &lb_envelope_windowed(&y, n, w).unwrap(), p).unwrap();
                    let webb = lb_webb(&x, &y, p, w).unwrap();
                    assert!(
                        kim <= d + tol,
                        "kim {kim} > dtw {d} ({w:?}, p={p}, {n}x{m})"
                    );
                    assert!(keogh <= webb, "keogh {keogh} > webb {webb} ({w:?}, p={p})");
                    assert!(
                        webb <= d + tol,
                        "webb {webb} > dtw {d} ({w:?}, p={p}, {n}x{m})"
                    );
                    tighter += usize::from(webb > keogh);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1500, "checked {checked}");
    assert!(
        tighter * 2 > checked,
        "LB_Webb tighter only {tighter}/{checked}"
    );
}

#[test]
fn bounds_sandwich_dtw_subconvex_p() {
    // p < 1: LB_Kim and LB_Keogh stay valid (monotone cost); LB_Webb is rejected.
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..40 {
        let (x, y) = (random_walk(&mut rng, 15), random_walk(&mut rng, 15));
        for w in windows() {
            let d = dtw_distance_windowed(&x, &y, 0.5, w).unwrap();
            assert!(lb_kim(&x, &y, 0.5).unwrap() <= d + 1e-12 * d.max(1.0));
            let env = lb_envelope_windowed(&y, 15, w).unwrap();
            assert!(lb_keogh(&x, &env, 0.5).unwrap() <= d + 1e-12 * d.max(1.0));
        }
    }
    assert!(matches!(
        lb_webb(&[1.0, 2.0], &[1.0, 2.0], 0.5, DtwWindow::None),
        Err(FdarError::InvalidParameter { .. })
    ));
}

#[test]
fn lb_webb_exact_on_identical_and_tight_cases() {
    let x = [0.0, 1.0, 3.0, 2.0, 0.0, -1.0, -2.0, 0.0];
    assert_eq!(
        lb_webb(&x, &x, 2.0, DtwWindow::SakoeChiba { radius: 2 }).unwrap(),
        0.0
    );
    // Constant series: the only path cost is n·|c1 − c2|^p, which LB_Webb reaches.
    let a = [1.0; 9];
    let b = [3.0; 9];
    let w = DtwWindow::SakoeChiba { radius: 2 };
    let d = dtw_distance_windowed(&a, &b, 2.0, w).unwrap();
    assert_eq!(d, 36.0);
    assert!(
        lb_webb(&a, &b, 2.0, w).unwrap()
            >= lb_keogh(&a, &lb_envelope(&b, 2).unwrap(), 2.0).unwrap()
    );
    assert!(lb_webb(&a, &b, 2.0, w).unwrap() <= d);
}

// --- EAPrunedDTW -------------------------------------------------------------------

#[test]
fn early_abandon_is_exact_and_abandons() {
    let mut rng = StdRng::seed_from_u64(99);
    let sizes = [
        (1usize, 1usize),
        (1, 6),
        (6, 1),
        (12, 12),
        (25, 18),
        (18, 25),
        (40, 40),
    ];
    for _rep in 0..6 {
        for &(n, m) in &sizes {
            let x = random_walk(&mut rng, n);
            let y = random_walk(&mut rng, m);
            for w in windows() {
                if w.row_bounds(n, m).is_ok() {
                    check_early_abandon_case(&x, &y, w);
                }
            }
        }
    }
}

/// Exactness (cutoff ≥ DTW → identical bits) and abandonment (cutoff < DTW → `None`)
/// for one pair of series under one window, over several exponents `p`.
fn check_early_abandon_case(x: &[f64], y: &[f64], w: DtwWindow) {
    let (n, m) = (x.len(), y.len());
    for &p in &[1.0, 2.0, 1.5, 0.5] {
        let d = dtw_distance_windowed(x, y, p, w).unwrap();
        for cutoff in [f64::INFINITY, d, d * 1.5 + 1e-9, d + 1.0] {
            let ea = dtw_distance_early_abandon(x, y, p, w, cutoff).unwrap();
            assert_eq!(
                ea.map(f64::to_bits),
                Some(d.to_bits()),
                "{w:?} p={p} {n}x{m} cutoff={cutoff}"
            );
        }
        if d > 0.0 {
            for cutoff in [d.next_down_compat(), d * 0.5, 0.0, -1.0] {
                let ea = dtw_distance_early_abandon(x, y, p, w, cutoff).unwrap();
                assert_eq!(ea, None, "{w:?} p={p} {n}x{m} cutoff={cutoff}");
            }
        }
    }
}

/// `f64::next_down` is stable only since Rust 1.86 (MSRV is 1.81).
trait NextDown {
    fn next_down_compat(self) -> f64;
}

impl NextDown for f64 {
    fn next_down_compat(self) -> f64 {
        debug_assert!(self.is_finite() && self > 0.0);
        f64::from_bits(self.to_bits() - 1)
    }
}

#[test]
fn early_abandon_errors() {
    let w = DtwWindow::None;
    assert!(matches!(
        dtw_distance_early_abandon(&[], &[1.0], 2.0, w, 1.0),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_distance_early_abandon(&[1.0], &[1.0], 2.0, w, f64::NAN),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        dtw_distance_early_abandon(&[1.0], &[1.0], -1.0, w, 1.0),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        dtw_distance_early_abandon(
            &[1.0, 2.0, 3.0],
            &[1.0; 7],
            2.0,
            DtwWindow::Itakura { max_slope: 1.0 },
            1.0
        ),
        Err(FdarError::InvalidParameter { .. })
    ));
}

// --- 1-NN --------------------------------------------------------------------------

fn to_matrix(rows: &[Vec<f64>]) -> FdMatrix {
    let (n, m) = (rows.len(), rows[0].len());
    let mut data = vec![0.0; n * m];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            data[i + j * n] = v;
        }
    }
    FdMatrix::from_column_major(data, n, m).unwrap()
}

fn brute_force(queries: &[Vec<f64>], refs: &[Vec<f64>], p: f64, w: DtwWindow) -> Vec<(usize, f64)> {
    queries
        .iter()
        .map(|q| {
            let mut best = (usize::MAX, f64::INFINITY);
            for (k, r) in refs.iter().enumerate() {
                let d = dtw_distance_windowed(q, r, p, w).unwrap();
                if d < best.1 {
                    best = (k, d);
                }
            }
            best
        })
        .collect()
}

fn check_nn(queries: &[Vec<f64>], refs: &[Vec<f64>], p: f64, w: DtwWindow) -> DtwSearchStats {
    let nn = dtw_nearest_neighbor(&to_matrix(queries), &to_matrix(refs), p, w).unwrap();
    let bf = brute_force(queries, refs, p, w);
    for (qi, &(k, d)) in bf.iter().enumerate() {
        assert_eq!(nn.indices[qi], k, "query {qi} {w:?} p={p}");
        assert_eq!(
            nn.distances[qi].to_bits(),
            d.to_bits(),
            "query {qi} {w:?} p={p}"
        );
    }
    let s = nn.stats;
    let total = s.pruned_lb_keogh
        + s.pruned_lb_kim
        + s.pruned_lb_webb
        + s.early_abandoned
        + s.dtw_completed;
    assert_eq!(total, queries.len() * refs.len());
    s
}

#[test]
fn nearest_neighbor_equals_brute_force() {
    let mut rng = StdRng::seed_from_u64(1234);
    for &(m_q, m_r) in &[(32usize, 32usize), (24, 30), (9, 9)] {
        let refs: Vec<Vec<f64>> = (0..60).map(|_| random_walk(&mut rng, m_r)).collect();
        let queries: Vec<Vec<f64>> = (0..15).map(|_| random_walk(&mut rng, m_q)).collect();
        for w in windows() {
            if w.row_bounds(m_q, m_r).is_err() {
                continue;
            }
            for &p in &[2.0, 1.0, 1.5, 0.5] {
                check_nn(&queries, &refs, p, w);
            }
        }
    }
}

#[test]
fn nearest_neighbor_prunes_on_clustered_data() {
    // Noisy copies of a few prototypes: most candidates must be pruned.
    let mut rng = StdRng::seed_from_u64(5);
    let m = 64;
    let protos: Vec<Vec<f64>> = (0..4).map(|_| random_walk(&mut rng, m)).collect();
    let noisy = |rng: &mut StdRng, k: usize| -> Vec<f64> {
        protos[k % 4]
            .iter()
            .map(|v| v + rng.gen_range(-0.2..0.2))
            .collect()
    };
    let refs: Vec<Vec<f64>> = (0..200).map(|k| noisy(&mut rng, k)).collect();
    let queries: Vec<Vec<f64>> = (0..20).map(|k| noisy(&mut rng, k)).collect();
    let s = check_nn(&queries, &refs, 2.0, DtwWindow::SakoeChiba { radius: 6 });
    let pruned = s.pruned_lb_keogh + s.pruned_lb_kim + s.pruned_lb_webb + s.early_abandoned;
    assert!(pruned * 10 > 8 * queries.len() * refs.len(), "{s:?}");
}

#[test]
fn nearest_neighbor_ties_and_exact_matches() {
    let mut rng = StdRng::seed_from_u64(11);
    let base: Vec<Vec<f64>> = (0..6).map(|_| random_walk(&mut rng, 20)).collect();
    // refs: duplicates of every base series -> exact ties resolved to the lowest index.
    let mut refs = base.clone();
    refs.extend(base.iter().rev().cloned());
    refs.push(base[3].clone());
    let mut queries = base.clone();
    queries.push(random_walk(&mut rng, 20));
    let w = DtwWindow::SakoeChiba { radius: 3 };
    let nn = dtw_nearest_neighbor(&to_matrix(&queries), &to_matrix(&refs), 2.0, w).unwrap();
    for k in 0..6 {
        assert_eq!(nn.indices[k], k);
        assert_eq!(nn.distances[k], 0.0);
    }
    check_nn(&queries, &refs, 2.0, w);
    check_nn(&queries, &refs, 1.0, DtwWindow::None);
}

#[test]
fn nearest_neighbor_errors() {
    let ok = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let empty = FdMatrix::zeros(0, 3);
    let w = DtwWindow::None;
    assert!(matches!(
        dtw_nearest_neighbor(&empty, &ok, 2.0, w),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_nearest_neighbor(&ok, &empty, 2.0, w),
        Err(FdarError::InvalidDimension { .. })
    ));
    let nan = FdMatrix::from_column_major(vec![1.0, f64::NAN, 3.0], 1, 3).unwrap();
    assert!(matches!(
        dtw_nearest_neighbor(&nan, &ok, 2.0, w),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        dtw_nearest_neighbor(&ok, &ok, 0.0, w),
        Err(FdarError::InvalidParameter { .. })
    ));
    let long = FdMatrix::from_column_major(vec![1.0; 7], 1, 7).unwrap();
    assert!(matches!(
        dtw_nearest_neighbor(&ok, &long, 2.0, DtwWindow::Itakura { max_slope: 1.0 }),
        Err(FdarError::InvalidParameter { .. })
    ));
}

#[test]
fn lb_input_errors() {
    assert!(matches!(
        lb_envelope(&[], 1),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        lb_envelope_windowed(&[1.0], 0, DtwWindow::None),
        Err(FdarError::InvalidDimension { .. })
    ));
    let env = lb_envelope(&[1.0, 2.0], 1).unwrap();
    assert!(matches!(
        lb_keogh(&[1.0], &env, 2.0),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        lb_keogh(&[1.0, 2.0], &env, f64::NAN),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        lb_kim(&[], &[1.0], 2.0),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        lb_webb(&[1.0], &[], 2.0, DtwWindow::None),
        Err(FdarError::InvalidDimension { .. })
    ));
}

// --- 1-NN stats pinned on the pre-k-NN implementation (commit 31900859) --------

#[test]
fn nearest_neighbor_output_and_stats_unchanged_by_knn_refactor() {
    // Indices, distance bits and stats produced by `dtw_nearest_neighbor` before
    // it became the k = 1 case of the top-k search (commit 31900859).
    #[allow(clippy::type_complexity)]
    let pinned: [([usize; 10], [u64; 10], [usize; 5]); 3] = [
        (
            [7, 8, 8, 17, 13, 37, 13, 36, 16, 38],
            [
                4617304198679872651,
                4611418004540579176,
                4620742900586740100,
                4624672678886321647,
                4626093857559264570,
                4616520089029740828,
                4617678148044995588,
                4623582023152587964,
                4620918529528532590,
                4620704312033080351,
            ],
            [376, 0, 12, 1, 11],
        ),
        (
            [7, 8, 0, 17, 1, 38, 13, 6, 16, 26],
            [
                4620957434298031193,
                4618096687274148782,
                4620735548348747665,
                4621707342479366156,
                4621373988948816129,
                4620179051893055024,
                4620059460847711520,
                4621714160428844811,
                4621627653638243834,
                4621961426120890318,
            ],
            [206, 3, 112, 47, 32],
        ),
        (
            [7, 8, 8, 17, 26, 38, 13, 0, 16, 38],
            [
                4623945771487284497,
                4623166829955067378,
                4625357653707336576,
                4625945941344419334,
                4625441326516924243,
                4624536851411270244,
                4624872432120479992,
                4625957832840429245,
                4624387881609656005,
                4625328297402792012,
            ],
            [274, 0, 0, 113, 13],
        ),
    ];
    let mut rng = StdRng::seed_from_u64(7373);
    let refs: Vec<Vec<f64>> = (0..40).map(|_| random_walk(&mut rng, 24)).collect();
    let queries: Vec<Vec<f64>> = (0..10).map(|_| random_walk(&mut rng, 24)).collect();
    let cases = [
        (DtwWindow::SakoeChiba { radius: 2 }, 2.0),
        (DtwWindow::None, 1.0),
        (DtwWindow::Itakura { max_slope: 2.0 }, 0.5),
    ];
    for ((w, p), (idx, bits, st)) in cases.into_iter().zip(pinned) {
        let (q, r) = (to_matrix(&queries), to_matrix(&refs));
        let nn = dtw_nearest_neighbor(&q, &r, p, w).unwrap();
        assert_eq!(nn.indices, idx);
        let got: Vec<u64> = nn.distances.iter().map(|d| d.to_bits()).collect();
        assert_eq!(got, bits);
        let s = nn.stats;
        assert_eq!(
            [
                s.pruned_lb_keogh,
                s.pruned_lb_kim,
                s.pruned_lb_webb,
                s.early_abandoned,
                s.dtw_completed
            ],
            st
        );
        let knn = dtw_k_nearest_neighbors(&q, &r, 1, p, w).unwrap();
        assert_eq!(knn.stats, nn.stats);
        for qi in 0..10 {
            assert_eq!(knn.indices[qi], vec![nn.indices[qi]]);
            assert_eq!(knn.distances[qi][0].to_bits(), nn.distances[qi].to_bits());
        }
    }
}

// --- k-NN ----------------------------------------------------------------------

/// Brute-force k-NN: stable sort of all distances by (distance, index).
fn brute_force_k(
    q: &[f64],
    refs: &[Vec<f64>],
    k: usize,
    p: f64,
    w: DtwWindow,
) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = refs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, dtw_distance_windowed(q, r, p, w).unwrap()))
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn check_knn(
    queries: &[Vec<f64>],
    refs: &[Vec<f64>],
    k: usize,
    p: f64,
    w: DtwWindow,
) -> DtwSearchStats {
    let knn = dtw_k_nearest_neighbors(&to_matrix(queries), &to_matrix(refs), k, p, w).unwrap();
    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_k(q, refs, k, p, w);
        let idx: Vec<usize> = bf.iter().map(|e| e.0).collect();
        let bits: Vec<u64> = bf.iter().map(|e| e.1.to_bits()).collect();
        let got: Vec<u64> = knn.distances[qi].iter().map(|d| d.to_bits()).collect();
        assert_eq!(knn.indices[qi], idx, "query {qi} k={k} {w:?} p={p}");
        assert_eq!(got, bits, "query {qi} k={k} {w:?} p={p}");
    }
    let s = knn.stats;
    let total = s.pruned_lb_keogh
        + s.pruned_lb_kim
        + s.pruned_lb_webb
        + s.early_abandoned
        + s.dtw_completed;
    assert_eq!(total, queries.len() * refs.len());
    s
}

#[test]
fn k_nearest_neighbors_equal_brute_force() {
    let mut rng = StdRng::seed_from_u64(4242);
    let mut pruned = 0usize;
    for &(m_q, m_r) in &[(28usize, 28usize), (20, 26), (8, 8)] {
        let refs: Vec<Vec<f64>> = (0..45).map(|_| random_walk(&mut rng, m_r)).collect();
        let queries: Vec<Vec<f64>> = (0..8).map(|_| random_walk(&mut rng, m_q)).collect();
        for w in windows() {
            if w.row_bounds(m_q, m_r).is_err() {
                continue;
            }
            for &p in &[2.0, 1.0, 0.5] {
                for k in [1usize, 2, 3, 7, 45] {
                    let s = check_knn(&queries, &refs, k, p, w);
                    pruned +=
                        s.pruned_lb_keogh + s.pruned_lb_kim + s.pruned_lb_webb + s.early_abandoned;
                }
            }
        }
    }
    assert!(pruned > 0);
}

#[test]
fn k_nearest_neighbors_ties_and_clusters() {
    let mut rng = StdRng::seed_from_u64(99);
    let base: Vec<Vec<f64>> = (0..5).map(|_| random_walk(&mut rng, 20)).collect();
    // Exact duplicates (ties broken by index) plus noisy copies of each base.
    let mut refs = Vec::new();
    for b in &base {
        refs.push(b.clone());
        refs.push(b.clone());
        for _ in 0..4 {
            refs.push(b.iter().map(|v| v + rng.gen_range(-0.05..0.05)).collect());
        }
    }
    let queries: Vec<Vec<f64>> = base
        .iter()
        .map(|b| b.iter().map(|v| v + 0.01).collect())
        .collect();
    for k in [2usize, 3, 6, 10] {
        let s = check_knn(&queries, &refs, k, 2.0, DtwWindow::SakoeChiba { radius: 2 });
        // Clustered data: most far candidates are discarded by the bounds.
        assert!(
            s.dtw_completed + s.early_abandoned < queries.len() * refs.len() / 2,
            "{s:?}"
        );
    }
    let knn = dtw_k_nearest_neighbors(
        &to_matrix(&base),
        &to_matrix(&refs),
        2,
        2.0,
        DtwWindow::None,
    )
    .unwrap();
    for (qi, idx) in knn.indices.iter().enumerate() {
        assert_eq!(idx, &vec![6 * qi, 6 * qi + 1]);
        assert_eq!(knn.distances[qi], vec![0.0, 0.0]);
    }
}

#[test]
fn k_nearest_neighbors_errors() {
    let r = to_matrix(&[X10.to_vec(), Y10.to_vec()]);
    let q = to_matrix(&[X10.to_vec()]);
    for k in [0usize, 3] {
        assert!(matches!(
            dtw_k_nearest_neighbors(&q, &r, k, 2.0, DtwWindow::None),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
    assert!(matches!(
        dtw_k_nearest_neighbors(&FdMatrix::zeros(0, 10), &r, 1, 2.0, DtwWindow::None),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_k_nearest_neighbors(&q, &r, 1, f64::NAN, DtwWindow::None),
        Err(FdarError::InvalidParameter { .. })
    ));
}

// --- LB_Improved / LB_Enhanced ---------------------------------------------------

struct LbRef {
    radius: usize,
    v: usize,
    x: &'static [f64],
    y: &'static [f64],
    improved_p1: f64,
    improved_p2: f64,
    enhanced_p2: f64,
}

// Reference values on equal-length pairs with a Sakoe-Chiba radius:
// - `improved_p1`: Lemire's lbimproved (github.com/lemire/lbimproved, include/dtw.h,
//   `computeEnvelope` + the two passes of `LB_Improved::test`), L1 cost;
// - `improved_p2` / `enhanced_p2`: Tan, Petitjean & Webb's LbEnhanced Java code
//   (github.com/ChangWeiTan/LbEnhanced, `LbImproved.distance(a, b, Ub, Lb, r)` and
//   `LbEnhanced.distance(a, b, U, L, w, v, +inf)` with (U, L) = `fillEnvelope(b, w)`),
//   squared cost. a = x (query), b = y (candidate).
const LB_REFS: &[LbRef] = &[
    LbRef {
        radius: 1,
        v: 1,
        x: &[
            1.1253424870850857,
            -0.6698305075519568,
            0.5068733433830535,
            -0.12061668975708784,
            -1.3944112911909121,
            -0.9200644412350818,
            -1.292664334695793,
            -0.9272123881401284,
        ],
        y: &[
            -2.0745499247520063,
            -2.6042712906080876,
            -1.9613334040620272,
            -1.4569288916479985,
            -2.186930307158304,
            -1.0532841034491058,
            -1.3890415116613837,
            0.017606803697514994,
        ],
        improved_p1: 10.22564855765277,
        improved_p2: 19.282766834258485,
        enhanced_p2: 18.45997148192715,
    },
    LbRef {
        radius: 2,
        v: 3,
        x: &[
            0.11965743063739503,
            0.7254500168347464,
            1.3239025430086127,
            -0.4608761740423226,
            -0.576317260083363,
            0.1038912157164118,
            0.47091500017356336,
            -0.12640569249829559,
        ],
        y: &[
            -0.6450436110093112,
            -0.17390418613190778,
            -0.8939011891475355,
            -3.19079244769951,
            -5.603650898219337,
            -5.742542134340256,
            -5.973773968923021,
            -4.415708162908089,
        ],
        improved_p1: 20.02289771248377,
        improved_p2: 62.19268076137391,
        enhanced_p2: 64.5030868975686,
    },
    LbRef {
        radius: 2,
        v: 2,
        x: &[
            -0.012374507191291031,
            -0.46692628185380824,
            1.124425238344242,
            -0.6640973288311935,
            -0.8327810836536443,
            1.3355738067080059,
            2.2377662466046497,
            1.6858996077144481,
            1.6617741013463438,
            1.6852207219851822,
            3.474887187173815,
            3.551309493273871,
            4.604297304487091,
        ],
        y: &[
            -1.0153344789672285,
            -1.7172851225461003,
            -2.8219583428284807,
            -4.319528869688545,
            -3.873437939544151,
            -3.9322887908205257,
            -3.7286812316333338,
            -3.524697521991458,
            -3.9457043255210555,
            -0.819966540465817,
            -1.2334766273732272,
            -1.5359550205153167,
            -2.4669531114099796,
        ],
        improved_p1: 43.06503420649836,
        improved_p2: 160.87862173387325,
        enhanced_p2: 180.82248487885218,
    },
    LbRef {
        radius: 3,
        v: 4,
        x: &[
            -1.0205039767985389,
            -1.5216847739766695,
            -3.457541504637277,
            -5.580037812564177,
            -5.611383765867907,
            -5.645049965314537,
            -8.017671576271505,
            -6.419445964692013,
            -5.15233103792994,
            -5.428146105666341,
            -4.820439451063823,
            -5.858915649587289,
            -5.620008109167962,
        ],
        y: &[
            0.1621209279157876,
            -0.854101212984751,
            0.4720959116184238,
            2.0971315842671494,
            3.434966714239396,
            3.429331705594156,
            5.150645521912541,
            5.042167511575768,
            5.156470912545427,
            5.008544251781758,
            5.112851932629157,
            6.265286763479315,
            5.4676682153095335,
        ],
        improved_p1: 92.51040403912211,
        improved_p2: 802.0678044493206,
        enhanced_p2: 792.8848751190561,
    },
    LbRef {
        radius: 1,
        v: 3,
        x: &[
            -1.0423060244464155,
            -1.900314311572759,
            -2.9125783124594773,
            -3.3082302021201864,
            -3.6103183814591584,
            -4.726380029403082,
            -4.602663297512027,
            -4.9411458990612855,
            -5.048503399077961,
            -4.451304772314838,
            -5.50121681548479,
            -6.938552345907153,
            -7.149189372138838,
            -6.300713743021689,
            -7.6034157724206475,
            -8.279115363706506,
            -8.096712984487432,
            -6.397142031776184,
            -4.089511488988457,
            -5.083614484722398,
        ],
        y: &[
            2.5302410554042467,
            1.4501891368818718,
            2.220881119095787,
            1.0205672175144447,
            1.6392063386858053,
            2.577077726152161,
            3.5092011680887216,
            3.627553064260471,
            3.579262975108085,
            1.184477646647213,
            1.9354394929321146,
            1.318186227289921,
            0.8840103295957132,
            0.996952142818163,
            1.4971080869389284,
            1.037925674472116,
            2.3173884204516977,
            1.6607568321851125,
            2.2579278162161804,
            2.362347072451106,
        ],
        improved_p1: 134.67226438021598,
        improved_p2: 928.347916536319,
        enhanced_p2: 905.2899729225376,
    },
    LbRef {
        radius: 4,
        v: 2,
        x: &[
            -2.183998386687357,
            -3.1545753901622615,
            -4.232816155598176,
            -4.644607308069759,
            -5.149902948978102,
            -5.289384035441355,
            -6.966418310282288,
            -8.686997162102518,
            -9.554053063853678,
            -7.236882949858691,
            -6.998499172010879,
            -5.409682254095933,
            -4.6914845342308835,
            -4.053835418486531,
            -3.0837385809277134,
            -2.150894330894055,
            -3.3461730159987466,
            -1.6049868109841274,
            -1.4426038071153708,
            -1.0819032725122861,
        ],
        y: &[
            -0.31558321005188167,
            0.2893151394629005,
            0.9033112763968714,
            1.9059025782585781,
            0.6567688614478695,
            -1.631877553284264,
            -1.148555821071132,
            -2.0793464955911394,
            -1.9755931981578112,
            -1.0238752448949529,
            -0.5609488911247988,
            -1.9962991472517801,
            -1.293135347583756,
            0.24072584610676717,
            1.1470300567056864,
            1.0182344023861185,
            1.2425400195518512,
            0.9252740288770107,
            0.21629566116070065,
            0.18642801708376852,
        ],
        improved_p1: 71.55189669386064,
        improved_p2: 262.83497350320215,
        enhanced_p2: 251.41351524633615,
    },
    LbRef {
        radius: 3,
        v: 5,
        x: &[
            -0.6258957286830534,
            1.3955296798531043,
            2.0219582891670793,
            3.555473497922076,
            3.0298805906557553,
            4.994334431235048,
            6.190927847499971,
            5.370548156815743,
            5.19659795819076,
            4.918497290008828,
            6.928081657855297,
            7.441536662194828,
            9.642028504596944,
            7.823335740356178,
            7.70101824177015,
            7.591053047649344,
            6.581081539279002,
            6.9818647642039755,
            6.062652874121016,
            7.617984432373813,
            7.425414854524288,
            7.854619621629869,
            7.623445595192814,
            6.600455191612628,
            6.943447527336049,
            6.147348586739071,
            6.079023313944773,
            5.508069535003566,
            6.008562124258678,
            5.18056452000442,
            4.006994259336265,
        ],
        y: &[
            0.5352900506163266,
            1.2333241961890877,
            1.1223279638833557,
            0.7190878572830135,
            0.8496249126076167,
            0.9384786943339628,
            2.528607828441849,
            -0.48543618542086175,
            -1.0261505074288537,
            -0.08071664824558555,
            -0.3001011934006783,
            -0.7229454002505891,
            0.14992676422633477,
            2.2167188904575497,
            1.3214601634357277,
            1.1770745561003353,
            0.8968671565897699,
            1.0538618731635134,
            0.7532418608211489,
            0.1998915797012406,
            -0.6617157183601631,
            -0.9610347262906964,
            -0.9417002117226038,
            -1.9526538562699054,
            -0.802647990690637,
            0.667268480615939,
            0.6475261715949655,
            -1.132514346112726,
            -0.3054985472027202,
            0.31914518376606005,
            -1.9490732604961916,
        ],
        improved_p1: 164.97179917131402,
        improved_p2: 782.7034008171142,
        enhanced_p2: 714.3487256394261,
    },
    LbRef {
        radius: 5,
        v: 3,
        x: &[
            0.09924694241336245,
            0.8703660516075015,
            0.9244606636286692,
            0.020697859411382025,
            -0.5297143756499397,
            -1.6305406790959172,
            -1.4991034460452153,
            -2.252029728976809,
            -2.2000674108432388,
            -2.1755137652193053,
            -2.8755554173182425,
            -2.4103400171476714,
            -2.5622488030729142,
            -3.6546151107711786,
            -3.33047528633025,
            -3.331788740310791,
            -2.512968630442066,
            -2.0235144662540727,
            -1.8252824766074423,
            -2.6380865282580417,
            -2.6475445512684788,
            -1.8683127852116803,
            -4.2440676861661135,
            -3.8156290120328853,
            -3.333756543586958,
            -5.063993510605501,
            -6.140115801741877,
            -5.016505883134033,
            -3.5914727878457455,
            -4.204331963471685,
            -5.625623033539575,
        ],
        y: &[
            -0.1832582510206248,
            0.35716679548933605,
            -1.0380393209606267,
            -0.5738218074141892,
            -0.0019788410895175135,
            -1.4479497044393286,
            -1.2845976613318486,
            0.3450342737485914,
            2.1671178486503466,
            2.3932142160462293,
            4.0474385617804485,
            3.8867982554515734,
            2.310801446673729,
            2.829911811693619,
            2.9531356217072555,
            2.7869493114588484,
            2.335549154397844,
            5.111337174479971,
            3.88097350480625,
            4.315177095013504,
            4.270431941275137,
            4.286622952106292,
            2.928789870683783,
            2.129746042008758,
            2.086299749832123,
            3.7755137589837084,
            3.165510374422898,
            3.397874259590302,
            3.9370065324760013,
            2.974380497301434,
            3.7321447803278245,
        ],
        improved_p1: 132.5125208815806,
        improved_p2: 680.2704435082669,
        enhanced_p2: 685.370046856675,
    },
    LbRef {
        radius: 0,
        v: 2,
        x: &[
            1.3908099901065685,
            0.7835528599355098,
            1.3247786126640175,
            0.9733806540289426,
            0.16965486874379598,
            2.932452481247257,
            2.688470674895624,
            3.3090964366445537,
            5.219455990035508,
            7.021326449507977,
            7.86807767535226,
            9.409199043848222,
            10.164156208949537,
            8.85774660458478,
            7.8499737290744624,
            6.6101540020825045,
            6.028176318557023,
        ],
        y: &[
            -0.6346522767242877,
            -0.274389206709958,
            -0.6502313561445343,
            0.22792352601370536,
            0.21539967522030018,
            1.2877056277200731,
            -0.0238975120495748,
            -0.035090910085979345,
            1.070729488022918,
            2.681203625512351,
            2.0886127977339912,
            2.3080575766225704,
            2.293796663887651,
            0.9057225219852216,
            1.5225221553538764,
            1.1873125770921296,
            1.392424274687973,
        ],
        improved_p1: 67.12880296507312,
        improved_p2: 366.91424256286876,
        enhanced_p2: 366.9142425628688,
    },
    LbRef {
        radius: 2,
        v: 1,
        x: &[
            0.2296432022566566,
            -0.6220957555017489,
            -0.692939139870242,
            -2.439805174028879,
            -2.2590728281953307,
            -1.671162708892699,
            -1.8812204187579873,
            1.107833873546043,
            1.0668469608982898,
            1.455768181637889,
            0.19808396284781593,
            1.5194840682121216,
            2.2710785454377214,
            2.7442548266130355,
            2.911005304845753,
            2.9311658454145366,
            2.9497392630623405,
            2.314603423260621,
            1.9829121549478752,
            0.49029062355479835,
            0.5597680351450963,
            0.8671436281276804,
            0.8812149833332189,
            0.2262284433775742,
            0.391567078574088,
        ],
        y: &[
            -1.1709520944198604,
            -0.3624250770222144,
            -1.2545786558093028,
            -0.38442265070271364,
            -0.00951699679999557,
            -1.0866608742784276,
            1.032479718312353,
            1.7627057865467846,
            2.393674983981466,
            2.679610578407471,
            0.7239768165866323,
            1.4592982727259631,
            1.378506033316036,
            0.8346938158645404,
            1.1452694591725432,
            1.910301431933237,
            2.591042541885444,
            2.7422607159506365,
            4.694665250391166,
            3.778486665203177,
            4.382685429328946,
            4.616122381806463,
            6.623617719740192,
            5.264673384567399,
            6.130236486775769,
        ],
        improved_p1: 36.47937601226852,
        improved_p2: 92.0581428595644,
        enhanced_p2: 95.8585453932454,
    },
    LbRef {
        radius: 11,
        v: 4,
        x: &[
            -1.4055994660577564,
            -1.0074033465206602,
            -0.4034606758803133,
            -0.06945273462111418,
            -1.1294622866145663,
            -1.2760665529791189,
            -1.739909794194992,
            -1.1371650834250238,
            -1.2336322329524845,
            -0.4807824728640927,
            0.2454410489485057,
            1.0929606333943305,
        ],
        y: &[
            -1.460215581951882,
            -2.010278777848932,
            -3.098120760042338,
            -2.6904333419150745,
            -3.5577195960877237,
            -2.1101771695911653,
            -2.542537999528469,
            -2.9852350143579134,
            -2.436114068480393,
            -2.6463202692743537,
            -2.189195128222968,
            -1.0370873574496589,
        ],
        improved_p1: 14.466860780554022,
        improved_p2: 17.796185783719082,
        enhanced_p2: 9.505639429945361,
    },
    LbRef {
        radius: 6,
        v: 6,
        x: &[
            0.24105572138539047,
            0.47100056785463684,
            1.673829121211353,
            1.3076105544952248,
            -0.3745938158809672,
            -0.7756384280770097,
            -0.41425481143245463,
            -1.18602135460774,
            -2.8436445181557315,
            -3.3679534742214616,
            -3.461842313117507,
            -4.130946840367497,
            -5.006226386085695,
            -3.878923170602711,
            -4.286474403646903,
            -3.93224680335079,
            -3.555459458506763,
            -3.5672950305444218,
            -4.10394144446343,
            -3.7105425805097507,
            -3.5417088041444855,
            -4.443500493103972,
            -3.6560073859698123,
            -5.4988034972099245,
            -3.629806381432393,
            -4.109838686080473,
            -4.645008787001624,
            -4.543675746915519,
            -5.508499690240239,
            -5.2492774528027235,
            -6.188775689371682,
            -6.682079487073305,
            -6.361642671979665,
            -6.610491478045206,
            -7.034317765119745,
            -5.366296776927272,
            -5.919032945243294,
            -4.393398723035423,
            -5.006657810594821,
            -4.323700005432333,
        ],
        y: &[
            2.0186717784526276,
            0.608331367505835,
            -1.0332131182684232,
            1.1213127640629004,
            1.6183946020427964,
            1.2641640013365716,
            1.5154358314063192,
            2.128672109974687,
            2.2111805229427497,
            2.053550447644947,
            2.7383559520419474,
            3.80018303800261,
            3.2218935164442115,
            1.54648150597648,
            3.0789591267231393,
            2.4983726382608236,
            2.412950960124438,
            4.242368550960442,
            4.351139876291499,
            5.822672099947637,
            7.7722545014400435,
            6.861182132499816,
            7.970567082688703,
            8.03970679940171,
            9.024966500437147,
            9.34734585269483,
            10.187540073711988,
            11.383533807368384,
            12.217876673801326,
            11.0826601384639,
            10.941193203699441,
            12.056723505454524,
            11.321883268965093,
            11.683958223454209,
            12.402827869537726,
            10.962641138711307,
            11.120111916512172,
            12.403665029662104,
            12.374373268905357,
            12.31641269441154,
        ],
        improved_p1: 338.73519303917044,
        improved_p2: 4011.2444410516773,
        enhanced_p2: 3980.5072859688516,
    },
];

#[test]
fn lb_improved_and_enhanced_match_reference_implementations() {
    for (c, r) in LB_REFS.iter().enumerate() {
        let w = DtwWindow::SakoeChiba { radius: r.radius };
        let i1 = lb_improved(r.x, r.y, 1.0, w).unwrap();
        let i2 = lb_improved(r.x, r.y, 2.0, w).unwrap();
        let e2 = lb_enhanced(r.x, r.y, 2.0, w, r.v).unwrap();
        assert!(
            close(i1, r.improved_p1, 1e-13),
            "case {c}: {i1} vs {}",
            r.improved_p1
        );
        assert!(
            close(i2, r.improved_p2, 1e-13),
            "case {c}: {i2} vs {}",
            r.improved_p2
        );
        assert!(
            close(e2, r.enhanced_p2, 1e-13),
            "case {c}: {e2} vs {}",
            r.enhanced_p2
        );
        for p in [1.0, 2.0] {
            let d = dtw_distance_windowed(r.x, r.y, p, w).unwrap();
            assert!(lb_improved(r.x, r.y, p, w).unwrap() <= d * (1.0 + 1e-12));
        }
    }
}

#[test]
fn lb_improved_and_enhanced_bound_dtw_on_random_pairs() {
    let mut rng = StdRng::seed_from_u64(2019);
    let sizes = [
        (16usize, 16usize),
        (30, 30),
        (7, 7),
        (20, 13),
        (11, 25),
        (3, 3),
        (6, 9),
        (2, 5),
        (1, 4),
    ];
    let (mut checked, mut imp_tighter, mut enh_tighter) = (0usize, 0usize, 0usize);
    for _rep in 0..10 {
        for &(n, m) in &sizes {
            let x = random_walk(&mut rng, n);
            let y = random_walk(&mut rng, m);
            for w in windows() {
                if w.row_bounds(n, m).is_err() {
                    continue;
                }
                let env = lb_envelope_windowed(&y, n, w).unwrap();
                for &p in &[1.0, 2.0, 1.5, 3.0, 0.5] {
                    let d = dtw_distance_windowed(&x, &y, p, w).unwrap();
                    let tol = 1e-12 * d.max(1.0);
                    let keogh = lb_keogh(&x, &env, p).unwrap();
                    if p >= 1.0 {
                        let imp = lb_improved(&x, &y, p, w).unwrap();
                        assert!(
                            keogh <= imp,
                            "keogh {keogh} > improved {imp} ({w:?}, p={p}, {n}x{m})"
                        );
                        assert!(
                            imp <= d + tol,
                            "improved {imp} > dtw {d} ({w:?}, p={p}, {n}x{m})"
                        );
                        imp_tighter += usize::from(imp > keogh);
                    } else {
                        assert!(matches!(
                            lb_improved(&x, &y, p, w),
                            Err(FdarError::InvalidParameter { .. })
                        ));
                    }
                    for v in [1usize, 2, 3, 5, 100] {
                        let enh = lb_enhanced(&x, &y, p, w, v).unwrap();
                        assert!(
                            enh <= d + tol,
                            "enhanced(v={v}) {enh} > dtw {d} ({w:?}, p={p}, {n}x{m})"
                        );
                        enh_tighter += usize::from(enh > keogh);
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 2000, "checked {checked}");
    assert!(
        imp_tighter > checked / 4,
        "LB_Improved tighter only {imp_tighter}"
    );
    assert!(
        enh_tighter > checked,
        "LB_Enhanced tighter only {enh_tighter}"
    );
}

#[test]
fn lb_enhanced_and_improved_exact_and_degenerate_cases() {
    let w = DtwWindow::SakoeChiba { radius: 2 };
    // Identical series: every bound is zero.
    assert_eq!(lb_improved(&X10, &X10, 2.0, w).unwrap(), 0.0);
    assert_eq!(lb_enhanced(&X10, &X10, 2.0, w, 3).unwrap(), 0.0);
    // Length 2: no room for bands, LB_Enhanced is LB_Keogh.
    let (a, b) = ([0.0, 3.0], [1.0, -1.0]);
    let keogh = lb_keogh(&a, &lb_envelope(&b, 2).unwrap(), 2.0).unwrap();
    assert_eq!(lb_enhanced(&a, &b, 2.0, w, 4).unwrap(), keogh);
    // Radius 0 (Euclidean band): LB_Improved and LB_Enhanced with all bands are
    // the squared Euclidean distance, i.e. the DTW itself.
    let w0 = DtwWindow::SakoeChiba { radius: 0 };
    let d = dtw_distance_windowed(&X10, &Y10, 2.0, w0).unwrap();
    assert!(close(
        lb_enhanced(&X10, &Y10, 2.0, w0, 100).unwrap(),
        d,
        1e-14
    ));
    assert!(lb_improved(&X10, &Y10, 2.0, w0).unwrap() <= d * (1.0 + 1e-14));
    // Errors.
    assert!(matches!(
        lb_enhanced(&X10, &Y10, 2.0, w, 0),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        lb_enhanced(&[], &Y10, 2.0, w, 1),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        lb_improved(&X10, &[], 2.0, w),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        lb_improved(&X10, &Y10, f64::NAN, w),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        lb_enhanced(&X10, &Y10, 2.0, DtwWindow::Itakura { max_slope: 0.5 }, 2),
        Err(FdarError::InvalidParameter { .. })
    ));
}
