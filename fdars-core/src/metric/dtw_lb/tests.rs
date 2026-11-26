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
