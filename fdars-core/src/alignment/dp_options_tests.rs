//! Tests for the configurable DP engine (ALN-DPACC #49): neighbourhood size,
//! `dp_scl` cost refinement, the piecewise-linear (fdasrsf `DP2`) edge cost
//! and candidate pruning.

// Configs are built the way external callers must (sealed struct: `default()`
// + field assignment), so this lint is silenced for the test module.
#![allow(clippy::field_reassign_with_default)]

use super::dp_golden::*;
use super::*;
use crate::alignment::{
    coprime_nbhd, dp_edge_rslope_span, dp_edge_weight_core, dp_edge_weight_linear_core,
    dp_edge_weight_opts, dp_path_opts, dp_prepare_srsfs, elastic_align_pair,
    elastic_align_pair_with_config, elastic_distance_with_config,
    elastic_self_distance_matrix_with_config, karcher_mean_with_config, COPRIME_NBHD_7,
};
use crate::helpers::linear_interp;
use crate::matrix::FdMatrix;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn bump(t: f64, c: f64, s: f64) -> f64 {
    (-(t - c).powi(2) / s).exp()
}

/// Same analytic SRSFs as `optim_tests.rs` / `golden_dp2.py`.
fn case_a(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = grid(m);
    let q1 = t
        .iter()
        .map(|&s| 3.0 * bump(s, 0.35, 0.01) - 2.0 * bump(s, 0.7, 0.01) + 0.2)
        .collect();
    let q2 = t
        .iter()
        .map(|&s| 3.0 * bump(s, 0.45, 0.01) - 2.0 * bump(s, 0.78, 0.01) + 0.2)
        .collect();
    (t, q1, q2)
}

fn case_b(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = grid(m);
    let pi = std::f64::consts::PI;
    let q1 = t.iter().map(|&s| (4.0 * pi * s).sin() + 0.5).collect();
    let q2 = t
        .iter()
        .map(|&s| (4.0 * pi * s.powf(1.3)).sin() + 0.5)
        .collect();
    (t, q1, q2)
}

fn maxabs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn cfg_with(nbhd_dim: usize, dp_scl: usize, cost: DpEdgeCost, prune: bool) -> WarpOptimConfig {
    let mut c = WarpOptimConfig::default();
    c.nbhd_dim = nbhd_dim;
    c.dp_scl = dp_scl;
    c.dp_cost = cost;
    c.dp_prune = prune;
    c
}

fn engine_opts<'a>(cfg: &WarpOptimConfig, nbhd: &'a [(usize, usize)]) -> DpEngineOptions<'a> {
    DpEngineOptions {
        moves: DpMoves {
            nbhd,
            prune: cfg.dp_prune,
        },
        scl: cfg.dp_scl,
        cost: cfg.dp_cost,
    }
}

/// The DP's own objective (sum of edge weights + `L2Gam` penalty along the
/// optimal path) under `cfg`.
fn dp_objective(q1: &[f64], q2: &[f64], t: &[f64], cfg: &WarpOptimConfig) -> f64 {
    let nbhd = coprime_nbhd(cfg.nbhd_dim);
    let opts = engine_opts(cfg, &nbhd);
    let lam = cfg.lambda;
    let pen = |sc, tc, sr, tr| dp_lambda_penalty(t, sc, tc, sr, tr, lam);
    let path = dp_path_opts(q1, q2, t, None, &opts, pen);
    let (q1n, q2n) = dp_prepare_srsfs(q1, q2, opts.scl);
    path.windows(2)
        .map(|w| {
            let ((sr, sc), (tr, tc)) = (w[0], w[1]);
            let (rs, span) = dp_edge_rslope_span(t, sc, tc, sr, tr);
            dp_edge_weight_opts(&q1n, &q2n, &opts, sc, tc, sr, tr, rs, span) + pen(sc, tc, sr, tr)
        })
        .sum()
}

/// DP-independent reference objective: `∫ (q1 − √γ'·q2∘γ)²` of the linearly
/// interpolated, L²(Euclidean)-normalised SRSFs and the piecewise-linear warp,
/// by the midpoint rule on `fine` sub-intervals per grid interval.
fn reference_cost(q1: &[f64], q2: &[f64], t: &[f64], gamma: &[f64], fine: usize) -> f64 {
    let n1 = q1.iter().map(|v| v * v).sum::<f64>().sqrt();
    let n2 = q2.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut acc = 0.0;
    for k in 0..t.len() - 1 {
        let h = t[k + 1] - t[k];
        let slope = (gamma[k + 1] - gamma[k]) / h;
        for j in 0..fine {
            let s = t[k] + (j as f64 + 0.5) / fine as f64 * h;
            let g = gamma[k] + slope * (s - t[k]);
            let r = linear_interp(t, q1, s) / n1 - slope.sqrt() * linear_interp(t, q2, g) / n2;
            acc += r * r * h / fine as f64;
        }
    }
    acc
}

fn assert_valid_warp(g: &[f64], t: &[f64]) {
    assert_eq!(g.len(), t.len());
    assert_eq!(g[0], t[0]);
    assert_eq!(g[g.len() - 1], t[t.len() - 1]);
    assert!(g.windows(2).all(|w| w[1] >= w[0]), "not monotone");
}

// ─── Neighbourhood generation ───────────────────────────────────────────────

#[test]
fn coprime_nbhd_sizes_and_order_match_fdasrvf() {
    // OEIS A018805: 1, 3, 7, 11, 19, 23, 35, 43, 55, 63.
    let counts: Vec<usize> = (1..=10).map(|d| coprime_nbhd(d).len()).collect();
    assert_eq!(counts, [1, 3, 7, 11, 19, 23, 35, 43, 55, 63]);
    assert_eq!(&*coprime_nbhd(7), &COPRIME_NBHD_7[..]);
    // fdasrsf DP.c `Nbrs[63]` (grid_dim 10), first and last entries in order.
    let n10 = coprime_nbhd(10);
    assert_eq!(
        &n10[..12],
        &[
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (2, 1),
            (2, 3)
        ]
    );
    assert_eq!(&n10[59..], &[(10, 1), (10, 3), (10, 7), (10, 9)]);
}

// ─── Bit-identity ───────────────────────────────────────────────────────────

#[test]
fn explicit_default_dp_options_reproduce_classic() {
    let (t, q1, q2) = case_a(51);
    let f1: Vec<f64> = q1
        .iter()
        .scan(0.0, |a, v| {
            *a += v * v.abs() / 50.0;
            Some(*a)
        })
        .collect();
    let f2: Vec<f64> = q2
        .iter()
        .scan(0.0, |a, v| {
            *a += v * v.abs() / 50.0;
            Some(*a)
        })
        .collect();
    let tn: Vec<f64> = (0..51).map(|i| (i as f64 / 50.0).powf(1.3)).collect();
    for lam in [0.0, 0.3] {
        let mut cfg = cfg_with(7, 1, DpEdgeCost::PiecewiseConstant, false);
        cfg.lambda = lam;
        for tt in [&t, &tn] {
            let classic = elastic_align_pair(&f1, &f2, tt, lam);
            let viacfg = elastic_align_pair_with_config(&f1, &f2, tt, &cfg).unwrap();
            assert_eq!(viacfg, classic, "lam={lam}");
        }
    }
}

/// Pruning only skips edges that cannot win the strict `<` (edge costs are
/// `≥ 0`), so every output is bit-identical to the unpruned engine — for both
/// cost models, refined grids, penalties, bands, non-uniform grids, distance
/// matrices and the Karcher mean.
#[test]
fn pruning_is_bit_identical_to_unpruned() {
    let tn: Vec<f64> = (0..45).map(|i| (i as f64 / 44.0).powf(1.4) * 2.0).collect();
    let qn1: Vec<f64> = tn.iter().map(|&s| (2.0 * s).sin() + 0.2 * s).collect();
    let qn2: Vec<f64> = tn.iter().map(|&s| (2.0 * s + 0.4).cos()).collect();
    let mut cases = vec![case_a(41), case_b(61)];
    cases.push((tn, qn1, qn2));
    for (t, q1, q2) in &cases {
        for (nbhd, scl, cost) in [
            (7, 1, DpEdgeCost::PiecewiseConstant),
            (10, 1, DpEdgeCost::PiecewiseLinear),
            (5, 3, DpEdgeCost::PiecewiseConstant),
        ] {
            for (lam, pen) in [
                (0.0, WarpObjectivePenalty::L2Gam),
                (0.2, WarpObjectivePenalty::L2Gam),
                (0.2, WarpObjectivePenalty::L2Psi),
            ] {
                let mut off = cfg_with(nbhd, scl, cost, false);
                off.lambda = lam;
                off.penalty = pen;
                let mut on = off.clone();
                on.dp_prune = true;
                for band in [None, Some(4)] {
                    let g_off = optimal_warp(q1, q2, t, &off, band);
                    let g_on = optimal_warp(q1, q2, t, &on, band);
                    let bits = |g: &[f64]| g.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&g_on),
                        bits(&g_off),
                        "{nbhd} {scl} {cost:?} {lam} {band:?}"
                    );
                }
            }
        }
    }
    // Group routines.
    let t = grid(40);
    let mut d = FdMatrix::zeros(5, 40);
    for i in 0..5 {
        for j in 0..40 {
            d[(i, j)] =
                bump(t[j], 0.3 + 0.07 * i as f64, 0.01) + 0.1 * (3.0 * t[j] + i as f64).sin();
        }
    }
    let off = cfg_with(10, 1, DpEdgeCost::PiecewiseLinear, false);
    let mut on = off.clone();
    on.dp_prune = true;
    assert_eq!(
        elastic_self_distance_matrix_with_config(&d, &t, &on).unwrap(),
        elastic_self_distance_matrix_with_config(&d, &t, &off).unwrap()
    );
    assert_eq!(
        karcher_mean_with_config(&d, &t, 5, 1e-4, &on).unwrap(),
        karcher_mean_with_config(&d, &t, 5, 1e-4, &off).unwrap()
    );
    // Classic options with pruning equal the classic engine.
    let classic = cfg_with(7, 1, DpEdgeCost::PiecewiseConstant, true);
    let f1 = d.row(0);
    let f2 = d.row(3);
    assert_eq!(
        elastic_align_pair_with_config(&f1, &f2, &t, &classic).unwrap(),
        elastic_align_pair(&f1, &f2, &t, 0.0)
    );
}

// ─── Edge-cost models ───────────────────────────────────────────────────────

/// The linear edge weight is the exact integral of the squared residual of
/// the linearly interpolated SRSFs: compare with a 200 000-point midpoint
/// rule (error O(h²) ≈ 1e-11) on edges of several slopes.
#[test]
fn linear_edge_weight_is_exact_integral() {
    let q1 = [0.3, -1.2, 0.8, 2.0, 0.1, -0.4, 1.5, 0.9];
    let q2 = [1.1, 0.4, -0.7, 0.2, 1.9, 0.6, -1.0, 0.3];
    let h = 0.1;
    let interp = |q: &[f64], x: f64| {
        let k = (x.floor() as usize).min(q.len() - 2);
        q[k] + (x - k as f64) * (q[k + 1] - q[k])
    };
    for (sc, tc, sr, tr) in [
        (0, 1, 0, 1),
        (0, 3, 1, 2),
        (1, 3, 0, 5),
        (2, 7, 1, 4),
        (0, 7, 0, 7),
    ] {
        let slope = (tr - sr) as f64 / (tc - sc) as f64;
        let rs = slope.sqrt();
        let span = (tc - sc) as f64 * h;
        let w = dp_edge_weight_linear_core(&q1, &q2, sc, tc, sr, tr, rs, span);
        let n = 200_000;
        let mut acc = 0.0;
        for k in 0..n {
            let u = sc as f64 + (k as f64 + 0.5) / n as f64 * (tc - sc) as f64; // q1 index coord
            let v = sr as f64 + slope * (u - sc as f64);
            let r = interp(&q1, u) - rs * interp(&q2, v);
            acc += r * r;
        }
        let reference = acc / n as f64 * span;
        assert!(
            (w - reference).abs() < 1e-9 * reference.max(1.0),
            "edge ({sc},{tc},{sr},{tr}): {w} vs {reference}"
        );
    }
}

/// The piecewise-constant weight on a `dp_scl`-refined grid converges to the
/// exact (linear) weight, at first order in `1/dp_scl`.
#[test]
fn refined_constant_cost_converges_to_linear_cost() {
    let (t, q1, q2) = case_a(31);
    let nbhd = coprime_nbhd(7);
    let edges = [(3, 6, 2, 9), (10, 17, 12, 15), (20, 21, 18, 25)];
    for &(sc, tc, sr, tr) in &edges {
        let (rs, span) = dp_edge_rslope_span(&t, sc, tc, sr, tr);
        let lin = {
            let o = engine_opts(&cfg_with(7, 1, DpEdgeCost::PiecewiseLinear, false), &nbhd);
            let (a, b) = dp_prepare_srsfs(&q1, &q2, 1);
            dp_edge_weight_opts(&a, &b, &o, sc, tc, sr, tr, rs, span)
        };
        let mut prev_err = f64::INFINITY;
        for scl in [1, 4, 16, 64] {
            let o = engine_opts(
                &cfg_with(7, scl, DpEdgeCost::PiecewiseConstant, false),
                &nbhd,
            );
            let (a, b) = dp_prepare_srsfs(&q1, &q2, scl);
            let w = dp_edge_weight_opts(&a, &b, &o, sc, tc, sr, tr, rs, span);
            let err = (w - lin).abs();
            assert!(err <= prev_err * 1.0001, "scl={scl}: {err} > {prev_err}");
            prev_err = err;
        }
        assert!(
            prev_err < 0.05 * lin,
            "edge ({sc},{tc},{sr},{tr}): {prev_err} vs {lin}"
        );
        // Refining the linear model changes nothing beyond rounding.
        let o = engine_opts(&cfg_with(7, 5, DpEdgeCost::PiecewiseLinear, false), &nbhd);
        let (a, b) = dp_prepare_srsfs(&q1, &q2, 5);
        let w5 = dp_edge_weight_opts(&a, &b, &o, sc, tc, sr, tr, rs, span);
        assert!((w5 - lin).abs() < 1e-12 * lin.max(1.0));
    }
    // dp_scl = 1 is literally the classic edge weight.
    let (a, b) = dp_prepare_srsfs(&q1, &q2, 1);
    let o = engine_opts(&WarpOptimConfig::default(), &nbhd);
    let (rs, span) = dp_edge_rslope_span(&t, 3, 6, 2, 9);
    assert_eq!(
        dp_edge_weight_opts(&a, &b, &o, 3, 6, 2, 9, rs, span).to_bits(),
        dp_edge_weight_core(&a, &b, 3, 6, 2, 9, rs, span).to_bits()
    );
}

// ─── Optimality ─────────────────────────────────────────────────────────────

/// A larger neighbourhood contains every move of a smaller one, so the DP
/// optimum of its own objective cannot increase.
#[test]
fn larger_neighbourhood_never_increases_dp_objective() {
    for (t, q1, q2) in [case_a(41), case_b(41), case_a(81), case_b(101)] {
        for cost in [DpEdgeCost::PiecewiseConstant, DpEdgeCost::PiecewiseLinear] {
            for lam in [0.0, 0.1] {
                let mut prev = f64::INFINITY;
                for d in [3, 5, 7, 10, 12] {
                    let mut c = cfg_with(d, 1, cost, false);
                    c.lambda = lam;
                    let obj = dp_objective(&q1, &q2, &t, &c);
                    assert!(
                        obj <= prev * (1.0 + 1e-12),
                        "m={} {cost:?} lam={lam} d={d}: {obj} > {prev}",
                        t.len()
                    );
                    prev = obj;
                }
            }
        }
    }
}

/// On a DP-independent objective (exact integral of the linearly interpolated
/// SRSFs along the warp), the more accurate engines align at least as well as
/// the classic default.
#[test]
fn accurate_options_lower_the_reference_cost() {
    for (t, q1, q2) in [case_a(41), case_b(41), case_a(101), case_b(81)] {
        let g0 = optimal_warp(&q1, &q2, &t, &WarpOptimConfig::default(), None);
        let c0 = reference_cost(&q1, &q2, &t, &g0, 64);
        for cfg in [
            cfg_with(10, 1, DpEdgeCost::PiecewiseLinear, true),
            cfg_with(10, 4, DpEdgeCost::PiecewiseConstant, true),
            cfg_with(7, 1, DpEdgeCost::PiecewiseLinear, true),
        ] {
            let g = optimal_warp(&q1, &q2, &t, &cfg, None);
            assert_valid_warp(&g, &t);
            let c = reference_cost(&q1, &q2, &t, &g, 64);
            assert!(
                c <= c0 * (1.0 + 1e-9),
                "m={} {:?}/{}/{:?}: {c} > default {c0}",
                t.len(),
                cfg.nbhd_dim,
                cfg.dp_scl,
                cfg.dp_cost
            );
        }
    }
}

/// `f` vs a warped copy `f∘γ` (γ = t + 0.25·t(1−t), the warp of the curve-shape
/// self-distance test): the elastic distance should be 0; the DP leaves a
/// discretisation floor that the accurate options reduce.
#[test]
fn warped_self_distance_residual_decreases() {
    let m = 80;
    let t = grid(m);
    let f = |s: f64| (2.0 * std::f64::consts::PI * s).sin() + 0.6 * bump(s, 0.7, 0.005) + s;
    let f1: Vec<f64> = t.iter().map(|&s| f(s)).collect();
    let f2: Vec<f64> = t.iter().map(|&s| f(s + 0.25 * s * (1.0 - s))).collect();
    let d0 = elastic_distance_with_config(&f1, &f2, &t, &WarpOptimConfig::default()).unwrap();
    let d_lin10 = elastic_distance_with_config(
        &f1,
        &f2,
        &t,
        &cfg_with(10, 1, DpEdgeCost::PiecewiseLinear, true),
    )
    .unwrap();
    let d_scl = elastic_distance_with_config(
        &f1,
        &f2,
        &t,
        &cfg_with(10, 4, DpEdgeCost::PiecewiseConstant, true),
    )
    .unwrap();
    eprintln!("self-distance: default {d0}  nbhd10+linear {d_lin10}  nbhd10+scl4 {d_scl}");
    assert!(d0 > 0.0);
    assert!(d_lin10 < d0, "{d_lin10} !< {d0}");
    assert!(d_scl < d0, "{d_scl} !< {d0}");
}

#[test]
fn unit_neighbourhood_gives_identity_warp() {
    let (t, q1, q2) = case_b(41);
    let g = optimal_warp(
        &q1,
        &q2,
        &t,
        &cfg_with(1, 1, DpEdgeCost::PiecewiseConstant, false),
        None,
    );
    assert!(maxabs(&g, &t) < 1e-12);
}

#[test]
fn dp_option_validation() {
    for (d, s) in [(0, 1), (65, 1), (7, 0), (7, 65)] {
        let c = cfg_with(d, s, DpEdgeCost::PiecewiseConstant, false);
        assert!(c.validate().is_err(), "nbhd_dim={d} dp_scl={s}");
    }
    assert!(cfg_with(64, 64, DpEdgeCost::PiecewiseLinear, true)
        .validate()
        .is_ok());
    let t = grid(10);
    let f: Vec<f64> = t.iter().map(|&s| s * s).collect();
    let bad = cfg_with(0, 1, DpEdgeCost::PiecewiseConstant, false);
    assert!(elastic_align_pair_with_config(&f, &f, &t, &bad).is_err());
}

// ─── fdasrsf 2.7.5 DP2 golden ───────────────────────────────────────────────

/// fdasrsf 2.7.5 `optimum_reparam(q1, t, q2, method="DP2", grid_dim=G,
/// penalty=P, lam=L)` (embedded in `dp_options_golden.rs`, generated by
/// `.planning/quick/261126-2ez-*/golden_dp2.py`) versus fdars with
/// `nbhd_dim = G` + `DpEdgeCost::PiecewiseLinear`: the same move set, move
/// order, SRSF normalisation, exact piecewise-linear edge integral, `l2gam`
/// penalty and warp interpolation. Documented tolerance: 1e-12 (sup norm; observed ≤ 2.3e-16) —
/// differences are floating-point only (fdasrsf walks the sub-intervals in
/// floating point with a 1e-6 breakpoint merge, fdars in exact integers).
#[test]
fn piecewise_linear_matches_fdasrsf_dp2_golden() {
    let cases: [(&str, (Vec<f64>, Vec<f64>, Vec<f64>), usize, f64, &[f64]); 5] = [
        ("A41_G10", case_a(41), 10, 0.0, &FDASRSF_DP2_A41_G10),
        ("B41_G10", case_b(41), 10, 0.0, &FDASRSF_DP2_B41_G10),
        ("A101_G10", case_a(101), 10, 0.0, &FDASRSF_DP2_A101_G10),
        ("B61_G7", case_b(61), 7, 0.0, &FDASRSF_DP2_B61_G7),
        (
            "A61_G10_L2GAM",
            case_a(61),
            10,
            0.05,
            &FDASRSF_DP2_A61_G10_L2GAM,
        ),
    ];
    for (name, (t, q1, q2), g, lam, golden) in cases {
        for prune in [false, true] {
            let mut c = cfg_with(g, 1, DpEdgeCost::PiecewiseLinear, prune);
            c.lambda = lam;
            let gam = optimal_warp(&q1, &q2, &t, &c, None);
            let err = maxabs(&gam, golden);
            eprintln!("{name} prune={prune}: sup |fdars − fdasrsf DP2| = {err:e}");
            assert!(err < 1e-12, "{name}: sup |fdars − fdasrsf DP2| = {err}");
        }
        // The classic engine is a different (piecewise-constant, 35-move)
        // discretisation; report how far it is from DP2 for context.
        let classic = optimal_warp(&q1, &q2, &t, &WarpOptimConfig::default(), None);
        eprintln!(
            "{name}: classic vs DP2 sup diff {}",
            maxabs(&classic, golden)
        );
    }
}
