//! Warp optimiser options for curve shapes (ALN-CAV #65): bit-identity of the
//! default path, the residual-floor diagnosis and the refined optimisers.
#![allow(clippy::field_reassign_with_default)]

use super::geometry::{self, Grid};
use super::*;
use crate::alignment::{DpEdgeCost, WarpObjectivePenalty, WarpOptimMethod};
use crate::test_helpers::ValuePin;
use std::f64::consts::PI;

fn sample(m: usize, f: impl Fn(f64) -> Vec<f64>) -> FdMatrix {
    let d = f(0.0).len();
    let mut b = FdMatrix::zeros(d, m);
    for j in 0..m {
        let v = f(j as f64 / (m - 1) as f64);
        for k in 0..d {
            b[(k, j)] = v[k];
        }
    }
    b
}

/// Open planar test curve (the `open1` curve of `tests.rs`).
fn c_open(t: f64) -> Vec<f64> {
    vec![t, 0.3 * (2.0 * PI * t).sin() + 0.1 * t * t]
}

fn c_open2(t: f64) -> Vec<f64> {
    let s = t + 0.15 * t * (1.0 - t);
    vec![
        s + 0.05 * (PI * s).sin(),
        0.35 * (2.0 * PI * s).sin() + 0.2 * s.powi(3),
    ]
}

fn c_3d(t: f64) -> Vec<f64> {
    vec![t, 0.3 * (2.0 * PI * t).sin(), 0.2 * (PI * t).cos()]
}

fn c_closed(t: f64) -> Vec<f64> {
    let a = 2.0 * PI * t;
    let r = 1.0 + 0.2 * (3.0 * a).cos() + 0.1 * a.sin();
    vec![r * a.cos(), r * a.sin()]
}

/// The warp of the #48 invariance test, `γ0(t) = t + 0.25 t(1 − t)`.
fn gam0(t: f64) -> f64 {
    t + 0.25 * t * (1.0 - t)
}

/// Exact inverse of [`gam0`].
fn gam0_inv(s: f64) -> f64 {
    // 0.25 t² − 1.25 t + s = 0, root in [0, 1]
    (1.25 - (1.5625 - s).sqrt()) / 0.5
}

fn push_alignment(out: &mut Vec<Vec<f64>>, a: &CurveShapeAlignment) {
    out.push(a.gamma.clone());
    out.push(a.beta2_aligned.as_slice().to_vec());
    out.push(a.q2_aligned.as_slice().to_vec());
    out.push(a.rotation.as_slice().to_vec());
    out.push(vec![a.distance, a.seed as f64]);
}

/// Every output field of the default-path probe (pairs, a closed pair, a Karcher mean).
fn default_path_outputs(cfgs: &[CurveShapeConfig]) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let m = 41;
    let pairs = [
        (sample(m, c_open), sample(m, c_open2)),
        (sample(m, c_3d), sample(m, |t| c_3d(t.powf(1.2)))),
    ];
    for cfg in cfgs {
        for (b1, b2) in &pairs {
            if cfg.metric != CurveShapeMetric::Srvf && b1.nrows() != 2 {
                continue;
            }
            push_alignment(&mut out, &curve_shape_align_pair(b1, b2, cfg).unwrap());
        }
    }
    let mut closed = CurveShapeConfig::default();
    closed.closed = true;
    let b1 = sample(m, c_closed);
    let b2 = sample(m, |t| {
        c_closed((t + 0.1 * (2.0 * PI * t).sin() / (2.0 * PI)) % 1.0)
    });
    push_alignment(
        &mut out,
        &curve_shape_align_pair(&b1, &b2, &closed).unwrap(),
    );
    let set = curves_to_set(&[
        sample(m, c_open),
        sample(m, c_open2),
        sample(m, |t| c_open(gam0(t))),
    ])
    .unwrap();
    let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 5, 1e-4).unwrap();
    out.push(km.mean.as_slice().to_vec());
    out.push(km.distances.clone());
    out
}

/// Exact output bits (same-process comparisons only: libm results differ across CPUs).
fn output_bits(fields: &[Vec<f64>]) -> Vec<Vec<u64>> {
    fields
        .iter()
        .map(|f| f.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn hash_cfgs() -> Vec<CurveShapeConfig> {
    let mut out = vec![CurveShapeConfig::default()];
    let mut c = CurveShapeConfig::default();
    c.lambda = 0.05;
    c.scale = false;
    out.push(c);
    let mut c = CurveShapeConfig::default();
    c.metric = CurveShapeMetric::Elastic { a: 1.0, b: 0.25 };
    out.push(c);
    out
}

/// Default-path curve-shape outputs are unchanged from before the warp optimiser option.
///
/// Originally an FNV-1a hash of the output bits (`0xd3bc_07b4_c482_b258`); that hash
/// depends on the platform libm's last-ulp transcendental results and failed on CI
/// runners. The summaries below were recorded on 925ef514, which reproduced that hash
/// bit-for-bit on the recording machine, and are compared at rtol 1e-10.
#[test]
fn default_warp_optim_bit_identical_to_pre_change() {
    let mut pin = ValuePin::new();
    for f in default_path_outputs(&hash_cfgs()) {
        pin.push(&f);
    }
    pin.assert_close(DEFAULT_PATH_PIN, 1e-10, "curve-shape default path");
}

const DEFAULT_PATH_PIN: &[f64] = &[
    19.137500000000003,
    19.137500000000003,
    12.41486830357143,
    13.245121951219513,
    -0.4137735164022369,
    9.910697368231794,
    1.6240306173209045,
    0.8023132435891951,
    26.8597311185642,
    54.47060977630928,
    41.5161081931503,
    13.517820833508798,
    1.999830786095852,
    2.0258466847588874,
    1.9999999999999993,
    1.253146228642787,
    0.108271497677494,
    0.108271497677494,
    0.011722717209327589,
    0.054135748838747,
    22.150000000000002,
    22.150000000000002,
    15.428918402777779,
    14.779471544715449,
    0.11547758809500756,
    14.257401161263823,
    2.270038683519835,
    -0.03269098775753729,
    16.29611366052412,
    63.09763424232555,
    41.302601588341794,
    7.817188153231527,
    2.9992799779582953,
    3.08868037901372,
    3.0000000000000013,
    1.6770363224078348,
    0.05505235265174344,
    0.05505235265174344,
    0.0030307615324919233,
    0.02752617632587172,
    19.137500000000003,
    19.137500000000003,
    12.41486830357143,
    13.245121951219513,
    -0.7540420614237049,
    18.060804709461596,
    5.3933670347219484,
    1.4620992115779194,
    36.25118415373871,
    73.51615313090983,
    75.6236627975583,
    18.244300742611763,
    1.9998307860958522,
    2.0258466847588914,
    1.9999999999999998,
    1.2531462286427875,
    0.1618402747546557,
    0.1618402747546557,
    0.02619227453266245,
    0.08092013737732785,
    22.112500000000004,
    22.112500000000004,
    15.421574652777778,
    14.776117886178863,
    0.12439153170185194,
    23.90728436309597,
    6.4023507500699335,
    -0.06849168624589452,
    20.915337390170713,
    81.57190585874726,
    68.94051057439505,
    10.113160377633449,
    2.9991988700525702,
    3.093975077479817,
    3.0000000000000018,
    1.6777426874639136,
    0.07229121150761608,
    0.07229121150761608,
    0.005226019261238884,
    0.03614560575380804,
    19.225,
    19.225,
    12.482682291666666,
    13.284451219512198,
    -0.5987053505356037,
    10.18495691876134,
    1.6695010342316154,
    0.6221513204628372,
    -7.245207176175254,
    53.114786632806776,
    41.66483603489018,
    -5.288260191783812,
    1.9992594636693835,
    2.0536800197893768,
    2.0,
    1.2563397343083638,
    0.17671617618995888,
    0.17671617618995888,
    0.031228606927200593,
    0.08835808809497944,
    20.525000000000002,
    20.525000000000002,
    14.123397817460319,
    14.134654471544716,
    0.06942562213497179,
    7.69847452997021,
    0.9151697261163131,
    -0.8690229546006499,
    2.1544428233668684,
    52.243940853557156,
    41.227948459699284,
    7.874394232131946,
    1.9999625437002573,
    2.0122027968020872,
    1.9999999999999996,
    1.248446558174932,
    0.07556393707590262,
    0.07556393707590262,
    0.00570990858641097,
    0.03778196853795131,
    -0.2570484562536639,
    10.465110602472866,
    1.8283682021227905,
    1.030380018001125,
    0.15919112948383018,
    0.15919112948383018,
    0.009414571159702777,
    0.11043666810474123,
];

// ─── Residual-floor diagnosis (GH #65) ──────────────────────────────────────

fn warp_cfg(method: WarpOptimMethod, nbhd: usize, cost: DpEdgeCost) -> WarpOptimConfig {
    let mut w = WarpOptimConfig::default();
    w.method = method;
    w.nbhd_dim = nbhd;
    w.dp_cost = cost;
    w
}

fn dp2() -> WarpOptimConfig {
    warp_cfg(WarpOptimMethod::Dp, 10, DpEdgeCost::PiecewiseLinear)
}

fn rbfgs() -> WarpOptimConfig {
    warp_cfg(WarpOptimMethod::Rbfgs, 7, DpEdgeCost::PiecewiseConstant)
}

/// The optimiser variants of the diagnosis table.
fn variants() -> Vec<(&'static str, WarpOptimConfig)> {
    vec![
        ("dp7", WarpOptimConfig::default()),
        ("dp10+lin", dp2()),
        (
            "dp20+lin",
            warp_cfg(WarpOptimMethod::Dp, 20, DpEdgeCost::PiecewiseLinear),
        ),
        ("rbfgs", rbfgs()),
    ]
}

fn shape_cfg(metric: CurveShapeMetric, warp: WarpOptimConfig) -> CurveShapeConfig {
    let mut c = CurveShapeConfig::default();
    c.metric = metric;
    c.warp_optim = warp;
    c
}

/// Distance of `β1` and the analytic `β1∘γ0` after applying the **exact**
/// inverse warp through the pipeline (linear resampling of the curve,
/// finite-difference SRVF, Procrustes, trapezoid distance) — the floor of
/// everything except the warp optimiser.
fn oracle_distance(m: usize, metric: CurveShapeMetric) -> f64 {
    let cfg = shape_cfg(metric, WarpOptimConfig::default());
    let grid = Grid::new(m);
    let p1 = prepare(&sample(m, c_open), &cfg, &grid).unwrap();
    let p2 = prepare(&sample(m, |t| c_open(gam0(t))), &cfg, &grid).unwrap();
    let ginv: Vec<f64> = grid.t.iter().map(|&s| gam0_inv(s)).collect();
    let bw = geometry::warp_curve(&p2.beta, &grid, &ginv);
    let qw = srvf_cfg(&bw, &cfg, &grid);
    let r = best_rotation(&p1.q, &qw, &cfg, &grid);
    let c = make_candidate(&p1.q, r, ginv, 0, &bw, &qw, &cfg, &grid);
    c.distance
}

/// Residual table `d(β, β∘γ0)` versus `m` for every optimiser (printed with
/// `--nocapture --ignored`); see the SUMMARY of quick task 261126-ic6.
#[test]
#[ignore = "diagnostic table, run with --ignored --nocapture"]
fn residual_floor_table() {
    let ms = [41usize, 81, 161, 321];
    for (label, metric) in [
        ("SRVF", CurveShapeMetric::Srvf),
        (
            "(a,b)=(2,0.5) λ=2",
            CurveShapeMetric::Elastic { a: 2.0, b: 0.5 },
        ),
    ] {
        eprintln!("\n{label}: d(β, β∘γ0), analytic warped curve");
        eprintln!(
            "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "m",
            "oracle",
            "dp7",
            "dp10+lin",
            "dp20+lin",
            "rbfgs",
            "dp7 it6",
            "rbfgs it6",
            "dp7 norot",
            "rbfgs nrt",
            "|γ7-γ*|∞"
        );
        for &m in &ms {
            let b1 = sample(m, c_open);
            let b2 = sample(m, |t| c_open(gam0(t)));
            let mut row = format!("{m:>5} {:>10.5}", oracle_distance(m, metric));
            for (_, w) in variants() {
                let d = curve_shape_distance(&b1, &b2, &shape_cfg(metric, w)).unwrap();
                row += &format!(" {d:>10.5}");
            }
            for (iters, rot) in [(6, true), (2, false)] {
                for w in [WarpOptimConfig::default(), rbfgs()] {
                    let mut c = shape_cfg(metric, w);
                    c.align_iter = iters;
                    c.rotation = rot;
                    let d = curve_shape_distance(&b1, &b2, &c).unwrap();
                    row += &format!(" {d:>10.5}");
                }
            }
            let res =
                curve_shape_align_pair(&b1, &b2, &shape_cfg(metric, WarpOptimConfig::default()))
                    .unwrap();
            let err = res
                .gamma
                .iter()
                .enumerate()
                .map(|(j, g)| (g - gam0_inv(j as f64 / (m - 1) as f64)).abs())
                .fold(0.0, f64::max);
            row += &format!(" {err:>10.5}");
            eprintln!("{row}");
        }
    }
    // 1-D elastic distance (the #49 case), f vs f∘γ0, unnormalised SRSF L².
    let bump = |t: f64, c: f64, s: f64| (-(t - c).powi(2) / s).exp();
    let f = |s: f64| (2.0 * PI * s).sin() + 0.6 * bump(s, 0.7, 0.005) + s;
    eprintln!("\n1-D elastic distance d(f, f∘γ0)");
    eprintln!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "m", "oracle", "dp7", "dp10+lin", "dp20+lin", "rbfgs"
    );
    for &m in &ms {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let f1: Vec<f64> = t.iter().map(|&s| f(s)).collect();
        let f2: Vec<f64> = t.iter().map(|&s| f(gam0(s))).collect();
        let ginv: Vec<f64> = t.iter().map(|&s| gam0_inv(s)).collect();
        let fa = crate::alignment::reparameterize_curve(&f2, &t, &ginv);
        let q1 = crate::alignment::srsf_transform(
            &FdMatrix::from_column_major(f1.clone(), 1, m).unwrap(),
            &t,
        );
        let qa =
            crate::alignment::srsf_transform(&FdMatrix::from_column_major(fa, 1, m).unwrap(), &t);
        let w = crate::helpers::simpsons_weights(&t);
        let oracle = crate::helpers::l2_distance(&q1.row(0), &qa.row(0), &w);
        let mut row = format!("{m:>5} {oracle:>10.5}");
        for (_, w) in variants() {
            let d = crate::alignment::elastic_distance_with_config(&f1, &f2, &t, &w).unwrap();
            row += &format!(" {d:>10.5}");
        }
        eprintln!("{row}");
    }
    // Closed contour: β∘(γ0 + k/(m−1)) (warp and an integer seed shift).
    eprintln!("\nclosed SRVF contour, warp + seed shift");
    eprintln!(
        "{:>5} {:>10} {:>10} {:>10} {:>10}",
        "m", "dp7", "dp10+lin", "rbfgs", "rbfgs it4"
    );
    for m in [61usize, 121] {
        let b1 = sample(m, c_closed);
        let b2 = sample(m, |t| closed_warped(t, m));
        let mut row = format!("{m:>5}");
        for (w, it) in [
            (WarpOptimConfig::default(), 2),
            (dp2(), 2),
            (rbfgs(), 2),
            (rbfgs(), 4),
        ] {
            let mut c = shape_cfg(CurveShapeMetric::Srvf, w);
            c.closed = true;
            c.align_iter = it;
            row += &format!(" {:>10.5}", curve_shape_distance(&b1, &b2, &c).unwrap());
        }
        eprintln!("{row}");
    }
}

/// The closed contour reparametrised by `γ0` and started `(m − 1)/5` samples later.
fn closed_warped(t: f64, m: usize) -> Vec<f64> {
    let k = (m - 1) / 5;
    c_closed((gam0(t) + k as f64 / (m - 1) as f64) % 1.0)
}

// ─── Refined optimisers: invariance with tighter bounds ─────────────────────

/// RBFGS with enough alternation rounds for the rotation to settle.
fn rbfgs_shape(metric: CurveShapeMetric) -> CurveShapeConfig {
    let mut c = shape_cfg(metric, rbfgs());
    c.align_iter = 4;
    c
}

fn self_distance(m: usize, cfg: &CurveShapeConfig) -> f64 {
    curve_shape_distance(&sample(m, c_open), &sample(m, |t| c_open(gam0(t))), cfg).unwrap()
}

fn assert_valid_warp(g: &[f64]) {
    assert!(g[0].abs() < 1e-12 && (g[g.len() - 1] - 1.0).abs() < 1e-12);
    assert!(g.windows(2).all(|w| w[1] >= w[0]), "non-monotone warp");
}

/// Explicit default options equal the default config; the generic R^d DP with
/// the classic options reproduces the classic R^d DP exactly.
#[test]
fn explicit_default_warp_optim_and_generic_dp_match_classic() {
    let mut cfgs = hash_cfgs();
    for c in &mut cfgs {
        c.warp_optim = WarpOptimConfig::default();
        c.warp_optim.dp_prune = false; // pruning is exact: still bit-identical
    }
    // same process, same libm: the explicit options must reproduce every output bit
    assert_eq!(
        output_bits(&default_path_outputs(&cfgs)),
        output_bits(&default_path_outputs(&hash_cfgs()))
    );
    let m = 41;
    let grid = Grid::new(m);
    let q1 = curve_srvf(&sample(m, c_3d), true).unwrap();
    let q2 = curve_srvf(&sample(m, |t| c_3d(t.powf(1.3))), true).unwrap();
    for lam in [0.0, 0.1] {
        let classic = geometry::dp_warp(&q1, &q2, &grid, lam);
        let generic = geometry::dp_warp_opts(&q1, &q2, &grid, lam, &WarpOptimConfig::default());
        assert_eq!(classic, generic, "lambda {lam}");
    }
}

/// Independent cross-check of the R^d DP options: a planar SRVF whose second
/// coordinate vanishes is a 1-D problem, so the R^d DP with fdasrsf `DP2`
/// options (63 moves, linear cost; golden-tested in 1-D) gives the 1-D warp.
#[test]
fn rd_dp_options_reduce_to_1d_engine() {
    let m = 61;
    let grid = Grid::new(m);
    let a: Vec<f64> = grid.t.iter().map(|&t| (5.0 * t).sin() + 1.3).collect();
    let b: Vec<f64> = grid
        .t
        .iter()
        .map(|&t| (5.0 * t.powf(1.4)).sin() + 1.3)
        .collect();
    let unit = |v: &[f64]| {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter().map(|x| x / n).collect::<Vec<f64>>()
    };
    let (a, b) = (unit(&a), unit(&b));
    let as_mat = |v: &[f64]| {
        let mut q = FdMatrix::zeros(2, m);
        for (j, &x) in v.iter().enumerate() {
            q[(0, j)] = x;
        }
        q
    };
    for (nbhd, cost, scl) in [
        (10, DpEdgeCost::PiecewiseLinear, 1),
        (7, DpEdgeCost::PiecewiseConstant, 3),
        (12, DpEdgeCost::PiecewiseConstant, 1),
    ] {
        let mut w = warp_cfg(WarpOptimMethod::Dp, nbhd, cost);
        w.dp_scl = scl;
        let g_rd = geometry::dp_warp_opts(&as_mat(&a), &as_mat(&b), &grid, 0.0, &w);
        let g_1d = crate::alignment::optim::optimal_warp(&a, &b, &grid.t, &w, None);
        let diff = g_rd
            .iter()
            .zip(&g_1d)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-12, "nbhd {nbhd} {cost:?} scl {scl}: {diff}");
    }
}

/// GH #65 diagnosis as a regression test: the classic DP's self-distance
/// floor does not shrink with `m` (slope quantisation), the DP2 options lower
/// it, and RBFGS reaches the resampling floor and keeps shrinking with `m`.
#[test]
fn rbfgs_removes_grid_independent_floor_srvf() {
    let dp = CurveShapeConfig::default();
    let fine = shape_cfg(CurveShapeMetric::Srvf, dp2());
    let rb = rbfgs_shape(CurveShapeMetric::Srvf);
    let (dp81, dp161) = (self_distance(81, &dp), self_distance(161, &dp));
    let (dp2_81, dp2_161) = (self_distance(81, &fine), self_distance(161, &fine));
    let (rb81, rb161) = (self_distance(81, &rb), self_distance(161, &rb));
    let (or81, or161) = (
        oracle_distance(81, CurveShapeMetric::Srvf),
        oracle_distance(161, CurveShapeMetric::Srvf),
    );
    eprintln!(
        "dp {dp81} {dp161} | dp2 {dp2_81} {dp2_161} | rbfgs {rb81} {rb161} | oracle {or81} {or161}"
    );
    // classic floor ≈ 0.018 at both sizes (does not halve)
    assert!(dp81 > 0.015 && dp161 > 0.015 && dp161 > 0.8 * dp81);
    // DP2 options: lower floor, still grid-independent
    assert!(dp2_81 < 0.7 * dp81 && dp2_161 < 0.7 * dp161);
    // RBFGS: an order of magnitude below the classic floor, within 1.5× of the
    // exact-warp (oracle) resampling floor, and shrinking with m
    assert!(rb81 < 0.003 && rb161 < 0.0015, "{rb81} {rb161}");
    assert!(rb81 < 1.5 * or81 && rb161 < 1.5 * or161);
    assert!(rb161 < 0.6 * rb81);
}

/// RBFGS undoes rotation + scale + translation + warp; the invariance test of
/// `tests::distance_invariant_to_reparametrisation_and_rotation` with a
/// tolerance 10× tighter (0.003 instead of the DP-floor bound 0.03), and the
/// alignment's warp approximates the exact inverse warp.
#[test]
fn rbfgs_invariance_rigid_and_warp_tight() {
    let m = 80;
    let b1 = sample(m, c_open);
    let r = {
        let (c, s) = (0.9f64.cos(), 0.9f64.sin());
        let mut r = FdMatrix::zeros(2, 2);
        r[(0, 0)] = c;
        r[(0, 1)] = -s;
        r[(1, 0)] = s;
        r[(1, 1)] = c;
        r
    };
    let mut b2 = geometry::scaled(&geometry::rotate(&r, &sample(m, |t| c_open(gam0(t)))), 0.4);
    for j in 0..m {
        b2[(0, j)] += 1.0;
        b2[(1, j)] -= 2.0;
    }
    let res = curve_shape_align_pair(&b1, &b2, &rbfgs_shape(CurveShapeMetric::Srvf)).unwrap();
    let dp = curve_shape_distance(&b1, &b2, &CurveShapeConfig::default()).unwrap();
    eprintln!("rbfgs {} dp {dp}", res.distance);
    assert!(res.distance < 0.003, "{}", res.distance);
    assert!(res.distance < dp / 5.0);
    assert_valid_warp(&res.gamma);
    let err = res
        .gamma
        .iter()
        .enumerate()
        .map(|(j, g)| (g - gam0_inv(j as f64 / (m - 1) as f64)).abs())
        .fold(0.0, f64::max);
    assert!(err < 5e-3, "warp error {err}");
    // the recovered rotation undoes R
    assert!((res.rotation[(1, 0)].atan2(res.rotation[(0, 0)]) + 0.9).abs() < 5e-3);
}

/// The (a,b) invariance test of `ab_tests::reparametrisation_invariance`
/// (bound 0.05 at the DP floor, ≈0.04 at λ = 2) with RBFGS: bound 0.015
/// (observed 0.001–0.007, 0.012 at λ = 2) and at least 3× below the DP.
#[test]
fn rbfgs_invariance_ab_metrics_tight() {
    let m = 101;
    let gam = |t: f64| t + 0.25 * (PI * t).sin() / PI;
    let c = |t: f64| vec![t, 0.3 * (3.0 * t).sin() + 0.1 * t * t];
    let c1 = sample(m, c);
    let c2 = sample(m, |t| c(gam(t)));
    for (a, b) in [(1.0, 0.5), (1.0, 2.0), (1.0, 1.0), (3.0, 1.0), (2.0, 0.5)] {
        let metric = CurveShapeMetric::Elastic { a, b };
        let dp =
            curve_shape_distance(&c1, &c2, &shape_cfg(metric, WarpOptimConfig::default())).unwrap();
        let rb = curve_shape_distance(&c1, &c2, &rbfgs_shape(metric)).unwrap();
        eprintln!("a={a} b={b}: dp {dp} rbfgs {rb}");
        assert!(rb < 0.015, "a={a} b={b}: {rb}");
        assert!(rb < dp / 3.0, "a={a} b={b}: rbfgs {rb} vs dp {dp}");
    }
}

/// Closed contours: warp + seed shift. RBFGS lowers the default DP residual
/// (≈0.029 at m = 121) by more than half.
#[test]
fn rbfgs_closed_contour_residual() {
    let m = 121;
    let b1 = sample(m, c_closed);
    let b2 = sample(m, |t| closed_warped(t, m));
    let mut dp = CurveShapeConfig::default();
    dp.closed = true;
    let mut rb = rbfgs_shape(CurveShapeMetric::Srvf);
    rb.closed = true;
    let d_dp = curve_shape_distance(&b1, &b2, &dp).unwrap();
    let res = curve_shape_align_pair(&b1, &b2, &rb).unwrap();
    eprintln!("closed: dp {d_dp} rbfgs {}", res.distance);
    assert!(res.distance < 0.02 && res.distance < 0.5 * d_dp);
    assert_valid_warp(&res.gamma);
    for q in [&res.q1, &res.q2_aligned] {
        assert!(closure_residual(q).unwrap().iter().all(|v| v.abs() < 1e-9));
    }
}

/// The refined optimisers flow through the group statistics (single DP call
/// site): the Karcher mean of warped copies has a smaller residual spread.
#[test]
fn rbfgs_karcher_mean_residuals_shrink() {
    let m = 61;
    let set = curves_to_set(&[
        sample(m, c_open),
        sample(m, |t| c_open(gam0(t))),
        sample(m, |t| c_open(t.powf(1.15))),
    ])
    .unwrap();
    let dp = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 10, 1e-5).unwrap();
    let rb =
        curve_shape_karcher_mean(&set, &rbfgs_shape(CurveShapeMetric::Srvf), 10, 1e-5).unwrap();
    let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    eprintln!("karcher: dp {:?} rbfgs {:?}", dp.distances, rb.distances);
    assert!(max(&rb.distances) < 0.5 * max(&dp.distances));
    assert!(max(&rb.distances) < 0.01);
}

#[test]
fn warp_optim_validation() {
    let mut c = CurveShapeConfig::default();
    c.warp_optim.lambda = 0.1;
    assert!(c.validate().is_err(), "warp_optim.lambda must stay 0");
    let mut c = CurveShapeConfig::default();
    c.warp_optim.nbhd_dim = 0;
    assert!(c.validate().is_err());
    let mut c = CurveShapeConfig::default();
    c.warp_optim.penalty = WarpObjectivePenalty::Roughness;
    assert!(c.validate().is_ok(), "no penalty weight: fine");
    c.lambda = 0.1;
    assert!(c.validate().is_err(), "DP cannot express roughness");
    c.warp_optim.method = WarpOptimMethod::Rbfgs;
    assert!(c.validate().is_ok());
    let m = 30;
    let b = sample(m, c_open);
    let d = curve_shape_distance(&b, &sample(m, |t| c_open(gam0(t))), &c).unwrap();
    assert!(d.is_finite());
}
