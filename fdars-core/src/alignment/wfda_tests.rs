#![allow(clippy::field_reassign_with_default)]

use super::golden::*;
use super::*;

const N: usize = 8;
const M: usize = 41;
/// fdapace's BOBYQA (minqa defaults: `npt = nknots + 2` interpolation points,
/// trust-region `rhoend` stop) returns knot heights accurate to ~1e-5 on this
/// problem (fdars' Nelder–Mead reaches equal or lower objectives); the
/// tolerances below bound that optimiser difference.
/// Registered curves differ more because the bimodal test curves are steep.
const WARP_TOL: f64 = 1e-4;
const ALIGNED_TOL: f64 = 2e-3;
const COST_REL_TOL: f64 = 1e-4;

fn golden_data() -> (FdMatrix, Vec<f64>) {
    let t: Vec<f64> = (0..M).map(|j| 10.0 * j as f64 / (M - 1) as f64).collect();
    let data = FdMatrix::from_column_major(Y_DATA.to_vec(), N, M).unwrap();
    (data, t)
}

fn golden_config(choice: WfdaChoice, cost: WfdaCostMeasure) -> WfdaConfig {
    let mut c = WfdaConfig::default();
    c.subset_prop = 1.0;
    c.choice = choice;
    c.cost_measure = cost;
    c
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Warps on fdapace's [0, 1] work scale (argvals = seq(0, 10, len = 41)).
fn work_scale(mat: &FdMatrix) -> Vec<f64> {
    mat.as_slice().iter().map(|v| v / 10.0).collect()
}

struct Golden<'a> {
    lambda: f64,
    h: &'a [f64],
    hinv: Option<&'a [f64]>,
    aligned: &'a [f64],
    costs: &'a [f64],
}

fn check_golden(res: &WfdaResult, g: &Golden, warp_tol: f64, value_tol: f64, cost_rel: f64) {
    assert!((res.lambda - g.lambda).abs() <= 1e-12 * g.lambda);
    let dh = max_abs_diff(&work_scale(&res.warps), g.h);
    assert!(dh < warp_tol, "h differs from fdapace by {dh}");
    if let Some(hinv) = g.hinv {
        let d = max_abs_diff(&work_scale(&res.inverse_warps), hinv);
        assert!(d < warp_tol, "hInv differs from fdapace by {d}");
    }
    let da = max_abs_diff(res.aligned.as_slice(), g.aligned);
    assert!(da < value_tol, "aligned differs from fdapace by {da}");
    for (c, gc) in res.costs.iter().zip(g.costs) {
        assert!(
            (c - gc).abs() <= cost_rel * gc.abs(),
            "cost {c} vs fdapace {gc}"
        );
    }
}

#[test]
fn golden_fdapace_legacy_truncated() {
    let (data, t) = golden_data();
    let cfg = golden_config(WfdaChoice::Truncated, WfdaCostMeasure::FdapaceLegacy);
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    let g = Golden {
        lambda: LEG_TRUNC_LAMBDA,
        h: &LEG_TRUNC_H,
        hinv: Some(&LEG_TRUNC_HINV),
        aligned: &LEG_TRUNC_ALIGNED,
        costs: &LEG_TRUNC_COSTS,
    };
    check_golden(&res, &g, WARP_TOL, ALIGNED_TOL, COST_REL_TOL);
}

#[test]
fn golden_fdapace_legacy_weighted() {
    let (data, t) = golden_data();
    let cfg = golden_config(WfdaChoice::Weighted, WfdaCostMeasure::FdapaceLegacy);
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    let g = Golden {
        lambda: LEG_WEIGHT_LAMBDA,
        h: &LEG_WEIGHT_H,
        hinv: None,
        aligned: &LEG_WEIGHT_ALIGNED,
        costs: &LEG_WEIGHT_COSTS,
    };
    check_golden(&res, &g, WARP_TOL, ALIGNED_TOL, COST_REL_TOL);
}

#[test]
fn golden_fixed_cost_truncated() {
    let (data, t) = golden_data();
    let cfg = golden_config(WfdaChoice::Truncated, WfdaCostMeasure::PairwiseFit);
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    let g = Golden {
        lambda: FIX_TRUNC_LAMBDA,
        h: &FIX_TRUNC_H,
        hinv: Some(&FIX_TRUNC_HINV),
        aligned: &FIX_TRUNC_ALIGNED,
        costs: &FIX_TRUNC_COSTS,
    };
    check_golden(&res, &g, WARP_TOL, ALIGNED_TOL, COST_REL_TOL);
}

#[test]
fn golden_fixed_cost_weighted() {
    let (data, t) = golden_data();
    let cfg = golden_config(WfdaChoice::Weighted, WfdaCostMeasure::PairwiseFit);
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    let g = Golden {
        lambda: FIX_WEIGHT_LAMBDA,
        h: &FIX_WEIGHT_H,
        hinv: None,
        aligned: &FIX_WEIGHT_ALIGNED,
        costs: &FIX_WEIGHT_COSTS,
    };
    check_golden(&res, &g, WARP_TOL, ALIGNED_TOL, COST_REL_TOL);
}

#[test]
fn golden_fixed_cost_three_knots_user_lambda() {
    let (data, t) = golden_data();
    let mut cfg = golden_config(WfdaChoice::Truncated, WfdaCostMeasure::PairwiseFit);
    cfg.nknots = 3;
    cfg.lambda = Some(1e-3);
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    let g = Golden {
        lambda: FIX_K3_L_LAMBDA,
        h: &FIX_K3_L_H,
        hinv: None,
        aligned: &FIX_K3_L_ALIGNED,
        costs: &FIX_K3_L_COSTS,
    };
    check_golden(&res, &g, WARP_TOL, ALIGNED_TOL, COST_REL_TOL);
}

// ---------------------------------------------------------------------------
// Invariants and hand-computed cases
// ---------------------------------------------------------------------------

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

fn bimodal(s: f64) -> f64 {
    1.5 * (-((s - 0.3) / 0.08_f64).powi(2)).exp() + (-((s - 0.7) / 0.1_f64).powi(2)).exp() + 0.2
}

/// `g(t) = t + b·t(1 − t)` and its inverse (monotone for |b| < 1).
fn quad_warp(b: f64, t: f64) -> f64 {
    t + b * t * (1.0 - t)
}

fn quad_warp_inv(b: f64, s: f64) -> f64 {
    if b.abs() < 1e-14 {
        return s;
    }
    ((1.0 + b) - ((1.0 + b) * (1.0 + b) - 4.0 * b * s).sqrt()) / (2.0 * b)
}

fn matrix_from_fn(n: usize, t: &[f64], f: impl Fn(usize, f64) -> f64) -> FdMatrix {
    let mut d = FdMatrix::zeros(n, t.len());
    for i in 0..n {
        for (j, &tj) in t.iter().enumerate() {
            d[(i, j)] = f(i, tj);
        }
    }
    d
}

#[test]
fn identically_shaped_curves_give_identity_warps() {
    let t = grid(31);
    // Power-of-two amplitudes: the normalised curves are bit-identical.
    let amps = [1.0, 2.0, 0.5, 4.0, 0.25];
    let data = matrix_from_fn(amps.len(), &t, |i, s| amps[i] * bimodal(s));
    for choice in [WfdaChoice::Truncated, WfdaChoice::Weighted] {
        let mut cfg = WfdaConfig::default();
        cfg.subset_prop = 1.0;
        cfg.choice = choice;
        let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
        for i in 0..amps.len() {
            assert!(max_abs_diff(&res.warps.row(i), &t) < 1e-12);
            assert!(max_abs_diff(&res.inverse_warps.row(i), &t) < 1e-12);
            assert!(max_abs_diff(&res.aligned.row(i), &data.row(i)) < 1e-12);
            assert!(res.costs[i] < 1e-24);
        }
    }
}

#[test]
fn recovers_exact_piecewise_linear_pairwise_warp() {
    // phi: knots (1/3, 2/3) -> heights (0.25, 0.6), representable with nknots = 2.
    let knots = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    let heights = [0.0, 0.25, 0.6, 1.0];
    let phi = |s: f64| linear_interp(&knots, &heights, s);
    let phi_inv = |s: f64| linear_interp(&heights, &knots, s);
    // Maximum at t = 1 for both curves, so the amplitude normalisation is exact.
    let f = |s: f64| {
        0.5 * (-((s - 0.25) / 0.08_f64).powi(2)).exp()
            + 0.4 * (-((s - 0.6) / 0.1_f64).powi(2)).exp()
            + 1.5 * s
            + 0.1
    };
    let t = grid(201);
    let data = matrix_from_fn(2, &t, |i, s| if i == 0 { f(s) } else { f(phi_inv(s)) });
    let mut cfg = WfdaConfig::default();
    cfg.subset_prop = 1.0;
    cfg.lambda = Some(0.0);
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    assert_eq!(res.partners, vec![vec![1], vec![0]]);
    let target: Vec<f64> = t.iter().map(|&s| phi(s)).collect();
    let target_inv: Vec<f64> = t.iter().map(|&s| phi_inv(s)).collect();
    // One partner: the inverse warp is the pairwise warp itself.
    assert_eq!(res.inverse_warps.row(0), res.pairwise_warps[0].row(0));
    let e = max_abs_diff(&res.inverse_warps.row(0), &target);
    assert!(e < 2e-3, "pairwise warp error {e}");
    let e = max_abs_diff(&res.warps.row(0), &target_inv);
    assert!(e < 2e-3, "warp error {e}");
    // Curve 0 registered onto curve 1's time scale reproduces curve 1.
    let e = max_abs_diff(&res.aligned.row(0), &data.row(1));
    assert!(e < 5e-3, "aligned error {e}");
}

#[test]
fn recovers_average_of_true_pairwise_warps() {
    // y_i = f ∘ γ_i with γ_i⁻¹ = g_{b_i}; the true pairwise warp is
    // h_ik = g_{b_k} ∘ γ_i, so h_i⁻¹ = mean_{k≠i} g_{b_k} ∘ γ_i.
    let b = [-0.4, -0.25, -0.1, 0.05, 0.3, 0.4];
    let n = b.len();
    let t = grid(101);
    let data = matrix_from_fn(n, &t, |i, s| bimodal(quad_warp_inv(b[i], s)));
    let mut cfg = WfdaConfig::default();
    cfg.subset_prop = 1.0;
    cfg.nknots = 7;
    cfg.truncation_quantile = 1.0; // keep every partner
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    for i in 0..n {
        let target: Vec<f64> = t
            .iter()
            .map(|&s| {
                let g = quad_warp_inv(b[i], s);
                (0..n)
                    .filter(|&k| k != i)
                    .map(|k| quad_warp(b[k], g))
                    .sum::<f64>()
                    / (n - 1) as f64
            })
            .collect();
        let e = max_abs_diff(&res.inverse_warps.row(i), &target);
        assert!(e < 1e-2, "curve {i}: inverse-warp error {e}");
        // Σ b = 0, so y_i ∘ h_i = f ∘ g_{−b_i/(n−1)}⁻¹.
        let predicted: Vec<f64> = t
            .iter()
            .map(|&s| bimodal(quad_warp_inv(-b[i] / (n - 1) as f64, s)))
            .collect();
        let e = max_abs_diff(&res.aligned.row(i), &predicted);
        // max|f'| ≈ 16.1, so a 1e-2 time error moves the curve by at most 0.162.
        assert!(e < 16.2 * 1e-2, "curve {i}: aligned error {e}");
    }
}

fn check_warp_properties(res: &WfdaResult, t: &[f64]) {
    let m = t.len();
    let mats = std::iter::once(&res.warps)
        .chain(std::iter::once(&res.inverse_warps))
        .chain(res.pairwise_warps.iter());
    for mat in mats {
        for r in 0..mat.nrows() {
            let h = mat.row(r);
            assert!(h.windows(2).all(|p| p[1] >= p[0]), "non-monotone warp");
            assert!((h[0] - t[0]).abs() < 1e-12 && (h[m - 1] - t[m - 1]).abs() < 1e-12);
        }
    }
    for i in 0..res.warps.nrows() {
        let (h, hinv) = (res.warps.row(i), res.inverse_warps.row(i));
        // Linear interpolation of the monotone h between grid nodes is off by
        // at most one cell increment of h.
        let spacing = h.windows(2).map(|p| p[1] - p[0]).fold(0.0, f64::max);
        for j in 0..m {
            // h is the grid inverse of h⁻¹ (R approx(x = hInv, y = t, xout = t)).
            assert!((linear_interp(&hinv, t, t[j]) - h[j]).abs() < 1e-12);
            assert!((linear_interp(t, &h, hinv[j]) - t[j]).abs() <= spacing + 1e-12);
        }
    }
}

#[test]
fn warps_are_monotone_inverse_pairs_pwl_and_random_search() {
    let (data, t) = golden_data();
    let mut cfg = WfdaConfig::default();
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    check_warp_properties(&res, &t);
    cfg.is_pwl = false;
    cfg.nknots = 0; // ignored by the random search
    let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
    check_warp_properties(&res, &t);
    assert_eq!(res, pairwise_synchronization(&data, &t, &cfg).unwrap());
}

#[test]
fn inverse_warp_is_the_truncated_or_weighted_pairwise_average() {
    let (data, t) = golden_data();
    for choice in [WfdaChoice::Truncated, WfdaChoice::Weighted] {
        let mut cfg = WfdaConfig::default();
        cfg.subset_prop = 1.0;
        cfg.choice = choice;
        let res = pairwise_synchronization(&data, &t, &cfg).unwrap();
        for i in 0..N {
            let costs = &res.pairwise_costs[i];
            let k = costs.len();
            assert_eq!(k, N - 1);
            let weights: Vec<f64> = match choice {
                WfdaChoice::Truncated => {
                    // Type-7 0.9 quantile of 7 costs: sorted[5] + 0.4 (sorted[6] − sorted[5]).
                    let mut s = costs.clone();
                    s.sort_by(f64::total_cmp);
                    let q = s[5] + 0.4 * (s[6] - s[5]);
                    let w: Vec<f64> = costs
                        .iter()
                        .map(|&c| if c <= q { 1.0 } else { 0.0 })
                        .collect();
                    assert_eq!(w.iter().sum::<f64>(), 6.0); // drops the worst partner
                    w
                }
                _ => costs.iter().map(|c| 1.0 / c).collect(),
            };
            let total: f64 = weights.iter().sum();
            for j in 0..M {
                let avg = (0..k)
                    .map(|r| weights[r] * res.pairwise_warps[i][(r, j)])
                    .sum::<f64>()
                    / total;
                assert!((avg - res.inverse_warps[(i, j)]).abs() < 1e-12);
            }
            let mean_cost = costs.iter().sum::<f64>() / k as f64;
            assert!((res.costs[i] - mean_cost).abs() < 1e-15);
        }
    }
}

#[test]
fn partner_subsets_are_seeded_and_sized_like_fdapace() {
    let t = grid(21);
    // K = round_half_even(0.5 · (n − 1)): n = 6 → 2, n = 8 → 4, n = 9 → 4.
    for (n, k) in [(6, 2), (8, 4), (9, 4)] {
        let data = matrix_from_fn(n, &t, |i, s| bimodal(quad_warp(0.1 * i as f64 - 0.3, s)));
        let res = pairwise_synchronization(&data, &t, &WfdaConfig::default()).unwrap();
        for (i, p) in res.partners.iter().enumerate() {
            assert_eq!(p.len(), k);
            assert!(!p.contains(&i) && p.iter().all(|&c| c < n));
            let mut q = p.clone();
            q.sort_unstable();
            q.dedup();
            assert_eq!(q.len(), k);
        }
        let again = pairwise_synchronization(&data, &t, &WfdaConfig::default()).unwrap();
        assert_eq!(res, again);
        let mut cfg = WfdaConfig::default();
        cfg.seed = 1;
        let other = pairwise_synchronization(&data, &t, &cfg).unwrap();
        assert_ne!(res.partners, other.partners);
    }
}

#[test]
fn quantile_and_grid_inverse_match_r() {
    // R: quantile(c(1,2,3,4,10), 0.9) = 7.6; quantile(c(3,1,2), 0.5) = 2.
    assert!((quantile_type7(&[1.0, 2.0, 3.0, 4.0, 10.0], 0.9) - 7.6).abs() < 1e-12);
    assert_eq!(quantile_type7(&[3.0, 1.0, 2.0], 0.5), 2.0);
    assert_eq!(quantile_type7(&[1.0, 2.0, 3.0, 4.0, 10.0], 1.0), 10.0);
    assert_eq!(quantile_type7(&[5.0], 0.9), 5.0);
    // R: approx(x = c(0,.5,.5,.75,1), y = c(0,.25,.5,.75,1), xout = y)$y
    //    = 0 0.1875 0.375 0.75 1 (ties = mean).
    let w = [0.0, 0.25, 0.5, 0.75, 1.0];
    let h = invert_on_grid(&w, &[0.0, 0.5, 0.5, 0.75, 1.0]);
    assert_eq!(h, vec![0.0, 0.1875, 0.375, 0.75, 1.0]);
}

#[test]
fn weighted_average_with_zero_costs_takes_the_limit() {
    let warps = vec![
        vec![0.0, 0.2, 1.0],
        vec![0.0, 0.9, 1.0],
        vec![0.0, 0.4, 1.0],
    ];
    let mut cfg = WfdaConfig::default();
    cfg.choice = WfdaChoice::Weighted;
    let avg = average_warps(&warps, &[0.0, 1.0, 0.0], &cfg);
    assert!((avg[1] - 0.3).abs() < 1e-15);
    // 1/cost weights 1, 1/2, 1/4: (0.2 + 0.45 + 0.1) / 1.75.
    let avg = average_warps(&warps, &[1.0, 2.0, 4.0], &cfg);
    assert!((avg[1] - 0.75 / 1.75).abs() < 1e-15);
}

#[test]
fn rejects_invalid_inputs() {
    let t = grid(11);
    let data = matrix_from_fn(4, &t, |i, s| bimodal(quad_warp(0.1 * i as f64, s)));
    let run = |cfg: &WfdaConfig| pairwise_synchronization(&data, &t, cfg);
    let mut bad: Vec<WfdaConfig> = Vec::new();
    for (field, value) in [
        ("subset", 0.0),
        ("subset", 1.5),
        ("quantile", 0.0),
        ("lambda", -1.0),
    ] {
        let mut c = WfdaConfig::default();
        match field {
            "subset" => c.subset_prop = value,
            "quantile" => c.truncation_quantile = value,
            _ => c.lambda = Some(value),
        }
        bad.push(c);
    }
    let mut c = WfdaConfig::default();
    c.nknots = 8;
    bad.push(c);
    let mut c = WfdaConfig::default();
    c.max_evals = 0;
    bad.push(c);
    let mut c = WfdaConfig::default();
    c.subset_prop = 0.1; // round(0.3) = 0 partners
    bad.push(c);
    for c in &bad {
        assert!(
            matches!(run(c), Err(FdarError::InvalidParameter { .. })),
            "{c:?}"
        );
    }
    let cfg = WfdaConfig::default();
    let one = matrix_from_fn(1, &t, |_, s| bimodal(s));
    assert!(pairwise_synchronization(&one, &t, &cfg).is_err());
    assert!(pairwise_synchronization(&data, &t[..10], &cfg).is_err());
    let mut zero = data.clone();
    for j in 0..t.len() {
        zero[(2, j)] = 0.0;
    }
    assert!(pairwise_synchronization(&zero, &t, &cfg).is_err());
    let mut nan = data.clone();
    nan[(0, 3)] = f64::NAN;
    assert!(pairwise_synchronization(&nan, &t, &cfg).is_err());
    let mut rev = t.clone();
    rev.swap(2, 3);
    assert!(pairwise_synchronization(&data, &rev, &cfg).is_err());
}
