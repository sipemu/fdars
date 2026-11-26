// Sealed-config idiom (`default()` + field assignment) mirrors external usage.
#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::test_helpers::{libm_canary, ValuePin};
use rand::SeedableRng;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Two-bump template.
fn g0(s: f64) -> f64 {
    (-((s - 0.35) / 0.1).powi(2)).exp() + 0.7 * (-((s - 0.7) / 0.08).powi(2)).exp()
}

/// Smoother two-bump template for the in-model studies.
fn g1(s: f64) -> f64 {
    (-((s - 0.35) / 0.15).powi(2)).exp() + 0.7 * (-((s - 0.7) / 0.12).powi(2)).exp()
}

/// γ_a(t) = (e^{a t} − 1)/(e^{a} − 1) and its inverse.
fn exp_warp(a: f64, t: f64) -> f64 {
    if a.abs() < 1e-12 {
        t
    } else {
        ((a * t).exp() - 1.0) / (a.exp() - 1.0)
    }
}
fn exp_warp_inv(a: f64, s: f64) -> f64 {
    if a.abs() < 1e-12 {
        s
    } else {
        (1.0 + s * (a.exp() - 1.0)).ln() / a
    }
}

const A: [f64; 6] = [-0.9, -0.5, -0.15, 0.2, 0.55, 0.85];

/// Centre a set of true warps: `γ_i ∘ γ_I`, γ_I the inverse Karcher mean — the
/// identifiable target of a group alignment.
fn centre(truth: &FdMatrix, t: &[f64]) -> FdMatrix {
    let (n, m) = truth.shape();
    let gam_i = sqrt_mean_inverse(truth, t);
    let mut centred = FdMatrix::zeros(n, m);
    for i in 0..n {
        let c = reparameterize_curve(&truth.row(i), t, &gam_i);
        for j in 0..m {
            centred[(i, j)] = c[j];
        }
    }
    centred
}

/// Curves `f_i = g0 ∘ γ_i⁻¹ + noise` (exponential warps) and the centred truth.
fn synth(m: usize, noise: f64, seed: u64) -> (FdMatrix, Vec<f64>, FdMatrix) {
    let t = grid(m);
    let n = A.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = FdMatrix::zeros(n, m);
    let mut truth = FdMatrix::zeros(n, m);
    for (i, &a) in A.iter().enumerate() {
        for j in 0..m {
            let e: f64 = rng.sample(StandardNormal);
            data[(i, j)] = g0(exp_warp_inv(a, t[j])) + noise * e;
            truth[(i, j)] = exp_warp(a, t[j]);
        }
    }
    let centred = centre(&truth, &t);
    (data, t, centred)
}

/// Warps exactly in the sampler's warp space (first harmonic), curves `g1 ∘ γ_i⁻¹ + noise`.
fn synth_in_model(m: usize, noise: f64, seed: u64) -> (FdMatrix, Vec<f64>, FdMatrix) {
    let t = grid(m);
    let coefs: [[f64; 2]; 5] = [
        [0.25, -0.1],
        [-0.2, 0.15],
        [0.1, 0.2],
        [-0.15, -0.2],
        [0.0, -0.05],
    ];
    let n = coefs.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = FdMatrix::zeros(n, m);
    let mut truth = FdMatrix::zeros(n, m);
    let basis = warp_tangent_basis(&t, 1);
    for (i, c) in coefs.iter().enumerate() {
        let v: Vec<f64> = (0..m)
            .map(|j| c[0] * basis[0][j] + c[1] * basis[1][j])
            .collect();
        let r = l2_norm_l2(&v, &t);
        let psi: Vec<f64> = v.iter().map(|&x| r.cos() + r.sin() * x / r).collect();
        let sq: Vec<f64> = psi.iter().map(|p| p * p).collect();
        let mut g = cumtrapz(&sq, &t);
        let tot = g[m - 1];
        g.iter_mut().for_each(|x| *x /= tot);
        // f_i = g0 ∘ γ_i⁻¹ evaluated on the grid via the inverse warp.
        let ginv = crate::warping::invert_gamma(&g, &t);
        for j in 0..m {
            let e: f64 = rng.sample(StandardNormal);
            data[(i, j)] = g1(ginv[j]) + noise * e;
            truth[(i, j)] = g[j];
        }
    }
    let centred = centre(&truth, &t);
    (data, t, centred)
}

fn small_cfg(seed: u64) -> BayesianGroupConfig {
    let mut c = BayesianGroupConfig::default();
    c.n_iter = 600;
    c.burn_in = 300;
    c.n_warp_harmonics = 3;
    c.n_template_basis = 16;
    c.seed = seed;
    c
}

fn max_abs_diff(a: &FdMatrix, b: &FdMatrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Fraction of interior grid points (all curves) where truth lies in the band.
fn coverage(res: &BayesianGroupResult, truth: &FdMatrix) -> f64 {
    let (n, m) = truth.shape();
    let mut hit = 0usize;
    let mut tot = 0usize;
    for i in 0..n {
        for j in 1..m - 1 {
            tot += 1;
            let v = truth[(i, j)];
            if v >= res.gammas_lower[(i, j)] - 1e-12 && v <= res.gammas_upper[(i, j)] + 1e-12 {
                hit += 1;
            }
        }
    }
    hit as f64 / tot as f64
}

fn mean_band_width(res: &BayesianGroupResult) -> f64 {
    let (n, m) = res.gammas.shape();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..m {
            s += res.gammas_upper[(i, j)] - res.gammas_lower[(i, j)];
        }
    }
    s / (n * m) as f64
}

fn is_warp(row: &[f64], a: f64, b: f64) -> bool {
    (row[0] - a).abs() < 1e-12
        && (row[row.len() - 1] - b).abs() < 1e-12
        && row.windows(2).all(|w| w[1] >= w[0] - 1e-12)
}

// ─── Building blocks ─────────────────────────────────────────────────────────

#[test]
fn template_basis_matches_scipy() {
    // fdasrsf 2.7.5 `utility_functions.create_basismatrix(linspace(0,1,11), 6, 3)`
    // (scipy BSpline.design_matrix), printed with repr().
    let reference: [[f64; 6]; 11] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.36, 0.56, 0.08000000000000002, 0.0, 0.0, 0.0],
        [
            0.03999999999999998,
            0.6399999999999999,
            0.32000000000000006,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.31999999999999984,
            0.6600000000000001,
            0.020000000000000035,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.07999999999999996,
            0.74,
            0.18000000000000005,
            0.0,
            0.0,
        ],
        [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.1799999999999998,
            0.7400000000000001,
            0.08000000000000014,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.01999999999999995,
            0.6599999999999998,
            0.32000000000000023,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.31999999999999984,
            0.6400000000000001,
            0.04000000000000007,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.07999999999999996,
            0.5599999999999999,
            0.3600000000000001,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let t = grid(11);
    let cols = template_bspline_basis(&t, 6, 3);
    for (r, row) in reference.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            assert!(
                (cols[k][r] - v).abs() < 1e-12,
                "B[{r},{k}] = {} vs {v}",
                cols[k][r]
            );
        }
    }
    // Partition of unity on a finer grid.
    let t = grid(57);
    let cols = template_bspline_basis(&t, 16, 3);
    for r in 0..t.len() {
        let s: f64 = cols.iter().map(|c| c[r]).sum();
        assert!((s - 1.0).abs() < 1e-12);
    }
}

#[test]
fn warp_basis_is_orthonormal_tangent() {
    let t = grid(41);
    let basis = warp_tangent_basis(&t, 4);
    for a in 0..basis.len() {
        assert!(
            trapz(&basis[a], &t).abs() < 1e-12,
            "basis {a} not tangent at 1"
        );
        for b in 0..basis.len() {
            let prod: Vec<f64> = basis[a].iter().zip(&basis[b]).map(|(x, y)| x * y).collect();
            let expect = if a == b { 1.0 } else { 0.0 };
            assert!((trapz(&prod, &t) - expect).abs() < 1e-12, "<{a},{b}>");
        }
    }
}

#[test]
fn warp_from_coef_gives_valid_warps() {
    let (data, t, _) = synth(31, 0.0, 1);
    let cfg = small_cfg(1);
    let prob = Problem::new(&data, &t, &cfg).unwrap();
    // Zero coefficients → identity, ψ = 1.
    let w = prob.warp_from_coef(0, vec![0.0; 6]).unwrap();
    for j in 0..t.len() {
        assert!((w.gamma[j] - t[j]).abs() < 1e-14);
        assert!((w.psi[j] - 1.0).abs() < 1e-14);
    }
    // A moderate shooting vector → monotone warp of [0,1], ‖ψ‖ = 1.
    let w = prob
        .warp_from_coef(0, vec![0.3, -0.2, 0.1, 0.05, 0.0, -0.1])
        .unwrap();
    assert!(is_warp(&w.gamma, 0.0, 1.0));
    assert!((l2_norm_l2(&w.psi, &t) - 1.0).abs() < 1e-3);
    // ‖v‖ > π/2 along a single mode forces ψ < 0 somewhere → rejected.
    assert!(prob
        .warp_from_coef(0, vec![2.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        .is_none());
}

#[test]
fn split_rhat_and_quantiles_hand_computed() {
    // numpy reference (BDA3 split-R̂ formula, halves of length 3):
    // chains [1..6] and [2,4,..,12] → 2.3021728866442674;
    // chains of length 7 (middle point dropped) → 0.8500815729444263.
    let r = split_rhat(&[
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    ]);
    assert!((r - 2.3021728866442674).abs() < 1e-12, "{r}");
    let r = split_rhat(&[
        vec![0.3, -1.2, 0.8, 0.1, -0.4, 1.1, 0.2],
        vec![-0.1, 0.5, -0.7, 0.9, 0.0, -0.3, 0.6],
    ]);
    assert!((r - 0.8500815729444263).abs() < 1e-12, "{r}");
    assert!(split_rhat(&[vec![1.0, 2.0, 3.0]]).is_nan());
    assert_eq!(split_rhat(&[vec![2.0; 8], vec![2.0; 8]]), 1.0);
    // numpy.percentile(1..8, [2.5, 50, 97.5]) = [1.175, 4.5, 7.825].
    let mut v: Vec<f64> = (1..=8).map(f64::from).collect();
    let (lo, md, hi) = sample_quantiles(&mut v, 0.95);
    assert!((lo - 1.175).abs() < 1e-12 && (md - 4.5).abs() < 1e-12 && (hi - 7.825).abs() < 1e-12);
}

// ─── Sampler behaviour ───────────────────────────────────────────────────────

#[test]
fn same_seed_is_bit_identical_and_chains_use_seed_plus_k() {
    let (data, t, _) = synth(31, 0.01, 2);
    let mut cfg = small_cfg(11);
    cfg.n_iter = 200;
    cfg.burn_in = 100;
    cfg.n_chains = 3;
    let a = bayesian_align_group(&data, &t, &cfg).unwrap();
    let b = bayesian_align_group(&data, &t, &cfg).unwrap();
    assert_eq!(a, b, "identical config must give bit-identical results");
    // Chain k of a multi-chain run is exactly the single-chain run seeded seed + k.
    let mut single = cfg.clone();
    single.n_chains = 1;
    single.seed = 13;
    let s = bayesian_align_group(&data, &t, &single).unwrap();
    assert_eq!(a.chains[2].seed, 13);
    assert_eq!(a.chains[2], s.chains[0]);
    // Different seeds give different chains.
    assert_ne!(a.chains[0].sigma2_trace, a.chains[1].sigma2_trace);
}

#[test]
fn recovers_known_warps_with_sane_acceptance() {
    let (data, t, truth) = synth(41, 0.0, 3);
    let mut cfg = small_cfg(5);
    cfg.n_iter = 3000;
    cfg.burn_in = 1500;
    cfg.n_chains = 2;
    let res = bayesian_align_group(&data, &t, &cfg).unwrap();
    let (n, m) = truth.shape();
    let mut ident = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            ident[(i, j)] = t[j];
        }
    }
    let err = max_abs_diff(&res.gammas, &truth);
    let base = max_abs_diff(&ident, &truth);
    assert!(
        base > 0.1,
        "synthetic warps must be far from identity ({base})"
    );
    assert!(
        err < 0.05,
        "posterior mean warp error {err} (identity baseline {base})"
    );
    // Aligned curves are much closer to each other than the raw curves.
    let spread = |mat: &FdMatrix| -> f64 {
        (0..m)
            .map(|j| {
                let col = mat.column(j);
                let mu = col.iter().sum::<f64>() / n as f64;
                col.iter().map(|v| (v - mu).powi(2)).sum::<f64>()
            })
            .sum()
    };
    assert!(spread(&res.aligned_data) < 0.2 * spread(&data));
    for (i, &acc) in res.warp_acceptance.iter().enumerate() {
        assert!(
            acc > 0.01 && acc < 0.9,
            "warp acceptance of curve {i}: {acc}"
        );
    }
    assert!(res.template_acceptance > 0.01 && res.template_acceptance < 0.9);
    // Template peaks where g0 does (0.35, up to the centring warp).
    let argmax = (0..m / 2)
        .max_by(|&a, &b| res.template[a].total_cmp(&res.template[b]))
        .unwrap();
    assert!(
        (t[argmax] - 0.35).abs() <= 0.1,
        "first template peak at {}",
        t[argmax]
    );
}

#[test]
fn posterior_summaries_are_consistent() {
    let (data, t, _) = synth(31, 0.01, 4);
    let mut cfg = small_cfg(9);
    cfg.n_chains = 2;
    let res = bayesian_align_group(&data, &t, &cfg).unwrap();
    let (n, m) = data.shape();
    let nd = 2 * (cfg.n_iter - cfg.burn_in);
    assert_eq!(res.sigma2_draws.len(), nd);
    assert_eq!(res.gamma_draws.len(), n);
    assert_eq!(res.gamma_draws[0].shape(), (nd, m));
    assert_eq!(res.template_srsf_draws.shape(), (nd, m));
    for i in 0..n {
        for mat in [
            &res.gammas,
            &res.gammas_median,
            &res.gammas_lower,
            &res.gammas_upper,
        ] {
            assert!(
                is_warp(&mat.row(i), 0.0, 1.0),
                "curve {i} summary is not a warp"
            );
        }
        for c in &res.chains {
            assert!(is_warp(&c.posterior_mean_gammas.row(i), 0.0, 1.0));
            assert!(is_warp(&c.mode_gammas.row(i), 0.0, 1.0));
        }
        for j in 0..m {
            assert!(res.gammas_lower[(i, j)] <= res.gammas_median[(i, j)] + 1e-12);
            assert!(res.gammas_median[(i, j)] <= res.gammas_upper[(i, j)] + 1e-12);
        }
    }
    for j in 0..m {
        assert!(res.template_srsf_lower[j] <= res.template_srsf[j] + 1e-12);
        assert!(res.template_srsf[j] <= res.template_srsf_upper[j] + 1e-12);
        assert!(res.template_lower[j] <= res.template_upper[j] + 1e-12);
    }
    assert!(res.sigma2_lower <= res.sigma2_mean && res.sigma2_mean <= res.sigma2_upper);
    // The chain mode is the retained draw with the largest log-likelihood.
    for c in &res.chains {
        let best = c
            .log_likelihood_trace
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        assert_eq!(c.mode_log_likelihood, best);
    }
    // Centred draws: the Karcher mean of each draw's warps is the identity, so the
    // posterior-mean warps are (approximately) centred as a set.
    let gi = sqrt_mean_inverse(&res.gammas, &t);
    for j in 0..m {
        assert!(
            (gi[j] - t[j]).abs() < 0.02,
            "posterior-mean warps not centred at {j}"
        );
    }
}

#[test]
fn more_noise_widens_bands() {
    let mut cfg = small_cfg(21);
    cfg.n_iter = 2000;
    cfg.burn_in = 1000;
    let (lo_data, t, _) = synth(41, 0.002, 6);
    let (hi_data, _, _) = synth(41, 0.03, 6);
    let lo = bayesian_align_group(&lo_data, &t, &cfg).unwrap();
    let hi = bayesian_align_group(&hi_data, &t, &cfg).unwrap();
    assert!(
        hi.sigma2_mean > 1.5 * lo.sigma2_mean,
        "σ² {} vs {}",
        hi.sigma2_mean,
        lo.sigma2_mean
    );
    let (wl, wh) = (mean_band_width(&lo), mean_band_width(&hi));
    assert!(wh > 1.2 * wl, "band width {wh} (noisy) vs {wl} (clean)");
}

#[test]
fn credible_bands_cover_truth() {
    // Small fixed-seed Monte-Carlo study, warps inside the model's warp space, noise added
    // to f (not to the SRSF): the model is misspecified, see the module docs "Calibration"
    // (in-model data: `calibration_tests`, ≈ nominal). Measured 0.874 (deterministic).
    let mut cov = Vec::new();
    for seed in 0..3u64 {
        let (data, t, truth) = synth_in_model(31, 0.02, 100 + seed);
        let mut cfg = small_cfg(7 + seed);
        cfg.n_warp_harmonics = 1;
        cfg.n_iter = 3000;
        cfg.burn_in = 1500;
        let res = bayesian_align_group(&data, &t, &cfg).unwrap();
        cov.push(coverage(&res, &truth));
    }
    let mean = cov.iter().sum::<f64>() / cov.len() as f64;
    println!("short-study coverage {mean:.3} {cov:?}");
    assert!(mean > 0.85, "mean coverage {mean} ({cov:?})");
}

#[test]
#[ignore = "long Monte-Carlo coverage study (10 data seeds x 4 chains x 20000 iterations)"]
fn credible_bands_cover_truth_long_study() {
    let mut cov = Vec::new();
    for seed in 0..10u64 {
        let (data, t, truth) = synth_in_model(31, 0.02, 200 + seed);
        let mut cfg = small_cfg(31 + 4 * seed);
        cfg.n_warp_harmonics = 1;
        cfg.n_iter = 20000;
        cfg.burn_in = 10000;
        cfg.n_chains = 4;
        let res = bayesian_align_group(&data, &t, &cfg).unwrap();
        cov.push(coverage(&res, &truth));
    }
    let mean = cov.iter().sum::<f64>() / cov.len() as f64;
    // Measured 0.829 (per-seed 0.57–1.00) for nominal 0.95 (2026-11-26, also 0.822 with
    // 30 template basis functions; unchanged by the #64 fix, which touches only template
    // summaries). The same samplers on data simulated from the model cover ≈ nominally
    // (0.92–0.96, `calibration_tests`), so the shortfall is misspecification: the SRSF of
    // f + iid noise is heteroscedastic, correlated and heavy-tailed where f' ≈ 0, not the
    // model's iid-Gaussian SRSF noise (module docs "Calibration").
    println!("long-study coverage {mean:.3} {cov:?}");
    assert!(mean > 0.80, "mean coverage {mean} ({cov:?})");
}

#[test]
fn rhat_near_one_on_easy_problem() {
    // Few curves, warps in the model space, wide posterior: 4 chains mix well.
    let (data, t, _) = synth_in_model(21, 0.03, 8);
    let mut cfg = small_cfg(3);
    cfg.n_warp_harmonics = 1;
    cfg.n_template_basis = 12;
    cfg.n_iter = 24000;
    cfg.burn_in = 8000;
    cfg.thin = 4;
    cfg.n_chains = 4;
    let res = bayesian_align_group(&data, &t, &cfg).unwrap();
    assert!(res.rhat_sigma2 < 1.1, "R̂(σ²) = {}", res.rhat_sigma2);
    assert!(
        res.rhat_log_likelihood < 1.1,
        "R̂(loglik) = {}",
        res.rhat_log_likelihood
    );
    for (i, &r) in res.rhat_warps.iter().enumerate() {
        assert!(r < 1.1, "R̂ of warp {i} = {r}");
    }
    // Chains agree: small between-chain spread of the posterior-mean warps.
    for (i, &s) in res.mode_spread.iter().enumerate() {
        assert!(s < 0.02, "mode spread of curve {i} = {s}");
    }
}

#[test]
fn general_domain_and_resampled_grid() {
    // Same curves on [2, 4] with a 25-point internal grid: outputs live on argvals.
    let (data, t, _) = synth(31, 0.0, 9);
    let argvals: Vec<f64> = t.iter().map(|&x| 2.0 + 2.0 * x).collect();
    let mut cfg = small_cfg(2);
    cfg.n_points = 25;
    let res = bayesian_align_group(&data, &argvals, &cfg).unwrap();
    let (n, m) = data.shape();
    assert_eq!(res.gammas.shape(), (n, m));
    for i in 0..n {
        assert!(is_warp(&res.gammas.row(i), 2.0, 4.0));
        assert!(is_warp(&res.gamma_draws[i].row(0), 2.0, 4.0));
    }
    // fmean starts at the mean of the curves' first values.
    let f0 = (0..n).map(|i| data[(i, 0)]).sum::<f64>() / n as f64;
    assert!((res.template[0] - f0).abs() < 1e-12);
}

#[test]
fn rejects_invalid_input() {
    let (data, t, _) = synth(21, 0.0, 1);
    let ok = small_cfg(1);
    assert!(bayesian_align_group(&data, &t[..20], &ok).is_err());
    let bad_t: Vec<f64> = t.iter().rev().copied().collect();
    assert!(bayesian_align_group(&data, &bad_t, &ok).is_err());
    let mut nan = data.clone();
    nan[(0, 3)] = f64::NAN;
    assert!(bayesian_align_group(&nan, &t, &ok).is_err());
    type Mutator = Box<dyn Fn(&mut BayesianGroupConfig)>;
    let cases: Vec<Mutator> = vec![
        Box::new(|c| c.burn_in = 0),
        Box::new(|c| c.burn_in = c.n_iter),
        Box::new(|c| c.thin = 0),
        Box::new(|c| c.n_chains = 0),
        Box::new(|c| c.alpha0 = 0.0),
        Box::new(|c| c.beta0 = -1.0),
        Box::new(|c| c.credible_level = 1.0),
        Box::new(|c| c.pcn_betas = vec![0.1]),
        Box::new(|c| c.pcn_betas = vec![0.0, 0.1, 0.2, 0.3]),
        Box::new(|c| c.pcn_probs = vec![0.0, 0.0, 0.0, 0.0]),
        Box::new(|c| c.warp_prior_sd = 0.0),
        Box::new(|c| c.n_warp_harmonics = 0),
        Box::new(|c| c.n_warp_harmonics = 10),
        Box::new(|c| c.n_template_basis = 2),
        Box::new(|c| c.n_template_basis = 22),
        Box::new(|c| c.n_points = 3),
    ];
    for (k, f) in cases.iter().enumerate() {
        let mut c = ok.clone();
        f(&mut c);
        assert!(
            matches!(
                bayesian_align_group(&data, &t, &c),
                Err(FdarError::InvalidParameter { .. })
            ),
            "case {k} should be rejected"
        );
    }
}

#[test]
fn thinning_keeps_every_kth_draw() {
    let (data, t, _) = synth(21, 0.01, 1);
    let mut cfg = small_cfg(4);
    cfg.n_iter = 301;
    cfg.burn_in = 100;
    let full = bayesian_align_group(&data, &t, &cfg).unwrap();
    cfg.thin = 4;
    let thin = bayesian_align_group(&data, &t, &cfg).unwrap();
    let expect: Vec<f64> = full.chains[0]
        .sigma2_trace
        .iter()
        .step_by(4)
        .copied()
        .collect();
    assert_eq!(thin.chains[0].sigma2_trace, expect);
}

/// fdasrsf 2.7.5 `fdawarp(f, t).time_warping_bayes(iter=20000, burnin=10000, nbasis=16,
/// npoints=41, initcoef=zeros((6, 6)))` on `synth(41, 0.0, _)` (no noise), posterior-mean
/// warps `obj.gam` averaged over numpy seeds 0..4 (script
/// `.planning/quick/261126-4r5-…/golden_time_warping_bayes.py`). Rows = curves.
/// fdasrsf's own seed-to-seed spread of these warps is 0.047 (max abs).
const FDASRSF_GAM: [[f64; 41]; 6] = [
    [
        0.00000, 0.02673, 0.05518, 0.08516, 0.11640, 0.14850, 0.18103, 0.21358, 0.24579, 0.27737,
        0.30814, 0.33800, 0.36694, 0.39498, 0.42221, 0.44871, 0.47457, 0.49989, 0.52472, 0.54912,
        0.57313, 0.59679, 0.62011, 0.64309, 0.66575, 0.68805, 0.70997, 0.73147, 0.75249, 0.77302,
        0.79306, 0.81264, 0.83186, 0.85088, 0.86990, 0.88919, 0.90904, 0.92977, 0.95167, 0.97501,
        1.00000,
    ],
    [
        0.00000, 0.02818, 0.05752, 0.08766, 0.11812, 0.14839, 0.17802, 0.20669, 0.23423, 0.26066,
        0.28614, 0.31092, 0.33530, 0.35957, 0.38395, 0.40856, 0.43345, 0.45852, 0.48366, 0.50868,
        0.53343, 0.55782, 0.58181, 0.60544, 0.62881, 0.65202, 0.67516, 0.69829, 0.72139, 0.74442,
        0.76728, 0.78989, 0.81219, 0.83423, 0.85614, 0.87815, 0.90054, 0.92365, 0.94779, 0.97319,
        1.00000,
    ],
    [
        0.00000, 0.02541, 0.05123, 0.07735, 0.10362, 0.12989, 0.15603, 0.18194, 0.20760, 0.23302,
        0.25824, 0.28333, 0.30839, 0.33350, 0.35872, 0.38410, 0.40963, 0.43531, 0.46106, 0.48683,
        0.51255, 0.53815, 0.56360, 0.58889, 0.61400, 0.63895, 0.66374, 0.68838, 0.71285, 0.73714,
        0.76124, 0.78513, 0.80883, 0.83237, 0.85582, 0.87925, 0.90278, 0.92652, 0.95058, 0.97506,
        1.00000,
    ],
    [
        0.00000, 0.02410, 0.04769, 0.07090, 0.09390, 0.11685, 0.13993, 0.16326, 0.18690, 0.21086,
        0.23509, 0.25950, 0.28398, 0.30840, 0.33270, 0.35686, 0.38089, 0.40487, 0.42889, 0.45307,
        0.47750, 0.50224, 0.52733, 0.55275, 0.57847, 0.60442, 0.63056, 0.65682, 0.68320, 0.70969,
        0.73631, 0.76307, 0.78996, 0.81696, 0.84400, 0.87096, 0.89770, 0.92409, 0.94999, 0.97531,
        1.00000,
    ],
    [
        0.00000, 0.02452, 0.04860, 0.07224, 0.09542, 0.11815, 0.14045, 0.16239, 0.18408, 0.20567,
        0.22734, 0.24926, 0.27160, 0.29448, 0.31797, 0.34208, 0.36674, 0.39185, 0.41727, 0.44287,
        0.46854, 0.49421, 0.51987, 0.54556, 0.57137, 0.59740, 0.62375, 0.65046, 0.67757, 0.70502,
        0.73273, 0.76058, 0.78843, 0.81614, 0.84361, 0.87074, 0.89748, 0.92378, 0.94964, 0.97504,
        1.00000,
    ],
    [
        0.00000, 0.02223, 0.04205, 0.06007, 0.07700, 0.09358, 0.11047, 0.12821, 0.14716, 0.16746,
        0.18900, 0.21144, 0.23437, 0.25737, 0.28012, 0.30252, 0.32465, 0.34674, 0.36913, 0.39217,
        0.41615, 0.44122, 0.46739, 0.49447, 0.52218, 0.55020, 0.57827, 0.60626, 0.63423, 0.66241,
        0.69112, 0.72073, 0.75152, 0.78357, 0.81672, 0.85045, 0.88401, 0.91649, 0.94702, 0.97495,
        1.00000,
    ],
];

/// fdasrsf `obj.fmean` of the same runs (averaged over the 4 seeds).
const FDASRSF_FMEAN: [f64; 41] = [
    0.00000, 0.00004, 0.00018, 0.00081, 0.00283, 0.00765, 0.01749, 0.03532, 0.06464, 0.10960,
    0.17615, 0.27289, 0.40566, 0.55036, 0.65907, 0.71013, 0.72049, 0.70089, 0.63737, 0.52913,
    0.40330, 0.29113, 0.20798, 0.15683, 0.13250, 0.12793, 0.14609, 0.19269, 0.26008, 0.32860,
    0.37906, 0.40430, 0.40128, 0.36426, 0.29945, 0.22368, 0.15122, 0.09440, 0.06045, 0.04452,
    0.03790,
];

#[test]
fn matches_fdasrsf_time_warping_bayes_loosely() {
    // Same data and closest-matching settings: 3 warp harmonics (fdasrsf: 6 coefficients,
    // a constant + 2.5 harmonics), 16 quadratic B-splines, Z-mixture defaults, identity
    // start and raw (uncentred) draws as in fdasrsf. MCMC output is compared
    // distributionally: posterior-mean warps within 0.06 (fdasrsf seed spread 0.047),
    // template peaks within 0.1 of fdasrsf's. (fdasrsf's template is visibly smeared —
    // peak 0.72 vs the true 1.0, σ² ≈ 0.4–0.8 — ours peaks at 0.93 with σ² ≈ 0.035.)
    let (data, t, _) = synth(41, 0.0, 0);
    let mut cfg = BayesianGroupConfig::default();
    cfg.n_iter = 20000;
    cfg.burn_in = 10000;
    cfg.thin = 5;
    cfg.n_warp_harmonics = 3;
    cfg.n_template_basis = 16;
    cfg.n_chains = 4;
    cfg.init_dp = false;
    cfg.center_draws = false;
    let res = bayesian_align_group(&data, &t, &cfg).unwrap();
    let mut worst: f64 = 0.0;
    for (i, row) in FDASRSF_GAM.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            worst = worst.max((res.gammas[(i, j)] - v).abs());
        }
    }
    assert!(worst < 0.06, "max |γ − γ_fdasrsf| = {worst}");
    let peak = |f: &[f64], lo: usize, hi: usize| -> f64 {
        t[(lo..hi).max_by(|&a, &b| f[a].total_cmp(&f[b])).unwrap()]
    };
    for (lo, hi) in [(0, 21), (21, 41)] {
        let (ours, theirs) = (peak(&res.template, lo, hi), peak(&FDASRSF_FMEAN, lo, hi));
        assert!(
            (ours - theirs).abs() <= 0.1,
            "template peak {ours} vs fdasrsf {theirs}"
        );
    }
}

// ─── Default-sampler bit-identity ───────────────────────────────────────────

/// Record one output field in a portable value pin (see [`ValuePin`]).
fn fnv(h: &mut ValuePin, xs: &[f64]) {
    h.push(xs);
}

/// Hash of every field that existed in the pCN-only release (065279f0).
fn result_hash(r: &BayesianGroupResult) -> ValuePin {
    let mut h = ValuePin::new();
    for m in [
        &r.gammas,
        &r.gammas_median,
        &r.gammas_lower,
        &r.gammas_upper,
        &r.aligned_data,
        &r.template_srsf_draws,
    ] {
        fnv(&mut h, m.as_slice());
    }
    for v in [
        &r.template_srsf,
        &r.template_srsf_lower,
        &r.template_srsf_upper,
        &r.template,
        &r.template_lower,
        &r.template_upper,
        &r.warp_acceptance,
        &r.rhat_warps,
        &r.mode_spread,
        &r.sigma2_draws,
    ] {
        fnv(&mut h, v);
    }
    fnv(
        &mut h,
        &[
            r.sigma2_mean,
            r.sigma2_lower,
            r.sigma2_upper,
            r.template_acceptance,
            r.rhat_sigma2,
            r.rhat_log_likelihood,
        ],
    );
    for g in &r.gamma_draws {
        fnv(&mut h, g.as_slice());
    }
    for c in &r.chains {
        fnv(
            &mut h,
            &[c.seed as f64, c.mode_log_likelihood, c.template_acceptance],
        );
        fnv(&mut h, c.posterior_mean_gammas.as_slice());
        fnv(&mut h, c.mode_gammas.as_slice());
        for v in [
            &c.mode_template_srsf,
            &c.template_srsf_mean,
            &c.warp_acceptance,
            &c.sigma2_trace,
            &c.log_likelihood_trace,
        ] {
            fnv(&mut h, v);
        }
    }
    h
}

/// Hash of the [`result_hash`] fields that do not depend on the re-centred template
/// (everything except the template SRSF / function summaries and draws).
fn warp_hash(r: &BayesianGroupResult) -> ValuePin {
    let mut h = ValuePin::new();
    for m in [
        &r.gammas,
        &r.gammas_median,
        &r.gammas_lower,
        &r.gammas_upper,
        &r.aligned_data,
    ] {
        fnv(&mut h, m.as_slice());
    }
    for v in [
        &r.warp_acceptance,
        &r.rhat_warps,
        &r.mode_spread,
        &r.sigma2_draws,
    ] {
        fnv(&mut h, v);
    }
    fnv(
        &mut h,
        &[
            r.sigma2_mean,
            r.sigma2_lower,
            r.sigma2_upper,
            r.template_acceptance,
            r.rhat_sigma2,
            r.rhat_log_likelihood,
        ],
    );
    for g in &r.gamma_draws {
        fnv(&mut h, g.as_slice());
    }
    for c in &r.chains {
        fnv(
            &mut h,
            &[c.seed as f64, c.mode_log_likelihood, c.template_acceptance],
        );
        fnv(&mut h, c.posterior_mean_gammas.as_slice());
        fnv(&mut h, c.mode_gammas.as_slice());
        for v in [&c.warp_acceptance, &c.sigma2_trace, &c.log_likelihood_trace] {
            fnv(&mut h, v);
        }
    }
    h
}

#[test]
fn default_pcn_sampler_is_bit_identical_to_pcn_release() {
    // Pinned on the pCN-only implementation (commit 065279f0) before the infHMC
    // sampler was added: the default sampler must keep producing identical results.
    // GH #64 changed only the re-centred template summaries (exact B-spline evaluation
    // instead of warping the linear interpolant): every field that does not depend on them
    // (`warp_hash`) and the whole `center_draws = false` result keep their pre-#64 hashes,
    // computed on c038fd46; the full default hash changed from 0x17c8a7166fdb441d.
    let (data, t, _) = synth(31, 0.01, 2);
    let mut cfg = small_cfg(11);
    cfg.n_iter = 300;
    cfg.burn_in = 100;
    cfg.thin = 2;
    cfg.n_chains = 2;
    let r = bayesian_align_group(&data, &t, &cfg).unwrap();
    result_hash(&r).assert_close(PCN_RESULT_PIN, 1e-10, "default pCN results");
    warp_hash(&r).assert_close(PCN_RELEASE_WARP_PIN, 1e-10, "pCN warp/σ² results");
    cfg.center_draws = false;
    result_hash(&bayesian_align_group(&data, &t, &cfg).unwrap()).assert_close(
        PCN_RELEASE_RAW_PIN,
        1e-10,
        "pCN center_draws=false results",
    );
}

// The pins below replace FNV-1a hashes of the output bits (warp 0x8544_396a_c9c4_e4f3,
// raw 0xe256_f6f0_c204_2509, full 0x423a_b6ba_faae_cb16), which depend on the platform
// libm's last-ulp results. They were recorded on 925ef514, which reproduced all three
// hashes bit-for-bit on the recording machine, and are compared at rtol 1e-10.

/// Pre-#64 (c038fd46) template-independent fields of the default run.
const PCN_RELEASE_WARP_PIN: &[f64] = &[
    93.02050455510377,
    93.02050455510377,
    63.478383989797564,
    62.658249282757495,
    93.05035653160375,
    93.05035653160375,
    63.54427668478698,
    62.68255696342813,
    91.53775966023242,
    91.53775966023242,
    62.05925360570497,
    61.93723804546742,
    94.46704366335942,
    94.46704366335942,
    64.82986813370763,
    63.333122860297046,
    49.81259783875675,
    50.19514440624196,
    30.784162485971173,
    23.813919071159,
    0.9030100334448161,
    0.9030100334448161,
    0.13672106575989085,
    0.5234113712374582,
    8.969476882640045,
    8.969476882640045,
    13.824037701991099,
    5.072821900515067,
    0.04181820269125868,
    0.04181820269125868,
    0.0002998917025309802,
    0.025510647454415725,
    24.386378163815397,
    24.386378163815397,
    3.0386124993750836,
    12.05718991025918,
    3.6197405468131434,
    3.6197405468131434,
    4.333629832965608,
    3.0424916948491934,
    3492.1527947841123,
    3492.1527947841123,
    2523.349851272,
    2277.48056150494,
    3345.817634750447,
    3345.817634750447,
    2347.7051328480898,
    2200.432018800899,
    3175.8081901911646,
    3175.8081901911646,
    2194.8806473603872,
    2129.3954257559353,
    3030.7952609155245,
    3030.7952609155245,
    2052.487871368722,
    2057.938617354927,
    2833.002753939271,
    2833.002753939271,
    1820.512003430461,
    1936.0977979045833,
    2726.115832055202,
    2726.115832055202,
    1757.9695382105863,
    1896.1403100199598,
    -43.063862205265615,
    65.95349431228902,
    3092.3938668403657,
    -32.227636119006505,
    93.01318621785269,
    93.01318621785269,
    63.45795541152461,
    62.65201137138325,
    93.0295182380317,
    93.0295182380317,
    63.50171043523531,
    62.657856602151625,
    0.8862876254180603,
    0.8862876254180603,
    0.13282849185132156,
    0.5161649944258639,
    12.04912883354833,
    12.04912883354833,
    1.4870144185881344,
    5.795099337840454,
    -6694.200245551715,
    6694.200245551715,
    454933.63244294754,
    -3174.5337785407464,
    -44.250336283788826,
    69.07976772191593,
    3355.1001076136017,
    -33.36198561617137,
    93.02396755117007,
    93.02396755117007,
    63.49385674972958,
    62.661170757672274,
    93.01652870818674,
    93.01652870818674,
    63.50211940186551,
    62.65101430160114,
    0.919732441471572,
    0.919732441471572,
    0.14249281327949348,
    0.5306577480490523,
    12.337249330267053,
    12.337249330267053,
    1.5515980807869476,
    5.982031152410852,
    -6907.441584517007,
    6907.441584517007,
    484018.986850159,
    -3279.1941871603517,
];
/// Pre-#64 (c038fd46) `center_draws = false` run.
const PCN_RELEASE_RAW_PIN: &[f64] = &[
    92.93831497874487,
    92.93831497874487,
    63.37887840215459,
    62.6034767618598,
    92.95626840963558,
    92.95626840963558,
    63.41852877640891,
    62.6217105258666,
    91.37855076046743,
    91.37855076046743,
    61.861885266163846,
    61.8382922394977,
    94.50044240638326,
    94.50044240638326,
    64.84231684143671,
    63.33256170549353,
    50.01592926532176,
    50.400061219507364,
    30.993523990302776,
    23.888236292481043,
    239.01852315702115,
    8402.593412895394,
    17007.876559575903,
    -680.5559798621681,
    1.1950926157850774,
    41.82548740705622,
    84.7359880090067,
    -3.3865994475354624,
    -3.825357436789856,
    41.66977509057972,
    84.3811536269616,
    -5.668974204911341,
    6.203634197306834,
    42.25571046892533,
    85.29111405814002,
    -1.0283807859036944,
    7.125830165785702,
    7.458489619530604,
    4.094145727801826,
    3.3282054940110726,
    3.315647248466703,
    6.893871496181865,
    2.8787587586620496,
    0.7533185449309496,
    10.698828558840528,
    10.709309830052055,
    6.377873105747441,
    5.720104091184199,
    0.9030100334448161,
    0.9030100334448161,
    0.13672106575989085,
    0.5234113712374582,
    8.886776079868001,
    8.886776079868001,
    13.30849049392379,
    5.15028959587724,
    0.043374932185970626,
    0.043374932185970626,
    0.0003234712492976655,
    0.026032816497298106,
    24.386378163815397,
    24.386378163815397,
    3.0386124993750836,
    12.05718991025918,
    3.6197405468131434,
    3.6197405468131434,
    4.333629832965608,
    3.0424916948491934,
    3490.194189168834,
    3490.194189168834,
    2520.062553308556,
    2276.061413721438,
    3343.610188416327,
    3343.610188416327,
    2344.337958419272,
    2198.901500573936,
    3173.3624098993773,
    3173.3624098993773,
    2191.403609257046,
    2127.7216144319796,
    3027.741991941185,
    3027.741991941185,
    2048.68632886246,
    2055.9681394134905,
    2830.094094023689,
    2830.094094023689,
    1817.2421534731807,
    1934.2724562723722,
    2722.9136209560756,
    2722.9136209560756,
    1754.620228136961,
    1894.140651155389,
    -43.063862205265615,
    65.95349431228902,
    3092.3938668403657,
    -32.227636119006505,
    92.88804064740438,
    92.88804064740438,
    63.322990818574304,
    62.57911420318285,
    92.949716793946,
    92.949716793946,
    63.58195906452482,
    62.69640573929895,
    1.7467420934605487,
    41.888107654405204,
    86.66742075146989,
    -2.863502492071629,
    0.9138768563220068,
    41.60621135309167,
    84.5751483448527,
    -3.3546293528037916,
    0.8862876254180603,
    0.8862876254180603,
    0.13282849185132156,
    0.5161649944258639,
    12.04912883354833,
    12.04912883354833,
    1.4870144185881344,
    5.795099337840454,
    -6694.200245551715,
    6694.200245551715,
    454933.63244294754,
    -3174.5337785407464,
    -44.250336283788826,
    69.07976772191593,
    3355.1001076136017,
    -33.36198561617137,
    92.98431282768675,
    92.98431282768675,
    63.42956318600125,
    62.62438900781434,
    92.94232893819971,
    92.94232893819971,
    63.38622873007385,
    62.591939815011074,
    0.41433282173408215,
    42.70719530120588,
    87.09107282122979,
    -3.8196712087495674,
    1.4763083752481543,
    42.133184912745286,
    85.13959373026074,
    -3.4185695422671327,
    0.919732441471572,
    0.919732441471572,
    0.14249281327949348,
    0.5306577480490523,
    12.337249330267053,
    12.337249330267053,
    1.5515980807869476,
    5.982031152410852,
    -6907.441584517007,
    6907.441584517007,
    484018.986850159,
    -3279.1941871603517,
];
/// Default run after the #64 template-centring fix.
const PCN_RESULT_PIN: &[f64] = &[
    93.02050455510377,
    93.02050455510377,
    63.478383989797564,
    62.658249282757495,
    93.05035653160375,
    93.05035653160375,
    63.54427668478698,
    62.68255696342813,
    91.53775966023242,
    91.53775966023242,
    62.05925360570497,
    61.93723804546742,
    94.46704366335942,
    94.46704366335942,
    64.82986813370763,
    63.333122860297046,
    49.81259783875675,
    50.19514440624196,
    30.784162485971173,
    23.813919071159,
    230.71725344439812,
    8380.853763682913,
    17009.10742060286,
    -684.6155441374651,
    1.1535862672220034,
    41.71092435397206,
    84.74302182380953,
    -3.407341172886311,
    -4.465418536994654,
    41.41042409921159,
    84.12804084627113,
    -5.95306904784355,
    6.249171070236719,
    42.451900717582475,
    85.9067389207816,
    -0.980141338700816,
    7.087503966613842,
    7.4154888400700845,
    4.059949397852577,
    3.314861456982183,
    3.215877929352223,
    6.792442364128994,
    2.8124230339294605,
    0.7147089317642045,
    10.697703079335213,
    10.7080023876087,
    6.392205433659035,
    5.71175670653095,
    0.9030100334448161,
    0.9030100334448161,
    0.13672106575989085,
    0.5234113712374582,
    8.969476882640045,
    8.969476882640045,
    13.824037701991099,
    5.072821900515067,
    0.04181820269125868,
    0.04181820269125868,
    0.0002998917025309802,
    0.025510647454415725,
    24.386378163815397,
    24.386378163815397,
    3.0386124993750836,
    12.05718991025918,
    3.6197405468131434,
    3.6197405468131434,
    4.333629832965608,
    3.0424916948491934,
    3492.1527947841123,
    3492.1527947841123,
    2523.349851272,
    2277.48056150494,
    3345.817634750447,
    3345.817634750447,
    2347.7051328480898,
    2200.432018800899,
    3175.8081901911646,
    3175.8081901911646,
    2194.8806473603872,
    2129.3954257559353,
    3030.7952609155245,
    3030.7952609155245,
    2052.487871368722,
    2057.938617354927,
    2833.002753939271,
    2833.002753939271,
    1820.512003430461,
    1936.0977979045833,
    2726.115832055202,
    2726.115832055202,
    1757.9695382105863,
    1896.1403100199598,
    -43.063862205265615,
    65.95349431228902,
    3092.3938668403657,
    -32.227636119006505,
    93.01318621785269,
    93.01318621785269,
    63.45795541152461,
    62.65201137138325,
    93.0295182380317,
    93.0295182380317,
    63.50171043523531,
    62.657856602151625,
    1.98152963551241,
    42.072670130607015,
    86.64840294219768,
    -2.7922528803351705,
    0.9377446681207161,
    41.486556982963506,
    84.52353708163648,
    -3.3222294105010484,
    0.8862876254180603,
    0.8862876254180603,
    0.13282849185132156,
    0.5161649944258639,
    12.04912883354833,
    12.04912883354833,
    1.4870144185881344,
    5.795099337840454,
    -6694.200245551715,
    6694.200245551715,
    454933.63244294754,
    -3174.5337785407464,
    -44.250336283788826,
    69.07976772191593,
    3355.1001076136017,
    -33.36198561617137,
    93.02396755117007,
    93.02396755117007,
    63.49385674972958,
    62.661170757672274,
    93.01652870818674,
    93.01652870818674,
    63.50211940186551,
    62.65101430160114,
    0.1339648591480715,
    42.53991667980759,
    87.0950854221543,
    -3.9312343128044236,
    1.3694278663232793,
    42.02557293182126,
    85.12751295844015,
    -3.4924529352715767,
    0.919732441471572,
    0.919732441471572,
    0.14249281327949348,
    0.5306577480490523,
    12.337249330267053,
    12.337249330267053,
    1.5515980807869476,
    5.982031152410852,
    -6907.441584517007,
    6907.441584517007,
    484018.986850159,
    -3279.1941871603517,
];

// ─── infHMC sampler ─────────────────────────────────────────────────────────

fn hmc_cfg(seed: u64) -> BayesianGroupConfig {
    let mut c = small_cfg(seed);
    c.sampler = BayesianSampler::InfHmc;
    c
}

#[test]
fn sse_gradient_matches_finite_differences() {
    let (data, t, _) = synth(31, 0.01, 1);
    let cfg = small_cfg(1);
    let prob = Problem::new(&data, &t, &cfg).unwrap();
    let template = prob.template_from_coef(&prob.template_coef(&prob.q[2]));
    let sse_at = |i: usize, c: &[f64]| -> f64 {
        sse(
            &template,
            &prob.warp_from_coef(i, c.to_vec()).unwrap().warped,
        )
    };
    // c = 0 itself is skipped: there γ = id puts every γ_j exactly on a knot of the
    // piecewise-linear q_i, where the SSE has a kink (one-sided slopes are used).
    let points: [[f64; 6]; 4] = [
        [0.002, -0.001, 0.0015, 0.0, -0.0005, 0.001],
        [0.3, -0.2, 0.1, 0.05, 0.0, -0.1],
        [-0.15, 0.25, -0.08, 0.12, 0.04, 0.03],
        [0.05, 0.4, 0.0, -0.2, 0.1, 0.0],
    ];
    let h = 1e-6;
    for i in [0, 3, 5] {
        for c in &points {
            let g = prob.sse_gradient(i, c, &template).unwrap();
            let gmax = g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            for k in 0..6 {
                let (mut cp, mut cm) = (c.to_vec(), c.to_vec());
                cp[k] += h;
                cm[k] -= h;
                let fd = (sse_at(i, &cp) - sse_at(i, &cm)) / (2.0 * h);
                assert!(
                    (g[k] - fd).abs() <= 1e-6 * gmax.max(1e-3),
                    "curve {i} c={c:?} k={k}: analytic {} vs FD {fd}",
                    g[k]
                );
            }
        }
    }
    // Outside the support (ψ < 0) there is no gradient.
    assert!(prob
        .sse_gradient(0, &[2.5, 0.0, 0.0, 0.0, 0.0, 0.0], &template)
        .is_none());
}

#[test]
fn dual_averaging_matches_fdasrsf_recursion() {
    // fdasrsf 2.7.5 `bayesian_functions.dual_avg_step` with eps0 = 0.1, target 0.65 and
    // acceptance probabilities 0.9, 0.2, 0.5 (numpy, printed with repr()).
    let mut da = hmc::DualAveraging::new(0.1);
    let got: Vec<f64> = [0.9, 0.2, 0.5]
        .iter()
        .map(|&a| da.update(a, 0.65))
        .collect();
    println!("{got:?} {}", da.averaged());
    let want = DUAL_AVG_REF;
    for (g, w) in got.iter().zip(&want[..3]) {
        assert!((g - w).abs() < 1e-12, "{g} vs {w}");
    }
    assert!((da.averaged() - want[3]).abs() < 1e-12);
}

const DUAL_AVG_REF: [f64; 4] = [
    std::f64::consts::FRAC_PI_2, // printed 1.5707963267948966 (the π/2 cap)
    0.6241250557782609,
    0.3935123830840553,
    0.6289411107167289,
];

/// Posterior mean / variance of the 2 warp coefficients of curve `i` (template and σ²
/// fixed, likelihood tempered by `beta`) by tensor-product quadrature on `centre ± w`,
/// restricted to valid warps.
fn quadrature_moments(
    prob: &Problem,
    state: &ChainState,
    (i, beta): (usize, f64),
    centre: [f64; 2],
    w: [f64; 2],
    k: usize,
) -> [f64; 4] {
    let var = &prob.warp_var;
    let mut logs = Vec::with_capacity(k * k);
    for a in 0..k {
        for b in 0..k {
            let c = [
                centre[0] - w[0] + 2.0 * w[0] * a as f64 / (k - 1) as f64,
                centre[1] - w[1] + 2.0 * w[1] * b as f64 / (k - 1) as f64,
            ];
            let lp = prob.warp_from_coef(i, c.to_vec()).map(|wp| {
                -sse(&state.template, &wp.warped) * beta / (2.0 * state.sigma2)
                    - 0.5 * (c[0] * c[0] / var[0] + c[1] * c[1] / var[1])
            });
            logs.push((c, lp.unwrap_or(f64::NEG_INFINITY)));
        }
    }
    let mx = logs.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let (mut z, mut m0, mut m1, mut s0, mut s1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (c, lp) in &logs {
        let p = (lp - mx).exp();
        z += p;
        m0 += p * c[0];
        m1 += p * c[1];
        s0 += p * c[0] * c[0];
        s1 += p * c[1] * c[1];
    }
    let (m0, m1) = (m0 / z, m1 / z);
    [m0, m1, s0 / z - m0 * m0, s1 / z - m1 * m1]
}

#[test]
fn hmc_kernel_leaves_warp_conditional_invariant() {
    // Curve 0, one harmonic, template and σ² held fixed: the infHMC chain of the two
    // warp coefficients must reproduce the exact conditional posterior moments, computed
    // independently by 2-D quadrature.
    // Measured (seed 77): means 0.24544 / −0.12646 vs 0.24556 / −0.12651, variances
    // 1.475e-4 / 4.457e-4 vs 1.425e-4 / 4.380e-4, ESS ≈ 9600 / 10600 of 20000, acc 0.68.
    check_hmc_conditional(1.0);
}

#[test]
fn tempered_hmc_kernel_leaves_tempered_conditional_invariant() {
    // Parallel tempering (GH #69): a replica at inverse temperature β must leave
    // `exp(−β·SSE/2σ²)·prior` invariant; checked against quadrature of that density.
    check_hmc_conditional(0.3);
}

/// infHMC on one curve's warp (fixed template / σ²) at inverse temperature `beta` vs
/// 2-D quadrature of the tempered conditional: means within 4 SE, variances within
/// 4 relative SE, acceptance near the target.
fn check_hmc_conditional(beta: f64) {
    let (data, t, _) = synth_in_model(31, 0.02, 100);
    let mut cfg = hmc_cfg(1);
    cfg.n_warp_harmonics = 1;
    let prob = Problem::new(&data, &t, &cfg).unwrap();
    let mut state = ChainState::init(&prob, true).unwrap();
    state.sigma2 = 0.5;
    // Coarse pass to locate the posterior, then a fine grid over ±10 posterior sd.
    let coarse = quadrature_moments(&prob, &state, (0, beta), [0.0, 0.0], [1.2, 1.2], 401);
    let exact = quadrature_moments(
        &prob,
        &state,
        (0, beta),
        [coarse[0], coarse[1]],
        [10.0 * coarse[2].sqrt(), 10.0 * coarse[3].sqrt()],
        401,
    );
    println!("quadrature {coarse:?} {exact:?}");
    let mut rng = StdRng::seed_from_u64(77);
    let mut da = hmc::DualAveraging::new(0.1);
    let mut eps = 0.1;
    let (mut x0, mut x1) = (Vec::new(), Vec::new());
    let mut acc = 0usize;
    let n_adapt = 1000;
    let n_keep = 20000;
    for it in 0..n_adapt + n_keep {
        let out = hmc::hmc_transition(&prob, &state, 0, (eps, beta), 10, &mut rng);
        if let Some((w, s)) = out.accepted {
            state.warps[0] = w;
            state.sse[0] = s;
            if it >= n_adapt {
                acc += 1;
            }
        }
        if it < n_adapt {
            eps = da.update(out.alpha, 0.65);
            if it + 1 == n_adapt {
                eps = da.averaged();
            }
        } else {
            x0.push(state.warps[0].coef[0]);
            x1.push(state.warps[0].coef[1]);
        }
    }
    let stats = |x: &[f64]| {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        let v = x.iter().map(|y| (y - m).powi(2)).sum::<f64>() / n;
        (m, v, geyer_ess(x))
    };
    let (ma, va, ea) = stats(&x0);
    let (mb, vb, eb) = stats(&x1);
    let acc = acc as f64 / n_keep as f64;
    println!("β {beta}: eps {eps} acc {acc} chain ({ma}, {mb}, {va}, {vb}) ess ({ea}, {eb})");
    for (m, v, e, mx, vx) in [
        (ma, va, ea, exact[0], exact[2]),
        (mb, vb, eb, exact[1], exact[3]),
    ] {
        let se = (vx / e).sqrt();
        assert!(
            (m - mx).abs() < 4.0 * se,
            "mean {m} vs exact {mx} (SE {se})"
        );
        let rel = (2.0 / e).sqrt();
        assert!(
            (v / vx - 1.0).abs() < 4.0 * rel,
            "variance {v} vs exact {vx}"
        );
    }
    assert!(acc > 0.5 && acc < 0.85, "acceptance {acc} (target 0.65)");
}

/// Pooled per-curve warp areas `∫(γ − t)` (unit domain) of a result's draws, per chain.
fn area_traces(res: &BayesianGroupResult, t: &[f64], n_chains: usize) -> Vec<Vec<Vec<f64>>> {
    let n = res.gamma_draws.len();
    let nd = res.gamma_draws[0].nrows() / n_chains;
    (0..n)
        .map(|i| {
            (0..n_chains)
                .map(|c| {
                    (0..nd)
                        .map(|k| {
                            let row = res.gamma_draws[i].row(c * nd + k);
                            let dev: Vec<f64> = row.iter().zip(t).map(|(g, x)| g - x).collect();
                            trapz(&dev, t)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Batch-means standard errors (20 batches per chain) of the mean and of the variance
/// (batch means of `(x − m)²`) — robust to slowly mixing components that a truncated
/// autocorrelation sum can miss.
fn batch_means_se(chains: &[Vec<f64>], m: f64) -> [f64; 2] {
    let mut bm: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for c in chains {
        let len = c.len() / 20;
        for b in c.chunks_exact(len).take(20) {
            bm[0].push(b.iter().sum::<f64>() / len as f64);
            bm[1].push(b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / len as f64);
        }
    }
    bm.map(|v| {
        let k = v.len() as f64;
        let mu = v.iter().sum::<f64>() / k;
        (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (k - 1.0) / k).sqrt()
    })
}

fn pooled_mean_var(chains: &[Vec<f64>]) -> (f64, f64) {
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    let n = all.len() as f64;
    let m = all.iter().sum::<f64>() / n;
    (m, all.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n)
}

fn tiny_problem_cfg(
    sampler: BayesianSampler,
    n_iter: usize,
    thin: usize,
    seed: u64,
) -> BayesianGroupConfig {
    let mut c = small_cfg(seed);
    c.sampler = sampler;
    c.n_warp_harmonics = 1;
    c.n_template_basis = 8;
    c.n_iter = n_iter;
    c.burn_in = n_iter / 5;
    c.thin = thin;
    c.n_chains = 4;
    c.center_every = 0;
    c
}

/// First `nc` curves of `synth_in_model(21, 0.03, 8)`.
fn tiny_data(nc: usize) -> (FdMatrix, Vec<f64>) {
    let (full, t, _) = synth_in_model(21, 0.03, 8);
    let mut data = FdMatrix::zeros(nc, 21);
    for i in 0..nc {
        for j in 0..21 {
            data[(i, j)] = full[(i, j)];
        }
    }
    (data, t)
}

/// Mixing summary of a run: ESS per retained draw of σ², log-likelihood and the warp
/// areas (min over curves), and the worst R̂.
fn mixing(res: &BayesianGroupResult) -> ([f64; 3], f64) {
    let nd = res.sigma2_draws.len() as f64;
    let ess_w = res.ess_warps.iter().copied().fold(f64::INFINITY, f64::min);
    let rhat = res
        .rhat_warps
        .iter()
        .copied()
        .chain([res.rhat_sigma2, res.rhat_log_likelihood])
        .fold(0.0, f64::max);
    (
        [res.ess_sigma2 / nd, res.ess_log_likelihood / nd, ess_w / nd],
        rhat,
    )
}

#[test]
fn infhmc_mixes_better_than_pcn_at_equal_iterations() {
    // Two curves, one harmonic, no centring (exact posterior target), 4 chains, the same
    // number of iterations for both samplers. Measured (release, 2026-11-26):
    //   3000 it: infHMC ESS/draw σ² 0.46, loglik 0.32, warps 0.42, max R̂ 1.003, acc 0.67–0.70;
    //            pCN    ESS/draw σ² 0.02, loglik 0.004, warps 0.004, max R̂ ≈ 2.5.
    let (data, t) = tiny_data(2);
    let pcn = bayesian_align_group(
        &data,
        &t,
        &tiny_problem_cfg(BayesianSampler::Pcn, 3000, 1, 3),
    )
    .unwrap();
    let hmc = bayesian_align_group(
        &data,
        &t,
        &tiny_problem_cfg(BayesianSampler::InfHmc, 3000, 1, 3),
    )
    .unwrap();
    let (ep, rp) = mixing(&pcn);
    let (eh, rh) = mixing(&hmc);
    println!(
        "pCN ESS/draw {ep:?} R̂ {rp}; infHMC ESS/draw {eh:?} R̂ {rh}; acc {:?}",
        hmc.warp_acceptance
    );
    assert!(rh < 1.05, "infHMC max R̂ {rh}");
    assert!(rp > rh, "pCN R̂ {rp} should exceed infHMC's {rh}");
    for k in 0..3 {
        assert!(eh[k] > 0.1, "infHMC ESS/draw {eh:?}");
        assert!(eh[k] > 5.0 * ep[k], "infHMC ESS/draw {eh:?} vs pCN {ep:?}");
    }
    for &a in &hmc.warp_acceptance {
        assert!(a > 0.5 && a < 0.85, "infHMC acceptance {a} (target 0.65)");
    }
    assert_eq!(
        hmc.template_acceptance, 1.0,
        "Gibbs template draw always moves"
    );
    for c in &hmc.chains {
        assert_eq!(c.warp_step_sizes.len(), 2);
        assert!(c
            .warp_step_sizes
            .iter()
            .all(|&e| e > 0.0 && e <= std::f64::consts::FRAC_PI_2));
    }
    assert!(pcn.chains[0].warp_step_sizes.is_empty());
    // Well-mixed, unimodal posterior: no curve flagged.
    assert!(
        hmc.multimodal_curves.iter().all(|&m| !m),
        "{:?}",
        hmc.warp_bimodality
    );
}

#[test]
#[ignore = "long invariance study: infHMC vs 1e6-iteration pCN on the same posterior (~1 min debug)"]
fn infhmc_and_long_pcn_agree_on_posterior_moments() {
    // Two curves, one harmonic, no centring: both kernels target the same posterior.
    // pCN needs ~1e6 iterations to converge here (R̂ ≈ 1.000); infHMC 5e4.
    // Measured (release, 2026-11-26, 4 chains each):
    //   σ² mean/var  infHMC 0.25355 / 0.00487   pCN 0.25266 / 0.00476
    //   warp areas   infHMC  0.08135 / −0.09326 pCN  0.08131 / −0.09314
    let (data, t) = tiny_data(2);
    let hmc = bayesian_align_group(
        &data,
        &t,
        &tiny_problem_cfg(BayesianSampler::InfHmc, 50_000, 1, 3),
    )
    .unwrap();
    let mut pcn_cfg = tiny_problem_cfg(BayesianSampler::Pcn, 1_000_000, 20, 3);
    pcn_cfg.injective_warps = true;
    let pcn = bayesian_align_group(&data, &t, &pcn_cfg).unwrap();
    assert!(mixing(&pcn).1 < 1.02 && mixing(&hmc).1 < 1.02);
    let sig = |r: &BayesianGroupResult| -> Vec<Vec<f64>> {
        r.chains.iter().map(|c| c.sigma2_trace.clone()).collect()
    };
    let mut pairs: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = vec![(sig(&hmc), sig(&pcn))];
    let (ah, ap) = (area_traces(&hmc, &t, 4), area_traces(&pcn, &t, 4));
    pairs.extend(ah.into_iter().zip(ap));
    for (h, p) in &pairs {
        let ((mh, vh), (mp, vp)) = (pooled_mean_var(h), pooled_mean_var(p));
        let (sh, sp) = (batch_means_se(h, mh), batch_means_se(p, mp));
        let se = (sh[0] * sh[0] + sp[0] * sp[0]).sqrt();
        let se_v = (sh[1] * sh[1] + sp[1] * sp[1]).sqrt();
        println!("mean {mh} vs {mp} (SE {se}), var {vh} vs {vp} (SE {se_v})");
        assert!((mh - mp).abs() < 4.0 * se, "means {mh} vs {mp} (SE {se})");
        assert!(
            (vh - vp).abs() < 4.0 * se_v,
            "variances {vh} vs {vp} (SE {se_v})"
        );
    }
}

#[test]
fn infhmc_is_deterministic_and_chains_use_seed_plus_k() {
    let (data, t, _) = synth(31, 0.01, 2);
    let mut cfg = hmc_cfg(11);
    cfg.n_iter = 200;
    cfg.burn_in = 100;
    cfg.n_chains = 3;
    let a = bayesian_align_group(&data, &t, &cfg).unwrap();
    let b = bayesian_align_group(&data, &t, &cfg).unwrap();
    assert_eq!(a, b, "identical config must give bit-identical results");
    let mut single = cfg.clone();
    single.n_chains = 1;
    single.seed = 13;
    let s = bayesian_align_group(&data, &t, &single).unwrap();
    assert_eq!(a.chains[2], s.chains[0]);
    assert_ne!(a.chains[0].sigma2_trace, a.chains[1].sigma2_trace);
    // The sampler choice matters: pCN on the same seed differs.
    let mut p = cfg.clone();
    p.sampler = BayesianSampler::Pcn;
    assert_ne!(
        bayesian_align_group(&data, &t, &p).unwrap().sigma2_draws,
        a.sigma2_draws
    );
}

#[test]
fn infhmc_recovers_known_warps() {
    let (data, t, truth) = synth(41, 0.0, 3);
    let mut cfg = hmc_cfg(5);
    cfg.n_iter = 1500;
    cfg.burn_in = 750;
    cfg.n_chains = 2;
    let res = bayesian_align_group(&data, &t, &cfg).unwrap();
    let err = max_abs_diff(&res.gammas, &truth);
    println!("infHMC warp error {err}, acc {:?}", res.warp_acceptance);
    assert!(
        err < 0.05,
        "posterior mean warp error {err} (identity baseline > 0.1)"
    );
    for i in 0..truth.nrows() {
        assert!(is_warp(&res.gammas.row(i), 0.0, 1.0));
    }
}

#[test]
fn infhmc_rejects_invalid_hmc_settings() {
    let (data, t, _) = synth(21, 0.0, 1);
    type Mutator = Box<dyn Fn(&mut BayesianGroupConfig)>;
    let cases: Vec<Mutator> = vec![
        Box::new(|c| c.hmc_step_size = 0.0),
        Box::new(|c| c.hmc_step_size = 2.0),
        Box::new(|c| c.hmc_max_leapfrog = 0),
        Box::new(|c| c.hmc_target_accept = 1.0),
    ];
    for (k, f) in cases.iter().enumerate() {
        let mut c = hmc_cfg(1);
        f(&mut c);
        assert!(
            matches!(
                bayesian_align_group(&data, &t, &c),
                Err(FdarError::InvalidParameter { .. })
            ),
            "case {k} should be rejected"
        );
        // The same settings are ignored by the pCN sampler.
        c.sampler = BayesianSampler::Pcn;
        c.n_iter = 20;
        c.burn_in = 10;
        assert!(bayesian_align_group(&data, &t, &c).is_ok(), "pCN case {k}");
    }
    // pCN mixture settings are ignored by infHMC.
    let mut c = hmc_cfg(1);
    c.n_iter = 20;
    c.burn_in = 10;
    c.pcn_betas = vec![];
    assert!(bayesian_align_group(&data, &t, &c).is_ok());
}

#[test]
fn geyer_ess_matches_ar1_theory() {
    // AR(1) with ρ: ESS = n (1 − ρ)/(1 + ρ). iid: ESS ≈ n.
    let mut rng = StdRng::seed_from_u64(5);
    let n = 40_000;
    for rho in [0.0f64, 0.5, 0.9] {
        let mut x = vec![0.0; n];
        for k in 1..n {
            let e: f64 = rng.sample(StandardNormal);
            x[k] = rho * x[k - 1] + (1.0 - rho * rho).sqrt() * e;
        }
        let ess = geyer_ess(&x);
        let want = n as f64 * (1.0 - rho) / (1.0 + rho);
        assert!(
            (ess / want - 1.0).abs() < 0.15,
            "ρ={rho}: ESS {ess} vs {want}"
        );
    }
    assert!(geyer_ess(&[1.0; 10]).is_nan());
    assert!(geyer_ess(&[1.0, 2.0, 3.0]).is_nan());
    let chains = vec![vec![0.1, -0.3, 0.2, 0.5, -0.1], vec![1.0; 5]];
    assert_eq!(multi_chain_ess(&chains), geyer_ess(&chains[0]));
}

#[test]
fn bimodality_coefficient_matches_scipy() {
    // scipy 1.18.1: (skew(x, bias=False)² + 1) /
    //   (kurtosis(x, bias=False) + 3 (n−1)² / ((n−2)(n−3))).
    let a = [0.3, -1.2, 0.8, 0.1, -0.4, 1.1, 0.2, 2.5, -0.7];
    let b = [0.0, 0.1, 0.2, 0.15, 5.0, 5.1, 4.9, 5.2];
    assert!((bimodality_coefficient(&a) - 0.2904365212080462).abs() < 1e-12);
    assert!((bimodality_coefficient(&b) - 0.47339751548510983).abs() < 1e-12);
    assert!(bimodality_coefficient(&[1.0; 6]).is_nan());
    assert!(bimodality_coefficient(&[1.0, 2.0, 3.0]).is_nan());
    // Large samples: Gaussian → 1/3, equal mixture N(±3, 1) → 1/(3 − 1.62) ≈ 0.725.
    let mut rng = StdRng::seed_from_u64(9);
    let g: Vec<f64> = (0..50_000)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();
    let mix: Vec<f64> = g
        .iter()
        .enumerate()
        .map(|(k, v)| v + if k % 2 == 0 { 3.0 } else { -3.0 })
        .collect();
    let (bg, bm) = (bimodality_coefficient(&g), bimodality_coefficient(&mix));
    assert!((bg - 1.0 / 3.0).abs() < 0.02, "Gaussian BC {bg}");
    assert!(
        (bm - 1.0 / 1.38).abs() < 0.02 && bm > 5.0 / 9.0,
        "mixture BC {bm}"
    );
}

#[test]
fn stuck_chain_in_poor_mode_is_flagged() {
    // Five curves, no centring: one of four infHMC chains (seed 3 + 3) settles in a poor
    // local mode (σ² ≈ 0.6 vs 0.22 for the others; measured 2026-11-26). infHMC moves
    // locally and cannot leave it, which the multi-chain diagnostics must expose.
    let (data, t) = tiny_data(5);
    let res = bayesian_align_group(
        &data,
        &t,
        &tiny_problem_cfg(BayesianSampler::InfHmc, 3000, 1, 3),
    )
    .unwrap();
    let means: Vec<f64> = res
        .chains
        .iter()
        .map(|c| c.sigma2_trace.iter().sum::<f64>() / c.sigma2_trace.len() as f64)
        .collect();
    println!(
        "chain σ² means {means:?}; BC {:?}; R̂ σ² {}",
        res.warp_bimodality, res.rhat_sigma2
    );
    let (lo, hi) = means
        .iter()
        .fold((f64::INFINITY, 0.0f64), |(a, b), &m| (a.min(m), b.max(m)));
    if hi > 2.0 * lo {
        assert!(res.rhat_sigma2 > 1.5, "R̂ must flag disagreeing chains");
        assert!(
            res.multimodal_curves.iter().any(|&m| m),
            "BC {:?}",
            res.warp_bimodality
        );
        assert!(res.mode_spread.iter().copied().fold(0.0, f64::max) > 0.01);
    } else {
        assert!(res.rhat_sigma2 < 1.1);
    }
}

#[test]
fn shooting_map_is_periodic_unless_injective() {
    // exp_1(v) depends on ‖v‖ through cos/sin, so v and v' = −v (2π − r)/r give the same
    // warp: the un-truncated (pCN default) coefficient posterior has a second shell. With
    // `injective_warps` (always on for infHMC) only the ‖v‖ < π preimage is admissible.
    let (data, t, _) = synth(31, 0.0, 1);
    let mut cfg = small_cfg(1);
    cfg.n_warp_harmonics = 1;
    let prob = Problem::new(&data, &t, &cfg).unwrap();
    let c = vec![0.12, -0.05];
    let r = (c[0] * c[0] + c[1] * c[1] as f64).sqrt();
    let k = -(2.0 * std::f64::consts::PI - r) / r;
    let anti: Vec<f64> = c.iter().map(|x| k * x).collect();
    let (w, wa) = (
        prob.warp_from_coef(0, c.clone()).unwrap(),
        prob.warp_from_coef(0, anti.clone()).unwrap(),
    );
    // Trapezoidal ‖v‖ equals the coefficient norm only up to quadrature error.
    assert!(max_abs_diff_vec(&w.gamma, &wa.gamma) < 1e-3);
    cfg.injective_warps = true;
    let inj = Problem::new(&data, &t, &cfg).unwrap();
    assert!(inj.warp_from_coef(0, anti).is_none());
    assert_eq!(inj.warp_from_coef(0, c).unwrap().gamma, w.gamma);
    cfg.injective_warps = false;
    cfg.sampler = BayesianSampler::InfHmc;
    assert!(Problem::new(&data, &t, &cfg).unwrap().injective);
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// [`result_hash`] plus the infHMC-specific and diagnostic fields.
fn hmc_result_hash(r: &BayesianGroupResult) -> ValuePin {
    let mut h = result_hash(r);
    for v in [&r.ess_warps, &r.warp_bimodality] {
        fnv(&mut h, v);
    }
    fnv(&mut h, &[r.ess_sigma2, r.ess_log_likelihood]);
    for c in &r.chains {
        fnv(&mut h, &c.warp_step_sizes);
    }
    h
}

#[test]
fn default_infhmc_sampler_is_bit_identical_to_pre_mode_jump_release() {
    // Pinned on 80fb0916 (before the GH #69 mode-jumping options were added): with the
    // new options at their defaults infHMC must keep producing identical results.
    let (data, t, _) = synth(31, 0.01, 2);
    let mut cfg = hmc_cfg(11);
    cfg.n_iter = 300;
    cfg.burn_in = 100;
    cfg.thin = 2;
    cfg.n_chains = 2;
    let full = hmc_result_hash(&bayesian_align_group(&data, &t, &cfg).unwrap());
    let (d5, t5) = tiny_data(5);
    let tiny = hmc_result_hash(
        &bayesian_align_group(
            &d5,
            &t5,
            &tiny_problem_cfg(BayesianSampler::InfHmc, 600, 1, 3),
        )
        .unwrap(),
    );
    // the output structure is portable
    assert_eq!(full.0.len(), INFHMC_RESULT_PIN.len());
    assert_eq!(tiny.0.len(), INFHMC_TINY_PIN.len());
    // HMC trajectories are chaotic: a one-ulp libm difference flips accept/reject
    // decisions (measured O(1) deviation with glibc's SSE2 instead of FMA kernels), so
    // the values are pinned only where libm matches the recording machine.
    let canary = libm_canary();
    if canary == INFHMC_RECORDING_LIBM_CANARY {
        full.assert_close(INFHMC_RESULT_PIN, 1e-10, "default infHMC results");
        tiny.assert_close(INFHMC_TINY_PIN, 1e-10, "default infHMC tiny results");
    } else {
        eprintln!(
            "infHMC value pin not enforced: libm canary {canary:#x} differs from the \
             recording machine's {INFHMC_RECORDING_LIBM_CANARY:#x}"
        );
    }
}

// Replace FNV-1a hashes 0xc01e_025f_f834_51b8 / 0x7d6b_9558_f89a_dbf3 (platform-libm
// dependent); recorded on 925ef514, which reproduced both bit-for-bit on the recording
// machine (glibc FMA kernels, x86-64), compared at rtol 1e-10.
const INFHMC_RECORDING_LIBM_CANARY: u64 = 0x9260_de3e_115e_bc05;
const INFHMC_RESULT_PIN: &[f64] = &[
    93.04027151519216,
    93.04027151519216,
    63.49938054714968,
    62.65417757733375,
    93.06894321655052,
    93.06894321655052,
    63.532383671035696,
    62.66475076804546,
    91.45591913078167,
    91.45591913078167,
    62.01110135887636,
    61.892875422989306,
    94.62630883297476,
    94.62630883297476,
    65.03109230659312,
    63.411738082071594,
    49.633041047729165,
    50.005353821406054,
    30.707468763510594,
    23.69647055117536,
    123.20404173134317,
    8467.171531946988,
    17623.248420701817,
    -784.9648702150497,
    0.6160202086566864,
    42.124168235558756,
    87.8393595451344,
    -3.9152941412113207,
    -4.92139335575075,
    42.23832690297437,
    88.38185770332419,
    -6.746349403114111,
    6.28339119475527,
    43.59074526330364,
    89.24671827999128,
    -1.0679223382692282,
    7.695315465719275,
    7.889974965172444,
    4.4493396970942145,
    3.7266386795483744,
    5.1591146306455125,
    7.096917049473251,
    3.4201719753170337,
    1.9962501441669431,
    10.520917331360952,
    10.52894541704721,
    6.312636142829264,
    5.607988958424754,
    4.530100334448161,
    4.530100334448161,
    3.4213515508775068,
    2.650222965440357,
    6.566939553904755,
    6.566939553904755,
    7.192015794198441,
    3.8452530890738883,
    0.012293684575450919,
    0.012293684575450919,
    3.453930839533349e-5,
    0.0064023152446801455,
    14.941232857724138,
    14.941232857724138,
    1.145731434921033,
    7.49387687621674,
    3.273409479278188,
    3.273409479278188,
    3.1094712049963813,
    2.6243128379891205,
    3528.830990819787,
    3528.830990819787,
    2556.605327507884,
    2292.3167388086376,
    3354.3486745011833,
    3354.3486745011833,
    2360.4929236123876,
    2206.6206389865783,
    3178.3780613291624,
    3178.3780613291624,
    2195.637317359907,
    2129.707764916265,
    3006.501898902037,
    3006.501898902037,
    2018.0593407548176,
    2040.9008785097064,
    2843.6514307605653,
    2843.6514307605653,
    1848.1758334230433,
    1950.3117758684893,
    2698.055810296026,
    2698.055810296026,
    1723.914319083992,
    1878.0794972921592,
    13.243796689807809,
    13.243796689807809,
    123.54703020557686,
    5.495864459871872,
    93.03678818311808,
    93.03678818311808,
    63.494869323051546,
    62.65079158309895,
    93.02540599643734,
    93.02540599643734,
    63.50223009155825,
    62.64213257294972,
    0.755348998435381,
    42.28887548700943,
    88.04758638396326,
    -3.76881876433081,
    0.5675940005241249,
    41.94593525884139,
    87.58746139829299,
    -3.9188698265895496,
    4.5719063545150505,
    4.5719063545150505,
    3.485520296193555,
    2.6711259754738017,
    7.336138541205199,
    7.336138541205199,
    0.5570205152478676,
    3.58369935361707,
    -1954.0909632432335,
    1956.5785566228492,
    49911.7767094306,
    -865.2644243779712,
    11.386621556259499,
    14.613378443740501,
    147.60299000272653,
    3.924414370839666,
    93.04388645519191,
    93.04388645519191,
    63.50318802226835,
    62.65694489482049,
    93.06984340872961,
    93.06984340872961,
    63.524054611306134,
    62.66315408205893,
    0.4874638029763346,
    42.51018256624698,
    88.56698243928976,
    -3.9662889954059577,
    0.664446416789248,
    42.30240121227611,
    88.11976661088907,
    -3.9117184558330904,
    4.48829431438127,
    4.48829431438127,
    3.3632509703470874,
    2.629319955406912,
    7.605094316518944,
    7.605094316518944,
    0.5887109196731652,
    3.798960082297471,
    -2166.9214152319446,
    2166.9214152319446,
    53898.26258821998,
    -1096.964577805573,
    145.33455716154452,
    145.33455716154452,
    3658.159153971297,
    79.5811495419847,
    2.154626066081058,
    2.154626066081058,
    0.7766670559345097,
    1.2894802608917948,
    80.1716716837054,
    80.1716716837054,
    3561.1219101046486,
    53.53924088411416,
    0.004090671344455122,
    0.004090671344455122,
    2.7963170736955053e-6,
    0.002356994075042032,
    0.004448655290133727,
    0.004448655290133727,
    3.306560862821875e-6,
    0.0025593406993125053,
];
const INFHMC_TINY_PIN: &[f64] = &[
    52.437329539775625,
    52.437329539775625,
    36.41940845879786,
    35.583340863374346,
    52.29564926096233,
    52.29564926096233,
    36.47893810531883,
    35.51456147622855,
    45.76488725882399,
    45.76488725882399,
    29.42240684130143,
    31.68545675737839,
    57.947095502264666,
    57.947095502264666,
    43.29410288400346,
    38.68188083009572,
    42.097230695164136,
    42.49505915763897,
    26.74624566857649,
    20.340677976320777,
    -1062.8817046545958,
    46524.53998831663,
    72977.13743369597,
    -9250.882965800267,
    -0.553584221174285,
    21.872968267870203,
    31.218516548324562,
    -4.834263664490398,
    -16.165618483523343,
    29.361340643741467,
    50.28505216794504,
    -14.040273164228926,
    15.276171892908549,
    22.106019447971274,
    32.73763227677796,
    4.5867169364652405,
    7.873615479843984,
    7.873615479843984,
    4.162613796247273,
    4.127192210588281,
    2.3232633802983793,
    4.202044186133332,
    1.4836846684857905,
    0.4289414331594623,
    11.591246867681562,
    11.591246867681562,
    8.233731875024072,
    6.358078826800989,
    3.7491652754590987,
    3.7491652754590987,
    2.812887749755435,
    2.234974958263773,
    31.80292470805023,
    31.80292470805023,
    248.12119178508257,
    18.623389370162872,
    0.5846862707293117,
    0.5846862707293117,
    0.09941755784135455,
    0.32835318615640247,
    633.5653563535924,
    633.5653563535924,
    280.2830575906049,
    393.26059778114126,
    13.521729500555677,
    13.521729500555677,
    73.42662452872207,
    11.817640182175648,
    23371.83237833352,
    23371.83237833352,
    17902.878657441943,
    15410.460221816185,
    18577.530761713,
    18577.530761713,
    12761.077092699283,
    12818.917146722679,
    21587.641849122956,
    21587.641849122956,
    14338.73431076439,
    13777.38383702572,
    16992.28503559283,
    16992.28503559283,
    11847.437064349391,
    12323.887241513887,
    19642.27342813046,
    19642.27342813046,
    13238.24254676294,
    13289.977860229372,
    -58.30462442505824,
    66.30462442505825,
    3891.866224747564,
    -39.53641628337216,
    52.38851082339279,
    52.38851082339279,
    36.606284868787185,
    35.57382899877827,
    52.392913260322096,
    52.392913260322096,
    36.58211680283783,
    35.57721939727726,
    0.28521809685877825,
    25.145676345141545,
    41.31092672554167,
    -4.471865821534418,
    -0.7262070265480001,
    24.583312647697962,
    40.06812355991284,
    -4.983333454361263,
    3.6994991652754585,
    3.6994991652754585,
    2.7394851184918663,
    2.201335559265442,
    108.47600758634438,
    108.47600758634438,
    25.037748343249508,
    54.335341086976555,
    -33457.92255454576,
    33457.92255454576,
    2337353.388907389,
    -16750.33456801183,
    -56.33762175598613,
    66.33762175598613,
    3779.303842680423,
    -38.55841450399075,
    52.387061601573116,
    52.387061601573116,
    36.60422287073409,
    35.573912477988955,
    52.375031811515676,
    52.375031811515676,
    36.56368204443264,
    35.56945454511673,
    -1.1208357943973057,
    25.490692159719345,
    41.82329104251251,
    -5.386834480369859,
    -0.7061223269206007,
    24.51992000461899,
    39.9399238016613,
    -4.931093583555474,
    3.7813021702838063,
    3.7813021702838063,
    2.8618398499446767,
    2.2607679465776296,
    106.81009778539452,
    106.81009778539452,
    24.307319651780045,
    53.50866632725268,
    -33236.528787990705,
    33236.528787990705,
    2306545.066435471,
    -16672.10454193729,
    -56.373737237068006,
    68.37373723706801,
    3916.483096918804,
    -38.915824824712004,
    52.38794657443801,
    52.38794657443801,
    36.60283696257874,
    35.57327500959825,
    52.38308371907015,
    52.38308371907015,
    36.56586205895067,
    35.56898577859319,
    -0.6700611092800002,
    24.5419548874869,
    40.21097349009415,
    -4.768579689558614,
    -0.699428968373077,
    24.57784614290715,
    40.10273353311492,
    -4.95491150697185,
    3.7946577629382308,
    3.7946577629382308,
    2.8812266409513905,
    2.2757929883138566,
    107.0329664683144,
    107.0329664683144,
    24.411506531609216,
    53.52650851831481,
    -33246.18830285091,
    33246.18830285091,
    2308136.171189951,
    -16617.058977841327,
    -109.7322257744446,
    123.7322257744446,
    13663.41253425591,
    -74.82148384962973,
    51.55359266964561,
    51.55359266964561,
    36.138911172473996,
    34.99790197115584,
    51.44634488901095,
    51.44634488901095,
    36.053273645095814,
    34.928742905755186,
    -0.7177787412041705,
    21.01141805811192,
    28.880062900535044,
    -4.812822774792974,
    -0.08257856285541845,
    21.566188298388873,
    29.2718648760741,
    -4.467716113072989,
    3.7212020033388984,
    3.7212020033388984,
    2.7800229096351456,
    2.2020033388981637,
    311.2462845135387,
    311.2462845135387,
    206.52648306396594,
    157.05699092937988,
    -60078.39650101378,
    60078.39650101378,
    7525996.761023368,
    -30133.08587789903,
    5073.263968011139,
    5073.263968011139,
    5249785.169244986,
    3048.309611874954,
    4.567865982979023,
    4.567865982979023,
    4.184638215102538,
    2.7556917651663886,
    1972.8440708686671,
    1972.8440708686671,
    2159244.918243973,
    1316.3893494968318,
    0.011964508583951972,
    0.011964508583951972,
    3.038613605514586e-5,
    0.007512074154156368,
    0.011650227953464154,
    0.011650227953464154,
    2.851900363335686e-5,
    0.007236761496109124,
    0.011480131107419293,
    0.011480131107419293,
    2.792794189669249e-5,
    0.007107511464432125,
    0.018888071905169647,
    0.018888071905169647,
    8.850443745379874e-5,
    0.012414053038290596,
];

// ─── Parallel tempering (GH #69) ────────────────────────────────────────────

fn chain_sigma2_means(res: &BayesianGroupResult) -> Vec<f64> {
    res.chains
        .iter()
        .map(|c| c.sigma2_trace.iter().sum::<f64>() / c.sigma2_trace.len() as f64)
        .collect()
}

fn pt_cfg(seed: u64, k: usize) -> BayesianGroupConfig {
    let mut c = tiny_problem_cfg(BayesianSampler::InfHmc, 3000, 1, seed);
    c.n_temperatures = k;
    c.pt_min_inv_temp = 0.1;
    c
}

/// Runs plain infHMC and infHMC with 6 tempered replicas on the stuck 5-curve problem;
/// asserts plain is stuck and tempering escapes. Returns (plain, tempered, seconds).
fn stuck_vs_tempered(seed: u64) -> (BayesianGroupResult, BayesianGroupResult, [f64; 2]) {
    let (data, t) = tiny_data(5);
    let t0 = std::time::Instant::now();
    let plain = bayesian_align_group(&data, &t, &pt_cfg(seed, 1)).unwrap();
    let t1 = std::time::Instant::now();
    let pt = bayesian_align_group(&data, &t, &pt_cfg(seed, 6)).unwrap();
    let secs = [(t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64()];
    let (mp, rp) = mixing(&plain);
    let (mt, rt) = mixing(&pt);
    println!(
        "seed {seed}: plain σ² {:.3?} R̂σ² {:.3} maxR̂ {rp:.3} ESS/draw {mp:.3?} {:.2}s | \
         PT6 σ² {:.3?} R̂σ² {:.4} maxR̂ {rt:.4} ESS/draw {mt:.3?} swaps {:.2?} {:.2}s",
        chain_sigma2_means(&plain),
        plain.rhat_sigma2,
        secs[0],
        chain_sigma2_means(&pt),
        pt.rhat_sigma2,
        pt.chains[0].swap_acceptance,
        secs[1]
    );
    // Reproduction: one chain sits in the poor mode (σ² ≈ 0.66 vs 0.22) and is flagged.
    let m = chain_sigma2_means(&plain);
    let (lo, hi) = m
        .iter()
        .fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
    assert!(
        hi > 2.0 * lo,
        "plain infHMC not stuck for seed {seed}: {m:?}"
    );
    assert!(plain.rhat_sigma2 > 1.5 && plain.multimodal_curves.iter().any(|&b| b));
    // Escape: every chain in the good mode, diagnostics clean.
    let m = chain_sigma2_means(&pt);
    for &v in &m {
        assert!((v - 0.224).abs() < 0.01, "tempered chain σ² means {m:?}");
    }
    assert!(pt.rhat_sigma2 < 1.01, "R̂(σ²) {}", pt.rhat_sigma2);
    assert!(rt < 1.02, "max R̂ {rt}");
    assert!(
        pt.multimodal_curves.iter().all(|&b| !b),
        "{:?}",
        pt.warp_bimodality
    );
    assert!(
        pt.mode_spread.iter().all(|&s| s < 0.01),
        "{:?}",
        pt.mode_spread
    );
    for c in &pt.chains {
        assert_eq!(c.swap_acceptance.len(), 5);
        // Rates are probabilities; some exchanges happen (a single level can be a
        // bottleneck with rate ≈ 0 in one chain: seed 13, pair 2).
        assert!(c.swap_acceptance.iter().all(|&a| (0.0..=1.0).contains(&a)));
        assert!(c.swap_acceptance.iter().any(|&a| a > 0.1));
        assert_eq!(c.warp_step_sizes.len(), 5);
    }
    assert!(plain.chains[0].swap_acceptance.is_empty());
    (plain, pt, secs)
}

#[test]
fn parallel_tempering_escapes_poor_mode() {
    // GH #69: 5 curves, no centring, 4 chains × 3000 it. Plain infHMC: chain 3 stuck at
    // σ² ≈ 0.66 (others 0.22), R̂(σ²) 3.38, all curves flagged multimodal. With 6 replicas
    // (β_min = 0.1) every chain finds the good mode. Measured (debug, 2026-11-26):
    //   plain σ² [0.224, 0.224, 0.224, 0.656], R̂σ² 3.383, ESS/draw [0.49, 0.22, 0.34], 2.5 s
    //   PT6   σ² [0.223, 0.222, 0.225, 0.224], R̂σ² 1.000, ESS/draw [0.61, 0.36, 0.52], 7.3 s
    //   swap acceptance [0.24, 0.16, 0.03, 0.33, 0.35]
    let (plain, pt, _) = stuck_vs_tempered(3);
    // ESS per draw does not drop (the good mode is sampled as well as by healthy chains).
    let (mp, _) = mixing(&plain);
    let (mt, _) = mixing(&pt);
    for k in 0..3 {
        assert!(mt[k] > 0.5 * mp[k], "PT ESS/draw {mt:?} vs plain {mp:?}");
    }
}

#[test]
#[ignore = "multi-seed stuck-chain study (~40 s debug)"]
fn parallel_tempering_escapes_poor_mode_all_stuck_seeds() {
    // Of seeds 3, 13, 23, 33, 43, 53 plain infHMC is stuck for 3, 13, 33, 53 (measured
    // 2026-11-26); PT6 (β_min 0.1) fixes all four (R̂(σ²) ≤ 1.002). Release run time per
    // seed: plain ≈ 0.4 s, PT6 ≈ 1.1 s.
    for seed in [3, 13, 33, 53] {
        stuck_vs_tempered(seed);
    }
}

/// Pooled moments of σ² and the warp areas of two runs agree within 4 batch-means SE.
fn assert_same_posterior(a: &BayesianGroupResult, b: &BayesianGroupResult, t: &[f64]) {
    let sig = |r: &BayesianGroupResult| -> Vec<Vec<f64>> {
        r.chains.iter().map(|c| c.sigma2_trace.clone()).collect()
    };
    let mut pairs: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = vec![(sig(a), sig(b))];
    let (aa, ab) = (
        area_traces(a, t, a.chains.len()),
        area_traces(b, t, b.chains.len()),
    );
    pairs.extend(aa.into_iter().zip(ab));
    for (x, y) in &pairs {
        let ((mx, vx), (my, vy)) = (pooled_mean_var(x), pooled_mean_var(y));
        let (sx, sy) = (batch_means_se(x, mx), batch_means_se(y, my));
        let se = (sx[0] * sx[0] + sy[0] * sy[0]).sqrt();
        let se_v = (sx[1] * sx[1] + sy[1] * sy[1]).sqrt();
        println!("mean {mx} vs {my} (SE {se}), var {vx} vs {vy} (SE {se_v})");
        assert!((mx - my).abs() < 4.0 * se, "means {mx} vs {my} (SE {se})");
        assert!(
            (vx - vy).abs() < 4.0 * se_v,
            "variances {vx} vs {vy} (SE {se_v})"
        );
    }
}

#[test]
fn parallel_tempering_preserves_the_posterior() {
    // Two curves, one harmonic, no centring (exact target, unimodal): the β = 1 replica of
    // a tempered run must sample the same posterior as plain infHMC (itself validated
    // against long pCN and quadrature).
    let (data, t) = tiny_data(2);
    let plain = bayesian_align_group(
        &data,
        &t,
        &tiny_problem_cfg(BayesianSampler::InfHmc, 6000, 1, 21),
    )
    .unwrap();
    let mut cfg = tiny_problem_cfg(BayesianSampler::InfHmc, 6000, 1, 31);
    cfg.n_temperatures = 4;
    let pt = bayesian_align_group(&data, &t, &cfg).unwrap();
    println!("swap acceptance {:?}", pt.chains[0].swap_acceptance);
    assert!(mixing(&pt).1 < 1.02);
    assert_same_posterior(&plain, &pt, &t);
}

#[test]
#[ignore = "long invariance study: tempered infHMC / tempered pCN vs 1e6-iteration pCN (~2 min release)"]
fn parallel_tempering_and_long_pcn_agree_on_posterior_moments() {
    // Same design as `infhmc_and_long_pcn_agree_on_posterior_moments`; the reference is
    // untempered injective pCN, so a wrong tempered kernel (HMC scale, Gibbs template or σ²)
    // or swap ratio would bias the β = 1 marginal.
    let (data, t) = tiny_data(2);
    let mut pcn_cfg = tiny_problem_cfg(BayesianSampler::Pcn, 1_000_000, 20, 3);
    pcn_cfg.injective_warps = true;
    let pcn = bayesian_align_group(&data, &t, &pcn_cfg).unwrap();
    let mut hmc_cfg = tiny_problem_cfg(BayesianSampler::InfHmc, 50_000, 1, 7);
    hmc_cfg.n_temperatures = 4;
    let hmc = bayesian_align_group(&data, &t, &hmc_cfg).unwrap();
    let mut pt_pcn_cfg = pcn_cfg.clone();
    pt_pcn_cfg.seed = 11;
    pt_pcn_cfg.n_temperatures = 3;
    pt_pcn_cfg.n_iter = 600_000;
    pt_pcn_cfg.burn_in = 120_000;
    let pt_pcn = bayesian_align_group(&data, &t, &pt_pcn_cfg).unwrap();
    println!(
        "R̂ pcn {} pt-hmc {} pt-pcn {}; swaps hmc {:?} pcn {:?}",
        mixing(&pcn).1,
        mixing(&hmc).1,
        mixing(&pt_pcn).1,
        hmc.chains[0].swap_acceptance,
        pt_pcn.chains[0].swap_acceptance
    );
    assert!(mixing(&pcn).1 < 1.02 && mixing(&hmc).1 < 1.02 && mixing(&pt_pcn).1 < 1.02);
    assert_same_posterior(&hmc, &pcn, &t);
    assert_same_posterior(&pt_pcn, &pcn, &t);
}

#[test]
fn replica_exchange_acceptance_matches_formula() {
    // Two replicas (β = 1, 0.4) with log-likelihoods ℓ_A, ℓ_B: a swap is accepted with
    // probability min(1, exp((1 − 0.4)(ℓ_B − ℓ_A))); temperatures stay with the levels.
    let (data, t) = tiny_data(2);
    let cfg = tiny_problem_cfg(BayesianSampler::InfHmc, 100, 1, 1);
    let prob = Problem::new(&data, &t, &cfg).unwrap();
    let base = ChainState::init(&prob, true).unwrap();
    let sse: f64 = base.sse.iter().sum();
    let ll = |s2: f64| prob.log_likelihood(sse, s2);
    let s2a = sse / (prob.n_curves() * prob.t_len()) as f64; // maximises ℓ
    let s2b = [1.1, 1.2, 1.3, 1.5, 2.0]
        .iter()
        .map(|f| f * s2a)
        .find(|&s| {
            let p = (0.6 * (ll(s) - ll(s2a))).exp();
            p > 0.2 && p < 0.8
        })
        .unwrap();
    let want = (0.6 * (ll(s2b) - ll(s2a))).exp();
    let mk = |s2: f64, beta: f64| {
        let mut state = base.clone();
        state.sigma2 = s2;
        Replica {
            state,
            beta,
            hmc: None,
            warp_accept: vec![0; 2],
            tmpl_accept: 0,
        }
    };
    let mut reps = vec![mk(s2a, 1.0), mk(s2b, 0.4)];
    let mut rng = StdRng::seed_from_u64(4);
    let mut swaps = vec![0usize];
    let n = 40_000;
    for _ in 0..n {
        exchange_states(&mut reps, &prob, 0, &mut rng, &mut swaps);
        if reps[0].state.sigma2 != s2a {
            reps.swap(0, 1);
            let (b0, b1) = (reps[0].beta, reps[1].beta);
            reps[0].beta = b1;
            reps[1].beta = b0;
        }
        assert_eq!((reps[0].beta, reps[1].beta), (1.0, 0.4));
    }
    let rate = swaps[0] as f64 / n as f64;
    let se = (want * (1.0 - want) / n as f64).sqrt();
    println!("swap rate {rate} vs {want} (SE {se})");
    assert!((rate - want).abs() < 4.0 * se, "rate {rate} vs {want}");
    // Odd iterations attempt pair (1, 2) only: nothing to do with two replicas.
    exchange_states(&mut reps, &prob, 1, &mut rng, &mut swaps);
    assert_eq!(reps[0].state.sigma2, s2a);
    // A swap towards a better state for the cold replica is always accepted.
    let mut reps = vec![mk(s2b, 1.0), mk(s2a, 0.4)];
    exchange_states(&mut reps, &prob, 0, &mut rng, &mut swaps);
    assert_eq!(reps[0].state.sigma2, s2a);
}

#[test]
fn temperature_ladder_and_validation() {
    let mut cfg = BayesianGroupConfig::default();
    assert_eq!(temperature_ladder(&cfg), vec![1.0]);
    cfg.n_temperatures = 3;
    cfg.pt_min_inv_temp = 0.04;
    let l = temperature_ladder(&cfg);
    assert_eq!(l[0], 1.0);
    assert!(
        (l[1] - 0.2).abs() < 1e-15 && (l[2] - 0.04).abs() < 1e-15,
        "{l:?}"
    );
    let (data, t, _) = synth(21, 0.0, 1);
    for (k, b) in [(0, 0.1), (2, 0.0), (2, 1.0), (2, f64::NAN)] {
        let mut c = hmc_cfg(1);
        c.n_temperatures = k;
        c.pt_min_inv_temp = b;
        assert!(
            matches!(
                bayesian_align_group(&data, &t, &c),
                Err(FdarError::InvalidParameter { .. })
            ),
            "({k}, {b}) should be rejected"
        );
    }
    // β_min is ignored without tempering; tempering also works with pCN.
    let mut c = small_cfg(1);
    c.n_iter = 40;
    c.burn_in = 20;
    c.pt_min_inv_temp = 5.0;
    assert!(bayesian_align_group(&data, &t, &c).is_ok());
    c.pt_min_inv_temp = 0.2;
    c.n_temperatures = 3;
    let r = bayesian_align_group(&data, &t, &c).unwrap();
    assert_eq!(r.chains[0].swap_acceptance.len(), 2);
    assert!(r.chains[0].warp_step_sizes.is_empty());
}
