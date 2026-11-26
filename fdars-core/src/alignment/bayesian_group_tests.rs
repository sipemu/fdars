// Sealed-config idiom (`default()` + field assignment) mirrors external usage.
#![allow(clippy::field_reassign_with_default)]

use super::*;
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

/// FNV-1a over the bit patterns of a float sequence.
fn fnv(h: &mut u64, xs: &[f64]) {
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// Hash of every field that existed in the pCN-only release (065279f0).
fn result_hash(r: &BayesianGroupResult) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
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
fn warp_hash(r: &BayesianGroupResult) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
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
    let (h, hw) = (result_hash(&r), warp_hash(&r));
    println!("pcn result hash {h:#018x}, warp hash {hw:#018x}");
    assert_eq!(
        hw, PCN_RELEASE_WARP_HASH,
        "pCN warp/σ² results changed: {hw:#018x}"
    );
    assert_eq!(h, PCN_RESULT_HASH, "default pCN results changed: {h:#018x}");
    cfg.center_draws = false;
    let h_raw = result_hash(&bayesian_align_group(&data, &t, &cfg).unwrap());
    assert_eq!(
        h_raw, PCN_RELEASE_RAW_HASH,
        "pCN center_draws=false results changed: {h_raw:#018x}"
    );
}

/// Pre-#64 (c038fd46) hash of the template-independent fields of the default run.
const PCN_RELEASE_WARP_HASH: u64 = 0x8544_396a_c9c4_e4f3;
/// Pre-#64 (c038fd46) full hash of the `center_draws = false` run.
const PCN_RELEASE_RAW_HASH: u64 = 0xe256_f6f0_c204_2509;
/// Full hash of the default run after the #64 template-centring fix.
const PCN_RESULT_HASH: u64 = 0x423a_b6ba_faae_cb16;

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
/// fixed) by tensor-product quadrature on `centre ± w`, restricted to valid warps.
fn quadrature_moments(
    prob: &Problem,
    state: &ChainState,
    i: usize,
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
                -sse(&state.template, &wp.warped) / (2.0 * state.sigma2)
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
    let (data, t, _) = synth_in_model(31, 0.02, 100);
    let mut cfg = hmc_cfg(1);
    cfg.n_warp_harmonics = 1;
    let prob = Problem::new(&data, &t, &cfg).unwrap();
    let mut state = ChainState::init(&prob, true).unwrap();
    state.sigma2 = 0.5;
    // Coarse pass to locate the posterior, then a fine grid over ±10 posterior sd.
    let coarse = quadrature_moments(&prob, &state, 0, [0.0, 0.0], [1.2, 1.2], 401);
    let exact = quadrature_moments(
        &prob,
        &state,
        0,
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
        let out = hmc::hmc_transition(&prob, &state, 0, eps, 10, &mut rng);
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
    println!("eps {eps} acc {acc} chain ({ma}, {mb}, {va}, {vb}) ess ({ea}, {eb})");
    // Measured (seed 77): means 0.24544 / −0.12646 vs 0.24556 / −0.12651, variances
    // 1.475e-4 / 4.457e-4 vs 1.425e-4 / 4.380e-4, ESS ≈ 9600 / 10600 of 20000, acc 0.68.
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
