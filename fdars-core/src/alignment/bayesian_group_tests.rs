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
    // Small fixed-seed Monte-Carlo study, warps inside the model's warp space.
    // Coverage of the 95% pointwise bands (interior points, 5 curves, 3 data
    // seeds) is asserted with a margin for MC error and the SRSF-noise
    // misspecification (noise is added to f, not to q); measured values in SUMMARY.
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
    assert!(mean > 0.75, "mean coverage {mean} ({cov:?})");
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
    // 30 template basis functions): pointwise bands under-cover somewhat because the
    // SRSF of noisy curves is not the model's iid-Gaussian SRSF noise and the warps are
    // posterior-centred. Asserted at the same margin as the short study.
    println!("long-study coverage {mean:.3} {cov:?}");
    assert!(mean > 0.75, "mean coverage {mean} ({cov:?})");
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
