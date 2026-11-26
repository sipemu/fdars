// Calibration of the credible bands of `bayesian_align_group` (GH #64).
//
// Data are simulated *from the model itself*: a known template SRSF `q*` in the span of
// the template B-spline basis, known warps in the span of the warp basis, and data SRSFs
// for which the discrete likelihood the sampler evaluates,
//
//     interp(q_i, γ_i(t_j)) · ψ_i(t_j) = q*(t_j) + ε_ij ,   ε_ij ~ N(0, σ²) iid,
//
// holds exactly: curve i is observed at the abscissae `s_ij = γ_i(t_j)` with values
// `(q*(t_j) + ε_ij) / ψ_i(t_j)` (the SRSF group action `(q* + ε, γ_i⁻¹)` sampled at those
// points), so the linear interpolant hits its nodes exactly at the true warp. (Data on the
// sampling grid itself cannot be exactly in-model for non-trivial warps: the map
// `q_i ↦ interp(q_i, γ_i(t))·ψ_i` is a linear-spline collocation matrix, non-singular only
// under the Schoenberg–Whitney condition `|γ_i(t_j) − t_j| < h`, and badly conditioned near
// it.) The sampler then runs on these SRSFs directly (bypassing f → SRSF).
#![allow(clippy::field_reassign_with_default)]

use super::*;
use rand::SeedableRng;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Smooth two-bump template function.
fn g1(s: f64) -> f64 {
    (-((s - 0.35) / 0.15).powi(2)).exp() + 0.7 * (-((s - 0.7) / 0.12).powi(2)).exp()
}

/// One in-model data set.
pub(super) struct InModel {
    pub prob: Problem,
    /// Curves whose SRSFs are (approximately) `prob.q`; used only for `f0` / aligned data.
    pub data: FdMatrix,
    pub t: Vec<f64>,
    /// True warps centred by their inverse Karcher mean (unit grid).
    pub warps_centred: FdMatrix,
    /// True template SRSF warped by the same centring warp.
    pub template_centred: Vec<f64>,
    /// True (uncentred) template SRSF and warps.
    pub q_star: Vec<f64>,
    pub truth: FdMatrix,
    /// Generating warp coefficients.
    pub coefs: Vec<Vec<f64>>,
}

/// `n` valid warps with coefficients ~ N(0, coef_sd² / j²) (harmonic j), then centred
/// (composed with their inverse Karcher mean, re-projected onto the warp basis; three
/// passes) so that the truth is (up to projection error) a fixed point of the sampler's
/// centring and the template `q*` stays in the span of the template basis.
fn centred_warps(prob: &Problem, n: usize, coef_sd: f64, rng: &mut StdRng) -> Vec<Warp> {
    let nk = prob.warp_basis.len();
    let mut warps: Vec<Warp> = (0..n)
        .map(|i| loop {
            let coef: Vec<f64> = (0..nk)
                .map(|k| coef_sd / (k / 2 + 1) as f64 * rng.sample::<f64, _>(StandardNormal))
                .collect();
            if let Some(w) = prob.warp_from_coef(i, coef) {
                if w.psi.iter().all(|&p| p > 0.05) {
                    break w;
                }
            }
        })
        .collect();
    for _ in 0..3 {
        let gam_i = centering_warp(&warps, &prob.t);
        warps = warps
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let g = reparameterize_curve(&w.gamma, &prob.t, &gam_i);
                prob.warp_from_coef(i, prob.coef_from_gamma(&g)).unwrap()
            })
            .collect();
    }
    warps
}

/// Simulate from the model: `n` curves, grid size `m`, noise sd `sigma`, warp
/// coefficients ~ N(0, coef_sd² / j²) (harmonic j), conditioned on a valid warp.
pub(super) fn simulate_in_model(
    cfg: &BayesianGroupConfig,
    n: usize,
    m: usize,
    sigma: f64,
    coef_sd: f64,
    seed: u64,
) -> InModel {
    let t = grid(m);
    let mut rng = StdRng::seed_from_u64(seed);
    // Placeholder curves (the template itself) to build the problem and its priors.
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            data[(i, j)] = g1(t[j]);
        }
    }
    let mut prob = Problem::new(&data, &t, cfg).unwrap();
    // True template: g1's SRSF projected onto the template basis.
    let g1t: Vec<f64> = t.iter().map(|&s| g1(s)).collect();
    let q_star = prob.template_from_coef(&prob.template_coef(&srsf_single(&g1t, &t)));
    let warps = centred_warps(&prob, n, coef_sd, &mut rng);
    let mut truth = FdMatrix::zeros(n, m);
    let mut grids = Vec::with_capacity(n);
    let coefs: Vec<Vec<f64>> = warps.iter().map(|w| w.coef.clone()).collect();
    for (i, w) in warps.into_iter().enumerate() {
        let qi: Vec<f64> = (0..m)
            .map(|j| (q_star[j] + sigma * rng.sample::<f64, _>(StandardNormal)) / w.psi[j])
            .collect();
        // Curve with that SRSF on its own abscissae (only f0 / aligned data use it).
        let qq: Vec<f64> = qi.iter().map(|v| v * v.abs()).collect();
        let f = cumtrapz(&qq, &w.gamma);
        for j in 0..m {
            data[(i, j)] = linear_interp(&w.gamma, &f, t[j]);
            truth[(i, j)] = w.gamma[j];
        }
        prob.q[i] = qi;
        grids.push(w.gamma);
    }
    prob.q_grid = Some(grids);
    let gam_i = sqrt_mean_inverse(&truth, &t);
    let mut warps_centred = FdMatrix::zeros(n, m);
    for i in 0..n {
        let c = reparameterize_curve(&truth.row(i), &t, &gam_i);
        for j in 0..m {
            warps_centred[(i, j)] = c[j];
        }
    }
    let template_centred = warp_srsf(&q_star, &t, &gam_i);
    InModel {
        prob,
        data,
        t,
        warps_centred,
        template_centred,
        q_star,
        truth,
        coefs,
    }
}

/// Run all chains of `cfg` on a prepared problem.
pub(super) fn run_chains(prob: &Problem, cfg: &BayesianGroupConfig) -> Vec<ChainRun> {
    (0..cfg.n_chains)
        .map(|k| run_chain(prob, cfg, cfg.seed.wrapping_add(k as u64)).unwrap())
        .collect()
}

/// Coverage of interior points of `truth` (n × m) by the bands.
pub(super) fn band_coverage(lower: &FdMatrix, upper: &FdMatrix, truth: &FdMatrix) -> f64 {
    let (n, m) = truth.shape();
    let mut hit = 0usize;
    for i in 0..n {
        for j in 1..m - 1 {
            let v = truth[(i, j)];
            if v >= lower[(i, j)] - 1e-12 && v <= upper[(i, j)] + 1e-12 {
                hit += 1;
            }
        }
    }
    hit as f64 / (n * (m - 2)) as f64
}

pub(super) fn vec_coverage(lower: &[f64], upper: &[f64], truth: &[f64]) -> f64 {
    let m = truth.len();
    let hit = (1..m - 1)
        .filter(|&j| truth[j] >= lower[j] - 1e-12 && truth[j] <= upper[j] + 1e-12)
        .count();
    hit as f64 / (m - 2) as f64
}

/// (warp coverage, template-SRSF coverage, mean warp band width) of one result.
pub(super) fn coverages(res: &BayesianGroupResult, sim: &InModel) -> [f64; 3] {
    let (n, m) = res.gammas.shape();
    let mut w = 0.0;
    for i in 0..n {
        for j in 0..m {
            w += res.gammas_upper[(i, j)] - res.gammas_lower[(i, j)];
        }
    }
    [
        band_coverage(&res.gammas_lower, &res.gammas_upper, &sim.warps_centred),
        vec_coverage(
            &res.template_srsf_lower,
            &res.template_srsf_upper,
            &sim.template_centred,
        ),
        w / (n * m) as f64,
    ]
}

/// Mean and Monte-Carlo standard error over data seeds.
pub(super) fn mean_se(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mu = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (n - 1.0);
    (mu, (var / n).sqrt())
}

pub(super) fn base_cfg(sampler: BayesianSampler, n_iter: usize, seed: u64) -> BayesianGroupConfig {
    let mut c = BayesianGroupConfig::default();
    c.sampler = sampler;
    c.injective_warps = true;
    c.n_warp_harmonics = 1;
    c.n_template_basis = 12;
    c.n_iter = n_iter;
    c.burn_in = n_iter / 2;
    c.n_chains = 4;
    c.seed = seed;
    c
}

#[test]
fn in_model_simulation_satisfies_the_likelihood_exactly() {
    // At the true warps the sampler's warped SRSFs are q* + ε with ε iid N(0, σ²).
    let cfg = base_cfg(BayesianSampler::Pcn, 100, 0);
    let sigma = 0.1;
    let sim = simulate_in_model(&cfg, 6, 41, sigma, 0.2, 3);
    let mut resid = Vec::new();
    for i in 0..6 {
        // The sampler's own warp of curve i at the generating coefficients.
        let w = sim.prob.warp_from_coef(i, sim.coefs[i].clone()).unwrap();
        assert_eq!(w.gamma, sim.prob.q_grid.as_ref().unwrap()[i]);
        resid.extend(w.warped.iter().zip(&sim.q_star).map(|(a, b)| a - b));
    }
    let n = resid.len() as f64;
    let mean = resid.iter().sum::<f64>() / n;
    let sd = (resid.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt();
    // 246 iid N(0, σ²) values: sample moments within Monte-Carlo error.
    assert!(
        mean.abs() < 0.03 && (sd / sigma - 1.0).abs() < 0.15,
        "mean {mean} sd {sd}"
    );
    // Noise-free: the true parameters fit exactly (same seed ⇒ same warps).
    let sim0 = simulate_in_model(&cfg, 6, 41, 0.0, 0.2, 3);
    for i in 0..6 {
        let w = sim0.prob.warp_from_coef(i, sim0.coefs[i].clone()).unwrap();
        assert_eq!(sim0.coefs[i], sim.coefs[i]);
        assert!(w
            .warped
            .iter()
            .zip(&sim0.q_star)
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }
    // The design is centred: centring moves the true warps only by the projection error.
    let shift = sim
        .truth
        .as_slice()
        .iter()
        .zip(sim.warps_centred.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(shift < 5e-3, "centring shift {shift}");
    // The centring warp of the centred truth is the identity.
    let gi = sqrt_mean_inverse(&sim.warps_centred, &sim.t);
    assert!(gi.iter().zip(&sim.t).all(|(a, b)| (a - b).abs() < 1e-3));
}

/// Coverage of the template SRSF bands when the draws are re-centred the pre-#64 way
/// (warping the linear interpolant of the template's grid values).
fn interp_template_coverage(sim: &InModel, runs: &[ChainRun], level: f64) -> f64 {
    let m = sim.t.len();
    let mut rows = Vec::new();
    for r in runs {
        for d in &r.draws {
            let warps: Vec<Warp> = d
                .warp_coefs
                .iter()
                .enumerate()
                .map(|(i, c)| sim.prob.warp_from_coef(i, c.clone()).unwrap())
                .collect();
            let gi = centering_warp(&warps, &sim.t);
            rows.push(warp_srsf(&d.template, &sim.t, &gi));
        }
    }
    let (lo, _, hi) = pointwise_bands(&rows_to_matrix(&rows, m), level);
    vec_coverage(&lo, &hi, &sim.template_centred)
}

/// Monte-Carlo study summary (means ± SE over data sets).
struct Study {
    warp: (f64, f64),
    template: (f64, f64),
    template_interp: (f64, f64),
    warp_raw: (f64, f64),
    template_raw: (f64, f64),
    sigma2_ratio: f64,
    max_rhat: f64,
}

/// `reps` in-model data sets (5 curves, 31 points, warp coefficient sd 0.1), 4 chains
/// each; centred (`center_draws = true`) and raw (`false`) summaries of the same chains.
/// σ² prior IG(1e-6, 1e-6) so that the prior does not floor σ² (see module docs).
fn study(sampler: BayesianSampler, sigma: f64, n_iter: usize, reps: u64) -> Study {
    let mut rows: [Vec<f64>; 6] = Default::default();
    let mut max_rhat: f64 = 0.0;
    for s in 0..reps {
        let mut cfg = base_cfg(sampler, n_iter, 1000 + 4 * s);
        cfg.alpha0 = 1e-6;
        cfg.beta0 = 1e-6;
        let sim = simulate_in_model(&cfg, 5, 31, sigma, 0.1, 500 + s);
        let runs = run_chains(&sim.prob, &cfg);
        let res = assemble(&sim.prob, &runs, &sim.data, &sim.t, &cfg);
        let c = coverages(&res, &sim);
        rows[0].push(c[0]);
        rows[1].push(c[1]);
        rows[2].push(interp_template_coverage(&sim, &runs, cfg.credible_level));
        rows[5].push(res.sigma2_mean / (sigma * sigma));
        max_rhat = res
            .rhat_warps
            .iter()
            .copied()
            .chain([res.rhat_sigma2])
            .fold(max_rhat, f64::max);
        cfg.center_draws = false;
        let raw = assemble(&sim.prob, &runs, &sim.data, &sim.t, &cfg);
        let c = coverages(&raw, &sim);
        rows[3].push(c[0]);
        rows[4].push(c[1]);
    }
    let st = Study {
        warp: mean_se(&rows[0]),
        template: mean_se(&rows[1]),
        template_interp: mean_se(&rows[2]),
        warp_raw: mean_se(&rows[3]),
        template_raw: mean_se(&rows[4]),
        sigma2_ratio: mean_se(&rows[5]).0,
        max_rhat,
    };
    println!(
        "{sampler:?} σ={sigma} n_iter={n_iter} reps={reps}: warp {:.3}±{:.3} template {:.3}±{:.3} \
         (pre-#64 interp centring {:.3}±{:.3}); raw draws: warp {:.3}±{:.3} template {:.3}±{:.3}; \
         σ²/σ²_true {:.3}; max R̂ {:.3}",
        st.warp.0,
        st.warp.1,
        st.template.0,
        st.template.1,
        st.template_interp.0,
        st.template_interp.1,
        st.warp_raw.0,
        st.warp_raw.1,
        st.template_raw.0,
        st.template_raw.1,
        st.sigma2_ratio,
        st.max_rhat
    );
    st
}

#[test]
fn credible_bands_calibrated_on_in_model_data() {
    // Short version of the long study: infHMC, small noise (where the pointwise SRSF
    // likelihood is close to a proper likelihood), 3 data sets × 2 chains × 1500 iterations.
    // Per-data-set coverage varies by ±0.05; values are deterministic. Measured
    // (2026-11-26): warp 0.959 (0.883, 0.993, 1.0), template 0.897 (0.828, 0.966, 0.897);
    // the pre-#64 interpolated template centring gives lower template coverage.
    let mut rows = [Vec::new(), Vec::new()];
    for s in 0..3u64 {
        let mut cfg = base_cfg(BayesianSampler::InfHmc, 1500, 70 + 4 * s);
        cfg.n_chains = 2;
        cfg.alpha0 = 1e-6;
        cfg.beta0 = 1e-6;
        let sim = simulate_in_model(&cfg, 5, 31, 0.02, 0.1, 80 + s);
        let runs = run_chains(&sim.prob, &cfg);
        let res = assemble(&sim.prob, &runs, &sim.data, &sim.t, &cfg);
        let c = coverages(&res, &sim);
        rows[0].push(c[0]);
        rows[1].push(c[1]);
    }
    let (w, t) = (mean_se(&rows[0]).0, mean_se(&rows[1]).0);
    println!(
        "short in-model coverage: warp {w:.3} {:?}, template {t:.3} {:?}",
        rows[0], rows[1]
    );
    assert!(
        w > SHORT_WARP_MIN && t > SHORT_TEMPLATE_MIN,
        "warp {w}, template {t}"
    );
}

const SHORT_WARP_MIN: f64 = 0.92;
const SHORT_TEMPLATE_MIN: f64 = 0.86;

#[test]
#[ignore = "long in-model calibration study (~6 min release): 40 data sets x 4 chains per row"]
fn credible_bands_calibrated_on_in_model_data_long_study() {
    // Measured (release, 2026-11-26), coverage of nominal-95% pointwise bands, mean ± MC SE
    // over data sets (centred summaries, the default):
    //   infHMC σ=0.01: warp 0.964±0.008, template 0.945±0.008 (pre-#64 0.899±0.012), σ² ×1.04
    //   infHMC σ=0.1 : warp 0.945±0.011, template 0.915±0.011 (pre-#64 0.827±0.014), σ² ×0.90
    //   pCN    σ=0.1 : warp 0.923±0.014, template 0.924±0.010 (pre-#64 0.909±0.012), σ² ×1.00
    //   infHMC σ=0.3 : warp 0.926±0.023, template 0.909±0.016 (pre-#64 0.847±0.021), σ² ×0.74
    // Raw (`center_draws = false`) warp bands over-cover (0.96–0.985): they carry the
    // non-identified common phase that wanders between centring steps.
    let hmc_small = study(BayesianSampler::InfHmc, 0.01, 8000, 40);
    let hmc = study(BayesianSampler::InfHmc, 0.1, 8000, 40);
    let pcn = study(BayesianSampler::Pcn, 0.1, 40000, 40);
    let hmc_big = study(BayesianSampler::InfHmc, 0.3, 8000, 20);
    // Calibrated where the pointwise SRSF likelihood is close to a proper likelihood.
    assert!(
        (hmc_small.warp.0 - 0.95).abs() < 0.03,
        "{:?}",
        hmc_small.warp
    );
    assert!(
        (hmc_small.template.0 - 0.95).abs() < 0.03,
        "{:?}",
        hmc_small.template
    );
    assert!(hmc.warp.0 > 0.92 && hmc.template.0 > 0.89);
    assert!(pcn.warp.0 > 0.89 && pcn.template.0 > 0.89);
    assert!(hmc_big.warp.0 > 0.88 && hmc_big.template.0 > 0.87);
    // The #64 fix: exact spline centring vs warping the interpolant.
    for st in [&hmc_small, &hmc, &hmc_big] {
        assert!(st.template.0 > st.template_interp.0 + 0.03);
    }
    // σ² shrinks with the noise level (noise-smoothing of the interpolated likelihood).
    assert!(hmc_small.sigma2_ratio > 0.95 && hmc_big.sigma2_ratio < 0.85);
    for st in [&hmc_small, &hmc, &pcn, &hmc_big] {
        assert!(st.warp_raw.0 > st.warp.0 && st.max_rhat < 1.05);
    }
}

#[test]
fn centred_template_uses_the_exact_spline() {
    // Warping the B-spline template by γ must agree with the spline evaluated at γ(t)
    // (reference: the same operation on a 16× finer grid, where linear interpolation is
    // accurate); the pre-#64 coarse `warp_srsf` has the interpolation (chord) error.
    let cfg = base_cfg(BayesianSampler::Pcn, 100, 0);
    let sim = simulate_in_model(&cfg, 3, 31, 0.0, 0.1, 9);
    let t = &sim.t;
    let coef = sim.prob.template_coef(&sim.q_star);
    let gamma: Vec<f64> = t
        .iter()
        .map(|&s| s + 0.04 * (std::f64::consts::PI * s).sin() * (1.0 - s).max(0.0))
        .collect();
    assert!(gamma.windows(2).all(|w| w[1] > w[0]));
    // Identity: exactly the template.
    let id = sim.prob.warp_template(&coef, t);
    let tc = sim.prob.template_from_coef(&coef);
    assert!(id.iter().zip(&tc).all(|(a, b)| (a - b).abs() < 1e-12));
    let exact = sim.prob.warp_template(&coef, &gamma);
    let coarse = warp_srsf(&sim.q_star, t, &gamma);
    let tf = grid(481);
    let qf = Problem::expand(&template_bspline_basis(&tf, coef.len(), 3), &coef, tf.len());
    let qf_at: Vec<f64> = gamma.iter().map(|&g| linear_interp(&tf, &qf, g)).collect();
    let dg = gradient_uniform(&gamma, 1.0 / 30.0);
    let reference: Vec<f64> = qf_at.iter().zip(&dg).map(|(q, d)| q * d.sqrt()).collect();
    let err = |v: &[f64]| {
        v.iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };
    let (e_exact, e_coarse) = (err(&exact), err(&coarse));
    println!("max error: exact spline {e_exact:.2e}, interpolated {e_coarse:.2e}");
    assert!(e_exact < 2e-3, "exact {e_exact}");
    assert!(
        e_coarse > 10.0 * e_exact,
        "coarse {e_coarse} vs exact {e_exact}"
    );
}
