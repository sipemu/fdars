//! Tests of exponential-family registration with GFPCA (ALN-EXPFAM, GH #60).
//!
//! Golden values: R registr 2.2.1 via
//! `.planning/quick/261126-8kh-aln-expfam-60-exponential-family-registr/golden_registr.R`
//! (see `golden.rs`), on data from the portable LCG [`sim`] (generated
//! identically in R).

use super::basis::{glm_fit, quantile_knots, CubicBs};
use super::golden::*;
use super::warp::{digamma, CurveProblem, ObsLoss, TemplateScale, WarpParam};
use super::*;

// ------------------------------------------------------------------ data ----

struct Lcg(f64);

impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = (1_664_525.0 * self.0 + 1_013_904_223.0) % 4_294_967_296.0;
        (self.0 + 0.5) / 4_294_967_296.0
    }
    fn norm(&mut self) -> f64 {
        let (u1, u2) = (self.unif(), self.unif());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn pois(&mut self, mu: f64) -> f64 {
        let u = self.unif();
        let (mut k, mut p) = (0.0, (-mu).exp());
        let mut f = p;
        while u > f && k < 1000.0 {
            k += 1.0;
            p *= mu / k;
            f += p;
        }
        k
    }
}

fn plogis(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Port of `sim()` in golden_registr.R. Returns the data and the true
/// registered times per curve.
fn sim(seed: u64, n: usize, m: usize, family: u8, incompl: u8) -> (IrregFdata, Vec<Vec<f64>>) {
    use std::f64::consts::PI;
    let mut r = Lcg(seed as f64);
    let grid: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let (mut ts, mut ys, mut truth) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let a = 0.5 * r.norm();
        let w = [r.unif(), r.unif(), r.unif()];
        let tot: f64 = w.iter().sum();
        let cw = [w[0] / tot, (w[0] + w[1]) / tot, (w[0] + w[1] + w[2]) / tot];
        let t_true: Vec<f64> = grid
            .iter()
            .map(|&s| {
                3.0 * s * (1.0 - s).powi(2) * cw[0]
                    + 3.0 * s * s * (1.0 - s) * cw[1]
                    + s.powi(3) * cw[2]
            })
            .collect();
        let lat: Vec<f64> = t_true
            .iter()
            .map(|&t| -1.5 * ((2.0 * PI * t).sin() + (2.0 * PI * t).cos()) + a * (PI * t).sin())
            .collect();
        let val: Vec<f64> = lat
            .iter()
            .map(|&l| match family {
                0 => l + 0.1 * r.norm(),
                1 => f64::from(u8::from(r.unif() < plogis(l))),
                2 => r.pois((0.5 * l + 1.0).exp()),
                _ => {
                    let e: f64 = (0..4).map(|_| -r.unif().ln()).sum();
                    (0.3 * l).exp() / 4.0 * e
                }
            })
            .collect();
        let mut lo = 0;
        let mut hi = m;
        if incompl == 1 || incompl == 3 {
            lo = (r.unif() * 0.3 * m as f64).floor() as usize;
        }
        if incompl == 2 || incompl == 3 {
            hi = m - (r.unif() * 0.3 * m as f64).floor() as usize;
        }
        ts.push(grid[lo..hi].to_vec());
        ys.push(val[lo..hi].to_vec());
        truth.push(t_true[lo..hi].to_vec());
    }
    (IrregFdata::from_lists(&ts, &ys), truth)
}

fn data_sum(d: &IrregFdata) -> f64 {
    d.values.iter().sum()
}

fn cfg(family: ExpFamily, npc: usize) -> RegisterFpcaConfig {
    RegisterFpcaConfig {
        family,
        fpca: GfpcaConfig {
            npc,
            ..GfpcaConfig::default()
        },
        ..RegisterFpcaConfig::default()
    }
}

fn split(flat: &[f64], k: usize) -> Vec<Vec<f64>> {
    flat.chunks(k).map(<[f64]>::to_vec).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Per-curve initial-step losses of the given coefficients (and gamma scales)
/// against the template with spline coefficients `coefs`, evaluated by fdars.
fn initial_losses_at(
    data: &IrregFdata,
    c: &RegisterFpcaConfig,
    coefs: &[f64],
    betas: &[Vec<f64>],
    scales: Option<&[f64]>,
) -> Vec<f64> {
    let cd = curve_data(data, "data").unwrap();
    let setup = build_setup(&cd, c).unwrap();
    let knots = quantile_knots(&cd.pooled_sorted_t(), c.fpca.kt - 4);
    (0..cd.t.len())
        .map(|i| {
            // R `loss_h` builds the template basis as `bs(c(t_min, t_max, h), …)`,
            // whose boundary knots stretch when R's (slightly infeasible) warp
            // leaves [t_min, t_max]; reproduce that to evaluate R's loss exactly.
            let h: Vec<f64> = setup.warp[i]
                .theta
                .iter()
                .map(|r| super::basis::dot(r, &betas[i]))
                .collect();
            let lo = h.iter().copied().fold(setup.t_min, f64::min);
            let hi = h.iter().copied().fold(setup.t_max, f64::max);
            let basis = CubicBs::new(lo, hi, &knots);
            let prob = CurveProblem {
                y: &cd.y[i],
                theta_h: &setup.warp[i].theta,
                template_basis: &basis,
                template_coefs: coefs,
                loss: ObsLoss {
                    family: c.family,
                    scale: TemplateScale::Link,
                },
                incompleteness: c.incompleteness,
                lambda_inc: c.lambda_inc,
                bounds: setup.bounds(i),
            };
            prob.loss_grad(&betas[i], scales.map_or(1.0, |s| s[i]), false)
                .0
        })
        .collect()
}

/// Whether R's coefficients satisfy registr's constraints exactly (R's
/// `constrOptim` runs with `ci − 1e-6`, so its optimum may violate them by up to
/// `1e-6` — monotonicity ties or endpoints just outside `[t_min, t_max]`).
fn r_feasible(beta: &[f64], lo: f64, hi: f64) -> bool {
    beta.windows(2).all(|w| w[1] >= w[0]) && beta[0] >= lo && beta[beta.len() - 1] <= hi
}

fn glm_coefs_of(data: &IrregFdata, family: ExpFamily) -> Vec<f64> {
    let cd = curve_data(data, "data").unwrap();
    let p = cd.pooled_sorted_t();
    let knots = quantile_knots(&p, 4);
    let basis = CubicBs::new(p[0], p[p.len() - 1], &knots);
    let rows: Vec<Vec<f64>> = cd.t.iter().flatten().map(|&x| basis.eval(x)).collect();
    let ys: Vec<f64> = cd.y.iter().flatten().copied().collect();
    glm_fit(&rows, &ys, family).unwrap()
}

/// Registered copy of `data` from R's coefficients (Kh = 4: no inner knots).
fn registered_with(data: &IrregFdata, betas: &[Vec<f64>]) -> IrregFdata {
    let (mut ts, mut ys) = (Vec::new(), Vec::new());
    for (i, b) in betas.iter().enumerate() {
        let (t, y) = data.get_obs(i);
        ts.push(evaluate_inverse_warp(b, &[], t[0], t[t.len() - 1], t).unwrap());
        ys.push(y.to_vec());
    }
    IrregFdata::from_lists(&ts, &ys)
}

/// `|cos|` of the principal angle between two vectors.
fn abs_cos(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    (d / (na * nb)).abs()
}

/// Smallest cosine of the principal angles between two column spaces with
/// orthonormal columns (`m × k`, column-major).
fn subspace_min_cos(a: &[f64], b: &[f64], m: usize, k: usize) -> f64 {
    let am = nalgebra::DMatrix::from_column_slice(m, k, a);
    let bm = nalgebra::DMatrix::from_column_slice(m, k, b);
    let s = (am.transpose() * bm).svd(false, false).singular_values;
    s.iter().copied().fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------- goldens ----

#[test]
fn sim_reproduces_r_data() {
    let cases: [(u64, usize, usize, u8, u8, f64); 7] = [
        (101, 15, 50, 0, 0, G_DATA_SUM),
        (202, 40, 60, 1, 0, B_DATA_SUM),
        (303, 15, 50, 2, 0, P_DATA_SUM),
        (404, 15, 50, 3, 0, GA_DATA_SUM),
        (505, 15, 50, 0, 2, TR_DATA_SUM),
        (505, 15, 50, 0, 1, LE_DATA_SUM),
        (505, 15, 50, 0, 3, FU_DATA_SUM),
    ];
    for (seed, n, m, fam, inc, sum) in cases {
        let (d, _) = sim(seed, n, m, fam, inc);
        assert!(
            (data_sum(&d) - sum).abs() < 1e-9 * sum.abs().max(1.0),
            "seed {seed}"
        );
    }
    assert_eq!(sim(505, 15, 50, 0, 2).0.total_points(), TR_DATA_LEN);
    assert_eq!(sim(505, 15, 50, 0, 1).0.total_points(), LE_DATA_LEN);
    assert_eq!(sim(505, 15, 50, 0, 3).0.total_points(), FU_DATA_LEN);
}

#[test]
fn template_glm_matches_r() {
    let cases = [
        (sim(101, 15, 50, 0, 0).0, ExpFamily::Gaussian, &G_GLM_COEFS),
        (sim(202, 40, 60, 1, 0).0, ExpFamily::Binomial, &B_GLM_COEFS),
        (sim(303, 15, 50, 2, 0).0, ExpFamily::Poisson, &P_GLM_COEFS),
        (sim(404, 15, 50, 3, 0).0, ExpFamily::Gamma, &GA_GLM_COEFS),
    ];
    for (d, fam, want) in cases {
        let got = glm_coefs_of(&d, fam);
        assert!(
            max_abs_diff(&got, want) < 1e-7,
            "{fam:?}: {got:?} vs {want:?}"
        );
    }
}

/// The loss evaluated by fdars at R's optimum (with R's template GLM
/// coefficients) equals R's loss — validating `loss_h` for every family and
/// incompleteness mode — and fdars' own optimum is at least as good on every
/// curve where R's coefficients are feasible, with warps close to R's.
#[test]
fn initial_registration_matches_r() {
    struct Case {
        data: IrregFdata,
        cfg: RegisterFpcaConfig,
        coefs: &'static [f64],
        beta: &'static [f64],
        scales: Option<&'static [f64]>,
        loss: f64,
        beta_tol: f64,
    }
    let with_inc = |inc| {
        let mut c = cfg(ExpFamily::Gaussian, 1);
        c.incompleteness = Some(inc);
        c.lambda_inc = 0.05;
        c
    };
    // Coefficient tolerances reflect R's optimiser accuracy: BFGS `reltol = 1e-8`
    // (flat binary likelihoods: ~1e-4), the adaptive log-barrier (`mu = 1e-4`)
    // that keeps R's optimum away from active monotonicity constraints (fdars
    // reaches lower losses with tied coefficients there, ~5e-4 apart) and
    // Nelder–Mead for poisson / gamma.
    let cases = [
        Case {
            data: sim(101, 15, 50, 0, 0).0,
            cfg: cfg(ExpFamily::Gaussian, 1),
            coefs: &G_GLM_COEFS_REGISTR,
            beta: &G_INIT_BETA,
            scales: None,
            loss: G_INIT_LOSS,
            beta_tol: 1e-3,
        },
        Case {
            data: sim(202, 40, 60, 1, 0).0,
            cfg: cfg(ExpFamily::Binomial, 1),
            coefs: &B_GLM_COEFS_REGISTR,
            beta: &B_INIT_BETA,
            scales: None,
            loss: B_INIT_LOSS,
            beta_tol: 1e-3,
        },
        Case {
            data: sim(303, 15, 50, 2, 0).0,
            cfg: cfg(ExpFamily::Poisson, 1),
            coefs: &P_GLM_COEFS_REGISTR,
            beta: &P_INIT_BETA,
            scales: None,
            loss: P_INIT_LOSS,
            beta_tol: 1e-2,
        },
        Case {
            data: sim(404, 15, 50, 3, 0).0,
            cfg: cfg(ExpFamily::Gamma, 1),
            coefs: &GA_GLM_COEFS_REGISTR,
            beta: &GA_INIT_BETA,
            scales: Some(&GA_INIT_SCALE),
            loss: GA_INIT_LOSS,
            beta_tol: 1e-2,
        },
        Case {
            data: sim(505, 15, 50, 0, 2).0,
            cfg: with_inc(CurveIncompleteness::Trailing),
            coefs: &TR_GLM_COEFS_REGISTR,
            beta: &TR_INIT_BETA,
            scales: None,
            loss: TR_INIT_LOSS,
            beta_tol: 1e-3,
        },
        Case {
            data: sim(505, 15, 50, 0, 1).0,
            cfg: with_inc(CurveIncompleteness::Leading),
            coefs: &LE_GLM_COEFS_REGISTR,
            beta: &LE_INIT_BETA,
            scales: None,
            loss: LE_INIT_LOSS,
            beta_tol: 1e-3,
        },
        Case {
            data: sim(505, 15, 50, 0, 3).0,
            cfg: with_inc(CurveIncompleteness::Full),
            coefs: &FU_GLM_COEFS_REGISTR,
            beta: &FU_INIT_BETA,
            scales: None,
            loss: FU_INIT_LOSS,
            beta_tol: 1e-3,
        },
    ];
    for case in cases {
        let tag = format!("{:?} {:?}", case.cfg.family, case.cfg.incompleteness);
        let r_beta = split(case.beta, 4);
        let r_curve = initial_losses_at(&case.data, &case.cfg, case.coefs, &r_beta, case.scales);
        let at_r: f64 = r_curve.iter().sum();
        assert!(
            (at_r - case.loss).abs() < 1e-10 * case.loss.abs(),
            "{tag}: fdars loss at R beta {at_r} vs R {}",
            case.loss
        );
        let ours = register_expfam(&case.data, None, &case.cfg).unwrap();
        assert!(
            ours.loss <= case.loss * (1.0 + 1e-6),
            "{tag}: ours {} vs R {}",
            ours.loss,
            case.loss
        );
        let range = case
            .data
            .argvals
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        for i in 0..r_beta.len() {
            let (t, _) = case.data.get_obs(i);
            let (lo, hi) = match case.cfg.incompleteness {
                None => (t[0], t[t.len() - 1]),
                Some(_) => range,
            };
            let d = max_abs_diff(&ours.hinv_beta[i], &r_beta[i]);
            if r_feasible(&r_beta[i], lo, hi) {
                // Same template up to R's glm tolerance (1e-8 relative deviance).
                assert!(
                    ours.curve_loss[i] <= r_curve[i] + 1e-6 * r_curve[i].abs(),
                    "{tag} curve {i}: ours {} vs R {}",
                    ours.curve_loss[i],
                    r_curve[i]
                );
                assert!(
                    d < case.beta_tol,
                    "{tag} curve {i}: max |beta - beta_R| = {d} ours {:?} {} R {:?} {}",
                    ours.hinv_beta[i],
                    ours.curve_loss[i],
                    r_beta[i],
                    r_curve[i]
                );
            } else {
                assert!(
                    d < 0.05,
                    "{tag} curve {i} (R infeasible): max |beta - beta_R| = {d}"
                );
            }
        }
    }
}

#[test]
fn fpca_gauss_matches_r() {
    let (d, _) = sim(101, 15, 50, 0, 0);
    let reg = registered_with(&d, &split(&G_INIT_BETA, 4));
    let c = GfpcaConfig {
        npc: 2,
        psi_init: Some(FdMatrix::from_column_major(G_PSI0.to_vec(), 8, 2).unwrap()),
        ..GfpcaConfig::default()
    };
    let fit = gfpca_variational(&reg, ExpFamily::Gaussian, &c).unwrap();
    assert!(max_abs_diff(&fit.knots, &G_FPCA_KNOTS) < 1e-12);
    assert!(max_abs_diff(&fit.mean, &G_FPCA_MU) < 1e-7);
    assert!(max_abs_diff(&fit.evalues, &G_FPCA_EVAL) < 1e-6 * G_FPCA_EVAL[0]);
    assert!((fit.sigma2.unwrap() - G_FPCA_SIGMA2).abs() < 1e-9);
    for l in 0..2 {
        let ours = fit.efunctions.column(l);
        let r = &G_FPCA_EFUN[l * 100..(l + 1) * 100];
        assert!(abs_cos(ours, r) > 1.0 - 1e-10);
    }
    // R subject_coefs is Kt x n column-major; fdars n x Kt.
    for i in 0..15 {
        let r = &G_FPCA_SUBJ[i * 8..(i + 1) * 8];
        assert!(max_abs_diff(&fit.subject_coefs.row(i), r) < 1e-7);
    }
}

#[test]
fn bfpca_matches_r() {
    let (d, _) = sim(202, 40, 60, 1, 0);
    let reg = registered_with(&d, &split(&B_INIT_BETA, 4));
    let c = GfpcaConfig {
        psi_init: Some(FdMatrix::from_column_major(B_PSI0.to_vec(), 8, 1).unwrap()),
        xi_init: Some(1.0),
        ..GfpcaConfig::default()
    };
    let fit = gfpca_variational(&reg, ExpFamily::Binomial, &c).unwrap();
    assert_eq!(fit.n_iter + 1, B_FPCA_NITER);
    assert!(max_abs_diff(&fit.alpha_coefs, &B_FPCA_ALPHA) < 1e-7);
    assert!(max_abs_diff(&fit.mean, &B_FPCA_MU) < 1e-7);
    assert!((fit.evalues[0] - B_FPCA_EVAL[0]).abs() < 1e-6 * B_FPCA_EVAL[0]);
    assert!(abs_cos(fit.efunctions.column(0), &B_FPCA_EFUN) > 1.0 - 1e-10);
    for i in 0..40 {
        assert!(max_abs_diff(&fit.subject_coefs.row(i), &B_FPCA_SUBJ[i * 8..(i + 1) * 8]) < 1e-7);
    }
    // The EM start (template GLM of the registered data) is R's too.
    let cd = curve_data(&reg, "data").unwrap();
    let p = cd.pooled_sorted_t();
    let basis = CubicBs::new(p[0], p[p.len() - 1], &quantile_knots(&p, 4));
    let rows: Vec<Vec<f64>> = cd.t.iter().flatten().map(|&x| basis.eval(x)).collect();
    let ys: Vec<f64> = cd.y.iter().flatten().copied().collect();
    assert!(
        max_abs_diff(
            &glm_fit(&rows, &ys, ExpFamily::Binomial).unwrap(),
            &B_FPCA_ALPHA0
        ) < 1e-7
    );
}

/// Full `register_fpca` against R: different random EM starts and optimisers,
/// so agreement is to tolerance. For scale, R's own results move by up to
/// 0.046 (gaussian, npc 2), 0.010 (binomial), 2e-4 (poisson) and 3e-3
/// (trailing) in max |Δβ| when only `fpca_seed` changes (checked with registr
/// 2.2.1, seeds 1, 7, 42). Observed fdars-vs-R: 1e-4, 0.015, 7e-4, 0.011.
#[test]
fn register_fpca_gaussian_matches_r() {
    let (d, _) = sim(101, 15, 50, 0, 0);
    let fit = register_fpca(&d, None, &cfg(ExpFamily::Gaussian, 2)).unwrap();
    assert!(fit.converged);
    assert_eq!(fit.iterations, G_FULL_ITER);
    let beta = fit.registration.hinv_beta.concat();
    assert!(
        max_abs_diff(&beta, &G_FULL_BETA) < 0.01,
        "{}",
        max_abs_diff(&beta, &G_FULL_BETA)
    );
    assert!(max_abs_diff(&fit.fpca.mean, &G_FULL_MU) < 0.02);
    let cos = subspace_min_cos(fit.fpca.efunctions.as_slice(), &G_FULL_EFUN, 100, 2);
    assert!(cos > 0.99, "subspace cos {cos}");
    assert!((fit.registration_loss[0] - G_FULL_REGLOSS[0]).abs() < 1e-6 * G_FULL_REGLOSS[0]);
    let last = *fit.registration_loss.last().unwrap();
    let r_last = *G_FULL_REGLOSS.last().unwrap();
    assert!(
        (last - r_last).abs() < 0.05 * r_last,
        "final reg loss {last} vs R {r_last}"
    );
}

#[test]
fn register_fpca_binomial_matches_r() {
    let (d, _) = sim(202, 40, 60, 1, 0);
    let fit = register_fpca(&d, None, &cfg(ExpFamily::Binomial, 1)).unwrap();
    assert!(fit.converged);
    assert_eq!(fit.iterations, B_FULL_ITER);
    let beta = fit.registration.hinv_beta.concat();
    assert!(
        max_abs_diff(&beta, &B_FULL_BETA) < 0.03,
        "{}",
        max_abs_diff(&beta, &B_FULL_BETA)
    );
    let range = B_FULL_MU.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - B_FULL_MU.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(max_abs_diff(&fit.fpca.mean, &B_FULL_MU) < 0.05 * range);
    assert!(abs_cos(fit.fpca.efunctions.column(0), &B_FULL_EFUN) > 0.99);
    let last = *fit.registration_loss.last().unwrap();
    let r_last = *B_FULL_REGLOSS.last().unwrap();
    assert!(last <= r_last * 1.01, "final reg loss {last} vs R {r_last}");
}

#[test]
fn register_fpca_poisson_and_incomplete_match_r() {
    let (d, _) = sim(303, 15, 50, 2, 0);
    let fit = register_fpca(&d, None, &cfg(ExpFamily::Poisson, 1)).unwrap();
    let beta = fit.registration.hinv_beta.concat();
    assert!(
        max_abs_diff(&beta, &P_FULL_BETA) < 0.005,
        "{}",
        max_abs_diff(&beta, &P_FULL_BETA)
    );
    let last = *fit.registration_loss.last().unwrap();
    assert!(last <= P_FULL_REGLOSS.last().unwrap() * 1.01);

    let (d, _) = sim(505, 15, 50, 0, 2);
    let mut c = cfg(ExpFamily::Gaussian, 1);
    c.incompleteness = Some(CurveIncompleteness::Trailing);
    c.lambda_inc = 0.05;
    let fit = register_fpca(&d, None, &c).unwrap();
    let beta = fit.registration.hinv_beta.concat();
    assert!(
        max_abs_diff(&beta, &TR_FULL_BETA) < 0.03,
        "{}",
        max_abs_diff(&beta, &TR_FULL_BETA)
    );
    let last = *fit.registration_loss.last().unwrap();
    let r_last = *TR_FULL_REGLOSS.last().unwrap();
    assert!((last - r_last).abs() < 0.05 * r_last, "{last} vs {r_last}");
}

// ------------------------------------------------------------- invariants ----

#[test]
fn digamma_matches_known_values() {
    // ψ(1) = −γ, ψ(1/2) = −γ − 2 ln 2, ψ(10) = H_9 − γ.
    let gamma = 0.577_215_664_901_532_9;
    assert!((digamma(1.0) + gamma).abs() < 1e-13);
    assert!((digamma(0.5) + gamma + 2.0 * 2f64.ln()).abs() < 1e-13);
    let h9: f64 = (1..10).map(|k| 1.0 / k as f64).sum();
    assert!((digamma(10.0) - (h9 - gamma)).abs() < 1e-13);
}

/// Analytic gradients (loss_h + parameterisation pullback, incl. the gamma
/// scale) agree with central finite differences.
#[test]
fn gradients_match_finite_differences() {
    let incs = [
        None,
        Some(CurveIncompleteness::Leading),
        Some(CurveIncompleteness::Trailing),
        Some(CurveIncompleteness::Full),
    ];
    let fams = [
        (ExpFamily::Gaussian, 0, TemplateScale::Link),
        (ExpFamily::Binomial, 1, TemplateScale::Link),
        (ExpFamily::Poisson, 2, TemplateScale::Link),
        (ExpFamily::Poisson, 2, TemplateScale::Mean),
        (ExpFamily::Gamma, 3, TemplateScale::Link),
        (ExpFamily::Gamma, 3, TemplateScale::Mean),
    ];
    for (family, code, scale) in fams {
        let (d, _) = sim(77, 4, 30, code, 3);
        let cd = curve_data(&d, "data").unwrap();
        for inc in incs {
            let mut c = cfg(family, 1);
            c.kh = 6;
            c.incompleteness = inc;
            c.lambda_inc = 0.3;
            let setup = build_setup(&cd, &c).unwrap();
            let basis = CubicBs::new(setup.t_min, setup.t_max, &[0.3, 0.5, 0.7]);
            // Positive template values on the mean scale; moderate on the link scale.
            let coefs: Vec<f64> = (0..7)
                .map(|k| 1.0 + 0.4 * ((k as f64) * 1.3).sin())
                .collect();
            let i = 1;
            let prob = CurveProblem {
                y: &cd.y[i],
                theta_h: &setup.warp[i].theta,
                template_basis: &basis,
                template_coefs: &coefs,
                loss: ObsLoss { family, scale },
                incompleteness: inc,
                lambda_inc: c.lambda_inc,
                bounds: setup.bounds(i),
            };
            let param = WarpParam {
                incompleteness: inc,
                kh: 6,
                bounds: setup.bounds(i),
            };
            let np = param.n_params();
            let f = |x: &[f64]| {
                let dec = param.decode(&x[..np]);
                let s = x[np].exp();
                let (l, gb, gs) = prob.loss_grad(&dec.beta, s, true);
                let mut g = param.pullback(&dec, &gb);
                g.push(gs * s);
                (l, g)
            };
            let x0: Vec<f64> = (0..=np).map(|k| 0.3 * ((k as f64) * 2.1).cos()).collect();
            let (_, g) = f(&x0);
            for k in 0..=np {
                let h = 1e-6;
                let (mut xp, mut xm) = (x0.clone(), x0.clone());
                xp[k] += h;
                xm[k] -= h;
                let fd = (f(&xp).0 - f(&xm).0) / (2.0 * h);
                let tol = 1e-5 * (1.0 + fd.abs());
                assert!(
                    (fd - g[k]).abs() < tol,
                    "{family:?} {scale:?} {inc:?} k={k}: fd {fd} vs {}",
                    g[k]
                );
            }
        }
    }
}

/// Decoded coefficients respect registr's constraints and encode/decode
/// round-trips.
#[test]
fn parameterisation_is_monotone_and_invertible() {
    let b = super::warp::WarpBounds {
        t_min: 0.0,
        t_max: 1.0,
        t_min_i: 0.2,
        t_max_i: 0.8,
    };
    for inc in [
        None,
        Some(CurveIncompleteness::Leading),
        Some(CurveIncompleteness::Trailing),
        Some(CurveIncompleteness::Full),
    ] {
        let p = WarpParam {
            incompleteness: inc,
            kh: 6,
            bounds: b,
        };
        for s in 0..20 {
            let x: Vec<f64> = (0..p.n_params())
                .map(|k| 3.0 * ((s * 7 + k) as f64 * 0.77).sin())
                .collect();
            let beta = p.decode(&x).beta;
            assert!(beta.windows(2).all(|w| w[1] > w[0]));
            assert!(beta[0] >= -1e-15 && beta[5] <= 1.0 + 1e-15);
            match inc {
                None => assert!((beta[0] - 0.2).abs() < 1e-15 && (beta[5] - 0.8).abs() < 1e-15),
                Some(CurveIncompleteness::Leading) => assert!((beta[5] - 0.8).abs() < 1e-15),
                Some(CurveIncompleteness::Trailing) => assert!((beta[0] - 0.2).abs() < 1e-15),
                Some(_) => {}
            }
            let back = p.decode(&p.encode(&beta)).beta;
            // encode moves the start inside by FRAC_EPS = 1e-3 of the span.
            assert!(max_abs_diff(&back, &beta) < 2e-3);
        }
    }
}

/// Root-mean-square distance of the (globally re-centred) registered times from
/// the truth; the model identifies warps only up to a common warp, so both are
/// compared after removing their cross-sectional mean at each grid point.
fn centred_rmse(est: &[Vec<f64>], truth: &[Vec<f64>]) -> f64 {
    let (n, m) = (est.len(), est[0].len());
    let mut ss = 0.0;
    for j in 0..m {
        let me: f64 = est.iter().map(|e| e[j]).sum::<f64>() / n as f64;
        let mt: f64 = truth.iter().map(|t| t[j]).sum::<f64>() / n as f64;
        for i in 0..n {
            ss += ((est[i][j] - me) - (truth[i][j] - mt)).powi(2);
        }
    }
    (ss / (n * m) as f64).sqrt()
}

#[test]
fn recovers_known_warps_binomial_and_poisson() {
    for (code, family, n, m, ratio) in [
        (1, ExpFamily::Binomial, 80, 100, 0.5),
        (2, ExpFamily::Poisson, 40, 100, 0.5),
    ] {
        let (d, truth) = sim(9_000 + code as u64, n, m, code, 0);
        let fit = register_fpca(&d, None, &cfg(family, 1)).unwrap();
        let ident: Vec<Vec<f64>> = (0..n).map(|i| d.get_obs(i).0.to_vec()).collect();
        let before = centred_rmse(&ident, &truth);
        let after = centred_rmse(&fit.registration.t_hat, &truth);
        assert!(
            after < ratio * before,
            "{family:?}: rmse {after} vs identity {before}"
        );
        for th in &fit.registration.t_hat {
            assert!(th.windows(2).all(|w| w[1] >= w[0]));
            assert!(th[0].abs() < 1e-12 && (th[m - 1] - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn identical_curves_give_identity_warps() {
    let t: Vec<f64> = (0..50).map(|j| j as f64 / 49.0).collect();
    let y: Vec<f64> = t
        .iter()
        .map(|&s| (2.0 * std::f64::consts::PI * s).sin() + s)
        .collect();
    let d = IrregFdata::from_lists(&vec![t.clone(); 6], &vec![y; 6]);
    let reg = register_expfam(&d, None, &cfg(ExpFamily::Gaussian, 1)).unwrap();
    for th in &reg.t_hat {
        assert!(max_abs_diff(th, &t) < 1e-3, "{}", max_abs_diff(th, &t));
    }
}

/// A large `λ_inc` pins the free endpoints to the observed ones; `λ_inc = 0`
/// lets them move.
#[test]
fn lambda_inc_controls_endpoints() {
    let (d, _) = sim(505, 15, 50, 0, 3);
    let gap = |lambda: f64, inc: CurveIncompleteness| {
        let mut c = cfg(ExpFamily::Gaussian, 1);
        c.incompleteness = Some(inc);
        c.lambda_inc = lambda;
        let reg = register_expfam(&d, None, &c).unwrap();
        (0..d.n_obs())
            .map(|i| {
                let (t, _) = d.get_obs(i);
                let th = &reg.t_hat[i];
                let (lo, hi) = (th[0] - t[0], th[th.len() - 1] - t[t.len() - 1]);
                match inc {
                    CurveIncompleteness::Leading => lo.abs(),
                    CurveIncompleteness::Trailing => hi.abs(),
                    CurveIncompleteness::Full => {
                        ((th[th.len() - 1] - th[0]) - (t[t.len() - 1] - t[0])).abs()
                    }
                }
            })
            .fold(0.0, f64::max)
    };
    for inc in [
        CurveIncompleteness::Leading,
        CurveIncompleteness::Trailing,
        CurveIncompleteness::Full,
    ] {
        let free = gap(0.0, inc);
        let pinned = gap(1e4, inc);
        assert!(pinned < 1e-3, "{inc:?}: pinned gap {pinned}");
        assert!(
            free > 10.0 * pinned,
            "{inc:?}: free {free} vs pinned {pinned}"
        );
    }
}

#[test]
fn register_fpca_regular_and_template_and_gamma() {
    let (d, _) = sim(404, 15, 50, 3, 0);
    let fit = register_fpca(&d, None, &cfg(ExpFamily::Gamma, 1)).unwrap();
    let sc = fit.registration.gamma_scales.as_ref().unwrap();
    assert_eq!(sc.len(), 15);
    assert!(sc.iter().all(|&s| s > 0.0 && s.is_finite()));
    assert!(fit.registration_loss.last().unwrap() < &fit.registration_loss[0]);

    // FdMatrix front end == IrregFdata on a common grid.
    let (g, _) = sim(101, 15, 50, 0, 0);
    let mut mat = FdMatrix::zeros(15, 50);
    for i in 0..15 {
        for (j, v) in g.get_obs(i).1.iter().enumerate() {
            mat[(i, j)] = *v;
        }
    }
    let t = g.get_obs(0).0.to_vec();
    let a = register_fpca_regular(&mat, &t, &cfg(ExpFamily::Gaussian, 1)).unwrap();
    let b = register_fpca(&g, None, &cfg(ExpFamily::Gaussian, 1)).unwrap();
    assert_eq!(a, b);
    // A template equal to the data gives the same initial registration.
    let r1 = register_expfam(&g, Some(&g), &cfg(ExpFamily::Gaussian, 1)).unwrap();
    let r2 = register_expfam(&g, None, &cfg(ExpFamily::Gaussian, 1)).unwrap();
    assert_eq!(r1, r2);
    let reg = b.registered_data(&g);
    assert_eq!(reg.argvals, b.registration.t_hat.concat());
}

#[test]
fn invalid_inputs_are_rejected() {
    let (g, _) = sim(101, 5, 20, 0, 0);
    let mut c = cfg(ExpFamily::Gaussian, 1);
    c.kh = 3;
    assert!(register_fpca(&g, None, &c).is_err());
    let mut c = cfg(ExpFamily::Gaussian, 9);
    c.fpca.kt = 8;
    assert!(register_fpca(&g, None, &c).is_err());
    let mut c = cfg(ExpFamily::Gaussian, 1);
    c.lambda_inc = -1.0;
    assert!(register_fpca(&g, None, &c).is_err());
    // Gaussian data are not binary / positive counts.
    assert!(register_fpca(&g, None, &cfg(ExpFamily::Binomial, 1)).is_err());
    assert!(register_fpca(&g, None, &cfg(ExpFamily::Gamma, 1)).is_err());
    // Unsorted argvals.
    let bad = IrregFdata::from_lists(
        &[vec![0.0, 0.5, 0.4], vec![0.0, 0.5, 1.0]],
        &[vec![1.0; 3], vec![1.0; 3]],
    );
    assert!(register_expfam(&bad, None, &RegisterFpcaConfig::default()).is_err());
    // Template range must match.
    let short: Vec<f64> = (0..10).map(|j| 0.8 * j as f64 / 9.0).collect();
    let h = IrregFdata::from_lists(&[short.clone(), short], &[vec![0.0; 10], vec![1.0; 10]]);
    assert!(register_expfam(&g, Some(&h), &RegisterFpcaConfig::default()).is_err());
    assert!(evaluate_inverse_warp(&[0.0, 1.0], &[], 0.0, 1.0, &[0.5]).is_err());
}

/// Curve-specific irregular grids (random 60 % subsamples, endpoints kept):
/// registration still recovers the known warps of binary data.
#[test]
fn recovers_known_warps_on_irregular_grids() {
    let (full, truth_full) = sim(9_001, 80, 100, 1, 0);
    let mut r = Lcg(4242.0);
    let (mut ts, mut ys, mut truth) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..full.n_obs() {
        let (t, y) = full.get_obs(i);
        let keep: Vec<usize> = (0..t.len())
            .filter(|&j| j == 0 || j == t.len() - 1 || r.unif() < 0.6)
            .collect();
        ts.push(keep.iter().map(|&j| t[j]).collect::<Vec<f64>>());
        ys.push(keep.iter().map(|&j| y[j]).collect::<Vec<f64>>());
        truth.push(keep.iter().map(|&j| truth_full[i][j]).collect::<Vec<f64>>());
    }
    assert!(ts.iter().any(|t| t.len() != ts[0].len()));
    let d = IrregFdata::from_lists(&ts, &ys);
    let fit = register_fpca(&d, None, &cfg(ExpFamily::Binomial, 1)).unwrap();
    // The grids differ, so the per-grid-point centring of `centred_rmse` is not
    // available: use the raw RMSE on each curve's own points.
    let rmse = |est: &[Vec<f64>]| {
        let (mut ss, mut k) = (0.0, 0usize);
        for (e, t) in est.iter().zip(&truth) {
            for (a, b) in e.iter().zip(t) {
                ss += (a - b).powi(2);
                k += 1;
            }
        }
        (ss / k as f64).sqrt()
    };
    let before = rmse(&ts);
    let after = rmse(&fit.registration.t_hat);
    assert!(after < 0.6 * before, "rmse {after} vs identity {before}");
    for th in &fit.registration.t_hat {
        assert!(th.windows(2).all(|w| w[1] >= w[0]));
    }
}
