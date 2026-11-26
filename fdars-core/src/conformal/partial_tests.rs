// Sealed-config idiom (`default()` + field assignment) mirrors external usage.
#![allow(clippy::field_reassign_with_default)]

use super::*;
use rand::prelude::*;
use rand_distr::Normal;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Two-bump base shape; the fixed template of the synthetic model.
fn base(z1: f64, z2: f64, s: f64) -> f64 {
    z1 * (-((s - 0.3) / 0.09).powi(2) / 2.0).exp() + z2 * (-((s - 0.7) / 0.09).powi(2) / 2.0).exp()
}

/// Synthetic warped data (adapted from the paper's simulation: two Gaussian bumps with
/// random amplitudes, composed with random warps). `z2` is correlated with `z1` so the
/// observed prefix carries information about the unobserved part; warps are the
/// fdasrsf exponential family `γ(t) = (e^{a t} − 1)/(e^a − 1)`, `a ~ U(−1, 1)`.
fn simulate(n: usize, t: &[f64], rng: &mut StdRng) -> FdMatrix {
    let m = t.len();
    let nrm = Normal::new(0.0, 1.0).unwrap();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let z1 = 1.0 + 0.3 * nrm.sample(rng);
        let z2 = z1 + 0.1 * nrm.sample(rng);
        let a: f64 = rng.gen_range(-1.0..1.0);
        for j in 0..m {
            let g = if a.abs() < 1e-8 {
                t[j]
            } else {
                ((a * t[j]).exp() - 1.0) / (a.exp() - 1.0)
            };
            data[(i, j)] = base(z1, z2, g);
        }
    }
    data
}

fn template(t: &[f64]) -> Vec<f64> {
    t.iter().map(|&s| base(1.0, 1.0, s)).collect()
}

fn rows_except(data: &FdMatrix, skip: usize) -> FdMatrix {
    let idx: Vec<usize> = (0..data.nrows()).filter(|&i| i != skip).collect();
    select_rows(data, &idx)
}

fn prefix(row: &[f64], m_obs: usize) -> Vec<f64> {
    row[..m_obs].to_vec()
}

// ─── Closed-form affine full-conformal interval ─────────────────────────────

/// Independent brute force: accept y on a fine grid, return hull.
fn brute_interval(a: &[f64], b: &[f64], c: f64, k: usize) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let n = 400_000;
    for s in 0..=n {
        let y = -20.0 + 40.0 * s as f64 / n as f64;
        let st = (y - c).abs();
        let below = a
            .iter()
            .zip(b)
            .filter(|(&ai, &bi)| (ai - bi * y).abs() < st)
            .count();
        if below < k {
            lo = lo.min(y);
            hi = hi.max(y);
        }
    }
    (lo, hi)
}

#[test]
fn affine_interval_matches_brute_force() {
    let mut rng = StdRng::seed_from_u64(7);
    for trial in 0..20 {
        let n = 5 + trial % 6;
        let a: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let b: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..0.6)).collect();
        let c = rng.gen_range(-1.0..1.0);
        let k = 1 + trial % n;
        let (lo, hi) = affine_conformal_interval(&a, &b, c, k);
        let (blo, bhi) = brute_interval(&a, &b, c, k);
        assert!(lo <= c && c <= hi, "centre must be accepted");
        assert!((lo - blo).abs() < 2e-4, "trial {trial}: lo {lo} vs {blo}");
        assert!((hi - bhi).abs() < 2e-4, "trial {trial}: hi {hi} vs {bhi}");
    }
}

#[test]
fn affine_interval_split_conformal_limit() {
    // b = 0: calibration scores do not depend on y, so the set is the classic split
    // conformal interval c ± (k-th smallest |a_i|).
    let a = [0.5, -0.1, 0.3, -0.9, 0.2];
    let b = [0.0; 5];
    let (lo, hi) = affine_conformal_interval(&a, &b, 1.0, 3);
    assert!(
        (lo - 0.7).abs() < 1e-12 && (hi - 1.3).abs() < 1e-12,
        "{lo} {hi}"
    );
}

#[test]
fn affine_interval_unbounded_when_test_dominates() {
    // b_i = 1 with a_i > c: for y → −∞ every calibration score |a_i − y| exceeds the
    // new score |y|, so the set is unbounded below; above, it ends at y = 1.
    let a = [2.0, 3.0];
    let b = [1.0, 1.0];
    let (lo, hi) = affine_conformal_interval(&a, &b, 0.0, 1);
    assert!(lo == f64::NEG_INFINITY, "lo = {lo}");
    assert!((hi - 1.0).abs() < 1e-12, "hi = {hi}");
}

// ─── Exact finite-sample invariant (rotation argument) ──────────────────────

/// Each of the n_2 + 1 exchangeable curves in turn plays the new curve. The fit is a
/// symmetric function of the augmented sample, so the number of rotations whose target
/// is covered is ≥ k = ⌈(1 − α)(n_2 + 1)⌉ deterministically — pointwise for the
/// amplitude band and for the phase set.
fn rotation_counts(
    dist: PartialPredictorDistance,
    m_obs: usize,
    alpha: f64,
) -> (Vec<usize>, usize, usize) {
    let t = grid(31);
    let m = t.len();
    let mut rng = StdRng::seed_from_u64(11);
    let n = 16;
    let data = simulate(n, &t, &mut rng);
    let tmpl = template(&t);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = alpha;
    cfg.distance = dist;
    cfg.n_phase_radii = 4;
    let mut amp_cov = vec![0usize; m];
    let mut ph_cov = 0usize;
    for r in 0..n {
        let model =
            partial_conformal_fit_with_template(&rows_except(&data, r), &t, &tmpl, &cfg).unwrap();
        let truth = elastic_align_pair(&tmpl, &data.row(r), &t, 0.0);
        let p = partial_conformal_predict(&model, &prefix(&data.row(r), m_obs), &cfg).unwrap();
        for j in 0..m {
            let v = truth.f_aligned[j];
            if p.amplitude_lower[j] - 1e-9 <= v && v <= p.amplitude_upper[j] + 1e-9 {
                amp_cov[j] += 1;
            }
        }
        if p.phase_set_contains(&truth.gamma).unwrap() {
            ph_cov += 1;
        }
    }
    let k = ((n as f64) * (1.0 - alpha) - 1e-12).ceil() as usize;
    (amp_cov, ph_cov, k)
}

#[test]
fn rotation_invariant_l2() {
    let (amp, ph, k) = rotation_counts(PartialPredictorDistance::L2, 14, 0.2);
    assert!(amp.iter().all(|&c| c >= k), "amp {amp:?} < k = {k}");
    assert!(ph >= k, "phase {ph} < k = {k}");
}

#[test]
fn rotation_invariant_srsf_full_observation() {
    let (amp, ph, k) = rotation_counts(PartialPredictorDistance::Srsf, 31, 0.3);
    assert!(amp.iter().all(|&c| c >= k), "amp {amp:?} < k = {k}");
    assert!(ph >= k, "phase {ph} < k = {k}");
}

// ─── Monte-Carlo marginal coverage ──────────────────────────────────────────

#[test]
fn monte_carlo_coverage_several_alpha() {
    let t = grid(31);
    let m = t.len();
    let tmpl = template(&t);
    let reps = 150;
    let n2 = 19;
    let alphas = [0.1, 0.2, 0.3];
    let mut amp_cov = [0.0f64; 3];
    let mut ph_cov = [0usize; 3];
    let mut rng = StdRng::seed_from_u64(2026);
    for _ in 0..reps {
        let data = simulate(n2 + 1, &t, &mut rng);
        let cal = rows_except(&data, n2);
        let m_obs = rng.gen_range(8..m); // random observation end s (Assumption 1)
        let truth = elastic_align_pair(&tmpl, &data.row(n2), &t, 0.0);
        let mut cfg = PartialConformalConfig::default();
        cfg.n_phase_radii = 2;
        let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
        for (a, &alpha) in alphas.iter().enumerate() {
            cfg.alpha = alpha;
            let p = partial_conformal_predict(&model, &prefix(&data.row(n2), m_obs), &cfg).unwrap();
            let covered = (0..m)
                .filter(|&j| {
                    let v = truth.f_aligned[j];
                    p.amplitude_lower[j] <= v && v <= p.amplitude_upper[j]
                })
                .count();
            amp_cov[a] += covered as f64 / m as f64;
            if p.phase_set_contains(&truth.gamma).unwrap() {
                ph_cov[a] += 1;
            }
        }
    }
    for (a, &alpha) in alphas.iter().enumerate() {
        let target = 1.0 - alpha;
        let upper = target + 1.0 / (n2 + 1) as f64;
        let sd = (target * alpha / reps as f64).sqrt();
        let amp = amp_cov[a] / reps as f64;
        let ph = ph_cov[a] as f64 / reps as f64;
        assert!(
            amp >= target - 3.0 * sd && amp <= upper + 3.0 * sd,
            "alpha {alpha}: amplitude coverage {amp}"
        );
        assert!(
            ph >= target - 3.0 * sd && ph <= upper + 3.0 * sd,
            "alpha {alpha}: phase coverage {ph}"
        );
    }
}

// ─── Efficiency / degeneracy / determinism ──────────────────────────────────

fn mean_width(p: &PartialConformalPrediction) -> f64 {
    p.amplitude_upper
        .iter()
        .zip(&p.amplitude_lower)
        .map(|(u, l)| u - l)
        .sum::<f64>()
        / p.amplitude_lower.len() as f64
}

#[test]
fn sets_shrink_with_more_calibration_data() {
    let t = grid(31);
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(5);
    let big = simulate(80, &t, &mut rng);
    let tests = simulate(10, &t, &mut rng);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.2;
    cfg.n_phase_radii = 4;
    let small_idx: Vec<usize> = (0..9).collect();
    let small =
        partial_conformal_fit_with_template(&select_rows(&big, &small_idx), &t, &tmpl, &cfg)
            .unwrap();
    let large = partial_conformal_fit_with_template(&big, &t, &tmpl, &cfg).unwrap();
    let (mut ws, mut wl, mut rs, mut rl) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..tests.nrows() {
        let obs = prefix(&tests.row(i), 16);
        let ps = partial_conformal_predict(&small, &obs, &cfg).unwrap();
        let pl = partial_conformal_predict(&large, &obs, &cfg).unwrap();
        ws += mean_width(&ps);
        wl += mean_width(&pl);
        rs += ps.phase_radius;
        rl += pl.phase_radius;
    }
    assert!(wl < ws, "amplitude width: n2=80 {wl} vs n2=9 {ws}");
    assert!(rl < rs, "phase radius: n2=80 {rl} vs n2=9 {rs}");
}

#[test]
fn full_observation_degenerates_sensibly() {
    let t = grid(31);
    let m = t.len();
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(9);
    let cal = simulate(30, &t, &mut rng);
    let tests = simulate(8, &t, &mut rng);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.2;
    cfg.n_phase_radii = 4;
    let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
    let (mut w_full, mut w_short) = (0.0, 0.0);
    for i in 0..tests.nrows() {
        let full = partial_conformal_predict(&model, &tests.row(i), &cfg).unwrap();
        // s = 1: the registration window is the whole template and the exact full
        // registration (the conformal target) is exposed.
        assert_eq!(full.registration.start_index, 0);
        assert_eq!(full.registration.end_index, m - 1);
        assert!((full.registration.domain_fraction - 1.0).abs() < 1e-12);
        let fr = full.full_registration.as_ref().expect("full registration");
        let direct = elastic_align_pair(&tmpl, &tests.row(i), &t, 0.0);
        assert_eq!(fr, &direct);
        assert!((full.observed_end - 1.0).abs() < 1e-12);
        let short = partial_conformal_predict(&model, &prefix(&tests.row(i), 5), &cfg).unwrap();
        assert!(short.full_registration.is_none());
        w_full += mean_width(&full);
        w_short += mean_width(&short);
    }
    assert!(w_full < w_short, "full {w_full} vs short-prefix {w_short}");
}

#[test]
fn deterministic_under_seed() {
    let t = grid(25);
    let mut rng = StdRng::seed_from_u64(3);
    let data = simulate(16, &t, &mut rng);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.25;
    cfg.karcher_max_iter = 5;
    cfg.n_phase_radii = 3;
    let m1 = partial_conformal_fit(&data, &t, &cfg).unwrap();
    let m2 = partial_conformal_fit(&data, &t, &cfg).unwrap();
    assert_eq!(m1, m2);
    assert_eq!(m1.training_indices.len() + m1.calibration_indices.len(), 16);
    let obs = prefix(&data.row(0), 12);
    let p1 = partial_conformal_predict(&m1, &obs, &cfg).unwrap();
    let p2 = partial_conformal_predict(&m2, &obs, &cfg).unwrap();
    assert_eq!(p1, p2);
    cfg.seed = 43;
    let m3 = partial_conformal_fit(&data, &t, &cfg).unwrap();
    assert_ne!(m1.calibration_indices, m3.calibration_indices);
}

#[test]
fn phase_set_structure() {
    let t = grid(31);
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(21);
    let cal = simulate(25, &t, &mut rng);
    let obs = simulate(1, &t, &mut rng).row(0);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.2;
    let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
    let p = partial_conformal_predict(&model, &prefix(&obs, 15), &cfg).unwrap();
    // The predicted centre has new-point score 0 and is always in the set.
    assert!(p.phase_set_contains(&p.phase_center).unwrap());
    // Every reported trial is a member and a valid warp inside the envelope.
    for r in 0..p.phase_accepted.nrows() {
        let g = p.phase_accepted.row(r);
        assert!(p.phase_set_contains(&g).unwrap());
        assert!(g.windows(2).all(|w| w[1] >= w[0] - 1e-12));
        assert!((g[0] - t[0]).abs() < 1e-9 && (g[t.len() - 1] - 1.0).abs() < 1e-9);
    }
    assert!(p.phase_radius > 0.0 && p.phase_radius < std::f64::consts::FRAC_PI_2);
    // A warp far from all calibration warps is excluded.
    let extreme: Vec<f64> = t.iter().map(|&s| s.powi(8)).collect();
    assert!(!p.phase_set_contains(&extreme).unwrap());
    // NW weights are a probability vector; centre lies inside the amplitude band.
    assert!((p.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for j in 0..t.len() {
        assert!(p.amplitude_lower[j] <= p.amplitude_center[j]);
        assert!(p.amplitude_center[j] <= p.amplitude_upper[j]);
    }
}

#[test]
fn elastic_distance_option_runs() {
    let t = grid(21);
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(4);
    let cal = simulate(10, &t, &mut rng);
    let obs = simulate(1, &t, &mut rng).row(0);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.3;
    cfg.distance = PartialPredictorDistance::Elastic;
    cfg.n_phase_radii = 3;
    let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
    let p = partial_conformal_predict(&model, &prefix(&obs, 12), &cfg).unwrap();
    assert!(p.amplitude_lower.iter().all(|v| v.is_finite()));
    assert!(p.amplitude_upper.iter().all(|v| v.is_finite()));
    assert!(p.bandwidth > 0.0);
}

#[test]
fn input_validation() {
    let t = grid(15);
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(1);
    let cal = simulate(5, &t, &mut rng);
    let cfg = PartialConformalConfig::default();
    let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
    // alpha = 0.1 needs n2 >= 9
    assert!(partial_conformal_predict(&model, &prefix(&cal.row(0), 8), &cfg).is_err());
    let mut ok = cfg.clone();
    ok.alpha = 0.3;
    assert!(partial_conformal_predict(&model, &[1.0], &ok).is_err());
    assert!(partial_conformal_predict(&model, &[0.0; 16], &ok).is_err());
    assert!(partial_conformal_predict(&model, &[0.0, f64::NAN], &ok).is_err());
    let mut bad = ok.clone();
    bad.bandwidth = Some(0.0);
    assert!(partial_conformal_predict(&model, &prefix(&cal.row(0), 8), &bad).is_err());
    let rev: Vec<f64> = t.iter().rev().copied().collect();
    assert!(partial_conformal_fit_with_template(&cal, &rev, &tmpl, &ok).is_err());
    assert!(partial_conformal_fit_with_template(&cal, &t, &tmpl[..3], &ok).is_err());
    let mut badc = ok;
    badc.cal_fraction = 1.0;
    assert!(partial_conformal_fit(&cal, &t, &badc).is_err());
}
