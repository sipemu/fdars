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

// ═══════════════════════════════════════════════════════════════════════════
// GH #72: simultaneous bands, held-out bandwidth tuning, fragments / sparse
// ═══════════════════════════════════════════════════════════════════════════

fn fnv(v: &[f64], h: &mut u64) {
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// Defaults are bit-identical to the pre-#72 code: FNV-1a hash of the full output
/// pinned on commit 9c856e31 before the new options were added.
#[test]
fn default_results_bit_identical_to_pre_72_pin() {
    let m = 41;
    let t = grid(m);
    let n = 30;
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let a = 0.8 + 0.4 * ((i * 13 % 7) as f64 / 6.0);
        let sh = 0.1 * ((i * 7 % 5) as f64 / 4.0 - 0.5);
        for j in 0..m {
            data[(i, j)] = a * (-(t[j] - 0.5 - sh).powi(2) / 0.02).exp()
                + 0.3 * a * (-(t[j] - 0.2).powi(2) / 0.01).exp();
        }
    }
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for dist in [
        PartialPredictorDistance::L2,
        PartialPredictorDistance::Srsf,
        PartialPredictorDistance::Elastic,
    ] {
        let mut cfg = PartialConformalConfig::default();
        cfg.distance = dist;
        cfg.alpha = 0.2;
        cfg.n_phase_radii = 3;
        let model = partial_conformal_fit(&data, &t, &cfg).unwrap();
        for mo in [10, 25, 41] {
            let p = partial_conformal_predict(&model, &data.row(3)[..mo], &cfg).unwrap();
            fnv(&p.amplitude_center, &mut h);
            fnv(&p.amplitude_lower, &mut h);
            fnv(&p.amplitude_upper, &mut h);
            fnv(&p.phase_center, &mut h);
            fnv(&p.phase_lower, &mut h);
            fnv(&p.phase_upper, &mut h);
            fnv(&[p.phase_radius, p.bandwidth], &mut h);
            fnv(&p.weights, &mut h);
            fnv(p.phase_accepted.as_slice(), &mut h);
            fnv(&p.registration.gamma, &mut h);
            assert!(p.amplitude_radius.is_none() && p.bandwidth_selection.is_none());
        }
    }
    assert_eq!(h, 0x2d35_858d_07e5_6e48);
}

// ─── Simultaneous band ──────────────────────────────────────────────────────

/// Full-conformal acceptance of a trial curve `y` under the max-type score.
fn sup_accept(resp: &FdMatrix, nw: &NwWeights, k: usize, sigma: &[f64], y: &[f64]) -> bool {
    let (n2, m) = resp.shape();
    let c: Vec<f64> = (0..m)
        .map(|j| (0..n2).map(|l| nw.test[l] * resp[(l, j)]).sum())
        .collect();
    let s_new = (0..m)
        .map(|j| (y[j] - c[j]).abs() / sigma[j])
        .fold(0.0_f64, f64::max);
    let below = (0..n2)
        .filter(|&i| {
            let bi = nw.b[i];
            let si = (0..m)
                .map(|j| {
                    let fit = (1.0 - bi) * nw.center_of(i, resp, j) + bi * y[j];
                    (resp[(i, j)] - fit).abs() / sigma[j]
                })
                .fold(0.0_f64, f64::max);
            si < s_new
        })
        .count();
    below < k
}

#[test]
fn simultaneous_band_contains_every_accepted_trial_curve() {
    // Independent brute force: random trial curves around the centre; every one the
    // full-conformal rule accepts must lie inside centre ± R σ, and the band must be
    // reached (some accepted trial has sup-score close to R).
    let t = grid(13);
    let m = t.len();
    let mut rng = StdRng::seed_from_u64(72);
    let cal = simulate(12, &t, &mut rng);
    let resp = cal.clone();
    let n2 = 11;
    let dist: Vec<f64> = (0..=n2)
        .flat_map(|i| {
            (0..=n2).map(move |j| {
                if i == j {
                    0.0
                } else {
                    0.2 + 0.07 * ((i * 5 + j * 5) % 7) as f64
                }
            })
        })
        .collect();
    let nw = NwWeights::new(&dist, n2, 0.3);
    let resp = select_rows(&resp, &(0..n2).collect::<Vec<_>>());
    let sigma: Vec<f64> = t.iter().map(|&s| 0.5 + s).collect();
    let k = rank_threshold(n2, 0.25).unwrap();
    let amp = simultaneous_band(&resp, &nw, k, &sigma);
    let r = amp.radius.unwrap();
    assert!(r.is_finite() && r > 0.0);
    let mut max_accepted = 0.0_f64;
    for trial in 0..4000 {
        let scale = 1.6 * r * (trial as f64 / 4000.0);
        let y: Vec<f64> = (0..m)
            .map(|j| amp.center[j] + scale * sigma[j] * rng.gen_range(-1.0..1.0))
            .collect();
        if sup_accept(&resp, &nw, k, &sigma, &y) {
            let sc = (0..m)
                .map(|j| (y[j] - amp.center[j]).abs() / sigma[j])
                .fold(0.0_f64, f64::max);
            assert!(
                sc <= r + 1e-12,
                "accepted trial with sup-score {sc} > R {r}"
            );
            max_accepted = max_accepted.max(sc);
        }
    }
    // The superset bound is not grossly loose.
    assert!(
        max_accepted > 0.7 * r,
        "max accepted {max_accepted} vs R {r}"
    );
}

#[test]
fn simultaneous_band_split_limit_is_kth_sup_residual() {
    // b_i = 0 (new point has no weight in the calibration fits): R is the k-th smallest
    // calibration sup-score sup_t |Y_i − m_i| / σ — the split-conformal sup-norm band.
    let mut rng = StdRng::seed_from_u64(3);
    let (n2, m) = (9, 6);
    let mut resp = FdMatrix::zeros(n2, m);
    for i in 0..n2 {
        for j in 0..m {
            resp[(i, j)] = rng.gen_range(-1.0..1.0);
        }
    }
    let mut u = vec![0.0; n2 * n2];
    for i in 0..n2 {
        for j in 0..n2 {
            if i != j {
                u[i * n2 + j] = 1.0 / (n2 - 1) as f64;
            }
        }
    }
    let nw = NwWeights {
        u,
        b: vec![0.0; n2],
        test: vec![1.0 / n2 as f64; n2],
        n2,
    };
    let sigma = vec![2.0; m];
    let k = rank_threshold(n2, 0.2).unwrap();
    let mut scores: Vec<f64> = (0..n2)
        .map(|i| {
            (0..m)
                .map(|j| {
                    let mi: f64 = (0..n2)
                        .filter(|&l| l != i)
                        .map(|l| resp[(l, j)])
                        .sum::<f64>()
                        / (n2 - 1) as f64;
                    (resp[(i, j)] - mi).abs() / 2.0
                })
                .fold(0.0_f64, f64::max)
        })
        .collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let amp = simultaneous_band(&resp, &nw, k, &sigma);
    assert!((amp.radius.unwrap() - scores[k - 1]).abs() < 1e-12);
    for j in 0..m {
        assert!((amp.upper[j] - amp.center[j] - 2.0 * scores[k - 1]).abs() < 1e-12);
    }
}

fn whole_curve_covered(p: &PartialConformalPrediction, truth: &[f64]) -> bool {
    truth
        .iter()
        .enumerate()
        .all(|(j, &v)| p.amplitude_lower[j] - 1e-9 <= v && v <= p.amplitude_upper[j] + 1e-9)
}

/// Rotation invariant for a general observation pattern and band type: each of the
/// n exchangeable curves plays the new curve; the number of rotations whose target is
/// covered is ≥ k deterministically (amplitude: pointwise per t, or the whole curve for
/// simultaneous bands; phase: membership).
fn rotation_counts_general(
    obs: &PartialObservation,
    cfg: &PartialConformalConfig,
) -> (Vec<usize>, usize, usize, usize) {
    let t = grid(31);
    let m = t.len();
    let mut rng = StdRng::seed_from_u64(11);
    let n = 16;
    let data = simulate(n, &t, &mut rng);
    let tmpl = template(&t);
    let mut amp_cov = vec![0usize; m];
    let (mut whole, mut ph_cov) = (0usize, 0usize);
    for r in 0..n {
        let model =
            partial_conformal_fit_with_template(&rows_except(&data, r), &t, &tmpl, cfg).unwrap();
        let truth = elastic_align_pair(&tmpl, &data.row(r), &t, 0.0);
        let values = extract_observed(obs, &t, &data.row(r));
        let p = partial_conformal_predict_observed(&model, obs, &values, cfg).unwrap();
        for j in 0..m {
            let v = truth.f_aligned[j];
            if p.amplitude_lower[j] - 1e-9 <= v && v <= p.amplitude_upper[j] + 1e-9 {
                amp_cov[j] += 1;
            }
        }
        whole += usize::from(whole_curve_covered(&p, &truth.f_aligned));
        ph_cov += usize::from(p.phase_set_contains(&truth.gamma).unwrap());
    }
    let k = ((n as f64) * (1.0 - cfg.alpha) - 1e-12).ceil() as usize;
    (amp_cov, whole, ph_cov, k)
}

#[test]
fn rotation_invariant_simultaneous_band() {
    let t = grid(31);
    for (obs, scale) in [
        (PartialObservation::Prefix(14), PartialBandScale::Constant),
        (
            PartialObservation::Prefix(31),
            PartialBandScale::Custom(t.iter().map(|&s| 0.3 + base(1.0, 1.0, s)).collect()),
        ),
        (
            PartialObservation::Fragments(vec![(3, 9), (18, 25)]),
            PartialBandScale::Constant,
        ),
    ] {
        let mut cfg = PartialConformalConfig::default();
        cfg.alpha = 0.2;
        cfg.n_phase_radii = 2;
        cfg.band = PartialAmplitudeBand::Simultaneous;
        cfg.band_scale = scale;
        let (_, whole, ph, k) = rotation_counts_general(&obs, &cfg);
        assert!(
            whole >= k,
            "{obs:?}: whole-curve coverage {whole} < k = {k}"
        );
        assert!(ph >= k, "{obs:?}: phase {ph} < k = {k}");
    }
}

#[test]
fn rotation_invariant_fragments_and_sparse() {
    for (obs, dist) in [
        (
            PartialObservation::Fragments(vec![(2, 8), (15, 22)]),
            PartialPredictorDistance::L2,
        ),
        (
            PartialObservation::Fragments(vec![(0, 5), (10, 14), (20, 30)]),
            PartialPredictorDistance::Srsf,
        ),
        (
            PartialObservation::Sparse(vec![0.07, 0.21, 0.38, 0.55, 0.81]),
            PartialPredictorDistance::L2,
        ),
    ] {
        let mut cfg = PartialConformalConfig::default();
        cfg.alpha = 0.2;
        cfg.distance = dist;
        cfg.n_phase_radii = 2;
        let (amp, _, ph, k) = rotation_counts_general(&obs, &cfg);
        assert!(
            amp.iter().all(|&c| c >= k),
            "{obs:?}: amp {amp:?} < k = {k}"
        );
        assert!(ph >= k, "{obs:?}: phase {ph} < k = {k}");
    }
}

#[test]
fn single_fragment_reproduces_prefix_exactly() {
    let t = grid(31);
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(8);
    let cal = simulate(20, &t, &mut rng);
    let new = simulate(1, &t, &mut rng).row(0);
    for dist in [PartialPredictorDistance::L2, PartialPredictorDistance::Srsf] {
        for band in [
            PartialAmplitudeBand::Pointwise,
            PartialAmplitudeBand::Simultaneous,
        ] {
            for n_obs in [12, 31] {
                let mut cfg = PartialConformalConfig::default();
                cfg.alpha = 0.2;
                cfg.distance = dist;
                cfg.band = band;
                cfg.n_phase_radii = 3;
                let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
                let p = partial_conformal_predict(&model, &new[..n_obs], &cfg).unwrap();
                let frag = PartialObservation::Fragments(vec![(0, n_obs - 1)]);
                let mut q =
                    partial_conformal_predict_observed(&model, &frag, &new[..n_obs], &cfg).unwrap();
                assert_eq!(q.fragment_registrations, vec![p.registration.clone()]);
                q.fragment_registrations.clear();
                assert_eq!(p, q);
                let pre = PartialObservation::Prefix(n_obs);
                let r =
                    partial_conformal_predict_observed(&model, &pre, &new[..n_obs], &cfg).unwrap();
                assert_eq!(p, r);
            }
        }
    }
}

#[test]
fn fragment_distance_is_length_weighted_sum() {
    // d_prod = Σ λ_j d_j with λ_j ∝ fragment length, checked against direct L² sums.
    let t = grid(21);
    let preds_full: Vec<Vec<f64>> = (0..3)
        .map(|i| t.iter().map(|&s| (3.0 * s + i as f64).sin()).collect())
        .collect();
    let frags = vec![(1usize, 5usize), (10, 18)];
    let obs = PartialObservation::Fragments(frags.clone());
    let preds: Vec<Vec<f64>> = preds_full
        .iter()
        .map(|p| extract_observed(&obs, &t, p))
        .collect();
    let cfg = PartialConformalConfig::default();
    let d = pattern_distances(&obs, &t, &preds, &cfg);
    let total = (t[5] - t[1]) + (t[18] - t[10]);
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let mut want = 0.0;
        for &(a, b) in &frags {
            let w = simpsons_weights(&t[a..=b]);
            let dj = weighted_l2(&preds_full[i][a..=b], &preds_full[j][a..=b], &w);
            want += (t[b] - t[a]) / total * dj;
        }
        assert!((d[i * 3 + j] - want).abs() < 1e-14);
        assert_eq!(d[i * 3 + j], d[j * 3 + i]);
    }
    // Sparse: Euclidean distance of linearly interpolated values.
    let sp = PartialObservation::Sparse(vec![0.125, 0.5, 0.9]);
    let ps: Vec<Vec<f64>> = preds_full
        .iter()
        .map(|p| extract_observed(&sp, &t, p))
        .collect();
    assert!((ps[0][1] - preds_full[0][10]).abs() < 1e-15);
    let lin = 0.5 * (preds_full[0][2] + preds_full[0][3]);
    assert!((ps[0][0] - lin).abs() < 1e-14);
    let ds = pattern_distances(&sp, &t, &ps, &cfg);
    let want: f64 = ps[0]
        .iter()
        .zip(&ps[1])
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((ds[1] - want).abs() < 1e-15);
}

// ─── Monte-Carlo coverage of the new options ────────────────────────────────

#[test]
fn monte_carlo_simultaneous_and_sparse_coverage() {
    let t = grid(31);
    let m = t.len();
    let tmpl = template(&t);
    let reps = 200;
    let n2 = 19;
    let alpha = 0.2;
    let sigma: Vec<f64> = t.iter().map(|&s| 0.2 + base(1.0, 1.0, s)).collect();
    let (mut sim_const, mut sim_mod, mut sparse_pw) = (0usize, 0usize, 0.0f64);
    let mut rng = StdRng::seed_from_u64(7272);
    for _ in 0..reps {
        let data = simulate(n2 + 1, &t, &mut rng);
        let cal = rows_except(&data, n2);
        let m_obs = rng.gen_range(8..m);
        let truth = elastic_align_pair(&tmpl, &data.row(n2), &t, 0.0);
        let mut cfg = PartialConformalConfig::default();
        cfg.alpha = alpha;
        cfg.n_phase_radii = 1;
        let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
        cfg.band = PartialAmplitudeBand::Simultaneous;
        let obs = prefix(&data.row(n2), m_obs);
        let p = partial_conformal_predict(&model, &obs, &cfg).unwrap();
        sim_const += usize::from(whole_curve_covered(&p, &truth.f_aligned));
        cfg.band_scale = PartialBandScale::Custom(sigma.clone());
        let p = partial_conformal_predict(&model, &obs, &cfg).unwrap();
        sim_mod += usize::from(whole_curve_covered(&p, &truth.f_aligned));
        // Sparse pattern at random (curve-independent) times, pointwise band.
        let mut times: Vec<f64> = (0..4).map(|_| rng.gen_range(0.0..1.0)).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sp = PartialObservation::Sparse(times);
        let vals = extract_observed(&sp, &t, &data.row(n2));
        let mut cfg2 = PartialConformalConfig::default();
        cfg2.alpha = alpha;
        cfg2.n_phase_radii = 1;
        let p = partial_conformal_predict_observed(&model, &sp, &vals, &cfg2).unwrap();
        let cov = (0..m)
            .filter(|&j| {
                let v = truth.f_aligned[j];
                p.amplitude_lower[j] <= v && v <= p.amplitude_upper[j]
            })
            .count();
        sparse_pw += cov as f64 / m as f64;
    }
    let target = 1.0 - alpha;
    let sd = (target * alpha / reps as f64).sqrt();
    let upper = target + 1.0 / (n2 + 1) as f64;
    let sc = sim_const as f64 / reps as f64;
    let sm = sim_mod as f64 / reps as f64;
    let spw = sparse_pw / reps as f64;
    eprintln!("simultaneous const {sc}, modulated {sm}, sparse pointwise {spw}");
    // Simultaneous bands are a (slightly conservative) superset of the conformal set.
    assert!(
        sc >= target - 3.0 * sd && sc <= upper + 0.05 + 3.0 * sd,
        "sim const {sc}"
    );
    assert!(
        sm >= target - 3.0 * sd && sm <= upper + 0.05 + 3.0 * sd,
        "sim mod {sm}"
    );
    assert!(
        spw >= target - 3.0 * sd && spw <= upper + 3.0 * sd,
        "sparse {spw}"
    );
}

// ─── Held-out bandwidth tuning ──────────────────────────────────────────────

#[test]
fn bandwidth_selection_matches_brute_force_leave_one_out() {
    // Independent re-computation of the tuning criterion through the public API:
    // for every candidate h, refit on T \ {τ} and predict τ with bandwidth h.
    let t = grid(25);
    let m = t.len();
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(31);
    let tune = simulate(12, &t, &mut rng);
    for band in [
        PartialAmplitudeBand::Pointwise,
        PartialAmplitudeBand::Simultaneous,
    ] {
        let mut cfg = PartialConformalConfig::default();
        cfg.alpha = 0.25;
        cfg.n_phase_radii = 1;
        cfg.band = band;
        let tm = partial_conformal_fit_with_template(&tune, &t, &tmpl, &cfg).unwrap();
        let obs = PartialObservation::Prefix(11);
        let sel = partial_conformal_select_bandwidth(&tm, &obs, &cfg).unwrap();
        assert_eq!(sel.n_tuning, 12);
        assert_eq!(sel.candidates.len(), sel.mean_width.len());
        assert!(sel.candidates.len() >= 3);
        assert!(sel.candidates.windows(2).all(|w| w[1] > w[0]));
        for (c, &h) in sel.candidates.iter().enumerate() {
            let mut fixed = cfg.clone();
            fixed.bandwidth = Some(h);
            let mut total = 0.0;
            for tau in 0..12 {
                let mt = partial_conformal_fit_with_template(
                    &rows_except(&tune, tau),
                    &t,
                    &tmpl,
                    &fixed,
                )
                .unwrap();
                let p =
                    partial_conformal_predict(&mt, &prefix(&tune.row(tau), 11), &fixed).unwrap();
                total += (0..m)
                    .map(|j| p.amplitude_upper[j] - p.amplitude_lower[j])
                    .sum::<f64>()
                    / m as f64;
            }
            let want = total / 12.0;
            assert!(
                (sel.mean_width[c] - want).abs() < 1e-9 * (1.0 + want.abs()),
                "{band:?} h={h}: {} vs {want}",
                sel.mean_width[c]
            );
        }
        let best = sel.mean_width.iter().copied().fold(f64::INFINITY, f64::min);
        let idx = sel
            .candidates
            .iter()
            .position(|&h| h == sel.bandwidth)
            .unwrap();
        assert_eq!(sel.mean_width[idx], best);
    }
}

#[test]
fn tuning_fraction_splits_and_selects() {
    let t = grid(25);
    let mut rng = StdRng::seed_from_u64(12);
    let data = simulate(40, &t, &mut rng);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.25;
    cfg.n_phase_radii = 1;
    cfg.karcher_max_iter = 5;
    cfg.tuning_fraction = Some(0.4);
    let model = partial_conformal_fit(&data, &t, &cfg).unwrap();
    let base = {
        let mut c = cfg.clone();
        c.tuning_fraction = None;
        partial_conformal_fit(&data, &t, &c).unwrap()
    };
    // Tuning ∪ calibration = the untuned calibration set, disjointly.
    let mut all: Vec<usize> = model
        .tuning_indices
        .iter()
        .chain(&model.calibration_indices)
        .copied()
        .collect();
    all.sort_unstable();
    let mut want = base.calibration_indices.clone();
    want.sort_unstable();
    assert_eq!(all, want);
    assert_eq!(model.tuning_indices.len(), 8);
    assert_eq!(model.template, base.template);
    for (r, &i) in model.calibration_indices.iter().enumerate() {
        assert_eq!(model.calibration_data.row(r), data.row(i));
    }
    let obs = prefix(&data.row(0), 12);
    let p = partial_conformal_predict(&model, &obs, &cfg).unwrap();
    let sel = p.bandwidth_selection.as_ref().expect("tuned");
    assert_eq!(sel.bandwidth, p.bandwidth);
    // Same selection as the public selector on a model holding only the tuning curves.
    let tm = partial_conformal_fit_with_template(&model.tuning_data, &t, &model.template, &{
        let mut c = cfg.clone();
        c.tuning_fraction = None;
        c
    })
    .unwrap();
    let direct =
        partial_conformal_select_bandwidth(&tm, &PartialObservation::Prefix(12), &cfg).unwrap();
    assert_eq!(&direct, sel);
    // An explicit bandwidth takes precedence over tuning.
    let mut fixed = cfg.clone();
    fixed.bandwidth = Some(0.37);
    let q = partial_conformal_predict(&model, &obs, &fixed).unwrap();
    assert!(q.bandwidth_selection.is_none());
    assert_eq!(q.bandwidth, 0.37);
}

#[test]
fn monte_carlo_coverage_with_tuned_bandwidth() {
    let t = grid(25);
    let m = t.len();
    let tmpl = template(&t);
    let reps = 150;
    let alpha = 0.2;
    let (mut pw, mut sim) = (0.0f64, 0usize);
    let mut rng = StdRng::seed_from_u64(9090);
    let mut n2 = 0;
    for rep in 0..reps {
        let data = simulate(31, &t, &mut rng);
        let cal = rows_except(&data, 30);
        let truth = elastic_align_pair(&tmpl, &data.row(30), &t, 0.0);
        let m_obs = rng.gen_range(6..m);
        let mut cfg = PartialConformalConfig::default();
        cfg.alpha = alpha;
        cfg.n_phase_radii = 1;
        cfg.tuning_fraction = Some(0.35);
        cfg.seed = rep as u64;
        let model = partial_conformal_fit_with_template(&cal, &t, &tmpl, &cfg).unwrap();
        n2 = model.calibration_data.nrows();
        let obs = prefix(&data.row(30), m_obs);
        let p = partial_conformal_predict(&model, &obs, &cfg).unwrap();
        assert!(p.bandwidth_selection.is_some());
        pw += (0..m)
            .filter(|&j| {
                let v = truth.f_aligned[j];
                p.amplitude_lower[j] <= v && v <= p.amplitude_upper[j]
            })
            .count() as f64
            / m as f64;
        cfg.band = PartialAmplitudeBand::Simultaneous;
        let p = partial_conformal_predict(&model, &obs, &cfg).unwrap();
        sim += usize::from(whole_curve_covered(&p, &truth.f_aligned));
    }
    let target = 1.0 - alpha;
    let sd = (target * alpha / reps as f64).sqrt();
    let upper = target + 1.0 / (n2 + 1) as f64;
    let pw = pw / reps as f64;
    let sim = sim as f64 / reps as f64;
    eprintln!("tuned: pointwise {pw}, simultaneous {sim} (n2 = {n2})");
    assert!(
        pw >= target - 3.0 * sd && pw <= upper + 3.0 * sd,
        "pointwise {pw}"
    );
    assert!(
        sim >= target - 3.0 * sd && sim <= upper + 0.05 + 3.0 * sd,
        "simultaneous {sim}"
    );
}

#[test]
fn training_sd_scale_and_new_option_validation() {
    let t = grid(21);
    let tmpl = template(&t);
    let mut rng = StdRng::seed_from_u64(2);
    let data = simulate(24, &t, &mut rng);
    let mut cfg = PartialConformalConfig::default();
    cfg.alpha = 0.3;
    cfg.n_phase_radii = 1;
    cfg.karcher_max_iter = 5;
    cfg.band = PartialAmplitudeBand::Simultaneous;
    cfg.band_scale = PartialBandScale::TrainingSd;
    let model = partial_conformal_fit(&data, &t, &cfg).unwrap();
    let sd = model.training_sd.clone().unwrap();
    let p = partial_conformal_predict(&model, &prefix(&data.row(0), 10), &cfg).unwrap();
    let r = p.amplitude_radius.unwrap();
    let max = sd.iter().copied().fold(0.0_f64, f64::max);
    for j in 0..t.len() {
        let s = sd[j].max(0.05 * max);
        assert!((p.amplitude_upper[j] - p.amplitude_center[j] - r * s).abs() < 1e-12);
    }
    // TrainingSd is unavailable without a training split.
    let cal = partial_conformal_fit_with_template(&data, &t, &tmpl, &cfg).unwrap();
    assert!(partial_conformal_predict(&cal, &prefix(&data.row(0), 10), &cfg).is_err());
    let mut c = cfg.clone();
    c.band_scale = PartialBandScale::Custom(vec![1.0; 5]);
    assert!(partial_conformal_predict(&model, &prefix(&data.row(0), 10), &c).is_err());
    c.band_scale = PartialBandScale::Custom(vec![0.0; t.len()]);
    assert!(partial_conformal_predict(&model, &prefix(&data.row(0), 10), &c).is_err());
    // Pointwise bands ignore the scale entirely.
    c.band = PartialAmplitudeBand::Pointwise;
    assert!(partial_conformal_predict(&model, &prefix(&data.row(0), 10), &c).is_ok());
    let mut c = cfg.clone();
    c.tuning_fraction = Some(1.0);
    assert!(partial_conformal_fit(&data, &t, &c).is_err());
    c.tuning_fraction = Some(0.3);
    c.bandwidth_candidates = vec![0.1, -1.0];
    assert!(partial_conformal_fit(&data, &t, &c).is_err());
    // Observation patterns.
    let ok = {
        let mut c = cfg.clone();
        c.band = PartialAmplitudeBand::Pointwise;
        c
    };
    let row = data.row(1);
    let bad = |obs: PartialObservation, vals: &[f64]| {
        partial_conformal_predict_observed(&model, &obs, vals, &ok).is_err()
    };
    assert!(bad(PartialObservation::Fragments(vec![]), &[]));
    assert!(bad(PartialObservation::Fragments(vec![(3, 3)]), &row[3..4]));
    assert!(bad(
        PartialObservation::Fragments(vec![(2, 6), (6, 9)]),
        &row[2..11]
    ));
    assert!(bad(
        PartialObservation::Fragments(vec![(15, 21)]),
        &row[15..21]
    ));
    assert!(bad(PartialObservation::Fragments(vec![(2, 6)]), &row[2..6]));
    assert!(bad(PartialObservation::Sparse(vec![0.5]), &[0.0]));
    assert!(bad(PartialObservation::Sparse(vec![0.5, 0.4]), &[0.0, 0.0]));
    assert!(bad(PartialObservation::Sparse(vec![0.5, 1.2]), &[0.0, 0.0]));
    assert!(bad(
        PartialObservation::Sparse(vec![0.1, 0.5]),
        &[0.0, f64::NAN]
    ));
    assert!(bad(PartialObservation::Prefix(5), &row[..4]));
    // Elastic needs >= 3 points per fragment.
    let mut el = ok.clone();
    el.distance = PartialPredictorDistance::Elastic;
    assert!(partial_conformal_predict_observed(
        &model,
        &PartialObservation::Fragments(vec![(2, 3)]),
        &row[2..4],
        &el
    )
    .is_err());
    // Tuning needs enough curves for alpha.
    let tiny = partial_conformal_fit_with_template(&select_rows(&data, &[0, 1, 2]), &t, &tmpl, &ok)
        .unwrap();
    assert!(
        partial_conformal_select_bandwidth(&tiny, &PartialObservation::Prefix(8), &ok).is_err()
    );
}
