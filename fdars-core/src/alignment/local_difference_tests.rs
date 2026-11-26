use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

fn gauss(t: f64, c: f64, s: f64) -> f64 {
    (-((t - c) / s).powi(2) / 2.0).exp()
}

/// Two-bump template; `extra` raises the second bump (amplitude effect).
fn template(t: f64, extra: f64) -> f64 {
    gauss(t, 0.3, 0.07) + (0.8 + extra) * gauss(t, 0.68, 0.09)
}

/// Individual nuisance warp `(e^{st} − 1)/(e^s − 1)`.
fn exp_warp(s: f64, t: f64) -> f64 {
    if s.abs() < 1e-12 {
        t
    } else {
        ((s * t).exp() - 1.0) / (s.exp() - 1.0)
    }
}

/// Localised time map `w(t) = t − δ·φ(t)` with `φ` a smooth bump on
/// `[0.45, 0.95]` (monotone for small δ). A curve `μ(w(t))` shows the
/// template feature at 0.68 *later* by ≈ δ, while the first feature is
/// untouched.
fn local_shift(delta: f64, t: f64) -> f64 {
    let (a, b) = (0.45, 0.95);
    if t <= a || t >= b {
        return t;
    }
    let u = (t - a) / (b - a);
    t - delta * (std::f64::consts::PI * u).sin().powi(2)
}

/// Groups A (label 0) and B (label 1), `n_per` curves each. B gets
/// `amp_extra` on the second bump and a localised delay `delta` of it.
fn simulate(
    seed: u64,
    n_per: usize,
    m: usize,
    amp_extra: f64,
    delta: f64,
) -> (FdMatrix, Vec<usize>) {
    let t = grid(m);
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 2 * n_per;
    let mut data = FdMatrix::zeros(n, m);
    let labels: Vec<usize> = (0..n).map(|i| usize::from(i >= n_per)).collect();
    for i in 0..n {
        let b = labels[i] == 1;
        let s: f64 = rng.gen_range(-0.2..0.2);
        let scale: f64 = rng.gen_range(0.9..1.1);
        for j in 0..m {
            let mut x = exp_warp(s, t[j]);
            if b {
                x = local_shift(delta, x);
            }
            let e: f64 = rng.gen_range(-0.08..0.08);
            data[(i, j)] = scale * template(x, if b { amp_extra } else { 0.0 }) + e;
        }
    }
    (data, labels)
}

fn quick_cfg() -> LocalDifferenceConfig {
    LocalDifferenceConfig {
        n_perm: 999,
        max_iter: 8,
        ..LocalDifferenceConfig::default()
    }
}

fn covers(regions: &[DifferenceRegion], t: f64) -> bool {
    regions.iter().any(|r| r.t_start <= t && t <= r.t_end)
}

#[test]
fn amplitude_bump_only_gives_amplitude_region_and_no_phase_region() {
    let m = 60;
    let (data, labels) = simulate(761, 20, m, 0.6, 0.0);
    let t = grid(m);
    let res = elastic_local_difference_test(&data, &t, &labels, &quick_cfg()).unwrap();
    let amp = &res.amplitude.regions;
    assert!(!amp.is_empty(), "amplitude bump not found");
    assert!(covers(amp, 0.68), "{amp:?}");
    for r in amp {
        // B's bump is higher: A − B < 0, located on the second bump.
        assert_eq!(r.direction, SignificanceDirection::Negative, "{r:?}");
        // ITP localisation is at the resolution of the basis supports
        // (knot spacing 0.1 here), so allow one support width of slack.
        assert!(r.t_start >= 0.3, "{r:?}");
    }
    assert!(
        res.phase.regions.is_empty(),
        "spurious phase regions {:?}",
        res.phase.regions
    );
}

#[test]
fn timing_shift_only_gives_phase_region_and_no_amplitude_region() {
    let m = 60;
    let (data, labels) = simulate(762, 20, m, 0.0, 0.1);
    let t = grid(m);
    let res = elastic_local_difference_test(&data, &t, &labels, &quick_cfg()).unwrap();
    let ph = &res.phase.regions;
    assert!(!ph.is_empty(), "timing shift not found");
    assert!(covers(ph, 0.68), "{ph:?}");
    for r in ph {
        // B's second feature is later, so A is earlier there.
        assert_eq!(LocalDifferenceTest::timing(r), Timing::AEarlier, "{r:?}");
        assert!(r.t_start >= 0.35, "{r:?}");
    }
    assert!(
        res.amplitude.regions.is_empty(),
        "spurious amplitude regions {:?}",
        res.amplitude.regions
    );
}

#[test]
fn scb_option_and_psi_representation_localise_too() {
    let m = 60;
    let t = grid(m);
    let mut cfg = quick_cfg();
    cfg.method = LocalTestMethod::Scb;
    cfg.scb_nb = 400;
    let (data, labels) = simulate(763, 20, m, 0.6, 0.0);
    let res = elastic_local_difference_test(&data, &t, &labels, &cfg).unwrap();
    assert!(
        covers(&res.amplitude.regions, 0.68),
        "{:?}",
        res.amplitude.regions
    );
    assert!(res.amplitude.lower.is_some() && res.amplitude.itp.is_none());
    assert!(res.phase.regions.is_empty(), "{:?}", res.phase.regions);

    let mut cfg = quick_cfg();
    cfg.phase_representation = PhaseRepresentation::Psi;
    let (data, labels) = simulate(764, 20, m, 0.0, 0.1);
    let res = elastic_local_difference_test(&data, &t, &labels, &cfg).unwrap();
    assert!(
        !res.phase.regions.is_empty(),
        "psi phase test found nothing"
    );
    assert!(res.phase.regions.iter().all(|r| r.t_start >= 0.35));
    // ψ of a warp on [0, 1] has unit L² norm.
    let w = crate::helpers::simpsons_weights(&t);
    let norm2: f64 = (0..m).map(|j| w[j] * res.phase_data[(0, j)].powi(2)).sum();
    assert!((norm2 - 1.0).abs() < 0.05, "{norm2}");
}

/// The alignment is label-free: permuting the labels leaves it unchanged.
#[test]
fn alignment_is_label_free() {
    let m = 40;
    let t = grid(m);
    let (data, labels) = simulate(765, 6, m, 0.5, 0.0);
    let mut cfg = quick_cfg();
    cfg.n_perm = 49;
    cfg.max_iter = 3;
    let r1 = elastic_local_difference_test(&data, &t, &labels, &cfg).unwrap();
    let flipped: Vec<usize> = labels.iter().map(|&l| 7 * (1 - l)).collect();
    let r2 = elastic_local_difference_test(&data, &t, &flipped, &cfg).unwrap();
    assert_eq!(r1.alignment, r2.alignment);
    assert_eq!(r2.group_labels, [0, 7]);
    // Effects flip sign with the group roles.
    for j in 0..m {
        assert!((r1.amplitude.effect[j] + r2.amplitude.effect[j]).abs() < 1e-12);
    }
    // Displacement phase data is γ − t.
    for j in 0..m {
        let d = r1.alignment.gammas[(0, j)] - t[j];
        assert!((r1.phase_data[(0, j)] - d).abs() < 1e-15);
    }
}

#[test]
fn input_validation() {
    let m = 20;
    let t = grid(m);
    let (data, labels) = simulate(1, 4, m, 0.0, 0.0);
    let cfg = quick_cfg();
    assert!(elastic_local_difference_test(&data, &t[..m - 1], &labels, &cfg).is_err());
    assert!(elastic_local_difference_test(&data, &t, &labels[1..], &cfg).is_err());
    let three: Vec<usize> = (0..8).map(|i| i % 3).collect();
    assert!(elastic_local_difference_test(&data, &t, &three, &cfg).is_err());
    let one = vec![0usize; 8];
    assert!(elastic_local_difference_test(&data, &t, &one, &cfg).is_err());
    let lonely: Vec<usize> = (0..8).map(|i| usize::from(i == 0)).collect();
    assert!(elastic_local_difference_test(&data, &t, &lonely, &cfg).is_err());
    let mut bad = cfg.clone();
    bad.alpha = 0.0;
    assert!(elastic_local_difference_test(&data, &t, &labels, &bad).is_err());
    let mut bad = cfg.clone();
    bad.scb_bandwidth = Some(-1.0);
    assert!(elastic_local_difference_test(&data, &t, &labels, &bad).is_err());
    let mut nan = data.clone();
    nan[(0, 0)] = f64::NAN;
    assert!(elastic_local_difference_test(&nan, &t, &labels, &cfg).is_err());
}

/// Null study: no group difference → the probability that *any* region is
/// flagged (family-wise over the domain; for the ITP this is bounded by the
/// full-domain interval test) should not exceed α beyond binomial error.
fn null_rejections(reps: u64, n_per: usize, m: usize, method: LocalTestMethod) -> (usize, usize) {
    let t = grid(m);
    let mut cfg = LocalDifferenceConfig {
        method,
        scb_nb: 299,
        n_perm: 199,
        max_iter: 5,
        ..LocalDifferenceConfig::default()
    };
    let (mut amp, mut ph) = (0, 0);
    for r in 0..reps {
        let (data, labels) = simulate(10_000 + r, n_per, m, 0.0, 0.0);
        cfg.seed = 100 + r;
        let res = elastic_local_difference_test(&data, &t, &labels, &cfg).unwrap();
        amp += usize::from(!res.amplitude.regions.is_empty());
        ph += usize::from(!res.phase.regions.is_empty());
    }
    (amp, ph)
}

#[test]
fn null_study_short() {
    // 20 replicates: P(Bin(20, 0.05) >= 5) ≈ 0.003.
    let (amp, ph) = null_rejections(20, 8, 40, LocalTestMethod::Itp);
    assert!(
        amp <= 4 && ph <= 4,
        "null rejections: amplitude {amp}/20, phase {ph}/20"
    );
}

#[test]
#[ignore = "long Monte-Carlo study (2 x 200 replicates, ~80 s debug); run with --ignored"]
fn null_study_long() {
    // 200 replicates at α = 0.05: mean 10, 3-sigma upper bound ≈ 19.
    // Recorded: ITP 0/200 and 0/200, SCB 6/200 (amplitude) and 5/200 (phase).
    for method in [LocalTestMethod::Itp, LocalTestMethod::Scb] {
        let (amp, ph) = null_rejections(200, 30, 50, method);
        eprintln!("{method:?} null family-wise rejections: amplitude {amp}/200, phase {ph}/200");
        assert!(
            amp <= 19 && ph <= 19,
            "{method:?}: amplitude {amp}/200, phase {ph}/200"
        );
    }
}
