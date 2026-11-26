// Sealed-config idiom (`default()` + field assignment) mirrors external usage.
#![allow(clippy::field_reassign_with_default)]

use super::*;
use std::f64::consts::PI;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Two-bump template of the golden script (decays to ~0 at both ends).
fn mu0(s: f64) -> f64 {
    (-((s - 0.45) / 0.12).powi(2)).exp() + 0.6 * (-((s - 0.7) / 0.08).powi(2)).exp()
}

const A: [f64; 6] = [1.05, 0.95, 1.0, 1.08, 0.93, 1.02];
const B: [f64; 6] = [-0.03, 0.04, 0.0, -0.06, 0.05, -0.01];
const AMP: [f64; 6] = [1.0, 1.2, 0.9, 1.1, 0.8, 1.05];

fn make(n: usize, m: usize, f: impl Fn(usize, f64) -> f64) -> (FdMatrix, Vec<f64>) {
    let t = grid(m);
    let mut x = FdMatrix::zeros(n, m);
    for i in 0..n {
        for (j, &tj) in t.iter().enumerate() {
            x[(i, j)] = f(i, tj);
        }
    }
    (x, t)
}

/// Frame-invariant relative warps `h_r⁻¹ ∘ hᵢ = (aᵢ/a_r)·t + (bᵢ − b_r)/a_r`.
///
/// Registration fixes `gᵢ = fᵢ ∘ hᵢ⁻¹` only up to a common reparametrisation
/// `gᵢ ↦ gᵢ ∘ K`, i.e. `hᵢ ↦ K⁻¹ ∘ hᵢ`, under which `h_r⁻¹ ∘ hᵢ` is invariant.
fn relative(a: &[f64], b: &[f64], r: usize) -> (Vec<f64>, Vec<f64>) {
    let ra = a.iter().map(|ai| ai / a[r]).collect();
    let rb = b.iter().map(|bi| (bi - b[r]) / a[r]).collect();
    (ra, rb)
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < tol, "{what}[{k}]: {a} vs {e} (tol {tol})");
    }
}

fn cfg(metric: AffineMetric, domain: AffineDomain) -> AffineRegistrationConfig {
    let mut c = AffineRegistrationConfig::default();
    c.metric = metric;
    c.domain = domain;
    c
}

// ---------------------------------------------------------------------------
// Metric / warping primitives (hand-computed)
// ---------------------------------------------------------------------------

#[test]
fn metric_hand_computed() {
    let g = [0.0, 0.5, 1.0];
    // L2: ∫(v1 − v2)² by trapezoid on diff² = [1, 1, 1] → 1.
    assert!(
        (metric_value(AffineMetric::L2, &g, &[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]) - 1.0).abs()
            < 1e-15
    );
    // NormalizedL2: v1 = [1,1,1], v2 = [0,0,0] → 1 / (1 + 0) = 1.
    assert!(
        (metric_value(AffineMetric::NormalizedL2, &g, &[1.0; 3], &[0.0; 3]) - 1.0).abs() < 1e-15
    );
    // Pearson: amplitude-scale invariant and uncentred.
    let v = [0.3, 1.0, 2.0];
    let w: Vec<f64> = v.iter().map(|x| 5.0 * x).collect();
    assert!(metric_value(AffineMetric::Pearson, &g, &v, &w).abs() < 1e-15);
    // Points (1, 0) vs (0, 1) after the first: ‖u − v‖² = (2 − 0)/(m−1)·… → √(1)/2.
    let p = metric_value(
        AffineMetric::Pearson,
        &g,
        &[9.0, 1.0, 0.0],
        &[9.0, 0.0, 1.0],
    );
    // s1 = s2 = 1/2 → u/‖u‖ = (√2, 0), v/‖v‖ = (0, √2): Σd² = 4 → √(4/2)/2.
    assert!((p - (2.0_f64).sqrt() / 2.0).abs() < 1e-15);
    assert_eq!(
        metric_value(
            AffineMetric::Pearson,
            &g,
            &[1.0, 1.0, 1.0],
            &[1.0, 0.0, 0.0]
        ),
        0.5
    );
}

#[test]
fn warp_curves_hand_computed() {
    let t = grid(11);
    let data = FdMatrix::from_column_major(t.clone(), 1, 11).unwrap(); // f(t) = t
    let g = affine_warp_curves(&data, &t, &[0.5], &[0.2], ShiftExtrapolation::Linear).unwrap();
    for (j, &tj) in t.iter().enumerate() {
        assert!((g[(0, j)] - (tj - 0.2) / 0.5).abs() < 1e-12); // f(h⁻¹(t)) = (t − 0.2)/0.5
    }
    let gb = affine_warp_curves(&data, &t, &[0.5], &[0.2], ShiftExtrapolation::Boundary).unwrap();
    assert_eq!(gb[(0, 0)], 0.0);
    assert_eq!(gb[(0, 10)], 1.0);
    assert!(affine_warp_curves(&data, &t, &[0.0], &[0.0], ShiftExtrapolation::Linear).is_err());
    assert!(
        affine_warp_curves(&data, &t, &[1.0, 1.0], &[0.0], ShiftExtrapolation::Linear).is_err()
    );
}

#[test]
fn identity_warp_gives_zero_distance() {
    let (x, t) = make(1, 51, |_, s| mu0(s));
    let c = x.row(0);
    for metric in [
        AffineMetric::L2,
        AffineMetric::NormalizedL2,
        AffineMetric::Pearson,
    ] {
        for domain in [AffineDomain::Full, AffineDomain::Overlap] {
            let d = warp_dissimilarity(&c, &t, &c, Warp::IDENTITY, &cfg(metric, domain));
            assert!(d.abs() < 1e-15, "{metric:?}/{domain:?}: {d}");
        }
    }
}

#[test]
fn center_warps_composes_with_inverse_mean() {
    let mut w: Vec<Warp> = A.iter().zip(&B).map(|(&a, &b)| Warp { a, b }).collect();
    let orig = w.clone();
    center_warps(&mut w, &[0, 1, 2, 3, 4, 5]);
    let ma = w.iter().map(|x| x.a).sum::<f64>() / 6.0;
    let mb = w.iter().map(|x| x.b).sum::<f64>() / 6.0;
    assert!((ma - 1.0).abs() < 1e-14 && mb.abs() < 1e-14);
    // Relative warps are unchanged by centring.
    let (ra0, rb0) = relative(&A, &B, 0);
    let a: Vec<f64> = w.iter().map(|x| x.a).collect();
    let b: Vec<f64> = w.iter().map(|x| x.b).collect();
    let (ra, rb) = relative(&a, &b, 0);
    assert_close(&ra, &ra0, 1e-14, "rel a");
    assert_close(&rb, &rb0, 1e-14, "rel b");
    assert_ne!(w, orig);
}

// ---------------------------------------------------------------------------
// Exact recovery of known synthetic warps
// ---------------------------------------------------------------------------

/// `fᵢ = μ0 ∘ hᵢ`, so registering to the fixed template μ0 recovers `(aᵢ, bᵢ)`.
#[test]
fn recovery_fixed_template_all_classes() {
    let m = 201;
    let t = grid(m);
    let tmpl: Vec<f64> = t.iter().map(|&s| mu0(s)).collect();
    let cases: [(LinearWarpClass, bool, bool); 3] = [
        (LinearWarpClass::Shift, false, true),
        (LinearWarpClass::Dilation, true, false),
        (LinearWarpClass::Affine, true, true),
    ];
    for (class, dil, sh) in cases {
        let a: Vec<f64> = A.iter().map(|&a| if dil { a } else { 1.0 }).collect();
        let b: Vec<f64> = B.iter().map(|&b| if sh { b } else { 0.0 }).collect();
        let (x, _) = make(6, m, |i, s| mu0(a[i] * s + b[i]));
        for (metric, domain) in [
            (AffineMetric::L2, AffineDomain::Full),
            (AffineMetric::Pearson, AffineDomain::Overlap),
            (AffineMetric::NormalizedL2, AffineDomain::Overlap),
        ] {
            let c = cfg(metric, domain);
            let r = affine_registration(&x, &t, class, ShiftTemplate::Fixed(&tmpl), &c).unwrap();
            let what = format!("{class:?}/{metric:?}/{domain:?}");
            assert_close(&r.dilations, &a, 2e-4, &format!("{what} a"));
            assert_close(&r.shifts, &b, 2e-4, &format!("{what} b"));
            assert!(r.converged, "{what}");
            assert!(
                r.distances.iter().all(|&d| d < 1e-3),
                "{what}: {:?}",
                r.distances
            );
            // Registered curves coincide with the template where observed.
            for i in 0..6 {
                for j in 40..160 {
                    assert!((r.registered_data[(i, j)] - tmpl[j]).abs() < 2e-3, "{what}");
                }
            }
        }
    }
}

/// Mean template: warps are recovered up to the common frame, which centring
/// pins to mean dilation 1 / mean shift 0.
#[test]
fn recovery_mean_template_relative_warps() {
    let m = 201;
    let (x, t) = make(6, m, |i, s| mu0(A[i] * s + B[i]));
    for (metric, domain) in [
        (AffineMetric::L2, AffineDomain::Full),
        (AffineMetric::Pearson, AffineDomain::Overlap),
    ] {
        let c = cfg(metric, domain);
        let r =
            affine_registration(&x, &t, LinearWarpClass::Affine, ShiftTemplate::Mean, &c).unwrap();
        let (ra, rb) = relative(&r.dilations, &r.shifts, 0);
        let (ta, tb) = relative(&A, &B, 0);
        assert_close(&ra, &ta, 1e-3, "rel a");
        assert_close(&rb, &tb, 1e-3, "rel b");
        let ma = r.dilations.iter().sum::<f64>() / 6.0;
        let mb = r.shifts.iter().sum::<f64>() / 6.0;
        assert!((ma - 1.0).abs() < 1e-12 && mb.abs() < 1e-12);
        assert!(r.converged);
    }
}

#[test]
fn shift_class_matches_iterative_shift_truth() {
    let m = 201;
    let (x, t) = make(6, m, |i, s| mu0(s - B[i]));
    let c = cfg(AffineMetric::L2, AffineDomain::Full);
    let r = affine_registration(&x, &t, LinearWarpClass::Shift, ShiftTemplate::Mean, &c).unwrap();
    // f_i(t) = μ0(t − Bᵢ) = μ0 ∘ hᵢ with hᵢ(t) = t − Bᵢ; centred.
    let mean_b = B.iter().sum::<f64>() / 6.0;
    let expect: Vec<f64> = B.iter().map(|b| -(b - mean_b)).collect();
    assert_close(&r.shifts, &expect, 2e-4, "shifts");
    assert!(r.dilations.iter().all(|&a| a == 1.0));
}

#[test]
fn aligned_data_is_a_fixed_point() {
    let (x, t) = make(5, 101, |i, s| (1.0 + 0.1 * i as f64) * mu0(s));
    for class in [
        LinearWarpClass::Shift,
        LinearWarpClass::Dilation,
        LinearWarpClass::Affine,
    ] {
        let c = cfg(AffineMetric::Pearson, AffineDomain::Overlap);
        let r = affine_registration(&x, &t, class, ShiftTemplate::Mean, &c).unwrap();
        assert_close(&r.dilations, &[1.0; 5], 1e-6, "a");
        assert_close(&r.shifts, &[0.0; 5], 1e-6, "b");
    }
    // Class None: identity warps, template = mean, registered = data.
    let r = affine_registration(
        &x,
        &t,
        LinearWarpClass::None,
        ShiftTemplate::Mean,
        &AffineRegistrationConfig::default(),
    )
    .unwrap();
    assert_eq!(r.registered_data, x);
    assert_eq!(r.n_iter, 1);
    assert_close(&r.template, &crate::fdata::mean_1d(&x), 1e-14, "mean");
}

#[test]
fn custom_template_and_validation() {
    let (x, t) = make(4, 51, |i, s| mu0(s + 0.02 * i as f64));
    let median = |reg: &FdMatrix| crate::fdata::mean_1d(reg);
    let c = AffineRegistrationConfig::default();
    let r1 = affine_registration(
        &x,
        &t,
        LinearWarpClass::Shift,
        ShiftTemplate::Custom(&median),
        &c,
    )
    .unwrap();
    let r2 = affine_registration(&x, &t, LinearWarpClass::Shift, ShiftTemplate::Mean, &c).unwrap();
    // Full domain: the Mean template is the extrapolated mean = the closure.
    assert_close(&r1.shifts, &r2.shifts, 1e-12, "custom vs mean");

    let mut bad = AffineRegistrationConfig::default();
    bad.domain = AffineDomain::Overlap; // L2 + overlap + affine is rejected
    assert!(
        affine_registration(&x, &t, LinearWarpClass::Affine, ShiftTemplate::Mean, &bad).is_err()
    );
    assert!(affine_registration(&x, &t, LinearWarpClass::Shift, ShiftTemplate::Mean, &bad).is_ok());
    bad = AffineRegistrationConfig::default();
    bad.max_dilation = 1.0;
    assert!(bad.validate(LinearWarpClass::Affine).is_err());
    bad = AffineRegistrationConfig::default();
    bad.grid_size = 2;
    assert!(bad.validate(LinearWarpClass::Shift).is_err());
    let short = [0.0; 3];
    assert!(affine_registration(
        &x,
        &t,
        LinearWarpClass::Shift,
        ShiftTemplate::Fixed(&short),
        &c
    )
    .is_err());
    assert!(affine_registration(
        &x,
        &t[..50],
        LinearWarpClass::Shift,
        ShiftTemplate::Mean,
        &c
    )
    .is_err());
}

#[test]
fn bounds_are_respected_with_fixed_template() {
    // True shift 0.3 exceeds the bound 0.15·L: the fit saturates at the bound.
    let (x, t) = make(1, 101, |_, s| mu0(s + 0.3));
    let tmpl: Vec<f64> = t.iter().map(|&s| mu0(s)).collect();
    let c = AffineRegistrationConfig::default();
    let r = affine_registration(
        &x,
        &t,
        LinearWarpClass::Shift,
        ShiftTemplate::Fixed(&tmpl),
        &c,
    )
    .unwrap();
    assert!((r.shifts[0] - 0.15).abs() < 1e-9, "{}", r.shifts[0]);
}

// ---------------------------------------------------------------------------
// Golden: R fdacluster 0.4.2 (`golden_affine.R` in the quick-task directory)
// ---------------------------------------------------------------------------
//
// fdacluster reports the warped grids; `(aᵢ, bᵢ)` are their exact linear fits
// and are compared through the frame-invariant relative warps hᵢ ∘ h₁⁻¹ (the
// two implementations pin the common frame differently: fdacluster re-centres
// a moving template grid, fdars centres the warps on a fixed grid). Both
// implementations stop on their own tolerances (fdacluster: NLopt BOBYQA
// xtol_rel + distance_relative_tolerance = 1e-3). Observed max deviations of
// the relative warps: affine/pearson 9.0e-4, dilation/pearson 2.2e-4,
// shift/l2 1.1e-4 (both implementations differ from the noise-free truth by up
// to ~3e-3 because of the perturbations), hence tolerances 2e-3 / 5e-4 / 5e-4.

/// Compare the relative warps (w.r.t. curve `r`) of fdars and fdacluster.
pub(crate) fn assert_golden(
    a: &[f64],
    b: &[f64],
    fa: &[f64],
    fb: &[f64],
    r: usize,
    tol: f64,
    what: &str,
) {
    let (ra, rb) = relative(a, b, r);
    let (ga, gb) = relative(fa, fb, r);
    let dev = ra
        .iter()
        .zip(&ga)
        .chain(rb.iter().zip(&gb))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    eprintln!("{what}: max |fdars - fdacluster| relative-warp deviation = {dev:.2e}");
    assert_close(&ra, &ga, tol, &format!("{what} rel a"));
    assert_close(&rb, &gb, tol, &format!("{what} rel b"));
}

fn golden_k1_data(dilation_only: bool) -> (FdMatrix, Vec<f64>) {
    make(6, 101, |i, s| {
        let b = if dilation_only { 0.0 } else { B[i] };
        AMP[i] * mu0(A[i] * s + b) + 0.02 * (2.0 * PI * s + (i + 1) as f64).sin()
    })
}

#[test]
fn golden_fdacluster_k1_affine_pearson() {
    // fdakmeans(x = t, y = y1, n_clusters = 1, seeds = 1, warping_class = "affine",
    //           metric = "pearson")  — fdacluster 0.4.2
    let (x, t) = golden_k1_data(false);
    let c = cfg(AffineMetric::Pearson, AffineDomain::Overlap);
    let r = affine_registration(&x, &t, LinearWarpClass::Affine, ShiftTemplate::Mean, &c).unwrap();
    let fa = [
        1.0437557708,
        0.9446948310,
        0.9995665748,
        1.0799586652,
        0.9341404107,
        1.0167211034,
    ];
    let fb = [
        -0.0268006745,
        0.0410589759,
        -0.0029372107,
        -0.0617396152,
        0.0476827707,
        -0.0073638600,
    ];
    assert_golden(
        &r.dilations,
        &r.shifts,
        &fa,
        &fb,
        0,
        2e-3,
        "k1 affine/pearson",
    );
}

#[test]
fn golden_fdacluster_k1_dilation_pearson() {
    // fdakmeans(..., warping_class = "dilation", metric = "pearson")
    let (x, t) = golden_k1_data(true);
    let c = cfg(AffineMetric::Pearson, AffineDomain::Overlap);
    let r =
        affine_registration(&x, &t, LinearWarpClass::Dilation, ShiftTemplate::Mean, &c).unwrap();
    let fa = [
        1.0469103164,
        0.9435215424,
        0.9915591053,
        1.0725333379,
        0.9283097735,
        1.0184083274,
    ];
    assert_golden(
        &r.dilations,
        &r.shifts,
        &fa,
        &[0.0; 6],
        0,
        5e-4,
        "k1 dilation/pearson",
    );
    assert!(r.shifts.iter().all(|&b| b == 0.0));
}

#[test]
fn golden_fdacluster_k1_shift_l2() {
    // fdakmeans(x = t, y = ys, ..., warping_class = "shift", metric = "l2")
    let (x, t) = make(6, 101, |i, s| {
        mu0(s + B[i]) + 0.02 * (2.0 * PI * s + (i + 1) as f64).sin()
    });
    let c = cfg(AffineMetric::L2, AffineDomain::Overlap);
    let r = affine_registration(&x, &t, LinearWarpClass::Shift, ShiftTemplate::Mean, &c).unwrap();
    let fb = [
        -0.0280048633,
        0.0408888059,
        -0.0005770392,
        -0.0590370900,
        0.0515663577,
        -0.0066098310,
    ];
    assert_golden(
        &r.dilations,
        &r.shifts,
        &[1.0; 6],
        &fb,
        0,
        5e-4,
        "k1 shift/l2",
    );
    assert!(r.dilations.iter().all(|&a| a == 1.0));
}
