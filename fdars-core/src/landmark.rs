//! Landmark-based registration for functional data.
//!
//! Aligns curves by mapping identified features (peaks, valleys, zero-crossings,
//! inflection points) to common target positions. Uses monotone cubic Hermite
//! interpolation (Fritsch-Carlson) to build smooth, monotone warping functions.
//!
//! - [`detect_landmarks`] — Detect features in a single curve
//! - [`landmark_register`] — Register curves given known landmark positions
//! - [`detect_and_register`] — Convenience: detect landmarks then register

use crate::alignment::reparameterize_curve;
use crate::matrix::FdMatrix;
use crate::seasonal::{compute_prominence, find_peaks_1d};

/// Kind of landmark feature to detect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LandmarkKind {
    /// Local maximum.
    Peak,
    /// Local minimum.
    Valley,
    /// Sign change (zero-crossing via linear interpolation).
    ZeroCrossing,
    /// Inflection point (second derivative zero-crossing).
    Inflection,
    /// User-specified landmark (not auto-detected).
    Custom,
}

/// A detected landmark on a curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Landmark {
    /// Position on the domain (t-value).
    pub position: f64,
    /// Kind of feature.
    pub kind: LandmarkKind,
    /// Curve value at the landmark.
    pub value: f64,
    /// Feature strength (prominence for peaks/valleys, 0.0 for others).
    pub prominence: f64,
}

/// Result of landmark registration.
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkResult {
    /// Registered (aligned) curves (n × m).
    pub registered: FdMatrix,
    /// Warping functions (n × m).
    pub gammas: FdMatrix,
    /// Detected landmarks per curve.
    pub landmarks: Vec<Vec<Landmark>>,
    /// Common target landmark positions.
    pub target_landmarks: Vec<f64>,
}

// ─── Landmark Detection ─────────────────────────────────────────────────────

/// Detect landmarks of a given kind in a single curve.
///
/// # Arguments
/// * `curve` — Curve values (length m)
/// * `argvals` — Evaluation points (length m)
/// * `kind` — Type of landmark to detect
/// * `min_prominence` — Minimum prominence to keep a peak/valley (ignored for other kinds)
///
/// # Returns
/// Vector of detected landmarks, sorted by position.
pub fn detect_landmarks(
    curve: &[f64],
    argvals: &[f64],
    kind: LandmarkKind,
    min_prominence: f64,
) -> Vec<Landmark> {
    let m = curve.len();
    if m < 3 || argvals.len() != m {
        return Vec::new();
    }

    match kind {
        LandmarkKind::Peak => detect_peaks(curve, argvals, min_prominence),
        LandmarkKind::Valley => detect_valleys(curve, argvals, min_prominence),
        LandmarkKind::ZeroCrossing => detect_zero_crossings(curve, argvals),
        LandmarkKind::Inflection => detect_inflections(curve, argvals),
        LandmarkKind::Custom => Vec::new(), // Custom landmarks are user-supplied
    }
}

fn detect_peaks(curve: &[f64], argvals: &[f64], min_prominence: f64) -> Vec<Landmark> {
    let peak_indices = find_peaks_1d(curve, 1);
    peak_indices
        .into_iter()
        .filter_map(|idx| {
            let prom = compute_prominence(curve, idx);
            if prom >= min_prominence {
                Some(Landmark {
                    position: argvals[idx],
                    kind: LandmarkKind::Peak,
                    value: curve[idx],
                    prominence: prom,
                })
            } else {
                None
            }
        })
        .collect()
}

fn detect_valleys(curve: &[f64], argvals: &[f64], min_prominence: f64) -> Vec<Landmark> {
    // Negate curve, detect peaks, then negate back
    let negated: Vec<f64> = curve.iter().map(|&v| -v).collect();
    let peak_indices = find_peaks_1d(&negated, 1);
    peak_indices
        .into_iter()
        .filter_map(|idx| {
            let prom = compute_prominence(&negated, idx);
            if prom >= min_prominence {
                Some(Landmark {
                    position: argvals[idx],
                    kind: LandmarkKind::Valley,
                    value: curve[idx],
                    prominence: prom,
                })
            } else {
                None
            }
        })
        .collect()
}

fn detect_zero_crossings(curve: &[f64], argvals: &[f64]) -> Vec<Landmark> {
    let m = curve.len();
    let mut landmarks = Vec::new();
    for i in 0..(m - 1) {
        if curve[i] * curve[i + 1] < 0.0 {
            // Linear interpolation to find exact zero crossing
            let frac = curve[i].abs() / (curve[i].abs() + curve[i + 1].abs());
            let t = argvals[i] + frac * (argvals[i + 1] - argvals[i]);
            landmarks.push(Landmark {
                position: t,
                kind: LandmarkKind::ZeroCrossing,
                value: 0.0,
                prominence: 0.0,
            });
        }
    }
    landmarks
}

fn detect_inflections(curve: &[f64], argvals: &[f64]) -> Vec<Landmark> {
    let m = curve.len();
    if m < 4 {
        return Vec::new();
    }
    // Compute second derivative via central differences
    let mut d2 = vec![0.0; m];
    for i in 1..(m - 1) {
        let h1 = argvals[i] - argvals[i - 1];
        let h2 = argvals[i + 1] - argvals[i];
        d2[i] = 2.0
            * (curve[i + 1] / (h2 * (h1 + h2)) - curve[i] / (h1 * h2)
                + curve[i - 1] / (h1 * (h1 + h2)));
    }
    // Find zero crossings of second derivative
    let mut landmarks = Vec::new();
    for i in 1..(m - 2) {
        if d2[i] * d2[i + 1] < 0.0 {
            let frac = d2[i].abs() / (d2[i].abs() + d2[i + 1].abs());
            let t = argvals[i] + frac * (argvals[i + 1] - argvals[i]);
            // Interpolate curve value at the inflection point
            let val = curve[i]
                + (curve[i + 1] - curve[i]) * (t - argvals[i]) / (argvals[i + 1] - argvals[i]);
            landmarks.push(Landmark {
                position: t,
                kind: LandmarkKind::Inflection,
                value: val,
                prominence: 0.0,
            });
        }
    }
    landmarks
}

// ─── Monotone Cubic Hermite Interpolation (Fritsch-Carlson) ─────────────────

/// Fritsch-Carlson monotonicity adjustment of tangent slopes.
///
/// For each interval, if the secant is near-zero both tangents are zeroed;
/// otherwise clamps (α²+β²) ≤ 9 to guarantee monotonicity.
fn fritsch_carlson_fix(delta: &[f64], d: &mut [f64]) {
    for i in 0..delta.len() {
        if delta[i].abs() < 1e-15 {
            d[i] = 0.0;
            d[i + 1] = 0.0;
        } else {
            let alpha = d[i] / delta[i];
            let beta = d[i + 1] / delta[i];
            let s2 = alpha * alpha + beta * beta;
            if s2 > 9.0 {
                let tau = 3.0 / s2.sqrt();
                d[i] = tau * alpha * delta[i];
                d[i + 1] = tau * beta * delta[i];
            }
        }
    }
}

/// Evaluate the piecewise cubic Hermite interpolant at a single point.
///
/// Binary-searches `x_knots` for the interval, then evaluates
/// the cubic Hermite basis with tangent slopes `d`.
fn hermite_eval(t: f64, x_knots: &[f64], y_knots: &[f64], d: &[f64]) -> f64 {
    let k = x_knots.len();
    let seg = if t <= x_knots[0] {
        0
    } else if t >= x_knots[k - 1] {
        k - 2
    } else {
        match x_knots.binary_search_by(|v| v.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => i.min(k - 2),
            Err(i) => (i - 1).min(k - 2),
        }
    };

    let h = x_knots[seg + 1] - x_knots[seg];
    if h.abs() < 1e-15 {
        return y_knots[seg];
    }

    let s = (t - x_knots[seg]) / h;
    let h00 = (1.0 + 2.0 * s) * (1.0 - s) * (1.0 - s);
    let h10 = s * (1.0 - s) * (1.0 - s);
    let h01 = s * s * (3.0 - 2.0 * s);
    let h11 = s * s * (s - 1.0);
    h00 * y_knots[seg] + h10 * h * d[seg] + h01 * y_knots[seg + 1] + h11 * h * d[seg + 1]
}

/// Build knot sequences from source/target landmarks, adding domain endpoints.
///
/// Returns `(x_knots, y_knots)` where `x_knots` = target positions, `y_knots` = source positions.
fn build_hermite_knots(
    source: &[f64],
    target: &[f64],
    t0: f64,
    t_end: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut x_knots = Vec::with_capacity(target.len() + 2);
    let mut y_knots = Vec::with_capacity(source.len() + 2);
    x_knots.push(t0);
    y_knots.push(t0);
    for (&s, &t) in source.iter().zip(target.iter()) {
        if t > t0 && t < t_end {
            x_knots.push(t);
            y_knots.push(s);
        }
    }
    x_knots.push(t_end);
    y_knots.push(t_end);
    (x_knots, y_knots)
}

/// Compute secant slopes and initial tangent slopes for Hermite interpolation.
fn hermite_tangents(x_knots: &[f64], y_knots: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = x_knots.len();
    let mut delta = vec![0.0; k - 1];
    for i in 0..(k - 1) {
        let dx = x_knots[i + 1] - x_knots[i];
        delta[i] = if dx.abs() < 1e-15 {
            0.0
        } else {
            (y_knots[i + 1] - y_knots[i]) / dx
        };
    }

    let mut d = vec![0.0; k];
    if k == 2 {
        d[0] = delta[0];
        d[1] = delta[0];
    } else {
        d[0] = delta[0];
        d[k - 1] = delta[k - 2];
        for i in 1..(k - 1) {
            d[i] = (delta[i - 1] + delta[i]) / 2.0;
        }
    }

    fritsch_carlson_fix(&delta, &mut d);
    (delta, d)
}

/// Build a monotone warping function that maps source landmarks to target landmarks.
///
/// Uses Fritsch-Carlson monotone cubic Hermite interpolation to ensure the
/// resulting warping function is strictly monotone (a valid diffeomorphism).
fn monotone_landmark_warp(source: &[f64], target: &[f64], argvals: &[f64]) -> Vec<f64> {
    let m = argvals.len();
    if source.is_empty() || source.len() != target.len() {
        return argvals.to_vec();
    }

    let t0 = argvals[0];
    let t_end = argvals[m - 1];

    let (x_knots, y_knots) = build_hermite_knots(source, target, t0, t_end);
    if x_knots.len() < 2 {
        return argvals.to_vec();
    }

    let (_delta, d) = hermite_tangents(&x_knots, &y_knots);

    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| hermite_eval(t, &x_knots, &y_knots, &d))
        .collect();

    // Post-process: clamp to domain and enforce monotonicity
    gamma[0] = t0;
    gamma[m - 1] = t_end;
    for i in 1..m {
        gamma[i] = gamma[i].clamp(t0, t_end);
        if gamma[i] < gamma[i - 1] {
            gamma[i] = gamma[i - 1];
        }
    }

    gamma
}

// ─── Registration ───────────────────────────────────────────────────────────

/// Compute target landmark positions: use provided values, or mean of per-curve landmarks.
fn compute_target_landmarks(landmarks: &[Vec<f64>], target: Option<&[f64]>, n: usize) -> Vec<f64> {
    if let Some(t) = target {
        return t.to_vec();
    }
    let min_count = landmarks.iter().map(std::vec::Vec::len).min().unwrap_or(0);
    if min_count == 0 {
        return Vec::new();
    }
    let mut mean_pos = vec![0.0; min_count];
    for lm in landmarks {
        for (j, &pos) in lm.iter().take(min_count).enumerate() {
            mean_pos[j] += pos;
        }
    }
    for v in &mut mean_pos {
        *v /= n as f64;
    }
    mean_pos
}

/// Register curves using known landmark positions.
///
/// Builds monotone warping functions that map each curve's landmarks to common
/// target positions, then reparameterizes the curves.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `landmarks` — Per-curve landmark positions (n vectors, each with K positions)
/// * `target` — Target landmark positions (K values). If `None`, uses mean positions.
///
/// # Returns
/// [`LandmarkResult`] containing registered curves, warps, and landmark info.
pub fn landmark_register(
    data: &FdMatrix,
    argvals: &[f64],
    landmarks: &[Vec<f64>],
    target: Option<&[f64]>,
) -> LandmarkResult {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m || landmarks.len() != n {
        return LandmarkResult {
            registered: data.clone(),
            gammas: FdMatrix::zeros(n, m),
            landmarks: landmarks
                .iter()
                .map(|lm| {
                    lm.iter()
                        .map(|&p| Landmark {
                            position: p,
                            kind: LandmarkKind::Custom,
                            value: 0.0,
                            prominence: 0.0,
                        })
                        .collect()
                })
                .collect(),
            target_landmarks: target.unwrap_or(&[]).to_vec(),
        };
    }

    let target_pos = compute_target_landmarks(landmarks, target, n);

    let k = target_pos.len();

    // Build warping functions and register
    let mut registered = FdMatrix::zeros(n, m);
    let mut gammas = FdMatrix::zeros(n, m);
    let mut landmark_info = Vec::with_capacity(n);

    for i in 0..n {
        let source: Vec<f64> = landmarks[i].iter().take(k).copied().collect();
        let gamma = monotone_landmark_warp(&source, &target_pos, argvals);
        let fi = data.row(i);
        let f_aligned = reparameterize_curve(&fi, argvals, &gamma);

        for j in 0..m {
            registered[(i, j)] = f_aligned[j];
            gammas[(i, j)] = gamma[j];
        }

        landmark_info.push(
            landmarks[i]
                .iter()
                .map(|&p| Landmark {
                    position: p,
                    kind: LandmarkKind::Custom,
                    value: 0.0,
                    prominence: 0.0,
                })
                .collect(),
        );
    }

    LandmarkResult {
        registered,
        gammas,
        landmarks: landmark_info,
        target_landmarks: target_pos,
    }
}

/// Detect landmarks and register curves in one step.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `kind` — Type of landmark to detect
/// * `min_prominence` — Minimum prominence for peak/valley detection
/// * `expected_count` — Expected number of landmarks per curve (uses first N)
///
/// # Returns
/// [`LandmarkResult`] with detected landmarks and registered curves.
pub fn detect_and_register(
    data: &FdMatrix,
    argvals: &[f64],
    kind: LandmarkKind,
    min_prominence: f64,
    expected_count: usize,
) -> LandmarkResult {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m {
        return LandmarkResult {
            registered: data.clone(),
            gammas: FdMatrix::zeros(n, m),
            landmarks: Vec::new(),
            target_landmarks: Vec::new(),
        };
    }

    // Detect landmarks for each curve
    let mut all_landmarks = Vec::with_capacity(n);
    let mut all_positions = Vec::with_capacity(n);

    for i in 0..n {
        let curve = data.row(i);
        let lms = detect_landmarks(&curve, argvals, kind, min_prominence);
        let positions: Vec<f64> = lms
            .iter()
            .take(expected_count)
            .map(|l| l.position)
            .collect();
        all_positions.push(positions);
        all_landmarks.push(lms);
    }

    // Register using detected positions
    let result = landmark_register(data, argvals, &all_positions, None);

    LandmarkResult {
        registered: result.registered,
        gammas: result.gammas,
        landmarks: all_landmarks,
        target_landmarks: result.target_landmarks,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn test_monotone_warp_identity() {
        let m = 50;
        let t = uniform_grid(m);
        let source = vec![0.25, 0.5, 0.75];
        let target = vec![0.25, 0.5, 0.75]; // same → identity
        let gamma = monotone_landmark_warp(&source, &target, &t);
        for j in 0..m {
            assert!(
                (gamma[j] - t[j]).abs() < 1e-10,
                "Identity warp should give gamma(t)=t at j={j}: got {}",
                gamma[j]
            );
        }
    }

    #[test]
    fn test_monotone_warp_hits_landmarks() {
        let m = 100;
        let t = uniform_grid(m);
        let source = vec![0.3, 0.6];
        let target = vec![0.4, 0.7];
        let gamma = monotone_landmark_warp(&source, &target, &t);

        // gamma(target[k]) should equal source[k] so that f(gamma(t)) moves
        // features from source positions to target positions
        for (&s, &tgt) in source.iter().zip(target.iter()) {
            let idx = (tgt * (m - 1) as f64).round() as usize;
            assert!(
                (gamma[idx] - s).abs() < 0.05,
                "Warp should map target {tgt} to source {s}, got {}",
                gamma[idx]
            );
        }
    }

    #[test]
    fn test_monotone_warp_monotonicity() {
        let m = 100;
        let t = uniform_grid(m);
        // Adversarial: big shifts
        let source = vec![0.2, 0.8];
        let target = vec![0.5, 0.6];
        let gamma = monotone_landmark_warp(&source, &target, &t);
        for j in 1..m {
            assert!(
                gamma[j] >= gamma[j - 1] - 1e-15,
                "Warp must be monotone at j={j}: {} < {}",
                gamma[j],
                gamma[j - 1]
            );
        }
    }

    #[test]
    fn test_monotone_warp_boundaries() {
        let m = 50;
        let t = uniform_grid(m);
        let source = vec![0.3, 0.7];
        let target = vec![0.4, 0.8];
        let gamma = monotone_landmark_warp(&source, &target, &t);
        assert!(
            (gamma[0] - t[0]).abs() < 1e-15,
            "Warp must start at domain start"
        );
        assert!(
            (gamma[m - 1] - t[m - 1]).abs() < 1e-15,
            "Warp must end at domain end"
        );
    }

    #[test]
    fn test_detect_peaks_sin() {
        let m = 200;
        let t = uniform_grid(m);
        // sin(2πt) has peak at t≈0.25
        let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let lms = detect_landmarks(&curve, &t, LandmarkKind::Peak, 0.1);
        assert!(
            !lms.is_empty(),
            "Should detect at least one peak in sin(2πt)"
        );
        assert!(
            (lms[0].position - 0.25).abs() < 0.02,
            "Peak should be near 0.25, got {}",
            lms[0].position
        );
    }

    #[test]
    fn test_detect_valleys() {
        let m = 200;
        let t = uniform_grid(m);
        // sin(2πt) has valley at t≈0.75
        let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let lms = detect_landmarks(&curve, &t, LandmarkKind::Valley, 0.1);
        assert!(
            !lms.is_empty(),
            "Should detect at least one valley in sin(2πt)"
        );
        assert!(
            (lms[0].position - 0.75).abs() < 0.02,
            "Valley should be near 0.75, got {}",
            lms[0].position
        );
    }

    #[test]
    fn test_detect_zero_crossings() {
        let m = 200;
        let t = uniform_grid(m);
        // sin(2πt) crosses zero at t=0, 0.5, 1.0
        let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let lms = detect_landmarks(&curve, &t, LandmarkKind::ZeroCrossing, 0.0);
        assert!(!lms.is_empty(), "Should detect zero crossings in sin(2πt)");
        // The crossing near 0.5 should be found
        let has_half = lms.iter().any(|l| (l.position - 0.5).abs() < 0.02);
        assert!(has_half, "Should detect zero crossing near t=0.5");
    }

    #[test]
    fn test_registration_reduces_variability() {
        let m = 100;
        let n = 5;
        let t = uniform_grid(m);

        // Create phase-shifted sinusoids: sin(2π(t - shift))
        let shifts = [0.0, 0.02, -0.03, 0.04, -0.01];
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                col_major[i + j * n] = (2.0 * PI * (t[j] - shifts[i])).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        let result = detect_and_register(&data, &t, LandmarkKind::Peak, 0.5, 1);

        // After registration, peak positions should be more aligned
        // Check by measuring variance of peak positions in registered curves
        let mut reg_peaks = Vec::new();
        for i in 0..n {
            let curve = result.registered.row(i);
            let lms = detect_landmarks(&curve, &t, LandmarkKind::Peak, 0.5);
            if let Some(p) = lms.first() {
                reg_peaks.push(p.position);
            }
        }

        if reg_peaks.len() >= 2 {
            let mean_pos: f64 = reg_peaks.iter().sum::<f64>() / reg_peaks.len() as f64;
            let variance: f64 = reg_peaks
                .iter()
                .map(|&p| (p - mean_pos).powi(2))
                .sum::<f64>()
                / reg_peaks.len() as f64;
            assert!(
                variance < 0.01,
                "After registration, peak position variance should be small, got {variance}"
            );
        }
    }

    #[test]
    fn test_single_curve_identity() {
        let m = 50;
        let t = uniform_grid(m);
        let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let data = FdMatrix::from_column_major(curve.clone(), 1, m).unwrap();
        let result = detect_and_register(&data, &t, LandmarkKind::Peak, 0.1, 1);
        // Single curve registration should be close to identity
        let max_diff: f64 = (0..m)
            .map(|j| (result.registered[(0, j)] - data[(0, j)]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 0.1,
            "Single curve should be nearly unchanged, max_diff={max_diff}"
        );
    }

    #[test]
    fn test_no_landmarks_identity() {
        let m = 50;
        let t = uniform_grid(m);
        // Constant curve → no peaks
        let data = FdMatrix::from_column_major(vec![1.0; m], 1, m).unwrap();
        let landmarks = vec![vec![]];
        let result = landmark_register(&data, &t, &landmarks, None);
        // With no landmarks, warp should be identity
        for j in 0..m {
            assert!(
                (result.gammas[(0, j)] - t[j]).abs() < 1e-10,
                "No-landmark warp should be identity at j={j}"
            );
        }
    }

    // ── Reference-value tests (scipy PCHIP) ─────────────────────────────────

    #[test]
    fn test_monotone_warp_reference_scipy_pchip() {
        // Reference: scipy.interpolate.PchipInterpolator([0,0.4,0.7,1], [0,0.3,0.6,1])
        // evaluated at linspace(0,1,11)
        let eval_11: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        let warp = monotone_landmark_warp(&[0.3, 0.6], &[0.4, 0.7], &eval_11);

        #[rustfmt::skip]
        let scipy_ref = [
            0.000000000000000, 0.064845278864971, 0.137206457925636,
            0.215964408023483, 0.300000000000000, 0.390737116764514,
            0.490606653620352, 0.600000000000000, 0.721164021164021,
            0.855026455026455, 1.000000000000000,
        ];

        assert_eq!(warp.len(), 11);
        for (j, (&actual, &expected)) in warp.iter().zip(scipy_ref.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "warp[{j}]: got {actual:.6}, expected {expected:.6}"
            );
        }
    }

    #[test]
    fn test_detect_landmarks_short_curve() {
        // m < 3 should return empty
        let t = vec![0.0, 1.0];
        let curve = vec![1.0, 2.0];
        assert!(detect_landmarks(&curve, &t, LandmarkKind::Peak, 0.0).is_empty());
    }

    #[test]
    fn test_detect_landmarks_mismatched_lengths() {
        let t = vec![0.0, 0.5, 1.0];
        let curve = vec![1.0, 2.0]; // wrong length
        assert!(detect_landmarks(&curve, &t, LandmarkKind::Peak, 0.0).is_empty());
    }

    #[test]
    fn test_detect_landmarks_custom_kind() {
        let t = uniform_grid(50);
        let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        // Custom kind always returns empty
        assert!(detect_landmarks(&curve, &t, LandmarkKind::Custom, 0.0).is_empty());
    }

    #[test]
    fn test_detect_inflections() {
        let m = 200;
        let t = uniform_grid(m);
        // sin(2πt) has inflections at t≈0 and t≈0.5
        let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let lms = detect_landmarks(&curve, &t, LandmarkKind::Inflection, 0.0);
        assert!(!lms.is_empty(), "Should detect inflection points");
        // Check kind
        assert!(lms
            .iter()
            .all(|l| matches!(l.kind, LandmarkKind::Inflection)));
    }

    #[test]
    fn test_detect_inflections_short_curve() {
        // m < 4 should return empty
        let t = vec![0.0, 0.5, 1.0];
        let curve = vec![0.0, 1.0, 0.0];
        let lms = detect_landmarks(&curve, &t, LandmarkKind::Inflection, 0.0);
        assert!(lms.is_empty());
    }

    #[test]
    fn test_detect_peaks_high_prominence_filter() {
        let m = 200;
        let t = uniform_grid(m);
        // Small bumps should be filtered by high prominence
        let curve: Vec<f64> = t.iter().map(|&ti| 0.01 * (2.0 * PI * ti).sin()).collect();
        let lms = detect_landmarks(&curve, &t, LandmarkKind::Peak, 1.0);
        assert!(lms.is_empty(), "High prominence should filter small bumps");
    }

    #[test]
    fn test_detect_valleys_high_prominence_filter() {
        let m = 200;
        let t = uniform_grid(m);
        let curve: Vec<f64> = t.iter().map(|&ti| 0.01 * (2.0 * PI * ti).sin()).collect();
        let lms = detect_landmarks(&curve, &t, LandmarkKind::Valley, 1.0);
        assert!(
            lms.is_empty(),
            "High prominence should filter small valleys"
        );
    }

    #[test]
    fn test_hermite_tangents_two_points() {
        // k=2 branch
        let (delta, d) = hermite_tangents(&[0.0, 1.0], &[0.0, 1.0]);
        assert_eq!(delta.len(), 1);
        assert!((delta[0] - 1.0).abs() < 1e-15);
        assert!((d[0] - 1.0).abs() < 1e-15);
        assert!((d[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_hermite_tangents_zero_dx() {
        // Two coincident knots → delta should be 0
        let (delta, _d) = hermite_tangents(&[0.5, 0.5, 1.0], &[1.0, 2.0, 3.0]);
        assert!((delta[0]).abs() < 1e-10, "Zero dx should give zero delta");
    }

    #[test]
    fn test_landmark_register_empty_data() {
        let data = FdMatrix::zeros(0, 0);
        let result = landmark_register(&data, &[], &[], None);
        assert_eq!(result.registered.nrows(), 0);
    }

    #[test]
    fn test_landmark_register_mismatched_argvals() {
        let m = 10;
        let _t = uniform_grid(m);
        let data = FdMatrix::from_column_major(vec![1.0; m], 1, m).unwrap();
        let wrong_t = vec![0.0; m + 1]; // wrong length
        let landmarks = vec![vec![0.5]];
        let result = landmark_register(&data, &wrong_t, &landmarks, None);
        // Should return early with cloned data
        assert_eq!(result.registered.nrows(), 1);
    }

    #[test]
    fn test_landmark_register_mismatched_n_landmarks() {
        let m = 10;
        let t = uniform_grid(m);
        let data = FdMatrix::from_column_major(vec![1.0; 2 * m], 2, m).unwrap();
        // Only provide 1 landmark vec for 2 curves
        let landmarks = vec![vec![0.5]];
        let result = landmark_register(&data, &t, &landmarks, None);
        assert_eq!(result.registered.nrows(), 2);
    }

    #[test]
    fn test_landmark_register_with_target() {
        let m = 100;
        let n = 3;
        let t = uniform_grid(m);
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                col_major[i + j * n] = (2.0 * PI * (t[j] - 0.02 * i as f64)).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let landmarks = vec![vec![0.25], vec![0.27], vec![0.29]];
        let target = vec![0.25];
        let result = landmark_register(&data, &t, &landmarks, Some(&target));
        assert_eq!(result.target_landmarks, vec![0.25]);
        assert_eq!(result.registered.shape(), (n, m));
    }

    #[test]
    fn test_detect_and_register_empty() {
        let data = FdMatrix::zeros(0, 0);
        let result = detect_and_register(&data, &[], LandmarkKind::Peak, 0.1, 1);
        assert_eq!(result.registered.nrows(), 0);
        assert!(result.landmarks.is_empty());
    }

    #[test]
    fn test_detect_and_register_mismatched_argvals() {
        let m = 10;
        let data = FdMatrix::from_column_major(vec![1.0; m], 1, m).unwrap();
        let wrong_t = vec![0.0; m + 1];
        let result = detect_and_register(&data, &wrong_t, LandmarkKind::Peak, 0.1, 1);
        assert_eq!(result.registered.nrows(), 1);
    }

    #[test]
    fn test_monotone_warp_reference_scipy_extreme() {
        // Reference: scipy PchipInterpolator([0,0.5,0.6,1], [0,0.2,0.8,1])
        // Extreme warp: source landmarks [0.2, 0.8] mapped to [0.5, 0.6]
        let eval_11: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        let warp = monotone_landmark_warp(&[0.2, 0.8], &[0.5, 0.6], &eval_11);

        #[rustfmt::skip]
        let scipy_ref = [
            0.000000000000000, 0.005903448275862, 0.025710344827586,
            0.062565517241379, 0.119613793103448, 0.200000000000000,
            0.800000000000000, 0.893750000000000, 0.955555555555556,
            0.989583333333333, 1.000000000000000,
        ];

        assert_eq!(warp.len(), 11);
        for (j, (&actual, &expected)) in warp.iter().zip(scipy_ref.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.02,
                "warp[{j}]: got {actual:.6}, expected {expected:.6}"
            );
        }

        // Verify monotonicity
        for j in 1..warp.len() {
            assert!(
                warp[j] >= warp[j - 1],
                "Monotonicity violated at {j}: {} < {}",
                warp[j],
                warp[j - 1]
            );
        }
    }
}
