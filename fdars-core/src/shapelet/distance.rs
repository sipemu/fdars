//! Shapelet distance core: per-window z-normalization and the sliding-window
//! minimum z-normalized Euclidean distance (`sdist`), plus the [`Shapelet`] type.
//!
//! This module provides the atomic numerical primitive that every downstream
//! shapelet phase (discovery, transform, classifier) builds on. It is pure
//! `&[f64]` arithmetic — no nalgebra conversion, no integration weights.
//!
//! # Shapelet distance definition
//!
//! Given a length-`L` shapelet `S` (stored **already z-normalized**) and a
//! series `T` of length `M ≥ L`, the shapelet distance is the minimum over all
//! `M - L + 1` sliding windows of the Euclidean distance between the shapelet
//! and the (independently, per-window) z-normalized window:
//!
//! ```text
//! sdist(S, T) = min_{t = 0 .. M-L}  || z(T[t : t+L]) - S ||_2
//! ```
//!
//! Each window is z-normalized **independently at comparison time** — never the
//! whole series once up front. This is what makes the distance scale- and
//! offset-invariant (it captures *shape*, not amplitude/offset).
//!
//! # z-normalization convention
//!
//! Z-normalization here uses the **population** standard deviation (`ddof = 0`),
//! matching the pyts convention. (sktime/aeon variants may use `ddof = 1`; that
//! divergence is intentional and noted here.) A constant or near-constant window
//! (population std ≤ `1e-12`) normalizes to the **zero vector** rather than
//! producing `NaN`/`Inf`.

use crate::error::FdarError;

/// Standard-deviation floor for the constant-window guard.
///
/// A window whose population standard deviation is at or below this threshold is
/// treated as constant and normalized to the zero vector (never divided by
/// ~zero, so the result is always finite).
const STD_EPS: f64 = 1e-12;

/// Z-normalize `src` into `dst` in place (population std, `ddof = 0`).
///
/// Subtracts the arithmetic mean and divides by the population standard
/// deviation. Uses a numerically stable two-pass computation (mean first, then
/// std from deviations) rather than the unstable `E[X²] - E[X]²` form.
///
/// **Constant-window guard:** if the population std is ≤ `1e-12` the window is
/// treated as constant and `dst` is filled with zeros. The output is therefore
/// always finite — never `NaN` or `Inf`.
///
/// This is the allocation-free variant intended for the hot sliding-window loop,
/// where `dst` is a reused scratch buffer.
///
/// # Panics
///
/// In debug builds, panics if `src.len() != dst.len()`. In release builds the
/// shorter length is used (no out-of-bounds access).
pub fn z_normalize_into(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(
        src.len(),
        dst.len(),
        "z_normalize_into: src and dst length mismatch"
    );
    let n = src.len().min(dst.len());
    if n == 0 {
        return;
    }
    let len_f = n as f64;
    // Pass 1: mean.
    let mut sum = 0.0;
    for &v in &src[..n] {
        sum += v;
    }
    let mean = sum / len_f;
    // Pass 2: population variance from deviations.
    let mut sq = 0.0;
    for &v in &src[..n] {
        let d = v - mean;
        sq += d * d;
    }
    let std = (sq / len_f).sqrt();
    if std <= STD_EPS {
        // Constant / near-constant window: zero vector, always finite.
        for d in &mut dst[..n] {
            *d = 0.0;
        }
        return;
    }
    let inv = 1.0 / std;
    for i in 0..n {
        dst[i] = (src[i] - mean) * inv;
    }
}

/// Z-normalize a window slice, returning a freshly allocated vector.
///
/// Population std (`ddof = 0`); constant windows (std ≤ `1e-12`) map to the zero
/// vector. See [`z_normalize_into`] for the in-place hot-loop variant.
///
/// # Examples
///
/// ```
/// use fdars_core::shapelet::z_normalize_window;
///
/// // A constant window normalizes to zeros (no NaN/Inf).
/// let z = z_normalize_window(&[5.0, 5.0, 5.0]);
/// assert_eq!(z, vec![0.0, 0.0, 0.0]);
///
/// // A non-constant window has mean ~0 and population std ~1.
/// let z = z_normalize_window(&[1.0, 2.0, 3.0]);
/// let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
/// assert!(mean.abs() < 1e-12);
/// ```
#[must_use]
pub fn z_normalize_window(slice: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; slice.len()];
    z_normalize_into(slice, &mut out);
    out
}

/// A discovered shapelet: a z-normalized discriminative subsequence plus its
/// provenance in the training set.
///
/// The `values` are stored **already z-normalized** so that [`shapelet_distance`]
/// (and downstream transform/predict paths) never re-normalize the shapelet
/// against test statistics.
///
/// `quality` is a placeholder here (0.0); it is populated by the discovery phase
/// (Phase 58) with a discriminative score (higher = better).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Shapelet {
    /// Z-normalized subsequence values.
    pub values: Vec<f64>,
    /// Index of the source training series this shapelet was extracted from.
    pub series_idx: usize,
    /// Start offset of the subsequence within the source series.
    pub start: usize,
    /// Length `L` of the subsequence.
    pub length: usize,
    /// Discriminative quality score (higher = better). 0.0 until set in discovery.
    pub quality: f64,
}

impl Shapelet {
    /// Build a shapelet from a source series slice, z-normalizing the window
    /// `series[start .. start + length]` and recording provenance.
    ///
    /// `quality` is initialized to 0.0 (set later during discovery).
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] if `length == 0` or the window
    /// `[start, start + length)` does not lie within `series`.
    pub fn from_source(
        series: &[f64],
        series_idx: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, FdarError> {
        if length == 0 {
            return Err(FdarError::InvalidDimension {
                parameter: "length",
                expected: "length >= 1".to_string(),
                actual: length.to_string(),
            });
        }
        let end = start
            .checked_add(length)
            .ok_or(FdarError::InvalidDimension {
                parameter: "start+length",
                expected: format!("<= series length {}", series.len()),
                actual: "overflow".to_string(),
            })?;
        if end > series.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "start+length",
                expected: format!("<= series length {}", series.len()),
                actual: end.to_string(),
            });
        }
        Ok(Self {
            values: z_normalize_window(&series[start..end]),
            series_idx,
            start,
            length,
            quality: 0.0,
        })
    }

    /// Length `L` of the shapelet.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether the shapelet has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Shapelet distance `sdist`: the minimum over sliding windows of the
/// z-normalized Euclidean distance between the (pre-normalized) shapelet and
/// each per-window-normalized window of `series`.
///
/// Returns `(min_distance, best_offset)` where `best_offset` is the start index
/// of the window achieving the minimum (the **first** such offset on ties, for
/// deterministic output).
///
/// # Early abandon
///
/// `best_so_far` is an upper bound on the distance we care about (from prior
/// windows or an external caller). The inner element loop compares the running
/// **squared** partial sum against `best_so_far²` and breaks as soon as it is
/// exceeded — pruning hopeless windows early. Because the squared partial sum is
/// monotonically non-decreasing, abandoning can only skip windows that cannot
/// beat the current best, so the returned minimum is **identical** to a full,
/// non-abandoned computation. Pass `best_so_far = f64::INFINITY` to disable
/// abandoning entirely.
///
/// The metric is plain (unweighted) Euclidean distance over z-normalized
/// values — deliberately *not* the Simpson/integration-weighted functional L2
/// used elsewhere in the crate.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the shapelet is empty or longer
/// than `series` (no valid window).
///
/// # Examples
///
/// ```
/// use fdars_core::shapelet::{shapelet_distance, z_normalize_window};
///
/// // Shapelet is a z-normalized motif; the series contains that exact motif at
/// // offset 2. (A non-linear motif so only the true window matches.)
/// let shapelet = z_normalize_window(&[1.0, 4.0, 2.0]);
/// let series = [0.0, 9.0, 1.0, 4.0, 2.0, 7.0];
/// let (dist, offset) = shapelet_distance(&shapelet, &series, f64::INFINITY).unwrap();
/// assert!(dist < 1e-9);
/// assert_eq!(offset, 2);
/// ```
#[must_use = "the shapelet distance and best-match offset should not be discarded"]
pub fn shapelet_distance(
    shapelet_z: &[f64],
    series: &[f64],
    best_so_far: f64,
) -> Result<(f64, usize), FdarError> {
    let l = shapelet_z.len();
    if l == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "shapelet_z",
            expected: "length >= 1".to_string(),
            actual: "0".to_string(),
        });
    }
    if l > series.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "shapelet_z.len",
            expected: format!("<= series length {}", series.len()),
            actual: l.to_string(),
        });
    }

    // Running best squared distance (compare everything in squared space; sqrt
    // only the final answer). Seed from the caller's bound so early-abandon can
    // prune from the very first window.
    let mut best_sq = if best_so_far.is_finite() {
        best_so_far * best_so_far
    } else {
        f64::INFINITY
    };
    let mut best_offset = 0usize;
    let mut found = false;

    // Reused scratch buffer for the per-window z-normalization (no per-window
    // allocation in the hot loop).
    let mut window_z = vec![0.0; l];

    let n_windows = series.len() - l + 1;
    for t in 0..n_windows {
        let window = &series[t..t + l];
        z_normalize_into(window, &mut window_z);

        // Accumulate squared Euclidean distance with early abandon.
        let mut acc = 0.0;
        let mut abandoned = false;
        for k in 0..l {
            let diff = window_z[k] - shapelet_z[k];
            acc += diff * diff;
            if acc > best_sq {
                abandoned = true;
                break;
            }
        }
        if abandoned {
            continue;
        }
        // acc <= best_sq here. Strict `<` keeps the first-minimum offset on ties.
        if !found || acc < best_sq {
            best_sq = acc;
            best_offset = t;
            found = true;
        }
    }

    // If every window abandoned against the caller's tight bound, no window beat
    // it; report the bound itself as the (non-improving) minimum at offset 0.
    let min_dist = if found { best_sq.sqrt() } else { best_so_far };
    Ok((min_dist, best_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn population_std(z: &[f64]) -> f64 {
        let n = z.len() as f64;
        let mean = z.iter().sum::<f64>() / n;
        (z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    #[test]
    fn test_znorm_constant_window() {
        // Exactly constant.
        let z = z_normalize_window(&[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(z, vec![0.0; 4]);
        assert!(z.iter().all(|v| v.is_finite()));

        // Near-constant: one element perturbed by 1e-15 must still be finite.
        let mut x = vec![5.0; 20];
        x[3] += 1e-15;
        let z = z_normalize_window(&x);
        assert!(
            z.iter().all(|v| v.is_finite()),
            "near-constant produced non-finite"
        );
    }

    #[test]
    fn test_znorm_mean_std() {
        let z = z_normalize_window(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-12, "mean not ~0: {mean}");
        assert!(
            (population_std(&z) - 1.0).abs() < 1e-12,
            "population std not ~1"
        );
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sdist_scale_offset_invariant() {
        // Shapelet = z-normed motif; series contains a plain motif in noise.
        let shapelet = z_normalize_window(&[1.0, 3.0, 2.0, 4.0]);
        let series = vec![0.5, 1.0, 3.0, 2.0, 4.0, 0.7, 0.2];

        let (d0, o0) = shapelet_distance(&shapelet, &series, f64::INFINITY).unwrap();

        // Offset by a constant.
        let shifted: Vec<f64> = series.iter().map(|v| v + 100.0).collect();
        let (d1, o1) = shapelet_distance(&shapelet, &shifted, f64::INFINITY).unwrap();

        // Scale by a positive constant.
        let scaled: Vec<f64> = series.iter().map(|v| v * 50.0).collect();
        let (d2, o2) = shapelet_distance(&shapelet, &scaled, f64::INFINITY).unwrap();

        assert!(
            (d0 - d1).abs() < 1e-10,
            "offset invariance failed: {d0} vs {d1}"
        );
        assert!(
            (d0 - d2).abs() < 1e-10,
            "scale invariance failed: {d0} vs {d2}"
        );
        assert_eq!(o0, o1);
        assert_eq!(o0, o2);
    }

    #[test]
    fn test_sdist_min_semantics() {
        // Series contains an exact copy of the shapelet's source motif at offset 3.
        let motif = [2.0, -1.0, 0.5, 3.0, 1.0];
        let shapelet = z_normalize_window(&motif);
        let mut series = vec![9.0, 8.0, 7.0]; // noise prefix
        series.extend_from_slice(&motif);
        series.extend_from_slice(&[6.0, 5.0]); // noise suffix

        let (dist, offset) = shapelet_distance(&shapelet, &series, f64::INFINITY).unwrap();
        assert!(dist < 1e-9, "exact-motif sdist not ~0: {dist}");
        assert_eq!(offset, 3, "wrong best-match offset");
    }

    #[test]
    fn test_sdist_early_abandon_identical() {
        let shapelet = z_normalize_window(&[0.0, 1.0, 0.5, -1.0, 2.0]);
        let series = vec![
            3.0, 1.0, -2.0, 0.4, 1.5, 0.0, 1.0, 0.5, -1.0, 2.0, 4.0, 2.2, -0.3,
        ];

        // Truth: no abandon.
        let (d_inf, o_inf) = shapelet_distance(&shapelet, &series, f64::INFINITY).unwrap();

        // Tight bound that is >= the true min: abandon must only prune, not
        // change the answer.
        let bound = d_inf + 0.5;
        let (d_tight, o_tight) = shapelet_distance(&shapelet, &series, bound).unwrap();
        assert!((d_inf - d_tight).abs() < 1e-12, "abandon changed the min");
        assert_eq!(o_inf, o_tight, "abandon changed the offset");

        // An exact-min bound must also reproduce the min.
        let (d_eq, o_eq) = shapelet_distance(&shapelet, &series, d_inf).unwrap();
        assert!((d_inf - d_eq).abs() < 1e-12);
        assert_eq!(o_inf, o_eq);
    }

    #[test]
    fn test_sdist_dimension_error() {
        let shapelet = z_normalize_window(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let series = [1.0, 2.0]; // shorter than the shapelet
        let err = shapelet_distance(&shapelet, &series, f64::INFINITY).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));

        // Empty shapelet also errors.
        let err = shapelet_distance(&[], &series, f64::INFINITY).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn test_shapelet_from_source() {
        let series = [0.0, 1.0, 2.0, 3.0, 4.0];
        let s = Shapelet::from_source(&series, 7, 1, 3).unwrap();
        assert_eq!(s.series_idx, 7);
        assert_eq!(s.start, 1);
        assert_eq!(s.length, 3);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.quality, 0.0);
        // values == z-norm of series[1..4]
        assert_eq!(s.values, z_normalize_window(&series[1..4]));

        // Out-of-range window errors.
        assert!(Shapelet::from_source(&series, 0, 3, 5).is_err());
        assert!(Shapelet::from_source(&series, 0, 0, 0).is_err());
    }
}
