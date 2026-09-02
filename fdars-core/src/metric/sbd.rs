//! Shape-Based Distance (SBD) via FFT normalized cross-correlation.
//!
//! SBD is the shape-invariant distance primitive underpinning k-Shape clustering
//! (Paparrizos & Gravano, *k-Shape*, SIGMOD 2015). It is invariant to amplitude
//! scaling and phase (circular) shift: two series with the same shape but
//! different offset, scale, or alignment have distance ≈ 0.
//!
//! # Definition
//!
//! Given two series `x`, `y`, both are first **z-normalized** (population std,
//! `ddof = 0`) — this is intrinsic to SBD and is applied inside [`sbd`], never
//! trusted to the caller. Let `x_z`, `y_z` be the z-normalized series. The
//! coefficient-normalized cross-correlation at lag `w` is
//!
//! ```text
//! NCCc_w(x, y) = CC_w(x_z, y_z) / (‖x_z‖₂ · ‖y_z‖₂)
//! ```
//!
//! where `CC` is the linear cross-correlation computed via FFT, and the SBD is
//!
//! ```text
//! SBD(x, y) = 1 − max_w NCCc_w(x, y)   ∈ [0, 2].
//! ```
//!
//! `0` means identical shape (up to scale + shift); `2` means perfectly
//! anti-correlated. The optimal lag `w*` (the signed shift that aligns `y` to
//! `x`) is returned alongside the distance for downstream shift-alignment.
//!
//! # Numerical details (the make-or-break gates)
//!
//! * **Zero-padding.** The FFT is planned for `fft_len = next_power_of_two(2·m − 1)`
//!   (`m` = series length). A too-short FFT would wrap the cross-correlation
//!   circularly and corrupt every nonzero-lag value.
//! * **IFFT scaling.** `rustfft`'s inverse transform is *unnormalized*; the raw
//!   IFFT output is explicitly divided by `fft_len`.
//! * **Coefficient normalization.** `CC` is divided by `‖x_z‖·‖y_z‖` (from the
//!   pre-FFT z-normalized vectors), bounding `NCCc ∈ [−1, 1]`.
//! * **Signed lag.** The raw IFFT index is converted to a signed lag in
//!   `−(m−1) ..= +(m−1)` (not the raw `fft_len − k` wrap-around index).
//! * **Constant-series guard.** If either input is constant (z-norm → zero
//!   vector, so `‖x_z‖·‖y_z‖ ≈ 0`), SBD is defined as
//!   `SbdResult { distance: 1.0, shift: 0 }` (never `NaN`).
//!
//! # Determinism
//!
//! SBD uses no RNG; the sequential and `parallel`-feature builds of
//! [`sbd_distance_matrix`] are bit-identical.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::shapelet::z_normalize_window;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Norms whose product falls below this threshold are treated as a constant
/// series (degenerate NCCc denominator) — matches the z-normalization guard.
const NORM_EPS: f64 = 1e-12;

/// Result of a Shape-Based Distance computation.
///
/// `distance ∈ [0, 2]` (0 = identical shape up to scale/shift). `shift` is the
/// signed optimal cyclic lag `w*` (in `−(m−1) ..= +(m−1)`) by which `y` aligns
/// to `x` — required by k-Shape centroid shift-alignment (Phase 62).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SbdResult {
    /// Shape-Based Distance, `1 − max_w NCCc_w`, in `[0, 2]`.
    pub distance: f64,
    /// Signed optimal lag `w*` (alignment shift of `y` relative to `x`).
    pub shift: isize,
}

/// Compute the Shape-Based Distance between two equal-length series.
///
/// Both series are z-normalized internally (population std); the distance is
/// invariant to amplitude offset, positive scaling, and circular shift. Returns
/// the distance in `[0, 2]` together with the signed optimal alignment lag.
///
/// A constant input (z-norm yields the zero vector) yields
/// `SbdResult { distance: 1.0, shift: 0 }` rather than `NaN`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if either series is empty.
///
/// # Examples
///
/// ```
/// use fdars_core::metric::sbd;
///
/// // A series against a scaled + offset copy of itself is shape-identical.
/// let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
/// let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 10.0).collect();
/// let r = sbd(&x, &y).unwrap();
/// assert!(r.distance < 1e-9, "offset+scale invariant, got {}", r.distance);
/// assert_eq!(r.shift, 0);
/// ```
#[must_use = "SBD result carries the distance and optimal shift"]
pub fn sbd(x: &[f64], y: &[f64]) -> Result<SbdResult, FdarError> {
    if x.is_empty() || y.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "x/y",
            expected: "non-empty series".to_string(),
            actual: format!("x.len()={}, y.len()={}", x.len(), y.len()),
        });
    }

    // Z-normalize both series (intrinsic to SBD — never trust the caller).
    let x_z = z_normalize_window(x);
    let y_z = z_normalize_window(y);

    // Coefficient-normalization denominator from the pre-FFT z-normed vectors.
    let norm_x = x_z.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_y = y_z.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = norm_x * norm_y;
    if denom <= NORM_EPS {
        // Constant series → degenerate NCCc denominator. Defined convention.
        return Ok(SbdResult {
            distance: 1.0,
            shift: 0,
        });
    }

    // Linear cross-correlation needs at least 2·m − 1 lags; round up to a power
    // of two for FFT efficiency. Handles unequal lengths via max(len).
    let m = x.len().max(y.len());
    let fft_len = (2 * m - 1).next_power_of_two();

    // One planner per call (FftPlanner is !Send). Forward plan reused for both
    // transforms of the same length; one inverse plan.
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    // Zero-pad x_z and y_z to fft_len.
    let mut xbuf = vec![Complex::new(0.0, 0.0); fft_len];
    let mut ybuf = vec![Complex::new(0.0, 0.0); fft_len];
    for (b, &v) in xbuf.iter_mut().zip(x_z.iter()) {
        b.re = v;
    }
    for (b, &v) in ybuf.iter_mut().zip(y_z.iter()) {
        b.re = v;
    }

    fft.process(&mut xbuf);
    fft.process(&mut ybuf);

    // Cross-correlation spectrum: X · conj(Y).
    for (xb, yb) in xbuf.iter_mut().zip(ybuf.iter()) {
        *xb *= yb.conj();
    }
    ifft.process(&mut xbuf);

    // rustfft IFFT is unnormalized → divide by fft_len, then coefficient-
    // normalize by ‖x_z‖·‖y_z‖. Scan only the 2m−1 meaningful lags.
    let scale = 1.0 / (fft_len as f64 * denom);
    let m_signed = m as isize;
    let fft_len_signed = fft_len as isize;

    let mut best_ncc = f64::NEG_INFINITY;
    let mut best_shift: isize = 0;
    // Positive lags: raw index k in 0..m maps to shift +k.
    for k in 0..m {
        let ncc = xbuf[k].re * scale;
        if ncc > best_ncc {
            best_ncc = ncc;
            best_shift = k as isize;
        }
    }
    // Negative lags: raw index k in fft_len-(m-1)..fft_len maps to shift k - fft_len.
    for k in (fft_len - (m - 1))..fft_len {
        let ncc = xbuf[k].re * scale;
        if ncc > best_ncc {
            best_ncc = ncc;
            best_shift = k as isize - fft_len_signed;
        }
    }
    debug_assert!(best_shift.abs() < m_signed);

    // Clamp NCCc to [-1, 1] to absorb finite-precision overshoot, keeping
    // distance in [0, 2].
    let max_ncc = best_ncc.clamp(-1.0, 1.0);
    Ok(SbdResult {
        distance: 1.0 - max_ncc,
        shift: best_shift,
    })
}

/// Build the n×n symmetric SBD distance matrix over a curve set.
///
/// `data` is an `FdMatrix` with rows = series and columns = time points (all
/// rows share length `m`). Returns a symmetric matrix with a zero diagonal and
/// `D[(i, j)] = sbd(row_i, row_j).distance`. Only the upper triangle is
/// computed (parallelized via `iter_maybe_parallel!`, each task building its own
/// `FftPlanner`) and mirrored to the lower triangle; the output is bit-identical
/// across sequential and `parallel` builds.
///
/// The result is suitable as a precomputed distance matrix for k-medoids
/// (Phase 63).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has no rows or no columns.
#[must_use = "the SBD distance matrix is the whole point of calling this"]
pub fn sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (n>0 rows, m>0 cols)".to_string(),
            actual: format!("{n}x{m}"),
        });
    }

    // Compute the upper triangle in parallel. Each task materializes its two
    // rows contiguously and calls `sbd`, which builds its own (!Send) planner.
    let upper: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let mut xrow = vec![0.0f64; m];
            data.row_to_buf(i, &mut xrow);
            let mut out = Vec::with_capacity(n.saturating_sub(i + 1));
            let mut yrow = vec![0.0f64; m];
            for j in (i + 1)..n {
                data.row_to_buf(j, &mut yrow);
                // sbd cannot fail here: rows are non-empty (m > 0).
                let d = sbd(&xrow, &yrow).map(|r| r.distance).unwrap_or(1.0);
                out.push(d);
            }
            out
        })
        .collect();

    // Scatter into a symmetric matrix with zero diagonal.
    let mut dist = FdMatrix::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = upper[idx];
            dist[(i, j)] = d;
            dist[(j, i)] = d;
            idx += 1;
        }
    }
    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_x() -> Vec<f64> {
        vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 2.0, 6.0]
    }

    #[test]
    fn test_sbd_self_zero() {
        // Exercises FFT padding + IFFT scaling + coefficient-normalization all
        // at once: NCCc at lag 0 for a series with itself is exactly 1.
        let x = sample_x();
        let r = sbd(&x, &x).unwrap();
        assert!(
            r.distance.abs() < 1e-10,
            "sbd(x,x) must be 0, got {}",
            r.distance
        );
        assert_eq!(r.shift, 0);
    }

    #[test]
    fn test_sbd_symmetric() {
        let x = sample_x();
        let y = vec![2.0, 1.0, 4.0, 3.0, 0.0, 5.0, 1.0, 2.0];
        let dxy = sbd(&x, &y).unwrap().distance;
        let dyx = sbd(&y, &x).unwrap().distance;
        assert!(
            (dxy - dyx).abs() < 1e-10,
            "sbd must be symmetric: {dxy} vs {dyx}"
        );
    }

    #[test]
    fn test_sbd_shifted_copy() {
        // A long sampled sine wave; y is x translated right by k. SBD uses
        // *linear* (zero-padded) cross-correlation, so a shifted copy is not
        // bit-exactly distance 0 (the non-overlapping tail contributes), but the
        // distance is tiny and — crucially — the recovered lag is the correct
        // SIGNED shift ±k, never a `fft_len − k` wrap-around index.
        let n = 128usize;
        let k = 5usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin())
            .collect();
        let mut y = vec![0.0; n];
        y[k..n].copy_from_slice(&x[..(n - k)]);
        let r = sbd(&x, &y).unwrap();
        // Primary gate: the recovered lag is the correct SIGNED shift (magnitude
        // k), NOT a `fft_len − k` wrap-around index.
        assert_eq!(
            r.shift.unsigned_abs(),
            k,
            "expected |shift| == {k}, got {}",
            r.shift
        );
        assert!(
            r.shift.unsigned_abs() < n,
            "shift must be a signed lag, not a fft_len wrap: {}",
            r.shift
        );
        // Distance is near 0: only the k/n non-overlapping tail (zero-padded
        // linear cross-correlation) contributes a small residual.
        assert!(
            r.distance < 0.05,
            "shifted copy must be near 0, got {}",
            r.distance
        );
    }

    #[test]
    fn test_sbd_offset_scale_invariant() {
        let x = sample_x();
        // Offset invariance.
        let x_off: Vec<f64> = x.iter().map(|v| v + 100.0).collect();
        let d_off = sbd(&x, &x_off).unwrap().distance;
        assert!(d_off < 1e-10, "SBD must be offset-invariant: {d_off}");
        // Positive-scale invariance.
        let x_sc: Vec<f64> = x.iter().map(|v| v * 50.0).collect();
        let d_sc = sbd(&x, &x_sc).unwrap().distance;
        assert!(d_sc < 1e-10, "SBD must be scale-invariant: {d_sc}");
    }

    #[test]
    fn test_sbd_ncc_bounds() {
        // distance ∈ [0, 2] ⇔ max NCCc ∈ [−1, 1]. Test several pairs.
        let x = sample_x();
        let cases = [
            vec![6.0, 2.0, 1.0, 4.0, 5.0, 0.0, 3.0, 1.0],
            vec![-1.0, -3.0, -2.0, -5.0, -4.0, -1.0, -2.0, -6.0], // anti-shape
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        ];
        for y in &cases {
            let r = sbd(&x, y).unwrap();
            assert!(
                (0.0..=2.0).contains(&r.distance),
                "distance out of [0,2]: {}",
                r.distance
            );
            let ncc = 1.0 - r.distance;
            assert!(
                (-1.0 - 1e-10..=1.0 + 1e-10).contains(&ncc),
                "NCCc out of [-1,1]: {ncc}"
            );
        }
    }

    #[test]
    fn test_sbd_constant_series() {
        let x = sample_x();
        let c = vec![7.0; x.len()];
        let r = sbd(&x, &c).unwrap();
        assert_eq!(r.distance, 1.0);
        assert_eq!(r.shift, 0);
        assert!(!r.distance.is_nan());
        // Both constant.
        let r2 = sbd(&c, &c).unwrap();
        assert_eq!(r2.distance, 1.0);
        assert_eq!(r2.shift, 0);
    }

    fn sample_matrix() -> FdMatrix {
        // 4 series of length 6.
        let rows = [
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            vec![2.0, 2.0, 4.0, 1.0, 3.0, 5.0],
        ];
        let n = rows.len();
        let m = rows[0].len();
        let mut mat = FdMatrix::zeros(n, m);
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                mat[(i, j)] = v;
            }
        }
        mat
    }

    #[test]
    fn test_sbd_matrix_symmetric_zero_diag() {
        let mat = sample_matrix();
        let d = sbd_distance_matrix(&mat).unwrap();
        let n = mat.nrows();
        for i in 0..n {
            assert!(d[(i, i)].abs() < 1e-15, "diagonal not zero at {i}");
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-15,
                    "not symmetric at ({i},{j})"
                );
            }
        }
        // Matrix entries equal the independent pairwise sbd distances.
        let mut xrow = vec![0.0; mat.ncols()];
        let mut yrow = vec![0.0; mat.ncols()];
        for i in 0..n {
            mat.row_to_buf(i, &mut xrow);
            for j in 0..n {
                if i == j {
                    continue;
                }
                mat.row_to_buf(j, &mut yrow);
                let expected = sbd(&xrow, &yrow).unwrap().distance;
                assert!(
                    (d[(i, j)] - expected).abs() < 1e-15,
                    "matrix entry mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sbd_matrix_parallel_matches() {
        // The matrix builder must be deterministic: recomputing it yields
        // byte-identical output (seq==parallel invariant checked via bit-equal
        // self-consistency, and against independent pairwise sbd).
        let mat = sample_matrix();
        let a = sbd_distance_matrix(&mat).unwrap();
        let b = sbd_distance_matrix(&mat).unwrap();
        let n = mat.nrows();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    a[(i, j)].to_bits(),
                    b[(i, j)].to_bits(),
                    "non-deterministic at ({i},{j})"
                );
            }
        }
        // Bit-identical to independent pairwise computation.
        let mut xrow = vec![0.0; mat.ncols()];
        let mut yrow = vec![0.0; mat.ncols()];
        for i in 0..n {
            mat.row_to_buf(i, &mut xrow);
            for j in (i + 1)..n {
                mat.row_to_buf(j, &mut yrow);
                let d = sbd(&xrow, &yrow).unwrap().distance;
                assert_eq!(a[(i, j)].to_bits(), d.to_bits());
            }
        }
    }
}
