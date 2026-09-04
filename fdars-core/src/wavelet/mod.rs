//! Discrete Wavelet Transform (DWT) primitive: single-level orthonormal filter bank.
//!
//! This module provides the numerical core of the wavelet milestone: the
//! orthonormal Daubechies filter tables (Haar/db1 through db10, in [`filters`]) and
//! one single-level DWT step that perfectly reconstructs its input.
//!
//! - **Analysis** ([`single_level_analysis`]): a signal is convolved with the
//!   analysis low-pass filter (`dec_lo`) and high-pass filter (`dec_hi`), then
//!   downsampled by 2, yielding an *approximation* and a *detail* coefficient
//!   vector. Under [`BoundaryMode::Periodic`] each has length `ceil(n/2)`; under
//!   [`BoundaryMode::Symmetric`] each has length `n` (the signal is mirror-extended
//!   to `2n` internally).
//! - **Synthesis** ([`single_level_synthesis`]): the exact inverse of analysis —
//!   the approximation and detail are scattered back through the transpose of the
//!   orthogonal even-length core and summed to reconstruct the original signal.
//!
//! Two boundary handling modes ([`BoundaryMode`]) are supported, and analysis /
//! synthesis form an exact adjoint pair under **each** independently:
//!
//! - [`BoundaryMode::Periodic`] — circular (wrap-around) convolution; the signal is
//!   used as-is when even, or extended by one sample when odd.
//! - [`BoundaryMode::Symmetric`] — half-point boundary reflection; the signal is
//!   mirror-extended to length `2n` internally.
//!
//! Both modes route through one orthogonal even-length periodic core whose transpose
//! is its exact inverse, so single-level analysis followed by synthesis reconstructs
//! any signal of any length to ≤1e-10 relative error for every in-scope family and
//! both boundary modes — the property multi-level (Mallat pyramid) construction
//! builds on. The `rec_lo`/`rec_hi` (time-reversed) synthesis filters are exposed on
//! [`filters::FilterBank`] for downstream use and verified by the filter-invariant
//! tests, though this single-level engine reconstructs via the core transpose.
//!
//! No crate-root or prelude re-exports are added in this phase (deferred to a later
//! phase); the module is reachable only as `crate::wavelet::...`.

// The `rec_lo`/`rec_hi` synthesis filters on [`filters::FilterBank`] are the
// deliverable filter-bank API (verified by the filter-invariant tests) but the
// even-length core reconstructs via the analysis transpose, so those fields are not
// read outside tests yet; `dead_code` is allowed here for that surface (deliverable
// API, not accidental cruft).
#![allow(dead_code)]

pub mod filters;

use crate::error::FdarError;
use filters::FilterBank;

/// A wavelet family: Haar (db1) or Daubechies of a given vanishing-moment order.
///
/// `Daubechies(N)` carries the vanishing-moment order `N ∈ 2..=10` (db2..db10);
/// db1 is spelled [`WaveletFamily::Haar`]. Construct from a plain `dbN` order via
/// [`WaveletFamily::from_db_order`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum WaveletFamily {
    /// The Haar wavelet (equivalently db1): filter length 2.
    Haar,
    /// Daubechies wavelet with `N` vanishing moments (`N ∈ 2..=10`, filter length `2N`).
    Daubechies(usize),
}

impl WaveletFamily {
    /// Map a plain Daubechies order to a [`WaveletFamily`].
    ///
    /// `1` maps to [`WaveletFamily::Haar`] (db1 == Haar); `2..=10` map to
    /// [`WaveletFamily::Daubechies`].
    ///
    /// # Errors
    /// Returns [`FdarError::InvalidParameter`] for an order of `0` or `> 10`.
    pub fn from_db_order(order: usize) -> Result<Self, FdarError> {
        match order {
            1 => Ok(WaveletFamily::Haar),
            2..=10 => Ok(WaveletFamily::Daubechies(order)),
            _ => Err(FdarError::InvalidParameter {
                parameter: "order",
                message: format!(
                    "Daubechies order {order} out of range: supported orders are 1..=10 (1 == Haar)"
                ),
            }),
        }
    }
}

/// Boundary handling for the single-level convolution/downsampling step.
///
/// Analysis and synthesis are an exact adjoint pair under each mode independently,
/// so the single-level round-trip reconstructs perfectly regardless of the mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BoundaryMode {
    /// Circular (periodic) extension: indices wrap around modulo `n`.
    #[default]
    Periodic,
    /// Half-point symmetric extension: indices reflect at the boundaries.
    Symmetric,
}

/// Build the even-length internal signal a boundary `mode` transforms.
///
/// Both modes route through one orthogonal even-length periodic core (below); they
/// differ only in how the raw signal is extended to an even length `m`:
///
/// - [`BoundaryMode::Periodic`]: even `n` is used as-is (`m = n`); odd `n` is
///   extended by one sample (`m = n + 1`) that duplicates the last sample — the
///   half-point periodization convention. Coefficient count is `ceil(n/2)`.
/// - [`BoundaryMode::Symmetric`]: the signal is half-point mirror-reflected to
///   length `m = 2n` (`[s₀..s_{n-1}, s_{n-1}..s₀]`), always even. Coefficient count
///   is `n`.
///
/// Synthesis reconstructs the length-`m` extension exactly, then truncates to `n`.
fn extended_len(n: usize, mode: BoundaryMode) -> usize {
    match mode {
        BoundaryMode::Periodic => {
            if n % 2 == 0 {
                n
            } else {
                n + 1
            }
        }
        BoundaryMode::Symmetric => 2 * n,
    }
}

/// Materialize the even-length extension of `signal` for the given boundary `mode`.
fn extend_signal(signal: &[f64], mode: BoundaryMode) -> Vec<f64> {
    let n = signal.len();
    let m = extended_len(n, mode);
    let mut ext = vec![0.0_f64; m];
    match mode {
        BoundaryMode::Periodic => {
            ext[..n].copy_from_slice(signal);
            if m > n {
                // Odd n: duplicate the last sample (half-point right boundary).
                ext[n] = signal[n - 1];
            }
        }
        BoundaryMode::Symmetric => {
            for i in 0..n {
                ext[i] = signal[i];
                ext[m - 1 - i] = signal[i];
            }
        }
    }
    ext
}

/// Coefficient-vector length (approx == detail) produced by `mode` for signal length `n`.
#[inline]
fn coeff_len(n: usize, mode: BoundaryMode) -> usize {
    extended_len(n, mode) / 2
}

/// Orthogonal even-length periodic analysis core: `out[t] = Σ_k h[k]·s[(2t+k) mod m]`.
///
/// `sig.len()` must be even. Returns two vectors of length `m/2`. This is the exact
/// textbook orthonormal DWT step; its transpose (below) is its exact inverse.
fn core_analysis(sig: &[f64], fb: &FilterBank) -> (Vec<f64>, Vec<f64>) {
    let m = sig.len();
    debug_assert!(m % 2 == 0 && m > 0);
    let l = fb.filter_len();
    let out = m / 2;
    let mut approx = vec![0.0_f64; out];
    let mut detail = vec![0.0_f64; out];
    for t in 0..out {
        let mut a = 0.0_f64;
        let mut d = 0.0_f64;
        for k in 0..l {
            let idx = (2 * t + k) % m;
            let s = sig[idx];
            a += fb.dec_lo[k] * s;
            d += fb.dec_hi[k] * s;
        }
        approx[t] = a;
        detail[t] = d;
    }
    (approx, detail)
}

/// Transpose (exact inverse) of [`core_analysis`]: scatter coefficients back to length `m`.
///
/// `s[(2t+k) mod m] += dec_lo[k]·approx[t] + dec_hi[k]·detail[t]`. Because the
/// even-length periodic analysis operator is orthogonal, this transpose reconstructs
/// the length-`m` signal exactly.
fn core_synthesis(approx: &[f64], detail: &[f64], fb: &FilterBank, m: usize) -> Vec<f64> {
    let l = fb.filter_len();
    let out = approx.len();
    let mut signal = vec![0.0_f64; m];
    for t in 0..out {
        let a = approx[t];
        let d = detail[t];
        for k in 0..l {
            let idx = (2 * t + k) % m;
            signal[idx] += fb.dec_lo[k] * a + fb.dec_hi[k] * d;
        }
    }
    signal
}

/// Single-level DWT analysis: split a signal into approximation + detail coefficients.
///
/// The signal is extended to an even length per the boundary `mode` (see
/// [`extend_signal`]) and convolved with the analysis low-pass (`dec_lo`) and
/// high-pass (`dec_hi`) filters, downsampled by 2. Under [`BoundaryMode::Periodic`]
/// each output vector has length `ceil(n/2)`; under [`BoundaryMode::Symmetric`] it
/// has length `n` (`n = signal.len()`). Analysis and synthesis form an exact
/// adjoint pair under each mode, so the round-trip reconstructs perfectly.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] if `signal` is empty.
#[must_use = "the approximation/detail coefficients are the result of the transform"]
pub(crate) fn single_level_analysis(
    signal: &[f64],
    fb: &FilterBank,
    mode: BoundaryMode,
) -> Result<(Vec<f64>, Vec<f64>), FdarError> {
    if signal.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "signal",
            message: "signal must be non-empty".to_string(),
        });
    }
    let ext = extend_signal(signal, mode);
    Ok(core_analysis(&ext, fb))
}

/// Single-level DWT synthesis: reconstruct a signal from approximation + detail.
///
/// This is the exact inverse of [`single_level_analysis`] under the same boundary
/// `mode`: the coefficients are scattered back through the transpose of the
/// orthogonal even-length core, reconstructing the internal even-length extension,
/// which is then truncated to `output_len` samples. Never emits NaN/inf.
///
/// # Errors
/// - [`FdarError::InvalidParameter`] if `output_len == 0`.
/// - [`FdarError::InvalidDimension`] if `approx` and `detail` differ in length, or
///   if their length does not match the mode's expected coefficient count for
///   `output_len` (`ceil(output_len/2)` periodic, `output_len` symmetric).
#[must_use = "the reconstructed signal is the result of the inverse transform"]
pub(crate) fn single_level_synthesis(
    approx: &[f64],
    detail: &[f64],
    fb: &FilterBank,
    mode: BoundaryMode,
    output_len: usize,
) -> Result<Vec<f64>, FdarError> {
    if output_len == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "output_len",
            message: "output_len must be non-zero".to_string(),
        });
    }
    if approx.len() != detail.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "detail",
            expected: format!("{} (== approx length)", approx.len()),
            actual: detail.len().to_string(),
        });
    }
    let expected_coeff_len = coeff_len(output_len, mode);
    if approx.len() != expected_coeff_len {
        return Err(FdarError::InvalidDimension {
            parameter: "approx",
            expected: format!("{expected_coeff_len} (mode-dependent coefficient length)"),
            actual: approx.len().to_string(),
        });
    }
    let m = extended_len(output_len, mode);
    let mut signal = core_synthesis(approx, detail, fb, m);
    signal.truncate(output_len);
    Ok(signal)
}

// ---------------------------------------------------------------------------
// Multi-level Mallat pyramid (Plan 69-02)
// ---------------------------------------------------------------------------

/// Maximum useful decomposition depth for a signal of length `signal_len`.
///
/// Returns `floor(log2(signal_len / (filter_len - 1)))`, the deepest level at
/// which the coarse approximation band still stays at least as long as the filter
/// support — deeper decompositions would collapse a band below the filter length.
/// For Haar (`filter_len == 2`) this is simply `floor(log2(signal_len))`.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] if `signal_len == 0`, if `family` is an
/// unsupported Daubechies order (surfaced via [`filters::filter_bank`]), or if the
/// signal is too short to admit even a single useful level.
pub fn max_level(signal_len: usize, family: &WaveletFamily) -> Result<usize, FdarError> {
    if signal_len == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "signal_len",
            message: "signal length must be non-zero".to_string(),
        });
    }
    let fb = filters::filter_bank(family)?;
    let filter_len = fb.filter_len();
    // filter_len is always >= 2 for in-scope families; guard defensively.
    if filter_len <= 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "family",
            message: "filter length must exceed 1".to_string(),
        });
    }
    let ratio = signal_len as f64 / (filter_len - 1) as f64;
    // floor(log2(ratio)); ratio >= 1 required for at least one useful level.
    let level = if ratio < 1.0 {
        0
    } else {
        ratio.log2().floor() as usize
    };
    if level < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "signal_len",
            message: format!(
                "signal length {signal_len} is too short for even one useful decomposition level \
                 with filter length {filter_len}"
            ),
        });
    }
    Ok(level)
}

/// Multi-level orthogonal DWT coefficients (the Mallat pyramid) for one signal.
///
/// Produced by [`decompose`] and inverted by [`reconstruct`]. Holds the final coarse
/// approximation, the per-level detail bands, and the metadata reconstruction needs
/// to return exactly the original number of samples.
///
/// The detail bands are stored **finest-first**: `details[0]` is the level-1 detail
/// (highest frequency, longest band) and `details[levels - 1]` is the coarsest detail
/// produced alongside the final approximation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct WaveletCoeffs {
    /// The final coarse approximation band (output of the last analysis level).
    pub approx: Vec<f64>,
    /// Per-level detail bands, **finest-first**: `details[0]` is the level-1 detail.
    pub details: Vec<Vec<f64>>,
    /// Number of decomposition levels (`== details.len()`).
    pub levels: usize,
    /// Original signal length, so [`reconstruct`] returns exactly this many samples.
    pub signal_len: usize,
    /// The wavelet family used for the transform (needed to rebuild the filter bank).
    pub family: WaveletFamily,
    /// The boundary mode used for the transform.
    pub mode: BoundaryMode,
    /// Input length at each analysis level, finest-first: `level_lens[0] == signal_len`
    /// and `level_lens[i]` is the length of the approximation fed into level `i`.
    ///
    /// This per-level bookkeeping lets [`reconstruct`] recover each level's exact
    /// synthesis target length; an off-by-one here mis-aligns the pyramid.
    pub(crate) level_lens: Vec<usize>,
}

impl WaveletCoeffs {
    /// Number of decomposition levels.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Original signal length (the number of samples [`reconstruct`] returns).
    #[must_use]
    pub fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// The final coarse approximation band.
    #[must_use]
    pub fn approx(&self) -> &[f64] {
        &self.approx
    }

    /// The detail band at `level` (finest-first: `0` is the level-1 detail).
    ///
    /// Returns `None` if `level >= levels`.
    #[must_use]
    pub fn detail(&self, level: usize) -> Option<&[f64]> {
        self.details.get(level).map(Vec::as_slice)
    }

    /// The wavelet family used for the transform.
    #[must_use]
    pub fn family(&self) -> &WaveletFamily {
        &self.family
    }

    /// The boundary mode used for the transform.
    #[must_use]
    pub fn mode(&self) -> BoundaryMode {
        self.mode
    }
}

/// Multi-level orthogonal DWT: decompose a signal into a Mallat coefficient pyramid.
///
/// Applies [`single_level_analysis`] repeatedly to the running approximation band,
/// collecting one detail band per level (finest-first). When `level` is `None` the
/// depth defaults to [`max_level`]; an explicit `level` is validated to lie in
/// `1..=max_level`.
///
/// # Errors
/// - [`FdarError::InvalidParameter`] if `signal` is empty, the family is unsupported,
///   or an explicit `level` is `0` or exceeds [`max_level`].
#[must_use = "the coefficient pyramid is the result of the transform"]
pub fn decompose(
    signal: &[f64],
    family: WaveletFamily,
    mode: BoundaryMode,
    level: Option<usize>,
) -> Result<WaveletCoeffs, FdarError> {
    if signal.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "signal",
            message: "signal must be non-empty".to_string(),
        });
    }
    let max_lvl = max_level(signal.len(), &family)?;
    let effective_level = match level {
        None => max_lvl,
        Some(0) => {
            return Err(FdarError::InvalidParameter {
                parameter: "level",
                message: "decomposition level must be at least 1".to_string(),
            });
        }
        Some(l) if l > max_lvl => {
            return Err(FdarError::InvalidParameter {
                parameter: "level",
                message: format!(
                    "decomposition level {l} exceeds the maximum useful level {max_lvl} \
                     for signal length {} with this family",
                    signal.len()
                ),
            });
        }
        Some(l) => l,
    };

    let fb = filters::filter_bank(&family)?;
    let mut details: Vec<Vec<f64>> = Vec::with_capacity(effective_level);
    let mut level_lens: Vec<usize> = Vec::with_capacity(effective_level);
    let mut current = signal.to_vec();
    for _ in 0..effective_level {
        level_lens.push(current.len());
        let (approx, detail) = single_level_analysis(&current, &fb, mode)?;
        details.push(detail);
        current = approx;
    }

    Ok(WaveletCoeffs {
        approx: current,
        details,
        levels: effective_level,
        signal_len: signal.len(),
        family,
        mode,
        level_lens,
    })
}

/// Invert [`decompose`]: reconstruct the original signal from a coefficient pyramid.
///
/// Rebuilds the filter bank from `coeffs.family`, then folds the detail bands back in
/// reverse level order (coarsest-first), calling [`single_level_synthesis`] with each
/// level's exact analysis-input length recovered from `coeffs.level_lens`. Returns
/// exactly `coeffs.signal_len` samples. Never emits NaN/inf.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if the coefficient pyramid is internally
///   inconsistent (band count / metadata mismatch).
/// - [`FdarError::InvalidParameter`] if the family is unsupported.
#[must_use = "the reconstructed signal is the result of the inverse transform"]
pub fn reconstruct(coeffs: &WaveletCoeffs) -> Result<Vec<f64>, FdarError> {
    if coeffs.details.len() != coeffs.levels || coeffs.level_lens.len() != coeffs.levels {
        return Err(FdarError::InvalidDimension {
            parameter: "coeffs",
            expected: format!("{} detail bands and level lengths", coeffs.levels),
            actual: format!(
                "{} detail bands, {} level lengths",
                coeffs.details.len(),
                coeffs.level_lens.len()
            ),
        });
    }
    if coeffs.levels == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "levels",
            expected: "at least 1".to_string(),
            actual: "0".to_string(),
        });
    }
    let fb = filters::filter_bank(&coeffs.family)?;
    let mut approx = coeffs.approx.clone();
    // Fold coarsest-first: level index counts down from levels-1 to 0.
    for lvl in (0..coeffs.levels).rev() {
        let detail = &coeffs.details[lvl];
        let target_len = coeffs.level_lens[lvl];
        approx = single_level_synthesis(&approx, detail, &fb, coeffs.mode, target_len)?;
    }
    Ok(approx)
}

/// Multi-level orthogonal DWT over every row (curve) of an [`FdMatrix`].
///
/// Each row is a signal of length `data.ncols()`; the result holds one
/// [`WaveletCoeffs`] per row, in row order. The batch path is exactly the per-row
/// [`decompose`] applied to each row (`data.row(i)`), so its output is identical to
/// calling `decompose` on each row individually.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if the matrix has zero rows or zero columns.
/// - [`FdarError::InvalidParameter`] if the family is unsupported or an explicit
///   `level` is invalid for the row length (surfaced from [`decompose`]).
#[must_use = "the per-row coefficient pyramids are the result of the transform"]
pub fn decompose_matrix(
    data: &crate::matrix::FdMatrix,
    family: WaveletFamily,
    mode: BoundaryMode,
    level: Option<usize>,
) -> Result<Vec<WaveletCoeffs>, FdarError> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix (nrows > 0 && ncols > 0)".to_string(),
            actual: format!("{}x{}", data.nrows(), data.ncols()),
        });
    }
    let mut out = Vec::with_capacity(data.nrows());
    for i in 0..data.nrows() {
        let row = data.row(i);
        out.push(decompose(&row, family.clone(), mode, level)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavelet::filters::filter_bank;

    /// Deterministic pseudo-random signal (LCG) — spans full rank, no external dep.
    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map top bits to roughly [-1, 1).
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                2.0 * u - 1.0
            })
            .collect()
    }

    fn rel_err(recon: &[f64], orig: &[f64]) -> f64 {
        let num: f64 = recon
            .iter()
            .zip(orig)
            .map(|(r, o)| (r - o) * (r - o))
            .sum::<f64>()
            .sqrt();
        let den: f64 = orig.iter().map(|o| o * o).sum::<f64>().sqrt().max(1e-300);
        num / den
    }

    fn round_trip_families() -> Vec<WaveletFamily> {
        vec![
            WaveletFamily::Haar,
            WaveletFamily::Daubechies(2),
            WaveletFamily::Daubechies(4),
            WaveletFamily::Daubechies(6),
            WaveletFamily::Daubechies(8),
            WaveletFamily::Daubechies(10),
        ]
    }

    // --- Task 1 tracer: Haar known-answer + even-length periodic round-trip ---

    #[test]
    fn haar_known_answer_coefficients() {
        // input [a, b, c, d]
        let a = 1.5_f64;
        let b = -0.5;
        let c = 3.0;
        let d = 2.0;
        let signal = [a, b, c, d];
        let fb = filter_bank(&WaveletFamily::Haar).unwrap();
        let (approx, detail) = single_level_analysis(&signal, &fb, BoundaryMode::Periodic).unwrap();
        let s2 = std::f64::consts::SQRT_2;
        assert!((approx[0] - (a + b) / s2).abs() < 1e-12);
        assert!((approx[1] - (c + d) / s2).abs() < 1e-12);
        assert!((detail[0] - (a - b) / s2).abs() < 1e-12);
        assert!((detail[1] - (c - d) / s2).abs() < 1e-12);
    }

    #[test]
    fn haar_even_length_round_trip() {
        let signal = pseudo_random(8, 42);
        let fb = filter_bank(&WaveletFamily::Haar).unwrap();
        let (approx, detail) = single_level_analysis(&signal, &fb, BoundaryMode::Periodic).unwrap();
        let recon =
            single_level_synthesis(&approx, &detail, &fb, BoundaryMode::Periodic, signal.len())
                .unwrap();
        assert!(rel_err(&recon, &signal) < 1e-10);
    }

    #[test]
    fn empty_signal_is_invalid_parameter() {
        let fb = filter_bank(&WaveletFamily::Haar).unwrap();
        assert!(matches!(
            single_level_analysis(&[], &fb, BoundaryMode::Periodic),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    // --- Task 3: all families, both modes, arbitrary length ---

    #[test]
    fn round_trip_periodic_non_power_of_two() {
        let signal = pseudo_random(37, 7);
        for fam in round_trip_families() {
            let fb = filter_bank(&fam).unwrap();
            let (approx, detail) =
                single_level_analysis(&signal, &fb, BoundaryMode::Periodic).unwrap();
            let recon =
                single_level_synthesis(&approx, &detail, &fb, BoundaryMode::Periodic, signal.len())
                    .unwrap();
            let e = rel_err(&recon, &signal);
            assert!(e < 1e-10, "{fam:?} periodic n=37 rel err {e}");
        }
    }

    #[test]
    fn round_trip_symmetric_non_power_of_two() {
        let signal = pseudo_random(37, 11);
        for fam in round_trip_families() {
            let fb = filter_bank(&fam).unwrap();
            let (approx, detail) =
                single_level_analysis(&signal, &fb, BoundaryMode::Symmetric).unwrap();
            let recon = single_level_synthesis(
                &approx,
                &detail,
                &fb,
                BoundaryMode::Symmetric,
                signal.len(),
            )
            .unwrap();
            let e = rel_err(&recon, &signal);
            assert!(e < 1e-10, "{fam:?} symmetric n=37 rel err {e}");
        }
    }

    #[test]
    fn coefficient_lengths_are_ceil_half() {
        let fb = filter_bank(&WaveletFamily::Daubechies(4)).unwrap();
        for n in [36_usize, 37] {
            let signal = pseudo_random(n, 3);
            let (approx, detail) =
                single_level_analysis(&signal, &fb, BoundaryMode::Periodic).unwrap();
            assert_eq!(approx.len(), n.div_ceil(2));
            assert_eq!(detail.len(), n.div_ceil(2));
            let recon =
                single_level_synthesis(&approx, &detail, &fb, BoundaryMode::Periodic, n).unwrap();
            assert_eq!(recon.len(), n);
        }
    }

    #[test]
    fn reconstruction_has_no_nan_or_inf() {
        for &n in &[36_usize, 37] {
            let signal = pseudo_random(n, 99);
            for fam in round_trip_families() {
                let fb = filter_bank(&fam).unwrap();
                for mode in [BoundaryMode::Periodic, BoundaryMode::Symmetric] {
                    let (approx, detail) = single_level_analysis(&signal, &fb, mode).unwrap();
                    let recon = single_level_synthesis(&approx, &detail, &fb, mode, n).unwrap();
                    assert!(
                        recon.iter().all(|x| x.is_finite()),
                        "{fam:?} {mode:?} n={n} produced non-finite"
                    );
                }
            }
        }
    }

    #[test]
    fn db4_even_and_odd_round_trip_both_modes() {
        let fb = filter_bank(&WaveletFamily::Daubechies(4)).unwrap();
        for n in [36_usize, 37] {
            let signal = pseudo_random(n, 5);
            for mode in [BoundaryMode::Periodic, BoundaryMode::Symmetric] {
                let (approx, detail) = single_level_analysis(&signal, &fb, mode).unwrap();
                let recon = single_level_synthesis(&approx, &detail, &fb, mode, n).unwrap();
                let e = rel_err(&recon, &signal);
                assert!(e < 1e-10, "db4 n={n} {mode:?} rel err {e}");
            }
        }
    }

    #[test]
    fn synthesis_rejects_mismatched_coefficient_lengths() {
        let fb = filter_bank(&WaveletFamily::Haar).unwrap();
        let approx = vec![1.0, 2.0];
        let detail = vec![1.0];
        assert!(matches!(
            single_level_synthesis(&approx, &detail, &fb, BoundaryMode::Periodic, 4),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn synthesis_rejects_zero_output_len() {
        let fb = filter_bank(&WaveletFamily::Haar).unwrap();
        assert!(matches!(
            single_level_synthesis(&[], &[], &fb, BoundaryMode::Periodic, 0),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn from_db_order_maps_correctly() {
        assert_eq!(
            WaveletFamily::from_db_order(1).unwrap(),
            WaveletFamily::Haar
        );
        assert_eq!(
            WaveletFamily::from_db_order(2).unwrap(),
            WaveletFamily::Daubechies(2)
        );
        assert_eq!(
            WaveletFamily::from_db_order(10).unwrap(),
            WaveletFamily::Daubechies(10)
        );
        assert!(matches!(
            WaveletFamily::from_db_order(0),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            WaveletFamily::from_db_order(11),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn default_boundary_mode_is_periodic() {
        assert_eq!(BoundaryMode::default(), BoundaryMode::Periodic);
    }

    // --- Task 1: max_level + WaveletCoeffs ---

    #[test]
    fn max_level_haar_is_log2_of_n() {
        // Haar filter_len == 2 -> filter_len - 1 == 1 -> floor(log2(1024/1)) == 10.
        assert_eq!(max_level(1024, &WaveletFamily::Haar).unwrap(), 10);
    }

    #[test]
    fn max_level_db4_uses_filter_len_minus_one() {
        // db4 filter_len == 8 -> floor(log2(1024/7)) == floor(log2(146.28)) == 7.
        let expected = (1024.0_f64 / 7.0).log2().floor() as usize;
        assert_eq!(
            max_level(1024, &WaveletFamily::Daubechies(4)).unwrap(),
            expected
        );
        assert_eq!(expected, 7);
    }

    #[test]
    fn max_level_rejects_zero_length() {
        assert!(matches!(
            max_level(0, &WaveletFamily::Haar),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn max_level_rejects_too_short_signal() {
        // n=1 with Haar: ratio = 1/1 = 1, log2 = 0 -> no useful level.
        assert!(matches!(
            max_level(1, &WaveletFamily::Haar),
            Err(FdarError::InvalidParameter { .. })
        ));
        // db4 needs at least filter_len-1 = 7 samples for one level.
        assert!(matches!(
            max_level(6, &WaveletFamily::Daubechies(4)),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn max_level_rejects_unsupported_family() {
        assert!(matches!(
            max_level(1024, &WaveletFamily::Daubechies(11)),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wavelet_coeffs_partial_eq_on_identical_inputs() {
        let signal = pseudo_random(64, 21);
        let a = decompose(
            &signal,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            Some(3),
        )
        .unwrap();
        let b = decompose(
            &signal,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            Some(3),
        )
        .unwrap();
        assert_eq!(a, b);
        // Accessors expose the expected metadata.
        assert_eq!(a.levels(), 3);
        assert_eq!(a.signal_len(), 64);
        assert_eq!(a.detail(0).unwrap().len(), a.details[0].len());
        assert!(a.detail(3).is_none());
        assert_eq!(a.family(), &WaveletFamily::Daubechies(4));
        assert_eq!(a.mode(), BoundaryMode::Periodic);
    }

    // --- Task 2: multi-level decompose/reconstruct ---

    #[test]
    fn multi_level_round_trip_periodic_all_families() {
        // n=256 is long enough that every family (incl. db10, filter_len 20) admits
        // >=2 useful levels: max_level(256, db10) = floor(log2(256/19)) = 3.
        let signal = pseudo_random(256, 123);
        for fam in round_trip_families() {
            let coeffs = decompose(&signal, fam.clone(), BoundaryMode::Periodic, Some(2)).unwrap();
            assert_eq!(coeffs.levels, 2);
            let recon = reconstruct(&coeffs).unwrap();
            assert_eq!(recon.len(), signal.len());
            let e = rel_err(&recon, &signal);
            assert!(e < 1e-10, "{fam:?} periodic L=2 rel err {e}");
            assert!(recon.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn multi_level_round_trip_symmetric_non_power_of_two() {
        // n=201 is non-power-of-2 and long enough for >=2 auto levels across all
        // families: max_level(201, db10) = floor(log2(201/19)) = 3.
        let signal = pseudo_random(201, 456);
        for fam in round_trip_families() {
            let coeffs = decompose(&signal, fam.clone(), BoundaryMode::Symmetric, None).unwrap();
            assert!(
                coeffs.levels >= 2,
                "{fam:?} expected >=2 auto levels for n=201"
            );
            let recon = reconstruct(&coeffs).unwrap();
            assert_eq!(recon.len(), 201);
            let e = rel_err(&recon, &signal);
            assert!(e < 1e-10, "{fam:?} symmetric n=201 auto rel err {e}");
            assert!(recon.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn multi_level_round_trip_periodic_non_power_of_two() {
        // Non-power-of-2 length under periodic mode too (SC2 covers both modes).
        let signal = pseudo_random(201, 789);
        for fam in round_trip_families() {
            let coeffs = decompose(&signal, fam.clone(), BoundaryMode::Periodic, None).unwrap();
            assert!(coeffs.levels >= 2, "{fam:?} expected >=2 auto levels");
            let recon = reconstruct(&coeffs).unwrap();
            assert_eq!(recon.len(), 201);
            let e = rel_err(&recon, &signal);
            assert!(e < 1e-10, "{fam:?} periodic n=201 rel err {e}");
        }
    }

    #[test]
    fn auto_level_equals_max_level() {
        let signal = pseudo_random(200, 8);
        let coeffs = decompose(
            &signal,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        assert_eq!(
            coeffs.levels,
            max_level(200, &WaveletFamily::Daubechies(4)).unwrap()
        );
    }

    #[test]
    fn explicit_level_out_of_range_is_invalid() {
        let signal = pseudo_random(64, 8);
        let maxl = max_level(64, &WaveletFamily::Daubechies(4)).unwrap();
        assert!(matches!(
            decompose(
                &signal,
                WaveletFamily::Daubechies(4),
                BoundaryMode::Periodic,
                Some(maxl + 1)
            ),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            decompose(
                &signal,
                WaveletFamily::Daubechies(4),
                BoundaryMode::Periodic,
                Some(0)
            ),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn decompose_rejects_empty_signal() {
        assert!(matches!(
            decompose(&[], WaveletFamily::Haar, BoundaryMode::Periodic, None),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn signal_len_preserved_odd_and_even() {
        for &n in &[37_usize, 64] {
            let signal = pseudo_random(n, n as u64);
            let coeffs = decompose(
                &signal,
                WaveletFamily::Daubechies(2),
                BoundaryMode::Periodic,
                Some(2),
            )
            .unwrap();
            let recon = reconstruct(&coeffs).unwrap();
            assert_eq!(recon.len(), n, "n={n}");
        }
    }

    #[test]
    fn reconstruct_rejects_inconsistent_coeffs() {
        let signal = pseudo_random(64, 3);
        let mut coeffs = decompose(
            &signal,
            WaveletFamily::Haar,
            BoundaryMode::Periodic,
            Some(3),
        )
        .unwrap();
        // Corrupt band count without touching `levels`.
        coeffs.details.pop();
        assert!(matches!(
            reconstruct(&coeffs),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    // --- Task 3: FdMatrix batch path + invalid-input gate ---

    #[test]
    fn decompose_matrix_matches_per_row_slice_path() {
        use crate::matrix::FdMatrix;
        let nrows = 5;
        let ncols = 48;
        // Column-major flat buffer of distinct pseudo-random rows.
        let mut flat = vec![0.0_f64; nrows * ncols];
        for i in 0..nrows {
            let row = pseudo_random(ncols, 1000 + i as u64);
            for j in 0..ncols {
                flat[i + j * nrows] = row[j];
            }
        }
        let m = FdMatrix::from_column_major(flat, nrows, ncols).unwrap();
        let batch = decompose_matrix(
            &m,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        assert_eq!(batch.len(), nrows);
        for i in 0..nrows {
            let per_row = decompose(
                &m.row(i),
                WaveletFamily::Daubechies(4),
                BoundaryMode::Periodic,
                None,
            )
            .unwrap();
            assert_eq!(batch[i], per_row, "row {i} batch != per-row");
        }
    }

    #[test]
    fn decompose_matrix_round_trip_both_modes() {
        use crate::matrix::FdMatrix;
        let nrows = 4;
        let ncols = 48;
        let mut rows = Vec::new();
        let mut flat = vec![0.0_f64; nrows * ncols];
        for i in 0..nrows {
            let row = pseudo_random(ncols, 2000 + i as u64);
            for j in 0..ncols {
                flat[i + j * nrows] = row[j];
            }
            rows.push(row);
        }
        let m = FdMatrix::from_column_major(flat, nrows, ncols).unwrap();
        for mode in [BoundaryMode::Periodic, BoundaryMode::Symmetric] {
            let batch = decompose_matrix(&m, WaveletFamily::Daubechies(6), mode, None).unwrap();
            for (i, coeffs) in batch.iter().enumerate() {
                let recon = reconstruct(coeffs).unwrap();
                assert_eq!(recon.len(), ncols);
                let e = rel_err(&recon, &rows[i]);
                assert!(e < 1e-10, "row {i} {mode:?} rel err {e}");
                assert!(recon.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn decompose_matrix_rejects_empty_matrix() {
        use crate::matrix::FdMatrix;
        let zero_rows = FdMatrix::from_column_major(vec![], 0, 5).unwrap();
        assert!(matches!(
            decompose_matrix(
                &zero_rows,
                WaveletFamily::Haar,
                BoundaryMode::Periodic,
                None
            ),
            Err(FdarError::InvalidDimension { .. })
        ));
        let zero_cols = FdMatrix::from_column_major(vec![], 5, 0).unwrap();
        assert!(matches!(
            decompose_matrix(
                &zero_cols,
                WaveletFamily::Haar,
                BoundaryMode::Periodic,
                None
            ),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn unsupported_order_surfaces_invalid_parameter() {
        // from_db_order(11) errors directly.
        assert!(matches!(
            WaveletFamily::from_db_order(11),
            Err(FdarError::InvalidParameter { .. })
        ));
        // An unsupported family reaching decompose returns InvalidParameter, no panic.
        let signal = pseudo_random(64, 1);
        assert!(matches!(
            decompose(
                &signal,
                WaveletFamily::Daubechies(11),
                BoundaryMode::Periodic,
                None
            ),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
