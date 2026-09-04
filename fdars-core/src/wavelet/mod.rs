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

// This module is the single-level DWT primitive foundation. Its filter-bank
// constructor and coefficient tables are consumed by the multi-level/batch surface
// in the next plan; until that lands, some `pub(crate)` items are reached only from
// this module's own tests, so `dead_code` is allowed here (they are the deliverable
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
}
