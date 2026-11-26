//! Contiguous significant regions on an evaluation grid.
//!
//! Shared by every procedure that turns a pointwise decision (a confidence
//! band excluding zero, an adjusted p-value below `α`) into intervals of the
//! domain:
//!
//! * [`significant_regions`] / [`significant_regions_from_se`] — index-only
//!   regions where a band `[lower, upper]` excludes zero (historically in
//!   `explain`; still re-exported there under the same names).
//! * [`difference_regions`] — regions carrying grid coordinates
//!   `[t_start, t_end]`, the direction of the group difference and (for
//!   p-value based procedures) the smallest adjusted p-value inside the region.
//!   Used by [`crate::inference::scb_two_sample_band`],
//!   [`crate::inference::itp_two_pop_regions`] and
//!   [`crate::alignment::elastic_local_difference_test`].
//!
//! # Example
//!
//! ```
//! use fdars_core::regions::{difference_regions, SignificanceDirection};
//!
//! let t = [0.0, 0.25, 0.5, 0.75, 1.0];
//! let effect = [0.1, 0.8, 0.9, -0.7, 0.0];
//! let significant = [false, true, true, true, false];
//! let regions = difference_regions(&t, &effect, &significant, None).unwrap();
//! // Split at the sign change of the effect.
//! assert_eq!(regions.len(), 2);
//! assert_eq!((regions[0].t_start, regions[0].t_end), (0.25, 0.5));
//! assert_eq!(regions[0].direction, SignificanceDirection::Positive);
//! assert_eq!(regions[1].direction, SignificanceDirection::Negative);
//! ```

use crate::error::FdarError;

/// Direction of a significant region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SignificanceDirection {
    /// The effect (e.g. `mean_A − mean_B`) is positive.
    Positive,
    /// The effect is negative.
    Negative,
}

/// A contiguous interval where the confidence band excludes zero.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SignificantRegion {
    /// Start index (inclusive).
    pub start_idx: usize,
    /// End index (inclusive).
    pub end_idx: usize,
    /// Direction of the effect.
    pub direction: SignificanceDirection,
}

/// A contiguous interval of the evaluation grid where two groups differ.
///
/// `[t_start, t_end] = [argvals[start_idx], argvals[end_idx]]` (grid points,
/// both inclusive). Within one region the direction of the effect is
/// constant: a sign change of the effect always starts a new region.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DifferenceRegion {
    /// First grid index of the region (inclusive).
    pub start_idx: usize,
    /// Last grid index of the region (inclusive).
    pub end_idx: usize,
    /// `argvals[start_idx]`.
    pub t_start: f64,
    /// `argvals[end_idx]`.
    pub t_end: f64,
    /// Sign of the effect inside the region (`Positive` ⇔ group A above B for
    /// the tested quantity; see the producing function for its meaning).
    pub direction: SignificanceDirection,
    /// Smallest pointwise adjusted p-value inside the region, for p-value based
    /// procedures (ITP); `None` for band-based procedures (SCB).
    pub min_adjusted_pvalue: Option<f64>,
}

/// Maximal runs `[start, end]` of consecutive indices sharing the same `Some`
/// label. `None` labels separate runs.
fn contiguous_runs<T: Copy + PartialEq>(
    n: usize,
    label: impl Fn(usize) -> Option<T>,
) -> Vec<(usize, usize, T)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        if let Some(d) = label(i) {
            let start = i;
            i += 1;
            while i < n && label(i) == Some(d) {
                i += 1;
            }
            runs.push((start, i - 1, d));
        } else {
            i += 1;
        }
    }
    runs
}

/// Identify contiguous regions where the CI `[lower, upper]` excludes zero.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `lower` is empty or
/// `lower.len() != upper.len()`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn significant_regions(
    lower: &[f64],
    upper: &[f64],
) -> Result<Vec<SignificantRegion>, FdarError> {
    if lower.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "lower",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }
    if lower.len() != upper.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "upper",
            expected: format!("{} (matching lower)", lower.len()),
            actual: format!("{}", upper.len()),
        });
    }
    Ok(
        contiguous_runs(lower.len(), |i| detect_direction(lower[i], upper[i]))
            .into_iter()
            .map(|(start_idx, end_idx, direction)| SignificantRegion {
                start_idx,
                end_idx,
                direction,
            })
            .collect(),
    )
}

/// Build CI from beta(t) +/- z * SE, then find significant regions.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `beta_t` is empty or
/// `beta_t.len() != beta_se.len()`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn significant_regions_from_se(
    beta_t: &[f64],
    beta_se: &[f64],
    z_alpha: f64,
) -> Result<Vec<SignificantRegion>, FdarError> {
    if beta_t.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "beta_t",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }
    if beta_t.len() != beta_se.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "beta_se",
            expected: format!("{} (matching beta_t)", beta_t.len()),
            actual: format!("{}", beta_se.len()),
        });
    }
    let lower: Vec<f64> = beta_t
        .iter()
        .zip(beta_se)
        .map(|(b, s)| b - z_alpha * s)
        .collect();
    let upper: Vec<f64> = beta_t
        .iter()
        .zip(beta_se)
        .map(|(b, s)| b + z_alpha * s)
        .collect();
    significant_regions(&lower, &upper)
}

/// Detect significance direction at a single point from CI bounds.
fn detect_direction(lower: f64, upper: f64) -> Option<SignificanceDirection> {
    if lower > 0.0 {
        Some(SignificanceDirection::Positive)
    } else if upper < 0.0 {
        Some(SignificanceDirection::Negative)
    } else {
        None
    }
}

/// Sign of an effect value (`None` for an exact zero or NaN).
fn effect_direction(e: f64) -> Option<SignificanceDirection> {
    if e > 0.0 {
        Some(SignificanceDirection::Positive)
    } else if e < 0.0 {
        Some(SignificanceDirection::Negative)
    } else {
        None
    }
}

/// Turn a pointwise significance mask into directed grid regions.
///
/// A grid point belongs to a region when `significant[j]` holds and
/// `effect[j]` is non-zero; maximal runs of such points with the same sign of
/// `effect` form one [`DifferenceRegion`]. When `pvalues` is given, each region
/// records the smallest `pvalues[j]` over its points.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] if `argvals` is empty or `effect`,
/// `significant` or `pvalues` do not have `argvals.len()` elements.
pub fn difference_regions(
    argvals: &[f64],
    effect: &[f64],
    significant: &[bool],
    pvalues: Option<&[f64]>,
) -> Result<Vec<DifferenceRegion>, FdarError> {
    let m = argvals.len();
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }
    let check = |name: &'static str, len: usize| {
        if len == m {
            Ok(())
        } else {
            Err(FdarError::InvalidDimension {
                parameter: name,
                expected: format!("{m} (matching argvals)"),
                actual: format!("{len}"),
            })
        }
    };
    check("effect", effect.len())?;
    check("significant", significant.len())?;
    if let Some(p) = pvalues {
        check("pvalues", p.len())?;
    }
    let label = |j: usize| {
        if significant[j] {
            effect_direction(effect[j])
        } else {
            None
        }
    };
    Ok(contiguous_runs(m, label)
        .into_iter()
        .map(|(start_idx, end_idx, direction)| DifferenceRegion {
            start_idx,
            end_idx,
            t_start: argvals[start_idx],
            t_end: argvals[end_idx],
            direction,
            min_adjusted_pvalue: pvalues.map(|p| {
                p[start_idx..=end_idx]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
            }),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difference_regions_splits_and_records_pvalues() {
        let t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let effect = [1.0, 2.0, 0.0, -1.0, -2.0, 3.0];
        let sig = [true, true, true, true, true, false];
        let p = [0.04, 0.01, 0.02, 0.03, 0.001, 0.5];
        let r = difference_regions(&t, &effect, &sig, Some(&p)).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].start_idx, r[0].end_idx), (0, 1));
        assert_eq!(r[0].min_adjusted_pvalue, Some(0.01));
        assert_eq!((r[1].t_start, r[1].t_end), (3.0, 4.0));
        assert_eq!(r[1].direction, SignificanceDirection::Negative);
        assert_eq!(r[1].min_adjusted_pvalue, Some(0.001));
    }

    #[test]
    fn difference_regions_rejects_length_mismatch() {
        assert!(difference_regions(&[0.0, 1.0], &[1.0], &[true, true], None).is_err());
        assert!(difference_regions(&[], &[], &[], None).is_err());
    }

    #[test]
    fn significant_regions_unchanged_semantics() {
        let lower = [0.1, 0.2, -1.0, -0.5, -0.4, 0.3];
        let upper = [1.0, 1.0, 1.0, -0.1, -0.2, 1.0];
        let r = significant_regions(&lower, &upper).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!((r[0].start_idx, r[0].end_idx), (0, 1));
        assert_eq!((r[1].start_idx, r[1].end_idx), (3, 4));
        assert_eq!(r[1].direction, SignificanceDirection::Negative);
        assert_eq!((r[2].start_idx, r[2].end_idx), (5, 5));
    }
}
