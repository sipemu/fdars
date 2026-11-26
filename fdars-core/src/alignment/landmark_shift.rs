//! Landmark shift registration.
//!
//! Shifts each curve rigidly so that its landmark lands on a common target
//! location τ (scikit-fda `landmark_shift_deltas` / `landmark_shift_registration`).
//!
//! * [`landmark_shift_deltas`] — one landmark per curve; target chosen by
//!   [`LandmarkLocation`] (mean by default, as scikit-fda).
//! * [`landmark_shift_deltas_multi`] — several landmarks per curve; the common
//!   shift is the least-squares fit `δᵢ = (1/K) Σₖ (τₖ − tᵢₖ)`.
//! * [`landmark_shift_registration`] — deltas + re-evaluation via
//!   [`crate::alignment::shift_curves`].
//!
//! Sign convention (as everywhere in fdars shift registration): the registered
//! curve is `fᵢ(t − δᵢ)`, so `δᵢ = τ − tᵢ` and `registered_i(τ) = fᵢ(tᵢ)`.
//! scikit-fda returns `dᵢ = tᵢ − τ = −δᵢ`.
//!
//! scikit-fda 0.10.1's `landmark_shift_registration` docstring claims the default
//! location is the mid-range `(max + min)/2`, but its code delegates to
//! `landmark_shift_deltas`, whose default is the mean; fdars follows the code
//! ([`LandmarkLocation::Mean`]) and offers [`LandmarkLocation::MidRange`]
//! explicitly.

use super::shift_iterative::{shift_curves, ShiftExtrapolation};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Target location τ for single-landmark shift registration.
#[derive(Clone, Copy, Default)]
#[non_exhaustive]
pub enum LandmarkLocation<'a> {
    /// Mean of the landmark positions (scikit-fda default).
    #[default]
    Mean,
    /// `(max + min) / 2`, which minimises the largest absolute shift.
    MidRange,
    /// A fixed target position.
    Fixed(f64),
    /// A closure mapping all landmark positions to the target (e.g. a median).
    Custom(&'a dyn Fn(&[f64]) -> f64),
}

impl std::fmt::Debug for LandmarkLocation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LandmarkLocation::Mean => f.write_str("Mean"),
            LandmarkLocation::MidRange => f.write_str("MidRange"),
            LandmarkLocation::Fixed(v) => f.debug_tuple("Fixed").field(v).finish(),
            LandmarkLocation::Custom(_) => f.write_str("Custom(<closure>)"),
        }
    }
}

/// Configuration of [`landmark_shift_registration`].
///
/// Construct with `LandmarkShiftConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LandmarkShiftConfig {
    /// Out-of-domain evaluation policy (default [`ShiftExtrapolation::Boundary`];
    /// scikit-fda's `FDataGrid` default corresponds to [`ShiftExtrapolation::Linear`]).
    pub extrapolation: ShiftExtrapolation,
    /// Keep only the grid points observed by every shifted curve (default `false`).
    pub restrict_domain: bool,
}

/// Result of [`landmark_shift_registration`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LandmarkShiftResult {
    /// Registered curves `fᵢ(t − δᵢ)` on `registered_argvals`.
    pub registered_data: FdMatrix,
    /// Grid of `registered_data` (equal to `argvals` unless `restrict_domain`).
    pub registered_argvals: Vec<f64>,
    /// Per-curve shifts `δᵢ = τ − tᵢ`.
    pub shifts: Vec<f64>,
    /// Target landmark location τ.
    pub target: f64,
}

fn check_finite(values: &[f64], parameter: &'static str) -> Result<(), FdarError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter,
            message: "all values must be finite".to_string(),
        });
    }
    Ok(())
}

fn resolve_location(landmarks: &[f64], location: LandmarkLocation<'_>) -> Result<f64, FdarError> {
    let target = match location {
        LandmarkLocation::Mean => landmarks.iter().sum::<f64>() / landmarks.len() as f64,
        LandmarkLocation::MidRange => {
            let max = landmarks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = landmarks.iter().copied().fold(f64::INFINITY, f64::min);
            0.5 * (max + min)
        }
        LandmarkLocation::Fixed(v) => v,
        LandmarkLocation::Custom(f) => f(landmarks),
    };
    if !target.is_finite() {
        return Err(FdarError::InvalidParameter {
            parameter: "location",
            message: format!("target location must be finite, got {target}"),
        });
    }
    Ok(target)
}

fn deltas_and_target(
    landmarks: &[f64],
    location: LandmarkLocation<'_>,
) -> Result<(Vec<f64>, f64), FdarError> {
    if landmarks.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "landmarks",
            expected: "at least one landmark".to_string(),
            actual: "0".to_string(),
        });
    }
    check_finite(landmarks, "landmarks")?;
    let target = resolve_location(landmarks, location)?;
    Ok((landmarks.iter().map(|&t| target - t).collect(), target))
}

/// Shifts that move each curve's landmark `tᵢ` to the target τ: `δᵢ = τ − tᵢ`.
///
/// Equals `−landmark_shift_deltas(fd, landmarks, location)` of scikit-fda 0.10.1
/// (fdars sign convention, see the module documentation).
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `landmarks` is empty.
/// * [`FdarError::InvalidParameter`] if a landmark or the resolved target is not finite.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{landmark_shift_deltas, LandmarkLocation};
///
/// let d = landmark_shift_deltas(&[0.2, 0.4, 0.6], LandmarkLocation::Mean).unwrap();
/// assert!((d[0] - 0.2).abs() < 1e-12 && d[1].abs() < 1e-12 && (d[2] + 0.2).abs() < 1e-12);
/// let d = landmark_shift_deltas(&[0.2, 0.4, 0.9], LandmarkLocation::MidRange).unwrap();
/// assert!((d[0] - 0.35).abs() < 1e-12);
/// ```
pub fn landmark_shift_deltas(
    landmarks: &[f64],
    location: LandmarkLocation<'_>,
) -> Result<Vec<f64>, FdarError> {
    deltas_and_target(landmarks, location).map(|(d, _)| d)
}

/// Least-squares common shift for several landmarks per curve.
///
/// For curve i with landmarks `tᵢ₁ … tᵢK` and targets `τ₁ … τK`, returns
/// `δᵢ = argmin_δ Σₖ (tᵢₖ + δ − τₖ)² = (1/K) Σₖ (τₖ − tᵢₖ)`. `targets = None`
/// uses the per-landmark mean over curves. With `K = 1` and `targets = None`
/// this equals [`landmark_shift_deltas`] with [`LandmarkLocation::Mean`].
/// (scikit-fda supports one landmark per curve only for 1-D domains.)
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `landmarks` is empty, the curves carry
///   different numbers of landmarks, a curve has none, or `targets.len() != K`.
/// * [`FdarError::InvalidParameter`] for non-finite landmarks or targets.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::landmark_shift_deltas_multi;
///
/// let lm = vec![vec![0.2, 0.6], vec![0.3, 0.7]];
/// let d = landmark_shift_deltas_multi(&lm, Some(&[0.25, 0.65])).unwrap();
/// assert!((d[0] - 0.05).abs() < 1e-12 && (d[1] + 0.05).abs() < 1e-12);
/// ```
pub fn landmark_shift_deltas_multi(
    landmarks: &[Vec<f64>],
    targets: Option<&[f64]>,
) -> Result<Vec<f64>, FdarError> {
    let k = landmarks.first().map_or(0, Vec::len);
    if landmarks.is_empty() || k == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "landmarks",
            expected: "at least one curve with at least one landmark".to_string(),
            actual: format!("{} curves, {k} landmarks", landmarks.len()),
        });
    }
    if let Some(bad) = landmarks.iter().find(|l| l.len() != k) {
        return Err(FdarError::InvalidDimension {
            parameter: "landmarks",
            expected: format!("{k} landmarks per curve"),
            actual: bad.len().to_string(),
        });
    }
    for lm in landmarks {
        check_finite(lm, "landmarks")?;
    }
    let tau: Vec<f64> = match targets {
        Some(t) if t.len() != k => {
            return Err(FdarError::InvalidDimension {
                parameter: "targets",
                expected: k.to_string(),
                actual: t.len().to_string(),
            })
        }
        Some(t) => {
            check_finite(t, "targets")?;
            t.to_vec()
        }
        None => (0..k)
            .map(|j| landmarks.iter().map(|l| l[j]).sum::<f64>() / landmarks.len() as f64)
            .collect(),
    };
    Ok(landmarks
        .iter()
        .map(|l| l.iter().zip(&tau).map(|(&t, &g)| g - t).sum::<f64>() / k as f64)
        .collect())
}

/// Shift each curve so that its landmark aligns with a common target.
///
/// Computes `δᵢ = τ − tᵢ` ([`landmark_shift_deltas`]) and returns
/// `fᵢ(t − δᵢ)` re-evaluated by [`crate::alignment::shift_curves`] with
/// `config.extrapolation` / `config.restrict_domain` — scikit-fda 0.10.1
/// `landmark_shift_registration(fd, landmarks, location, restrict_domain,
/// extrapolation)`. For several landmarks per curve combine
/// [`landmark_shift_deltas_multi`] with [`crate::alignment::shift_curves`].
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `data` is empty, `argvals.len() != m`
///   or `landmarks.len() != n`.
/// * [`FdarError::InvalidParameter`] for an invalid grid or non-finite
///   landmarks / target.
/// * [`FdarError::ComputationFailed`] if the restricted domain is empty.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{landmark_shift_registration, LandmarkLocation,
///     LandmarkShiftConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 101;
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let peaks = [0.4, 0.5, 0.6];
/// let mut data = FdMatrix::zeros(3, m);
/// for (i, &p) in peaks.iter().enumerate() {
///     for (j, &t) in argvals.iter().enumerate() {
///         data[(i, j)] = (-(t - p).powi(2) / 0.005).exp();
///     }
/// }
/// let res = landmark_shift_registration(&data, &argvals, &peaks, LandmarkLocation::Mean,
///     &LandmarkShiftConfig::default()).unwrap();
/// assert!((res.target - 0.5).abs() < 1e-12);
/// for i in 0..3 {
///     assert!((res.registered_data[(i, 50)] - 1.0).abs() < 1e-9); // all peaks at t = 0.5
/// }
/// ```
#[must_use = "landmark_shift_registration returns the registered curves"]
pub fn landmark_shift_registration(
    data: &FdMatrix,
    argvals: &[f64],
    landmarks: &[f64],
    location: LandmarkLocation<'_>,
    config: &LandmarkShiftConfig,
) -> Result<LandmarkShiftResult, FdarError> {
    if landmarks.len() != data.nrows() {
        return Err(FdarError::InvalidDimension {
            parameter: "landmarks",
            expected: data.nrows().to_string(),
            actual: landmarks.len().to_string(),
        });
    }
    let (shifts, target) = deltas_and_target(landmarks, location)?;
    let out = shift_curves(
        data,
        argvals,
        &shifts,
        config.extrapolation,
        config.restrict_domain,
    )?;
    Ok(LandmarkShiftResult {
        registered_data: out.data,
        registered_argvals: out.argvals,
        shifts,
        target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const C: [f64; 5] = [0.03, -0.05, 0.08, -0.02, 0.06];
    const A: [f64; 5] = [0.3, 0.25, 0.35, 0.2, 0.4];
    const B: [f64; 5] = [0.1, -0.1, 0.05, 0.0, -0.05];
    const LM: [f64; 5] = [0.31, 0.19, 0.37, 0.24, 0.33];

    /// Same data as `golden_shift.py` (see shift_iterative tests).
    fn golden_data() -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..51).map(|j| j as f64 / 50.0).collect();
        let mut x = FdMatrix::zeros(5, 51);
        for i in 0..5 {
            for (j, &tj) in t.iter().enumerate() {
                x[(i, j)] = (2.0 * PI * (tj - C[i])).sin()
                    + A[i] * (4.0 * PI * (tj - C[i])).cos()
                    + B[i] * tj;
            }
        }
        (x, t)
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < tol, "{what}[{k}]: {a} vs {e}");
        }
    }

    fn neg(v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| -x).collect()
    }

    // Golden: scikit-fda 0.10.1 `landmark_shift_deltas(fd, LM)` and
    // `landmark_shift_deltas(fd, LM, location=0.3)` (negated: fdars sign).
    #[test]
    fn golden_skfda_deltas() {
        let d = landmark_shift_deltas(&LM, LandmarkLocation::Mean).unwrap();
        let skfda = [
            0.02200000000000002,
            -0.09799999999999998,
            0.08200000000000002,
            -0.04799999999999999,
            0.04200000000000004,
        ];
        assert_close(&d, &neg(&skfda), 1e-15, "mean");
        let d = landmark_shift_deltas(&LM, LandmarkLocation::Fixed(0.3)).unwrap();
        let skfda = [
            0.010000000000000009,
            -0.10999999999999999,
            0.07,
            -0.06,
            0.030000000000000027,
        ];
        assert_close(&d, &neg(&skfda), 1e-15, "fixed");
    }

    // Golden: scikit-fda 0.10.1 `landmark_shift_registration(fd, LM)` (FDataGrid
    // default linear extrapolation) and `landmark_shift_registration(
    // FDataGrid(extrapolation="bounds"), LM, location=0.3, restrict_domain=True)`.
    #[test]
    fn golden_skfda_registration() {
        let (x, t) = golden_data();
        let cfg = LandmarkShiftConfig {
            extrapolation: ShiftExtrapolation::Linear,
            ..Default::default()
        };
        let r = landmark_shift_registration(&x, &t, &LM, LandmarkLocation::Mean, &cfg).unwrap();
        assert!((r.target - 0.288).abs() < 1e-15);
        let row1: Vec<f64> = (0..3).map(|j| r.registered_data[(1, j)]).collect();
        assert_close(
            &row1,
            &[
                0.15913741740423681,
                0.23100146343779748,
                0.30286550947135815,
            ],
            1e-12,
            "mean row1",
        );
        let row2: Vec<f64> = (48..51).map(|j| r.registered_data[(2, j)]).collect();
        assert_close(
            &row2,
            &[0.1472572217855016, 0.3336722301876769, 0.5200872385898525],
            1e-12,
            "mean row2",
        );

        let cfg = LandmarkShiftConfig {
            restrict_domain: true,
            ..Default::default()
        };
        let r =
            landmark_shift_registration(&x, &t, &LM, LandmarkLocation::Fixed(0.3), &cfg).unwrap();
        assert_eq!(r.registered_argvals.len(), 41);
        assert!((r.registered_argvals[0] - 0.12).abs() < 1e-12);
        assert!((r.registered_argvals[40] - 0.92).abs() < 1e-12);
        let row1: Vec<f64> = (0..3).map(|j| r.registered_data[(1, j)]).collect();
        assert_close(
            &row1,
            &[0.5472032659854648, 0.610703453436255, 0.6582704681326929],
            1e-12,
            "restrict row1",
        );
        let row2: Vec<f64> = (38..41).map(|j| r.registered_data[(2, j)]).collect();
        assert_close(
            &row2,
            &[
                -0.7018335634871113,
                -0.5225998641773357,
                -0.3374218000601542,
            ],
            1e-12,
            "restrict row2",
        );
    }

    // Invariant: after registration every curve has its landmark value at τ.
    #[test]
    fn landmarks_land_on_target() {
        let (x, t) = golden_data();
        // landmarks on grid points so that fᵢ(tᵢ) is a stored value
        let lm = [0.3, 0.2, 0.36, 0.24, 0.34];
        let r = landmark_shift_registration(
            &x,
            &t,
            &lm,
            LandmarkLocation::Fixed(0.3),
            &LandmarkShiftConfig::default(),
        )
        .unwrap();
        for (i, &li) in lm.iter().enumerate() {
            let j_src = (li * 50.0_f64).round() as usize;
            assert!((r.registered_data[(i, 15)] - x[(i, j_src)]).abs() < 1e-12);
        }
    }

    #[test]
    fn locations_and_multi() {
        let d = landmark_shift_deltas(&LM, LandmarkLocation::Mean).unwrap();
        assert!(d.iter().sum::<f64>().abs() < 1e-15);
        let mid = landmark_shift_deltas(&LM, LandmarkLocation::MidRange).unwrap();
        // midrange 0.28: max |δ| is minimal and attained at both extremes
        assert!((mid[1] - 0.09).abs() < 1e-15 && (mid[2] + 0.09).abs() < 1e-15);
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        let d = landmark_shift_deltas(&LM, LandmarkLocation::Custom(&median)).unwrap();
        assert!(d[0].abs() < 1e-15); // 0.31 is the median

        let single: Vec<Vec<f64>> = LM.iter().map(|&v| vec![v]).collect();
        assert_eq!(
            landmark_shift_deltas_multi(&single, None).unwrap(),
            landmark_shift_deltas(&LM, LandmarkLocation::Mean).unwrap()
        );
        // two landmarks, each curve a pure shift of the template [0.3, 0.7]
        let multi = vec![vec![0.35, 0.75], vec![0.2, 0.6], vec![0.3, 0.7]];
        let d = landmark_shift_deltas_multi(&multi, Some(&[0.3, 0.7])).unwrap();
        assert_close(&d, &[-0.05, 0.1, 0.0], 1e-15, "multi");
    }

    #[test]
    fn validation() {
        let (x, t) = golden_data();
        let cfg = LandmarkShiftConfig::default();
        assert!(matches!(
            landmark_shift_registration(&x, &t, &LM[..4], LandmarkLocation::Mean, &cfg),
            Err(FdarError::InvalidDimension {
                parameter: "landmarks",
                ..
            })
        ));
        assert!(matches!(
            landmark_shift_deltas(&[], LandmarkLocation::Mean),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            landmark_shift_deltas(&[0.1, f64::NAN], LandmarkLocation::Mean),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            landmark_shift_deltas(&[0.1], LandmarkLocation::Fixed(f64::INFINITY)),
            Err(FdarError::InvalidParameter {
                parameter: "location",
                ..
            })
        ));
        assert!(matches!(
            landmark_shift_deltas_multi(&[vec![0.1, 0.2], vec![0.3]], None),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            landmark_shift_deltas_multi(&[vec![0.1, 0.2]], Some(&[0.1])),
            Err(FdarError::InvalidDimension {
                parameter: "targets",
                ..
            })
        ));
        let restrict = LandmarkShiftConfig {
            restrict_domain: true,
            ..Default::default()
        };
        assert!(matches!(
            landmark_shift_registration(
                &x,
                &t,
                &[0.0, 1.2, 0.5, 0.5, 0.5],
                LandmarkLocation::Fixed(0.5),
                &restrict
            ),
            Err(FdarError::ComputationFailed { .. })
        ));
    }
}
