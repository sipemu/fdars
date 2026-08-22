//! Correlation-matrix `MetricSpace` backend (FRE-02-02).
//!
//! Objects are correlation matrices (SPD with unit diagonal) stored as flat
//! column-major `Vec<f64>` of length `d*d`. The distance is the element-wise
//! Frobenius norm; the weighted Fréchet mean is the weighted average projected
//! back to a correlation matrix by unit-diagonal renormalization
//! `M̄[i,j] = M[i,j] / sqrt(M[i,i]·M[j,j])`.
//!
//! # Divergence from R `frechet` 0.3.0
//!
//! R uses a correlation-manifold geometry; this backend uses the simpler
//! Frobenius distance with a unit-diagonal-renormalization projection for the
//! mean. Numeric results differ; the capability (distance + weighted mean on a
//! correlation-response space) matches. The renormalization requires positive
//! averaged diagonal entries — a non-positive diagonal returns an error rather
//! than producing an invalid correlation matrix.

use crate::error::FdarError;
use crate::frechet::MetricSpace;
use crate::helpers::NUMERICAL_EPS;

/// Correlation-matrix response space (FRE-02-02).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationMatrixSpace {
    /// Matrix dimension `d` (objects are `d*d` flat vectors).
    pub d: usize,
}

impl CorrelationMatrixSpace {
    /// Construct a correlation-matrix space of dimension `d`.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `d < 1`.
    pub fn new(d: usize) -> Result<Self, FdarError> {
        if d < 1 {
            return Err(FdarError::InvalidParameter {
                parameter: "d",
                message: "matrix dimension must be >= 1".to_string(),
            });
        }
        Ok(Self { d })
    }

    fn check_len(&self, obj: &[f64], name: &'static str) -> Result<(), FdarError> {
        let dd = self.d * self.d;
        if obj.len() != dd {
            return Err(FdarError::InvalidDimension {
                parameter: name,
                expected: format!("{dd} elements (d*d)"),
                actual: format!("{} elements", obj.len()),
            });
        }
        Ok(())
    }
}

impl MetricSpace for CorrelationMatrixSpace {
    type Object = Vec<f64>;

    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError> {
        self.check_len(a, "a")?;
        self.check_len(b, "b")?;
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt())
    }

    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError> {
        if objects.is_empty() {
            return Err(FdarError::InvalidDimension {
                parameter: "objects",
                expected: "at least 1 object".to_string(),
                actual: "0 objects".to_string(),
            });
        }
        if weights.len() != objects.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "weights",
                expected: format!("{} weights (matching objects)", objects.len()),
                actual: format!("{} weights", weights.len()),
            });
        }
        let dd = self.d * self.d;
        for (i, o) in objects.iter().enumerate() {
            if o.len() != dd {
                return Err(FdarError::InvalidDimension {
                    parameter: "objects",
                    expected: format!("each object has {dd} elements"),
                    actual: format!("object {i} has {} elements", o.len()),
                });
            }
        }
        let sw: f64 = weights.iter().sum();
        if sw.abs() < NUMERICAL_EPS {
            return Err(FdarError::ComputationFailed {
                operation: "CorrelationMatrixSpace::weighted_frechet_mean",
                detail: "sum of weights is ~0; cannot normalize the barycenter".to_string(),
            });
        }
        // Element-wise weighted average.
        let mut m = vec![0.0f64; dd];
        for (o, &w) in objects.iter().zip(weights.iter()) {
            for (k, mk) in m.iter_mut().enumerate() {
                *mk += w * o[k];
            }
        }
        for x in &mut m {
            *x /= sw;
        }
        // Renormalize to unit diagonal; guard positive diagonal entries.
        let d = self.d;
        for i in 0..d {
            if m[i + i * d] <= 0.0 {
                return Err(FdarError::ComputationFailed {
                    operation: "CorrelationMatrixSpace::weighted_frechet_mean",
                    detail: format!(
                        "averaged diagonal entry {i} is non-positive ({}); cannot renormalize to a correlation matrix",
                        m[i + i * d]
                    ),
                });
            }
        }
        let mut result = vec![0.0f64; dd];
        for i in 0..d {
            for j in 0..d {
                result[i + j * d] = m[i + j * d] / (m[i + i * d] * m[j + j * d]).sqrt();
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A valid 2×2 correlation matrix with off-diagonal r (flat column-major).
    fn corr2(r: f64) -> Vec<f64> {
        vec![1.0, r, r, 1.0]
    }

    #[test]
    fn correlation_distance_of_identical_is_zero() {
        let s = CorrelationMatrixSpace::new(2).unwrap();
        let a = corr2(0.4);
        assert!(s.distance(&a, &a).unwrap() < 1e-12);
    }

    #[test]
    fn correlation_mean_of_identical_recovers() {
        let s = CorrelationMatrixSpace::new(2).unwrap();
        let a = corr2(0.5);
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone()], &[0.5, 0.5])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10, "x={x} y={y}");
        }
    }

    #[test]
    fn correlation_mean_has_unit_diagonal() {
        let s = CorrelationMatrixSpace::new(2).unwrap();
        let m = s
            .weighted_frechet_mean(&[corr2(0.2), corr2(0.8)], &[0.3, 0.7])
            .unwrap();
        assert!((m[0] - 1.0).abs() < 1e-10);
        assert!((m[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn correlation_rejects_non_positive_diagonal() {
        // Signed weights whose average produces a non-positive diagonal.
        let s = CorrelationMatrixSpace::new(2).unwrap();
        let a = corr2(0.1);
        let b = corr2(0.2);
        // weights sum to 1 but drive diagonal negative: 2*a - 1*b has diagonal 2-1=1 (ok),
        // use weights that make diagonal <= 0: -1 and 0.5 → sum -0.5, diagonal (-1+0.5)/-0.5 ... instead
        // force via near-zero-diagonal average: weights 1 and -1 sum 0 → caught by weight guard.
        // Use a degenerate object with zero diagonal.
        let degen = vec![0.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            s.weighted_frechet_mean(&[degen, a, b], &[1.0, 0.0, 0.0]),
            Err(FdarError::ComputationFailed { .. })
        ));
    }

    #[test]
    fn correlation_rejects_dimension_mismatch() {
        let s = CorrelationMatrixSpace::new(2).unwrap();
        let a = corr2(0.3);
        let bad = vec![1.0, 0.0, 0.0];
        assert!(matches!(
            s.distance(&a, &bad),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
