//! Point-process (intensity/count) `MetricSpace` backend (FRE-02-05).
//!
//! Objects are intensity or count vectors on a shared grid, stored as `Vec<f64>`
//! of length `m`. The metric is the L2 (Euclidean) distance; the weighted Fréchet
//! mean is the weighted average. Non-negativity of intensities is not enforced —
//! supplying valid intensities/counts is the caller's responsibility; only
//! dimensions are validated.
//!
//! # Divergence from R `frechet` 0.3.0
//!
//! R's point-process response geometry may use an intensity-transform or
//! Fisher–Rao metric; this backend uses a plain L2 metric on the intensity
//! vector. The capability (distance + weighted Fréchet mean over point-process
//! responses) matches.

use crate::error::FdarError;
use crate::frechet::MetricSpace;
use crate::helpers::NUMERICAL_EPS;

/// Point-process response space over length-`m` intensity/count vectors (FRE-02-05).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointProcessSpace {
    /// Grid length `m` (objects are intensity/count vectors of length `m`).
    pub m: usize,
}

impl PointProcessSpace {
    /// Construct a point-process space over length-`m` intensity vectors.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `m < 1`.
    pub fn new(m: usize) -> Result<Self, FdarError> {
        if m < 1 {
            return Err(FdarError::InvalidParameter {
                parameter: "m",
                message: "intensity grid length must be >= 1".to_string(),
            });
        }
        Ok(Self { m })
    }

    fn check_len(&self, obj: &[f64], name: &'static str) -> Result<(), FdarError> {
        if obj.len() != self.m {
            return Err(FdarError::InvalidDimension {
                parameter: name,
                expected: format!("{} elements", self.m),
                actual: format!("{} elements", obj.len()),
            });
        }
        Ok(())
    }
}

impl MetricSpace for PointProcessSpace {
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
        for (i, o) in objects.iter().enumerate() {
            if o.len() != self.m {
                return Err(FdarError::InvalidDimension {
                    parameter: "objects",
                    expected: format!("each object has {} elements", self.m),
                    actual: format!("object {i} has {} elements", o.len()),
                });
            }
        }
        let sw: f64 = weights.iter().sum();
        if sw.abs() < NUMERICAL_EPS {
            return Err(FdarError::ComputationFailed {
                operation: "PointProcessSpace::weighted_frechet_mean",
                detail: "sum of weights is ~0; cannot normalize the barycenter".to_string(),
            });
        }
        let mut m = vec![0.0f64; self.m];
        for (o, &w) in objects.iter().zip(weights.iter()) {
            for (k, mk) in m.iter_mut().enumerate() {
                *mk += w * o[k];
            }
        }
        for x in &mut m {
            *x /= sw;
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_process_distance_of_identical_is_zero() {
        let s = PointProcessSpace::new(3).unwrap();
        let a = vec![0.5, 1.5, 2.0];
        assert!(s.distance(&a, &a).unwrap() < 1e-12);
    }

    #[test]
    fn point_process_distance_orthonormal_is_sqrt2() {
        let s = PointProcessSpace::new(3).unwrap();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((s.distance(&a, &b).unwrap() - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn point_process_mean_of_identical_recovers() {
        let s = PointProcessSpace::new(3).unwrap();
        let a = vec![0.5, 1.5, 2.0];
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone()], &[0.3, 0.7])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn point_process_rejects_dimension_mismatch() {
        let s = PointProcessSpace::new(3).unwrap();
        let a = vec![1.0, 0.0, 0.0];
        let bad = vec![1.0, 0.0];
        assert!(matches!(
            s.distance(&a, &bad),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
