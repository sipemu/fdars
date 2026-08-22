//! Network (graph-Laplacian) `MetricSpace` backend (FRE-02-04).
//!
//! Objects are graph Laplacians of `d`-node graphs, stored as flat column-major
//! `Vec<f64>` of length `d*d`. The metric is the element-wise Frobenius distance;
//! the weighted Fréchet mean is the weighted average, which stays a valid
//! Laplacian for non-negative weights (Laplacians form a convex cone). Laplacian
//! structure is not re-validated per call — supplying valid Laplacians is the
//! caller's responsibility.
//!
//! # Divergence from R `frechet` 0.3.0
//!
//! R's network-response geometry may use a different graph metric; this backend
//! uses Frobenius distance on the Laplacian representation. The capability
//! (distance + weighted Fréchet mean over network responses) matches.

use crate::error::FdarError;
use crate::frechet::MetricSpace;
use crate::helpers::NUMERICAL_EPS;

/// Network response space over `d`-node graph Laplacians (FRE-02-04).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NetworkSpace {
    /// Number of graph nodes `d` (objects are `d*d` flat Laplacians).
    pub d: usize,
}

impl NetworkSpace {
    /// Construct a network space over `d`-node graphs.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `d < 1`.
    pub fn new(d: usize) -> Result<Self, FdarError> {
        if d < 1 {
            return Err(FdarError::InvalidParameter {
                parameter: "d",
                message: "number of nodes must be >= 1".to_string(),
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

impl MetricSpace for NetworkSpace {
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
                operation: "NetworkSpace::weighted_frechet_mean",
                detail: "sum of weights is ~0; cannot normalize the barycenter".to_string(),
            });
        }
        let mut m = vec![0.0f64; dd];
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

    // 3-node path-graph Laplacian (flat column-major, symmetric so layout-agnostic).
    fn laplacian_path3() -> Vec<f64> {
        // rows: [1,-1,0; -1,2,-1; 0,-1,1]
        vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0]
    }

    fn laplacian_triangle3() -> Vec<f64> {
        // complete graph on 3 nodes: [2,-1,-1; -1,2,-1; -1,-1,2]
        vec![2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0]
    }

    fn row_sums(m: &[f64], d: usize) -> Vec<f64> {
        (0..d)
            .map(|i| (0..d).map(|j| m[i + j * d]).sum::<f64>())
            .collect()
    }

    #[test]
    fn network_distance_of_identical_is_zero() {
        let s = NetworkSpace::new(3).unwrap();
        let a = laplacian_path3();
        assert!(s.distance(&a, &a).unwrap() < 1e-12);
    }

    #[test]
    fn network_mean_of_identical_recovers() {
        let s = NetworkSpace::new(3).unwrap();
        let a = laplacian_path3();
        let m = s
            .weighted_frechet_mean(&[a.clone(), a.clone()], &[0.5, 0.5])
            .unwrap();
        for (x, y) in m.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn network_mean_preserves_row_sums() {
        let s = NetworkSpace::new(3).unwrap();
        let m = s
            .weighted_frechet_mean(&[laplacian_path3(), laplacian_triangle3()], &[0.4, 0.6])
            .unwrap();
        for rs in row_sums(&m, 3) {
            assert!(rs.abs() < 1e-10, "row sum {rs} != 0");
        }
    }

    #[test]
    fn network_rejects_dimension_mismatch() {
        let s = NetworkSpace::new(3).unwrap();
        let a = laplacian_path3();
        let bad = vec![0.0; 4];
        assert!(matches!(
            s.distance(&a, &bad),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
